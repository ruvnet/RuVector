//! ruview-csi-bench — ADR-183 Tier 3 iter 17
//!
//! Cosine-separability benchmark comparing:
//!   A) 128-dim contrastive CSI embeddings (CsiEmbedderCpu)
//!   B) Text-feature baseline (normalised numeric features as a proxy for
//!      the NL-summary → text-encoder pipeline)
//!
//! Metric: **separability ratio** = mean intra-class cosine sim /
//!         mean inter-class cosine sim.  Higher = better cluster purity.
//! Goal (ADR-183 §17): CSI ratio ≥ 2× text-feature ratio.
//!
//! Usage:
//!   ruview-csi-bench --model /usr/local/share/ruvector/model.safetensors
//!   ruview-csi-bench --model MODEL_PATH --samples 20 --noise 0.05

use std::path::PathBuf;

#[cfg(feature = "csi-embed")]
use ruvector_hailo::{CsiEmbedderCpu, CsiFeatures};

fn print_usage() {
    eprintln!(
        "usage: ruview-csi-bench --model PATH [--lora PATH] [--samples N] [--noise F]"
    );
    std::process::exit(1);
}

/// Parse simple --key value args without pulling in clap.
fn parse_args() -> (PathBuf, Option<PathBuf>, usize, f32) {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<PathBuf> = None;
    let mut lora_path: Option<PathBuf> = None;
    let mut samples = 10usize;
    let mut noise = 0.03f32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(PathBuf::from(&args[i]));
            }
            "--lora" => {
                i += 1;
                lora_path = Some(PathBuf::from(&args[i]));
            }
            "--samples" => {
                i += 1;
                samples = args[i].parse().unwrap_or(10);
            }
            "--noise" => {
                i += 1;
                noise = args[i].parse().unwrap_or(0.03);
            }
            "--help" | "-h" => print_usage(),
            _ => {}
        }
        i += 1;
    }
    let model_path = model_path
        .unwrap_or_else(|| PathBuf::from("/usr/local/share/ruvector/model.safetensors"));
    (model_path, lora_path, samples, noise)
}

/// Five synthetic activity archetypes — span the 8-dim CSI feature space.
#[derive(Debug, Clone, Copy)]
struct ActivityClass {
    name: &'static str,
    breathing_bpm: f32,
    breathing_conf: f32,
    heart_rate_bpm: f32,
    hr_conf: f32,
    motion: f32,
    snr_db: f32,
    peak_br: f32,
    peak_hr: f32,
}

const ACTIVITIES: &[ActivityClass] = &[
    ActivityClass {
        name: "resting",
        breathing_bpm: 14.0,
        breathing_conf: 0.9,
        heart_rate_bpm: 62.0,
        hr_conf: 0.85,
        motion: 0.05,
        snr_db: 28.0,
        peak_br: 0.7,
        peak_hr: 0.6,
    },
    ActivityClass {
        name: "exercising",
        breathing_bpm: 26.0,
        breathing_conf: 0.8,
        heart_rate_bpm: 110.0,
        hr_conf: 0.75,
        motion: 0.85,
        snr_db: 18.0,
        peak_br: 0.9,
        peak_hr: 0.85,
    },
    ActivityClass {
        name: "sleeping",
        breathing_bpm: 10.0,
        breathing_conf: 0.95,
        heart_rate_bpm: 52.0,
        hr_conf: 0.9,
        motion: 0.01,
        snr_db: 35.0,
        peak_br: 0.5,
        peak_hr: 0.4,
    },
    ActivityClass {
        name: "stressed",
        breathing_bpm: 20.0,
        breathing_conf: 0.65,
        heart_rate_bpm: 95.0,
        hr_conf: 0.7,
        motion: 0.3,
        snr_db: 22.0,
        peak_br: 0.6,
        peak_hr: 0.75,
    },
    ActivityClass {
        name: "absent",
        breathing_bpm: 0.0,
        breathing_conf: 0.0,
        heart_rate_bpm: 0.0,
        hr_conf: 0.0,
        motion: 0.0,
        snr_db: 8.0,
        peak_br: 0.0,
        peak_hr: 0.0,
    },
];

/// Reproducible pseudo-random noise — LCG with per-sample seed.
fn lcg_noise(seed: u64, amplitude: f32) -> f32 {
    let v = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let f = (v >> 33) as f32 / (u32::MAX as f32);
    (f - 0.5) * 2.0 * amplitude
}

#[cfg(feature = "csi-embed")]
fn activity_to_features(act: &ActivityClass, sample: usize, noise: f32) -> CsiFeatures {
    let n = |base: f32, idx: u64| -> f32 {
        (base + lcg_noise(idx * 1000 + sample as u64, noise)).clamp(0.0, 1.0)
    };
    CsiFeatures {
        breathing_bpm_norm:      n(act.breathing_bpm / 30.0, 1),
        breathing_confidence:    n(act.breathing_conf, 2),
        heart_rate_bpm_norm:     n(act.heart_rate_bpm / 120.0, 3),
        heart_rate_confidence:   n(act.hr_conf, 4),
        motion_score:            n(act.motion, 5),
        log_snr_norm:            n(act.snr_db / 40.0, 6),
        peak_amp_breathing_norm: n(act.peak_br, 7),
        peak_amp_hr_norm:        n(act.peak_hr, 8),
    }
}

/// 8-dim normalised feature vector (the text-baseline proxy).
fn activity_to_text_features(act: &ActivityClass, sample: usize, noise: f32) -> Vec<f32> {
    let n = |base: f32, idx: u64| -> f32 {
        (base + lcg_noise(idx * 1000 + sample as u64, noise)).clamp(0.0, 1.0)
    };
    vec![
        n(act.breathing_bpm / 30.0, 1),
        n(act.breathing_conf, 2),
        n(act.heart_rate_bpm / 120.0, 3),
        n(act.hr_conf, 4),
        n(act.motion, 5),
        n(act.snr_db / 40.0, 6),
        n(act.peak_br, 7),
        n(act.peak_hr, 8),
    ]
}

/// L2-normalise a Vec<f32> in place; returns the norm for diagnostics.
fn l2_norm_inplace(v: &mut Vec<f32>) -> f32 {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter_mut().for_each(|x| *x /= norm);
    norm
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Separability ratio: mean intra-class cosine / mean inter-class cosine.
/// Ratio > 1 means the embeddings cluster by class.
fn separability(embeddings: &[Vec<Vec<f32>>]) -> (f32, f32, f32) {
    let n_classes = embeddings.len();
    let mut intra_sum = 0.0f32;
    let mut intra_cnt = 0usize;
    let mut inter_sum = 0.0f32;
    let mut inter_cnt = 0usize;

    for (ci, class_embs) in embeddings.iter().enumerate() {
        // Intra-class pairs
        for i in 0..class_embs.len() {
            for j in (i + 1)..class_embs.len() {
                intra_sum += cosine(&class_embs[i], &class_embs[j]);
                intra_cnt += 1;
            }
        }
        // Inter-class pairs
        for cj in (ci + 1)..n_classes {
            for ei in class_embs {
                for ej in &embeddings[cj] {
                    inter_sum += cosine(ei, ej);
                    inter_cnt += 1;
                }
            }
        }
    }

    let intra = if intra_cnt > 0 { intra_sum / intra_cnt as f32 } else { 0.0 };
    let inter = if inter_cnt > 0 { inter_sum / inter_cnt as f32 } else { 0.0 };
    let ratio = if inter.abs() > 1e-6 { intra / inter } else { f32::INFINITY };
    (intra, inter, ratio)
}

fn main() {
    let (model_path, lora_path, samples, noise) = parse_args();

    println!("=== ruview-csi-bench (ADR-183 Tier 3 iter 18) ===");
    println!("model:   {}", model_path.display());
    if let Some(ref lp) = lora_path {
        println!("lora:    {}", lp.display());
    } else {
        println!("lora:    (none — base model only)");
    }
    println!("samples: {samples} per class");
    println!("noise:   {noise:.3} (σ of additive Gaussian)");
    println!("classes: {}", ACTIVITIES.iter().map(|a| a.name).collect::<Vec<_>>().join(", "));
    println!();

    // ── Text-feature baseline ──────────────────────────────────────────────
    // Use the raw 8-dim normalised feature vector as a proxy for the
    // "text-encoder on NL summary" baseline. This is a conservative
    // comparison: a real text encoder would perform *worse* because it
    // must recover numeric magnitudes from prose.
    let text_embeddings: Vec<Vec<Vec<f32>>> = ACTIVITIES
        .iter()
        .map(|act| {
            (0..samples)
                .map(|s| {
                    let mut v = activity_to_text_features(act, s, noise);
                    l2_norm_inplace(&mut v);
                    v
                })
                .collect()
        })
        .collect();

    let (text_intra, text_inter, text_ratio) = separability(&text_embeddings);
    println!("Text-feature baseline (8-dim L2-normalised):");
    println!("  intra-class cosine:  {text_intra:.4}");
    println!("  inter-class cosine:  {text_inter:.4}");
    println!("  separability ratio:  {text_ratio:.3}x");
    println!();

    // ── CSI contrastive embeddings ─────────────────────────────────────────
    #[cfg(not(feature = "csi-embed"))]
    {
        eprintln!("CSI embedding path requires --features csi-embed");
        std::process::exit(1);
    }

    #[cfg(feature = "csi-embed")]
    {
        let embedder = match CsiEmbedderCpu::open_with_lora(&model_path, lora_path.as_deref()) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("Cannot load CSI model from {}: {err}", model_path.display());
                eprintln!("Pass --model PATH [--lora PATH] or set RUVIEW_CSI_MODEL");
                std::process::exit(1);
            }
        };
        if embedder.has_lora() {
            println!("LoRA adapter loaded — applying rank-4 room-specific residual update.");
        }

        let csi_embeddings: Vec<Vec<Vec<f32>>> = ACTIVITIES
            .iter()
            .map(|act| {
                (0..samples)
                    .map(|s| {
                        let features = activity_to_features(act, s, noise);
                        embedder.embed(&features).to_vec()
                    })
                    .collect()
            })
            .collect();

        let (csi_intra, csi_inter, csi_ratio) = separability(&csi_embeddings);
        let label = if embedder.has_lora() {
            "CSI + LoRA embeddings (128-dim, rank-4 adapter)"
        } else {
            "CSI contrastive embeddings (128-dim, base model)"
        };
        println!("{label}:");
        println!("  intra-class cosine:  {csi_intra:.4}");
        println!("  inter-class cosine:  {csi_inter:.4}");
        println!("  separability ratio:  {csi_ratio:.3}x");
        println!();

        let improvement = csi_ratio / text_ratio;
        let target_met = improvement >= 2.0;
        println!("Improvement over text-feature baseline: {improvement:.2}x");
        println!(
            "ADR-183 §17 target (≥ 2×): {}",
            if target_met { "PASS ✓" } else { "FAIL ✗" }
        );
        println!();

        // ── Latency benchmark (ADR-183 §7 target: p99 < 12 ms) ───────────────
        {
            const WARMUP: usize = 100;
            const ITERS: usize = 10_000;
            let probe_features = activity_to_features(&ACTIVITIES[0], 0, 0.0);

            // Warm up JIT / branch predictors.
            for _ in 0..WARMUP {
                let _ = embedder.embed(&probe_features);
            }

            let mut latencies_us: Vec<u64> = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let t0 = std::time::Instant::now();
                let _ = embedder.embed(&probe_features);
                latencies_us.push(t0.elapsed().as_micros() as u64);
            }
            latencies_us.sort_unstable();

            let p50 = latencies_us[ITERS / 2];
            let p99 = latencies_us[ITERS * 99 / 100];
            let p99_9 = latencies_us[ITERS * 999 / 1000];
            let mean = latencies_us.iter().sum::<u64>() / ITERS as u64;

            println!("Forward-pass latency (CPU, {ITERS} iters, release build):");
            println!("  mean:  {mean} µs");
            println!("  p50:   {p50} µs");
            println!("  p99:   {p99} µs  ({:.3} ms)", p99 as f64 / 1000.0);
            println!("  p99.9: {p99_9} µs");
            let latency_ok = p99 < 12_000;
            println!(
                "ADR-183 §7 target (p99 < 12 ms): {}",
                if latency_ok { "PASS ✓" } else { "FAIL ✗" }
            );
            if !latency_ok {
                std::process::exit(1);
            }
        }

        if !target_met {
            if embedder.has_lora() {
                eprintln!(
                    "\nNote: LoRA adapter loaded but improvement still < 2×. The adapter \
                     may need more fine-tuning epochs (ADR-183 iter 19: SONA online adapt)."
                );
            } else {
                eprintln!(
                    "\nNote: run with --lora /usr/local/share/ruvector/node-N.json to apply \
                     per-room LoRA adapters (ADR-183 iter 18)."
                );
            }
            std::process::exit(1);
        }
    }
}
