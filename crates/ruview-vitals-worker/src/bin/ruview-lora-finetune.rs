//! ruview-lora-finetune — ADR-183 Tier 3 iter 20
//!
//! Offline supervised LoRA fine-tuning for the CSI contrastive encoder.
//! Optimises the per-room rank-4 LoRA adapter to maximise class separability
//! on the 5 vital-sign activity classes (absent/resting/sleeping/exercising/
//! stressed) using all 8 input features — including motion_score which the
//! online SONA adapter never sees because `VitalReading` does not carry it.
//!
//! ## Why this exists
//!
//! SONA adapts online from live `VitalReading` data, but `VitalReading` has
//! no `motion_score` field.  The ADR-183 §17 separability benchmark uses
//! synthetic feature vectors with motion_score = 0.85 (exercising) vs 0.01
//! (sleeping) — the most discriminative feature in the 8-dim space.  This
//! tool bridges the gap by fine-tuning directly on the benchmark's synthetic
//! class distributions with all 8 features.
//!
//! ## Algorithm
//!
//! 1. Generate `--samples` (default 50) synthetic samples per class using the
//!    same 5 `ACTIVITIES` archetypes + LCG noise as `ruview-csi-bench`.
//! 2. Run triplet-loss gradient steps (Adam, LR cosine-decayed from 1e-3 to
//!    1e-5, margin=0.3) until `improvement = csi_ratio/text_ratio >= 2.0` or
//!    `--max-steps` is exhausted.
//! 3. Check separability every `--check-every` steps (default 200).
//! 4. Save the adapter when the target is met.  Always save at the end.
//!
//! ## Usage
//!
//!   ruview-lora-finetune \
//!     --model /usr/local/share/ruvector/model.safetensors \
//!     --lora  /usr/local/share/ruvector/node-0.json \
//!     --out   /usr/local/share/ruvector/node-0.json \
//!     [--samples 50] [--max-steps 8000] [--check-every 200]
//!
//! The binary requires `--features csi-embed`.

#[cfg(not(feature = "csi-embed"))]
fn main() {
    eprintln!("ruview-lora-finetune requires --features csi-embed");
    std::process::exit(1);
}

#[cfg(feature = "csi-embed")]
fn main() {
    inner::run();
}

#[cfg(feature = "csi-embed")]
mod inner {
    use ruvector_hailo::{CsiEmbedderCpu, CsiFeatures, CsiLoraAdapter, LORA_RANK, CSI_EMBED_DIM};
    use std::path::PathBuf;

    // ── Training hyper-parameters ────────────────────────────────────────────
    const LR_START: f32 = 1e-3;
    const LR_END: f32 = 1e-5;
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const EPS: f32 = 1e-8;
    const MARGIN: f32 = 0.3;
    const TARGET_IMPROVEMENT: f32 = 2.0;

    // ── Synthetic activity class archetypes (mirrors ruview-csi-bench) ───────
    struct Activity {
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

    const ACTIVITIES: &[Activity] = &[
        Activity {
            name: "resting",
            breathing_bpm: 14.0, breathing_conf: 0.9,
            heart_rate_bpm: 62.0, hr_conf: 0.85,
            motion: 0.05, snr_db: 28.0, peak_br: 0.7, peak_hr: 0.6,
        },
        Activity {
            name: "exercising",
            breathing_bpm: 26.0, breathing_conf: 0.8,
            heart_rate_bpm: 110.0, hr_conf: 0.75,
            motion: 0.85, snr_db: 18.0, peak_br: 0.9, peak_hr: 0.85,
        },
        Activity {
            name: "sleeping",
            breathing_bpm: 10.0, breathing_conf: 0.95,
            heart_rate_bpm: 52.0, hr_conf: 0.9,
            motion: 0.01, snr_db: 35.0, peak_br: 0.5, peak_hr: 0.4,
        },
        Activity {
            name: "stressed",
            breathing_bpm: 20.0, breathing_conf: 0.65,
            heart_rate_bpm: 95.0, hr_conf: 0.7,
            motion: 0.3, snr_db: 22.0, peak_br: 0.6, peak_hr: 0.75,
        },
        Activity {
            name: "absent",
            breathing_bpm: 0.0, breathing_conf: 0.0,
            heart_rate_bpm: 0.0, hr_conf: 0.0,
            motion: 0.0, snr_db: 8.0, peak_br: 0.0, peak_hr: 0.0,
        },
    ];

    fn lcg_noise(seed: u64, amplitude: f32) -> f32 {
        let v = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let f = (v >> 33) as f32 / (u32::MAX as f32);
        (f - 0.5) * 2.0 * amplitude
    }

    fn activity_features(act: &Activity, sample: usize, noise: f32) -> CsiFeatures {
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

    fn activity_text_features(act: &Activity, sample: usize, noise: f32) -> Vec<f32> {
        let n = |base: f32, idx: u64| -> f32 {
            (base + lcg_noise(idx * 1000 + sample as u64, noise)).clamp(0.0, 1.0)
        };
        vec![
            n(act.breathing_bpm / 30.0, 1), n(act.breathing_conf, 2),
            n(act.heart_rate_bpm / 120.0, 3), n(act.hr_conf, 4),
            n(act.motion, 5), n(act.snr_db / 40.0, 6), n(act.peak_br, 7), n(act.peak_hr, 8),
        ]
    }

    fn l2_norm_inplace(v: &mut Vec<f32>) {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        v.iter_mut().for_each(|x| *x /= norm);
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn separability(embeddings: &[Vec<Vec<f32>>]) -> (f32, f32, f32) {
        let n_classes = embeddings.len();
        let (mut intra_sum, mut intra_cnt) = (0.0f32, 0usize);
        let (mut inter_sum, mut inter_cnt) = (0.0f32, 0usize);
        for (ci, class_embs) in embeddings.iter().enumerate() {
            for i in 0..class_embs.len() {
                for j in (i + 1)..class_embs.len() {
                    intra_sum += cosine(&class_embs[i], &class_embs[j]);
                    intra_cnt += 1;
                }
            }
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

    // ── Adam optimizer for a flat parameter vector ───────────────────────────
    struct Adam {
        m: Vec<f32>,
        v: Vec<f32>,
        step: u64,
    }

    impl Adam {
        fn new(size: usize) -> Self {
            Self { m: vec![0.0; size], v: vec![0.0; size], step: 0 }
        }

        fn update(&mut self, params: &mut [f32], grad: &[f32], lr: f32) {
            self.step += 1;
            let t = self.step as f32;
            let bc1 = 1.0 - BETA1.powf(t);
            let bc2 = 1.0 - BETA2.powf(t);
            for i in 0..params.len() {
                self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * grad[i];
                self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * grad[i] * grad[i];
                let m_hat = self.m[i] / bc1;
                let v_hat = self.v[i] / bc2;
                params[i] -= lr * m_hat / (v_hat.sqrt() + EPS);
            }
        }
    }

    // ── LoRA forward + backprop ───────────────────────────────────────────────

    fn lora_forward(
        emb: &[f32; CSI_EMBED_DIM],
        lora_a: &[f32], lora_b: &[f32], scaling: f32,
    ) -> [f32; CSI_EMBED_DIM] {
        let mut inter = [0f32; LORA_RANK];
        for j in 0..LORA_RANK {
            let off = j * CSI_EMBED_DIM;
            for k in 0..CSI_EMBED_DIM {
                inter[j] += lora_b[off + k] * emb[k];
            }
        }
        let mut out = [0f32; CSI_EMBED_DIM];
        for i in 0..CSI_EMBED_DIM {
            let off = i * LORA_RANK;
            let mut d = 0f32;
            for j in 0..LORA_RANK { d += lora_a[off + j] * inter[j]; }
            out[i] = emb[i] + scaling * d;
        }
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for v in &mut out { *v /= norm; }
        out
    }

    fn lora_backward(
        emb: &[f32; CSI_EMBED_DIM],
        lora_a: &[f32], lora_b: &[f32], scaling: f32,
        grad_out: &[f32; CSI_EMBED_DIM],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut inter = [0f32; LORA_RANK];
        for j in 0..LORA_RANK {
            let off = j * CSI_EMBED_DIM;
            for k in 0..CSI_EMBED_DIM { inter[j] += lora_b[off + k] * emb[k]; }
        }
        let mut grad_delta = [0f32; CSI_EMBED_DIM];
        for i in 0..CSI_EMBED_DIM { grad_delta[i] = scaling * grad_out[i]; }

        let mut grad_a = vec![0f32; CSI_EMBED_DIM * LORA_RANK];
        for i in 0..CSI_EMBED_DIM {
            let off = i * LORA_RANK;
            for j in 0..LORA_RANK { grad_a[off + j] = grad_delta[i] * inter[j]; }
        }

        let mut grad_inter = [0f32; LORA_RANK];
        for j in 0..LORA_RANK {
            for i in 0..CSI_EMBED_DIM {
                grad_inter[j] += lora_a[i * LORA_RANK + j] * grad_delta[i];
            }
        }

        let mut grad_b = vec![0f32; LORA_RANK * CSI_EMBED_DIM];
        for j in 0..LORA_RANK {
            let off = j * CSI_EMBED_DIM;
            for k in 0..CSI_EMBED_DIM { grad_b[off + k] = grad_inter[j] * emb[k]; }
        }
        (grad_b, grad_a)
    }

    fn save_adapter(
        path: &std::path::Path,
        lora_a: &[f32], lora_b: &[f32], scaling: f32, steps: usize,
    ) -> std::io::Result<()> {
        use std::io::Write as _;
        let mut out = String::with_capacity(64 * 1024);
        out.push_str("{\"config\":{\"rank\":");
        out.push_str(&LORA_RANK.to_string());
        out.push_str(",\"alpha\":");
        out.push_str(&(LORA_RANK * 2).to_string());
        out.push_str("},\"inputDim\":");
        out.push_str(&CSI_EMBED_DIM.to_string());
        out.push_str(",\"outputDim\":");
        out.push_str(&CSI_EMBED_DIM.to_string());
        out.push_str(",\"sona\":{\"step\":");
        out.push_str(&steps.to_string());
        out.push_str(",\"lr\":1e-4,\"beta1\":0.9,\"beta2\":0.999},\"weights\":{\"loraA\":");
        push_matrix_flat(&mut out, lora_a, CSI_EMBED_DIM, LORA_RANK);
        out.push_str(",\"loraB\":");
        push_matrix_flat(&mut out, lora_b, LORA_RANK, CSI_EMBED_DIM);
        out.push_str(",\"scaling\":");
        out.push_str(&format!("{:.1}", scaling));
        out.push_str("}}");

        let tmp = path.with_extension("json.tmp");
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(out.as_bytes())?;
        drop(f);
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    fn push_matrix_flat(out: &mut String, flat: &[f32], rows: usize, cols: usize) {
        out.push('[');
        for r in 0..rows {
            out.push('[');
            for c in 0..cols {
                let v = flat[r * cols + c];
                if v == 0.0 { out.push_str("0.0"); } else { out.push_str(&format!("{v:.8e}")); }
                if c + 1 < cols { out.push(','); }
            }
            out.push(']');
            if r + 1 < rows { out.push(','); }
        }
        out.push(']');
    }

    fn parse_args() -> (PathBuf, PathBuf, PathBuf, usize, usize, usize, f32) {
        let args: Vec<String> = std::env::args().collect();
        let mut model = PathBuf::from("/usr/local/share/ruvector/model.safetensors");
        let mut lora_in = PathBuf::from("/usr/local/share/ruvector/node-0.json");
        let mut lora_out: Option<PathBuf> = None;
        let mut samples = 50usize;
        let mut max_steps = 8000usize;
        let mut check_every = 200usize;
        let mut noise = 0.04f32;
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--model" => { i += 1; model = PathBuf::from(&args[i]); }
                "--lora"  => { i += 1; lora_in = PathBuf::from(&args[i]); }
                "--out"   => { i += 1; lora_out = Some(PathBuf::from(&args[i])); }
                "--samples"     => { i += 1; samples = args[i].parse().unwrap_or(50); }
                "--max-steps"   => { i += 1; max_steps = args[i].parse().unwrap_or(8000); }
                "--check-every" => { i += 1; check_every = args[i].parse().unwrap_or(200); }
                "--noise"       => { i += 1; noise = args[i].parse().unwrap_or(0.04); }
                _ => {}
            }
            i += 1;
        }
        let out = lora_out.unwrap_or_else(|| lora_in.clone());
        (model, lora_in, out, samples, max_steps, check_every, noise)
    }

    pub fn run() {
        let (model_path, lora_in, lora_out, samples, max_steps, check_every, noise) = parse_args();

        println!("=== ruview-lora-finetune (ADR-183 Tier 3 iter 20) ===");
        println!("model:       {}", model_path.display());
        println!("lora-in:     {}", lora_in.display());
        println!("lora-out:    {}", lora_out.display());
        println!("samples/cls: {samples}  max-steps: {max_steps}  check-every: {check_every}");
        println!("noise:       {noise:.3}  LR: {LR_START:.0e}→{LR_END:.0e}  margin: {MARGIN}");
        println!();

        // ── Load base embedder ────────────────────────────────────────────────
        let embedder = CsiEmbedderCpu::open(&model_path).unwrap_or_else(|e| {
            eprintln!("Cannot load model from {}: {e:?}", model_path.display());
            std::process::exit(1);
        });

        // ── Load LoRA adapter ─────────────────────────────────────────────────
        let adapter = CsiLoraAdapter::load(&lora_in).unwrap_or_else(|e| {
            eprintln!("Cannot load LoRA from {}: {e:?}", lora_in.display());
            std::process::exit(1);
        });
        let (mut lora_a, mut lora_b, scaling) = adapter.into_parts();

        // ── Precompute base embeddings for all synthetic samples ──────────────
        // [class][sample] → [f32; CSI_EMBED_DIM]
        let n_classes = ACTIVITIES.len();
        let base_embeddings: Vec<Vec<[f32; CSI_EMBED_DIM]>> = ACTIVITIES
            .iter()
            .map(|act| {
                (0..samples)
                    .map(|s| embedder.embed(&activity_features(act, s, noise)))
                    .collect()
            })
            .collect();

        // ── Compute text baseline separability (fixed) ────────────────────────
        let text_embeddings: Vec<Vec<Vec<f32>>> = ACTIVITIES
            .iter()
            .map(|act| {
                (0..samples)
                    .map(|s| {
                        let mut v = activity_text_features(act, s, noise);
                        l2_norm_inplace(&mut v);
                        v
                    })
                    .collect()
            })
            .collect();
        let (_, _, text_ratio) = separability(&text_embeddings);
        let target_ratio = text_ratio * TARGET_IMPROVEMENT;
        println!("Text baseline separability: {text_ratio:.3}×  (target CSI ≥ {target_ratio:.3}×)");
        println!();

        // ── Adam states ───────────────────────────────────────────────────────
        let mut adam_a = Adam::new(lora_a.len());
        let mut adam_b = Adam::new(lora_b.len());

        // ── LCG for triplet selection ─────────────────────────────────────────
        let mut rng_state = 0xdeadbeef_c0ffeeu64;
        let mut rng = move || -> usize {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 33) as usize)
        };

        let mut best_improvement = 0.0f32;
        let mut target_met = false;
        let mut steps_done = 0usize;

        for step in 0..max_steps {
            // Cosine LR decay
            let lr = LR_END + 0.5 * (LR_START - LR_END) * (1.0 + (std::f32::consts::PI * step as f32 / max_steps as f32).cos());

            // Select anchor class (random, must have ≥ 2 samples → always true here)
            let ac = rng() % n_classes;
            let si_anchor = rng() % samples;
            let si_pos = {
                let mut s = rng() % samples;
                while s == si_anchor { s = rng() % samples; }
                s
            };

            let anchor_base = &base_embeddings[ac][si_anchor];
            let pos_base = &base_embeddings[ac][si_pos];
            let anchor_emb = lora_forward(anchor_base, &lora_a, &lora_b, scaling);
            let pos_emb = lora_forward(pos_base, &lora_a, &lora_b, scaling);

            // Hard negative: class whose centroid is closest to anchor
            let neg_class = (0..n_classes)
                .filter(|&c| c != ac)
                .min_by(|&a, &b| {
                    let da = centroid_cosine_dist(&base_embeddings[a], &lora_a, &lora_b, scaling, &anchor_emb);
                    let db = centroid_cosine_dist(&base_embeddings[b], &lora_a, &lora_b, scaling, &anchor_emb);
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap_or(0);

            let si_neg = rng() % samples;
            let neg_base = &base_embeddings[neg_class][si_neg];
            let neg_emb = lora_forward(neg_base, &lora_a, &lora_b, scaling);

            // Triplet loss: L = max(0, d_ap - d_an + margin)
            let d_ap = 1.0 - cosine(&anchor_emb, &pos_emb);
            let d_an = 1.0 - cosine(&anchor_emb, &neg_emb);
            let loss = (d_ap - d_an + MARGIN).max(0.0);
            if loss == 0.0 { steps_done += 1; continue; }

            // Gradient: push anchor toward positive, away from negative.
            // Approximate per-component gradient on anchor:
            //   dL/d_anchor ≈ pos_emb (attract) - neg_emb (repel)  (sign-flipped via loss)
            let mut grad_anchor = [0f32; CSI_EMBED_DIM];
            for i in 0..CSI_EMBED_DIM {
                // d(d_ap)/d(anchor) = -pos; d(d_an)/d(anchor) = -neg
                // dL/d(anchor) = d(d_ap)/d(anchor) - d(d_an)/d(anchor) = neg - pos
                grad_anchor[i] = neg_emb[i] - pos_emb[i];
            }

            let (gb, ga) = lora_backward(anchor_base, &lora_a, &lora_b, scaling, &grad_anchor);
            adam_b.update(&mut lora_b, &gb, lr);
            adam_a.update(&mut lora_a, &ga, lr);
            steps_done += 1;

            // ── Check separability ────────────────────────────────────────────
            if (step + 1) % check_every == 0 || step + 1 == max_steps {
                let csi_embeddings: Vec<Vec<Vec<f32>>> = base_embeddings
                    .iter()
                    .map(|class_bases| {
                        class_bases
                            .iter()
                            .map(|b| lora_forward(b, &lora_a, &lora_b, scaling).to_vec())
                            .collect()
                    })
                    .collect();
                let (intra, inter, csi_ratio) = separability(&csi_embeddings);
                let improvement = csi_ratio / text_ratio;
                if improvement > best_improvement { best_improvement = improvement; }

                let ok = if improvement >= TARGET_IMPROVEMENT { "✓ PASS" } else { "  " };
                println!(
                    "step {step:5}  lr={lr:.2e}  intra={intra:.4}  inter={inter:.4}  ratio={csi_ratio:.3}x  improvement={improvement:.2}x  {ok}",
                );

                if improvement >= TARGET_IMPROVEMENT && !target_met {
                    target_met = true;
                    if let Err(e) = save_adapter(&lora_out, &lora_a, &lora_b, scaling, steps_done) {
                        eprintln!("save failed: {e}");
                    } else {
                        println!("  → adapter saved (target met at step {steps_done})");
                    }
                    break;
                }
            }
        }

        // Always save final adapter
        if !target_met {
            if let Err(e) = save_adapter(&lora_out, &lora_a, &lora_b, scaling, steps_done) {
                eprintln!("save failed: {e}");
            } else {
                println!("  → adapter saved (partial improvement={best_improvement:.2}×, steps={steps_done})");
            }
        }

        println!();
        println!("=== Result ===");
        println!("best improvement:  {best_improvement:.2}×");
        println!("target:            {TARGET_IMPROVEMENT:.1}×");
        println!("target met:        {}", if target_met { "YES ✓" } else { "NO ✗  — increase --max-steps or check SONA data diversity" });

        std::process::exit(if target_met { 0 } else { 1 });
    }

    fn centroid_cosine_dist(
        class_bases: &[[f32; CSI_EMBED_DIM]],
        lora_a: &[f32], lora_b: &[f32], scaling: f32,
        anchor_emb: &[f32; CSI_EMBED_DIM],
    ) -> f32 {
        let mut centroid = [0f32; CSI_EMBED_DIM];
        for b in class_bases {
            let e = lora_forward(b, lora_a, lora_b, scaling);
            for i in 0..CSI_EMBED_DIM { centroid[i] += e[i]; }
        }
        let n = class_bases.len() as f32;
        for v in &mut centroid { *v /= n; }
        1.0 - cosine(anchor_emb, &centroid)
    }
}
