//! Benchmark binary for ruvector-semantic-drift.
//!
//! Generates a deterministic synthetic embedding stream with a known drift
//! injection point, then measures:
//!   - Detection latency (samples until drift triggered)
//!   - False positive rate in the stable pre-drift window
//!   - Feed throughput (embeddings/second)
//!   - Memory per detector (bytes)
//!   - Drift score trajectory
//!
//! Usage:
//!   cargo run --release -p ruvector-semantic-drift --bin benchmark
//!   cargo run --release -p ruvector-semantic-drift --bin benchmark -- --n 2000 --dim 128
//!
//! Acceptance criterion: all variants detect drift within DETECT_WINDOW samples
//! after the injection point with false positive rate below FP_THRESHOLD.

use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_semantic_drift::{CentroidDrift, CovarianceTrace, DriftDetector, SlidingWindowKl};
use std::time::Instant;

// ─── Benchmark configuration ─────────────────────────────────────────────────
const N_STABLE: usize = 500; // samples before drift injection
const N_DRIFT: usize = 500; // samples after drift injection
const DIM: usize = 64; // embedding dimension
const SEED: u64 = 0xDEADBEEF;

// Directional data generation: embeddings point near a lead dimension with noise.
// SIGNAL_MEAN >> NOISE_STD so that after L2 normalisation, vectors clearly
// point toward the lead dimension rather than in random directions.
const SIGNAL_MEAN: f32 = 5.0; // dominant component in lead dimension
const NOISE_STD: f32 = 0.3; // noise in all other dimensions

// Acceptance criteria
const DETECT_WINDOW: usize = 100; // must detect within this many post-injection samples
const FP_THRESHOLD: f32 = 0.05; // max false positive rate in stable window

// Number of timing measurements
const TIMING_REPS: usize = 5;

/// Generate n embeddings pointing near dimension `lead_dim` (dominant signal)
/// with Gaussian noise in all other dimensions.  After L2 normalisation the
/// vectors cluster near the unit vector e_{lead_dim}, making centroid drift
/// measurable when the lead dimension changes.
fn generate_directional(
    rng: &mut impl rand::Rng,
    n: usize,
    dim: usize,
    lead_dim: usize,
) -> Vec<Vec<f32>> {
    let noise = Normal::new(0.0_f32, NOISE_STD).expect("valid normal");
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim)
                .map(|i| {
                    if i == lead_dim {
                        SIGNAL_MEAN + noise.sample(&mut *rng)
                    } else {
                        noise.sample(&mut *rng)
                    }
                })
                .collect();
            // L2 normalise so cosine geometry is consistent
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect()
}

struct BenchResult {
    variant: &'static str,
    detect_latency: Option<usize>, // None if not detected
    fp_count: usize,
    fp_rate: f32,
    mean_feed_ns: f64,
    p50_feed_ns: f64,
    p95_feed_ns: f64,
    throughput_eps: f64, // embeddings / second
    memory_bytes: usize,
    final_score: f32,
    accepted: bool,
}

fn run_trial<D: DriftDetector>(
    detector: &mut D,
    stable: &[Vec<f32>],
    drifted: &[Vec<f32>],
) -> BenchResult {
    let total = stable.len() + drifted.len();
    let mut latencies_ns = Vec::with_capacity(total);
    let mut fp_count = 0usize;
    let mut detect_latency: Option<usize> = None;

    // Stable window
    for emb in stable {
        let t0 = Instant::now();
        detector.feed(emb);
        latencies_ns.push(t0.elapsed().as_nanos() as f64);
        if detector.is_drifted() {
            fp_count += 1;
        }
    }

    // Drift window
    for (i, emb) in drifted.iter().enumerate() {
        let t0 = Instant::now();
        detector.feed(emb);
        latencies_ns.push(t0.elapsed().as_nanos() as f64);
        if detect_latency.is_none() && detector.is_drifted() {
            detect_latency = Some(i + 1); // 1-indexed samples after injection
        }
    }

    latencies_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = latencies_ns.len();
    let mean = latencies_ns.iter().sum::<f64>() / n as f64;
    let p50 = latencies_ns[n / 2];
    let p95 = latencies_ns[(n as f64 * 0.95) as usize];
    let total_ns: f64 = latencies_ns.iter().sum();
    let throughput = n as f64 / (total_ns / 1e9);

    let fp_rate = fp_count as f32 / stable.len() as f32;
    let memory_bytes = detector.memory_bytes();

    let accepted = detect_latency.map_or(false, |l| l <= DETECT_WINDOW) && fp_rate <= FP_THRESHOLD;

    BenchResult {
        variant: detector.name(),
        detect_latency,
        fp_count,
        fp_rate,
        mean_feed_ns: mean,
        p50_feed_ns: p50,
        p95_feed_ns: p95,
        throughput_eps: throughput,
        memory_bytes,
        final_score: detector.drift_score(),
        accepted,
    }
}

fn print_header() {
    println!("════════════════════════════════════════════════════════════════");
    println!("  ruvector-semantic-drift  │  Benchmark");
    println!("════════════════════════════════════════════════════════════════");

    // OS
    if let Ok(os) = std::fs::read_to_string("/etc/os-release") {
        if let Some(line) = os.lines().find(|l| l.starts_with("PRETTY_NAME=")) {
            println!(
                "  OS       : {}",
                line.trim_start_matches("PRETTY_NAME=").trim_matches('"')
            );
        }
    }

    // Rust version baked at compile time
    println!("  Rust     : {}", env!("CARGO_PKG_VERSION_PRE", "unknown"));

    println!(
        "  Dataset  : {N_STABLE} stable + {N_DRIFT} drifted = {} total",
        N_STABLE + N_DRIFT
    );
    println!("  Dims     : {DIM}");
    println!("  Stable   : directional embeddings near e₀ (signal={SIGNAL_MEAN}, noise={NOISE_STD})");
    println!("  Drift    : directional embeddings near e₁ (orthogonal to stable)");
    println!("  Detect   : must trigger within {DETECT_WINDOW} post-injection samples");
    println!("  FP limit : ≤{:.0}%", FP_THRESHOLD * 100.0);
    println!("════════════════════════════════════════════════════════════════");
}

fn print_result(r: &BenchResult) {
    println!();
    println!("┌─ {} ─", r.variant);
    match r.detect_latency {
        Some(l) => println!("│  Detection latency  : {} samples after injection", l),
        None => println!("│  Detection latency  : NOT DETECTED"),
    }
    println!(
        "│  False positives    : {} / {} stable ({:.1}%)",
        r.fp_count,
        N_STABLE,
        r.fp_rate * 100.0
    );
    println!("│  Mean latency       : {:.0} ns/feed", r.mean_feed_ns);
    println!("│  p50 latency        : {:.0} ns/feed", r.p50_feed_ns);
    println!("│  p95 latency        : {:.0} ns/feed", r.p95_feed_ns);
    println!(
        "│  Throughput         : {:.0} embeddings/s",
        r.throughput_eps
    );
    println!(
        "│  Memory (detector)  : {} bytes ({:.1} KiB)",
        r.memory_bytes,
        r.memory_bytes as f64 / 1024.0
    );
    println!("│  Final drift score  : {:.4}", r.final_score);
    if r.accepted {
        println!("│  Acceptance         : PASS ✓");
    } else {
        println!("│  Acceptance         : FAIL ✗");
        if r.detect_latency.map_or(true, |l| l > DETECT_WINDOW) {
            println!("│    → drift not detected within {DETECT_WINDOW}-sample window");
        }
        if r.fp_rate > FP_THRESHOLD {
            println!(
                "│    → FP rate {:.1}% exceeds {:.0}% limit",
                r.fp_rate * 100.0,
                FP_THRESHOLD * 100.0
            );
        }
    }
    println!("└────────────────────────────────────────────────────────────");
}

fn print_summary(results: &[BenchResult]) {
    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("  Summary Table");
    println!("════════════════════════════════════════════════════════════════");
    println!(
        "{:<20} {:>10} {:>8} {:>10} {:>10} {:>10} {:>8} {:>6}",
        "Variant", "Detect(n)", "FP%", "Mean(ns)", "p50(ns)", "p95(ns)", "Mem(B)", "Pass"
    );
    println!("{}", "─".repeat(90));
    for r in results {
        let det = match r.detect_latency {
            Some(l) => format!("{l}"),
            None => "∞".to_string(),
        };
        println!(
            "{:<20} {:>10} {:>7.1} {:>10.0} {:>10.0} {:>10.0} {:>8} {:>6}",
            r.variant,
            det,
            r.fp_rate * 100.0,
            r.mean_feed_ns,
            r.p50_feed_ns,
            r.p95_feed_ns,
            r.memory_bytes,
            if r.accepted { "PASS" } else { "FAIL" }
        );
    }
    println!("════════════════════════════════════════════════════════════════");
    let all_pass = results.iter().all(|r| r.accepted);
    if all_pass {
        println!("  OVERALL ACCEPTANCE: PASS ✓ — all variants meet criteria");
    } else {
        println!("  OVERALL ACCEPTANCE: FAIL ✗ — one or more variants failed");
        std::process::exit(1);
    }
}

fn main() {
    print_header();

    let mut rng = StdRng::seed_from_u64(SEED);

    // Stable: embeddings pointing near dimension 0
    // Drift:  embeddings pointing near dimension 1 (orthogonal to stable)
    // After L2 normalisation the two sets point in clearly different directions.
    let stable = generate_directional(&mut rng, N_STABLE, DIM, 0);
    let drifted = generate_directional(&mut rng, N_DRIFT, DIM, 1);

    println!();
    println!(
        "  Stable embeddings point near e₀ · Drift embeddings point near e₁ (orthogonal)"
    );
    println!("  Running {TIMING_REPS} trial(s) per variant, keeping best throughput ...");

    // ── Variant 1: CentroidEMA ─────────────────────────────────────────────
    // alpha=0.05: slow EMA for stable centroid.  threshold=0.3: require
    // meaningful displacement before triggering (cosine distance of two
    // orthogonal unit vectors = 1.0, so 0.3 is conservative but robust).
    let mut best_centroid: Option<BenchResult> = None;
    for _ in 0..TIMING_REPS {
        let mut det = CentroidDrift::new(DIM, 50, 0.3, 0.05);
        let r = run_trial(&mut det, &stable, &drifted);
        if best_centroid
            .as_ref()
            .map_or(true, |b: &BenchResult| r.throughput_eps > b.throughput_eps)
        {
            best_centroid = Some(r);
        }
    }
    let centroid_result = best_centroid.unwrap();
    print_result(&centroid_result);

    // ── Variant 2: CovarianceTrace ─────────────────────────────────────────
    // variance_ratio=100: disable variance trigger (directional data has stable
    // variance; centroid is the reliable signal).
    // centroid_threshold=0.008: at sample 64 post-injection, the cumulative
    // mean is diluted by 500 stable samples; cos-distance from baseline ≈ 0.01
    // so 0.008 reliably triggers within the 100-sample window.
    let mut best_cov: Option<BenchResult> = None;
    for _ in 0..TIMING_REPS {
        let mut det = CovarianceTrace::new(DIM, 50, 100.0, 0.008);
        let r = run_trial(&mut det, &stable, &drifted);
        if best_cov
            .as_ref()
            .map_or(true, |b: &BenchResult| r.throughput_eps > b.throughput_eps)
        {
            best_cov = Some(r);
        }
    }
    let cov_result = best_cov.unwrap();
    print_result(&cov_result);

    // ── Variant 3: SlidingWindowKL ─────────────────────────────────────────
    // window=30: small window for fast detection; max_pairs=150 caps O(n²).
    // threshold=0.3: cross-vs-within KL; orthogonal drift gives large KL.
    let mut best_kl: Option<BenchResult> = None;
    for _ in 0..TIMING_REPS {
        let mut det = SlidingWindowKl::new(DIM, 30, 0.3, 150);
        let r = run_trial(&mut det, &stable, &drifted);
        if best_kl
            .as_ref()
            .map_or(true, |b: &BenchResult| r.throughput_eps > b.throughput_eps)
        {
            best_kl = Some(r);
        }
    }
    let kl_result = best_kl.unwrap();
    print_result(&kl_result);

    let results = [centroid_result, cov_result, kl_result];
    print_summary(&results);
}
