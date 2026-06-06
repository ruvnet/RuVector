//! # ruvector-drift benchmark
//!
//! Measures drift detection performance for three variants:
//!   1. MeanShift — EMA mean-shift distance (baseline)
//!   2. CUSUM     — cumulative sum on reference-mean projections
//!   3. MMD-RFF   — Maximum Mean Discrepancy via Random Fourier Features
//!
//! Dataset: D-dimensional Gaussian vectors, split into a reference phase
//! (mean = 0) and a drift phase (mean = `drift_magnitude`).  All numbers are
//! produced by a deterministic `rand::rngs::StdRng` — no external data needed.
//!
//! ## Usage
//!
//!   cargo run --release -p ruvector-drift
//!   cargo run --release -p ruvector-drift -- --dim 128 --n 2000 --drift 2.0

use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_drift::{
    CusumDetector, DriftDetector, DriftReport, MeanShiftDetector, MmdRffDetector,
};
use std::time::Instant;

// ── CLI-style constants (override via environment for CI) ────────────────────

/// Vector dimension.
const DIM: usize = 128;
/// Total insertions (half reference, half drift).
const N: usize = 2_000;
/// Drift magnitude (L2 shift in mean vector).
const DRIFT: f32 = 2.0;
/// Number of random query probes for latency measurement.
const QUERIES: usize = 1_000;
/// Warm-up count for reference phase.
const WARM_UP: usize = N / 2;
/// CUSUM slack (half expected shift).
const CUSUM_SLACK: f64 = 1.0;
/// CUSUM alert threshold.
const CUSUM_THRESH: f32 = 5.0;
/// MeanShift alert threshold (L2 distance in embedding space).
const MEAN_THRESH: f32 = 0.5;
/// MMD-RFF alert threshold.
const MMD_THRESH: f32 = 0.05;
/// MMD-RFF feature count.
const RFF_FEATURES: usize = 256;
/// EMA alpha for MeanShift and MMD.
const ALPHA: f64 = 0.05;

fn main() {
    print_header();

    let mut rng = StdRng::seed_from_u64(42);

    // ── Generate dataset ─────────────────────────────────────────────────────
    // Phase 1: reference — N(0, 1) per dimension.
    let reference_vecs: Vec<Vec<f32>> = (0..WARM_UP)
        .map(|_| sample_gaussian(&mut rng, DIM, 0.0, 1.0))
        .collect();

    // Phase 2: drift — N(drift, 1) per dimension (all dims shifted).
    let drift_vecs: Vec<Vec<f32>> = (0..N - WARM_UP)
        .map(|_| sample_gaussian(&mut rng, DIM, DRIFT as f64, 1.0))
        .collect();

    println!("Dataset");
    println!("  reference phase : {WARM_UP} vectors, D={DIM}, mean=0");
    println!(
        "  drift phase     : {} vectors, D={DIM}, mean={DRIFT}",
        N - WARM_UP
    );
    println!("  drift magnitude : {DRIFT} (L2 per-dim shift)");
    println!("  latency queries : {QUERIES}");
    println!();

    // ── Run each variant ─────────────────────────────────────────────────────
    let ms_report = run_mean_shift(&reference_vecs, &drift_vecs, &mut rng);
    let cs_report = run_cusum(&reference_vecs, &drift_vecs, &mut rng);
    let mmd_report = run_mmd_rff(&reference_vecs, &drift_vecs, &mut rng);

    // ── Print results table ──────────────────────────────────────────────────
    print_table(&[ms_report.clone(), cs_report.clone(), mmd_report.clone()]);

    // ── Latency breakdown ────────────────────────────────────────────────────
    measure_latency(&[
        ("MeanShift", MEAN_THRESH),
        ("CUSUM", CUSUM_THRESH),
        ("MMD-RFF", MMD_THRESH),
    ]);

    // ── Acceptance test ──────────────────────────────────────────────────────
    println!("\n── Acceptance Test ─────────────────────────────────────────────");
    let mut pass = true;
    for report in &[&ms_report, &cs_report, &mmd_report] {
        let detected = report.detection_lag.is_some();
        let status = if detected { "PASS" } else { "FAIL" };
        println!(
            "  {:<12} detect={} baseline={:.4} trigger={:.4} → {status}",
            report.variant, detected, report.baseline_score, report.trigger_score
        );
        if !detected {
            pass = false;
        }
    }
    println!();
    if pass {
        println!("  ✓ All three detectors correctly identified the injected drift.");
        println!("  ACCEPTANCE RESULT: PASS");
    } else {
        eprintln!(
            "  ✗ One or more detectors failed to detect drift within the observation window."
        );
        eprintln!("  ACCEPTANCE RESULT: FAIL");
        std::process::exit(1);
    }
}

// ── Variant runners ──────────────────────────────────────────────────────────

fn run_mean_shift(
    ref_vecs: &[Vec<f32>],
    drift_vecs: &[Vec<f32>],
    _rng: &mut StdRng,
) -> DriftReport {
    let mut det = MeanShiftDetector::new(DIM, WARM_UP, ALPHA);

    // Reference phase
    for v in ref_vecs {
        det.insert(v);
    }
    let baseline_score = det.drift_score();

    // Drift phase — find first detection
    let mut detection_lag = None;
    let mut trigger_score = 0.0_f32;
    for (i, v) in drift_vecs.iter().enumerate() {
        det.insert(v);
        if detection_lag.is_none() && det.is_drifted(MEAN_THRESH) {
            detection_lag = Some(i + 1);
            trigger_score = det.drift_score();
        }
    }
    let final_score = det.drift_score();

    DriftReport {
        variant: "MeanShift".into(),
        reference_count: ref_vecs.len(),
        observation_count: drift_vecs.len(),
        baseline_score,
        trigger_score,
        detection_lag,
        final_score,
        memory_bytes: det.memory_bytes(),
    }
}

fn run_cusum(ref_vecs: &[Vec<f32>], drift_vecs: &[Vec<f32>], _rng: &mut StdRng) -> DriftReport {
    let mut det = CusumDetector::new(DIM, WARM_UP, CUSUM_SLACK);

    for v in ref_vecs {
        det.insert(v);
    }
    let baseline_score = det.drift_score();

    let mut detection_lag = None;
    let mut trigger_score = 0.0_f32;
    for (i, v) in drift_vecs.iter().enumerate() {
        det.insert(v);
        if detection_lag.is_none() && det.is_drifted(CUSUM_THRESH) {
            detection_lag = Some(i + 1);
            trigger_score = det.drift_score();
        }
    }
    let final_score = det.drift_score();

    DriftReport {
        variant: "CUSUM".into(),
        reference_count: ref_vecs.len(),
        observation_count: drift_vecs.len(),
        baseline_score,
        trigger_score,
        detection_lag,
        final_score,
        memory_bytes: det.memory_bytes(),
    }
}

fn run_mmd_rff(ref_vecs: &[Vec<f32>], drift_vecs: &[Vec<f32>], _rng: &mut StdRng) -> DriftReport {
    let mut det = MmdRffDetector::new(DIM, RFF_FEATURES, WARM_UP, 1.0, ALPHA, 99);

    for v in ref_vecs {
        det.insert(v);
    }
    let baseline_score = det.drift_score();

    let mut detection_lag = None;
    let mut trigger_score = 0.0_f32;
    for (i, v) in drift_vecs.iter().enumerate() {
        det.insert(v);
        if detection_lag.is_none() && det.is_drifted(MMD_THRESH) {
            detection_lag = Some(i + 1);
            trigger_score = det.drift_score();
        }
    }
    let final_score = det.drift_score();

    DriftReport {
        variant: "MMD-RFF".into(),
        reference_count: ref_vecs.len(),
        observation_count: drift_vecs.len(),
        baseline_score,
        trigger_score,
        detection_lag,
        final_score,
        memory_bytes: det.memory_bytes(),
    }
}

// ── Latency measurement ──────────────────────────────────────────────────────

fn measure_latency(variants: &[(&str, f32)]) {
    let mut rng = StdRng::seed_from_u64(777);
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|_| sample_gaussian(&mut rng, DIM, 0.0, 1.0))
        .collect();

    println!("── Insert Latency (ns/vector, {QUERIES} probes) ────────────────────");

    // MeanShift
    {
        let mut det = MeanShiftDetector::new(DIM, WARM_UP, ALPHA);
        let t0 = Instant::now();
        for q in &queries {
            det.insert(q);
        }
        let elapsed = t0.elapsed().as_nanos() as f64;
        let mean_ns = elapsed / QUERIES as f64;
        let score = det.drift_score();
        println!(
            "  MeanShift   mean={mean_ns:>8.1} ns/insert  score_after={score:.4}  mem={}B",
            det.memory_bytes()
        );
    }
    {
        let mut det = CusumDetector::new(DIM, WARM_UP, CUSUM_SLACK);
        let t0 = Instant::now();
        for q in &queries {
            det.insert(q);
        }
        let elapsed = t0.elapsed().as_nanos() as f64;
        let mean_ns = elapsed / QUERIES as f64;
        let score = det.drift_score();
        println!(
            "  CUSUM       mean={mean_ns:>8.1} ns/insert  score_after={score:.4}  mem={}B",
            det.memory_bytes()
        );
    }
    {
        let mut det = MmdRffDetector::new(DIM, RFF_FEATURES, WARM_UP, 1.0, ALPHA, 99);
        let t0 = Instant::now();
        for q in &queries {
            det.insert(q);
        }
        let elapsed = t0.elapsed().as_nanos() as f64;
        let mean_ns = elapsed / QUERIES as f64;
        let score = det.drift_score();
        println!(
            "  MMD-RFF     mean={mean_ns:>8.1} ns/insert  score_after={score:.4}  mem={}B",
            det.memory_bytes()
        );
    }
    println!();
    for (name, thresh) in variants {
        println!("  {name:<12} threshold={thresh:.4}");
    }
    println!();
}

// ── Print helpers ────────────────────────────────────────────────────────────

fn print_header() {
    println!("══════════════════════════════════════════════════════════════════");
    println!("  ruvector-drift  Streaming Semantic Drift Detection Benchmark");
    println!("══════════════════════════════════════════════════════════════════");
    println!("  OS   : {}", std::env::consts::OS);
    println!("  Arch : {}", std::env::consts::ARCH);
    println!("  Dims : {DIM}");
    println!("  N    : {N}");
    println!("  Drift: {DRIFT}");
    println!();
}

fn print_table(reports: &[DriftReport]) {
    println!("── Detection Results ───────────────────────────────────────────────");
    println!(
        "{:<12} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Variant", "Ref#", "Drift#", "Baseline", "FinalScore", "Lag(vecs)", "Mem(B)"
    );
    println!("{}", "─".repeat(78));
    for r in reports {
        let lag = r.detection_lag.map_or("NONE".into(), |l| l.to_string());
        println!(
            "{:<12} {:>8} {:>10} {:>10.4} {:>10.4} {:>10} {:>10}",
            r.variant,
            r.reference_count,
            r.observation_count,
            r.baseline_score,
            r.final_score,
            lag,
            r.memory_bytes
        );
    }
    println!();
}

// ── Dataset generation ───────────────────────────────────────────────────────

/// Sample a D-dimensional Gaussian vector N(mean, std) using Box-Muller.
fn sample_gaussian(rng: &mut StdRng, dim: usize, mean: f64, std: f64) -> Vec<f32> {
    use std::f64::consts::PI;
    let mut out = Vec::with_capacity(dim);
    while out.len() < dim {
        let u1 = rng.gen::<f64>().max(1e-14);
        let u2 = rng.gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt() * std;
        let theta = 2.0 * PI * u2;
        out.push((mean + r * theta.cos()) as f32);
        if out.len() < dim {
            out.push((mean + r * theta.sin()) as f32);
        }
    }
    out.truncate(dim);
    out
}
