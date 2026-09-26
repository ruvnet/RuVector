//! Semantic drift detection benchmark.
//!
//! Measures drift score, detection latency, and false-positive rate
//! for three detector variants under abrupt and gradual distribution shift.
//!
//! Run with:
//!   cargo run --release -p ruvector-drift-detect --bin benchmark

use std::time::Instant;

use ruvector_drift_detect::{
    dataset::{sample_gradual_drift, sample_normal, sample_partial_drift},
    CentroidDriftDetector, DriftDetector, DriftReport, GlobalStatsDriftDetector,
    NeighborhoodDriftDetector,
};

// ─── Configuration ───────────────────────────────────────────────────────────

const BASELINE_N: usize = 5_000;
const DRIFT_N: usize = 2_000;
const CONTROL_N: usize = 2_000;
const DIMS: usize = 128;
const K_CENTROIDS: usize = 32;
const K_NEIGHBORS: usize = 10;
const N_ANCHORS: usize = 80;
// Shift all 128 dims by 3σ for unambiguous detection signal
const DRIFT_DIMS: usize = 128;
const DRIFT_MEAN: f32 = 3.0;

// Per-variant thresholds: each variant has a different natural scale.
const THRESHOLD_GLOBAL: f64 = 2.0;
const THRESHOLD_CENTROID: f64 = 0.3;
const THRESHOLD_NEIGHBORHOOD: f64 = 0.3;

// ─── Timing helpers ──────────────────────────────────────────────────────────

fn time_observe<D: DriftDetector>(det: &mut D, vecs: &[Vec<f32>]) -> u64 {
    let t = Instant::now();
    for v in vecs {
        det.observe(v);
    }
    t.elapsed().as_nanos() as u64 / vecs.len().max(1) as u64
}

fn time_score<D: DriftDetector>(det: &D) -> (f64, u64) {
    let t = Instant::now();
    let s = det.drift_score();
    (s, t.elapsed().as_nanos() as u64)
}

// ─── Result types ─────────────────────────────────────────────────────────────

struct VariantResult {
    variant: &'static str,
    observe_ns: u64,
    drift_score: f64,
    score_ns: u64,
    control_score: f64,
    threshold: f64,
}

impl VariantResult {
    fn detected(&self) -> bool {
        self.drift_score >= self.threshold
    }
    fn false_positive(&self) -> bool {
        self.control_score >= self.threshold
    }
    fn pass(&self) -> bool {
        self.detected() && !self.false_positive()
    }
}

// ─── Per-variant runners ──────────────────────────────────────────────────────

fn run_global(
    baseline: &[Vec<f32>],
    drifted: &[Vec<f32>],
    control: &[Vec<f32>],
) -> VariantResult {
    // Drift run
    let mut det = GlobalStatsDriftDetector::new(DIMS);
    for v in baseline { det.observe(v); }
    det.snapshot();
    let obs_ns = time_observe(&mut det, drifted);
    let (drift_score, score_ns) = time_score(&det);

    // False-positive control: fresh detector, same distribution as baseline
    let mut det2 = GlobalStatsDriftDetector::new(DIMS);
    for v in baseline { det2.observe(v); }
    det2.snapshot();
    for v in control { det2.observe(v); }
    let (control_score, _) = time_score(&det2);

    VariantResult {
        variant: "GlobalStats",
        observe_ns: obs_ns,
        drift_score,
        score_ns,
        control_score,
        threshold: THRESHOLD_GLOBAL,
    }
}

fn run_centroid(
    baseline: &[Vec<f32>],
    drifted: &[Vec<f32>],
    control: &[Vec<f32>],
) -> VariantResult {
    let mut det = CentroidDriftDetector::new(DIMS, K_CENTROIDS);
    for v in baseline { det.observe(v); }
    det.snapshot();
    let obs_ns = time_observe(&mut det, drifted);
    let (drift_score, score_ns) = time_score(&det);

    let mut det2 = CentroidDriftDetector::new(DIMS, K_CENTROIDS);
    for v in baseline { det2.observe(v); }
    det2.snapshot();
    for v in control { det2.observe(v); }
    let (control_score, _) = time_score(&det2);

    VariantResult {
        variant: "CentroidDrift(K=32)",
        observe_ns: obs_ns,
        drift_score,
        score_ns,
        control_score,
        threshold: THRESHOLD_CENTROID,
    }
}

fn run_neighborhood(
    baseline: &[Vec<f32>],
    drifted: &[Vec<f32>],
    control: &[Vec<f32>],
) -> VariantResult {
    let mut det = NeighborhoodDriftDetector::new(DIMS, K_NEIGHBORS, N_ANCHORS);
    for v in baseline { det.observe(v); }
    det.snapshot();
    let obs_ns = time_observe(&mut det, drifted);
    let (drift_score, score_ns) = time_score(&det);

    let mut det2 = NeighborhoodDriftDetector::new(DIMS, K_NEIGHBORS, N_ANCHORS);
    for v in baseline { det2.observe(v); }
    det2.snapshot();
    for v in control { det2.observe(v); }
    let (control_score, _) = time_score(&det2);

    VariantResult {
        variant: "NeighborhoodRecall",
        observe_ns: obs_ns,
        drift_score,
        score_ns,
        control_score,
        threshold: THRESHOLD_NEIGHBORHOOD,
    }
}

// ─── Gradual drift (no false-positive test) ───────────────────────────────────

struct GradualResult {
    variant: &'static str,
    observe_ns: u64,
    drift_score: f64,
    score_ns: u64,
}

fn run_gradual_all(baseline: &[Vec<f32>], gradual: &[Vec<f32>]) -> Vec<GradualResult> {
    vec![
        {
            let mut det = GlobalStatsDriftDetector::new(DIMS);
            for v in baseline { det.observe(v); }
            det.snapshot();
            let obs_ns = time_observe(&mut det, gradual);
            let (score, sns) = time_score(&det);
            GradualResult { variant: "GlobalStats", observe_ns: obs_ns, drift_score: score, score_ns: sns }
        },
        {
            let mut det = CentroidDriftDetector::new(DIMS, K_CENTROIDS);
            for v in baseline { det.observe(v); }
            det.snapshot();
            let obs_ns = time_observe(&mut det, gradual);
            let (score, sns) = time_score(&det);
            GradualResult { variant: "CentroidDrift(K=32)", observe_ns: obs_ns, drift_score: score, score_ns: sns }
        },
        {
            let mut det = NeighborhoodDriftDetector::new(DIMS, K_NEIGHBORS, N_ANCHORS);
            for v in baseline { det.observe(v); }
            det.snapshot();
            let obs_ns = time_observe(&mut det, gradual);
            let (score, sns) = time_score(&det);
            GradualResult { variant: "NeighborhoodRecall", observe_ns: obs_ns, drift_score: score, score_ns: sns }
        },
    ]
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    print_header();

    let baseline = sample_normal(BASELINE_N, DIMS, 0.0, 1.0, 1001);
    let drifted = sample_partial_drift(DRIFT_N, DIMS, DRIFT_DIMS, DRIFT_MEAN, 1.0, 2002);
    let control = sample_normal(CONTROL_N, DIMS, 0.0, 1.0, 3003);
    let gradual = sample_gradual_drift(DRIFT_N, DIMS, 0.3, DRIFT_MEAN, 4004);

    // ── Abrupt drift scenario ─────────────────────────────────────────────
    println!("\n=== Scenario A: Abrupt Full Drift ({DRIFT_DIMS}/{DIMS} dims shifted {DRIFT_MEAN}σ) ===\n");
    println!(
        "  {:<24} {:>10} {:>10} {:>12} {:>12} {:>10} {:>8}",
        "Variant", "Drift Score", "Ctrl Score", "Observe(ns)", "Score(ns)", "Threshold", "Pass?"
    );
    println!("  {}", "-".repeat(92));

    let results = vec![
        run_global(&baseline, &drifted, &control),
        run_centroid(&baseline, &drifted, &control),
        run_neighborhood(&baseline, &drifted, &control),
    ];

    for r in &results {
        println!(
            "  {:<24} {:>10.4} {:>10.4} {:>12} {:>12} {:>10.2} {:>8}",
            r.variant, r.drift_score, r.control_score,
            r.observe_ns, r.score_ns, r.threshold,
            if r.pass() { "PASS" } else { "FAIL" }
        );
    }

    // ── Gradual drift scenario ────────────────────────────────────────────
    println!("\n=== Scenario B: Gradual Drift (30%→100% ramp over 2000 vectors) ===\n");
    println!(
        "  {:<24} {:>10} {:>12} {:>12} {:>8}",
        "Variant", "Drift Score", "Observe(ns)", "Score(ns)", "Detects?"
    );
    println!("  {}", "-".repeat(72));

    let gradual_results = run_gradual_all(&baseline, &gradual);
    for r in &gradual_results {
        println!(
            "  {:<24} {:>10.4} {:>12} {:>12} {:>8}",
            r.variant, r.drift_score, r.observe_ns, r.score_ns,
            if r.drift_score > 0.1 { "YES" } else { "WEAK" }
        );
    }

    // ── Memory estimates ──────────────────────────────────────────────────
    println!("\n=== Memory Estimates (n={BASELINE_N}, D={DIMS}) ===\n");
    println!("  GlobalStats:         {} bytes  (2 × D × 8 bytes f64)", 2 * DIMS * 8);
    println!(
        "  CentroidDrift(K=32): {} bytes  (2 × K × D × 4 bytes f32)",
        2 * K_CENTROIDS * DIMS * 4
    );
    println!(
        "  NeighborhoodRecall:  {} bytes  (n × D × 4 bytes f32  [stores all vectors])",
        BASELINE_N * DIMS * 4
    );

    // ── Acceptance test ───────────────────────────────────────────────────
    println!("\n=== Acceptance Test ===\n");
    let mut all_pass = true;
    for r in &results {
        if r.detected() && !r.false_positive() {
            println!("  PASS: {} drift={:.4} (≥{:.2}) ctrl={:.4} (<{:.2})",
                r.variant, r.drift_score, r.threshold, r.control_score, r.threshold);
        } else if !r.detected() {
            println!("  FAIL: {} did not detect drift (score={:.4} < {:.2})",
                r.variant, r.drift_score, r.threshold);
            all_pass = false;
        } else {
            println!("  FAIL: {} false positive (ctrl score={:.4} ≥ {:.2})",
                r.variant, r.control_score, r.threshold);
            all_pass = false;
        }
    }

    println!();
    if all_pass {
        println!("  RESULT: PASS — all acceptance criteria met");
    } else {
        println!("  RESULT: FAIL — one or more criteria not met");
        std::process::exit(1);
    }

    // ── DriftReport summary ───────────────────────────────────────────────
    println!("\n=== DriftReport Summary ===\n");
    for r in &results {
        DriftReport {
            variant: r.variant,
            baseline_n: BASELINE_N,
            post_snapshot_n: DRIFT_N,
            drift_score: r.drift_score,
            is_drifted: r.detected(),
            threshold: r.threshold,
            observe_ns_mean: r.observe_ns,
            score_ns: r.score_ns,
        }
        .print();
    }
}

fn print_header() {
    println!("ruvector-drift-detect benchmark");
    println!("================================");
    println!("Dataset:     baseline={BASELINE_N} drift={DRIFT_N} control={CONTROL_N}");
    println!("Dimensions:  {DIMS}");
    println!("Drift type:  full ({DRIFT_DIMS}/{DIMS} dims shifted {DRIFT_MEAN}σ) + separate gradual");
    println!("Thresholds:  GlobalStats={THRESHOLD_GLOBAL}  CentroidDrift={THRESHOLD_CENTROID}  Neighborhood={THRESHOLD_NEIGHBORHOOD}");
    println!("Variants:    GlobalStats | CentroidDrift(K={K_CENTROIDS}) | NeighborhoodRecall(A={N_ANCHORS},k={K_NEIGHBORS})");
    println!();
    println!("OS:          {}", std::env::consts::OS);
    println!("Arch:        {}", std::env::consts::ARCH);
}
