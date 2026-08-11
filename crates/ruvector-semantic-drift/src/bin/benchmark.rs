//! Semantic drift benchmark for RuVector.
//!
//! Runs three detector variants against a controlled drift stream:
//!   1. WindowCentroid  — centroid-distance between sliding windows
//!   2. ProjectionDrift — RMS shift across 64 random projections
//!   3. SentinelQuery   — Jaccard degradation of sentinel top-k sets
//!
//! Dataset: n_base baseline vectors N(0, I), then n_drift drifted vectors
//!          N(shift·e₀, I) where e₀ is the unit vector in dimension 0.
//!
//! Acceptance test: all detectors must alert within `max_lag` vectors
//!                  after drift onset, with zero false positives in the
//!                  baseline phase.

use ruvector_semantic_drift::{
    dataset::{baseline, drifted},
    DriftDetector, ProjectionDriftDetector, SentinelQueryDetector, WindowCentroidDetector,
};
use std::time::{Duration, Instant};

const DIM: usize = 128;
const N_BASE: usize = 5_000;
const N_DRIFT: usize = 5_000;
// 8σ shift in dimension 0.  Centroid/projection detect via dimensional mean
// shift; sentinel detects via mean-all-distance ratio ≈ 64/(256+1) ≈ 0.25.
const DRIFT_SHIFT: f32 = 8.0;
const WINDOW: usize = 500;
const K_PROJ: usize = 64;
const N_SENTINEL: usize = 10;
const SNAPSHOT: usize = 300;
const TOP_K: usize = 5;
// sentinel threshold = 0.15 gives headroom: ratio ≈ 0.25 >> 0.15.
const SENTINEL_THRESHOLD: f32 = 0.15;
const MAX_LAG: usize = 2_000; // maximum vectors after onset before detection

fn percentile(sorted: &[u128], pct: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 * pct / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

struct BenchResult {
    name: String,
    update_times_ns: Vec<u128>,
    detection_lag: Option<usize>,
    false_positives: usize,
    memory_bytes: usize,
}

impl BenchResult {
    fn mean_ns(&self) -> f64 {
        if self.update_times_ns.is_empty() {
            return 0.0;
        }
        self.update_times_ns.iter().sum::<u128>() as f64 / self.update_times_ns.len() as f64
    }

    fn throughput_qps(&self) -> f64 {
        let mean = self.mean_ns();
        if mean == 0.0 {
            return 0.0;
        }
        1_000_000_000.0 / mean
    }
}

fn run_variant(
    mut det: Box<dyn DriftDetector>,
    base_vecs: &[Vec<f32>],
    drift_vecs: &[Vec<f32>],
) -> BenchResult {
    let name = det.name().to_owned();
    let mut update_times_ns: Vec<u128> = Vec::with_capacity(N_BASE + N_DRIFT);
    let mut false_positives = 0usize;
    let mut detection_lag: Option<usize> = None;

    // Baseline phase — count false positives.
    for v in base_vecs {
        let t0 = Instant::now();
        det.update(v);
        update_times_ns.push(t0.elapsed().as_nanos());

        if det.is_drifted() {
            false_positives += 1;
        }
    }

    // Drift phase — find detection lag.
    for (i, v) in drift_vecs.iter().enumerate() {
        let t0 = Instant::now();
        det.update(v);
        update_times_ns.push(t0.elapsed().as_nanos());

        if detection_lag.is_none() && det.is_drifted() {
            detection_lag = Some(i + 1);
        }
    }

    let memory_bytes = det.memory_bytes();

    BenchResult {
        name,
        update_times_ns,
        detection_lag,
        false_positives,
        memory_bytes,
    }
}

fn format_memory(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1_024 {
        format!("{:.0} KB", bytes as f64 / 1_024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn format_ns(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.1} ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.1} µs", ns / 1_000.0)
    } else {
        format!("{:.1} ns", ns)
    }
}

fn main() {
    // Print environment info.
    let rust_ver = std::process::Command::new("rustc")
        .args(["--version"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|_| "unknown".to_owned());

    println!("=== RuVector Semantic Drift Detector Benchmark ===");
    println!("Date       : 2026-08-01");
    println!("OS         : {}", std::env::consts::OS);
    println!("Arch       : {}", std::env::consts::ARCH);
    println!("Rust       : {rust_ver}");
    println!("Dataset    : {N_BASE} baseline + {N_DRIFT} drifted vectors");
    println!("Dimensions : {DIM}");
    println!("Drift shift: {DRIFT_SHIFT}σ in dimension 0");
    println!("Window     : {WINDOW} (centroid/projection detectors)");
    println!("Projections: {K_PROJ} (projection detector)");
    println!("Sentinels  : {N_SENTINEL} × top-{TOP_K} (sentinel detector)");
    println!("Snapshot   : {SNAPSHOT} (sentinel detector)");
    println!("Max lag    : {MAX_LAG} vectors after onset");
    println!();

    // Generate datasets.
    let t_gen = Instant::now();
    let base_vecs = baseline(N_BASE, DIM, 42);
    let drift_vecs = drifted(N_DRIFT, DIM, DRIFT_SHIFT, 42);
    let gen_ms: Duration = t_gen.elapsed();
    println!(
        "Dataset generation: {:.1} ms",
        gen_ms.as_secs_f64() * 1000.0
    );
    println!();

    // Detectors under test.
    let detectors: Vec<Box<dyn DriftDetector>> = vec![
        Box::new(WindowCentroidDetector::new(DIM, WINDOW, 0.10)),
        Box::new(ProjectionDriftDetector::new(DIM, K_PROJ, WINDOW, 0.15, 99)),
        Box::new(SentinelQueryDetector::new(
            DIM, N_SENTINEL, TOP_K, SNAPSHOT, SENTINEL_THRESHOLD, 77,
        )),
    ];

    let mut results: Vec<BenchResult> = detectors
        .into_iter()
        .map(|det| run_variant(det, &base_vecs, &drift_vecs))
        .collect();

    // Sort update_times for percentile calculation.
    for r in &mut results {
        r.update_times_ns.sort_unstable();
    }

    // Print results table.
    println!("Variant              | Mean update   | p50 update    | p95 update    | Updates/sec   | Detect lag | FP  | Memory");
    println!("---------------------|---------------|---------------|---------------|---------------|------------|-----|-------");

    let mut all_pass = true;
    for r in &results {
        let mean = r.mean_ns();
        let p50 = percentile(&r.update_times_ns, 50.0) as f64;
        let p95 = percentile(&r.update_times_ns, 95.0) as f64;
        let qps = r.throughput_qps();
        let lag_str = match r.detection_lag {
            Some(l) => format!("{l}"),
            None => "MISS".to_owned(),
        };
        let mem = format_memory(r.memory_bytes);

        println!(
            "{:<20} | {:<13} | {:<13} | {:<13} | {:<13} | {:<10} | {:<3} | {}",
            r.name,
            format_ns(mean),
            format_ns(p50),
            format_ns(p95),
            format!("{:.0}", qps),
            lag_str,
            r.false_positives,
            mem
        );

        // Acceptance check.
        let detected = r.detection_lag.map(|l| l <= MAX_LAG).unwrap_or(false);
        let no_fp = r.false_positives == 0;
        if !detected || !no_fp {
            all_pass = false;
        }
    }

    println!();
    println!("Acceptance criteria:");
    println!("  - Detect drift within {MAX_LAG} vectors after onset");
    println!("  - Zero false positives during {N_BASE}-vector baseline phase");
    println!();

    for r in &results {
        let detected = r.detection_lag.map(|l| l <= MAX_LAG).unwrap_or(false);
        let no_fp = r.false_positives == 0;
        let status = if detected && no_fp { "PASS" } else { "FAIL" };
        let det_str = match r.detection_lag {
            Some(l) if l <= MAX_LAG => format!("detected at lag={l}"),
            Some(l) => format!("detected late at lag={l}"),
            None => "NOT DETECTED".to_owned(),
        };
        let fp_str = if no_fp {
            "0 false positives".to_owned()
        } else {
            format!("{} false positives", r.false_positives)
        };
        println!("  {:<20} [{status}] — {det_str}, {fp_str}", r.name);
    }

    println!();
    if all_pass {
        println!("ACCEPTANCE: PASS — all detectors meet criteria.");
    } else {
        println!("ACCEPTANCE: FAIL — one or more detectors failed criteria.");
        std::process::exit(1);
    }
}
