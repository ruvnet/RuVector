//! Standalone benchmark for ruvector-drift detectors.
//!
//! Generates synthetic vector streams with controlled drift and measures:
//! - Observation throughput (vectors/sec)
//! - Mean / p50 / p95 latency per observation
//! - True positive rate (drift correctly detected on drifted data)
//! - False positive rate (alert on no-drift data)
//! - Memory estimate
//!
//! Run: cargo run --release -p ruvector-drift --bin benchmark

use rand::{rngs::SmallRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_drift::{
    centroid::CentroidDriftDetector, graph::GraphDriftDetector, mmd::MmdDriftDetector,
    DriftDetector, DriftReport,
};
use std::time::Instant;

// ── dataset parameters ─────────────────────────────────────────────────────

const DIMS: usize = 128;
const REF_SIZE: usize = 1_000;
const QUERY_SIZE: usize = 1_000;
const WINDOW_SIZE: usize = 500;

// ── helpers ────────────────────────────────────────────────────────────────

fn normal_vecs(n: usize, dims: usize, mean: f32, std: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let dist = Normal::new(mean, std).unwrap();
    (0..n)
        .map(|_| (0..dims).map(|_| dist.sample(&mut rng)).collect())
        .collect()
}

/// Gaussian mixture: half from N(mean, 1) and half from N(-mean, 1)
fn gmm_vecs(n: usize, dims: usize, separation: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let dist_a = Normal::new(separation, 1.0f32).unwrap();
    let dist_b = Normal::new(-separation, 1.0f32).unwrap();
    (0..n)
        .map(|i| {
            let dist = if i % 2 == 0 { &dist_a } else { &dist_b };
            (0..dims).map(|_| dist.sample(&mut rng)).collect()
        })
        .collect()
}

/// Measure per-observation latency and return sorted sample in nanoseconds.
fn measure_latencies<D: DriftDetector>(det: &mut D, vecs: &[Vec<f32>]) -> Vec<u64> {
    let mut latencies = Vec::with_capacity(vecs.len());
    for v in vecs {
        let t0 = Instant::now();
        det.observe(v);
        latencies.push(t0.elapsed().as_nanos() as u64);
    }
    latencies.sort_unstable();
    latencies
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    let idx = ((sorted.len() as f64 * p / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn throughput(latencies: &[u64]) -> f64 {
    let total_ns: u64 = latencies.iter().sum();
    if total_ns == 0 {
        return f64::INFINITY;
    }
    (latencies.len() as f64) / (total_ns as f64 / 1e9)
}

struct BenchResult {
    method: &'static str,
    dataset: &'static str,
    n_ref: usize,
    n_queries: usize,
    dims: usize,
    mean_lat_ns: f64,
    p50_ns: u64,
    p95_ns: u64,
    throughput_vps: f64,
    memory_bytes: usize,
    report: DriftReport,
}

impl BenchResult {
    fn print_header() {
        println!(
            "{:<12} {:<22} {:>6} {:>6} {:>4} {:>10} {:>8} {:>8} {:>12} {:>12} {:>8} {:<8}",
            "Method",
            "Dataset",
            "N_ref",
            "N_qry",
            "Dim",
            "Mean(ns)",
            "p50(ns)",
            "p95(ns)",
            "QPS",
            "Mem(bytes)",
            "DriftMag",
            "Alert?"
        );
        println!("{}", "-".repeat(130));
    }

    fn print(&self) {
        println!(
            "{:<12} {:<22} {:>6} {:>6} {:>4} {:>10.1} {:>8} {:>8} {:>12.0} {:>12} {:>8.4} {:<8}",
            self.method,
            self.dataset,
            self.n_ref,
            self.n_queries,
            self.dims,
            self.mean_lat_ns,
            self.p50_ns,
            self.p95_ns,
            self.throughput_vps,
            self.memory_bytes,
            self.report.magnitude,
            if self.report.drift_detected {
                "DRIFT"
            } else {
                "ok"
            }
        );
    }
}

// ── memory estimates ───────────────────────────────────────────────────────

fn centroid_mem(dims: usize, window: usize) -> usize {
    // ref_centroid + cur_sum + cur_buffer of vecs
    2 * dims * 4 + window * dims * 4
}

fn mmd_mem(dims: usize, n_features: usize, window: usize) -> usize {
    // weights + biases + ref_mean + cur_mean + eviction buf
    n_features * dims * 4 + n_features * 4 + 2 * n_features * 4 + window * dims * 4
}

fn graph_mem(ref_size: usize, window: usize, dims: usize) -> usize {
    (ref_size + window) * dims * 4
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() {
    // Print environment
    println!("=== ruvector-drift benchmark ===\n");
    if let Ok(v) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        print!("Rust:  {}", String::from_utf8_lossy(&v.stdout));
    }
    println!("OS:    {}", std::env::consts::OS);
    println!("Arch:  {}", std::env::consts::ARCH);
    println!("Dims:  {DIMS}  |  Ref size: {REF_SIZE}  |  Query size: {QUERY_SIZE}  |  Window: {WINDOW_SIZE}");
    println!();

    let ref_data = normal_vecs(REF_SIZE, DIMS, 0.0, 1.0, 42);

    // Three query datasets:
    // 1. Null: same distribution N(0,1)
    // 2. Centroid shift: N(2.0, 1.0) — clear mean shift
    // 3. GMM structural: mixture N(±3, 1) — same global mean≈0, different structure
    let datasets: &[(&str, Vec<Vec<f32>>)] = &[
        (
            "null (no drift)",
            normal_vecs(QUERY_SIZE, DIMS, 0.0, 1.0, 99),
        ),
        (
            "centroid shift+2σ",
            normal_vecs(QUERY_SIZE, DIMS, 2.0, 1.0, 77),
        ),
        ("GMM structural", gmm_vecs(QUERY_SIZE, DIMS, 3.0, 55)),
    ];

    let sigma = (DIMS as f32).sqrt(); // heuristic bandwidth
    const N_FEATURES: usize = 128;
    const K: usize = 10;

    let mut all_results: Vec<BenchResult> = Vec::new();

    BenchResult::print_header();

    for (ds_name, query_vecs) in datasets {
        // ── Centroid ──
        {
            let mut det = CentroidDriftDetector::new(&ref_data, WINDOW_SIZE, 0.3);
            let lats = measure_latencies(&mut det, query_vecs);
            let mean_lat = lats.iter().sum::<u64>() as f64 / lats.len() as f64;
            let report = det.report();
            let mem = centroid_mem(DIMS, WINDOW_SIZE);
            let r = BenchResult {
                method: "centroid",
                dataset: ds_name,
                n_ref: REF_SIZE,
                n_queries: QUERY_SIZE,
                dims: DIMS,
                mean_lat_ns: mean_lat,
                p50_ns: percentile(&lats, 50.0),
                p95_ns: percentile(&lats, 95.0),
                throughput_vps: throughput(&lats),
                memory_bytes: mem,
                report,
            };
            r.print();
            all_results.push(r);
        }

        // ── MMD-RFF ──
        {
            let mut det = MmdDriftDetector::new(&ref_data, N_FEATURES, sigma, WINDOW_SIZE, 0.05);
            let lats = measure_latencies(&mut det, query_vecs);
            let mean_lat = lats.iter().sum::<u64>() as f64 / lats.len() as f64;
            let report = det.report();
            let mem = mmd_mem(DIMS, N_FEATURES, WINDOW_SIZE);
            let r = BenchResult {
                method: "mmd-rff",
                dataset: ds_name,
                n_ref: REF_SIZE,
                n_queries: QUERY_SIZE,
                dims: DIMS,
                mean_lat_ns: mean_lat,
                p50_ns: percentile(&lats, 50.0),
                p95_ns: percentile(&lats, 95.0),
                throughput_vps: throughput(&lats),
                memory_bytes: mem,
                report,
            };
            r.print();
            all_results.push(r);
        }

        // ── Graph k-NN ──
        {
            // Graph uses ref_size=200 for tractable O(n^2) at report time
            let ref_small = normal_vecs(200, DIMS, 0.0, 1.0, 42);
            let query_small = &query_vecs[..200];
            let mut det = GraphDriftDetector::new(&ref_small, K, 200, 0.25);
            let lats = measure_latencies(&mut det, query_small);
            let mean_lat = lats.iter().sum::<u64>() as f64 / lats.len() as f64;
            let report = det.report();
            let mem = graph_mem(200, 200, DIMS);
            let r = BenchResult {
                method: "graph-knn",
                dataset: ds_name,
                n_ref: 200,
                n_queries: 200,
                dims: DIMS,
                mean_lat_ns: mean_lat,
                p50_ns: percentile(&lats, 50.0),
                p95_ns: percentile(&lats, 95.0),
                throughput_vps: throughput(&lats),
                memory_bytes: mem,
                report,
            };
            r.print();
            all_results.push(r);
        }
    }

    println!();

    // ── Acceptance test ──────────────────────────────────────────────────
    println!("=== Acceptance Test ===");
    println!("Criterion: centroid, mmd-rff, and graph-knn must all:");
    println!("  - NOT alert on null (no-drift) data");
    println!("  - ALERT on centroid-shift+2σ drifted data");
    println!();

    let null_centroid = all_results
        .iter()
        .find(|r| r.method == "centroid" && r.dataset == "null (no drift)")
        .unwrap();
    let drift_centroid = all_results
        .iter()
        .find(|r| r.method == "centroid" && r.dataset == "centroid shift+2σ")
        .unwrap();
    let null_mmd = all_results
        .iter()
        .find(|r| r.method == "mmd-rff" && r.dataset == "null (no drift)")
        .unwrap();
    let drift_mmd = all_results
        .iter()
        .find(|r| r.method == "mmd-rff" && r.dataset == "centroid shift+2σ")
        .unwrap();
    let null_graph = all_results
        .iter()
        .find(|r| r.method == "graph-knn" && r.dataset == "null (no drift)")
        .unwrap();
    let drift_graph = all_results
        .iter()
        .find(|r| r.method == "graph-knn" && r.dataset == "centroid shift+2σ")
        .unwrap();

    let checks = [
        (
            "centroid / null  → no alert",
            !null_centroid.report.drift_detected,
        ),
        (
            "centroid / drift → ALERT",
            drift_centroid.report.drift_detected,
        ),
        (
            "mmd-rff  / null  → no alert",
            !null_mmd.report.drift_detected,
        ),
        ("mmd-rff  / drift → ALERT", drift_mmd.report.drift_detected),
        (
            "graph-knn/ null  → no alert",
            !null_graph.report.drift_detected,
        ),
        (
            "graph-knn/ drift → ALERT",
            drift_graph.report.drift_detected,
        ),
    ];

    let mut passed = true;
    for (label, ok) in &checks {
        println!("  [{}] {}", if *ok { "PASS" } else { "FAIL" }, label);
        if !ok {
            passed = false;
        }
    }

    println!();
    if passed {
        println!("ACCEPTANCE RESULT: PASS — all detectors behave correctly");
    } else {
        println!("ACCEPTANCE RESULT: FAIL — see above");
        std::process::exit(1);
    }
}
