//! Streaming memory-admission benchmark.
//!
//! Compares three online cluster-admission policies for streaming agent
//! memory:
//!   1. NearestCentroidThreshold – baseline: fixed cosine threshold
//!   2. MincutGatedAdmission     – candidate A: global min-cut, fixed tau
//!   3. AdaptiveMincutAdmission  – candidate B: global min-cut, self-calibrating tau
//!
//! ## Matched-budget calibration
//!
//! An early, uncalibrated run of this benchmark (fixed THRESHOLD=0.55,
//! TAU=0.35, preserved in the nightly research doc as raw evidence) produced
//! a degenerate baseline: at that threshold, 3289 of 4000 points spawned
//! their own cluster, giving a trivially "pure" (0.999) but useless
//! (recall@10 = 0.06) result — purity alone is gameable by over-splitting,
//! exactly the kind of ungrounded metric the nightly promotion gate exists
//! to catch. Comparing two online-clustering policies fairly means comparing
//! them at the *same* final cluster count (the same downstream reindex /
//! memory budget), not at independently hand-picked thresholds. This
//! binary-searches the baseline's threshold to match candidate A's natural
//! cluster count under a fixed `tau`, then reports purity/recall at that
//! matched budget. Candidate B's `tau` is *not* calibrated (defeats its own
//! purpose); it is reported at whatever cluster count its self-calibrating
//! threshold naturally lands on.
//!
//! Run:
//!   cargo run --release -p ruvector-memory-admission --bin benchmark
//!
//! Environment overrides:
//!   N_POINTS=4000 K_TRUE=8 DIMS=64 N_QUERIES=300 TAU=0.005

use ruvector_memory_admission::dataset::{StreamConfig, StreamDataset};
use ruvector_memory_admission::policy::{
    AdaptiveMincutAdmission, AdmissionPolicy, MincutGatedAdmission, NearestCentroidThreshold,
};
use ruvector_memory_admission::sq_l2;
use std::time::Instant;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

const K_EVAL: usize = 10;
const MAX_CLUSTERS: usize = 48; // computational safety valve, well above the 3x-K_true acceptance bound
const CALIBRATION_ITERATIONS: usize = 25;

// ─── acceptance thresholds (fixed before the calibrated run; see hypothesis
//     in docs/adr and docs/research/nightly) ────────────────────────────────
const MIN_PURITY_GAIN_A_PP: f64 = 0.0; // candidate A must not lose purity at matched cluster budget
const MIN_RECALL_GAIN_A_PP: f64 = 2.0; // candidate A must gain >= 2pp recall@10 at matched budget
const MAX_RECALL_REGRESSION_B_PP: f64 = 2.0; // candidate B (unmatched, self-calibrated) tolerance vs baseline
const MAX_MEAN_LATENCY_US: f64 = 500.0; // absolute ceiling: this is a write-path admission decision, not a hot query
const MAX_CLUSTER_COUNT_FACTOR: usize = 3; // final cluster count <= 3x K_true

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
fn mean_u128(vals: &[u128]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<u128>() as f64 / vals.len() as f64
}

struct RunResult {
    name: String,
    n_clusters: usize,
    purity: f64,
    recall_at_10: f64,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    mean_sim_ops: f64,
    mem_kb: f64,
}

/// Run one policy over the full stream, then evaluate purity and held-out
/// recall.
fn run_policy(
    mut policy: impl AdmissionPolicy,
    ds: &StreamDataset,
    queries: &[(Vec<f32>, usize)],
) -> RunResult {
    let n = ds.points.len();
    let mut assigned_cluster = vec![0usize; n];
    let mut latencies_us: Vec<u128> = Vec::with_capacity(n);
    let mut sim_ops_sum: u64 = 0;

    for (i, p) in ds.points.iter().enumerate() {
        let t0 = Instant::now();
        let d = policy.admit(&p.vector);
        latencies_us.push(t0.elapsed().as_micros());
        sim_ops_sum += d.sim_ops as u64;
        assigned_cluster[i] = d.cluster_id;
    }

    // ── purity: majority true-label fraction per final cluster ──────────
    let n_clusters = policy.n_clusters();
    let mut cluster_label_counts: Vec<Vec<usize>> = vec![vec![0usize; ds.k_true]; n_clusters];
    for (i, p) in ds.points.iter().enumerate() {
        cluster_label_counts[assigned_cluster[i]][p.true_cluster] += 1;
    }
    let correct: usize = cluster_label_counts
        .iter()
        .map(|counts| counts.iter().copied().max().unwrap_or(0))
        .sum();
    let purity = correct as f64 / n as f64;

    // ── held-out recall@10: does the query's assigned cluster contain the
    //    true top-10 nearest neighbours from the full stream corpus? ─────
    let mut recall_sum = 0f64;
    for (q, _true_label) in queries {
        let mut dists: Vec<(usize, f32)> = ds
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, sq_l2(q, &p.vector)))
            .collect();
        dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_top10: Vec<usize> = dists.iter().take(K_EVAL).map(|(i, _)| *i).collect();

        let decision = policy.decide(q);
        let query_cluster = decision.cluster_id;
        let hits = true_top10
            .iter()
            .filter(|&&id| assigned_cluster[id] == query_cluster)
            .count();
        recall_sum += hits as f64 / K_EVAL as f64;
    }
    let recall_at_10 = recall_sum / queries.len() as f64;

    latencies_us.sort_unstable();
    let mem_kb = (n_clusters * ds.dims * 4) as f64 / 1024.0;

    RunResult {
        name: policy.name().to_string(),
        n_clusters,
        purity,
        recall_at_10,
        mean_us: mean_u128(&latencies_us),
        p50_us: percentile(&latencies_us, 0.50) as f64,
        p95_us: percentile(&latencies_us, 0.95) as f64,
        mean_sim_ops: sim_ops_sum as f64 / n as f64,
        mem_kb,
    }
}

fn clusters_at_threshold(ds: &StreamDataset, threshold: f32) -> usize {
    let mut p = NearestCentroidThreshold::new(threshold);
    for pt in &ds.points {
        p.admit(&pt.vector);
    }
    p.n_clusters()
}

/// Binary search `threshold` so `NearestCentroidThreshold`'s final cluster
/// count matches `target` as closely as possible. Cluster count is
/// monotonically non-decreasing in `threshold` (a higher bar to merge means
/// more spawns), which is what makes this search well-defined.
fn calibrate_threshold(ds: &StreamDataset, target: usize) -> (f32, usize) {
    let mut lo = 0.0f32;
    let mut hi = 1.0f32;
    let mut best_threshold = lo;
    let mut best_clusters = clusters_at_threshold(ds, lo);
    let mut best_diff = best_clusters.abs_diff(target);

    for _ in 0..CALIBRATION_ITERATIONS {
        let mid = (lo + hi) / 2.0;
        let n = clusters_at_threshold(ds, mid);
        let diff = n.abs_diff(target);
        if diff < best_diff {
            best_diff = diff;
            best_threshold = mid;
            best_clusters = n;
        }
        if n > target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    (best_threshold, best_clusters)
}

fn print_header() {
    println!(
        "{:<26} {:>8} {:>8} {:>10} {:>10} {:>9} {:>9} {:>10} {:>9}",
        "Variant",
        "Clusters",
        "Purity",
        "Recall@10",
        "Mean(µs)",
        "p50(µs)",
        "p95(µs)",
        "SimOps",
        "Mem(KB)"
    );
    println!("{}", "-".repeat(112));
}
fn print_row(r: &RunResult) {
    println!(
        "{:<26} {:>8} {:>8.4} {:>10.4} {:>10.2} {:>9.0} {:>9.0} {:>10.1} {:>9.1}",
        r.name,
        r.n_clusters,
        r.purity,
        r.recall_at_10,
        r.mean_us,
        r.p50_us,
        r.p95_us,
        r.mean_sim_ops,
        r.mem_kb
    );
}

fn main() {
    println!("=== RuVector Memory Admission Benchmark ===");
    println!("OS:   {}", std::env::consts::OS);
    println!("Arch: {}", std::env::consts::ARCH);
    println!();

    let n_points = env_usize("N_POINTS", 4000);
    let k_true = env_usize("K_TRUE", 8);
    let dims = env_usize("DIMS", 64);
    let n_queries = env_usize("N_QUERIES", 300);
    let tau = env_f32("TAU", 0.005);

    let cfg = StreamConfig {
        n_points,
        k_true,
        dims,
        ..StreamConfig::default()
    };
    let ds = StreamDataset::generate(&cfg);
    let queries = ds.held_out_queries(&cfg, n_queries, 0xC0FF_EE00);

    // ── candidate A first, to establish the cluster-count target ─────────
    let candidate_a = run_policy(MincutGatedAdmission::new(tau, MAX_CLUSTERS), &ds, &queries);
    let target_clusters = candidate_a.n_clusters;

    // ── calibrate the baseline threshold to match that budget ────────────
    let (calibrated_threshold, calibrated_clusters) = calibrate_threshold(&ds, target_clusters);
    let baseline = run_policy(
        NearestCentroidThreshold::new(calibrated_threshold),
        &ds,
        &queries,
    );

    // ── candidate B: same bootstrap tau, but NOT calibrated — reports
    //    wherever its self-calibrating threshold naturally lands ──────────
    let candidate_b = run_policy(
        AdaptiveMincutAdmission::new(1.0, MAX_CLUSTERS, tau),
        &ds,
        &queries,
    );

    println!("Dataset:");
    println!("  Stream points:  {n_points}");
    println!("  True clusters:  {k_true}");
    println!("  Dimensions:     {dims}");
    println!("  Held-out qrys:  {n_queries}");
    println!("  Candidate tau:  {tau:.4}");
    println!(
        "  Max clusters:   {MAX_CLUSTERS} (safety valve; acceptance bound is {}x K_true = {})",
        MAX_CLUSTER_COUNT_FACTOR,
        MAX_CLUSTER_COUNT_FACTOR * k_true
    );
    println!();
    println!("Matched-budget calibration:");
    println!("  Candidate A cluster count (target): {target_clusters}");
    println!(
        "  Calibrated baseline threshold:      {calibrated_threshold:.4} -> {calibrated_clusters} clusters ({CALIBRATION_ITERATIONS} search iterations)"
    );
    println!(
        "  Candidate B cluster count (NOT calibrated, self-tuned): {}",
        candidate_b.n_clusters
    );
    println!();

    println!("Results:");
    print_header();
    print_row(&baseline);
    print_row(&candidate_a);
    print_row(&candidate_b);
    println!();

    // ── acceptance checks ────────────────────────────────────────────────
    let purity_gain_a_pp = (candidate_a.purity - baseline.purity) * 100.0;
    let recall_gain_a_pp = (candidate_a.recall_at_10 - baseline.recall_at_10) * 100.0;
    let recall_regress_b_pp = (baseline.recall_at_10 - candidate_b.recall_at_10) * 100.0;
    let cluster_bound = MAX_CLUSTER_COUNT_FACTOR * k_true;

    let a_purity_pass = purity_gain_a_pp >= MIN_PURITY_GAIN_A_PP;
    let a_recall_pass = recall_gain_a_pp >= MIN_RECALL_GAIN_A_PP;
    let a_latency_pass = candidate_a.mean_us <= MAX_MEAN_LATENCY_US;
    let a_cluster_pass = candidate_a.n_clusters <= cluster_bound;
    let a_pass = a_purity_pass && a_recall_pass && a_latency_pass && a_cluster_pass;

    // Candidate B's claim is narrower: without any hand-tuned matching to
    // the baseline's budget, does self-calibration still land close enough
    // to be practically useful (small recall tolerance), while respecting
    // the same latency and cluster-count bounds?
    let b_recall_pass = recall_regress_b_pp <= MAX_RECALL_REGRESSION_B_PP;
    let b_latency_pass = candidate_b.mean_us <= MAX_MEAN_LATENCY_US;
    let b_cluster_pass = candidate_b.n_clusters <= cluster_bound;
    let b_pass = b_recall_pass && b_latency_pass && b_cluster_pass;

    println!("Acceptance criteria — Candidate A (MincutGatedAdmission, fixed tau, matched cluster budget):");
    println!(
        "  purity gain vs matched baseline >= {MIN_PURITY_GAIN_A_PP:.1}pp: {purity_gain_a_pp:>7.2}pp -> {}",
        if a_purity_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  recall@10 gain vs matched baseline >= {MIN_RECALL_GAIN_A_PP:.1}pp: {recall_gain_a_pp:>7.2}pp -> {}",
        if a_recall_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  mean latency <= {MAX_MEAN_LATENCY_US:.0}µs:                 {:>7.2}µs -> {}",
        candidate_a.mean_us,
        if a_latency_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  final clusters <= {cluster_bound} ({MAX_CLUSTER_COUNT_FACTOR}x K_true):          {:>7}   -> {}",
        candidate_a.n_clusters,
        if a_cluster_pass { "PASS" } else { "FAIL" }
    );
    println!();

    println!("Acceptance criteria — Candidate B (AdaptiveMincutAdmission, self-calibrating tau, NOT matched):");
    println!(
        "  recall@10 regression <= {MAX_RECALL_REGRESSION_B_PP:.1}pp vs matched baseline: {recall_regress_b_pp:>7.2}pp -> {}",
        if b_recall_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  mean latency <= {MAX_MEAN_LATENCY_US:.0}µs:                 {:>7.2}µs -> {}",
        candidate_b.mean_us,
        if b_latency_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  final clusters <= {cluster_bound} ({MAX_CLUSTER_COUNT_FACTOR}x K_true):          {:>7}   -> {}",
        candidate_b.n_clusters,
        if b_cluster_pass { "PASS" } else { "FAIL" }
    );
    println!();

    let overall = a_pass && b_pass;
    println!(
        "Overall: {}",
        if overall {
            "ACCEPT — all mandatory thresholds passed"
        } else if a_pass || b_pass {
            "PARTIAL — at least one candidate passed, see per-candidate results above"
        } else {
            "REJECT — no candidate passed all mandatory thresholds"
        }
    );

    if !overall {
        std::process::exit(1);
    }
}
