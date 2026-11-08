//! Streaming memory-admission benchmark.
//!
//! Compares four online cluster-admission policies for streaming agent
//! memory, on both a static stream and a regime-shift ("drift") stream:
//!   1. NearestCentroidThreshold    – baseline: fixed cosine threshold
//!   2. MincutGatedAdmission        – candidate A: global min-cut, fixed tau
//!   3. AdaptiveMincutAdmission     – candidate B: global min-cut, self-calibrating tau
//!      (negative result: drifts and blows through the cluster-count safety valve)
//!   4. GuardedConditionedAdmission – candidate C: global min-cut, tau conditioned on
//!      local features and recalibrated by a `darwin_guard`-screened `(1+1)`-ES
//!      targeting a warm-up-measured spawn rate (addresses candidate B's failure)
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

use ruvector_memory_admission::conditioned::GuardedConditionedAdmission;
use ruvector_memory_admission::dataset::{DriftStreamConfig, StreamConfig, StreamDataset};
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

// Candidate C (GuardedConditionedAdmission) — fixed BEFORE the benchmark run
// that produced this file's committed numbers (see docs/adr and the nightly
// research doc). Candidate C's claim is narrower than "beats the baseline":
// it must be a safe, zero-hand-tuned DROP-IN for candidate A (no worse,
// within noise) on the regime it was tuned for, AND it must beat a
// transplanted-unchanged candidate A once the regime drifts — the one thing
// a fixed tau structurally cannot do.
const MAX_CLUSTER_FACTOR_VS_A: f64 = 1.5; // candidate C must not blow up the way candidate B did (which hit 48 vs A's ~17)
const MAX_RECALL_REGRESSION_C_VS_A_PP: f64 = 1.0; // tighter than B's 2pp tolerance: C aims to replace A, not just beat the naive baseline
const MAX_PURITY_REGRESSION_C_VS_A_PP: f64 = 1.0;
const MIN_DRIFT_RECALL_GAIN_C_VS_A_PP: f64 = 2.0; // the adaptivity claim: self-calibration must earn its keep under drift
const DRIFT_SEED: u64 = 0x0FEE_D000;
const ES_SEED: u64 = 0xC0DE_1234;

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

    // ── candidate C: same bootstrap tau as A/B, self-calibrates via a
    //    guarded, feature-conditioned (1+1)-ES instead of an unguarded
    //    global cut-weight statistic ────────────────────────────────────
    let candidate_c = run_policy(
        GuardedConditionedAdmission::new(tau, MAX_CLUSTERS, ES_SEED),
        &ds,
        &queries,
    );
    // Re-run a fresh instance to read guard stats (run_policy consumes the
    // policy by value for its own cost accounting, same as A/B's pattern).
    let mut c_for_stats = GuardedConditionedAdmission::new(tau, MAX_CLUSTERS, ES_SEED);
    for pt in &ds.points {
        c_for_stats.admit(&pt.vector);
    }
    let c_guard_stats = c_for_stats.guard_stats();
    let c_target_rate = c_for_stats.target_spawn_rate();

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
    println!(
        "  Candidate C cluster count (NOT calibrated, guarded self-tuned): {}",
        candidate_c.n_clusters
    );
    println!(
        "  Candidate C warm-up target spawn rate: {}",
        c_target_rate
            .map(|r| format!("{r:.4}"))
            .unwrap_or_else(|| "n/a (stream shorter than warmup_len)".to_string())
    );
    println!(
        "  Candidate C guard: {} attempts, {} accepted, {} rejected (non_finite={}, out_of_bounds={}, degenerate={}, not_improving={})",
        c_guard_stats.attempts,
        c_guard_stats.accepted,
        c_guard_stats.attempts - c_guard_stats.accepted,
        c_guard_stats.rejected_non_finite,
        c_guard_stats.rejected_out_of_bounds,
        c_guard_stats.rejected_degenerate,
        c_guard_stats.rejected_not_improving,
    );
    println!();

    println!("Results (static stream):");
    print_header();
    print_row(&baseline);
    print_row(&candidate_a);
    print_row(&candidate_b);
    print_row(&candidate_c);
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

    // Candidate C's primary claim: a safe, zero-hand-tuned drop-in for A —
    // no worse within a tight tolerance, and specifically not a repeat of
    // candidate B's uncontrolled blow-up toward the safety valve.
    let cluster_bound_c_vs_a =
        (candidate_a.n_clusters as f64 * MAX_CLUSTER_FACTOR_VS_A).ceil() as usize;
    let recall_regress_c_vs_a_pp = (candidate_a.recall_at_10 - candidate_c.recall_at_10) * 100.0;
    let purity_regress_c_vs_a_pp = (candidate_a.purity - candidate_c.purity) * 100.0;

    let c_cluster_pass = candidate_c.n_clusters <= cluster_bound_c_vs_a;
    let c_recall_pass = recall_regress_c_vs_a_pp <= MAX_RECALL_REGRESSION_C_VS_A_PP;
    let c_purity_pass = purity_regress_c_vs_a_pp <= MAX_PURITY_REGRESSION_C_VS_A_PP;
    let c_latency_pass = candidate_c.mean_us <= MAX_MEAN_LATENCY_US;
    let c_static_pass = c_cluster_pass && c_recall_pass && c_purity_pass && c_latency_pass;

    println!("Acceptance criteria — Candidate C (GuardedConditionedAdmission, guarded self-calibrating tau, static stream):");
    println!(
        "  final clusters <= {cluster_bound_c_vs_a} ({MAX_CLUSTER_FACTOR_VS_A}x candidate A's {}): {:>7}   -> {}",
        candidate_a.n_clusters,
        candidate_c.n_clusters,
        if c_cluster_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  recall@10 regression vs candidate A <= {MAX_RECALL_REGRESSION_C_VS_A_PP:.1}pp: {recall_regress_c_vs_a_pp:>7.2}pp -> {}",
        if c_recall_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  purity regression vs candidate A <= {MAX_PURITY_REGRESSION_C_VS_A_PP:.1}pp:    {purity_regress_c_vs_a_pp:>7.2}pp -> {}",
        if c_purity_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  mean latency <= {MAX_MEAN_LATENCY_US:.0}µs:                           {:>7.2}µs -> {}",
        candidate_c.mean_us,
        if c_latency_pass { "PASS" } else { "FAIL" }
    );
    println!();

    // ── drift scenario: does self-calibration earn its keep once the
    //    geometry it was tuned for stops holding? ───────────────────────
    let drift_cfg = DriftStreamConfig {
        n_points,
        k_true,
        dims,
        ..DriftStreamConfig::default()
    };
    let (drift_ds, switch_at) = StreamDataset::generate_drift(&drift_cfg);
    let drift_queries_b =
        StreamDataset::held_out_queries_drift(&drift_cfg, n_queries, 0xD41F_7000, true);

    // Candidate A, TRANSPLANTED UNCHANGED: the same fixed tau calibrated for
    // the static (regime A) run above, never retuned for the drift.
    let candidate_a_drift = run_policy(
        MincutGatedAdmission::new(tau, MAX_CLUSTERS),
        &drift_ds,
        &drift_queries_b,
    );
    // Candidate C, run fresh on the drift stream from t=0 — its warm-up and
    // guarded recalibration see the regime switch happen live.
    let candidate_c_drift = run_policy(
        GuardedConditionedAdmission::new(tau, MAX_CLUSTERS, ES_SEED),
        &drift_ds,
        &drift_queries_b,
    );

    println!("Drift scenario (regime switch at point {switch_at}/{n_points}, regime B centres crowded by {:.2}):", drift_cfg.regime_b_crowding);
    println!("  Held-out queries drawn from regime B geometry only (post-drift).");
    print_header();
    print_row(&candidate_a_drift);
    print_row(&candidate_c_drift);
    println!();

    let drift_recall_gain_c_pp =
        (candidate_c_drift.recall_at_10 - candidate_a_drift.recall_at_10) * 100.0;
    let drift_recall_pass = drift_recall_gain_c_pp >= MIN_DRIFT_RECALL_GAIN_C_VS_A_PP;
    let drift_latency_pass = candidate_c_drift.mean_us <= MAX_MEAN_LATENCY_US;
    let drift_cluster_pass = candidate_c_drift.n_clusters <= cluster_bound;
    let drift_pass = drift_recall_pass && drift_latency_pass && drift_cluster_pass;

    println!(
        "Acceptance criteria — Candidate C vs transplanted-unchanged Candidate A, under drift:"
    );
    println!(
        "  recall@10 gain (regime-B queries) >= {MIN_DRIFT_RECALL_GAIN_C_VS_A_PP:.1}pp: {drift_recall_gain_c_pp:>7.2}pp -> {}",
        if drift_recall_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  mean latency <= {MAX_MEAN_LATENCY_US:.0}µs:                      {:>7.2}µs -> {}",
        candidate_c_drift.mean_us,
        if drift_latency_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  final clusters <= {cluster_bound} ({MAX_CLUSTER_COUNT_FACTOR}x K_true):               {:>7}   -> {}",
        candidate_c_drift.n_clusters,
        if drift_cluster_pass { "PASS" } else { "FAIL" }
    );
    println!();

    let c_pass = c_static_pass && drift_pass;
    let overall = a_pass && b_pass && c_pass;
    println!(
        "Overall: {}",
        if overall {
            "ACCEPT — all mandatory thresholds passed"
        } else if a_pass || b_pass || c_pass {
            "PARTIAL — at least one candidate passed, see per-candidate results above"
        } else {
            "REJECT — no candidate passed all mandatory thresholds"
        }
    );
    println!(
        "Candidate C overall: {} (static: {}, drift: {})",
        if c_pass { "PASS" } else { "FAIL" },
        if c_static_pass { "PASS" } else { "FAIL" },
        if drift_pass { "PASS" } else { "FAIL" }
    );

    if !overall {
        std::process::exit(1);
    }
}
