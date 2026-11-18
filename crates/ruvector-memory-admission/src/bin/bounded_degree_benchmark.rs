//! Bounded-degree admission benchmark — nightly 2026-11-18.
//!
//! Answers ADR-344's own Next Research item 1: does restricting the
//! min-cut admission graph to each point's `K` nearest existing centroids
//! (not all `C`) preserve `MincutGatedAdmission`'s quality gain while
//! closing its O(C^3) cost ceiling?
//!
//! Two parts:
//!
//! 1. **Scaling sweep** — fixes `C` exactly (via the benchmark-only
//!    `from_centroids` constructors, bypassing the stream) and measures raw
//!    `decide()` latency at each `C`, for the full-graph policy and several
//!    `K` values. Finds the `C*` where the full graph crosses the existing
//!    500µs write-path budget (`MAX_MEAN_LATENCY_US`, reused unchanged from
//!    the ADR-344 benchmark this nightly extends) and checks whether
//!    bounded-degree stays under it there.
//! 2. **Quality-preservation run** — a realistic stream (more true clusters
//!    than ADR-344's run, so natural `C` grows past every tested `K`),
//!    matched-budget calibrated exactly as ADR-344's benchmark was, scoring
//!    purity/recall@10/latency for the baseline, the full-graph policy
//!    (quality reference), and each bounded-degree `K`.
//!
//! Run:
//!   cargo run --release -p ruvector-memory-admission --bin bounded_degree_benchmark
//!
//! Environment overrides:
//!   TAU=0.005 N_POINTS=6000 K_TRUE=24 DIMS=64 N_QUERIES=400

use ruvector_memory_admission::bounded_degree::BoundedDegreeAdmission;
use ruvector_memory_admission::dataset::{Lcg64, StreamConfig, StreamDataset};
use ruvector_memory_admission::policy::{
    AdmissionPolicy, MincutGatedAdmission, NearestCentroidThreshold,
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

// Shared, unmodified from the ADR-344 benchmark this nightly extends — the
// write-path latency budget is a production constraint, not something this
// run gets to redefine after seeing results.
const MAX_MEAN_LATENCY_US: f64 = 500.0;
const K_EVAL: usize = 10;
const CALIBRATION_ITERATIONS: usize = 25;
const MAX_CLUSTERS_SAFETY_VALVE: usize = 256;

// Fixed before running (see docs/adr/ADR-361 and the nightly README):
const MAX_QUALITY_DELTA_PP: f64 = 1.0; // |purity or recall delta vs full graph| must stay <= this

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

fn random_unit_vecs(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg64(seed);
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dims).map(|_| rng.gaussian()).collect();
            ruvector_memory_admission::dataset::normalise(&mut v);
            v
        })
        .collect()
}

// ─── Part 1: scaling sweep ──────────────────────────────────────────────────

struct ScalePoint {
    variant: String,
    c: usize,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    mean_sim_ops: f64,
}

fn time_decide(policy: &dyn AdmissionPolicy, queries: &[Vec<f32>]) -> (Vec<u128>, f64) {
    let mut lat = Vec::with_capacity(queries.len());
    let mut ops_sum = 0u64;
    for q in queries {
        let t0 = Instant::now();
        let d = policy.decide(q);
        lat.push(t0.elapsed().as_micros());
        ops_sum += d.sim_ops as u64;
    }
    lat.sort_unstable();
    (lat, ops_sum as f64 / queries.len() as f64)
}

fn run_scaling_sweep(dims: usize) -> Vec<ScalePoint> {
    let cluster_counts = [8usize, 16, 32, 64, 96, 128, 192, 256];
    let k_values = [4usize, 8, 16, 32];
    let n_query_reps = 80usize;

    let mut rows = Vec::new();
    for &c in &cluster_counts {
        let centroids = random_unit_vecs(c, dims, 0x0005_CA1E_0001 ^ (c as u64));
        let queries = random_unit_vecs(n_query_reps, dims, 0x0005_CA1E_0002 ^ (c as u64));

        let full = MincutGatedAdmission::from_centroids(centroids.clone());
        let (lat, ops) = time_decide(&full, &queries);
        rows.push(ScalePoint {
            variant: "MincutGatedAdmission (full)".to_string(),
            c,
            mean_us: mean_u128(&lat),
            p50_us: percentile(&lat, 0.50) as f64,
            p95_us: percentile(&lat, 0.95) as f64,
            mean_sim_ops: ops,
        });

        for &k in &k_values {
            let bounded = BoundedDegreeAdmission::from_centroids(centroids.clone(), k);
            let (lat, ops) = time_decide(&bounded, &queries);
            rows.push(ScalePoint {
                variant: format!("BoundedDegreeAdmission(K={k})"),
                c,
                mean_us: mean_u128(&lat),
                p50_us: percentile(&lat, 0.50) as f64,
                p95_us: percentile(&lat, 0.95) as f64,
                mean_sim_ops: ops,
            });
        }
    }
    rows
}

// ─── Part 2: quality preservation at a realistic matched setting ──────────

struct QualityResult {
    name: String,
    n_clusters: usize,
    purity: f64,
    recall_at_10: f64,
    mean_us: f64,
    p95_us: f64,
}

fn run_policy_quality(
    mut policy: impl AdmissionPolicy,
    ds: &StreamDataset,
    queries: &[(Vec<f32>, usize)],
) -> QualityResult {
    let n = ds.points.len();
    let mut assigned = vec![0usize; n];
    let mut lat: Vec<u128> = Vec::with_capacity(n);

    for (i, p) in ds.points.iter().enumerate() {
        let t0 = Instant::now();
        let d = policy.admit(&p.vector);
        lat.push(t0.elapsed().as_micros());
        assigned[i] = d.cluster_id;
    }

    let n_clusters = policy.n_clusters();
    let mut label_counts = vec![vec![0usize; ds.k_true]; n_clusters];
    for (i, p) in ds.points.iter().enumerate() {
        label_counts[assigned[i]][p.true_cluster] += 1;
    }
    let correct: usize = label_counts
        .iter()
        .map(|c| c.iter().copied().max().unwrap_or(0))
        .sum();
    let purity = correct as f64 / n as f64;

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
        let query_cluster = policy.decide(q).cluster_id;
        let hits = true_top10
            .iter()
            .filter(|&&id| assigned[id] == query_cluster)
            .count();
        recall_sum += hits as f64 / K_EVAL as f64;
    }
    let recall_at_10 = recall_sum / queries.len() as f64;

    lat.sort_unstable();
    QualityResult {
        name: policy.name().to_string(),
        n_clusters,
        purity,
        recall_at_10,
        mean_us: mean_u128(&lat),
        p95_us: percentile(&lat, 0.95) as f64,
    }
}

fn clusters_at_threshold(ds: &StreamDataset, threshold: f32) -> usize {
    let mut p = NearestCentroidThreshold::new(threshold);
    for pt in &ds.points {
        p.admit(&pt.vector);
    }
    p.n_clusters()
}

fn calibrate_threshold(ds: &StreamDataset, target: usize) -> (f32, usize) {
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    let mut best = (lo, clusters_at_threshold(ds, lo));
    let mut best_diff = best.1.abs_diff(target);
    for _ in 0..CALIBRATION_ITERATIONS {
        let mid = (lo + hi) / 2.0;
        let n = clusters_at_threshold(ds, mid);
        let diff = n.abs_diff(target);
        if diff < best_diff {
            best_diff = diff;
            best = (mid, n);
        }
        if n > target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    best
}

fn clusters_at_tau_bounded(ds: &StreamDataset, tau: f32, k_neighbors: usize) -> usize {
    let mut p = BoundedDegreeAdmission::new(tau, MAX_CLUSTERS_SAFETY_VALVE, k_neighbors);
    for pt in &ds.points {
        p.admit(&pt.vector);
    }
    p.n_clusters()
}

/// Same matched-budget discipline the ADR-344 benchmark uses for its
/// baseline, applied here to `BoundedDegreeAdmission`'s `tau`: a first,
/// uncalibrated Part 2 run (same `tau` as the full graph, below) found
/// `BoundedDegreeAdmission(K=4|8)` landing on a different final cluster
/// count than the full-graph reference, confounding the purity/recall
/// comparison (more clusters mechanically tends toward higher purity). This
/// calibrates `tau` per `K` so the comparison is at the *same* cluster
/// budget as the full-graph reference — the only way a quality delta
/// between them is actually informative. `tau` is monotonic in cluster
/// count for the same reason `NearestCentroidThreshold`'s threshold is
/// (higher bar to merge => more spawns), which is what makes this
/// well-defined.
fn calibrate_tau_bounded(ds: &StreamDataset, k_neighbors: usize, target: usize) -> (f32, usize) {
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    let mut best = (lo, clusters_at_tau_bounded(ds, lo, k_neighbors));
    let mut best_diff = best.1.abs_diff(target);
    for _ in 0..CALIBRATION_ITERATIONS {
        let mid = (lo + hi) / 2.0;
        let n = clusters_at_tau_bounded(ds, mid, k_neighbors);
        let diff = n.abs_diff(target);
        if diff < best_diff {
            best_diff = diff;
            best = (mid, n);
        }
        if n > target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    best
}

fn main() {
    println!("=== RuVector Bounded-Degree Admission Benchmark (nightly 2026-11-18) ===");
    println!(
        "OS: {}  Arch: {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!();

    let dims = env_usize("DIMS", 64);

    // ── Part 1: scaling sweep ─────────────────────────────────────────────
    println!(
        "--- Part 1: decide() latency vs cluster count C (fixed centroids, not committed) ---"
    );
    println!(
        "{:<32} {:>6} {:>10} {:>9} {:>9} {:>9}",
        "Variant", "C", "Mean(µs)", "p50(µs)", "p95(µs)", "SimOps"
    );
    println!("{}", "-".repeat(80));
    let scale_rows = run_scaling_sweep(dims);
    for r in &scale_rows {
        println!(
            "{:<32} {:>6} {:>10.2} {:>9.0} {:>9.0} {:>9.1}",
            r.variant, r.c, r.mean_us, r.p50_us, r.p95_us, r.mean_sim_ops
        );
    }
    println!();

    let crossover = scale_rows
        .iter()
        .filter(|r| r.variant == "MincutGatedAdmission (full)")
        .find(|r| r.mean_us > MAX_MEAN_LATENCY_US);
    match crossover {
        Some(r) => println!(
            "Full-graph mean latency crosses the {MAX_MEAN_LATENCY_US:.0}µs budget at C={} ({:.2}µs).",
            r.c, r.mean_us
        ),
        None => println!(
            "Full-graph mean latency stayed under the {MAX_MEAN_LATENCY_US:.0}µs budget for every C swept — sweep range too small to find the ceiling."
        ),
    }
    let bounded_at_crossover: Vec<&ScalePoint> = match crossover {
        Some(co) => scale_rows
            .iter()
            .filter(|r| r.c == co.c && r.variant != "MincutGatedAdmission (full)")
            .collect(),
        None => Vec::new(),
    };
    for r in &bounded_at_crossover {
        println!(
            "  at that C: {:<28} {:>8.2}µs -> {}",
            r.variant,
            r.mean_us,
            if r.mean_us <= MAX_MEAN_LATENCY_US {
                "PASS (under budget)"
            } else {
                "FAIL (still over budget)"
            }
        );
    }
    let scaling_claim_holds = !bounded_at_crossover.is_empty()
        && bounded_at_crossover
            .iter()
            .any(|r| r.mean_us <= MAX_MEAN_LATENCY_US);
    println!();

    // ── Part 2: quality preservation at a realistic matched setting ───────
    println!(
        "--- Part 2: quality preservation, matched cluster budget, C grown past every tested K ---"
    );
    let n_points = env_usize("N_POINTS", 6000);
    let k_true = env_usize("K_TRUE", 24);
    let n_queries = env_usize("N_QUERIES", 400);
    let tau = env_f32("TAU", 0.005);
    let k_values = [4usize, 8, 16, 32];

    let cfg = StreamConfig {
        n_points,
        k_true,
        dims,
        ..StreamConfig::default()
    };
    let ds = StreamDataset::generate(&cfg);
    let queries = ds.held_out_queries(&cfg, n_queries, 0xC0FF_EE01);

    let full = run_policy_quality(
        MincutGatedAdmission::new(tau, MAX_CLUSTERS_SAFETY_VALVE),
        &ds,
        &queries,
    );
    let (calibrated_threshold, _) = calibrate_threshold(&ds, full.n_clusters);
    let baseline = run_policy_quality(
        NearestCentroidThreshold::new(calibrated_threshold),
        &ds,
        &queries,
    );

    println!(
        "  Target cluster count (from full-graph candidate): {}",
        full.n_clusters
    );
    println!("  Calibrated baseline threshold: {calibrated_threshold:.4}");
    println!();
    println!(
        "{:<36} {:>8} {:>8} {:>10} {:>10} {:>9}",
        "Variant", "Clusters", "Purity", "Recall@10", "Mean(µs)", "p95(µs)"
    );
    println!("{}", "-".repeat(86));
    let print_q = |label: &str, r: &QualityResult| {
        println!(
            "{:<36} {:>8} {:>8.4} {:>10.4} {:>10.2} {:>9.0}",
            label, r.n_clusters, r.purity, r.recall_at_10, r.mean_us, r.p95_us
        );
    };
    print_q("NearestCentroidThreshold (matched)", &baseline);
    print_q("MincutGatedAdmission (full graph)", &full);

    println!();
    println!("  RAW (disclosed, NOT used for acceptance): same tau as full graph, cluster count unmatched —");
    println!("  preserved as evidence of the confound this run found, same way ADR-344's own first run disclosed its uncalibrated baseline.");
    for &k in &k_values {
        let r = run_policy_quality(
            BoundedDegreeAdmission::new(tau, MAX_CLUSTERS_SAFETY_VALVE, k),
            &ds,
            &queries,
        );
        print_q(&format!("BoundedDegreeAdmission(K={k}) [raw]"), &r);
    }
    println!();

    println!("  MATCHED BUDGET (tau calibrated per K to the full graph's {} clusters) — this is what acceptance below is judged on:", full.n_clusters);
    let mut bounded_results = Vec::new();
    for &k in &k_values {
        let (calibrated_tau, _) = calibrate_tau_bounded(&ds, k, full.n_clusters);
        let r = run_policy_quality(
            BoundedDegreeAdmission::new(calibrated_tau, MAX_CLUSTERS_SAFETY_VALVE, k),
            &ds,
            &queries,
        );
        print_q(
            &format!("BoundedDegreeAdmission(K={k}) [matched, tau={calibrated_tau:.4}]"),
            &r,
        );
        bounded_results.push((k, r));
    }
    println!();

    println!("Acceptance — quality preservation at matched cluster budget (|delta vs full-graph| <= {MAX_QUALITY_DELTA_PP:.1}pp), and a real latency win:");
    let mut any_k_passes = false;
    for (k, r) in &bounded_results {
        let purity_delta_pp = (r.purity - full.purity).abs() * 100.0;
        let recall_delta_pp = (r.recall_at_10 - full.recall_at_10).abs() * 100.0;
        let faster = r.mean_us < full.mean_us;
        let pass = purity_delta_pp <= MAX_QUALITY_DELTA_PP
            && recall_delta_pp <= MAX_QUALITY_DELTA_PP
            && faster;
        println!(
            "  K={k:<4} purity_delta={purity_delta_pp:>5.2}pp recall_delta={recall_delta_pp:>5.2}pp mean_us={:.2} (full={:.2}, faster={faster}) -> {}",
            r.mean_us,
            full.mean_us,
            if pass { "PASS" } else { "FAIL" }
        );
        any_k_passes |= pass;
    }
    println!();

    let overall = scaling_claim_holds && any_k_passes;
    println!(
        "Scaling claim (bounded-degree stays under budget where full graph does not): {}",
        if scaling_claim_holds { "PASS" } else { "FAIL" }
    );
    println!("Quality-preservation claim (at least one K preserves quality within {MAX_QUALITY_DELTA_PP:.1}pp and is faster): {}", if any_k_passes { "PASS" } else { "FAIL" });
    println!(
        "Overall: {}",
        if overall {
            "ACCEPT — hypothesis supported"
        } else {
            "REJECT — hypothesis not supported by this run"
        }
    );

    if !overall {
        std::process::exit(1);
    }
}
