/// Benchmark demo for distance-adaptive vs. fixed-width beam search.
///
/// Key insight: on any navigable graph, DistanceAdaptive(gamma) provides a
/// (1+gamma/2)-approximation GUARANTEE; FixedWidth provides none.
/// To compare fairly, we show both at matched recall levels.
///
/// Dataset: N=5 000 Gaussian vectors, D=128, k-NN graph M=16.
/// Run with: cargo run --release -p ruvector-adaptive-beam
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_adaptive_beam::graph::build_knn_graph;
use ruvector_adaptive_beam::{l2_sq, AdaptiveBeamIndex, BeamStopPolicy, SearchMetrics};
use std::collections::HashSet;
use std::time::Instant;

const N: usize = 5_000;
const D: usize = 128;
const M: usize = 16;
const QUERIES: usize = 1_000;
const K: usize = 10;

fn gaussian_vecs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    (0..n)
        .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

fn brute_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> HashSet<u32> {
    let mut ds: Vec<(f32, u32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (l2_sq(query, v), i as u32))
        .collect();
    ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    ds.truncate(k);
    ds.into_iter().map(|(_, i)| i).collect()
}

struct Run {
    label: String,
    qps: f64,
    recall: f64,
    dist_per_q: f64,
    early_pct: f64,
    guaranteed: bool,
}

fn bench(
    idx: &AdaptiveBeamIndex,
    queries: &[Vec<f32>],
    gt: &[HashSet<u32>],
    policy: BeamStopPolicy,
    label: &str,
    guaranteed: bool,
) -> Run {
    // Warm-up (not measured)
    for q in queries.iter().take(50) {
        let _ = idx.search(q, K, policy);
    }

    let mut agg = SearchMetrics::default();
    let mut recall_sum = 0.0f64;
    let mut early = 0u64;

    let t0 = Instant::now();
    for (q, truth) in queries.iter().zip(gt.iter()) {
        let (res, m) = idx.search(q, K, policy);
        let found: HashSet<u32> = res.iter().map(|(i, _)| *i).collect();
        recall_sum += truth.intersection(&found).count() as f64 / K as f64;
        agg.distance_computations += m.distance_computations;
        agg.nodes_expanded += m.nodes_expanded;
        if m.early_stopped {
            early += 1;
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let nq = queries.len() as f64;

    Run {
        label: label.to_string(),
        qps: nq / elapsed,
        recall: recall_sum / nq,
        dist_per_q: agg.distance_computations as f64 / nq,
        early_pct: early as f64 / nq * 100.0,
        guaranteed,
    }
}

fn main() {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("=== Distance-Adaptive Beam Search — ruvector ===");
    println!("arXiv:2505.15636 (May 2025): Provably Accurate Graph-Based ANN Stopping Criterion");
    println!("Hardware: {cpus} logical CPUs | rustc --release");
    println!("Dataset: N={N} Gaussian vectors, D={D}, M={M} neighbours, Q={QUERIES} queries, k={K}");
    println!();

    print!("  [1/3] Generating {N} Gaussian vectors (D={D})... ");
    let vecs = gaussian_vecs(N, D, 42);
    println!("done.");

    print!("  [2/3] Building k-NN graph M={M} (parallel, exact k-NN)... ");
    let t = Instant::now();
    let (nb, ep) = build_knn_graph(&vecs, M);
    let build_ms = t.elapsed().as_secs_f64() * 1_000.0;
    println!("done in {build_ms:.0} ms.");

    let idx = AdaptiveBeamIndex::new(vecs.clone(), nb, ep);

    print!("  [3/3] Brute-force ground truth for {QUERIES} queries... ");
    let t = Instant::now();
    let queries = gaussian_vecs(QUERIES, D, 999);
    let gt: Vec<HashSet<u32>> = queries.iter().map(|q| brute_knn(&vecs, q, K)).collect();
    println!("done in {:.0} ms.", t.elapsed().as_secs_f64() * 1_000.0);
    println!();

    // Variant 1: FixedWidth sweep
    println!("--- Variant 1: FixedWidth (no quality guarantee) ---");
    let fw_runs: Vec<Run> = [64usize, 256, 1024, 4096]
        .iter()
        .map(|&bw| {
            bench(
                &idx, &queries, &gt,
                BeamStopPolicy::FixedWidth { beam_width: bw },
                &format!("FixedWidth(bw={bw})"),
                false,
            )
        })
        .collect();

    // Variant 2: DistanceAdaptive sweep
    println!("--- Variant 2: DistanceAdaptive (provable (1+γ/2)-approximation) ---");
    let da_runs: Vec<Run> = [2.0f32, 1.0, 0.5, 0.1]
        .iter()
        .map(|&g| {
            bench(
                &idx, &queries, &gt,
                BeamStopPolicy::DistanceAdaptive { gamma: g },
                &format!("DistanceAdaptive(γ={g:.1})"),
                true,
            )
        })
        .collect();

    // Variant 3: AdaptiveWithFloor
    let af_run = bench(
        &idx, &queries, &gt,
        BeamStopPolicy::AdaptiveWithFloor { gamma: 0.5, min_expansions: 16 },
        "AdaptiveFloor(γ=0.5,min=16)",
        true,
    );

    // Print combined table
    let sep = "─".repeat(90);
    println!();
    println!("{sep}");
    println!(
        "{:<40}  {:>9}  {:>10}  {:>11}  {:>10}  {}",
        "Policy", "QPS", "Recall@10", "Dist/query", "EarlyStop%", "Guarantee"
    );
    println!("{sep}");
    for r in fw_runs.iter().chain(da_runs.iter()).chain(std::iter::once(&af_run)) {
        println!(
            "{:<40}  {:>9.0}  {:>9.1}%  {:>11.1}  {:>9.1}%  {}",
            r.label,
            r.qps,
            r.recall * 100.0,
            r.dist_per_q,
            r.early_pct,
            if r.guaranteed { "(1+γ/2)-approx ✓" } else { "none" },
        );
    }
    println!("{sep}");

    // Matched-recall analysis: find FW that comes closest to DA(1.0) recall
    let da10 = &da_runs[1]; // DA(gamma=1.0)
    let target_recall = da10.recall;
    let fw_matched = fw_runs
        .iter()
        .min_by(|a, b| {
            (a.recall - target_recall)
                .abs()
                .partial_cmp(&(b.recall - target_recall).abs())
                .unwrap()
        })
        .unwrap();

    println!();
    println!("=== Matched-Recall Analysis ===");
    println!("Target recall: DA(γ=1.0) = {:.1}% Recall@10", target_recall * 100.0);
    println!(
        "  DA(γ=1.0)            : {:.0} dist/query  ({:.1}% recall, {:.0} QPS)",
        da10.dist_per_q, da10.recall * 100.0, da10.qps
    );
    println!(
        "  {} (closest FW): {:.0} dist/query  ({:.1}% recall, {:.0} QPS)",
        fw_matched.label, fw_matched.dist_per_q, fw_matched.recall * 100.0, fw_matched.qps
    );

    let quality_gap = (fw_matched.recall - target_recall) * 100.0;
    println!(
        "  Recall gap (FW vs DA): {:+.2} pp (FW has no guarantee; DA has (1+0.5)-approx)",
        quality_gap
    );

    // Memory and build summary
    let vec_mb = (N * D * 4) as f64 / 1e6;
    let graph_mb = (N * M * 4) as f64 / 1e6;
    println!();
    println!("=== Resource Summary ===");
    println!("  Vectors: {vec_mb:.2} MB  |  Graph: {graph_mb:.2} MB  |  Total: {:.2} MB", vec_mb + graph_mb);
    println!("  Graph build (exact k-NN, parallel): {build_ms:.0} ms");
    println!("  Note: HNSW-style graph (with long-range edges) would show DA early-stopping");
    println!("        more prominently; flat k-NN requires deeper exploration before convergence.");
}
