//! Benchmark: Entropy-Adaptive ANN Search
//!
//! Measures recall@10 and latency for three search variants on a clustered
//! synthetic dataset designed to contain both "easy" (low-entropy) queries
//! and "hard" (high-entropy) queries.
//!
//! Run:
//!   cargo run --release -p ruvector-entropy-ann --bin benchmark

use ruvector_entropy_ann::{
    dataset::{clustered_vectors, ground_truth, random_unit_vectors},
    graph::{FlatGraph, GraphConfig},
    metrics::LatencyStats,
    recall_at_k,
    search::{EntropyScaledEf, EntropyThresholdBeam, FixedEfSearch, Searcher},
};
use std::time::Instant;

fn system_info() {
    println!("=== Entropy-Adaptive ANN Benchmark ===");
    println!();

    // OS
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    println!("OS:           {os} / {arch}");

    // Rust version: run `rustc --version` for the exact compiler
    println!("Rust:         (see: rustc --version)");

    // CPU count
    let ncpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    println!("CPU threads:  {ncpu}");
    println!();
}

// 16D: distance concentration is much weaker than 64D, so the softmin
// entropy signal shows clear variation between easy/hard queries.
// N=2000 keeps brute-force O(n²) graph build fast (< 1s) for the PoC.
const N: usize = 2_000;
const DIM: usize = 16;
const N_CLUSTERS: usize = 10;
const CLUSTER_NOISE: f32 = 0.20;
const K: usize = 10;
const EF: usize = 50;
/// Matched-budget control: the measured mean ef_actual chosen by
/// EntropyScaledEf on this dataset (alpha=1.5 saturates the scale at
/// ef_max_factor=2.5 for every query → ef_actual = 122–124, mean ≈ 124).
/// A FixedEf run at this budget makes the "adaptive" claim falsifiable:
/// if FixedEf(124) matches EntropyScaledEf's recall, the gain is budget,
/// not entropy. (Measured: it matches to four decimal places.)
const EF_MATCHED: usize = 124;
const N_QUERIES_EASY: usize = 200;
const N_QUERIES_HARD: usize = 200;
const N_QUERIES_MIXED: usize = 400;
const GRAPH_K: usize = 16;

fn compute_recall(
    searcher: &dyn Searcher,
    queries: &[Vec<f32>],
    corpus: &[Vec<f32>],
    k: usize,
) -> f32 {
    let total: f32 = queries
        .iter()
        .map(|q| {
            let gt = ground_truth(q, corpus, k);
            let hits = searcher.search(q, k);
            recall_at_k(&gt, &hits, k)
        })
        .sum();
    total / queries.len() as f32
}

fn print_row(
    name: &str,
    query_type: &str,
    n_queries: usize,
    recall: f32,
    stats: &LatencyStats,
    mem_kb: usize,
    accept: bool,
) {
    let check = if accept { "PASS" } else { "FAIL" };
    println!(
        "  {name:<22} {query_type:<8} n={n_queries:<5} recall={recall:.3}  \
        mean={:6.1}µs  p50={:6.1}µs  p95={:7.1}µs  \
        {:.0} qps  mem={}KB  [{check}]",
        stats.mean_us, stats.p50_us, stats.p95_us, stats.throughput_qps, mem_kb,
    );
}

fn main() {
    system_info();

    println!("Dataset:");
    println!("  N (corpus) : {N}");
    println!("  Dimensions : {DIM}");
    println!("  Clusters   : {N_CLUSTERS}  noise={CLUSTER_NOISE}");
    println!("  k (recall) : {K}");
    println!("  ef_search  : {EF}");
    println!("  Graph K    : {GRAPH_K}");
    println!();

    // Build corpus — clustered to produce both easy and hard queries
    println!("Building corpus...");
    let t0 = Instant::now();
    let corpus = clustered_vectors(N, DIM, N_CLUSTERS, CLUSTER_NOISE, 42);
    println!("  corpus built in {:.1}ms", t0.elapsed().as_millis());

    // Build graph
    println!("Building flat graph (k={GRAPH_K})...");
    let t1 = Instant::now();
    let graph = FlatGraph::build(
        corpus.clone(),
        GraphConfig {
            k_neighbours: GRAPH_K,
        },
    );
    println!("  graph built in {:.1}ms", t1.elapsed().as_millis());
    println!();

    // Query sets
    // Easy: queries near cluster centroids (same distribution, tight clusters)
    let easy_queries = clustered_vectors(N_QUERIES_EASY, DIM, N_CLUSTERS, 0.02, 101);
    // Hard: completely random unit vectors — maximally ambiguous
    let hard_queries = random_unit_vectors(N_QUERIES_HARD, DIM, 202);
    // Mixed: the default cluster distribution
    let mixed_queries = clustered_vectors(N_QUERIES_MIXED, DIM, N_CLUSTERS, CLUSTER_NOISE, 303);

    let mem_bytes =
        graph.len() * DIM * 4 + graph.adjacency.iter().map(|a| a.len() * 8).sum::<usize>();
    let mem_kb = mem_bytes / 1024;

    // ── Variant definitions ───────────────────────────────────────────────────
    let fixed = FixedEfSearch {
        graph: &graph,
        ef_search: EF,
    };
    // Matched-budget control at EntropyScaledEf's measured mean ef_actual.
    let fixed_matched = FixedEfSearch {
        graph: &graph,
        ef_search: EF_MATCHED,
    };
    // Temperature=0.1 gives sharper contrast in 16D than 0.3.
    // In higher dimensions (64D+) distances concentrate and entropy
    // becomes less discriminative; see research doc for analysis.
    let entropy_thresh = EntropyThresholdBeam {
        graph: &graph,
        ef_search: EF,
        h_stop: 0.6,
        temperature: 0.1,
    };
    let entropy_scaled = EntropyScaledEf {
        graph: &graph,
        base_ef: EF,
        probe_depth: 5,
        alpha: 1.5,
        ef_min_factor: 0.8,
        ef_max_factor: 2.5,
        temperature: 0.1,
    };

    println!("Benchmarking variants on easy / hard / mixed query sets...");
    println!();
    println!(
        "  {:<22} {:<8} {:<7} {:<12} {:<16} {:<16} {:<16} {:<14} {:<10}",
        "Variant",
        "Queries",
        "N",
        "Recall@10",
        "Mean latency",
        "p50 latency",
        "p95 latency",
        "Throughput",
        "Result"
    );
    println!("  {}", "-".repeat(120));

    // Flat single-layer graph: without HNSW upper layers the entry routing
    // is limited to brute-force. Recall is bounded by graph connectivity in
    // the neighbourhood of the query. These thresholds reflect the flat-graph
    // constraint; a multi-layer HNSW would push past 0.95 at the same ef.
    //
    // Mixed queries use a different seed (303) than the corpus (42), so their
    // cluster centroids differ — they are partially out-of-distribution. The
    // threshold is set to 0.65 rather than 0.70 to reflect this honestly.
    let recall_threshold_easy = 0.80_f32;
    let recall_threshold_hard = 0.55_f32;
    let recall_threshold_mixed = 0.65_f32;

    // Run all variants on all query sets
    struct Run<'a> {
        searcher: &'a dyn Searcher,
        /// Display name (distinguishes the two FixedEf budgets).
        name: &'static str,
        queries: &'a [Vec<f32>],
        label: &'static str,
        threshold: f32,
    }

    let variants: [(&dyn Searcher, &'static str); 4] = [
        (&fixed, "FixedEf(50)"),
        (&fixed_matched, "FixedEf(124,matched)"),
        (&entropy_thresh, "EntropyThreshold"),
        (&entropy_scaled, "EntropyScaledEf"),
    ];
    let query_sets: [(&[Vec<f32>], &'static str, f32); 3] = [
        (&easy_queries, "easy", recall_threshold_easy),
        (&hard_queries, "hard", recall_threshold_hard),
        (&mixed_queries, "mixed", recall_threshold_mixed),
    ];
    let runs: Vec<Run> = variants
        .iter()
        .flat_map(|&(searcher, name)| {
            query_sets
                .iter()
                .map(move |&(queries, label, threshold)| Run {
                    searcher,
                    name,
                    queries,
                    label,
                    threshold,
                })
        })
        .collect();

    let mut all_pass = true;
    for run in &runs {
        let recall = compute_recall(run.searcher, run.queries, &corpus, K);
        let n = run.queries.len();
        let k_val = K;
        let searcher_ref = run.searcher;
        // Ground truth is deliberately NOT computed inside the timed closure:
        // the brute-force scan would dominate (and hide) the search latency.
        // The timed closure runs the ANN search only.
        let (_, stats) = LatencyStats::measure(n, |i| searcher_ref.search(&run.queries[i], k_val));
        let pass = recall >= run.threshold;
        if !pass {
            all_pass = false;
        }
        print_row(run.name, run.label, n, recall, &stats, mem_kb, pass);
    }

    println!();
    println!("─── Entropy Distribution Analysis ───");
    println!();

    // Analyse entropy of easy vs hard queries to validate the entropy signal
    use ruvector_entropy_ann::heap_entropy;

    let analyze_entropy = |queries: &[Vec<f32>], label: &str| {
        let mut entropies: Vec<f32> = queries
            .iter()
            .map(|q| {
                // Get top-20 distances from fixed-ef search, compute entropy
                let hits = fixed.search(q, 20);
                let dists: Vec<f32> = hits.iter().map(|h| h.dist).collect();
                heap_entropy(&dists, 0.1)
            })
            .collect();
        entropies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = entropies.len();
        let mean = entropies.iter().sum::<f32>() / n as f32;
        let p50 = entropies[n / 2];
        let p95 = entropies[(n as f32 * 0.95) as usize];
        println!(
            "  {label:<8} query entropy — mean={mean:.3}  p50={p50:.3}  p95={p95:.3}  \
            (lower=easier)"
        );
    };

    analyze_entropy(&easy_queries, "easy");
    analyze_entropy(&hard_queries, "hard");
    analyze_entropy(&mixed_queries, "mixed");

    println!();
    println!("─── Memory Estimate ───");
    let bytes_per_vec = DIM * 4;
    let bytes_per_edge = 8; // (f32 dist², usize id)
    let total_edge_bytes = N * GRAPH_K * bytes_per_edge;
    let total_vec_bytes = N * bytes_per_vec;
    let total_bytes = total_vec_bytes + total_edge_bytes;
    println!(
        "  {N} vectors × {DIM}D × 4B = {} KB vectors",
        total_vec_bytes / 1024
    );
    println!(
        "  {N} nodes × {GRAPH_K} edges × 8B = {} KB adjacency",
        total_edge_bytes / 1024
    );
    println!(
        "  Total index: {} KB = {:.1} MB",
        total_bytes / 1024,
        total_bytes as f32 / 1_048_576.0
    );

    println!();
    println!("─── Acceptance Result ───");
    if all_pass {
        println!("  PASS — all variants meet recall thresholds");
    } else {
        println!("  FAIL — one or more variants below threshold");
        std::process::exit(1);
    }
}
