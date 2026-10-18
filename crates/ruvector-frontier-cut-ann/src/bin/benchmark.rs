//! Benchmark: Frontier-Cut Conductance vs. Heap-Entropy Adaptive ANN Search
//!
//! Same dataset generators, same `FlatGraph`, same `FixedEfSearch` baseline,
//! and the same benchmark methodology (easy/hard/mixed query sets, matched-
//! budget control, recall thresholds) as the 2026-08-13 entropy-adaptive-ann
//! nightly (ADR-303), so the two signals are directly comparable. Only the
//! per-query adaptive signal differs: frontier-cut conductance (`phi`)
//! instead of heap-distance entropy (`H`).
//!
//! Run:
//!   cargo run --release -p ruvector-frontier-cut-ann --bin benchmark

use ruvector_frontier_cut_ann::cut::{probe_phi, CutThresholdBeam, FrontierCutScaledEf};
use ruvector_frontier_cut_ann::dataset::{clustered_vectors, ground_truth, random_unit_vectors};
use ruvector_frontier_cut_ann::graph::{FlatGraph, GraphConfig};
use ruvector_frontier_cut_ann::{recall_at_k, FixedEfSearch, LatencyStats, Searcher};
use std::time::Instant;

fn system_info() {
    println!("=== Frontier-Cut Conductance ANN Benchmark ===");
    println!();
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    println!("OS:           {os} / {arch}");
    println!("Rust:         (see: rustc --version)");
    let ncpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    println!("CPU threads:  {ncpu}");
    println!();
}

// Identical dataset parameters to ruvector-entropy-ann's benchmark, so the
// two benchmarks measure the same corpus/graph/query distribution.
const N: usize = 2_000;
const DIM: usize = 16;
const N_CLUSTERS: usize = 10;
const CLUSTER_NOISE: f32 = 0.20;
const K: usize = 10;
const EF: usize = 50;
const N_QUERIES_EASY: usize = 200;
const N_QUERIES_HARD: usize = 200;
const N_QUERIES_MIXED: usize = 400;
const GRAPH_K: usize = 16;
const PROBE_DEPTH: usize = 5;
const ALPHA: f32 = 1.5;
const EF_MIN_FACTOR: f32 = 0.8;
const EF_MAX_FACTOR: f32 = 2.5;

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

fn ef_stats(values: &[usize]) -> (f32, usize, usize) {
    let n = values.len().max(1);
    let mean = values.iter().sum::<usize>() as f32 / n as f32;
    let min = *values.iter().min().unwrap_or(&0);
    let max = *values.iter().max().unwrap_or(&0);
    (mean, min, max)
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

    println!("Building corpus...");
    let t0 = Instant::now();
    let corpus = clustered_vectors(N, DIM, N_CLUSTERS, CLUSTER_NOISE, 42);
    println!("  corpus built in {:.1}ms", t0.elapsed().as_millis());

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

    let easy_queries = clustered_vectors(N_QUERIES_EASY, DIM, N_CLUSTERS, 0.02, 101);
    let hard_queries = random_unit_vectors(N_QUERIES_HARD, DIM, 202);
    let mixed_queries = clustered_vectors(N_QUERIES_MIXED, DIM, N_CLUSTERS, CLUSTER_NOISE, 303);

    let mem_bytes =
        graph.len() * DIM * 4 + graph.adjacency.iter().map(|a| a.len() * 8).sum::<usize>();
    let mem_kb = mem_bytes / 1024;

    let fixed = FixedEfSearch {
        graph: &graph,
        ef_search: EF,
    };
    let cut_scaled = FrontierCutScaledEf {
        graph: &graph,
        base_ef: EF,
        probe_depth: PROBE_DEPTH,
        alpha: ALPHA,
        ef_min_factor: EF_MIN_FACTOR,
        ef_max_factor: EF_MAX_FACTOR,
    };

    // ── Matched-budget control (computed, not hardcoded) ─────────────────────
    // Measure the mean ef_actual FrontierCutScaledEf actually chooses on the
    // mixed query set, then build a FixedEf control at that exact budget.
    // If FixedEf(matched) reproduces FrontierCutScaledEf's recall, the gain is
    // "bigger budget", not "conductance" — this is the same falsifiability
    // check that refuted EntropyScaledEf in ADR-303.
    let predicted_efs: Vec<usize> = mixed_queries
        .iter()
        .map(|q| cut_scaled.predicted_ef(q))
        .collect();
    let (ef_actual_mean, ef_actual_min, ef_actual_max) = ef_stats(&predicted_efs);
    let ef_matched = ef_actual_mean.round() as usize;

    println!("─── Adaptivity Check (does ef_actual vary across queries?) ───");
    println!();
    for (label, queries) in [
        ("easy", &easy_queries),
        ("hard", &hard_queries),
        ("mixed", &mixed_queries),
    ] {
        let efs: Vec<usize> = queries.iter().map(|q| cut_scaled.predicted_ef(q)).collect();
        let (mean, min, max) = ef_stats(&efs);
        println!(
            "  {label:<8} ef_actual — mean={mean:6.1}  min={min:3}  max={max:3}  \
            (constant ef across min/max ⇒ no adaptivity, same failure as ADR-303)"
        );
    }
    println!();
    println!(
        "  mixed-set mean ef_actual = {ef_actual_mean:.1} (range {ef_actual_min}-{ef_actual_max}) \
        → matched-budget control = FixedEf({ef_matched})"
    );
    println!();

    let fixed_matched = FixedEfSearch {
        graph: &graph,
        ef_search: ef_matched,
    };
    let cut_thresh = CutThresholdBeam {
        graph: &graph,
        ef_search: EF,
        phi_stop: 0.15,
        window_steps: 3,
        min_steps: 3,
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

    // Same honest thresholds as ruvector-entropy-ann (same flat-graph
    // constraint, same corpus/query distributions).
    let recall_threshold_easy = 0.80_f32;
    let recall_threshold_hard = 0.55_f32;
    let recall_threshold_mixed = 0.65_f32;

    struct Run<'a> {
        searcher: &'a dyn Searcher,
        name: &'static str,
        queries: &'a [Vec<f32>],
        label: &'static str,
        threshold: f32,
    }

    let fixed_matched_label: &'static str =
        Box::leak(format!("FixedEf({ef_matched},matched)").into_boxed_str());

    let variants: Vec<(&dyn Searcher, &'static str)> = vec![
        (&fixed, "FixedEf(50)"),
        (&fixed_matched, fixed_matched_label),
        (&cut_thresh, "CutThreshold"),
        (&cut_scaled, "FrontierCutScaledEf"),
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
    let mut fixed_recall_mixed = 0.0f32;
    let mut cut_scaled_recall_mixed = 0.0f32;
    let mut fixed_matched_recall_mixed = 0.0f32;
    for run in &runs {
        let recall = compute_recall(run.searcher, run.queries, &corpus, K);
        let n = run.queries.len();
        let k_val = K;
        let searcher_ref = run.searcher;
        let (_, stats) = LatencyStats::measure(n, |i| searcher_ref.search(&run.queries[i], k_val));
        let pass = recall >= run.threshold;
        if !pass {
            all_pass = false;
        }
        if run.label == "mixed" && run.name == "FixedEf(50)" {
            fixed_recall_mixed = recall;
        }
        if run.label == "mixed" && run.name == "FrontierCutScaledEf" {
            cut_scaled_recall_mixed = recall;
        }
        if run.label == "mixed" && run.name == fixed_matched_label {
            fixed_matched_recall_mixed = recall;
        }
        print_row(run.name, run.label, n, recall, &stats, mem_kb, pass);
    }

    println!();
    println!("─── Conductance (phi) Distribution Analysis ───");
    println!();
    let analyze_phi = |queries: &[Vec<f32>], label: &str| {
        let mut phis: Vec<f32> = queries
            .iter()
            .map(|q| probe_phi(&graph, q, PROBE_DEPTH, EF).phi)
            .collect();
        phis.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = phis.len();
        let mean = phis.iter().sum::<f32>() / n as f32;
        let p50 = phis[n / 2];
        let p95 = phis[(n as f32 * 0.95) as usize];
        println!(
            "  {label:<8} phi — mean={mean:.3}  p50={p50:.3}  p95={p95:.3}  \
            (higher phi ⇒ more boundary-crossing edges in the probe window)"
        );
    };
    analyze_phi(&easy_queries, "easy");
    analyze_phi(&hard_queries, "hard");
    analyze_phi(&mixed_queries, "mixed");

    println!();
    println!("─── Falsifiable Claim Check ───");
    println!();
    let adaptivity_real = {
        let easy_efs: Vec<usize> = easy_queries
            .iter()
            .map(|q| cut_scaled.predicted_ef(q))
            .collect();
        let hard_efs: Vec<usize> = hard_queries
            .iter()
            .map(|q| cut_scaled.predicted_ef(q))
            .collect();
        let (easy_mean, _, _) = ef_stats(&easy_efs);
        let (hard_mean, _, _) = ef_stats(&hard_efs);
        (easy_mean - hard_mean).abs() >= 2.0
    };
    println!(
        "  Adaptivity (easy/hard mean ef_actual differ by >= 2): {}",
        if adaptivity_real {
            "YES"
        } else {
            "NO (saturated, same failure mode as ADR-303)"
        }
    );
    let gain_over_matched_budget = cut_scaled_recall_mixed - fixed_matched_recall_mixed;
    println!(
        "  FrontierCutScaledEf recall={cut_scaled_recall_mixed:.4} vs matched-budget \
        FixedEf({ef_matched}) recall={fixed_matched_recall_mixed:.4}  (delta={gain_over_matched_budget:+.4})"
    );
    println!(
        "  FrontierCutScaledEf recall={cut_scaled_recall_mixed:.4} vs FixedEf(50) \
        recall={fixed_recall_mixed:.4}  (delta={:+.4})",
        cut_scaled_recall_mixed - fixed_recall_mixed
    );

    println!();
    println!("─── Memory Estimate ───");
    let bytes_per_vec = DIM * 4;
    let bytes_per_edge = 8;
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
