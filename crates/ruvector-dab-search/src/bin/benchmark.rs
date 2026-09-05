//! Benchmark: Distance-Adaptive Beam (DAB) Search
//!
//! Measures recall@10, latency, and per-query distance-computation work for
//! FixedEf (baseline) vs AdaptiveGamma (candidate) on the same clustered
//! synthetic dataset and graph construction used by ADR-303
//! (`ruvector-entropy-ann`), so the two nightly experiments are directly
//! comparable.
//!
//! Run:
//!   cargo run --release -p ruvector-dab-search --bin benchmark

use ruvector_dab_search::{
    dataset::{clustered_vectors, ground_truth, random_unit_vectors},
    graph::{FlatGraph, GraphConfig},
    metrics::{LatencyStats, WorkStats},
    recall_at_k,
    search::{AdaptiveGamma, FixedEf, Searcher},
};
use std::time::Instant;

// Dataset parameters match ADR-303's benchmark exactly (same seeds, same
// sizes) so this experiment controls for dataset/graph-construction quality
// as a confound against that prior nightly's negative result.
const N: usize = 2_000;
const DIM: usize = 16;
const N_CLUSTERS: usize = 10;
const CLUSTER_NOISE: f32 = 0.20;
const K: usize = 10;
const GRAPH_K: usize = 16;
/// Entry-routing seeds probed per query (see graph.rs docs — replaces both
/// entropy-ann's O(n) brute-force entry scan and this crate's earlier,
/// broken single-fixed-entry design).
const NUM_ENTRY_SEEDS: usize = 40;
const N_QUERIES_EASY: usize = 200;
const N_QUERIES_HARD: usize = 200;
const N_QUERIES_MIXED: usize = 400;

/// Pre-registered primary gamma. Chosen as the midpoint of the paper's valid
/// range (0, 2] before any benchmark was run on this dataset; the sweep
/// below over {0.2, 1.0} is reported as exploratory context only and does
/// not change which gamma the acceptance test below uses.
const GAMMA_PRIMARY: f32 = 0.5;
const GAMMA_SWEEP: [f32; 2] = [0.2, 1.0];

/// Production safety cap for candidate B, fixed in advance (not tuned to
/// results): roughly FixedEf(50)'s typical expansion count on this dataset.
const CAPPED_MAX_EXPANSIONS: u64 = 40;

/// High-recall reference budget for the recall-floor test.
const EF_REFERENCE: usize = 100;
/// Apples-to-apples budget with ADR-303's default baseline.
const EF_DEFAULT: usize = 50;

// ─── acceptance thresholds (fixed before the first run of this file) ───────
/// AdaptiveGamma(primary) must expand measurably more on hard queries than
/// easy queries: this is the direct test for "does it actually adapt",
/// contrasting with ADR-303's measured EntropyScaledEf, whose ef_actual was
/// 122-124 for every query regardless of difficulty (ratio ~= 1.00).
const ADAPT_RATIO_MIN: f64 = 1.15;
/// AdaptiveGamma(primary) recall on each query set must be within this many
/// absolute recall points of FixedEf(EF_REFERENCE) on the same set.
const RECALL_FLOOR_DELTA: f32 = 0.03;
/// On the hard query set specifically, AdaptiveGamma(primary) must beat a
/// FixedEf baseline whose ef is calibrated (on the MIXED set only, to avoid
/// leaking the hard-set test into its own calibration) to match
/// AdaptiveGamma's mean distance-computation budget on the mixed set.
const MATCHED_BUDGET_HARD_ADVANTAGE_MIN: f32 = 0.02;

fn system_info() {
    println!("=== Distance-Adaptive Beam (DAB) Search Benchmark ===");
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

fn eval_recall(
    searcher: &dyn Searcher,
    queries: &[Vec<f32>],
    corpus: &[Vec<f32>],
    k: usize,
) -> f32 {
    let total: f32 = queries
        .iter()
        .map(|q| {
            let gt = ground_truth(q, corpus, k);
            let out = searcher.search(q, k);
            recall_at_k(&gt, &out.hits, k)
        })
        .sum();
    total / queries.len() as f32
}

fn eval_work(searcher: &dyn Searcher, queries: &[Vec<f32>], k: usize) -> WorkStats {
    let mut w = WorkStats::new();
    for q in queries {
        w.record(searcher.search(q, k).dist_computations);
    }
    w
}

/// Linear-scan calibration: find the FixedEf `ef` whose mean distance-
/// computation count on `queries` is closest to `target`.
fn calibrate_ef_for_budget(
    graph: &FlatGraph,
    queries: &[Vec<f32>],
    k: usize,
    target: f64,
) -> usize {
    let mut best_ef = k;
    let mut best_diff = f64::MAX;
    for ef in (k..=300).step_by(2) {
        let searcher = FixedEf {
            graph,
            ef_search: ef,
        };
        let mean = eval_work(&searcher, queries, k).mean();
        let diff = (mean - target).abs();
        if diff < best_diff {
            best_diff = diff;
            best_ef = ef;
        }
    }
    best_ef
}

/// Linear-scan calibration: find the FixedEf `ef` whose recall on `queries`
/// is closest to `target_recall`.
fn calibrate_ef_for_recall(
    graph: &FlatGraph,
    queries: &[Vec<f32>],
    corpus: &[Vec<f32>],
    k: usize,
    target_recall: f32,
) -> usize {
    let mut best_ef = k;
    let mut best_diff = f32::MAX;
    for ef in (k..=300).step_by(2) {
        let searcher = FixedEf {
            graph,
            ef_search: ef,
        };
        let recall = eval_recall(&searcher, queries, corpus, k);
        let diff = (recall - target_recall).abs();
        if diff < best_diff {
            best_diff = diff;
            best_ef = ef;
        }
    }
    best_ef
}

fn print_row(
    name: &str,
    query_type: &str,
    n_queries: usize,
    recall: f32,
    work: &WorkStats,
    stats: &LatencyStats,
) {
    println!(
        "  {name:<24} {query_type:<8} n={n_queries:<5} recall={recall:.3}  \
        dist_comp(mean={:6.1} sd={:5.1} min={:4} max={:4})  \
        lat_mean={:6.1}us  {:.0} qps",
        work.mean(),
        work.stddev(),
        work.min(),
        work.max(),
        stats.mean_us,
        stats.throughput_qps,
    );
}

fn main() {
    system_info();

    println!("Dataset (identical construction to ADR-303):");
    println!("  N (corpus) : {N}");
    println!("  Dimensions : {DIM}");
    println!("  Clusters   : {N_CLUSTERS}  noise={CLUSTER_NOISE}");
    println!("  k (recall) : {K}");
    println!("  Graph K    : {GRAPH_K}");
    println!("  gamma      : primary={GAMMA_PRIMARY}, sweep={GAMMA_SWEEP:?}");
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
            num_entry_seeds: NUM_ENTRY_SEEDS,
        },
    );
    println!(
        "  graph built in {:.1}ms, entry_seeds={}",
        t1.elapsed().as_millis(),
        graph.entry_seeds.len()
    );
    println!();

    let easy_queries = clustered_vectors(N_QUERIES_EASY, DIM, N_CLUSTERS, 0.02, 101);
    let hard_queries = random_unit_vectors(N_QUERIES_HARD, DIM, 202);
    let mixed_queries = clustered_vectors(N_QUERIES_MIXED, DIM, N_CLUSTERS, CLUSTER_NOISE, 303);

    let mem_kb = graph.memory_bytes() / 1024;

    // ── Variant definitions ────────────────────────────────────────────────
    let fixed_default = FixedEf {
        graph: &graph,
        ef_search: EF_DEFAULT,
    };
    let fixed_reference = FixedEf {
        graph: &graph,
        ef_search: EF_REFERENCE,
    };
    let adaptive_primary = AdaptiveGamma {
        graph: &graph,
        gamma: GAMMA_PRIMARY,
        max_expansions: None,
    };
    let adaptive_capped = AdaptiveGamma {
        graph: &graph,
        gamma: GAMMA_PRIMARY,
        max_expansions: Some(CAPPED_MAX_EXPANSIONS),
    };
    let adaptive_sweep: Vec<AdaptiveGamma> = GAMMA_SWEEP
        .iter()
        .map(|&g| AdaptiveGamma {
            graph: &graph,
            gamma: g,
            max_expansions: None,
        })
        .collect();

    println!("─── Recall / Work / Latency by variant and query set ───");
    println!();

    let query_sets: [(&[Vec<f32>], &str); 3] = [
        (&easy_queries, "easy"),
        (&hard_queries, "hard"),
        (&mixed_queries, "mixed"),
    ];

    let mut all_variants: Vec<&dyn Searcher> = vec![
        &fixed_default,
        &fixed_reference,
        &adaptive_primary,
        &adaptive_capped,
    ];
    for s in &adaptive_sweep {
        all_variants.push(s);
    }

    // recall/work tables, keyed by (variant name, query label)
    use std::collections::HashMap;
    let mut recall_table: HashMap<(String, &str), f32> = HashMap::new();
    let mut work_table: HashMap<(String, &str), WorkStats> = HashMap::new();

    for &searcher in &all_variants {
        for &(queries, label) in &query_sets {
            let recall = eval_recall(searcher, queries, &corpus, K);
            let work = eval_work(searcher, queries, K);
            let (_, stats) =
                LatencyStats::measure(queries.len(), |i| searcher.search(&queries[i], K));
            print_row(
                &searcher.name(),
                label,
                queries.len(),
                recall,
                &work,
                &stats,
            );
            recall_table.insert((searcher.name(), label), recall);
            work_table.insert((searcher.name(), label), work);
        }
    }
    println!();
    println!("  Index memory: {mem_kb} KB");
    println!();

    // ── Acceptance test 1: adaptivity ratio (hard/easy mean dist_comp) ─────
    println!("─── Test 1: Does the stopping rule actually adapt per query? ───");
    let name_primary = adaptive_primary.name();
    let easy_work = work_table.get(&(name_primary.clone(), "easy")).unwrap();
    let hard_work = work_table.get(&(name_primary.clone(), "hard")).unwrap();
    let adapt_ratio = hard_work.mean() / easy_work.mean().max(1e-9);
    let test1_pass = adapt_ratio >= ADAPT_RATIO_MIN;
    println!(
        "  {name_primary}: mean dist_comp easy={:.1} hard={:.1} ratio(hard/easy)={:.3} (threshold >= {ADAPT_RATIO_MIN})",
        easy_work.mean(),
        hard_work.mean(),
        adapt_ratio
    );
    println!(
        "  Contrast — ADR-303 measured EntropyScaledEf's ef_actual at 122-124 for EVERY \
        query (ratio ~= 1.00), which is why it was rejected. This test is the same question \
        asked of a different signal."
    );
    println!("  [{}]", if test1_pass { "PASS" } else { "FAIL" });
    println!();

    // ── Acceptance test 2: recall floor vs high-recall reference ───────────
    println!("─── Test 2: Recall floor vs FixedEf({EF_REFERENCE}) reference ───");
    let mut test2_pass = true;
    for &(_, label) in &query_sets {
        let ref_recall = *recall_table.get(&(fixed_reference.name(), label)).unwrap();
        let adaptive_recall = *recall_table.get(&(name_primary.clone(), label)).unwrap();
        let ok = adaptive_recall >= ref_recall - RECALL_FLOOR_DELTA;
        test2_pass &= ok;
        println!(
            "  {label:<6} reference={ref_recall:.3} adaptive={adaptive_recall:.3} \
            delta={:+.3} (floor: adaptive >= reference - {RECALL_FLOOR_DELTA}) [{}]",
            adaptive_recall - ref_recall,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    println!();

    // ── Acceptance test 3: matched-budget control on the hard set ─────────
    println!("─── Test 3: Matched-budget control (crux test) ───");
    let target_budget = work_table
        .get(&(name_primary.clone(), "mixed"))
        .unwrap()
        .mean();
    let matched_ef = calibrate_ef_for_budget(&graph, &mixed_queries, K, target_budget);
    let fixed_matched = FixedEf {
        graph: &graph,
        ef_search: matched_ef,
    };
    let matched_budget_actual = eval_work(&fixed_matched, &mixed_queries, K).mean();
    println!(
        "  Calibrated on MIXED set only: FixedEf(ef={matched_ef}) has mean dist_comp={matched_budget_actual:.1} \
        (target from {name_primary} on mixed = {target_budget:.1})"
    );
    let adaptive_recall_hard = *recall_table.get(&(name_primary.clone(), "hard")).unwrap();
    let matched_recall_hard = eval_recall(&fixed_matched, &hard_queries, &corpus, K);
    let advantage = adaptive_recall_hard - matched_recall_hard;
    let test3_pass = advantage >= MATCHED_BUDGET_HARD_ADVANTAGE_MIN;
    println!(
        "  On HARD queries at ~matched average budget: {name_primary} recall={adaptive_recall_hard:.3} \
        vs FixedEf({matched_ef},matched) recall={matched_recall_hard:.3}  advantage={advantage:+.3} \
        (threshold >= {MATCHED_BUDGET_HARD_ADVANTAGE_MIN})"
    );
    println!(
        "  This is the test ADR-303 could not pass: does adaptively reallocating budget toward \
        harder queries beat a flat allocation at the same average cost?"
    );
    println!("  [{}]", if test3_pass { "PASS" } else { "FAIL" });
    println!();

    // ── Headline number: cost reduction at matched recall (paper's metric) ─
    println!("─── Headline: cost at matched recall (arXiv:2505.15636's own metric) ───");
    let adaptive_recall_mixed = *recall_table.get(&(name_primary.clone(), "mixed")).unwrap();
    let recall_matched_ef =
        calibrate_ef_for_recall(&graph, &mixed_queries, &corpus, K, adaptive_recall_mixed);
    let fixed_recall_matched = FixedEf {
        graph: &graph,
        ef_search: recall_matched_ef,
    };
    let fixed_recall_matched_work = eval_work(&fixed_recall_matched, &mixed_queries, K).mean();
    let adaptive_work_mixed = work_table
        .get(&(name_primary.clone(), "mixed"))
        .unwrap()
        .mean();
    let reduction_pct = if fixed_recall_matched_work > 0.0 {
        100.0 * (1.0 - adaptive_work_mixed / fixed_recall_matched_work)
    } else {
        0.0
    };
    println!(
        "  On MIXED queries at matched recall ({adaptive_recall_mixed:.3}): FixedEf(ef={recall_matched_ef}) \
        needs {fixed_recall_matched_work:.1} dist_comp/query vs {name_primary}'s {adaptive_work_mixed:.1} \
        ({reduction_pct:+.1}% change)"
    );
    println!();

    // ── Overall acceptance ──────────────────────────────────────────────────
    println!("─── Acceptance Result ───");
    let verdict = if !test1_pass || !test2_pass {
        "REJECT"
    } else if !test3_pass {
        "INCONCLUSIVE"
    } else {
        "ACCEPT"
    };
    println!(
        "  Test 1 (adapts per query):        {}",
        if test1_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Test 2 (recall floor):             {}",
        if test2_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Test 3 (beats matched budget):     {}",
        if test3_pass { "PASS" } else { "FAIL" }
    );
    println!("  VERDICT: {verdict}");

    if verdict == "REJECT" {
        std::process::exit(1);
    }
}
