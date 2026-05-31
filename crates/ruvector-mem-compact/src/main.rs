//! Benchmark binary for graph-cut memory compaction.
//!
//! Runs two benchmark suites:
//! 1. Moderate Gaussian dataset — compares ID-exact recall@K across strategies.
//! 2. High-redundancy episodic dataset — compares cluster-coverage recall.
//!
//! Usage:
//!   cargo run --release -p ruvector-mem-compact
//!   cargo run --release -p ruvector-mem-compact -- --n 5000 --dims 128

use ruvector_mem_compact::dataset::{
    generate_concept_queries, generate_queries, generate_redundant_store, generate_store,
    DatasetParams, RedundantParams,
};
use ruvector_mem_compact::metrics::{
    cluster_coverage_recall, memory_kb, query_latencies, recall_at_k, CompactionMetrics,
};
use ruvector_mem_compact::{
    AgeTtlCompactor, GraphCutCompactor, MemoryCompactor, ThresholdCompactor,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = get_arg(&args, "--n", 2_000usize);
    let dims = get_arg(&args, "--dims", 128usize);
    let k = get_arg(&args, "--k", 10usize);
    let nq = get_arg(&args, "--queries", 50usize);
    let nconcepts = get_arg(&args, "--concepts", 40usize);
    let copies = get_arg(&args, "--copies", 20usize);

    println!("=== RuVector Graph-Cut Memory Compaction Benchmark ===");
    println!("OS      : {}", std::env::consts::OS);
    println!("Arch    : {}", std::env::consts::ARCH);
    println!();

    // ── Suite A: Moderate Gaussian data ──────────────────────────────────────
    println!("--- Suite A: Moderate Gaussian data (ID-exact recall@{k}) ---");
    println!("N={n}, D={dims}, clusters=10, std=0.2, queries={nq}, target=50%");
    println!();

    let params_a = DatasetParams {
        n,
        dims,
        n_clusters: 10,
        cluster_std: 0.2,
        seed: 42,
    };
    let store_a = generate_store(&params_a);
    let queries_a = generate_queries(&params_a, nq);
    let target = 0.50f32;

    let threshold_c = ThresholdCompactor::new(0.90);
    let graph_cut_c = GraphCutCompactor::new(8, 0.70);
    let strategies_a: Vec<(&dyn MemoryCompactor, &str)> = vec![
        (&AgeTtlCompactor, "AgeTtl"),
        (&threshold_c, "ThresholdMerge"),
        (&graph_cut_c, "GraphCutCompact"),
    ];

    let mut suite_a: Vec<CompactionMetrics> = Vec::new();
    for (compactor, _) in &strategies_a {
        let result = compactor.compact(&store_a, target);
        let compacted = store_a.apply(&result.kept_ids);
        let rec_id = recall_at_k(&queries_a, &store_a, &compacted, k);
        let rec_clust = 0.0; // not labelled in suite A
        let (q_mean, q_p50, q_p95) = query_latencies(&queries_a, &compacted, k);
        let mem = memory_kb(compacted.len(), dims);
        suite_a.push(CompactionMetrics {
            variant: compactor.name(),
            n_before: n,
            n_after: compacted.len(),
            compaction_pct: 1.0 - result.kept_ratio,
            recall_id_exact: rec_id,
            recall_cluster_cov: rec_clust,
            compact_ms: result.duration.as_secs_f32() * 1000.0,
            query_us_mean: q_mean,
            query_us_p50: q_p50,
            query_us_p95: q_p95,
            memory_kb: mem,
            acceptance: false, // set below
        });
    }

    CompactionMetrics::print_header();
    for m in &suite_a {
        m.print_row();
    }
    println!();

    // ── Suite B: High-redundancy episodic data ────────────────────────────────
    println!("--- Suite B: High-redundancy episodic data (cluster-coverage recall@5) ---");
    println!("concepts={nconcepts}, copies/concept={copies}, D=64, dup_noise=0.04, target=50%");
    println!("Insight: AgeTtl drops ALL copies of old concepts; GraphCut retains one per concept.");
    println!();

    let params_b = RedundantParams {
        n_concepts: nconcepts,
        copies_per_concept: copies,
        dims: 64,
        dup_noise: 0.04,
        seed: 1337,
    };
    let store_b = generate_redundant_store(&params_b);
    let queries_b = generate_concept_queries(&params_b);

    // Cluster label: entry i belongs to concept (i / copies_per_concept).
    let labels_b: Vec<usize> = (0..nconcepts * copies).map(|i| i / copies).collect();
    let k_cov = 5;

    let threshold_b = ThresholdCompactor::new(0.90);
    let graph_cut_b = GraphCutCompactor::new(8, 0.70);
    let strategies_b: Vec<(&dyn MemoryCompactor, &str)> = vec![
        (&AgeTtlCompactor, "AgeTtl"),
        (&threshold_b, "ThresholdMerge"),
        (&graph_cut_b, "GraphCutCompact"),
    ];

    let nb = store_b.len();
    let mut suite_b: Vec<CompactionMetrics> = Vec::new();
    for (compactor, _) in &strategies_b {
        let result = compactor.compact(&store_b, target);
        let compacted = store_b.apply(&result.kept_ids);
        let rec_clust = cluster_coverage_recall(&queries_b, &store_b, &compacted, &labels_b, k_cov);
        let rec_id = recall_at_k(&queries_b, &store_b, &compacted, 1);
        let (q_mean, q_p50, q_p95) = query_latencies(&queries_b, &compacted, k_cov);
        let mem = memory_kb(compacted.len(), 64);
        let accept = compactor.name() != "GraphCutCompact" || rec_clust >= 0.90;
        suite_b.push(CompactionMetrics {
            variant: compactor.name(),
            n_before: nb,
            n_after: compacted.len(),
            compaction_pct: 1.0 - result.kept_ratio,
            recall_id_exact: rec_id,
            recall_cluster_cov: rec_clust,
            compact_ms: result.duration.as_secs_f32() * 1000.0,
            query_us_mean: q_mean,
            query_us_p50: q_p50,
            query_us_p95: q_p95,
            memory_kb: mem,
            acceptance: accept,
        });
    }

    CompactionMetrics::print_header();
    for m in &suite_b {
        m.print_row();
    }
    println!();

    // ── Acceptance gate ───────────────────────────────────────────────────────
    println!("=== Acceptance Gate ===");
    println!("Criterion: GraphCutCompact cluster-coverage recall@{k_cov} >= 90.0% (Suite B)");
    println!();

    let gc_b = suite_b
        .iter()
        .find(|m| m.variant == "GraphCutCompact")
        .unwrap();
    let ttl_b = suite_b.iter().find(|m| m.variant == "AgeTtl").unwrap();
    let gc_a = suite_a
        .iter()
        .find(|m| m.variant == "GraphCutCompact")
        .unwrap();
    let ttl_a = suite_a.iter().find(|m| m.variant == "AgeTtl").unwrap();

    println!(
        "Suite A — ID-exact recall@{k}: GraphCut {:.1}%  AgeTtl {:.1}%  Δ={:+.1} pp",
        gc_a.recall_id_exact * 100.0,
        ttl_a.recall_id_exact * 100.0,
        (gc_a.recall_id_exact - ttl_a.recall_id_exact) * 100.0,
    );
    println!(
        "Suite B — Cluster-cov recall@{k_cov}: GraphCut {:.1}%  AgeTtl {:.1}%  Δ={:+.1} pp",
        gc_b.recall_cluster_cov * 100.0,
        ttl_b.recall_cluster_cov * 100.0,
        (gc_b.recall_cluster_cov - ttl_b.recall_cluster_cov) * 100.0,
    );

    if gc_b.acceptance {
        println!("RESULT: PASS — GraphCutCompact meets cluster-coverage recall threshold.");
    } else {
        eprintln!("RESULT: FAIL — GraphCutCompact cluster-coverage recall below threshold.");
        std::process::exit(1);
    }
}

fn get_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T
where
    T::Err: std::fmt::Debug,
{
    for w in args.windows(2) {
        if w[0] == flag {
            return w[1].parse().unwrap_or(default);
        }
    }
    default
}
