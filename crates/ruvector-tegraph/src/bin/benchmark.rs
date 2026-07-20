/// Typed-Edge Navigable Graph (TENG) — nightly benchmark
///
/// Measures three retrieval variants:
///   1. VectorOnly  — pure NSW beam search (baseline)
///   2. EdgeExpand  — NSW + typed-edge candidate expansion
///   3. EdgeConstrained — NSW filtered to SameDocument-connected nodes
///
/// Metrics: vector recall@10, semantic recall@10, mean/p50/p95 latency (μs),
/// QPS, and memory estimate.
///
/// Run:
///   cargo run --release -p ruvector-tegraph --bin benchmark
use ruvector_tegraph::{
    dataset::{generate, generate_queries, DatasetConfig},
    recall_at_k, recall_at_k_pairs,
    variants::{EdgeConstraint, TengIndex},
};
use std::time::Instant;

// ─── benchmark parameters ────────────────────────────────────────────────────

const N_DOCS: usize       = 100;
const NODES_PER_DOC: usize = 50;   // total = 5,000 nodes
const DIMS: usize          = 128;
const M: usize             = 16;   // NSW connections per node
const EF_CONSTRUCTION: usize = 64;
const EF_SEARCH: usize     = 80;
const K: usize             = 10;
const N_QUERIES: usize     = 500;
const EDGE_FACTOR: f32     = 0.30;  // typed-edge blend weight for EdgeExpand

// ─── acceptance thresholds ───────────────────────────────────────────────────

const MIN_VECTOR_RECALL_BASELINE: f32   = 0.65;
const MIN_VECTOR_RECALL_EXPAND: f32     = 0.60;
const MIN_SEMANTIC_RECALL_EXPAND: f32   = 0.70;
const MIN_CONSTRAINED_RECALL: f32       = 0.55;
const MIN_QPS_BASELINE: f64             = 500.0;

// ─── latency helpers ─────────────────────────────────────────────────────────

fn percentile(sorted_us: &[u128], p: f64) -> u128 {
    if sorted_us.is_empty() { return 0; }
    let idx = ((sorted_us.len() as f64 * p / 100.0) as usize)
        .min(sorted_us.len() - 1);
    sorted_us[idx]
}

fn memory_estimate_bytes(n: usize, dims: usize, m: usize, avg_edges: f32) -> usize {
    let vec_bytes   = n * dims * 4;               // f32 embeddings
    let adj_bytes   = n * m * 2 * 8;             // NSW adjacency (usize pairs)
    let edge_bytes  = (n as f32 * avg_edges * 16.0) as usize; // TypedEdge ~16B each
    let meta_bytes  = n * 24;                     // id + doc_id + vec pointer
    vec_bytes + adj_bytes + edge_bytes + meta_bytes
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    print_header();

    let cfg = DatasetConfig {
        n_docs: N_DOCS,
        nodes_per_doc: NODES_PER_DOC,
        dims: DIMS,
        same_doc_edges: 4,
        ref_edges: 2,
        cooccur_edges: 2,
        seed: 0xc0de_cafe_1234_5678,
    };

    let n_total = cfg.total_nodes();
    let avg_edges = (cfg.same_doc_edges + cfg.ref_edges + cfg.cooccur_edges + 1) as f32;

    // ── build ────────────────────────────────────────────────────────────────
    println!("Building TENG index ({} nodes, {} dims)...", n_total, DIMS);
    let nodes = generate(&cfg);
    let t_build = Instant::now();
    let index = TengIndex::build(nodes, M, EF_CONSTRUCTION, EF_SEARCH);
    let build_ms = t_build.elapsed().as_millis();
    println!("  Build complete in {}ms", build_ms);
    println!();

    // ── queries ──────────────────────────────────────────────────────────────
    let queries = generate_queries(DIMS, N_QUERIES, cfg.seed);

    // ── Variant 1: VectorOnly ────────────────────────────────────────────────
    let mut vo_recalls   = Vec::with_capacity(N_QUERIES);
    let mut vo_latencies = Vec::with_capacity(N_QUERIES);
    for q in &queries {
        let t = Instant::now();
        let result = index.search_vector_only(q, K);
        vo_latencies.push(t.elapsed().as_micros());
        let gt = index.brute_force_knn(q, K);
        vo_recalls.push(recall_at_k_pairs(&result, &gt, K));
    }
    vo_latencies.sort_unstable();
    let vo_mean = vo_latencies.iter().sum::<u128>() as f64 / N_QUERIES as f64;
    let vo_p50  = percentile(&vo_latencies, 50.0);
    let vo_p95  = percentile(&vo_latencies, 95.0);
    let vo_qps  = 1_000_000.0 / vo_mean;
    let vo_recall = vo_recalls.iter().sum::<f32>() / N_QUERIES as f32;

    // ── Variant 2: EdgeExpand ────────────────────────────────────────────────
    let mut ee_vec_recalls  = Vec::with_capacity(N_QUERIES);
    let mut ee_sem_recalls  = Vec::with_capacity(N_QUERIES);
    let mut ee_latencies    = Vec::with_capacity(N_QUERIES);
    for q in &queries {
        let t = Instant::now();
        let result = index.search_edge_expand(q, K, EDGE_FACTOR);
        ee_latencies.push(t.elapsed().as_micros());
        let gt_vec = index.brute_force_knn(q, K);
        let gt_sem = index.semantic_ground_truth(q, K);
        ee_vec_recalls.push(recall_at_k_pairs(&result, &gt_vec, K));
        ee_sem_recalls.push(recall_at_k(&result, &gt_sem, K));
    }
    ee_latencies.sort_unstable();
    let ee_mean     = ee_latencies.iter().sum::<u128>() as f64 / N_QUERIES as f64;
    let ee_p50      = percentile(&ee_latencies, 50.0);
    let ee_p95      = percentile(&ee_latencies, 95.0);
    let ee_qps      = 1_000_000.0 / ee_mean;
    let ee_vec_rec  = ee_vec_recalls.iter().sum::<f32>() / N_QUERIES as f32;
    let ee_sem_rec  = ee_sem_recalls.iter().sum::<f32>() / N_QUERIES as f32;

    // ── Variant 3: EdgeConstrained ───────────────────────────────────────────
    let constraint = EdgeConstraint::Type(ruvector_tegraph::EdgeType::SameDocument);
    let mut ec_recalls   = Vec::with_capacity(N_QUERIES);
    let mut ec_latencies = Vec::with_capacity(N_QUERIES);
    for q in &queries {
        let t = Instant::now();
        let result = index.search_edge_constrained(q, K, &constraint);
        ec_latencies.push(t.elapsed().as_micros());
        let constraint_gt_type = EdgeConstraint::Type(ruvector_tegraph::EdgeType::SameDocument);
        let gt = index.constrained_ground_truth(q, K, &constraint_gt_type);
        ec_recalls.push(recall_at_k_pairs(&result, &gt, K));
    }
    ec_latencies.sort_unstable();
    let ec_mean   = ec_latencies.iter().sum::<u128>() as f64 / N_QUERIES as f64;
    let ec_p50    = percentile(&ec_latencies, 50.0);
    let ec_p95    = percentile(&ec_latencies, 95.0);
    let ec_qps    = 1_000_000.0 / ec_mean;
    let ec_recall = ec_recalls.iter().sum::<f32>() / N_QUERIES as f32;

    let mem_bytes = memory_estimate_bytes(n_total, DIMS, M, avg_edges);

    // ── results table ────────────────────────────────────────────────────────
    println!("┌─────────────────────┬───────────┬───────────┬────────────┬──────────┬──────────┬──────────┬────────────────┬──────────────────┐");
    println!("│ Variant             │ Vec R@10  │ Sem R@10  │ Mean (μs)  │ p50 (μs) │ p95 (μs) │ QPS      │ Mem (MB)       │ Constraints      │");
    println!("├─────────────────────┼───────────┼───────────┼────────────┼──────────┼──────────┼──────────┼────────────────┼──────────────────┤");
    println!(
        "│ VectorOnly          │ {:<9.3} │ {:<9} │ {:<10.1} │ {:<8} │ {:<8} │ {:<8.0} │ {:<14.2} │ None             │",
        vo_recall, "—",
        vo_mean, vo_p50, vo_p95, vo_qps,
        mem_bytes as f64 / 1_048_576.0,
    );
    println!(
        "│ EdgeExpand(f={:.2})  │ {:<9.3} │ {:<9.3} │ {:<10.1} │ {:<8} │ {:<8} │ {:<8.0} │ {:<14.2} │ edge_factor={:.2} │",
        EDGE_FACTOR,
        ee_vec_rec, ee_sem_rec,
        ee_mean, ee_p50, ee_p95, ee_qps,
        mem_bytes as f64 / 1_048_576.0,
        EDGE_FACTOR,
    );
    println!(
        "│ EdgeConstrained     │ {:<9.3} │ {:<9} │ {:<10.1} │ {:<8} │ {:<8} │ {:<8.0} │ {:<14.2} │ SameDocument     │",
        ec_recall, "—",
        ec_mean, ec_p50, ec_p95, ec_qps,
        mem_bytes as f64 / 1_048_576.0,
    );
    println!("└─────────────────────┴───────────┴───────────┴────────────┴──────────┴──────────┴──────────┴────────────────┴──────────────────┘");
    println!();

    // ── dataset summary ──────────────────────────────────────────────────────
    println!("Dataset:  {} nodes  ·  {} dims  ·  {} queries  ·  k={}", n_total, DIMS, N_QUERIES, K);
    println!("Graph:    M={}  ef_construction={}  ef_search={}", M, EF_CONSTRUCTION, EF_SEARCH);
    println!("Build:    {}ms", build_ms);
    println!("Mem est:  {:.2} MB", mem_bytes as f64 / 1_048_576.0);
    println!();

    // ── semantic recall delta ────────────────────────────────────────────────
    let sem_delta = ee_sem_rec - vo_recall;
    println!(
        "Semantic recall improvement (EdgeExpand vs VectorOnly): {:+.3}  ({:.1}% relative)",
        sem_delta,
        sem_delta / vo_recall.max(1e-6) * 100.0,
    );
    println!();

    // ── acceptance test ──────────────────────────────────────────────────────
    println!("──────── Acceptance Tests ────────");
    let mut all_pass = true;

    let checks: &[(&str, bool, &str)] = &[
        (
            "VectorOnly vector recall ≥ threshold",
            vo_recall >= MIN_VECTOR_RECALL_BASELINE,
            &format!("{:.3} ≥ {:.3}", vo_recall, MIN_VECTOR_RECALL_BASELINE),
        ),
        (
            "EdgeExpand vector recall ≥ threshold",
            ee_vec_rec >= MIN_VECTOR_RECALL_EXPAND,
            &format!("{:.3} ≥ {:.3}", ee_vec_rec, MIN_VECTOR_RECALL_EXPAND),
        ),
        (
            "EdgeExpand semantic recall ≥ threshold",
            ee_sem_rec >= MIN_SEMANTIC_RECALL_EXPAND,
            &format!("{:.3} ≥ {:.3}", ee_sem_rec, MIN_SEMANTIC_RECALL_EXPAND),
        ),
        (
            "EdgeConstrained recall ≥ threshold",
            ec_recall >= MIN_CONSTRAINED_RECALL,
            &format!("{:.3} ≥ {:.3}", ec_recall, MIN_CONSTRAINED_RECALL),
        ),
        (
            "VectorOnly QPS ≥ threshold",
            vo_qps >= MIN_QPS_BASELINE,
            &format!("{:.0} ≥ {:.0}", vo_qps, MIN_QPS_BASELINE),
        ),
    ];

    for (label, pass, detail) in checks {
        let marker = if *pass { "PASS" } else { "FAIL" };
        println!("  [{}] {} — {}", marker, label, detail);
        if !pass { all_pass = false; }
    }

    println!();
    if all_pass {
        println!("Result: ALL PASS — TENG nightly benchmark complete.");
    } else {
        println!("Result: ONE OR MORE CHECKS FAILED — see table above.");
        std::process::exit(1);
    }
}

fn print_header() {
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!(" ruvector-tegraph: Typed-Edge Navigable Graph (TENG) Nightly Benchmark");
    println!(" RuVector 2026 — Three-variant hybrid vector+semantic retrieval");
    println!("══════════════════════════════════════════════════════════════════════════════");
    // Runtime info
    println!("OS:   {}", std::env::consts::OS);
    println!("Arch: {}", std::env::consts::ARCH);
    println!();
}
