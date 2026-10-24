//! Bounded RAG MinCut benchmark
//!
//! Measures three retrieval variants across two dataset sizes.
//! Run: cargo run --release -p ruvector-bounded-rag --bin benchmark
use ruvector_bounded_rag::sparse_knn::SparseKnnConfig;
use ruvector_bounded_rag::{
    BoundedRetriever, Corpus, GraphBfsRetriever, MinCutRetriever, Query, RetrieverConfig,
    SparseKnnMinCutRetriever, TopKRetriever,
};

use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::time::Instant;

struct BenchCase {
    n_chunks: usize,
    n_queries: usize,
    dim: usize,
    n_clusters: usize,
    budget: usize,
    edge_threshold: f32,
    seed_threshold: f32,
    /// Dense MinCutBounded is O(n^2*d); skip it above this crate's known
    /// cost cliff so the benchmark finishes in reasonable wall-clock time.
    run_dense_mincut: bool,
    /// Candidate A (unbounded threshold graph) degrades to dense-graph cost
    /// on tight clusters at scale (measured: ~1000x slower than Candidate B
    /// at n=8000) — skip it above the cliff for the same wall-clock reason.
    /// Candidate B (degree-capped) always runs; it is the point of this
    /// comparison.
    run_sparse_threshold: bool,
}

fn build_corpus(case: &BenchCase, rng: &mut StdRng) -> (Corpus, Vec<Query>) {
    let normal = Normal::new(0.0_f32, 0.08).unwrap();
    let per_cluster = case.n_chunks / case.n_clusters;

    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(case.n_chunks);
    let mut labels: Vec<u32> = Vec::with_capacity(case.n_chunks);

    for cluster in 0..case.n_clusters {
        for _ in 0..per_cluster {
            let mut v = vec![0.0_f32; case.dim];
            v[cluster % case.dim] = 1.0;
            for x in v.iter_mut() {
                *x += normal.sample(rng);
            }
            vecs.push(v);
            labels.push(cluster as u32);
        }
    }
    let corpus = Corpus::from_vecs(vecs).with_labels(labels);

    let mut queries: Vec<Query> = Vec::with_capacity(case.n_queries);
    for i in 0..case.n_queries {
        let target_cluster = i % case.n_clusters;
        let mut qv = vec![0.0_f32; case.dim];
        qv[target_cluster % case.dim] = 1.0;
        for x in qv.iter_mut() {
            *x += normal.sample(rng) * 0.05;
        }
        queries.push(Query::new(qv).with_relevant([target_cluster as u32]));
    }

    (corpus, queries)
}

struct Stats {
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    throughput_qps: f64,
    mean_precision: f64,
    mean_budget_util: f64,
}

fn run_variant(retriever: &dyn BoundedRetriever, corpus: &Corpus, queries: &[Query]) -> Stats {
    let mut latencies_us: Vec<f64> = Vec::with_capacity(queries.len());
    let mut precisions: Vec<f64> = Vec::with_capacity(queries.len());
    let mut budgets: Vec<f64> = Vec::with_capacity(queries.len());

    let total_start = Instant::now();
    for q in queries {
        let t0 = Instant::now();
        let result = retriever.retrieve(corpus, q);
        let elapsed = t0.elapsed().as_micros() as f64;
        latencies_us.push(elapsed);
        precisions.push(result.precision(corpus, q) as f64);
        budgets.push(result.budget_utilisation as f64);
    }
    let total_elapsed = total_start.elapsed().as_secs_f64();

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = latencies_us.len();
    let mean_us = latencies_us.iter().sum::<f64>() / n as f64;
    let p50_us = latencies_us[n / 2];
    let p95_us = latencies_us[(n * 95 / 100).min(n - 1)];
    let throughput_qps = queries.len() as f64 / total_elapsed;
    let mean_precision = precisions.iter().sum::<f64>() / n as f64;
    let mean_budget_util = budgets.iter().sum::<f64>() / n as f64;

    Stats {
        mean_us,
        p50_us,
        p95_us,
        throughput_qps,
        mean_precision,
        mean_budget_util,
    }
}

fn print_header() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          ruvector-bounded-rag: MinCut Context Window Benchmark             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Print system info
    println!("Platform : {}", std::env::consts::OS);
    println!("Arch     : {}", std::env::consts::ARCH);
    println!("Rust     : {}", env!("CARGO_PKG_RUST_VERSION", "unknown"));
    println!();
}

fn print_case_header(case: &BenchCase) {
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!(
        "Dataset  : {} chunks × {}d, {} clusters, {} queries, budget={}",
        case.n_chunks, case.dim, case.n_clusters, case.n_queries, case.budget
    );
    println!(
        "Config   : edge_threshold={:.2}  seed_threshold={:.2}",
        case.edge_threshold, case.seed_threshold
    );
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>12} {:>10} {:>12}",
        "Variant", "Mean(μs)", "p50(μs)", "p95(μs)", "QPS", "Precision", "BudgetUtil"
    );
    println!("{}", "─".repeat(90));
}

fn print_row(name: &str, s: &Stats) {
    println!(
        "{:<20} {:>10.1} {:>10.1} {:>10.1} {:>12.0} {:>9.3} {:>12.3}",
        name, s.mean_us, s.p50_us, s.p95_us, s.throughput_qps, s.mean_precision, s.mean_budget_util
    );
}

fn run_case(case: &BenchCase) {
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    let (corpus, queries) = build_corpus(case, &mut rng);

    let cfg = RetrieverConfig {
        budget: case.budget,
        edge_threshold: case.edge_threshold,
        seed_threshold: case.seed_threshold,
    };

    print_case_header(case);

    let topk = TopKRetriever::new(cfg.clone());
    let bfs = GraphBfsRetriever::new(cfg.clone());
    // Candidate A: cheaper *discovery* of the same unbounded threshold
    // graph MinCutBounded uses (isolates construction cost).
    let sparse_threshold = SparseKnnMinCutRetriever::new(cfg.clone());
    // Candidate B: mutual top-k degree cap, informed by Candidate A's
    // result that threshold graphs stay dense on tight clusters regardless
    // of discovery method. k scales with budget so the cap doesn't starve
    // the retriever of enough candidates to fill it.
    let max_degree = (case.budget * 2).max(10);
    let sparse_capped =
        SparseKnnMinCutRetriever::new(cfg.clone()).with_lsh_config(SparseKnnConfig {
            max_degree: Some(max_degree),
            ..SparseKnnConfig::default()
        });

    let s_topk = run_variant(&topk, &corpus, &queries);
    let s_bfs = run_variant(&bfs, &corpus, &queries);
    let s_sparse_capped = run_variant(&sparse_capped, &corpus, &queries);

    print_row("TopK (baseline)", &s_topk);
    print_row("GraphBFS", &s_bfs);

    let s_mc = if case.run_dense_mincut {
        let mc = MinCutRetriever::new(cfg.clone());
        let s = run_variant(&mc, &corpus, &queries);
        print_row("MinCutBounded (dense)", &s);
        Some(s)
    } else {
        println!(
            "{:<20} {:>10}",
            "MinCutBounded (dense)", "SKIPPED (O(n^2) cost cliff — see n=3000 case)"
        );
        None
    };

    let s_sparse_threshold = if case.run_sparse_threshold {
        let s = run_variant(&sparse_threshold, &corpus, &queries);
        print_row("SparseKnn (A: threshold)", &s);
        Some(s)
    } else {
        println!(
            "{:<20} {:>10}",
            "SparseKnn (A: threshold)",
            "SKIPPED (degrades to dense-graph cost on tight clusters — see n=3000 case)"
        );
        None
    };

    print_row(
        &format!("SparseKnn (B: k={max_degree} cap)"),
        &s_sparse_capped,
    );

    println!();

    if let Some(ref s_mc) = s_mc {
        if let Some(ref s_a) = s_sparse_threshold {
            println!(
                "Candidate A (threshold) vs dense: {:.2}x mean latency, precision {:.3} -> {:.3}",
                s_mc.mean_us / s_a.mean_us,
                s_mc.mean_precision,
                s_a.mean_precision
            );
        }
        println!(
            "Candidate B (k={max_degree} cap)  vs dense: {:.2}x mean latency, precision {:.3} -> {:.3}",
            s_mc.mean_us / s_sparse_capped.mean_us,
            s_mc.mean_precision,
            s_sparse_capped.mean_precision
        );
    }

    // Acceptance check — every variant must clear the precision floor.
    let threshold = 0.70;
    let mut pass = s_topk.mean_precision >= threshold
        && s_bfs.mean_precision >= threshold
        && s_sparse_capped.mean_precision >= threshold;
    if let Some(ref s_a) = s_sparse_threshold {
        pass = pass && s_a.mean_precision >= threshold;
    }
    if let Some(ref s_mc) = s_mc {
        pass = pass && s_mc.mean_precision >= threshold;
    }

    if pass {
        println!("✓ PASS — all run variants achieved precision >= {threshold:.2}");
    } else {
        println!("✗ FAIL — a variant fell below the {threshold:.2} precision threshold");
    }

    // Memory estimate (rough)
    let chunk_bytes = case.n_chunks * case.dim * 4;
    let dense_edges = case.n_chunks * case.n_chunks / 2; // upper bound
    let dense_graph_bytes = dense_edges * 8; // (usize, f32)
    println!(
        "Memory   : chunks≈{}KB  dense-graph≈{}KB (upper bound, for reference)",
        chunk_bytes / 1024,
        dense_graph_bytes / 1024
    );
    println!();
}

fn print_sparsity_stats(n: usize, dim: usize) {
    use ruvector_bounded_rag::sparse_knn::{build_sparse_edges, SparseKnnConfig};

    let mut rng = StdRng::seed_from_u64(0xA11CE);
    let normal = Normal::new(0.0_f32, 0.1).unwrap();
    let n_clusters = 6usize;
    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = vec![0.0_f32; dim];
        v[i % n_clusters % dim] = 1.0;
        for x in v.iter_mut() {
            *x += normal.sample(&mut rng);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x /= norm.max(1e-10);
        }
        vecs.push(v);
    }
    let (_edges, stats) = build_sparse_edges(&vecs, dim, 0.70, &SparseKnnConfig::default());
    let dense_pairs = n * (n - 1) / 2;
    println!(
        "n={n:<6} dense_pairs={dense_pairs:<12} lsh_candidate_pairs={:<10} reduction={:.1}x",
        stats.candidate_pairs_checked,
        dense_pairs as f64 / stats.candidate_pairs_checked.max(1) as f64
    );
}

fn main() {
    print_header();

    // Small case: 200 chunks, quick sanity
    run_case(&BenchCase {
        n_chunks: 200,
        n_queries: 50,
        dim: 64,
        n_clusters: 4,
        budget: 20,
        edge_threshold: 0.70,
        seed_threshold: 0.45,
        run_dense_mincut: true,
        run_sparse_threshold: true,
    });

    // Medium case: 1000 chunks
    run_case(&BenchCase {
        n_chunks: 1000,
        n_queries: 100,
        dim: 64,
        n_clusters: 5,
        budget: 30,
        edge_threshold: 0.70,
        seed_threshold: 0.45,
        run_dense_mincut: true,
        run_sparse_threshold: true,
    });

    // Larger case: 3000 chunks — this is the documented dense MinCut cost
    // cliff from the 2026-07-25 nightly report (~1.27s mean per query).
    run_case(&BenchCase {
        n_chunks: 3000,
        n_queries: 30,
        dim: 32,
        n_clusters: 6,
        budget: 40,
        edge_threshold: 0.72,
        seed_threshold: 0.45,
        run_dense_mincut: true,
        run_sparse_threshold: true,
    });

    // Beyond the cliff: dense MinCut AND Candidate A (unbounded threshold)
    // are both skipped — both degrade to dense-graph flow-network cost on
    // tight clusters (measured at n=3000). Candidate B (degree-capped)
    // keeps running — this is the regime the hypothesis is actually about.
    run_case(&BenchCase {
        n_chunks: 8000,
        n_queries: 20,
        dim: 32,
        n_clusters: 6,
        budget: 40,
        edge_threshold: 0.72,
        seed_threshold: 0.45,
        run_dense_mincut: false,
        run_sparse_threshold: false,
    });

    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("  Sparse graph construction cost (LSH candidate pairs vs. dense all-pairs)");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    for &n in &[200usize, 1000, 3000, 8000, 20000] {
        print_sparsity_stats(n, 32);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("  Benchmark complete. All numbers from release build on this hardware.");
    println!("═══════════════════════════════════════════════════════════════════════════════");
}
