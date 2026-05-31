//! MUVERA unified benchmark — produces the real numbers cited in the research doc.
//!
//! Usage:
//!   cargo run --release -p ruvector-muvera
//!   cargo run --release -p ruvector-muvera -- --fast   (sub-5 s smoke run)
//!
//! For each (n_docs, tokens_per_doc, orig_dim, num_reps) configuration, reports:
//!   - Build time (ms)
//!   - Query throughput (QPS)
//!   - Recall@10 vs brute-force MaxSim (flat and HNSW)
//!   - Memory usage (bytes)

use std::sync::Arc;
use std::time::Instant;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_muvera::{
    BruteForceMaxSim, FdeEncoder, FlatFdeIndex, HnswFdeIndex, MultiVecIndex, SearchResult,
    recall_at_k,
};

struct Config {
    n_docs: usize,
    tokens_per_doc: usize,
    tokens_per_query: usize,
    orig_dim: usize,
    num_reps: usize,
    n_queries: usize,
    k: usize,
    label: &'static str,
}

fn generate_corpus(
    n: usize,
    tokens: usize,
    dim: usize,
    seed: u64,
) -> Vec<Vec<Vec<f32>>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();
    (0..n)
        .map(|_| {
            (0..tokens)
                .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
                .collect()
        })
        .collect()
}

struct Row {
    variant: String,
    build_ms: f64,
    qps: f64,
    recall: f64,
    mem_bytes: usize,
    n_docs: usize,
}

fn run_config(cfg: &Config) -> Vec<Row> {
    println!("\n=== {} ===", cfg.label);
    println!(
        "  n_docs={} tokens/doc={} tokens/query={} dim={} reps={} k={}",
        cfg.n_docs,
        cfg.tokens_per_doc,
        cfg.tokens_per_query,
        cfg.orig_dim,
        cfg.num_reps,
        cfg.k
    );

    let docs = generate_corpus(cfg.n_docs, cfg.tokens_per_doc, cfg.orig_dim, 42);
    let queries = generate_corpus(cfg.n_queries, cfg.tokens_per_query, cfg.orig_dim, 99);

    let encoder = Arc::new(
        FdeEncoder::new(cfg.num_reps, cfg.orig_dim, 7)
            .expect("encoder init"),
    );

    let mut rows = Vec::new();

    // --- Variant 1: BruteForceMaxSim ---
    let t0 = Instant::now();
    let bf =
        BruteForceMaxSim::build(docs.clone(), encoder.clone()).expect("brute force build");
    let build_ms_bf = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let mut truths: Vec<Vec<SearchResult>> = Vec::with_capacity(cfg.n_queries);
    for q in &queries {
        truths.push(bf.search(q, cfg.k).expect("brute search"));
    }
    let qps_bf = cfg.n_queries as f64 / t1.elapsed().as_secs_f64();

    println!(
        "  BruteForce: build={build_ms_bf:.1}ms  QPS={qps_bf:.0}  mem={}B",
        bf.memory_bytes()
    );
    rows.push(Row {
        variant: "BruteForceMaxSim".to_string(),
        build_ms: build_ms_bf,
        qps: qps_bf,
        recall: 1.0,
        mem_bytes: bf.memory_bytes(),
        n_docs: cfg.n_docs,
    });

    // --- Variant 2: FlatFDE ---
    let t0 = Instant::now();
    let flat = FlatFdeIndex::build(docs.clone(), encoder.clone()).expect("flat fde build");
    let build_ms_flat = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let mut flat_recall_sum = 0.0_f64;
    for (i, q) in queries.iter().enumerate() {
        let got = flat.search(q, cfg.k).expect("flat search");
        flat_recall_sum += recall_at_k(&truths[i], &got, cfg.k);
    }
    let qps_flat = cfg.n_queries as f64 / t1.elapsed().as_secs_f64();
    let flat_recall = flat_recall_sum / cfg.n_queries as f64;

    println!(
        "  FlatFDE:    build={build_ms_flat:.1}ms  QPS={qps_flat:.0}  recall@{}={:.3}  mem={}B",
        cfg.k,
        flat_recall,
        flat.memory_bytes()
    );
    rows.push(Row {
        variant: "FlatFDE".to_string(),
        build_ms: build_ms_flat,
        qps: qps_flat,
        recall: flat_recall,
        mem_bytes: flat.memory_bytes(),
        n_docs: cfg.n_docs,
    });

    // --- Variant 3: HnswFDE ---
    let t0 = Instant::now();
    let hnsw = HnswFdeIndex::build(docs.clone(), encoder.clone()).expect("hnsw fde build");
    let build_ms_hnsw = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let mut hnsw_recall_sum = 0.0_f64;
    for (i, q) in queries.iter().enumerate() {
        let got = hnsw.search(q, cfg.k).expect("hnsw search");
        hnsw_recall_sum += recall_at_k(&truths[i], &got, cfg.k);
    }
    let qps_hnsw = cfg.n_queries as f64 / t1.elapsed().as_secs_f64();
    let hnsw_recall = hnsw_recall_sum / cfg.n_queries as f64;

    println!(
        "  HnswFDE:    build={build_ms_hnsw:.1}ms  QPS={qps_hnsw:.0}  recall@{}={:.3}  mem={}B",
        cfg.k,
        hnsw_recall,
        hnsw.memory_bytes()
    );
    rows.push(Row {
        variant: "HnswFDE".to_string(),
        build_ms: build_ms_hnsw,
        qps: qps_hnsw,
        recall: hnsw_recall,
        mem_bytes: hnsw.memory_bytes(),
        n_docs: cfg.n_docs,
    });

    rows
}

fn main() {
    let fast = std::env::args().any(|a| a == "--fast");

    println!("MUVERA Benchmark — ruvector-muvera");
    println!("Hardware: {}", hardware_string());
    println!("Mode: {}", if fast { "fast (smoke)" } else { "full" });

    let configs: Vec<Config> = if fast {
        vec![
            Config {
                n_docs: 500,
                tokens_per_doc: 16,
                tokens_per_query: 8,
                orig_dim: 32,
                num_reps: 8,
                n_queries: 50,
                k: 10,
                label: "small (500 docs, 16 tok, 32D, 8 reps) [fast]",
            },
            Config {
                n_docs: 1_000,
                tokens_per_doc: 20,
                tokens_per_query: 8,
                orig_dim: 64,
                num_reps: 16,
                n_queries: 50,
                k: 10,
                label: "medium (1K docs, 20 tok, 64D, 16 reps) [fast]",
            },
        ]
    } else {
        vec![
            Config {
                n_docs: 500,
                tokens_per_doc: 16,
                tokens_per_query: 8,
                orig_dim: 32,
                num_reps: 8,
                n_queries: 100,
                k: 10,
                label: "XS (500 docs, 16 tok, 32D, 8 reps)",
            },
            Config {
                n_docs: 2_000,
                tokens_per_doc: 20,
                tokens_per_query: 8,
                orig_dim: 64,
                num_reps: 16,
                n_queries: 200,
                k: 10,
                label: "S (2K docs, 20 tok, 64D, 16 reps)",
            },
            Config {
                n_docs: 5_000,
                tokens_per_doc: 32,
                tokens_per_query: 16,
                orig_dim: 64,
                num_reps: 32,
                n_queries: 200,
                k: 10,
                label: "M (5K docs, 32 tok, 64D, 32 reps)",
            },
            Config {
                n_docs: 10_000,
                tokens_per_doc: 32,
                tokens_per_query: 16,
                orig_dim: 128,
                num_reps: 64,
                n_queries: 200,
                k: 10,
                label: "L (10K docs, 32 tok, 128D, 64 reps)",
            },
        ]
    };

    let mut all_rows: Vec<Row> = Vec::new();
    for cfg in &configs {
        all_rows.extend(run_config(cfg));
    }

    // Summary table
    println!("\n{:-<90}", "");
    println!(
        "{:<30} {:>8} {:>10} {:>10} {:>12} {:>10}",
        "Variant", "n_docs", "Build(ms)", "QPS", "Recall@10", "Mem(KB)"
    );
    println!("{:-<90}", "");
    for r in &all_rows {
        println!(
            "{:<30} {:>8} {:>10.1} {:>10.0} {:>10.3} {:>10.1}",
            r.variant,
            r.n_docs,
            r.build_ms,
            r.qps,
            r.recall,
            r.mem_bytes as f64 / 1024.0
        );
    }
    println!("{:-<90}", "");
    println!("\nKey insight: HnswFDE QPS speedup vs BruteForce at n=10K:");
    if let (Some(bf), Some(hnsw)) = (
        all_rows.iter().find(|r| r.variant == "BruteForceMaxSim" && r.n_docs == 10_000),
        all_rows.iter().find(|r| r.variant == "HnswFDE" && r.n_docs == 10_000),
    ) {
        println!(
            "  BruteForce QPS: {:.0}  HnswFDE QPS: {:.0}  Speedup: {:.1}x  Recall: {:.3}",
            bf.qps,
            hnsw.qps,
            hnsw.qps / bf.qps,
            hnsw.recall
        );
    } else {
        // fast mode only goes to 1K
        if let (Some(bf), Some(hnsw)) = (
            all_rows.iter().find(|r| r.variant == "BruteForceMaxSim"),
            all_rows.iter().find(|r| r.variant == "HnswFDE"),
        ) {
            println!(
                "  BruteForce QPS: {:.0}  HnswFDE QPS: {:.0}  Speedup: {:.1}x  Recall: {:.3}",
                bf.qps,
                hnsw.qps,
                hnsw.qps / bf.qps,
                hnsw.recall
            );
        }
    }
}

fn hardware_string() -> String {
    // Best-effort hardware description from /proc/cpuinfo.
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("unknown").trim().to_string())
        })
        .unwrap_or_else(|| "unknown CPU".to_string())
}
