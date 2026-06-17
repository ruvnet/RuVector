//! Hybrid sparse-dense benchmark binary.
//!
//! Usage:
//!   cargo run --release -p ruvector-hybrid --bin hybrid-benchmark
//!   cargo run --release -p ruvector-hybrid --bin hybrid-benchmark -- --n-docs 50000
//!
//! Two dataset regimes are reported:
//! - Random (uncorrelated): tokens are independent of vectors. Honest
//!   baseline showing that sparse noise degrades dense recall.
//! - Structured (cluster-aligned): tokens correlate with vector clusters.
//!   Shows RRF recall gain when lexical and semantic signals align.

use std::time::Instant;

use ruvector_hybrid::{
    dataset::{generate, generate_structured, DatasetConfig},
    recall_at_k, DenseFlat, HybridRrf, HybridSearch, SparseBm25,
};

fn percentile(mut v: Vec<u128>, p: f64) -> u128 {
    v.sort_unstable();
    let idx = ((v.len() as f64 * p / 100.0) as usize).min(v.len().saturating_sub(1));
    v[idx]
}

fn bytes_to_mb(b: usize) -> f64 {
    b as f64 / 1_048_576.0
}

struct BenchResult {
    variant: &'static str,
    build_ms: u128,
    mean_us: f64,
    p50_us: u128,
    p95_us: u128,
    qps: f64,
    memory_mb: f64,
    recall: f64,
}

fn run_variant<I, F>(
    variant: &'static str,
    index: &mut I,
    gen: F,
    cfg: &DatasetConfig,
) -> BenchResult
where
    I: HybridSearch,
    F: Fn(&DatasetConfig) -> (Vec<ruvector_hybrid::Document>, Vec<ruvector_hybrid::Query>),
{
    let (docs, queries) = gen(cfg);

    let t0 = Instant::now();
    for d in &docs {
        index.insert(d.id, &d.vector, &d.tokens);
    }
    let build_ms = t0.elapsed().as_millis();

    let k = cfg.k;
    let mut latencies_us: Vec<u128> = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f64;

    let t_all = Instant::now();
    for q in &queries {
        let tq = Instant::now();
        let results = index.search(&q.vector, &q.tokens, k);
        latencies_us.push(tq.elapsed().as_micros());
        total_recall += recall_at_k(&results, &q.ground_truth) as f64;
    }
    let total_us = t_all.elapsed().as_micros() as f64;

    let n = queries.len();
    BenchResult {
        variant,
        build_ms,
        mean_us: latencies_us.iter().copied().sum::<u128>() as f64 / n as f64,
        p50_us: percentile(latencies_us.clone(), 50.0),
        p95_us: percentile(latencies_us, 95.0),
        qps: n as f64 / (total_us / 1_000_000.0),
        memory_mb: bytes_to_mb(index.memory_bytes()),
        recall: total_recall / n as f64,
    }
}

fn print_table(results: &[BenchResult]) {
    println!(
        "  {:<16}   {:>10}   {:>12}   {:>10}   {:>10}   {:>10}   {:>10}   {:>12}",
        "Variant", "Build(ms)", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Mem(MB)", "Recall@10"
    );
    println!("  {}", "-".repeat(108));
    for r in results {
        println!(
            "  {:<16}   {:>10}   {:>12.1}   {:>10}   {:>10}   {:>10.0}   {:>10.2}   {:>11.1}%",
            r.variant,
            r.build_ms,
            r.mean_us,
            r.p50_us,
            r.p95_us,
            r.qps,
            r.memory_mb,
            r.recall * 100.0,
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut n_docs = 10_000usize;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--n-docs" && i + 1 < args.len() {
            n_docs = args[i + 1].parse().unwrap_or(n_docs);
            i += 2;
        } else {
            i += 1;
        }
    }

    println!("=== ruvector-hybrid benchmark ===");
    println!("OS:       {}", std::env::consts::OS);
    println!("Arch:     {}", std::env::consts::ARCH);
    println!();

    // ── Section 1: Random (uncorrelated) dataset ─────────────────────────────
    let rand_cfg = DatasetConfig {
        n_docs,
        dim: 128,
        vocab_size: 1_000,
        tokens_per_doc: 20,
        n_queries: 200,
        seed: 42,
        k: 10,
    };
    println!(
        "── Section 1: Random (uncorrelated tokens) — {} docs, dim={}, vocab={}, queries={}",
        rand_cfg.n_docs, rand_cfg.dim, rand_cfg.vocab_size, rand_cfg.n_queries
    );
    println!("   Tokens are independent of vectors; sparse signal is noise w.r.t. dense GT.");

    let rand_results: Vec<BenchResult> = vec![
        {
            let mut idx = DenseFlat::new(rand_cfg.dim);
            run_variant("DenseFlat", &mut idx, generate, &rand_cfg)
        },
        {
            let mut idx = SparseBm25::new();
            run_variant("SparseBm25", &mut idx, generate, &rand_cfg)
        },
        {
            let mut idx = HybridRrf::new(rand_cfg.dim);
            run_variant("HybridRrf", &mut idx, generate, &rand_cfg)
        },
    ];
    print_table(&rand_results);
    println!();

    // ── Section 2: Structured (topic-model) dataset ──────────────────────────
    // Cap at 5 000 docs: with 32 topics there are C(32,3)=4 960 combinations.
    // Above ~5 000 docs, multiple docs share the same 3-topic set → equal BM25
    // scores within the relevant group → random sparse ranks → recall drops.
    // Section 1 already covers throughput; Section 2 validates recall.
    let struct_n_docs = n_docs.min(5_000);
    let struct_cfg = DatasetConfig {
        n_docs: struct_n_docs,
        dim: 128,
        vocab_size: 32, // 32 latent topics; token IDs = topic IDs
        tokens_per_doc: 3,
        n_queries: 200,
        seed: 42,
        k: 10,
    };
    println!(
        "── Section 2: Structured (topic model) — {} docs, dim={}, 32 topics, queries={}",
        struct_cfg.n_docs, struct_cfg.dim, struct_cfg.n_queries
    );
    println!("   Tokens = topic IDs; cosine and BM25 agree on relevant docs.");
    println!("   (capped at 5 000 docs: above that, C(32,3)=4 960 topic combos create BM25 ties)");

    let struct_results: Vec<BenchResult> = vec![
        {
            let mut idx = DenseFlat::new(struct_cfg.dim);
            run_variant("DenseFlat", &mut idx, generate_structured, &struct_cfg)
        },
        {
            let mut idx = SparseBm25::new();
            run_variant("SparseBm25", &mut idx, generate_structured, &struct_cfg)
        },
        {
            let mut idx = HybridRrf::new(struct_cfg.dim);
            run_variant("HybridRrf", &mut idx, generate_structured, &struct_cfg)
        },
    ];
    print_table(&struct_results);
    println!();

    // ── Acceptance checks ─────────────────────────────────────────────────────
    let dense_rand = rand_results
        .iter()
        .find(|r| r.variant == "DenseFlat")
        .unwrap();
    let sparse_rand = rand_results
        .iter()
        .find(|r| r.variant == "SparseBm25")
        .unwrap();
    let dense_struct = struct_results
        .iter()
        .find(|r| r.variant == "DenseFlat")
        .unwrap();
    let hybrid_struct = struct_results
        .iter()
        .find(|r| r.variant == "HybridRrf")
        .unwrap();

    let mut pass = true;

    let dense_rand_ok = dense_rand.recall >= 0.999;
    if !dense_rand_ok {
        eprintln!(
            "FAIL: DenseFlat (random) recall={:.1}% expected ≥ 99.9%",
            dense_rand.recall * 100.0
        );
        pass = false;
    }

    let dense_struct_ok = dense_struct.recall >= 0.999;
    if !dense_struct_ok {
        eprintln!(
            "FAIL: DenseFlat (structured) recall={:.1}% expected ≥ 99.9%",
            dense_struct.recall * 100.0
        );
        pass = false;
    }

    // 60% is well above SparseBm25-alone (~33%) on topic-model data.
    // The remaining gap to 100% is explained by equal BM25 scores within
    // same-topic document groups — see research doc for details.
    let hybrid_struct_ok = hybrid_struct.recall >= 0.60;
    if !hybrid_struct_ok {
        eprintln!(
            "FAIL: HybridRrf (structured) recall={:.1}% expected ≥ 60%",
            hybrid_struct.recall * 100.0
        );
        pass = false;
    }

    let sparse_qps_ok = sparse_rand.qps >= 500.0;
    if !sparse_qps_ok {
        eprintln!("FAIL: SparseBm25 QPS={:.0} expected ≥ 500", sparse_rand.qps);
        pass = false;
    }

    println!("Acceptance checks:");
    println!(
        "  DenseFlat random recall ≥ 99.9%:      {}",
        if dense_rand_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "  DenseFlat structured recall ≥ 99.9%:  {}",
        if dense_struct_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "  HybridRrf structured recall ≥ 60%:    {}",
        if hybrid_struct_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "  SparseBm25 QPS ≥ 500:                 {}",
        if sparse_qps_ok { "PASS" } else { "FAIL" }
    );
    println!();

    if pass {
        println!("RESULT: PASS — all acceptance checks satisfied.");
    } else {
        eprintln!("RESULT: FAIL — one or more acceptance checks failed.");
        std::process::exit(1);
    }
}
