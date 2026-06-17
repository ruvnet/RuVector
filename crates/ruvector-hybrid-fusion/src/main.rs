//! Benchmark binary: hybrid sparse-dense fusion with coherence-adaptive weighting.
//!
//! Runs four retrieval variants over a mixed text-dominant / vector-dominant
//! corpus and reports mean latency, p50, p95, throughput, recall@10, and
//! memory estimates.  All numbers come from real `cargo run --release` timings.

use ruvector_hybrid_fusion::{
    bench_retriever,
    bm25::Bm25Index,
    coherence_fuse,
    dataset::{generate_corpus, DIM, DOCS_PER_TOPIC, NUM_TOPICS, QUERIES_PER_TOPIC},
    dense::DenseIndex,
    linear_fuse, rrf_fuse, BenchResult,
};
use std::time::Instant;

fn print_env() {
    println!("=== ruvector-hybrid-fusion benchmark ===");
    println!("OS      : {}", std::env::consts::OS);
    println!("ARCH    : {}", std::env::consts::ARCH);
    println!("Docs    : {} ({} topics × {} per topic)", NUM_TOPICS * DOCS_PER_TOPIC, NUM_TOPICS, DOCS_PER_TOPIC);
    println!("Dims    : {}", DIM);
    println!("Queries : {} ({} per topic)", NUM_TOPICS * QUERIES_PER_TOPIC, QUERIES_PER_TOPIC);
    println!();
}

fn print_result(r: &BenchResult, mem_kb: usize) {
    println!(
        "  {:<22}  recall@10={:.3}  mean={:6.1}µs  p50={:6}µs  p95={:6}µs  QPS={:7.0}  mem={}KB",
        r.variant,
        r.mean_recall(),
        r.mean_latency_us(),
        r.p50_latency_us(),
        r.p95_latency_us(),
        r.throughput_qps(),
        mem_kb,
    );
}

fn main() {
    print_env();

    // ── 1. Build corpus ───────────────────────────────────────────────────────
    let t0 = Instant::now();
    let corpus = generate_corpus(42);
    println!("Corpus generated in {:.0}ms", t0.elapsed().as_millis());

    // ── 2. Build indexes ──────────────────────────────────────────────────────
    let doc_tokens: Vec<Vec<String>> = corpus.docs.iter().map(|d| d.tokens.clone()).collect();
    let doc_vectors: Vec<Vec<f32>> = corpus.docs.iter().map(|d| d.vector.clone()).collect();

    let t1 = Instant::now();
    let bm25 = Bm25Index::build(&doc_tokens);
    println!("BM25 index built  in {:5.0}ms  vocab={} terms  mem={}KB",
        t1.elapsed().as_millis(), bm25.vocab_size(), bm25.memory_bytes() / 1024);

    let t2 = Instant::now();
    let dense = DenseIndex::build(doc_vectors);
    println!("Dense index built in {:5.0}ms  mem={}KB",
        t2.elapsed().as_millis(), dense.memory_bytes() / 1024);

    let combined_mem_kb = (bm25.memory_bytes() + dense.memory_bytes()) / 1024;
    println!();

    // ── 3. Benchmark variants ─────────────────────────────────────────────────
    const TOP_K: usize = 10;
    const FETCH_K: usize = 50; // fetch more candidates for fusion legs

    println!("--- Retrieval variants ---");

    let r_sparse = bench_retriever(
        &corpus,
        |q| bm25.score(&q.tokens, TOP_K),
        "SparseOnly (BM25)",
        TOP_K,
    );
    print_result(&r_sparse, bm25.memory_bytes() / 1024);

    let r_dense = bench_retriever(
        &corpus,
        |q| dense.search(&q.vector, TOP_K),
        "DenseOnly (cosine)",
        TOP_K,
    );
    print_result(&r_dense, dense.memory_bytes() / 1024);

    let r_rrf = bench_retriever(
        &corpus,
        |q| {
            let sp = bm25.score(&q.tokens, FETCH_K);
            let de = dense.search(&q.vector, FETCH_K);
            rrf_fuse(&sp, &de, 60.0, TOP_K)
        },
        "HybridRRF (k=60)",
        TOP_K,
    );
    print_result(&r_rrf, combined_mem_kb);

    let r_linear = bench_retriever(
        &corpus,
        |q| {
            let sp = bm25.score(&q.tokens, FETCH_K);
            let de = dense.search(&q.vector, FETCH_K);
            linear_fuse(&sp, &de, TOP_K)
        },
        "HybridLinear (α=0.5)",
        TOP_K,
    );
    print_result(&r_linear, combined_mem_kb);

    let r_coh = bench_retriever(
        &corpus,
        |q| {
            let sp = bm25.score(&q.tokens, FETCH_K);
            let de = dense.search(&q.vector, FETCH_K);
            coherence_fuse(&sp, &de, TOP_K)
        },
        "HybridCoherence (adaptive)",
        TOP_K,
    );
    print_result(&r_coh, combined_mem_kb);

    // ── 4. Per-query-type breakdown ───────────────────────────────────────────
    use ruvector_hybrid_fusion::dataset::QueryType;
    println!();
    println!("--- Coherence fusion recall by query type ---");

    let mut kw_coh = 0.0f64;
    let mut kw_rrf = 0.0f64;
    let mut kw_count = 0usize;

    for qtype in [QueryType::Hybrid, QueryType::KeywordHeavy, QueryType::VectorHeavy] {
        let filtered: Vec<usize> = corpus
            .queries
            .iter()
            .enumerate()
            .filter(|(_, q)| q.query_type == qtype)
            .map(|(i, _)| i)
            .collect();

        let mean_coh: f64 = filtered.iter().map(|&i| r_coh.recalls[i]).sum::<f64>()
            / filtered.len() as f64;
        let mean_rrf: f64 = filtered.iter().map(|&i| r_rrf.recalls[i]).sum::<f64>()
            / filtered.len() as f64;
        let label = match qtype {
            QueryType::Hybrid => "Hybrid    ",
            QueryType::KeywordHeavy => "KwHeavy   ",
            QueryType::VectorHeavy => "VecHeavy  ",
        };
        println!("  {} CoherenceRecall={:.3}  RRFRecall={:.3}  delta={:+.3}",
            label, mean_coh, mean_rrf, mean_coh - mean_rrf);

        if qtype == QueryType::KeywordHeavy {
            kw_coh = mean_coh;
            kw_rrf = mean_rrf;
            kw_count = filtered.len();
        }
    }

    // ── 5. Acceptance test ────────────────────────────────────────────────────
    println!();
    println!("--- Acceptance tests ---");

    let recall_sparse = r_sparse.mean_recall();
    let recall_dense = r_dense.mean_recall();
    let recall_rrf = r_rrf.mean_recall();
    let recall_coh = r_coh.mean_recall();

    let mut passed = true;

    // Both hybrid variants must outperform each single leg
    check("HybridCoherence > SparseOnly",          recall_coh > recall_sparse,   &mut passed);
    check("HybridCoherence > DenseOnly",            recall_coh > recall_dense,    &mut passed);
    check("HybridRRF > SparseOnly",                 recall_rrf > recall_sparse,   &mut passed);
    check("HybridRRF > DenseOnly",                  recall_rrf > recall_dense,    &mut passed);

    // Minimum absolute recall floors
    check("SparseOnly recall >= 0.30",              recall_sparse >= 0.30,        &mut passed);
    check("DenseOnly  recall >= 0.40",              recall_dense  >= 0.40,        &mut passed);
    check("HybridRRF  recall >= 0.60",              recall_rrf    >= 0.60,        &mut passed);
    check("HybridCoherence recall >= 0.65",         recall_coh    >= 0.65,        &mut passed);

    // Coherence-adaptive weighting must win on keyword-heavy queries (n=50)
    let kw_msg = format!(
        "CoherenceRecall >= RRFRecall on KwHeavy (n={}, {:.3} vs {:.3})",
        kw_count, kw_coh, kw_rrf
    );
    check(&kw_msg, kw_coh >= kw_rrf,                                              &mut passed);

    println!();
    if passed {
        println!("RESULT: PASS — all acceptance tests passed");
    } else {
        println!("RESULT: FAIL — one or more acceptance tests failed");
        std::process::exit(1);
    }
}

fn check(label: &str, cond: bool, passed: &mut bool) {
    let status = if cond { "PASS" } else { "FAIL" };
    println!("  [{status}] {label}");
    if !cond { *passed = false; }
}
