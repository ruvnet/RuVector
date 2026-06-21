//! Benchmark binary: compare ScoreFusion, RRF, and RSF on a synthetic corpus.
//!
//! Synthetic design
//! ─────────────────
//! N_TOPICS topics, each with DOCS_PER_TOPIC documents.
//! Each document: vector = topic_centre + Uniform(−0.15, 0.15) noise (128-D),
//!                tokens = TOKENS_PER_DOC words drawn from topic vocabulary.
//! Each query targets one topic:
//!                vector = near topic_centre + smaller noise,
//!                tokens = QUERY_TOKENS words from that topic's vocabulary.
//!
//! Ground truth
//! ─────────────
//! For each query, brute-force combined score:
//!   combined(d) = 0.5 · cosine_norm(d) + 0.5 · bm25_norm(d)
//! where cosine_norm ∈ [0,1] = (cosine − min) / (max − min)
//! and   bm25_norm   ∈ [0,1] = bm25_score / max_bm25 (or 0 if no BM25 match).
//! Top-K by combined = ground truth for that query.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ruvector_hybrid::{
    recall_at_k, Bm25Index, DenseSearch, Document, FlatDenseIndex, HybridSearch, RrfHybridIndex,
    RsfHybridIndex, ScoreFusionIndex, SearchResult, SparseSearch,
};

// ── Dataset parameters ────────────────────────────────────────────────────────
const N_TOPICS: usize = 20;
const DOCS_PER_TOPIC: usize = 500;
const N_DOCS: usize = N_TOPICS * DOCS_PER_TOPIC;
const DIM: usize = 128;
const VOCAB_PER_TOPIC: usize = 25;
const TOKENS_PER_DOC: usize = 6;
const QUERY_TOKENS: usize = 3;
const N_QUERIES: usize = 500;
const K: usize = 10;
const SEED: u64 = 42;

// ── Ground-truth helpers ──────────────────────────────────────────────────────

fn cosine_score(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn compute_ground_truth(
    docs: &[Document],
    bm25: &Bm25Index,
    q_tokens: &[&str],
    q_vec: &[f32],
    k: usize,
) -> Vec<usize> {
    // BM25 scores (fetch full corpus to get max)
    let bm25_all = bm25.search(q_tokens, N_DOCS);
    let bm25_max = bm25_all.first().map(|r| r.score).unwrap_or(1.0).max(1e-10);
    let bm25_map: std::collections::HashMap<usize, f32> = bm25_all
        .iter()
        .map(|r| (r.id, r.score / bm25_max))
        .collect();

    // Cosine scores for all docs
    let cosines: Vec<f32> = docs
        .iter()
        .map(|d| cosine_score(q_vec, &d.vector))
        .collect();
    let cos_min = cosines.iter().cloned().fold(f32::INFINITY, f32::min);
    let cos_max = cosines.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let cos_range = (cos_max - cos_min).max(1e-10);

    let mut combined: Vec<(usize, f32)> = docs
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let c_norm = (cosines[i] - cos_min) / cos_range;
            let b_norm = bm25_map.get(&i).cloned().unwrap_or(0.0);
            (i, 0.5 * c_norm + 0.5 * b_norm)
        })
        .collect();

    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    combined.into_iter().take(k).map(|(id, _)| id).collect()
}

// ── Dataset generation ────────────────────────────────────────────────────────

fn generate_corpus(rng: &mut StdRng) -> Vec<Document> {
    // Topic centres: random unit vectors
    let centres: Vec<Vec<f32>> = (0..N_TOPICS)
        .map(|_| {
            let v: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect();

    let mut docs = Vec::with_capacity(N_DOCS);
    for (t, centre) in centres.iter().enumerate() {
        for d in 0..DOCS_PER_TOPIC {
            let id = t * DOCS_PER_TOPIC + d;
            let vector: Vec<f32> = centre
                .iter()
                .map(|&c| c + rng.gen::<f32>() * 0.30 - 0.15)
                .collect();
            let tokens: Vec<String> = (0..TOKENS_PER_DOC)
                .map(|_| format!("t{}w{}", t, rng.gen_range(0..VOCAB_PER_TOPIC)))
                .collect();
            docs.push(Document { id, tokens, vector });
        }
    }
    docs
}

struct Query {
    tokens: Vec<String>,
    vector: Vec<f32>,
    ground_truth: Vec<usize>,
}

fn generate_queries(docs: &[Document], bm25: &Bm25Index, rng: &mut StdRng) -> Vec<Query> {
    (0..N_QUERIES)
        .map(|_| {
            let topic = rng.gen_range(0..N_TOPICS);
            // Query vector ≈ mean of a few same-topic docs + tiny noise
            let anchor_idx = topic * DOCS_PER_TOPIC + rng.gen_range(0..DOCS_PER_TOPIC / 5);
            let anchor = &docs[anchor_idx].vector;
            let vector: Vec<f32> = anchor
                .iter()
                .map(|&v| v + rng.gen::<f32>() * 0.10 - 0.05)
                .collect();
            let tokens: Vec<String> = (0..QUERY_TOKENS)
                .map(|_| format!("t{}w{}", topic, rng.gen_range(0..VOCAB_PER_TOPIC)))
                .collect();
            let token_refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
            let ground_truth = compute_ground_truth(docs, bm25, &token_refs, &vector, K);
            Query {
                tokens,
                vector,
                ground_truth,
            }
        })
        .collect()
}

// ── Stats helpers ─────────────────────────────────────────────────────────────

fn percentile(sorted: &[u128], p: f64) -> u128 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx = ((sorted.len() as f64 * p / 100.0).ceil() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn mean_us(durations: &[u128]) -> f64 {
    durations.iter().sum::<u128>() as f64 / durations.len() as f64 / 1_000.0
}

// ── Benchmark runner ──────────────────────────────────────────────────────────

fn run_dense(idx: &FlatDenseIndex, queries: &[Query]) -> (Vec<SearchResult>, Vec<u128>, Vec<f32>) {
    let mut recalls = Vec::with_capacity(queries.len());
    let mut latencies = Vec::with_capacity(queries.len());
    let mut last_results = Vec::new();
    for q in queries {
        let start = Instant::now();
        let results = idx.search(&q.vector, K);
        latencies.push(start.elapsed().as_nanos());
        recalls.push(recall_at_k(&results, &q.ground_truth));
        last_results = results;
    }
    (last_results, latencies, recalls)
}

fn run_sparse(idx: &Bm25Index, queries: &[Query]) -> (Vec<SearchResult>, Vec<u128>, Vec<f32>) {
    let mut recalls = Vec::with_capacity(queries.len());
    let mut latencies = Vec::with_capacity(queries.len());
    let mut last_results = Vec::new();
    for q in queries {
        let token_refs: Vec<&str> = q.tokens.iter().map(String::as_str).collect();
        let start = Instant::now();
        let results = idx.search(&token_refs, K);
        latencies.push(start.elapsed().as_nanos());
        recalls.push(recall_at_k(&results, &q.ground_truth));
        last_results = results;
    }
    (last_results, latencies, recalls)
}

fn run_hybrid<H: HybridSearch>(
    idx: &H,
    queries: &[Query],
) -> (Vec<SearchResult>, Vec<u128>, Vec<f32>) {
    let mut recalls = Vec::with_capacity(queries.len());
    let mut latencies = Vec::with_capacity(queries.len());
    let mut last_results = Vec::new();
    for q in queries {
        let token_refs: Vec<&str> = q.tokens.iter().map(String::as_str).collect();
        let start = Instant::now();
        let results = idx.search(&token_refs, &q.vector, K);
        latencies.push(start.elapsed().as_nanos());
        recalls.push(recall_at_k(&results, &q.ground_truth));
        last_results = results;
    }
    (last_results, latencies, recalls)
}

fn print_row(name: &str, recalls: &[f32], latencies_ns: &mut [u128], mem_kb: usize) {
    let recall_mean = recalls.iter().sum::<f32>() / recalls.len() as f32;
    latencies_ns.sort_unstable();
    let mean = mean_us(latencies_ns);
    let p50 = percentile(latencies_ns, 50.0) as f64 / 1_000.0;
    let p95 = percentile(latencies_ns, 95.0) as f64 / 1_000.0;
    let qps = 1_000_000.0 / mean;
    println!(
        "{:<16} | {:>8.1}% | {:>9.1}μs | {:>8.1}μs | {:>8.1}μs | {:>8.0} | {:>7} KB",
        name,
        recall_mean * 100.0,
        mean,
        p50,
        p95,
        qps as u64,
        mem_kb,
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("Nightly RuVector Research — Hybrid Sparse-Dense Search");
    println!("=======================================================");
    println!("Crate  : ruvector-hybrid");
    println!("Date   : 2026-06-17");
    println!();
    println!("Dataset");
    println!("  Docs      : {N_DOCS}");
    println!("  Dimensions: {DIM}");
    println!("  Topics    : {N_TOPICS}");
    println!("  Vocab size: {}", N_TOPICS * VOCAB_PER_TOPIC);
    println!("  Queries   : {N_QUERIES}");
    println!("  k         : {K}");
    println!();

    let mut rng = StdRng::seed_from_u64(SEED);

    print!("Generating corpus ({N_DOCS} docs × {DIM}D)... ");
    let t0 = Instant::now();
    let docs = generate_corpus(&mut rng);
    println!("{:.1}ms", t0.elapsed().as_millis());

    print!("Building BM25 index... ");
    let t1 = Instant::now();
    let bm25_idx = Bm25Index::build(&docs);
    println!("{:.1}ms", t1.elapsed().as_millis());

    print!("Building dense index... ");
    let t2 = Instant::now();
    let dense_idx = FlatDenseIndex::build(&docs);
    println!("{:.1}ms", t2.elapsed().as_millis());

    print!("Building hybrid indices (ScoreFusion, RRF, RSF)... ");
    let t3 = Instant::now();
    let sf_idx = ScoreFusionIndex::build(&docs);
    let rrf_idx = RrfHybridIndex::build(&docs);
    let rsf_idx = RsfHybridIndex::build(&docs);
    println!("{:.1}ms", t3.elapsed().as_millis());

    print!("Computing combined ground truth for {N_QUERIES} queries... ");
    let t4 = Instant::now();
    let queries = generate_queries(&docs, &bm25_idx, &mut rng);
    println!("{:.1}ms", t4.elapsed().as_millis());

    // ── Memory estimates ──
    let bm25_mem_kb = bm25_idx.posting_bytes() / 1024;
    let dense_mem_kb = dense_idx.byte_size() / 1024;
    let rrf_mem_kb = bm25_mem_kb + dense_mem_kb; // stores both

    println!();
    println!("Memory Estimates");
    println!("  BM25 postings : {} KB", bm25_mem_kb);
    println!(
        "  Dense vectors : {} KB ({} × {} × 4B)",
        dense_mem_kb, N_DOCS, DIM
    );
    println!("  Hybrid indices: {} KB each (BM25 + dense)", rrf_mem_kb);

    println!();
    println!("Benchmark Results");
    println!("{:-<90}", "");
    println!(
        "{:<16} | {:>9} | {:>10} | {:>9} | {:>9} | {:>8} | {:>8}",
        "Variant", "Recall@10", "Mean lat", "p50 lat", "p95 lat", "QPS", "Memory"
    );
    println!("{:-<90}", "");

    let (_, mut dense_lat, dense_rec) = run_dense(&dense_idx, &queries);
    print_row("Dense (exact)", &dense_rec, &mut dense_lat, dense_mem_kb);

    let (_, mut sparse_lat, sparse_rec) = run_sparse(&bm25_idx, &queries);
    print_row("BM25 (sparse)", &sparse_rec, &mut sparse_lat, bm25_mem_kb);

    let (_, mut sf_lat, sf_rec) = run_hybrid(&sf_idx, &queries);
    print_row("ScoreFusion α=0.7", &sf_rec, &mut sf_lat, rrf_mem_kb);

    let (_, mut rrf_lat, rrf_rec) = run_hybrid(&rrf_idx, &queries);
    print_row("RRF k=60", &rrf_rec, &mut rrf_lat, rrf_mem_kb);

    let (_, mut rsf_lat, rsf_rec) = run_hybrid(&rsf_idx, &queries);
    print_row("RSF α=0.5", &rsf_rec, &mut rsf_lat, rrf_mem_kb);

    println!("{:-<90}", "");

    // ── Acceptance tests ──
    let dense_recall = dense_rec.iter().sum::<f32>() / dense_rec.len() as f32;
    let sparse_recall = sparse_rec.iter().sum::<f32>() / sparse_rec.len() as f32;
    let rrf_recall = rrf_rec.iter().sum::<f32>() / rrf_rec.len() as f32;
    let rsf_recall = rsf_rec.iter().sum::<f32>() / rsf_rec.len() as f32;
    let sf_recall = sf_rec.iter().sum::<f32>() / sf_rec.len() as f32;

    println!();
    println!("Acceptance Tests");

    let mut all_pass = true;

    macro_rules! check {
        ($cond:expr, $msg:expr) => {{
            let pass = $cond;
            println!("  {} ... {}", $msg, if pass { "PASS" } else { "FAIL" });
            if !pass {
                all_pass = false;
            }
        }};
    }

    // On a 50/50 combined GT with topic-isolated vocabulary, BM25 dominates because
    // within-topic cosine scores are nearly uniform (all same-topic docs cluster),
    // while BM25 varies significantly on keyword overlap. This is a known property
    // of keyword-biased ground truth — see research document for full discussion.

    // BM25 captures keyword-biased GT well (expected ≥ 70%)
    check!(
        sparse_recall >= 0.70,
        format!("BM25 recall@10 ≥ 70% (got {:.1}%)", sparse_recall * 100.0)
    );
    // All hybrid variants beat dense alone (any keyword signal helps)
    check!(
        rrf_recall > dense_recall,
        format!(
            "RRF recall > dense recall ({:.1}% > {:.1}%)",
            rrf_recall * 100.0,
            dense_recall * 100.0
        )
    );
    check!(
        rsf_recall > dense_recall,
        format!(
            "RSF recall > dense recall ({:.1}% > {:.1}%)",
            rsf_recall * 100.0,
            dense_recall * 100.0
        )
    );
    check!(
        sf_recall > dense_recall,
        format!(
            "ScoreFusion recall > dense recall ({:.1}% > {:.1}%)",
            sf_recall * 100.0,
            dense_recall * 100.0
        )
    );
    // RSF with equal weighting (α=0.5) recovers near-BM25 performance on keyword GT
    check!(
        rsf_recall >= 0.65,
        format!("RSF recall@10 ≥ 65% (got {:.1}%)", rsf_recall * 100.0)
    );
    // RRF provides a robust minimum baseline (rank fusion, score-agnostic)
    check!(
        rrf_recall >= 0.40,
        format!("RRF recall@10 ≥ 40% (got {:.1}%)", rrf_recall * 100.0)
    );
    // Sanity: no negative recalls
    check!(
        sf_recall >= 0.0 && rrf_recall >= 0.0 && rsf_recall >= 0.0,
        "All recalls are non-negative"
    );

    // Key insight: RSF (Weaviate-style) with α=0.5 matches BM25 on keyword-heavy GT.
    // RRF (Qdrant-style, fixed k=60) is more conservative — better when GT is balanced.
    let rsf_gap = (sparse_recall - rsf_recall).abs();
    println!(
        "\n  Insight: RSF gap vs BM25 = {:.1}pp (smaller = RSF better matches BM25 quality)",
        rsf_gap * 100.0
    );
    let rrf_gap = (sparse_recall - rrf_recall).abs();
    println!(
        "  Insight: RRF gap vs BM25 = {:.1}pp (larger gap = RRF is more conservative/balanced)",
        rrf_gap * 100.0
    );

    println!();
    if all_pass {
        println!("All acceptance tests PASSED.");
    } else {
        println!("Some acceptance tests FAILED — see details above.");
        std::process::exit(1);
    }
}
