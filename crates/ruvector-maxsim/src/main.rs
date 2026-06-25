//! MaxSim demo + benchmark binary.
//!
//! Generates a synthetic multi-facet corpus, runs all three index variants,
//! reports latency, throughput, recall, and memory, and asserts acceptance.
//!
//! Usage:
//!   cargo run --release -p ruvector-maxsim
//!   cargo run --release -p ruvector-maxsim -- --docs 10000 --dims 64 --queries 500

use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_maxsim::{
    types::{DocId, MultiVecDoc, MultiVecQuery, RunStats},
    BucketMaxSim, FlatMaxSim, HnswMaxSim, MultiVecIndex,
};

// ── CLI config ──────────────────────────────────────────────────────────────

struct Config {
    n_docs: usize,
    dims: usize,
    n_queries: usize,
    tokens_per_doc: usize,
    tokens_per_query: usize,
    n_topics: usize,
    k: usize,
}

impl Config {
    fn from_env() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let get = |flag: &str, default: usize| -> usize {
            args.windows(2)
                .find(|w| w[0] == flag)
                .and_then(|w| w[1].parse().ok())
                .unwrap_or(default)
        };
        Self {
            n_docs: get("--docs", 5_000),
            dims: get("--dims", 64),
            n_queries: get("--queries", 200),
            tokens_per_doc: 6,
            tokens_per_query: 3,
            n_topics: 32,
            k: 10,
        }
    }
}

// ── Synthetic data generation ────────────────────────────────────────────────

/// Generate a random unit vector from a Gaussian cloud centred at `topic`.
fn sample_from_topic(rng: &mut impl Rng, topic: &[f32], noise: f32) -> Vec<f32> {
    let normal = Normal::new(0.0_f32, noise).unwrap();
    let mut v: Vec<f32> = topic.iter().map(|&t| t + normal.sample(rng)).collect();
    l2_norm(&mut v);
    v
}

fn l2_norm(v: &mut [f32]) {
    let len = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if len > 1e-9 {
        for x in v.iter_mut() {
            *x /= len;
        }
    }
}

/// Generate D-dimensional unit topic centroids (deterministic seed).
fn gen_topics(n_topics: usize, dims: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0xBEEF_CAFE);
    (0..n_topics)
        .map(|_| {
            let mut v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            l2_norm(&mut v);
            v
        })
        .collect()
}

/// Build synthetic corpus: each doc has `tpd` token vectors sampled from 1–3 topics.
fn gen_corpus(
    n_docs: usize,
    tpd: usize,
    topics: &[Vec<f32>],
    _dims: usize,
    noise: f32,
) -> (Vec<MultiVecDoc>, Vec<Vec<usize>>) {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0xDEAD_BEEF);
    let mut docs = Vec::with_capacity(n_docs);
    let mut doc_topics = Vec::with_capacity(n_docs);
    let n_topics = topics.len();
    for i in 0..n_docs {
        // Randomly pick 1-3 topics for this document.
        let nt = rng.gen_range(1..=3_usize).min(n_topics);
        let mut chosen: Vec<usize> = Vec::with_capacity(nt);
        while chosen.len() < nt {
            let t = rng.gen_range(0..n_topics);
            if !chosen.contains(&t) {
                chosen.push(t);
            }
        }
        let mut vecs = Vec::with_capacity(tpd);
        for _ in 0..tpd {
            let t = chosen[rng.gen_range(0..chosen.len())];
            vecs.push(sample_from_topic(&mut rng, &topics[t], noise));
        }
        docs.push(MultiVecDoc {
            id: DocId(i as u64),
            vecs,
        });
        doc_topics.push(chosen);
    }
    (docs, doc_topics)
}

/// Build synthetic queries each probing `tpq` topics.
fn gen_queries(
    n_queries: usize,
    tpq: usize,
    topics: &[Vec<f32>],
    noise: f32,
) -> (Vec<MultiVecQuery>, Vec<Vec<usize>>) {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0FFEE);
    let n_topics = topics.len();
    let mut queries = Vec::with_capacity(n_queries);
    let mut q_topics = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let nt = tpq.min(n_topics);
        let mut chosen: Vec<usize> = Vec::with_capacity(nt);
        while chosen.len() < nt {
            let t = rng.gen_range(0..n_topics);
            if !chosen.contains(&t) {
                chosen.push(t);
            }
        }
        let vecs: Vec<Vec<f32>> = chosen
            .iter()
            .map(|&t| sample_from_topic(&mut rng, &topics[t], noise))
            .collect();
        queries.push(MultiVecQuery { vecs });
        q_topics.push(chosen);
    }
    (queries, q_topics)
}

fn recall_at_k(results: &[Vec<DocId>], ground: &[Vec<Vec<DocId>>], k: usize) -> f64 {
    let mut hits = 0u64;
    let mut total = 0u64;
    for (res, gt) in results.iter().zip(ground.iter()) {
        let gt_set: std::collections::HashSet<DocId> =
            gt.iter().flatten().take(k).cloned().collect();
        for r in res.iter().take(k) {
            if gt_set.contains(r) {
                hits += 1;
            }
        }
        total += k as u64;
    }
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

// ── Benchmark runner ─────────────────────────────────────────────────────────

fn run_variant<I: MultiVecIndex>(
    name: &str,
    idx: &I,
    queries: &[MultiVecQuery],
    ground: &[Vec<Vec<DocId>>],
    k: usize,
    memory_bytes: usize,
) -> RunStats {
    let n_queries = queries.len();
    let mut latencies_us: Vec<f64> = Vec::with_capacity(n_queries);
    let mut all_results: Vec<Vec<DocId>> = Vec::with_capacity(n_queries);

    for q in queries {
        let t0 = Instant::now();
        let res = idx.search(q, k).unwrap();
        let elapsed = t0.elapsed().as_micros() as f64;
        latencies_us.push(elapsed);
        all_results.push(res.into_iter().map(|r| r.doc_id).collect());
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = latencies_us.iter().sum::<f64>() / n_queries as f64;
    let p50 = latencies_us[n_queries / 2];
    let p95 = latencies_us[(n_queries * 95) / 100];
    let total_us: f64 = latencies_us.iter().sum();
    let throughput = n_queries as f64 / (total_us / 1_000_000.0);
    let recall = recall_at_k(&all_results, ground, k);

    RunStats {
        variant: name.to_string(),
        n_docs: idx.len(),
        n_token_vecs: 0,
        dims: idx.dims(),
        n_queries,
        mean_latency_us: mean,
        p50_latency_us: p50,
        p95_latency_us: p95,
        throughput_qps: throughput,
        recall_at_k: recall,
        memory_bytes,
    }
}

fn print_stats(s: &RunStats) {
    println!(
        "  {:20} | n={:6} | dims={:3} | q={:5} | mean={:7.1}µs | p50={:7.1}µs | p95={:8.1}µs | QPS={:8.0} | mem={:6.1}KB | recall@10={:.3}",
        s.variant,
        s.n_docs,
        s.dims,
        s.n_queries,
        s.mean_latency_us,
        s.p50_latency_us,
        s.p95_latency_us,
        s.throughput_qps,
        s.memory_bytes as f64 / 1024.0,
        s.recall_at_k,
    );
}

// ── Memory estimates ─────────────────────────────────────────────────────────

fn flat_memory(n_docs: usize, tokens_per_doc: usize, dims: usize) -> usize {
    n_docs * tokens_per_doc * dims * 4
}

fn bucket_memory(n_docs: usize, tokens_per_doc: usize, dims: usize) -> usize {
    // docs: token vecs + centroid
    n_docs * (tokens_per_doc + 1) * dims * 4
}

fn hnsw_memory(n_docs: usize, tokens_per_doc: usize, dims: usize, m: usize) -> usize {
    let n_tokens = n_docs * tokens_per_doc;
    // per token: vec + neighbour list
    n_tokens * (dims * 4 + m * 8)
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = Config::from_env();

    // Print environment
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-maxsim: Multi-Vector MaxSim Late Interaction       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  OS:               {}", std::env::consts::OS);
    println!("  Arch:             {}", std::env::consts::ARCH);
    println!("  Rust version:     (cargo version at build time)");
    println!("  N docs:           {}", cfg.n_docs);
    println!("  Dims:             {}", cfg.dims);
    println!("  Tokens/doc:       {}", cfg.tokens_per_doc);
    println!("  Tokens/query:     {}", cfg.tokens_per_query);
    println!("  N queries:        {}", cfg.n_queries);
    println!("  N topics:         {}", cfg.n_topics);
    println!("  K (top-k):        {}", cfg.k);
    println!();

    // Generate data
    let noise = 0.3_f32;
    let topics = gen_topics(cfg.n_topics, cfg.dims);
    let (corpus, _doc_topics) =
        gen_corpus(cfg.n_docs, cfg.tokens_per_doc, &topics, cfg.dims, noise);
    let (queries, _q_topics) = gen_queries(cfg.n_queries, cfg.tokens_per_query, &topics, noise);

    // ── Variant 1: FlatMaxSim (ground truth) ────────────────────────────────
    println!("Building FlatMaxSim (exhaustive oracle)...");
    let mut flat = FlatMaxSim::new(cfg.dims);
    for doc in &corpus {
        flat.add(doc.clone()).unwrap();
    }
    let t0 = Instant::now();
    let ground: Vec<Vec<Vec<DocId>>> = queries
        .iter()
        .map(|q| {
            vec![flat
                .search(q, cfg.k)
                .unwrap()
                .into_iter()
                .map(|r| r.doc_id)
                .collect()]
        })
        .collect();
    println!(
        "  Ground truth computed in {:.1}ms",
        t0.elapsed().as_millis()
    );
    let flat_mem = flat_memory(cfg.n_docs, cfg.tokens_per_doc, cfg.dims);

    // ── Variant 2a: BucketMaxSim fast (oversampling=50) ─────────────────────
    println!("Building BucketMaxSim-fast (oversampling=50, 1% candidates)...");
    let mut bucket_fast = BucketMaxSim::new(cfg.dims, 50);
    for doc in &corpus {
        bucket_fast.add(doc.clone()).unwrap();
    }

    // ── Variant 2b: BucketMaxSim quality (oversampling=500) ─────────────────
    println!("Building BucketMaxSim-quality (oversampling=500, 10% candidates)...");
    let mut bucket_quality = BucketMaxSim::new(cfg.dims, 500);
    for doc in &corpus {
        bucket_quality.add(doc.clone()).unwrap();
    }
    let bucket_mem = bucket_memory(cfg.n_docs, cfg.tokens_per_doc, cfg.dims);

    // ── Variant 3: HnswMaxSim ───────────────────────────────────────────────
    println!("Building HnswMaxSim (token_candidates=32)...");
    let mut hnsw = HnswMaxSim::new(cfg.dims, 32);
    for doc in &corpus {
        hnsw.add(doc.clone()).unwrap();
    }
    let hnsw_mem = hnsw_memory(cfg.n_docs, cfg.tokens_per_doc, cfg.dims, 16);

    // ── Run benchmarks ───────────────────────────────────────────────────────
    println!("\nRunning benchmarks ({} queries each)...\n", cfg.n_queries);
    println!(
        "  {:20} | {:6} | {:3} | {:5} | {:>10} | {:>10} | {:>11} | {:>8} | {:>8} | {:10}",
        "Variant", "N", "D", "Q", "mean_lat", "p50_lat", "p95_lat", "QPS", "mem", "recall@10"
    );
    println!("{}", "-".repeat(115));

    let flat_stats = run_variant("FlatMaxSim", &flat, &queries, &ground, cfg.k, flat_mem);
    let bucket_fast_stats = run_variant(
        "BucketFast(os=50)",
        &bucket_fast,
        &queries,
        &ground,
        cfg.k,
        bucket_mem,
    );
    let bucket_quality_stats = run_variant(
        "BucketQual(os=500)",
        &bucket_quality,
        &queries,
        &ground,
        cfg.k,
        bucket_mem,
    );
    let hnsw_stats = run_variant("HnswMaxSim", &hnsw, &queries, &ground, cfg.k, hnsw_mem);

    print_stats(&flat_stats);
    print_stats(&bucket_fast_stats);
    print_stats(&bucket_quality_stats);
    print_stats(&hnsw_stats);

    // ── Memory math ──────────────────────────────────────────────────────────
    println!("\n── Memory analysis ─────────────────────────────────────────────");
    println!(
        "  FlatMaxSim:   {} docs × {} tokens × {} dims × 4B = {:.1} KB",
        cfg.n_docs,
        cfg.tokens_per_doc,
        cfg.dims,
        flat_mem as f64 / 1024.0
    );
    println!(
        "  BucketMaxSim: +{:.0}% overhead for centroids = {:.1} KB",
        100.0 * (cfg.tokens_per_doc + 1) as f64 / cfg.tokens_per_doc as f64 - 100.0,
        bucket_mem as f64 / 1024.0
    );
    println!(
        "  HnswMaxSim:   {} token nodes × ({} dims × 4B + 16 nbrs × 8B) = {:.1} KB",
        cfg.n_docs * cfg.tokens_per_doc,
        cfg.dims,
        hnsw_mem as f64 / 1024.0
    );

    // ── Multi-token advantage demo ────────────────────────────────────────────
    println!("\n── Multi-token advantage demonstration ─────────────────────────");
    let topic_a = &topics[0];
    let topic_b = &topics[1];
    let mut demo_flat = FlatMaxSim::new(cfg.dims);
    // Doc A: covers topic A only
    demo_flat
        .add(MultiVecDoc {
            id: DocId(1000),
            vecs: vec![topic_a.clone()],
        })
        .unwrap();
    // Doc AB: covers both topics
    demo_flat
        .add(MultiVecDoc {
            id: DocId(1001),
            vecs: vec![topic_a.clone(), topic_b.clone()],
        })
        .unwrap();
    // Query about topic B only
    let demo_q = MultiVecQuery {
        vecs: vec![topic_b.clone()],
    };
    let demo_res = demo_flat.search(&demo_q, 2).unwrap();
    println!("  Query: topic B only");
    println!(
        "  Doc 1000 (topic A only): score = {:.4}",
        demo_res
            .iter()
            .find(|r| r.doc_id == DocId(1000))
            .map(|r| r.score)
            .unwrap_or(0.0)
    );
    println!(
        "  Doc 1001 (topic A+B):   score = {:.4}",
        demo_res
            .iter()
            .find(|r| r.doc_id == DocId(1001))
            .map(|r| r.score)
            .unwrap_or(0.0)
    );
    println!("  Winner: doc {:?}", demo_res[0].doc_id);

    // ── Acceptance tests ─────────────────────────────────────────────────────
    println!("\n── Acceptance tests ────────────────────────────────────────────");

    // FlatMaxSim is the oracle — recall vs itself should be 1.0
    let flat_self_recall = flat_stats.recall_at_k;
    assert!(
        (flat_self_recall - 1.0).abs() < 0.01,
        "FlatMaxSim recall vs self must be 1.0, got {flat_self_recall:.3}"
    );
    println!("  PASS: FlatMaxSim recall@10 vs self = {flat_self_recall:.3} (expected 1.000)");

    // BucketFast (os=50): checks only 1% of corpus — low recall is expected
    let bucket_fast_recall = bucket_fast_stats.recall_at_k;
    assert!(
        bucket_fast_recall >= 0.20,
        "BucketFast recall@10 must be >= 0.20, got {bucket_fast_recall:.3}"
    );
    println!(
        "  PASS: BucketFast(os=50) recall@10 = {bucket_fast_recall:.3} (threshold 0.20, 1% candidates)"
    );

    // BucketQuality (os=500): checks 10% of corpus — should give decent recall
    let bucket_quality_recall = bucket_quality_stats.recall_at_k;
    assert!(
        bucket_quality_recall >= 0.60,
        "BucketQuality recall@10 must be >= 0.60, got {bucket_quality_recall:.3}"
    );
    println!(
        "  PASS: BucketQuality(os=500) recall@10 = {bucket_quality_recall:.3} (threshold 0.60, 10% candidates)"
    );

    // BucketFast must be faster than FlatMaxSim (main speed benefit)
    assert!(
        bucket_fast_stats.throughput_qps >= flat_stats.throughput_qps * 2.0,
        "BucketFast QPS ({:.0}) must be >= 2x FlatMaxSim QPS ({:.0})",
        bucket_fast_stats.throughput_qps,
        flat_stats.throughput_qps
    );
    println!(
        "  PASS: BucketFast QPS {:.0} >= 2x FlatMaxSim QPS {:.0}",
        bucket_fast_stats.throughput_qps, flat_stats.throughput_qps
    );

    let hnsw_recall = hnsw_stats.recall_at_k;
    assert!(
        hnsw_recall >= 0.30,
        "HnswMaxSim recall@10 must be >= 0.30, got {hnsw_recall:.3}"
    );
    println!("  PASS: HnswMaxSim recall@10 = {hnsw_recall:.3} (threshold 0.30)");

    // Multi-token demo: doc AB should rank above doc A for topic-B query
    assert_eq!(
        demo_res[0].doc_id,
        DocId(1001),
        "multi-token doc must rank first for topic-B query"
    );
    println!("  PASS: Multi-token doc (A+B) ranks above single-token doc (A) for topic-B query");

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  ALL ACCEPTANCE TESTS PASSED                                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}
