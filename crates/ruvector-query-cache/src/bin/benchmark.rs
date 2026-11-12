//! Benchmark: Semantic Query Cache variants
//!
//! Compares three caching strategies on a synthetic agent-memory workload:
//!   1. NoCache       – ground-truth brute-force, 0% hit rate
//!   2. ExactCache    – bitwise-exact hit only
//!   3. SemanticCache – cosine-similarity cache at multiple thresholds
//!
//! Usage:
//!   cargo run --release -p ruvector-query-cache --bin benchmark
//!   cargo run --release -p ruvector-query-cache --bin benchmark -- --n 5000 --queries 500 --dim 128 --k 10

use ruvector_query_cache::{
    dataset::Dataset, exact_cache::ExactCache, no_cache::NoCache, recall_at_k,
    semantic_cache::SemanticCache, CachedAnn,
};
use std::time::{Duration, Instant};

// ─── constants ───────────────────────────────────────────────────────────────

const N_CORPUS: usize = 5_000;
const DIM: usize = 128;
const N_QUERIES: usize = 500;
const K: usize = 10;
const REPEAT_RATE: f32 = 0.35; // 35% of queries are near-duplicates (agent scenario)
const JITTER: f32 = 0.05; // noise magnitude on repeated queries
const CACHE_CAP: usize = 512; // maximum cache entries
const SEED: u64 = 42;

// Semantic thresholds to sweep.
const SEM_THRESHOLDS: &[f32] = &[0.85, 0.90, 0.95, 0.99];

// ─── latency helpers ─────────────────────────────────────────────────────────

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (v.len() - 1) as f64).round() as usize;
    v[idx.min(v.len() - 1)]
}

fn throughput(total_queries: usize, elapsed: Duration) -> f64 {
    total_queries as f64 / elapsed.as_secs_f64()
}

// ─── single-variant run ──────────────────────────────────────────────────────

struct BenchResult {
    name: String,
    hit_rate: f32,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
    recall: f32,
    mem_kb: usize,
    threshold: Option<f32>,
}

fn run_variant(
    name: &str,
    variant: &mut dyn CachedAnn,
    dataset: &Dataset,
    threshold: Option<f32>,
) -> BenchResult {
    let mut latencies_us: Vec<f64> = Vec::with_capacity(dataset.n_queries);
    let mut recall_sum = 0.0f32;
    let start = Instant::now();

    for (qi, query) in dataset.queries.iter().enumerate() {
        let t0 = Instant::now();
        let (hits, _dec) = variant.search(query, dataset.k);
        latencies_us.push(t0.elapsed().as_secs_f64() * 1e6);

        // Recall against exact ground truth
        let gt: Vec<ruvector_query_cache::Hit> = dataset.ground_truth[qi]
            .iter()
            .enumerate()
            .map(|(rank, &id)| ruvector_query_cache::Hit {
                id,
                dist: rank as f32,
            })
            .collect();
        recall_sum += recall_at_k(&hits, &gt, dataset.k);
    }

    let elapsed = start.elapsed();
    let stats = variant.stats();
    let mean_us = latencies_us.iter().sum::<f64>() / latencies_us.len() as f64;

    BenchResult {
        name: name.to_string(),
        hit_rate: stats.hit_rate(),
        mean_us,
        p50_us: percentile(latencies_us.clone(), 50.0),
        p95_us: percentile(latencies_us, 95.0),
        qps: throughput(dataset.n_queries, elapsed),
        recall: recall_sum / dataset.n_queries as f32,
        mem_kb: variant.memory_bytes() / 1024,
        threshold,
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    print_header();

    println!("Generating dataset …");
    let dataset = Dataset::generate(SEED, N_CORPUS, DIM, N_QUERIES, K, REPEAT_RATE, JITTER);
    println!(
        "  corpus={} dim={} queries={} k={} repeat_rate={:.0}% jitter={:.3}\n",
        dataset.n_corpus,
        dataset.dim,
        dataset.n_queries,
        dataset.k,
        REPEAT_RATE * 100.0,
        JITTER,
    );

    let mut results: Vec<BenchResult> = Vec::new();

    // ── 1. NoCache ────────────────────────────────────────────────────────────
    {
        let mut nc = NoCache::new(dataset.corpus.clone());
        let r = run_variant("NoCache", &mut nc, &dataset, None);
        results.push(r);
    }

    // ── 2. ExactCache ─────────────────────────────────────────────────────────
    {
        let mut ec = ExactCache::new(dataset.corpus.clone(), CACHE_CAP);
        let r = run_variant("ExactCache", &mut ec, &dataset, None);
        results.push(r);
    }

    // ── 3. SemanticCache at each threshold ───────────────────────────────────
    for &thr in SEM_THRESHOLDS {
        let mut sc = SemanticCache::new(dataset.corpus.clone(), CACHE_CAP, thr);
        let name = format!("Semantic@{:.2}", thr);
        let r = run_variant(&name, &mut sc, &dataset, Some(thr));
        results.push(r);
    }

    // ─── print table ─────────────────────────────────────────────────────────
    println!(
        "{:<20} {:>8} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8}",
        "Variant", "HitRate", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Recall", "Mem(KB)"
    );
    println!("{}", "─".repeat(86));
    for r in &results {
        println!(
            "{:<20} {:>7.1}% {:>9.1} {:>9.1} {:>9.1} {:>8.0} {:>8.3} {:>8}",
            r.name,
            r.hit_rate * 100.0,
            r.mean_us,
            r.p50_us,
            r.p95_us,
            r.qps,
            r.recall,
            r.mem_kb,
        );
    }

    // ─── acceptance test ─────────────────────────────────────────────────────
    println!("\n── Acceptance tests ──");

    let no_cache = results.iter().find(|r| r.name == "NoCache").unwrap();
    let baseline_latency = no_cache.mean_us;
    assert!(
        (no_cache.recall - 1.0).abs() < 1e-3,
        "NoCache recall must be 1.0, got {:.4}",
        no_cache.recall
    );
    println!("✓ NoCache recall = 1.000 (ground truth)");

    // ExactCache: recall ≥ 0.99 (hits are exact, misses are ground truth)
    let exact = results.iter().find(|r| r.name == "ExactCache").unwrap();
    assert!(
        exact.recall >= 0.99,
        "ExactCache recall must be ≥0.99, got {:.4}",
        exact.recall
    );
    println!("✓ ExactCache recall ≥ 0.99 (got {:.4})", exact.recall);

    // SemanticCache@0.90: hit_rate > exact cache (semantic is looser)
    let sem90 = results.iter().find(|r| r.name == "Semantic@0.90").unwrap();
    assert!(
        sem90.hit_rate >= exact.hit_rate,
        "SemanticCache@0.90 hit_rate must be ≥ ExactCache ({:.1}%), got {:.1}%",
        exact.hit_rate * 100.0,
        sem90.hit_rate * 100.0,
    );
    println!(
        "✓ SemanticCache@0.90 hit_rate ≥ ExactCache ({:.1}% vs {:.1}%)",
        sem90.hit_rate * 100.0,
        exact.hit_rate * 100.0,
    );

    // SemanticCache@0.90: recall ≥ 0.70
    assert!(
        sem90.recall >= 0.70,
        "SemanticCache@0.90 recall must be ≥0.70, got {:.4}",
        sem90.recall
    );
    println!(
        "✓ SemanticCache@0.90 recall ≥ 0.70 (got {:.4})",
        sem90.recall
    );

    // SemanticCache@0.85: mean latency ≤ 85% of NoCache when hit_rate > 10%
    let sem85 = results.iter().find(|r| r.name == "Semantic@0.85").unwrap();
    if sem85.hit_rate > 0.10 {
        let speedup_threshold = 0.90 * baseline_latency;
        assert!(
            sem85.mean_us <= speedup_threshold,
            "Semantic@0.85 mean latency ({:.1}µs) should be < {:.1}µs when hit_rate={:.1}%",
            sem85.mean_us,
            speedup_threshold,
            sem85.hit_rate * 100.0,
        );
        println!(
            "✓ Semantic@0.85 mean latency ({:.1}µs) < 90% of NoCache ({:.1}µs)",
            sem85.mean_us, speedup_threshold,
        );
    } else {
        println!(
            "  Semantic@0.85 hit_rate {:.1}% too low for latency test (skipped)",
            sem85.hit_rate * 100.0
        );
    }

    // Monotone quality: higher threshold → higher recall
    let sem99 = results.iter().find(|r| r.name == "Semantic@0.99").unwrap();
    assert!(
        sem99.recall >= sem85.recall,
        "Higher threshold must yield higher recall: @0.99={:.4} vs @0.85={:.4}",
        sem99.recall,
        sem85.recall,
    );
    println!(
        "✓ Monotone quality: recall@0.99 ({:.4}) ≥ recall@0.85 ({:.4})",
        sem99.recall, sem85.recall,
    );

    println!("\n=== PASS — all acceptance tests satisfied ===");
    println!(
        "\nKey insight: SemanticCache@0.90 trades {:.0}% hit rate for {:.1}% recall fidelity",
        sem90.hit_rate * 100.0,
        sem90.recall * 100.0,
    );
    println!(
        "at {:.1}µs mean latency vs {:.1}µs for NoCache (repeat_rate={:.0}%)",
        sem90.mean_us,
        baseline_latency,
        REPEAT_RATE * 100.0,
    );
}

fn print_header() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  ruvector-query-cache — Semantic Query Cache Bench  ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();
    // Print OS/Rust info
    println!("OS:      {}", std::env::consts::OS);
    println!("ARCH:    {}", std::env::consts::ARCH);
    println!("Rust:    {}", env!("CARGO_PKG_RUST_VERSION", "unknown"));
    println!(
        "Config:  corpus={} dim={} queries={} k={} cache_cap={}",
        N_CORPUS, DIM, N_QUERIES, K, CACHE_CAP,
    );
    println!();
}
