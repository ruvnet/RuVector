//! Benchmark: semantic query cache for ANN search.
//!
//! Measures cache hit rate, latency per query, and throughput for three variants:
//!   1. ExactCache   — bit-identical key match (baseline)
//!   2. LinearCache  — fixed-threshold cosine scan
//!   3. AdaptiveCache — self-tuning cosine scan
//!
//! Dataset: N unit-normalised random vectors of dimension DIM.
//! Workload: N_UNIQUE distinct queries + N_DUP near-duplicates (epsilon noise).

use ruvector_semantic_cache::{
    adaptive::{AdaptiveCache, TuneParams},
    dataset::{brute_force_topk, build_workload, Dataset},
    linear::LinearCache,
    ExactCache, SearchResult, SemanticCache,
};
use std::time::Instant;

// ── Configuration ────────────────────────────────────────────────────────────

const N_VECTORS: usize = 10_000;
const DIM: usize = 128;
const K: usize = 10;
const N_UNIQUE: usize = 600;
const N_DUP: usize = 400;
const EPSILON: f32 = 0.04; // noise magnitude for near-duplicates
const CACHE_CAP: usize = 64;
const LINEAR_THRESHOLD: f32 = 0.97;
const ADAPTIVE_INIT_THRESHOLD: f32 = 0.95;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Run a cache variant over the workload; return per-query latencies (ns).
fn run_variant<C: SemanticCache>(
    cache: &mut C,
    dataset: &Dataset,
    workload: &[ruvector_semantic_cache::dataset::QueryEntry],
) -> Vec<u64> {
    let mut latencies = Vec::with_capacity(workload.len());

    for entry in workload {
        let t0 = Instant::now();

        let result = cache.query(&entry.vec);
        if result.is_none() {
            // Cache miss — run ANN and time it separately.
            let ann_t0 = Instant::now();
            let ids = brute_force_topk(&entry.vec, &dataset.vectors, K);
            let ann_ns = ann_t0.elapsed().as_nanos() as u64;
            let results: Vec<SearchResult> = ids
                .iter()
                .enumerate()
                .map(|(rank, &id)| SearchResult {
                    id,
                    distance: rank as f32,
                })
                .collect();
            cache.insert(entry.vec.clone(), results);
            cache.record_ann_latency(ann_ns);
        }

        latencies.push(t0.elapsed().as_nanos() as u64);
    }

    latencies
}

/// Compute mean, p50, p95 from a slice of nanosecond latencies.
fn percentiles(mut lats: Vec<u64>) -> (f64, u64, u64) {
    lats.sort_unstable();
    let n = lats.len();
    let mean = lats.iter().sum::<u64>() as f64 / n as f64;
    let p50 = lats[n / 2];
    let p95 = lats[(n * 95) / 100];
    (mean, p50, p95)
}

fn throughput(lats: &[u64]) -> f64 {
    let total_ns: u64 = lats.iter().sum();
    if total_ns == 0 {
        return 0.0;
    }
    lats.len() as f64 / (total_ns as f64 / 1_000_000_000.0)
}

/// Recall@k: fraction of queries where ground-truth top-1 appears in cache results.
fn recall_at_1(
    cache_results: &[Option<Vec<SearchResult>>],
    workload: &[ruvector_semantic_cache::dataset::QueryEntry],
) -> f64 {
    let hit_queries: Vec<_> = cache_results
        .iter()
        .zip(workload.iter())
        .filter(|(r, _)| r.is_some())
        .collect();
    if hit_queries.is_empty() {
        return 1.0; // no hits → vacuously correct
    }
    let correct = hit_queries
        .iter()
        .filter(|(res, entry)| res.as_ref().unwrap().iter().any(|r| r.id == entry.nn_idx))
        .count();
    correct as f64 / hit_queries.len() as f64
}

// ── Recall-tracking run ───────────────────────────────────────────────────────

fn run_with_recall<C: SemanticCache>(
    cache: &mut C,
    dataset: &Dataset,
    workload: &[ruvector_semantic_cache::dataset::QueryEntry],
) -> (Vec<u64>, Vec<Option<Vec<SearchResult>>>) {
    let mut latencies = Vec::with_capacity(workload.len());
    let mut cache_results: Vec<Option<Vec<SearchResult>>> = Vec::with_capacity(workload.len());

    for entry in workload {
        let t0 = Instant::now();
        let result = cache.query(&entry.vec);

        if result.is_none() {
            let ann_t0 = Instant::now();
            let ids = brute_force_topk(&entry.vec, &dataset.vectors, K);
            let ann_ns = ann_t0.elapsed().as_nanos() as u64;
            let results: Vec<SearchResult> = ids
                .iter()
                .enumerate()
                .map(|(rank, &id)| SearchResult {
                    id,
                    distance: rank as f32,
                })
                .collect();
            cache.insert(entry.vec.clone(), results);
            cache.record_ann_latency(ann_ns);
            cache_results.push(None);
        } else {
            cache_results.push(result);
        }

        latencies.push(t0.elapsed().as_nanos() as u64);
    }

    (latencies, cache_results)
}

// ── Print helpers ─────────────────────────────────────────────────────────────

fn separator() {
    println!("{}", "─".repeat(80));
}

fn print_result(
    name: &str,
    lats: &[u64],
    stats: &ruvector_semantic_cache::CacheStats,
    recall: f64,
    acceptance_passed: bool,
) {
    let (mean, p50, p95) = percentiles(lats.to_vec());
    let qps = throughput(lats);
    let mean_ann = stats.mean_ann_latency_ns();
    let hit_rate = stats.hit_rate();

    println!("Variant          : {}", name);
    println!(
        "Queries          : {}  Hits: {}  Misses: {}  Evictions: {}",
        stats.queries, stats.hits, stats.misses, stats.evictions
    );
    println!("Hit rate         : {:.1}%", hit_rate * 100.0);
    println!("Recall@1 on hits : {:.3}", recall);
    println!("Mean latency     : {:.1} µs", mean / 1_000.0);
    println!("p50 latency      : {:.1} µs", p50 as f64 / 1_000.0);
    println!("p95 latency      : {:.1} µs", p95 as f64 / 1_000.0);
    println!("Throughput       : {:.0} QPS", qps);
    println!(
        "Mean ANN latency : {:.1} µs  (misses only)",
        mean_ann / 1_000.0
    );
    println!(
        "Acceptance       : {}",
        if acceptance_passed {
            "PASS ✓"
        } else {
            "FAIL ✗"
        }
    );
    separator();
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    // System info
    separator();
    println!("RuVector Semantic Query Cache — Benchmark");
    separator();
    println!("OS            : {}", std::env::consts::OS);
    println!("Arch          : {}", std::env::consts::ARCH);
    println!("Dataset N     : {}", N_VECTORS);
    println!("Dimensions    : {}", DIM);
    println!(
        "Workload      : {} unique + {} near-dup = {} queries",
        N_UNIQUE,
        N_DUP,
        N_UNIQUE + N_DUP
    );
    println!("Dup epsilon   : {}", EPSILON);
    println!("k (top-k)     : {}", K);
    println!("Cache capacity: {}", CACHE_CAP);
    println!("Linear thresh : {}", LINEAR_THRESHOLD);
    println!("Adaptive init : {}", ADAPTIVE_INIT_THRESHOLD);
    separator();

    // Build dataset
    print!("Building dataset ({} × {})...", N_VECTORS, DIM);
    let t = Instant::now();
    let dataset = Dataset::random(N_VECTORS, DIM, 0xABCD_1234);
    println!(" {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);

    // Build workload
    print!("Building workload ({} queries)...", N_UNIQUE + N_DUP);
    let t = Instant::now();
    let workload = build_workload(&dataset, N_UNIQUE, N_DUP, EPSILON, 0xBEEF_CAFE);
    println!(" {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
    separator();

    let mut all_pass = true;

    // ── Variant 1: ExactCache (baseline) ─────────────────────────────────────
    {
        let mut cache = ExactCache::new(CACHE_CAP);
        let (lats, cr) = run_with_recall(&mut cache, &dataset, &workload);
        let recall = recall_at_1(&cr, &workload);
        let stats = cache.stats().clone();

        // Exact cache: hit rate is low (only bit-identical hits), recall on hits must be 1.0.
        let pass = recall >= 0.99;
        all_pass &= pass;
        print_result("ExactCache (baseline)", &lats, &stats, recall, pass);
    }

    // ── Variant 2: LinearCache (fixed threshold) ──────────────────────────────
    {
        let mut cache = LinearCache::new(CACHE_CAP, LINEAR_THRESHOLD);
        let (lats, cr) = run_with_recall(&mut cache, &dataset, &workload);
        let recall = recall_at_1(&cr, &workload);
        let stats = cache.stats().clone();

        // LinearCache: must achieve ≥25% hit rate on the dup-heavy workload,
        // and recall on hits must be ≥ 0.85.
        let hit_rate = stats.hit_rate();
        let pass = hit_rate >= 0.25 && recall >= 0.85;
        all_pass &= pass;
        print_result("LinearCache (threshold=0.97)", &lats, &stats, recall, pass);
    }

    // ── Variant 3: AdaptiveCache ─────────────────────────────────────────────
    {
        let params = TuneParams {
            tune_interval: 50,
            step: 0.01,
            min_threshold: 0.90,
            max_threshold: 0.9999,
            target_hit_rate: 0.30,
            max_false_positive_rate: 0.05,
        };
        let mut cache = AdaptiveCache::new(CACHE_CAP, ADAPTIVE_INIT_THRESHOLD, params);
        let (lats, cr) = run_with_recall(&mut cache, &dataset, &workload);
        let recall = recall_at_1(&cr, &workload);
        let stats = cache.stats().clone();

        // AdaptiveCache: must achieve ≥ LinearCache hit rate (or close to it)
        // with recall ≥ 0.85.
        let hit_rate = stats.hit_rate();
        let pass = hit_rate >= 0.20 && recall >= 0.85;
        all_pass &= pass;
        print_result("AdaptiveCache (init=0.95)", &lats, &stats, recall, pass);
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    separator();
    println!(
        "OVERALL: {}",
        if all_pass {
            "ALL TESTS PASSED ✓"
        } else {
            "SOME TESTS FAILED ✗"
        }
    );
    separator();

    if !all_pass {
        std::process::exit(1);
    }
}
