//! Semantic Query Cache — benchmark binary.
//!
//! Compares three variants on a clustered synthetic workload that mimics
//! agent memory access patterns (repeated queries with slight perturbations):
//!
//!   1. NoCache        — every query hits the DB (linear scan, no caching).
//!   2. LinearCache    — exhaustive cosine scan over cached entries.
//!   3. ShardedCache   — LSH-bucketed cache; lookup only scans matching shard.
//!
//! Run:
//!   cargo run --release -p ruvector-semantic-cache --bin benchmark
//!
//! Env vars to override defaults:
//!   N_VECS=10000 N_CLUSTERS=50 N_QUERIES=500 DIMS=128 NOISE=0.05
//!   cargo run --release -p ruvector-semantic-cache --bin benchmark

use ruvector_semantic_cache::{
    dataset::{brute_force_top_k, recall_at_k, Dataset, DatasetConfig},
    linear::LinearScanCache,
    sharded::ShardedCache,
    NoCache, QueryCache,
};
use std::time::Instant;

// ─── parameters ──────────────────────────────────────────────────────────────

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn n_vecs() -> usize {
    env_usize("N_VECS", 10_000)
}
fn n_clusters() -> usize {
    env_usize("N_CLUSTERS", 50)
}
fn n_queries() -> usize {
    env_usize("N_QUERIES", 500)
}
fn dims() -> usize {
    env_usize("DIMS", 128)
}
fn noise() -> f32 {
    // noise_std=0.02 → within-cluster cosine sim ≈ 0.95 in dim=128
    // (noise_power = 0.02² × 128 = 0.051 → |c+ε| ≈ 1.025 → sim ≈ 0.951)
    // This keeps queries above the 0.92 cache threshold while staying realistic.
    env_f32("NOISE", 0.02)
}

const K: usize = 10;
const SEED: u64 = 0xC0DE_CAFE_BABE_9999;
const CACHE_THRESHOLD: f32 = 0.92;
const TTL_TICKS: u64 = u64::MAX; // no TTL for benchmark

// ─── acceptance thresholds ───────────────────────────────────────────────────

/// Linear cache hit rate must exceed this for clustered queries.
const MIN_HIT_RATE_LINEAR: f32 = 0.80;
/// Sharded cache hit rate may be slightly lower due to bucket boundaries.
const MIN_HIT_RATE_SHARDED: f32 = 0.70;
/// Mean recall of all queries (hits return prior-query results with some loss;
/// misses return exact brute-force results with recall=1.0).
/// Expected: ~10% misses at 1.0 + ~90% hits at ~0.70-0.80 = ~0.73-0.82 total.
const MIN_MEAN_RECALL: f32 = 0.60;
/// Speedup of cached variant vs NoCache (mean latency ratio).
const MIN_SPEEDUP: f64 = 3.0;

// ─── percentile helper ───────────────────────────────────────────────────────

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ─── result row ──────────────────────────────────────────────────────────────

struct Row {
    name: &'static str,
    hit_rate: f32,
    mean_us: f64,
    p50_us: u128,
    p95_us: u128,
    qps: f64,
    mem_bytes: usize,
    mean_recall: f32,
    pass: bool,
}

// ─── run one variant ─────────────────────────────────────────────────────────

fn run_variant(
    name: &'static str,
    dataset: &Dataset,
    cache: &mut dyn QueryCache,
    min_hit_rate: f32,
) -> Row {
    let nq = dataset.queries.len();
    let mut latencies: Vec<u128> = Vec::with_capacity(nq);
    let mut hit_count = 0usize;
    let mut total_recall = 0.0f32;
    let mut tick = 0u64;

    for (i, query) in dataset.queries.iter().enumerate() {
        let t0 = Instant::now();

        let result = match cache.lookup(query, CACHE_THRESHOLD, tick, TTL_TICKS) {
            Some(cached) => {
                hit_count += 1;
                cached
            }
            None => {
                let ids = brute_force_top_k(query, &dataset.base, K);
                // Only insert if not NoCache (len stays 0).
                cache.insert(query.clone(), ids.clone(), tick);
                ids
            }
        };

        let elapsed = t0.elapsed().as_micros();
        latencies.push(elapsed);
        tick += 1;

        // Measure result quality vs ground truth.
        let recall = recall_at_k(&dataset.ground_truth[i], &result);
        total_recall += recall;
    }

    latencies.sort_unstable();
    let mean_us = latencies.iter().sum::<u128>() as f64 / latencies.len() as f64;
    let p50 = percentile(&latencies, 50.0);
    let p95 = percentile(&latencies, 95.0);
    let total_secs = latencies.iter().sum::<u128>() as f64 / 1_000_000.0;
    let qps = nq as f64 / total_secs;
    let hit_rate = hit_count as f32 / nq as f32;
    let mean_recall = total_recall / nq as f32;
    let mem_bytes = cache.memory_bytes(dataset.config.dims);
    // Semantic caches trade some recall for speed; cached hits return results
    // from a similar prior query (not the exact query), so mean_recall < 1.0
    // is expected and documented. The acceptance criterion is a floor.
    let pass = if name == "NoCache" {
        true
    } else {
        hit_rate >= min_hit_rate && mean_recall >= MIN_MEAN_RECALL
    };

    Row {
        name,
        hit_rate,
        mean_us,
        p50_us: p50,
        p95_us: p95,
        qps,
        mem_bytes,
        mean_recall,
        pass,
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    // Print environment info.
    println!("=== Semantic Query Cache Benchmark ===");
    println!();
    let rust_ver = std::env::var("RUSTUP_TOOLCHAIN").unwrap_or_else(|_| "stable".to_string());
    println!("Rust toolchain  : {rust_ver}");
    println!(
        "OS              : {}",
        std::env::var("OSTYPE").unwrap_or_else(|_| {
            if cfg!(target_os = "linux") {
                "linux".to_string()
            } else if cfg!(target_os = "macos") {
                "macos".to_string()
            } else {
                "unknown".to_string()
            }
        })
    );
    println!("Target arch     : {}", std::env::consts::ARCH);

    let nv = n_vecs();
    let nc = n_clusters();
    let nq = n_queries();
    let d = dims();
    let noise_std = noise();

    println!();
    println!("Dataset");
    println!("  base vectors  : {nv}");
    println!("  clusters      : {nc}");
    println!("  queries       : {nq}");
    println!("  dims          : {d}");
    println!("  noise_std     : {noise_std:.3}");
    println!("  k             : {K}");
    println!("  threshold     : {CACHE_THRESHOLD:.2}");
    println!();
    println!("Generating dataset...");
    let t_gen = Instant::now();
    let dataset = Dataset::generate(DatasetConfig {
        n_vecs: nv,
        n_clusters: nc,
        n_queries: nq,
        dims: d,
        noise_std,
        k: K,
        seed: SEED,
    });
    println!("  Generated in  : {:.1}ms", t_gen.elapsed().as_millis());
    println!("  Ground truth  : {nq} × {K} IDs");
    println!();

    // ── Variant 1: NoCache ──
    println!("Running NoCache...");
    let mut no_cache = NoCache::new();
    let row_no = run_variant("NoCache", &dataset, &mut no_cache, 0.0);

    // ── Variant 2: LinearScanCache ──
    println!("Running LinearScanCache...");
    let mut linear = LinearScanCache::new(nc * 2); // capacity = 2× clusters
    let row_lin = run_variant("LinearCache", &dataset, &mut linear, MIN_HIT_RATE_LINEAR);

    // ── Variant 3: ShardedCache ──
    println!("Running ShardedCache...");
    let mut sharded = ShardedCache::new(nc * 4, d, SEED);
    let row_sh = run_variant("ShardedCache", &dataset, &mut sharded, MIN_HIT_RATE_SHARDED);

    // ── Print results table ──
    println!();
    println!(
        "{:<16} {:>8} {:>10} {:>8} {:>8} {:>10} {:>10} {:>8} {:>7}",
        "Variant", "HitRate", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Mem(KB)", "Recall", "PASS"
    );
    println!("{}", "-".repeat(92));

    for row in [&row_no, &row_lin, &row_sh] {
        let mem_kb = row.mem_bytes as f64 / 1024.0;
        let hit_pct = row.hit_rate * 100.0;
        let pass_str = if row.pass { "PASS" } else { "FAIL" };
        println!(
            "{:<16} {:>7.1}% {:>10.1} {:>8} {:>8} {:>10.0} {:>10.1} {:>8.3} {:>7}",
            row.name,
            hit_pct,
            row.mean_us,
            row.p50_us,
            row.p95_us,
            row.qps,
            mem_kb,
            row.mean_recall,
            pass_str
        );
    }

    // ── Acceptance summary ──
    println!();
    println!("=== Acceptance ===");
    println!(
        "  LinearCache hit rate >= {:.0}%  : {} ({:.1}%)",
        MIN_HIT_RATE_LINEAR * 100.0,
        if row_lin.hit_rate >= MIN_HIT_RATE_LINEAR {
            "PASS"
        } else {
            "FAIL"
        },
        row_lin.hit_rate * 100.0,
    );
    println!(
        "  ShardedCache hit rate >= {:.0}% : {} ({:.1}%)",
        MIN_HIT_RATE_SHARDED * 100.0,
        if row_sh.hit_rate >= MIN_HIT_RATE_SHARDED {
            "PASS"
        } else {
            "FAIL"
        },
        row_sh.hit_rate * 100.0,
    );
    // Recall: cached hits return results from a prior similar query, so
    // mean_recall < 1.0 is expected by design. The floor ensures we are
    // not returning completely uncorrelated results.
    println!(
        "  LinearCache mean recall >= {:.0}%  : {} ({:.3})",
        MIN_MEAN_RECALL * 100.0,
        if row_lin.mean_recall >= MIN_MEAN_RECALL { "PASS" } else { "FAIL" },
        row_lin.mean_recall,
    );
    println!(
        "  ShardedCache mean recall >= {:.0}% : {} ({:.3})",
        MIN_MEAN_RECALL * 100.0,
        if row_sh.mean_recall >= MIN_MEAN_RECALL { "PASS" } else { "FAIL" },
        row_sh.mean_recall,
    );

    // Effective speedup from caching.
    let lin_speedup = if row_lin.mean_us > 0.0 { row_no.mean_us / row_lin.mean_us } else { 0.0 };
    let sh_speedup = if row_sh.mean_us > 0.0 { row_no.mean_us / row_sh.mean_us } else { 0.0 };
    println!(
        "  LinearCache speedup >= {:.0}x       : {} ({:.2}x)",
        MIN_SPEEDUP,
        if lin_speedup >= MIN_SPEEDUP { "PASS" } else { "FAIL" },
        lin_speedup,
    );
    println!(
        "  ShardedCache speedup >= {:.0}x      : {} ({:.2}x)",
        MIN_SPEEDUP,
        if sh_speedup >= MIN_SPEEDUP { "PASS" } else { "FAIL" },
        sh_speedup,
    );

    let all_pass = row_lin.pass
        && row_sh.pass
        && lin_speedup >= MIN_SPEEDUP
        && sh_speedup >= MIN_SPEEDUP;
    println!();
    if all_pass {
        println!("✓ All acceptance criteria PASSED");
    } else {
        eprintln!("✗ One or more acceptance criteria FAILED");
        std::process::exit(1);
    }
}
