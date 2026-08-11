//! Benchmark: Hot-Warm-Cold Tiered Vector Quantization
//!
//! Measures recall@10, latency, throughput, and memory for three variants:
//!   1. FlatF32  – ground-truth brute-force scan, full f32 precision
//!   2. FlatU8   – all-quantized scalar u8 single tier
//!   3. HWC      – hot/warm/cold tiered with access-pattern-driven compaction

use ruvector_tiered_quant::{
    dataset::{generate_clustered_dataset, generate_dataset},
    recall_at_k, FlatF32Index, FlatU8Index, Hit, HotWarmColdIndex, TieredIndex,
};
use std::time::{Duration, Instant};

// ─── config ──────────────────────────────────────────────────────────────────

const N: usize = 10_000;
const DIMS: usize = 128;
const N_QUERIES: usize = 500;
const K: usize = 10;
const SEED: u64 = 31415926;
const HOT_THRESHOLD: u32 = 20;
const WARM_THRESHOLD: u32 = 5;

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    print_header();

    // For uniform random data with 60% cold (1-bit BQ at 128-dim), expected recall ≈ 0.40.
    println!(
        "\n=== Dataset: uniform random (n={N}, dims={DIMS}, queries={N_QUERIES}, k={K}) ===\n"
    );
    run_suite("uniform", 0.35, || generate_dataset(N, DIMS, N_QUERIES, SEED));

    // For clustered data with hot-biased queries, most queries hit hot/warm tier → higher recall.
    println!("\n=== Dataset: clustered (n={N}, dims={DIMS}, hot_cluster=1000, queries={N_QUERIES}) ===\n");
    run_suite("clustered", 0.70, || {
        generate_clustered_dataset(N, DIMS, N_QUERIES, 1_000, SEED)
    });

    println!("\n=== Small-n smoke (n=2000, dims=64) ===\n");
    run_suite("small", 0.35, || generate_dataset(2_000, 64, 200, 99));
}

// ─── suite ───────────────────────────────────────────────────────────────────

fn run_suite<F>(label: &str, hwc_recall_threshold: f32, make_data: F)
where
    F: Fn() -> (Vec<Vec<f32>>, Vec<Vec<f32>>),
{
    let (vecs, queries) = make_data();
    let n = vecs.len();
    let dims = vecs[0].len();

    // Build ground truth
    let mut gt_idx = FlatF32Index::new(dims);
    for (i, v) in vecs.iter().enumerate() {
        gt_idx.insert(i, v.clone());
    }

    // Precompute ground truth
    let ground_truths: Vec<Vec<Hit>> = queries.iter().map(|q| gt_idx.query(q, K)).collect();

    // ── Variant 1: FlatF32 ─────────────────────────────────────────────────
    {
        let mut idx = FlatF32Index::new(dims);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(i, v.clone());
        }
        let mem = n * dims * 4;
        let results = bench_flat_f32(&idx, &queries, &ground_truths);
        print_row(label, "FlatF32", n, dims, &results, mem, 1.000);
    }

    // ── Variant 2: FlatU8 ──────────────────────────────────────────────────
    {
        let mut idx = FlatU8Index::new(dims);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(i, v.clone());
        }
        // bytes + min(f32) + scale(f32) per vector
        let mem = n * (dims + dims * 4 * 2);
        let results = bench_flat_u8(&idx, &queries, &ground_truths);
        let recall = results.avg_recall;
        print_row(label, "FlatU8", n, dims, &results, mem, recall);
    }

    // ── Variant 3: HotWarmCold ─────────────────────────────────────────────
    {
        let mut idx = HotWarmColdIndex::new(dims, HOT_THRESHOLD, WARM_THRESHOLD);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(i, v.clone());
        }

        // Simulate access pattern: first 20% of vectors are "hot" (accessed many times)
        let hot_count = n / 5;
        for _ in 0..(HOT_THRESHOLD as usize + 5) {
            for id in 0..hot_count {
                idx.access(id);
            }
        }
        // Next 20% are "warm" (accessed a moderate number of times)
        let warm_count = n / 5;
        for _ in 0..(WARM_THRESHOLD as usize + 2) {
            for id in hot_count..(hot_count + warm_count) {
                idx.access(id);
            }
        }
        // Remaining 60% stay cold (no accesses)
        idx.compact();

        let stats = idx.stats();
        let mem = stats.total_bytes;
        let results = bench_hwc(&idx, &queries, &ground_truths);
        let recall = results.avg_recall;
        print_row(label, "HotWarmCold", n, dims, &results, mem, recall);

        println!(
            "  HWC tier distribution: hot={} ({:.1}%)  warm={} ({:.1}%)  cold={} ({:.1}%)",
            stats.hot_count,
            stats.hot_pct(),
            stats.warm_count,
            stats.warm_pct(),
            stats.cold_count,
            stats.cold_pct(),
        );
        println!(
            "  HWC compression ratio vs all-f32: {:.2}x",
            stats.compression_ratio(dims)
        );

        // Acceptance check — threshold varies by dataset type (see main comments).
        let pass = recall >= hwc_recall_threshold;
        println!(
            "  Acceptance (recall ≥ {hwc_recall_threshold:.2}): {} (got {:.3})",
            if pass { "PASS ✓" } else { "FAIL ✗" },
            recall
        );
        if !pass {
            std::process::exit(1);
        }
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

struct BenchResults {
    avg_recall: f32,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
}

fn bench_flat_f32(idx: &FlatF32Index, queries: &[Vec<f32>], gts: &[Vec<Hit>]) -> BenchResults {
    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f32;
    for (q, gt) in queries.iter().zip(gts.iter()) {
        let t0 = Instant::now();
        let hits = idx.query(q, K);
        latencies.push(t0.elapsed());
        total_recall += recall_at_k(gt, &hits, K);
    }
    summarize(latencies, total_recall / queries.len() as f32)
}

fn bench_flat_u8(idx: &FlatU8Index, queries: &[Vec<f32>], gts: &[Vec<Hit>]) -> BenchResults {
    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f32;
    for (q, gt) in queries.iter().zip(gts.iter()) {
        let t0 = Instant::now();
        let hits = idx.query(q, K);
        latencies.push(t0.elapsed());
        total_recall += recall_at_k(gt, &hits, K);
    }
    summarize(latencies, total_recall / queries.len() as f32)
}

fn bench_hwc(idx: &HotWarmColdIndex, queries: &[Vec<f32>], gts: &[Vec<Hit>]) -> BenchResults {
    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f32;
    for (q, gt) in queries.iter().zip(gts.iter()) {
        let t0 = Instant::now();
        let hits = idx.query(q, K);
        latencies.push(t0.elapsed());
        total_recall += recall_at_k(gt, &hits, K);
    }
    summarize(latencies, total_recall / queries.len() as f32)
}

fn summarize(mut latencies: Vec<Duration>, avg_recall: f32) -> BenchResults {
    latencies.sort();
    let n = latencies.len();
    let total_us: f64 = latencies.iter().map(|d| d.as_micros() as f64).sum();
    let mean_us = total_us / n as f64;
    let p50_us = latencies[n / 2].as_micros() as f64;
    let p95_us = latencies[(n as f64 * 0.95) as usize].as_micros() as f64;
    let total_secs = latencies.iter().map(|d| d.as_secs_f64()).sum::<f64>();
    let qps = n as f64 / total_secs;
    BenchResults {
        avg_recall,
        mean_us,
        p50_us,
        p95_us,
        qps,
    }
}

fn print_row(
    suite: &str,
    variant: &str,
    n: usize,
    dims: usize,
    r: &BenchResults,
    mem_bytes: usize,
    recall: f32,
) {
    let mem_mb = mem_bytes as f64 / 1024.0 / 1024.0;
    println!(
        "  [{suite}] {variant:<16} n={n:>6}  dims={dims:>4}  recall={recall:.3}  \
         mean={:.1}µs  p50={:.1}µs  p95={:.1}µs  QPS={:.0}  mem={mem_mb:.1}MB",
        r.mean_us, r.p50_us, r.p95_us, r.qps
    );
}

fn print_header() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-tiered-quant  –  HWC Tiered Vector Quantization       ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  OS:   {:<60}║", std::env::consts::OS);
    println!("║  Arch: {:<60}║", std::env::consts::ARCH);
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("Rust: {}", rust_version());
}

fn rust_version() -> &'static str {
    // Populated by build.rs or just a placeholder at compile time.
    env!("CARGO_PKG_VERSION")
}
