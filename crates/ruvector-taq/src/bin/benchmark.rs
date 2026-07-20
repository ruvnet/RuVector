//! TAQ benchmark: compare FullPrecision, UniformSQ8, UniformSQ4, and TAQ.
//!
//! Dataset: deterministic Gaussian-like vectors.
//! Metrics: recall@10, memory bytes, p50/p95/mean query latency.
//!
//! Run: cargo run --release -p ruvector-taq --bin benchmark

use ruvector_taq::{
    index::{FullPrecisionIndex, TaqIndex, UniformSq4Index, UniformSq8Index, VectorIndex},
    metrics::{ground_truth_knn, recall_at_k},
};
use std::time::{Duration, Instant};

// ── Dataset parameters ─────────────────────────────────────────────────────

const N_VECTORS: usize = 5_000;
const N_QUERIES: usize = 500;
const DIM: usize = 64;
const K: usize = 10;
const SEED_DATA: u64 = 0xDEAD_BEEF;
const SEED_QUERY: u64 = 0xCAFE_BABE;

// ── Deterministic LCG RNG ──────────────────────────────────────────────────

fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // High bits are better quality for LCG
    ((*state >> 33) as f32) / ((1u64 << 31) as f32) - 1.0 // [-1, 1)
}

/// Generate a dataset of n vectors with dim dimensions using Box-Muller pairs.
fn generate_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = Vec::with_capacity(dim);
            let mut i = 0;
            while i < dim {
                // Box-Muller: pairs of uniforms → Gaussian pair
                let u1 = (lcg_next(&mut state) * 0.5 + 0.5).max(1e-9);
                let u2 = lcg_next(&mut state) * 0.5 + 0.5;
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                v.push(r * theta.cos());
                if i + 1 < dim {
                    v.push(r * theta.sin());
                }
                i += 2;
            }
            v.truncate(dim);
            v
        })
        .collect()
}

// ── Timing utilities ───────────────────────────────────────────────────────

fn measure_search_latencies<I: VectorIndex>(
    index: &I,
    queries: &[Vec<f32>],
    k: usize,
) -> (Vec<Vec<usize>>, Vec<Duration>) {
    let mut results = Vec::with_capacity(queries.len());
    let mut latencies = Vec::with_capacity(queries.len());
    for q in queries {
        let t0 = Instant::now();
        let res = index.search(q, k);
        latencies.push(t0.elapsed());
        results.push(res);
    }
    (results, latencies)
}

fn percentile_us(mut lats: Vec<Duration>, p: f64) -> f64 {
    lats.sort_unstable();
    let idx = ((lats.len() as f64 * p).ceil() as usize).saturating_sub(1);
    lats[idx.min(lats.len() - 1)].as_micros() as f64
}

fn mean_us(lats: &[Duration]) -> f64 {
    let sum: u128 = lats.iter().map(|d| d.as_micros()).sum();
    sum as f64 / lats.len() as f64
}

fn throughput_qps(lats: &[Duration]) -> f64 {
    let sum_secs: f64 = lats.iter().map(|d| d.as_secs_f64()).sum();
    lats.len() as f64 / sum_secs
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  RuVector TAQ Benchmark — Topology-Aware Quantization   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("OS:        {}", std::env::consts::OS);
    println!("Arch:      {}", std::env::consts::ARCH);
    println!("Vectors:   {}", N_VECTORS);
    println!("Dims:      {}", DIM);
    println!("Queries:   {}", N_QUERIES);
    println!("K:         {}", K);
    println!("Seed data: 0x{:X}", SEED_DATA);
    println!();

    // ── Generate data ────────────────────────────────────────────────────
    eprint!("Generating dataset... ");
    let dataset = generate_dataset(N_VECTORS, DIM, SEED_DATA);
    let queries = generate_dataset(N_QUERIES, DIM, SEED_QUERY);
    eprintln!("done.");

    // ── Ground truth ─────────────────────────────────────────────────────
    eprint!("Computing exact ground truth... ");
    let t0 = Instant::now();
    let gt = ground_truth_knn(&queries, &dataset, K);
    let gt_elapsed = t0.elapsed();
    eprintln!("done in {:.2}s", gt_elapsed.as_secs_f64());

    // ── Build indices ────────────────────────────────────────────────────
    let builds: Vec<(
        &str,
        Box<dyn Fn() -> (String, Vec<Vec<usize>>, Vec<Duration>, usize)>,
    )> = vec![
        (
            "FullPrecision-f32",
            Box::new(|| {
                let idx = FullPrecisionIndex::build(dataset.clone(), DIM);
                let mem = idx.memory_bytes();
                let (res, lat) = measure_search_latencies(&idx, &queries, K);
                (idx.name().to_string(), res, lat, mem)
            }),
        ),
        (
            "UniformSQ8",
            Box::new(|| {
                let idx = UniformSq8Index::build(dataset.clone(), DIM);
                let mem = idx.memory_bytes();
                let (res, lat) = measure_search_latencies(&idx, &queries, K);
                (idx.name().to_string(), res, lat, mem)
            }),
        ),
        (
            "UniformSQ4",
            Box::new(|| {
                let idx = UniformSq4Index::build(dataset.clone(), DIM);
                let mem = idx.memory_bytes();
                let (res, lat) = measure_search_latencies(&idx, &queries, K);
                (idx.name().to_string(), res, lat, mem)
            }),
        ),
        (
            "TAQ-mixed(hub=SQ8,leaf=SQ4)",
            Box::new(|| {
                let idx = TaqIndex::build(dataset.clone(), DIM);
                let mem = idx.memory_bytes();
                let (res, lat) = measure_search_latencies(&idx, &queries, K);
                (idx.name().to_string(), res, lat, mem)
            }),
        ),
    ];

    let fp_mem_bytes = N_VECTORS * DIM * 4;

    println!("─────────────────────────────────────────────────────────────────────────────────────────────────────");
    println!(
        "{:<36} {:>9} {:>10} {:>10} {:>10} {:>11} {:>12} {:>9}",
        "Variant", "Recall@10", "Mean(μs)", "p50(μs)", "p95(μs)", "QPS", "Mem(KB)", "Mem%"
    );
    println!("─────────────────────────────────────────────────────────────────────────────────────────────────────");

    let mut taq_recall = 0.0f64;
    let mut taq_mem = 0usize;
    let mut sq8_recall = 0.0f64;
    let mut sq8_mem = 0usize;
    let mut sq4_recall = 0.0f64;
    let mut sq4_mem = 0usize;

    for (label, build_fn) in &builds {
        eprint!("Building {}... ", label);
        let (name, results, latencies, mem) = build_fn();
        eprintln!("done.");

        let recall = recall_at_k(&gt, &results, K);
        let mean = mean_us(&latencies);
        let p50 = percentile_us(latencies.clone(), 0.50);
        let p95 = percentile_us(latencies.clone(), 0.95);
        let qps = throughput_qps(&latencies);
        let mem_kb = mem / 1024;
        let mem_pct = mem as f64 / fp_mem_bytes as f64 * 100.0;

        println!(
            "{:<36} {:>9.4} {:>10.1} {:>10.1} {:>10.1} {:>11.0} {:>12} {:>8.1}%",
            name, recall, mean, p50, p95, qps, mem_kb, mem_pct
        );

        if name.contains("TAQ") {
            taq_recall = recall;
            taq_mem = mem;
        } else if name.contains("SQ8") {
            sq8_recall = recall;
            sq8_mem = mem;
        } else if name.contains("SQ4") {
            sq4_recall = recall;
            sq4_mem = mem;
        }
    }

    println!("─────────────────────────────────────────────────────────────────────────────────────────────────────");
    println!();

    // ── TAQ topology breakdown ──────────────────────────────────────────
    {
        let taq = TaqIndex::build(dataset.clone(), DIM);
        let hub_pct = taq.hub_count as f64 / N_VECTORS as f64 * 100.0;
        println!("TAQ Topology Breakdown:");
        println!("  Hub nodes (SQ8): {} ({:.1}%)", taq.hub_count, hub_pct);
        println!(
            "  Leaf nodes (SQ4): {} ({:.1}%)",
            taq.leaf_count,
            100.0 - hub_pct
        );
        println!(
            "  Avg bits/dim: {:.2}",
            (taq.hub_count as f64 * 8.0 + taq.leaf_count as f64 * 4.0) / N_VECTORS as f64
        );
    }

    println!();
    println!("Memory Math:");
    println!(
        "  f32 baseline:  {} KB = N * D * 4 bytes",
        fp_mem_bytes / 1024
    );
    println!(
        "  SQ8:           {} KB = N * D * 1 byte  (4x compression)",
        sq8_mem / 1024
    );
    println!(
        "  SQ4:           {} KB = N * D * 0.5 bytes (8x compression)",
        sq4_mem / 1024
    );
    println!(
        "  TAQ:           {} KB = hubs*D*1 + leaves*D*0.5 (mixed)",
        taq_mem / 1024
    );
    println!();

    // ── Acceptance test ─────────────────────────────────────────────────
    // TAQ must beat or match SQ4 recall while using ≤ SQ8 memory.
    println!("Acceptance Test:");
    let recall_pass = taq_recall >= sq4_recall;
    let mem_pass = taq_mem <= sq8_mem;
    let sq4_better = taq_recall >= sq4_recall - 0.001; // tolerance

    println!(
        "  [{}] TAQ recall@10 ({:.4}) >= UniformSQ4 recall@10 ({:.4})",
        if recall_pass { "PASS" } else { "FAIL" },
        taq_recall,
        sq4_recall
    );
    println!(
        "  [{}] TAQ memory ({} KB) <= UniformSQ8 memory ({} KB)",
        if mem_pass { "PASS" } else { "FAIL" },
        taq_mem / 1024,
        sq8_mem / 1024
    );
    println!(
        "  [INFO] TAQ vs SQ8 recall delta: {:.4}  (SQ8 is the high-precision baseline)",
        taq_recall - sq8_recall
    );
    println!(
        "  [INFO] TAQ vs SQ4 recall delta: {:.4}  (positive = TAQ wins on recall)",
        taq_recall - sq4_recall
    );

    let passed = recall_pass && mem_pass;
    println!();
    if passed {
        println!("ACCEPTANCE RESULT: PASS — TAQ achieves better recall than SQ4 at ≤ SQ8 memory.");
    } else {
        println!("ACCEPTANCE RESULT: FAIL — see diagnostics above.");
        std::process::exit(1);
    }
}
