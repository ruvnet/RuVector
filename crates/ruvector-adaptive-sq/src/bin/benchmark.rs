//! Adaptive SQ benchmark: compare Uniform8, Uniform16, AdaptiveSQ
//! on a deterministic clustered dataset.
//!
//! Run:
//!   cargo run --release -p ruvector-adaptive-sq --bin benchmark
//!
//! Optional env vars:
//!   ASQ_N=5000          # dataset size  (default 5000)
//!   ASQ_DIM=32          # dimensions    (default 32)
//!   ASQ_Q=200           # query count   (default 200)
//!   ASQ_K=10            # top-k         (default 10)
//!   ASQ_SEED=42         # random seed   (default 42)

use ruvector_adaptive_sq::{
    coherence::density_scores,
    dataset::{generate, generate_queries, ground_truth, recall_at_k},
    index::{AdaptiveSqIndex, SqIndex, Uniform16Index, Uniform8Index},
};
use std::time::Instant;

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn bench_index<I: SqIndex>(
    idx: &I,
    queries: &[Vec<f32>],
    vectors: &[Vec<f32>],
    k: usize,
) -> (Vec<f32>, Vec<u128>) {
    let mut recalls = Vec::with_capacity(queries.len());
    let mut latencies_ns = Vec::with_capacity(queries.len());

    for q in queries {
        let gt = ground_truth(q, vectors);
        let t0 = Instant::now();
        let res = idx.search(q, k);
        let elapsed = t0.elapsed().as_nanos();
        latencies_ns.push(elapsed);
        recalls.push(recall_at_k(&res, &gt, k));
    }
    (recalls, latencies_ns)
}

fn main() {
    let n: usize = std::env::var("ASQ_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5000);
    let dim: usize = std::env::var("ASQ_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let n_queries: usize = std::env::var("ASQ_Q")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let k: usize = std::env::var("ASQ_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let seed: u64 = std::env::var("ASQ_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);

    // ── Header ────────────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!("  ruvector-adaptive-sq  Benchmark");
    println!("══════════════════════════════════════════════════════════════════");
    println!("  OS      : {}", std::env::consts::OS);
    println!("  Arch    : {}", std::env::consts::ARCH);
    println!("  Dataset : N={n}, dim={dim}");
    println!("  Clusters: 4 tight (σ=0.025), 6 loose (σ=0.30), 25% tight");
    println!("  Queries : {n_queries}");
    println!("  k       : {k}");
    println!("  Seed    : {seed}");
    println!("══════════════════════════════════════════════════════════════════");

    // ── Dataset ───────────────────────────────────────────────────────────
    let dataset = generate(n, dim, 4, 6, 0.25, 0.025, 0.30, seed);
    let queries = generate_queries(&dataset, n_queries, 0.01, seed.wrapping_add(1));
    println!(
        "\nDataset:  {} tight, {} loose vectors",
        dataset.tight_indices().len(),
        dataset.loose_indices().len()
    );

    // ── Build indices ──────────────────────────────────────────────────────
    println!("\nBuilding indices ...");

    let t0 = Instant::now();
    let u8_idx = Uniform8Index::build(&dataset.vectors, dim);
    println!("  Uniform8    built in {:>5} µs", t0.elapsed().as_micros());

    let t0 = Instant::now();
    let u16_idx = Uniform16Index::build(&dataset.vectors, dim);
    println!("  Uniform16   built in {:>5} µs", t0.elapsed().as_micros());

    let knn_k = 12usize;
    let threshold_factor = 0.60f32;
    let t0 = Instant::now();
    // Print density score progress since it dominates build time at large N.
    eprintln!("  [AdaptiveSQ] computing density scores (k={knn_k}, N={n}) ...");
    let scores = density_scores(&dataset.vectors, knn_k);
    let adp_idx = AdaptiveSqIndex::build(&dataset.vectors, dim, knn_k, threshold_factor);
    let build_us = t0.elapsed().as_micros();
    println!(
        "  AdaptiveSQ  built in {:>5} µs  (HP={:.1}%, LP={:.1}%)",
        build_us,
        adp_idx.hp_ratio() * 100.0,
        (1.0 - adp_idx.hp_ratio()) * 100.0
    );

    // ── Query benchmark ────────────────────────────────────────────────────
    println!("\nRunning {n_queries} queries (k={k}) ...");

    let (u8_recalls, mut u8_lat) = bench_index(&u8_idx, &queries, &dataset.vectors, k);
    let (u16_recalls, mut u16_lat) = bench_index(&u16_idx, &queries, &dataset.vectors, k);
    let (adp_recalls, mut adp_lat) = bench_index(&adp_idx, &queries, &dataset.vectors, k);

    u8_lat.sort_unstable();
    u16_lat.sort_unstable();
    adp_lat.sort_unstable();

    let mean_ns = |v: &[u128]| v.iter().sum::<u128>() / v.len() as u128;
    let to_us = |ns: u128| ns as f64 / 1_000.0;
    let qps = |ns: u128| {
        if ns == 0 {
            0.0
        } else {
            1_000_000_000.0 / ns as f64
        }
    };

    let mean_recall = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;

    let u8_mean = mean_ns(&u8_lat);
    let u16_mean = mean_ns(&u16_lat);
    let adp_mean = mean_ns(&adp_lat);

    let u8_recall = mean_recall(&u8_recalls);
    let u16_recall = mean_recall(&u16_recalls);
    let adp_recall = mean_recall(&adp_recalls);

    // ── Results table ──────────────────────────────────────────────────────
    println!();
    println!(
        "{:<12} │ {:>10} │ {:>9} │ {:>9} │ {:>9} │ {:>8} │ {:>10} │ {:>6}",
        "Variant", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Mem(KB)", "Recall@K", "HP%"
    );
    println!("{}", "─".repeat(90));

    let print_row = |name: &str, lat: &[u128], recall: f32, mem_b: usize, hp: f32| {
        let mean_us = to_us(mean_ns(lat));
        let p50_us = to_us(percentile(lat, 50.0));
        let p95_us = to_us(percentile(lat, 95.0));
        let qps_v = qps(mean_ns(lat));
        let mem_kb = mem_b as f64 / 1024.0;
        println!(
            "{:<12} │ {:>10.1} │ {:>9.1} │ {:>9.1} │ {:>9.0} │ {:>8.1} │ {:>10.4} │ {:>5.1}%",
            name,
            mean_us,
            p50_us,
            p95_us,
            qps_v,
            mem_kb,
            recall,
            hp * 100.0
        );
    };

    print_row("Uniform8", &u8_lat, u8_recall, u8_idx.memory_bytes(), 0.0);
    print_row(
        "Uniform16",
        &u16_lat,
        u16_recall,
        u16_idx.memory_bytes(),
        0.0,
    );
    print_row(
        "AdaptiveSQ",
        &adp_lat,
        adp_recall,
        adp_idx.memory_bytes(),
        adp_idx.hp_ratio(),
    );

    // ── Memory comparison ──────────────────────────────────────────────────
    println!();
    let u8_mem = u8_idx.memory_bytes();
    let u16_mem = u16_idx.memory_bytes();
    let adp_mem = adp_idx.memory_bytes();
    println!(
        "Memory vs Uniform16: AdaptiveSQ uses {:.1}% of 16-bit storage",
        adp_mem as f64 / u16_mem as f64 * 100.0
    );
    println!(
        "Memory vs Uniform8:  AdaptiveSQ uses {:.1}% of 8-bit storage",
        adp_mem as f64 / u8_mem as f64 * 100.0
    );

    // ── HP routing analysis ────────────────────────────────────────────────
    println!();
    let hp_threshold = {
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        mean * threshold_factor
    };
    let tight_ids = dataset.tight_indices();
    let tight_hp = tight_ids
        .iter()
        .filter(|&&id| scores[id] <= hp_threshold)
        .count();
    let loose_ids = dataset.loose_indices();
    let loose_lp = loose_ids
        .iter()
        .filter(|&&id| scores[id] > hp_threshold)
        .count();
    println!("Routing analysis (threshold_factor={threshold_factor}):");
    println!(
        "  Tight cluster vectors → HP : {}/{} = {:.1}%",
        tight_hp,
        tight_ids.len(),
        tight_hp as f64 / tight_ids.len() as f64 * 100.0
    );
    println!(
        "  Loose cluster vectors → LP : {}/{} = {:.1}%",
        loose_lp,
        loose_ids.len(),
        loose_lp as f64 / loose_ids.len() as f64 * 100.0
    );

    // ── Acceptance tests ───────────────────────────────────────────────────
    println!();
    println!("Acceptance tests:");

    let recall_threshold = 0.93 * u16_recall;
    let recall_pass = adp_recall >= recall_threshold;
    println!(
        "  [{}] Recall: AdaptiveSQ {:.4} ≥ 0.93 × U16 {:.4} = {:.4}",
        if recall_pass { "PASS" } else { "FAIL" },
        adp_recall,
        u16_recall,
        recall_threshold
    );

    let mem_limit = (u16_mem as f64 * 0.75) as usize;
    let mem_pass = adp_mem <= mem_limit;
    println!(
        "  [{}] Memory: AdaptiveSQ {} KB ≤ 75% of U16 {} KB = {} KB",
        if mem_pass { "PASS" } else { "FAIL" },
        adp_mem / 1024,
        u16_mem / 1024,
        mem_limit / 1024
    );

    println!();
    if recall_pass && mem_pass {
        println!("✓ All acceptance tests PASSED");
        std::process::exit(0);
    } else {
        eprintln!("✗ One or more acceptance tests FAILED");
        std::process::exit(1);
    }
}
