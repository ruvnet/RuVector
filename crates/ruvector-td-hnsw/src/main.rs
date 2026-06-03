//! Benchmark binary for ruvector-td-hnsw.
//!
//! Run: cargo run --release -p ruvector-td-hnsw --bin td-benchmark
//!
//! Prints: dataset info, per-variant latency/throughput/fresh-recall, acceptance result.

use std::time::{Duration, Instant};

use rand::{Rng, SeedableRng};
use ruvector_td_hnsw::{DecayConfig, IndexVariant, TdIndex};

fn generate_dataset(n: usize, dims: usize, now_secs: u64, seed: u64) -> Vec<(u64, Vec<f32>, u64)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            // 70% old (> 24h), 20% medium (1-24h), 10% fresh (< 1h)
            let ts = match i % 10 {
                0 => now_secs - rng.gen_range(3601..86400),   // 1-24h
                1 => now_secs - rng.gen_range(0..3600),       // fresh
                _ => now_secs - rng.gen_range(86400..604800), // >24h old
            };
            (i as u64, v, ts)
        })
        .collect()
}

fn generate_queries(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 1);
    (0..n)
        .map(|_| (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect()
}

struct BenchResult {
    #[allow(dead_code)]
    variant: IndexVariant,
    n_vectors: usize,
    dims: usize,
    n_queries: usize,
    mean_latency_us: f64,
    p50_latency_us: f64,
    p95_latency_us: f64,
    throughput_qps: f64,
    memory_bytes: usize,
    fresh_recall: f64,
    #[allow(dead_code)]
    gated_fraction: f64,
}

fn bench_variant(
    variant: IndexVariant,
    config: DecayConfig,
    data: &[(u64, Vec<f32>, u64)],
    queries: &[Vec<f32>],
    now_secs: u64,
    k: usize,
    fresh_thresh_secs: u64,
) -> BenchResult {
    let dims = data[0].1.len();
    let mut idx = TdIndex::new(variant, config);

    for (id, vec, ts) in data {
        idx.insert(*id, vec.clone(), *ts);
    }

    // Memory estimate: vector floats + id + timestamp
    let bytes_per_entry = dims * 4 + 8 + 8;
    let memory_bytes = data.len() * bytes_per_entry;

    let mut latencies: Vec<Duration> = Vec::with_capacity(queries.len());
    let mut fresh_total = 0.0f64;
    let mut gated_total = 0usize;

    for q in queries {
        let t0 = Instant::now();
        let results = idx.search(q, k, now_secs);
        let elapsed = t0.elapsed();
        latencies.push(elapsed);

        let fresh = results
            .iter()
            .filter(|r| r.age_secs <= fresh_thresh_secs)
            .count();
        fresh_total += fresh as f64 / k as f64;

        // Count gated (missing from baseline k)
        gated_total += k.saturating_sub(results.len());
    }

    latencies.sort_unstable();
    let n = latencies.len();
    let sum: Duration = latencies.iter().sum();
    let mean_us = sum.as_micros() as f64 / n as f64;
    let p50_us = latencies[n / 2].as_micros() as f64;
    let p95_us = latencies[n * 95 / 100].as_micros() as f64;
    let throughput = n as f64 / sum.as_secs_f64();
    let fresh_recall = fresh_total / n as f64;
    let gated_fraction = gated_total as f64 / (n * k) as f64;

    BenchResult {
        variant,
        n_vectors: data.len(),
        dims,
        n_queries: n,
        mean_latency_us: mean_us,
        p50_latency_us: p50_us,
        p95_latency_us: p95_us,
        throughput_qps: throughput,
        memory_bytes,
        fresh_recall,
        gated_fraction,
    }
}

fn main() {
    let n_vectors = 10_000;
    let dims = 128;
    let n_queries = 1_000;
    let k = 10;
    let now_secs = 604800u64; // 7-day epoch
    let fresh_thresh_secs = 3600u64; // 1 hour = "fresh"
    let half_life_secs = 3600.0f64;
    let decay_strength = 3.0f32;

    // Print environment info
    println!("=== ruvector-td-hnsw Benchmark ===");
    println!("OS:          {}", std::env::consts::OS);
    println!("ARCH:        {}", std::env::consts::ARCH);
    println!("Rust:        {}", env!("CARGO_PKG_VERSION"));
    println!(
        "Crate:       ruvector-td-hnsw v{}",
        env!("CARGO_PKG_VERSION")
    );
    println!();
    println!("Dataset:     {} vectors, {} dims", n_vectors, dims);
    println!("Queries:     {}", n_queries);
    println!("Top-k:       {}", k);
    println!(
        "Half-life:   {}s ({:.1}h)",
        half_life_secs as u64,
        half_life_secs / 3600.0
    );
    println!("Decay str:   {:.1}", decay_strength);
    println!("Fresh thresh:{} s", fresh_thresh_secs);
    println!("Distribution: 70% old (>24h), 20% medium (1-24h), 10% fresh (<1h)");
    println!();

    let data = generate_dataset(n_vectors, dims, now_secs, 12345);
    let queries = generate_queries(n_queries, dims, 99999);

    let variants: Vec<(IndexVariant, DecayConfig, &str)> = vec![
        (
            IndexVariant::Baseline,
            DecayConfig::no_decay(),
            "Baseline (no decay)",
        ),
        (
            IndexVariant::TemporalDecay,
            DecayConfig::standard(decay_strength, half_life_secs),
            "TemporalDecay",
        ),
        (
            IndexVariant::CoherenceGated,
            DecayConfig::with_gate(decay_strength, half_life_secs, 43200.0, 1.5),
            "CoherenceGated",
        ),
    ];

    let mut results: Vec<(&str, BenchResult)> = Vec::new();
    for (variant, config, label) in &variants {
        print!("Benchmarking {}... ", label);
        use std::io::Write as _;
        std::io::stdout().flush().ok();
        let r = bench_variant(
            *variant,
            config.clone(),
            &data,
            &queries,
            now_secs,
            k,
            fresh_thresh_secs,
        );
        println!("done");
        results.push((label, r));
    }

    println!();
    println!("{:-<100}", "");
    println!(
        "{:<22} {:>8} {:>8} {:>8} {:>10} {:>12} {:>10} {:>10} {:>8}",
        "Variant", "N", "Dims", "Queries", "Mean µs", "Throughput", "p50 µs", "p95 µs", "FreshRec"
    );
    println!("{:-<100}", "");

    let baseline_fresh = results[0].1.fresh_recall;

    for (label, r) in &results {
        println!(
            "{:<22} {:>8} {:>8} {:>8} {:>10.1} {:>12.0} {:>10.1} {:>10.1} {:>8.3}",
            label,
            r.n_vectors,
            r.dims,
            r.n_queries,
            r.mean_latency_us,
            r.throughput_qps,
            r.p50_latency_us,
            r.p95_latency_us,
            r.fresh_recall,
        );
    }
    println!("{:-<100}", "");

    println!();
    println!(
        "Memory estimate per variant: {} KB ({} bytes/entry × {} entries)",
        results[0].1.memory_bytes / 1024,
        results[0].1.memory_bytes / n_vectors,
        n_vectors,
    );

    println!();
    println!("=== Acceptance Tests ===");
    let td_fresh = results[1].1.fresh_recall;
    let cg_fresh = results[2].1.fresh_recall;

    let pass_td = td_fresh > baseline_fresh;
    let pass_cg = cg_fresh >= td_fresh * 0.85; // gate may reduce k slightly

    println!(
        "[{}] TD fresh recall ({:.3}) > Baseline ({:.3})",
        if pass_td { "PASS" } else { "FAIL" },
        td_fresh,
        baseline_fresh
    );
    println!(
        "[{}] CG fresh recall ({:.3}) within 15% of TD ({:.3})",
        if pass_cg { "PASS" } else { "FAIL" },
        cg_fresh,
        td_fresh
    );

    let all_pass = pass_td && pass_cg;
    println!();
    if all_pass {
        println!("ACCEPTANCE: PASS — temporal decay improves fresh recall.");
    } else {
        println!("ACCEPTANCE: FAIL — see results above.");
        std::process::exit(1);
    }
}
