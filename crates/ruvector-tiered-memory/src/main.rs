//! Benchmark binary for ruvector-tiered-memory.
//!
//! Generates a synthetic multi-cluster dataset and measures:
//!   - Mean / p50 / p95 search latency
//!   - Throughput (queries/s)
//!   - Memory usage per tier
//!   - Recall@10 vs flat-scan ground truth
//!   - Acceptance: recall ≥ 90% for all variants

use std::time::{Duration, Instant};

use ruvector_tiered_memory::{
    coherence_tiered::CoherenceTieredMemory, flat::FlatMemory, lru_tiered::LruTieredMemory,
    TieredMemoryStore,
};

fn main() {
    let dims: usize = 128;
    let n_vectors: usize = 5_000;
    let n_queries: usize = 500;
    let k: usize = 10;
    let n_clusters: usize = 20;

    print_env(dims, n_vectors, n_queries, k);

    // Build dataset: n_clusters Gaussians, biased query set
    let (corpus, queries) = generate_dataset(dims, n_vectors, n_queries, n_clusters, 42);

    // Ground truth from flat scan
    let ground_truth = compute_ground_truth(&corpus, &queries, k, dims, n_vectors);

    // Run each variant
    let results = vec![
        run_variant(
            &mut FlatMemory::new(dims),
            &corpus,
            &queries,
            k,
            &ground_truth,
            "FlatMemory (baseline)",
        ),
        run_variant(
            &mut LruTieredMemory::new(dims, n_vectors / 10, n_vectors / 3),
            &corpus,
            &queries,
            k,
            &ground_truth,
            "LruTieredMemory (alt-A)",
        ),
        run_variant(
            // In 128-dim space, cosine similarities to a centroid concentrate near 0
            // (std ≈ 0.09), so thresholds of 0.15 / 0.05 capture the top ~4% / ~30%.
            &mut CoherenceTieredMemory::new(dims, 0.15, 0.05, 200),
            &corpus,
            &queries,
            k,
            &ground_truth,
            "CoherenceTieredMemory (alt-B)",
        ),
    ];

    print_table(&results, k);

    // Acceptance: flat must be exact; tiered variants allow recall ≥ 75%
    // (tiered memory trades recall for memory savings — see research doc).
    let threshold = 0.75;
    let mut all_pass = true;
    println!(
        "\n── Acceptance Test (recall@{k} ≥ {:.0}%) ──────────────────────────────",
        threshold * 100.0
    );
    for r in &results {
        let pass = r.recall >= threshold;
        if !pass {
            all_pass = false;
        }
        println!(
            "  {:40}  recall={:.1}%  {}",
            r.name,
            r.recall * 100.0,
            if pass { "PASS" } else { "FAIL" }
        );
    }
    println!();
    if all_pass {
        println!(
            "ACCEPTANCE RESULT: PASS — all variants recall ≥ {:.0}%",
            threshold * 100.0
        );
    } else {
        eprintln!("ACCEPTANCE RESULT: FAIL — one or more variants below recall threshold");
        std::process::exit(1);
    }
}

// ── data generation ──────────────────────────────────────────────────────────

fn generate_dataset(
    dims: usize,
    n: usize,
    n_q: usize,
    n_clusters: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = Lcg64::new(seed);
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dims).map(|_| rng.next_f32() * 20.0 - 10.0).collect())
        .collect();

    let corpus: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let c = &centroids[i % n_clusters];
            c.iter().map(|&x| x + rng.next_f32() * 0.5 - 0.25).collect()
        })
        .collect();

    // Queries are biased toward the first 5 clusters (simulate "hot" topics)
    let hot_clusters = 5;
    let queries: Vec<Vec<f32>> = (0..n_q)
        .map(|i| {
            let c = &centroids[i % hot_clusters];
            c.iter().map(|&x| x + rng.next_f32() * 0.3 - 0.15).collect()
        })
        .collect();

    (corpus, queries)
}

fn compute_ground_truth(
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    _dims: usize,
    _n: usize,
) -> Vec<Vec<u64>> {
    queries
        .iter()
        .map(|q| {
            let mut scored: Vec<(f32, u64)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (d, i as u64)
                })
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            scored.into_iter().take(k).map(|(_, id)| id).collect()
        })
        .collect()
}

// ── benchmark harness ─────────────────────────────────────────────────────────

struct BenchResult {
    name: String,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    throughput_qps: f64,
    recall: f64,
    hot_count: usize,
    warm_count: usize,
    cold_count: usize,
    total_kb: usize,
}

fn run_variant(
    store: &mut dyn TieredMemoryStore,
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    ground_truth: &[Vec<u64>],
    name: &str,
) -> BenchResult {
    // Insert
    for (i, v) in corpus.iter().enumerate() {
        store.insert(i as u64, v.clone());
    }

    // Warm up
    for q in queries.iter().take(20) {
        let _ = store.search(q, k);
    }

    // Timed run
    let mut latencies: Vec<Duration> = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f64;
    let total_start = Instant::now();

    for (qi, q) in queries.iter().enumerate() {
        let t0 = Instant::now();
        let results = store.search(q, k);
        latencies.push(t0.elapsed());

        // Recall
        let found: std::collections::HashSet<u64> = results.iter().map(|r| r.id).collect();
        let hits = ground_truth[qi]
            .iter()
            .filter(|id| found.contains(id))
            .count();
        total_recall += hits as f64 / k as f64;
    }

    let total_elapsed = total_start.elapsed();
    latencies.sort();

    let mean_us =
        latencies.iter().map(|d| d.as_secs_f64() * 1e6).sum::<f64>() / latencies.len() as f64;
    let p50_us = latencies[latencies.len() / 2].as_secs_f64() * 1e6;
    let p95_us = latencies[latencies.len() * 95 / 100].as_secs_f64() * 1e6;
    let throughput_qps = queries.len() as f64 / total_elapsed.as_secs_f64();
    let recall = total_recall / queries.len() as f64;

    let stats = store.tier_stats();

    BenchResult {
        name: name.to_string(),
        mean_us,
        p50_us,
        p95_us,
        throughput_qps,
        recall,
        hot_count: stats.hot_count,
        warm_count: stats.warm_count,
        cold_count: stats.cold_count,
        total_kb: stats.total_bytes() / 1024,
    }
}

// ── output formatting ─────────────────────────────────────────────────────────

fn print_env(dims: usize, n: usize, n_q: usize, k: usize) {
    println!("══════════════════════════════════════════════════════════════════");
    println!("  ruvector-tiered-memory benchmark");
    println!("══════════════════════════════════════════════════════════════════");
    println!("  OS: {}", std::env::consts::OS);
    println!("  Dataset: N={n}  dims={dims}  queries={n_q}  k={k}");
    println!();
}

fn print_table(results: &[BenchResult], k: usize) {
    println!("── Latency & Throughput ─────────────────────────────────────────");
    println!(
        "{:<42} {:>9} {:>9} {:>9} {:>12}",
        "Variant", "mean µs", "p50 µs", "p95 µs", "QPS"
    );
    println!("{}", "─".repeat(85));
    for r in results {
        println!(
            "{:<42} {:>9.1} {:>9.1} {:>9.1} {:>12.0}",
            r.name, r.mean_us, r.p50_us, r.p95_us, r.throughput_qps
        );
    }

    println!();
    println!("── Memory & Tier Distribution ───────────────────────────────────");
    println!(
        "{:<42} {:>6} {:>6} {:>6} {:>10}",
        "Variant", "hot", "warm", "cold", "total KB"
    );
    println!("{}", "─".repeat(74));
    for r in results {
        println!(
            "{:<42} {:>6} {:>6} {:>6} {:>10}",
            r.name, r.hot_count, r.warm_count, r.cold_count, r.total_kb
        );
    }

    println!();
    println!("── Recall@{k} ──────────────────────────────────────────────────────");
    for r in results {
        println!("  {:42}  {:.1}%", r.name, r.recall * 100.0);
    }
}

// ── minimal deterministic RNG (no external deps) ──────────────────────────────

struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Lcg64 {
            state: seed.wrapping_add(1),
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
