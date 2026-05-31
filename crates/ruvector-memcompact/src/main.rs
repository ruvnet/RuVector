//! Agent Memory Compaction Benchmark
//!
//! Generates a synthetic agent memory store of 500 entries organised into 20
//! semantic clusters, then compacts to a budget of 25 entries using three
//! strategies and measures cluster-level recall@10.
//!
//! Usage:
//!   cargo run --release -p ruvector-memcompact
//!   cargo run --release -p ruvector-memcompact -- --n 1000 --clusters 40 --budget 50
//!
//! Environment variables:
//!   BENCH_N        override number of entries (default 500)
//!   BENCH_CLUSTERS override cluster count      (default 20)
//!   BENCH_BUDGET   override budget             (default 25)
//!   BENCH_DIM      override dimension          (default 64)
//!   BENCH_QUERIES  override query count        (default 50)

use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use ruvector_memcompact::{
    compaction::{AgeEviction, CompactionStrategy, GraphCutCompaction, ImportanceEviction},
    memory::{MemoryEntry, MemoryStore},
    metrics::{mean_recall_at_k, percentile},
};

// ── Dataset generation ───────────────────────────────────────────────────────

/// Generate one unit-normalised centroid per cluster.
fn gen_centroids(k: usize, dim: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    use rand::Rng;
    (0..k)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            normalize(&mut v);
            v
        })
        .collect()
}

fn normalize(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Build the synthetic memory store.
///
/// - `n` entries split evenly across `k` clusters.
/// - Each entry: centroid + Normal(0, sigma) noise, then L2-normalised.
/// - Timestamps are sequential (oldest entries belong to lowest cluster ids).
/// - Importance scores are uniform-random so ImportanceEviction samples
///   roughly uniformly across clusters.
fn generate_dataset(
    n: usize,
    k: usize,
    dim: usize,
    sigma: f32,
    seed: u64,
) -> (MemoryStore, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let centroids = gen_centroids(k, dim, &mut rng);
    // Per-dimension std = sigma/sqrt(dim) so that ‖ε‖ ≈ sigma ≈ 0.15.
    let per_dim_sigma = sigma / (dim as f32).sqrt();
    let noise_dist = Normal::new(0.0f32, per_dim_sigma).unwrap();

    let per_cluster = n / k;
    let mut store = MemoryStore::new();
    let mut id: u64 = 0;

    use rand::Rng;
    for (c_idx, centroid) in centroids.iter().enumerate() {
        for entry_in_cluster in 0..per_cluster {
            let mut v: Vec<f32> = centroid
                .iter()
                .map(|&x| x + noise_dist.sample(&mut rng))
                .collect();
            normalize(&mut v);

            // Sequential timestamps: entries in cluster 0 are oldest.
            let timestamp = (c_idx * per_cluster + entry_in_cluster) as u64;
            let importance: f32 = rng.gen(); // uniform [0, 1]

            let entry = MemoryEntry::new(id, v, timestamp, importance).with_cluster(c_idx as u64);
            store.push(entry);
            id += 1;
        }
    }

    (store, centroids)
}

/// Generate query vectors: one per centroid (perturbed slightly).
/// Uses the same sigma/√D scaling as the dataset generator.
fn generate_queries(centroids: &[Vec<f32>], sigma: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed + 9999);
    let dim = centroids.first().map(|c| c.len()).unwrap_or(64);
    let per_dim = sigma * 0.5 / (dim as f32).sqrt();
    let dist = Normal::new(0.0f32, per_dim).unwrap();
    centroids
        .iter()
        .map(|c| {
            let mut v: Vec<f32> = c.iter().map(|&x| x + dist.sample(&mut rng)).collect();
            normalize(&mut v);
            v
        })
        .collect()
}

// ── Latency helpers ──────────────────────────────────────────────────────────

/// Time a closure and return (result, elapsed_ns).
fn timed<T, F: FnOnce() -> T>(f: F) -> (T, u128) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_nanos())
}

/// Measure per-query search latency over `queries` against `store`.
fn query_latencies_ns(queries: &[Vec<f32>], store: &MemoryStore, k: usize) -> Vec<f64> {
    queries
        .iter()
        .map(|q| {
            let (_, ns) = timed(|| ruvector_memcompact::metrics::top_k(q, store, k));
            ns as f64
        })
        .collect()
}

// ── Formatting helpers ───────────────────────────────────────────────────────

fn ns_to_us(ns: f64) -> f64 {
    ns / 1_000.0
}
fn ns_to_ms(ns: u128) -> f64 {
    ns as f64 / 1_000_000.0
}
fn kb(bytes: usize) -> f64 {
    bytes as f64 / 1024.0
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    // ── Configuration ──────────────────────────────────────────────────
    let n: usize = env_usize("BENCH_N", 500);
    let k: usize = env_usize("BENCH_CLUSTERS", 20);
    let budget: usize = env_usize("BENCH_BUDGET", 25);
    let dim: usize = env_usize("BENCH_DIM", 64);
    let n_q: usize = env_usize("BENCH_QUERIES", 50);
    // σ is the desired *total* noise magnitude (fraction of centroid length).
    // We scale per-dimension to σ/√D so ‖ε‖ ≈ σ, keeping within-cluster
    // cosine similarity ≈ 1/(1+σ²) ≈ 0.978 >> threshold, enabling clean clustering.
    let sigma: f32 = 0.15;
    let krecall: usize = 10;
    let threshold: f32 = 0.70;

    // ── Environment ────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      ruvector-memcompact  |  Agent Memory Compaction Bench   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    println!("  OS:            {os} / {arch}");
    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(line) = info.lines().find(|l| l.starts_with("model name")) {
            if let Some(cpu) = line.splitn(2, ':').nth(1) {
                println!("  CPU:          {}", cpu.trim());
            }
        }
    }
    println!();

    // ── Dataset ────────────────────────────────────────────────────────
    println!("  Dataset configuration");
    println!("  ─────────────────────");
    println!("  Entries (N):   {n}");
    println!("  Dimensions:    {dim}");
    println!("  Clusters (K):  {k}  ({} entries/cluster)", n / k);
    println!(
        "  Budget:        {budget}  ({:.0}% of N, {} cluster reps if K≤budget)",
        budget as f64 / n as f64 * 100.0,
        budget
    );
    println!("  Noise σ:       {sigma}");
    println!("  Sim threshold: {threshold}");
    println!(
        "  Queries:       {n_q}  (one per cluster centroid ×{:.1})",
        n_q as f64 / k as f64
    );
    println!("  Recall@k:      {krecall}");
    println!();

    let (original, centroids) = generate_dataset(n, k, dim, sigma, 42);
    // Use the first n_q centroids as queries (n_q ≤ k).
    let q_centroids: Vec<Vec<f32>> = centroids.iter().take(n_q).cloned().collect();
    let queries = generate_queries(&q_centroids, sigma, 42);

    let orig_bytes = original.memory_bytes();

    // ── Strategies ─────────────────────────────────────────────────────
    let strategies: Vec<Box<dyn CompactionStrategy>> = vec![
        Box::new(AgeEviction),
        Box::new(ImportanceEviction),
        Box::new(GraphCutCompaction::with_threshold(threshold)),
    ];

    println!(
        "  {:22} {:>9} {:>8} {:>9} {:>9} {:>9} {:>11} {:>10} {:>9}",
        "Strategy",
        "Compacted",
        "Compact",
        "Query",
        "Query",
        "Query",
        "Memory",
        "Reduction",
        "Recall"
    );
    println!(
        "  {:22} {:>9} {:>8} {:>9} {:>9} {:>9} {:>11} {:>10} {:>9}",
        "", "entries", "ms", "mean μs", "p50 μs", "p95 μs", "KB", "%", "@10"
    );
    println!("  {}", "─".repeat(103));

    let mut gc_recall = 0.0f32;
    let mut age_recall = 0.0f32;

    for strategy in &strategies {
        let (compacted, compact_ns) = timed(|| strategy.compact(&original, budget));

        let recall = mean_recall_at_k(&queries, &original, &compacted, krecall);

        let mut q_latencies = query_latencies_ns(&queries, &compacted, krecall);
        q_latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let mean_q = ns_to_us(q_latencies.iter().sum::<f64>() / q_latencies.len() as f64);
        let p50_q = ns_to_us(percentile(&q_latencies, 0.50));
        let p95_q = ns_to_us(percentile(&q_latencies, 0.95));
        let comp_kb = kb(compacted.memory_bytes());
        let reduction = 100.0 * (1.0 - compacted.memory_bytes() as f64 / orig_bytes as f64);

        println!(
            "  {:<22} {:>9} {:>8.3} {:>9.2} {:>9.2} {:>9.2} {:>11.1} {:>9.1}% {:>8.1}%",
            strategy.name(),
            compacted.len(),
            ns_to_ms(compact_ns),
            mean_q,
            p50_q,
            p95_q,
            comp_kb,
            reduction,
            recall * 100.0,
        );

        match strategy.name() {
            "GraphCutCompaction" => gc_recall = recall,
            "AgeEviction" => age_recall = recall,
            _ => {}
        }
    }

    println!();
    println!(
        "  Original store: {} entries  ({:.1} KB)",
        original.len(),
        kb(orig_bytes)
    );
    println!();

    // ── Acceptance test ────────────────────────────────────────────────
    let gc_threshold = 0.75f32;
    let margin = 0.10f32; // GraphCut must beat AgeEviction by ≥ 10 pp

    let recall_pass = gc_recall >= gc_threshold;
    let margin_pass = gc_recall - age_recall >= margin;
    let all_pass = recall_pass && margin_pass;

    println!("  Acceptance criteria");
    println!("  ───────────────────");
    println!(
        "  GraphCutCompaction recall@{krecall} ≥ {:.0}%  →  {:.1}%  {}",
        gc_threshold * 100.0,
        gc_recall * 100.0,
        if recall_pass { "PASS ✓" } else { "FAIL ✗" }
    );
    println!(
        "  GraphCut beats AgeEviction by ≥ {:.0} pp  →  {:.1} pp  {}",
        margin * 100.0,
        (gc_recall - age_recall) * 100.0,
        if margin_pass { "PASS ✓" } else { "FAIL ✗" }
    );
    println!();
    println!(
        "  Overall: {}",
        if all_pass { "ACCEPT ✓" } else { "FAIL ✗" }
    );

    if !all_pass {
        std::process::exit(1);
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
