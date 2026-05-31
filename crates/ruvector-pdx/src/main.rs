//! PDX benchmark harness.
//!
//! Measures throughput (queries/sec), recall@10, and memory for three backends:
//!   1. RowMajorIndex — row-major brute-force baseline
//!   2. PdxFlatIndex  — PDX columnar layout, no pruning
//!   3. PdxPruneIndex — PDX columnar layout + exponential lower-bound pruning
//!
//! All measurements use the same clustered-Gaussian corpus and query set.
//!
//! Usage:
//!   cargo run --release -p ruvector-pdx
//!   cargo run --release -p ruvector-pdx -- --fast   (quick smoke test)

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashSet;
use std::time::Instant;

use ruvector_pdx::{AnnIndex, PdxFlatIndex, PdxPruneIndex, RowMajorIndex};

// ── data generation ───────────────────────────────────────────────────────────

/// Gaussian-clustered corpus: n_clusters centroids in [-1,1]^D, σ=0.5 per dim.
fn gen_corpus(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();
    let noise = Normal::new(0.0f64, 0.5).unwrap();
    (0..n)
        .map(|_| {
            let c = &centroids[rng.gen_range(0..n_clusters)];
            c.iter().map(|&x| x + noise.sample(&mut rng) as f32).collect()
        })
        .collect()
}

fn gen_queries(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 9999);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.5f32..1.5)).collect())
        .collect()
}

// ── ground truth ──────────────────────────────────────────────────────────────

fn exact_top_k(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f32 = query.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
            (i, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.total_cmp(&b.1));
    dists[..k].iter().map(|(i, _)| *i).collect()
}

fn recall_at_k(truth: &[usize], got: &[usize]) -> f64 {
    let truth_set: HashSet<usize> = truth.iter().copied().collect();
    let found = got.iter().filter(|id| truth_set.contains(id)).count();
    found as f64 / truth.len() as f64
}

// ── benchmark runner ──────────────────────────────────────────────────────────

struct BenchResult {
    label: String,
    n: usize,
    dim: usize,
    recall_at_10: f64,
    qps: f64,
    memory_mb: f64,
    build_ms: f64,
}

fn run_bench(
    index: &mut dyn AnnIndex,
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
    k: usize,
) -> BenchResult {
    let label = index.label().to_string();
    let n = corpus.len();
    let dim = index.dim();

    // Build
    let build_start = Instant::now();
    for (i, v) in corpus.iter().enumerate() {
        index.add(i, v.clone()).unwrap();
    }
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;

    let memory_mb = index.memory_bytes() as f64 / (1024.0 * 1024.0);

    // Warmup
    for q in queries.iter().take(5) {
        let _ = index.search(q, k).unwrap();
    }

    // Timed search
    let q_count = queries.len();
    let t0 = Instant::now();
    let mut total_recall = 0.0f64;
    for (qi, q) in queries.iter().enumerate() {
        let results = index.search(q, k).unwrap();
        let got: Vec<usize> = results.iter().map(|r| r.id).collect();
        total_recall += recall_at_k(&ground_truth[qi], &got);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let qps = q_count as f64 / elapsed;
    let recall_at_10 = total_recall / q_count as f64;

    BenchResult { label, n, dim, recall_at_10, qps, memory_mb, build_ms }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let fast = std::env::args().any(|a| a == "--fast");
    let k = 10;
    let n_queries = if fast { 50 } else { 200 };
    let n_clusters = 50;
    let block_size = 64; // matches u64 bitmask capacity in PdxPruneIndex

    let configs: &[(usize, usize)] = if fast {
        &[(5_000, 128), (5_000, 512)]
    } else {
        &[(10_000, 96), (10_000, 384), (50_000, 128), (50_000, 384)]
    };

    println!("PDX Columnar Vector Layout — Benchmark");
    println!("Hardware: x86_64 Linux, rustc --release, no hand-written SIMD");
    println!("Metric: recall@{k}, QPS, memory, build-time");
    println!("{:-<90}", "");
    println!(
        "{:<22} {:>7} {:>6} {:>10} {:>12} {:>10} {:>10}",
        "Variant", "n", "D", "Recall@10", "QPS", "Mem(MB)", "Build(ms)"
    );
    println!("{:-<90}", "");

    for &(n, dim) in configs {
        let corpus = gen_corpus(n, dim, n_clusters, 42);
        let queries = gen_queries(n_queries, dim, 42);

        // Ground truth (exact)
        let ground_truth: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| exact_top_k(&corpus, q, k))
            .collect();

        // Variant 1: RowMajorIndex
        let mut row = RowMajorIndex::new(dim, block_size);
        let r1 = run_bench(&mut row, &corpus, &queries, &ground_truth, k);

        // Variant 2: PdxFlatIndex (columnar, no pruning)
        let mut pdx = PdxFlatIndex::new(dim, block_size);
        let r2 = run_bench(&mut pdx, &corpus, &queries, &ground_truth, k);

        // Variant 3: PdxPruneIndex (columnar + lower-bound pruning)
        let first_check = (dim / 8).max(8).min(dim);
        let mut pdxp = PdxPruneIndex::new(dim, block_size, first_check);
        let r3 = run_bench(&mut pdxp, &corpus, &queries, &ground_truth, k);

        for r in [r1, r2, r3] {
            println!(
                "{:<22} {:>7} {:>6} {:>9.1}% {:>12.0} {:>10.3} {:>10.1}",
                r.label, r.n, r.dim, r.recall_at_10 * 100.0, r.qps, r.memory_mb, r.build_ms
            );
        }
        println!("{:-<90}", "");
    }

    println!("Done. See docs/research/nightly/2026-05-08-pdx-columnar-scan/README.md for analysis.");
}
