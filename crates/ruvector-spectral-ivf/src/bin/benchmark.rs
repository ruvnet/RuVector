//! Benchmark: Spectral IVF vs KMeans IVF — latency, throughput, recall.
//!
//! Run:  cargo run --release -p ruvector-spectral-ivf --bin benchmark
//!
//! Measures all three variants on a deterministic synthetic corpus and prints
//! a comparison table. All numbers come from actual Rust execution on this machine.

use ruvector_spectral_ivf::index::{AnnIndex, CoherenceSpectralIvf, KMeansIvf, SpectralIvf};
use std::time::Instant;

// ── Dataset parameters (change these to stress-test) ─────────────────────────
const N: usize = 800; // corpus size
const DIM: usize = 64; // vector dimension
const N_QUERIES: usize = 200;
const K: usize = 10; // top-k
const NPROBE: usize = 4; // partitions to probe
const N_PARTS: usize = 8; // partitions to build
const KNN_K: usize = 10; // kNN graph degree
                         // ─────────────────────────────────────────────────────────────────────────────

fn main() {
    print_header();

    let (corpus, queries) = gen_corpus(N, DIM, N_QUERIES);
    let ground_truth = brute_force_ground_truth(&corpus, &queries, K);

    let results = vec![
        bench_variant(
            "KMeansIvf",
            build_kmeans(&corpus),
            &corpus,
            &queries,
            &ground_truth,
            K,
            NPROBE,
        ),
        bench_variant(
            "SpectralIvf",
            build_spectral(&corpus),
            &corpus,
            &queries,
            &ground_truth,
            K,
            NPROBE,
        ),
        bench_variant(
            "CoherenceSpectralIvf",
            build_coherence(&corpus),
            &corpus,
            &queries,
            &ground_truth,
            K,
            NPROBE,
        ),
    ];

    print_results(&results, N, DIM, N_QUERIES, K, NPROBE);
    check_acceptance(&results);
}

// ── Builder functions ─────────────────────────────────────────────────────────

fn build_kmeans(corpus: &[Vec<f32>]) -> (Box<dyn AnnIndex>, u128) {
    let mut idx = KMeansIvf::new(N_PARTS);
    let t = Instant::now();
    idx.build(corpus);
    (Box::new(idx), t.elapsed().as_millis())
}

fn build_spectral(corpus: &[Vec<f32>]) -> (Box<dyn AnnIndex>, u128) {
    let mut idx = SpectralIvf::new(N_PARTS, KNN_K);
    let t = Instant::now();
    idx.build(corpus);
    (Box::new(idx), t.elapsed().as_millis())
}

fn build_coherence(corpus: &[Vec<f32>]) -> (Box<dyn AnnIndex>, u128) {
    let mut idx = CoherenceSpectralIvf::new(N_PARTS, KNN_K);
    let t = Instant::now();
    idx.build(corpus);
    (Box::new(idx), t.elapsed().as_millis())
}

// ── Benchmark one variant ─────────────────────────────────────────────────────

struct BenchResult {
    name: String,
    build_ms: u128,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    throughput_qps: f64,
    recall_at_k: f64,
    mem_bytes: usize,
}

fn bench_variant(
    name: &str,
    (idx, build_ms): (Box<dyn AnnIndex>, u128),
    _corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
    k: usize,
    nprobe: usize,
) -> BenchResult {
    let mut latencies_us: Vec<f64> = Vec::with_capacity(queries.len());
    let mut total_recall = 0usize;

    for (qi, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let results = idx.search(q, k, nprobe);
        latencies_us.push(t.elapsed().as_secs_f64() * 1e6);

        let result_ids: Vec<usize> = results.iter().map(|r| r.id).collect();
        total_recall += ground_truth[qi]
            .iter()
            .filter(|id| result_ids.contains(id))
            .count();
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = latencies_us.len() as f64;
    let mean_us = latencies_us.iter().sum::<f64>() / n;
    let p50_us = latencies_us[(latencies_us.len() as f64 * 0.50) as usize];
    let p95_us = latencies_us[(latencies_us.len() as f64 * 0.95) as usize];
    let throughput_qps = 1e6 / mean_us;
    let recall_at_k = total_recall as f64 / (queries.len() * k) as f64;
    let mem_bytes = idx.memory_bytes();

    BenchResult {
        name: name.to_string(),
        build_ms,
        mean_us,
        p50_us,
        p95_us,
        throughput_qps,
        recall_at_k,
        mem_bytes,
    }
}

// ── Brute-force ground truth ──────────────────────────────────────────────────

fn brute_force_ground_truth(
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(usize, f32)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_distance(q, v)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(k);
            dists.into_iter().map(|(i, _)| i).collect()
        })
        .collect()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        return 1.0;
    }
    1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
}

// ── Deterministic dataset generation ─────────────────────────────────────────

/// Generate `n` vectors in `dim` dimensions with `n_clusters` natural clusters,
/// plus `n_queries` query vectors drawn from the same distribution.
fn gen_corpus(n: usize, dim: usize, n_queries: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n_clusters = 8;
    let mut corpus = Vec::with_capacity(n + n_queries);

    for i in 0..(n + n_queries) {
        let cluster = i % n_clusters;
        // Cluster centroid: one-hot-ish in the first `n_clusters` dims.
        let mut v = vec![0.05f32; dim];
        v[cluster] = 1.0 + lcg(i as u64 * 7 + 1, 100) as f32 * 0.01;
        // Add small noise to other dims.
        for d in 0..dim {
            let noise = lcg(i as u64 * 1000 + d as u64, 100) as f32 * 0.005 - 0.25;
            v[d] += noise;
            // Clamp negative values to 0 to keep everything non-negative.
            v[d] = v[d].max(0.0);
        }
        corpus.push(v);
    }

    let queries = corpus.split_off(n);
    (corpus, queries)
}

fn lcg(seed: u64, n: usize) -> usize {
    let x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (x >> 33) as usize % n
}

// ── Output ────────────────────────────────────────────────────────────────────

fn print_header() {
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!(" ruvector-spectral-ivf  ·  Spectral vs k-Means IVF benchmark");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!(" Rust      : (run `rustc --version` for version info)");
    println!(" OS        : {}", std::env::consts::OS);
    println!(" Arch      : {}", std::env::consts::ARCH);
    println!(" N         : {N} vectors");
    println!(" Dim       : {DIM}");
    println!(" Queries   : {N_QUERIES}");
    println!(" K         : {K}");
    println!(" N_parts   : {N_PARTS}");
    println!(" nprobe    : {NPROBE}");
    println!();
}

fn print_results(
    results: &[BenchResult],
    n: usize,
    dim: usize,
    nq: usize,
    k: usize,
    nprobe: usize,
) {
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9}",
        "Variant", "Build(ms)", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Recall@K", "Mem(KB)"
    );
    println!("──────────────────────────────────────────────────────────────────────────────");
    for r in results {
        println!(
            "{:<24} {:>8} {:>8.1} {:>8.1} {:>9.1} {:>9.0} {:>8.3} {:>9.1}",
            r.name,
            r.build_ms,
            r.mean_us,
            r.p50_us,
            r.p95_us,
            r.throughput_qps,
            r.recall_at_k,
            r.mem_bytes as f64 / 1024.0,
        );
    }
    println!("──────────────────────────────────────────────────────────────────────────────");
    println!();
    println!("Dataset: n={n}, dim={dim}, queries={nq}, k={k}, nprobe={nprobe}, n_parts={N_PARTS}");
    println!("Memory estimate = raw f32 storage (vectors + representatives), no compression.");
    println!("Recall@K = fraction of true top-{k} found by ANN (brute-force ground truth).");
    println!();
}

fn check_acceptance(results: &[BenchResult]) {
    println!("── Acceptance check ──────────────────────────────────────────────────────────");
    let threshold = 0.60f64;
    let mut all_pass = true;
    for r in results {
        let pass = r.recall_at_k >= threshold;
        let mark = if pass { "PASS" } else { "FAIL" };
        println!(
            "  [{mark}] {}: recall@{K}={:.3} (threshold ≥ {threshold:.2})",
            r.name, r.recall_at_k
        );
        if !pass {
            all_pass = false;
        }
    }
    println!();
    if all_pass {
        println!("✓ All variants pass recall acceptance threshold.");
    } else {
        println!("✗ One or more variants failed recall acceptance. Check nprobe / n_parts.");
        std::process::exit(1);
    }
}
