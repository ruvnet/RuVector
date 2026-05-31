//! RVQ benchmark harness: three variants measured on synthetic clustered data.
//!
//! Produces the numbers quoted in docs/research/nightly/2026-05-16-residual-vq/README.md
//!
//!   cargo run --release -p ruvector-residual-vq --bin rvq-demo
//!   cargo run --release -p ruvector-residual-vq --bin rvq-demo -- --fast

#![allow(clippy::needless_range_loop)]

use std::collections::HashSet;
use std::time::Instant;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

use ruvector_residual_vq::{AnnIndex, RvqBeamIndex, RvqEncoder, RvqGreedyIndex, RvqRerankIndex};

// ── Dataset ──────────────────────────────────────────────────────────────────

/// Gaussian-clustered synthetic data mirroring embedding distributions.
/// 100 clusters in [-3, 3]^D, σ=0.4 per cluster.
fn gen_clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let cr = Uniform::new(-3.0f32, 3.0);
    let noise = Normal::new(0.0f32, 0.4).unwrap();
    let centers: Vec<Vec<f32>> =
        (0..n_clusters).map(|_| (0..dim).map(|_| cr.sample(&mut rng)).collect()).collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % n_clusters];
            c.iter().map(|&x| x + noise.sample(&mut rng)).collect()
        })
        .collect()
}

// ── Brute-force baseline ──────────────────────────────────────────────────────

fn brute_force_top_k(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(id, v)| {
            let d: f32 = query.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
            (id, d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}

fn recall_at_k(results: &[ruvector_residual_vq::SearchResult], truth: &[usize]) -> f32 {
    let truth_set: HashSet<usize> = truth.iter().copied().collect();
    let hits = results.iter().filter(|r| truth_set.contains(&r.id)).count();
    hits as f32 / truth.len() as f32
}

// ── Measurement helpers ───────────────────────────────────────────────────────

struct BenchRow {
    label: String,
    n: usize,
    build_ms: f64,
    encode_us_per_vec: f64,
    search_qps: f64,
    recall10: f32,
    mem_mb: f64,
    bits_per_vec: usize,
    compress_ratio: f64,
}

fn measure_greedy(
    label: &str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    n_codebooks: usize,
    k_centroids: usize,
) -> BenchRow {
    let dim = vectors[0].len();
    let n = vectors.len();
    let bytes_orig = dim * 4;

    let t0 = Instant::now();
    let idx = RvqGreedyIndex::build(vectors, n_codebooks, k_centroids, 15);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Encode speed: re-encode all vectors outside the index
    let enc = RvqEncoder::train(vectors, n_codebooks, k_centroids, 15, 42);
    let t1 = Instant::now();
    let enc_n = n.min(2000);
    for v in &vectors[..enc_n] {
        let _ = enc.encode_greedy(v);
    }
    let encode_us = t1.elapsed().as_secs_f64() * 1e6 / enc_n as f64;

    // Search QPS
    let q_loops = (200 / queries.len().max(1)).max(1);
    let t2 = Instant::now();
    for _ in 0..q_loops {
        for q in queries {
            let _ = idx.search(q, 10);
        }
    }
    let qps = (q_loops * queries.len()) as f64 / t2.elapsed().as_secs_f64();

    // Recall
    let avg_recall = queries
        .iter()
        .zip(truth.iter())
        .map(|(q, t)| recall_at_k(&idx.search(q, 10), t))
        .sum::<f32>()
        / queries.len() as f32;

    let bits = n_codebooks * 8;
    let bytes_coded = bits / 8;

    BenchRow {
        label: label.to_string(),
        n,
        build_ms,
        encode_us_per_vec: encode_us,
        search_qps: qps,
        recall10: avg_recall,
        mem_mb: idx.memory_bytes() as f64 / 1e6,
        bits_per_vec: bits,
        compress_ratio: bytes_orig as f64 / bytes_coded as f64,
    }
}

fn measure_beam(
    label: &str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    n_codebooks: usize,
    k_centroids: usize,
    beam_width: usize,
) -> BenchRow {
    let dim = vectors[0].len();
    let n = vectors.len();
    let bytes_orig = dim * 4;

    let t0 = Instant::now();
    let idx = RvqBeamIndex::build(vectors, n_codebooks, k_centroids, 15, beam_width);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Encode speed (beam path)
    let enc = RvqEncoder::train(vectors, n_codebooks, k_centroids, 15, 42);
    let t1 = Instant::now();
    let enc_n = n.min(500);
    for v in &vectors[..enc_n] {
        let _ = enc.encode_beam(v, beam_width);
    }
    let encode_us = t1.elapsed().as_secs_f64() * 1e6 / enc_n as f64;

    // Search QPS (same ADC path as greedy)
    let q_loops = (200 / queries.len().max(1)).max(1);
    let t2 = Instant::now();
    for _ in 0..q_loops {
        for q in queries {
            let _ = idx.search(q, 10);
        }
    }
    let qps = (q_loops * queries.len()) as f64 / t2.elapsed().as_secs_f64();

    let avg_recall = queries
        .iter()
        .zip(truth.iter())
        .map(|(q, t)| recall_at_k(&idx.search(q, 10), t))
        .sum::<f32>()
        / queries.len() as f32;

    let bits = n_codebooks * 8;
    let bytes_coded = bits / 8;

    BenchRow {
        label: label.to_string(),
        n,
        build_ms,
        encode_us_per_vec: encode_us,
        search_qps: qps,
        recall10: avg_recall,
        mem_mb: idx.memory_bytes() as f64 / 1e6,
        bits_per_vec: bits,
        compress_ratio: bytes_orig as f64 / bytes_coded as f64,
    }
}

fn measure_rerank(
    label: &str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    n_codebooks: usize,
    k_centroids: usize,
    rerank_factor: usize,
) -> BenchRow {
    let dim = vectors[0].len();
    let n = vectors.len();
    let bytes_orig = dim * 4;

    let t0 = Instant::now();
    let idx = RvqRerankIndex::build(vectors, n_codebooks, k_centroids, 15, rerank_factor);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Encode speed (same as greedy)
    let enc = RvqEncoder::train(vectors, n_codebooks, k_centroids, 15, 42);
    let t1 = Instant::now();
    let enc_n = n.min(2000);
    for v in &vectors[..enc_n] {
        let _ = enc.encode_greedy(v);
    }
    let encode_us = t1.elapsed().as_secs_f64() * 1e6 / enc_n as f64;

    let q_loops = (200 / queries.len().max(1)).max(1);
    let t2 = Instant::now();
    for _ in 0..q_loops {
        for q in queries {
            let _ = idx.search(q, 10);
        }
    }
    let qps = (q_loops * queries.len()) as f64 / t2.elapsed().as_secs_f64();

    let avg_recall = queries
        .iter()
        .zip(truth.iter())
        .map(|(q, t)| recall_at_k(&idx.search(q, 10), t))
        .sum::<f32>()
        / queries.len() as f32;

    let bits = n_codebooks * 8;
    let bytes_coded = bits / 8;

    BenchRow {
        label: label.to_string(),
        n,
        build_ms,
        encode_us_per_vec: encode_us,
        search_qps: qps,
        recall10: avg_recall,
        mem_mb: idx.memory_bytes() as f64 / 1e6,
        bits_per_vec: bits,
        compress_ratio: bytes_orig as f64 / bytes_coded as f64,
    }
}

fn print_table(rows: &[BenchRow]) {
    println!(
        "\n{:<28} {:>7} {:>10} {:>12} {:>10} {:>9} {:>8} {:>10} {:>10}",
        "Variant", "N", "Build(ms)", "EncUs/vec", "SearchQPS", "Recall@10",
        "Mem(MB)", "Bits/vec", "Compress"
    );
    println!("{}", "-".repeat(108));
    for r in rows {
        println!(
            "{:<28} {:>7} {:>10.1} {:>12.2} {:>10.0} {:>9.3} {:>8.3} {:>10} {:>9.1}x",
            r.label, r.n, r.build_ms, r.encode_us_per_vec, r.search_qps,
            r.recall10, r.mem_mb, r.bits_per_vec, r.compress_ratio
        );
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let fast_mode = std::env::args().any(|a| a == "--fast");
    let sizes: &[usize] = if fast_mode { &[1_000, 5_000] } else { &[1_000, 10_000, 50_000] };
    let n_queries = if fast_mode { 20 } else { 100 };
    let dim = 128;
    let n_clusters = 100;

    println!("ruvector-residual-vq benchmark");
    println!("Hardware: {} logical CPUs", num_cpus());
    println!("Mode    : {}", if fast_mode { "fast (reduced)" } else { "full" });
    println!("Dim     : {dim}  Clusters: {n_clusters}  Queries: {n_queries}");
    println!("N codebooks: 8  K centroids: 64  k-means iters: 15");

    for &n in sizes {
        println!("\n══ N = {n} ══");
        let data = gen_clustered(n, dim, n_clusters, 1);
        let queries: Vec<Vec<f32>> = gen_clustered(n_queries, dim, n_clusters, 999);
        let truth: Vec<Vec<usize>> =
            queries.iter().map(|q| brute_force_top_k(&data, q, 10)).collect();

        let mut rows = Vec::new();

        // Variant A — RVQ-Greedy (beam=1), ADC scoring
        rows.push(measure_greedy(
            "RVQ-Greedy  (A)",
            &data,
            &queries,
            &truth,
            8,
            64,
        ));

        // Variant B — RVQ-Beam4, ADC scoring
        rows.push(measure_beam(
            "RVQ-Beam4   (B)",
            &data,
            &queries,
            &truth,
            8,
            64,
            4,
        ));

        // Variant C — RVQ-Rerank×5, greedy + exact L2 top-50
        rows.push(measure_rerank(
            "RVQ-Rerank×5 (C)",
            &data,
            &queries,
            &truth,
            8,
            64,
            5,
        ));

        print_table(&rows);

        // Summary for this N
        let a = &rows[0];
        let c = &rows[2];
        println!(
            "\n  Compression : {:.0}x vs raw f32 ({} bits/vec vs {} bits)",
            a.compress_ratio,
            a.bits_per_vec,
            dim * 32
        );
        println!(
            "  Rerank gain : +{:.1}% recall (C vs A)",
            (c.recall10 - a.recall10) * 100.0
        );
        rows.clear();
    }

    println!("\nDone.");
}

fn num_cpus() -> usize {
    // Simple detection without pulling in num_cpus crate
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}
