//! End-to-end benchmark binary for ruvector-lvq.
//!
//! Generates a synthetic dataset, builds three indexes (fp32 baseline,
//! LVQ-8, LVQ-8x8 with reranking), and reports memory + latency + recall
//! against the fp32 ground truth. The numbers printed here are the ones
//! pasted verbatim into the research document.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ruvector_lvq::{FlatF32, FlatLvqIndex, IndexKind, LvqError};

fn main() -> Result<(), LvqError> {
    let dim: usize = std::env::var("LVQ_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let n: usize = std::env::var("LVQ_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    let nq: usize = std::env::var("LVQ_NQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let k: usize = std::env::var("LVQ_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    println!("== ruvector-lvq bench ==");
    println!("dim = {dim}, n = {n}, nq = {nq}, k = {k}");

    // Synthetic dataset: cluster mixture so distances are non-trivial.
    let (data, queries) = make_clustered_dataset(n, nq, dim, 42);

    // Ground truth: fp32 brute force.
    let mut gt = FlatF32::new(dim);
    let t = Instant::now();
    for v in data.chunks_exact(dim) {
        gt.push(v)?;
    }
    println!(
        "fp32 build:           {:>8.2} ms   {:>10} bytes",
        t.elapsed().as_secs_f64() * 1e3,
        gt.byte_size()
    );

    // LVQ-8.
    let mut lvq8 = FlatLvqIndex::new_lvq8(dim);
    let t = Instant::now();
    lvq8.extend_from_flat(&data)?;
    println!(
        "LVQ-8 build:          {:>8.2} ms   {:>10} bytes",
        t.elapsed().as_secs_f64() * 1e3,
        lvq8.byte_size()
    );

    // LVQ-8x8.
    let mut lvq8x8 = FlatLvqIndex::new_lvq8x8(dim);
    let t = Instant::now();
    lvq8x8.extend_from_flat(&data)?;
    println!(
        "LVQ-8x8 build:        {:>8.2} ms   {:>10} bytes",
        t.elapsed().as_secs_f64() * 1e3,
        lvq8x8.byte_size()
    );

    // Search.
    let truth = run_search(&queries, dim, k, |q, k| gt.search_l2(q, k).unwrap());

    println!();
    println!(
        "{:<28} {:>10} {:>10} {:>10}",
        "variant", "lat ms", "qps", "recall@10"
    );

    bench("fp32 (ground truth)", &queries, dim, k, &truth, |q, k| {
        gt.search_l2(q, k).unwrap()
    });

    bench("LVQ-8", &queries, dim, k, &truth, |q, k| {
        lvq8.search_l2(q, k).unwrap()
    });

    bench("LVQ-8x8 (full scan)", &queries, dim, k, &truth, |q, k| {
        lvq8x8.search_l2(q, k).unwrap()
    });

    bench(
        "LVQ-8x8 (rerank, 5x)",
        &queries,
        dim,
        k,
        &truth,
        |q, k| lvq8x8.search_l2_reranked(q, k, k * 5).unwrap(),
    );

    bench(
        "LVQ-8x8 (rerank, 10x)",
        &queries,
        dim,
        k,
        &truth,
        |q, k| lvq8x8.search_l2_reranked(q, k, k * 10).unwrap(),
    );

    println!();
    println!(
        "memory savings: fp32={:.2} MB  lvq8={:.2} MB  lvq8x8={:.2} MB",
        gt.byte_size() as f64 / 1.048_576e6,
        lvq8.byte_size() as f64 / 1.048_576e6,
        lvq8x8.byte_size() as f64 / 1.048_576e6
    );
    println!(
        "lvq8 / fp32 ratio:    {:.3}",
        lvq8.byte_size() as f64 / gt.byte_size() as f64
    );
    println!(
        "lvq8x8 / fp32 ratio:  {:.3}",
        lvq8x8.byte_size() as f64 / gt.byte_size() as f64
    );

    println!();
    println!("kind discriminants exposed: {:?}", IndexKind::Lvq8x8);
    Ok(())
}

fn make_clustered_dataset(
    n: usize,
    nq: usize,
    dim: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_clusters = 32;
    let mut centers = Vec::with_capacity(n_clusters * dim);
    for _ in 0..n_clusters * dim {
        centers.push(rng.gen_range(-1.0_f32..1.0));
    }

    let mut data = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = rng.gen_range(0..n_clusters);
        for d in 0..dim {
            let center = centers[c * dim + d];
            data.push(center + rng.gen_range(-0.15_f32..0.15));
        }
    }

    let mut queries = Vec::with_capacity(nq * dim);
    for _ in 0..nq {
        let c = rng.gen_range(0..n_clusters);
        for d in 0..dim {
            let center = centers[c * dim + d];
            queries.push(center + rng.gen_range(-0.20_f32..0.20));
        }
    }
    (data, queries)
}

type Hits = Vec<ruvector_lvq::SearchHit>;

fn run_search<F: FnMut(&[f32], usize) -> Hits>(
    queries: &[f32],
    dim: usize,
    k: usize,
    mut f: F,
) -> Vec<Vec<u32>> {
    queries
        .chunks_exact(dim)
        .map(|q| f(q, k).into_iter().map(|h| h.id).collect())
        .collect()
}

fn bench<F: FnMut(&[f32], usize) -> Hits>(
    label: &str,
    queries: &[f32],
    dim: usize,
    k: usize,
    truth: &[Vec<u32>],
    mut f: F,
) {
    // Warmup.
    for q in queries.chunks_exact(dim).take(8) {
        let _ = f(q, k);
    }

    let mut total_hits = 0usize;
    let total_queries = queries.len() / dim;
    let t = Instant::now();
    for (i, q) in queries.chunks_exact(dim).enumerate() {
        let approx: Vec<u32> = f(q, k).into_iter().map(|h| h.id).collect();
        for id in &approx {
            if truth[i].contains(id) {
                total_hits += 1;
            }
        }
    }
    let elapsed = t.elapsed().as_secs_f64();
    let lat_ms = elapsed * 1e3 / total_queries as f64;
    let qps = total_queries as f64 / elapsed;
    let recall = total_hits as f64 / (k * total_queries) as f64;
    println!(
        "{:<28} {:>10.3} {:>10.0} {:>10.3}",
        label, lat_ms, qps, recall
    );
}
