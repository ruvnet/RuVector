//! RaBitQ benchmark binary — produces real timing and recall numbers.
//!
//! Runs three backends on Gaussian-cluster data (which mimics real embedding
//! distributions like SIFT, GloVe, or OpenAI text-embedding-3):
//!
//!   A) FlatF32Index    — exact brute-force baseline
//!   B) RabitqIndex     — 1-bit angular scan, no reranking
//!   C) RabitqPlusIndex — 1-bit scan + exact top-K reranking (variable factor)
//!
//! Key insight: on clustered data RaBitQ's XNOR-popcount scan quickly identifies
//! the right neighbourhood, then exact reranking lifts recall to near-100%.
//! At n=5K the rerank cost is small; at n=100K the 17.5x memory saving matters.
//!
//! Usage: cargo run --release -p ruvector-rabitq

use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::collections::HashSet;
use std::time::Instant;

use ruvector_rabitq::index::{AnnIndex, FlatF32Index, RabitqIndex, RabitqPlusIndex};

/// Gaussian-clustered data mimicking real embedding distributions.
///
/// Pure uniform Gaussian in D=128 suffers from distance concentration (all pairwise
/// distances nearly equal). Clustered data with std ≈ 15% of centroid spread gives
/// the structure that binary quantization can exploit, matching workloads like SIFT,
/// GloVe, OpenAI text-embedding-3, or other structured dense vector spaces.
fn generate_clustered(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::Rng as _;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroid_range = Uniform::new(-2.0f32, 2.0);
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..d).map(|_| centroid_range.sample(&mut rng)).collect())
        .collect();
    // std=0.6 gives ~15% noise relative to centroid spread [-2,2]:
    // enough separation that k-NN structure is clear at D=128.
    let noise = Normal::new(0.0f64, 0.6).unwrap();
    (0..n)
        .map(|_| {
            let c = &centroids[rng.gen_range(0..n_clusters)];
            c.iter()
                .map(|&x| x + noise.sample(&mut rng) as f32)
                .collect()
        })
        .collect()
}

fn recall_at_k(truth: &[usize], got: &[usize]) -> f64 {
    let truth_set: HashSet<usize> = truth.iter().copied().collect();
    got.iter().filter(|id| truth_set.contains(id)).count() as f64 / truth.len() as f64
}

fn run_search<I: AnnIndex>(
    label: &str,
    index: &I,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
    k: usize,
) -> f64 {
    let t = Instant::now();
    let mut total_recall = 0.0f64;
    for (i, q) in queries.iter().enumerate() {
        let res = index.search(q, k).unwrap();
        let ids: Vec<usize> = res.into_iter().map(|r| r.id).collect();
        total_recall += recall_at_k(&ground_truth[i], &ids);
    }
    let nq = queries.len();
    let elapsed = t.elapsed();
    let qps = nq as f64 / elapsed.as_secs_f64();
    let recall = total_recall / nq as f64;
    let mb = index.memory_bytes() as f64 / 1_048_576.0;
    println!(
        "  [{label:<22}] recall@{k}={:5.1}%  QPS={:6.0}  mem={:5.1}MB  lat={:.3}ms",
        recall * 100.0,
        qps,
        mb,
        elapsed.as_secs_f64() / nq as f64 * 1000.0,
    );
    recall
}

fn main() {
    let d = 128usize;
    let k = 10usize;
    let n_clusters = 100usize;
    let seed = 42u64;

    println!("=== RaBitQ Nightly Benchmark ===");
    println!("d={d}  k={k}  clusters={n_clusters}  data=Gaussian-cluster (std=0.6)");
    println!("CPU arch: {}", std::env::consts::ARCH);
    println!();

    // ── Experiment 1: recall vs rerank factor at n=5K ──────────────────────────
    {
        let n = 5_000;
        let nq = 200;
        println!("── Exp 1: recall vs rerank factor  (n={n}, nq={nq}) ──");

        let all = generate_clustered(n + nq, d, n_clusters, seed);
        let (db, q) = all.split_at(n);
        let db = db.to_vec();
        let queries = q.to_vec();

        let mut exact_idx = FlatF32Index::new(d);
        for (id, v) in db.iter().enumerate() {
            exact_idx.add(id, v.clone()).unwrap();
        }

        let ground_truth: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| {
                exact_idx
                    .search(q, k)
                    .unwrap()
                    .into_iter()
                    .map(|r| r.id)
                    .collect()
            })
            .collect();

        run_search("FlatF32 (exact)", &exact_idx, &queries, &ground_truth, k);

        let mut rq_idx = RabitqIndex::new(d, seed);
        for (id, v) in db.iter().enumerate() {
            rq_idx.add(id, v.clone()).unwrap();
        }
        run_search("RaBitQ 1-bit (no rerank)", &rq_idx, &queries, &ground_truth, k);

        for &factor in &[2usize, 5, 10, 20] {
            let mut idx = RabitqPlusIndex::new(d, seed, factor);
            for (id, v) in db.iter().enumerate() {
                idx.add(id, v.clone()).unwrap();
            }
            let label = format!("RaBitQ+ rerank×{factor}");
            run_search(&label, &idx, &queries, &ground_truth, k);
        }
        println!();
    }

    // ── Experiment 2: throughput at n=50K ──────────────────────────────────────
    {
        let n = 50_000;
        let nq = 500;
        println!("── Exp 2: throughput & memory at n={n} ──");

        let t_gen = Instant::now();
        let all = generate_clustered(n + nq, d, n_clusters, seed + 1);
        println!("  Data generation: {:.2}s", t_gen.elapsed().as_secs_f64());

        let (db, q) = all.split_at(n);
        let db = db.to_vec();
        let queries = q.to_vec();

        let t_build = Instant::now();
        let mut exact_idx = FlatF32Index::new(d);
        let mut rq_idx = RabitqIndex::new(d, seed);
        let mut rq_plus10 = RabitqPlusIndex::new(d, seed, 10);
        for (id, v) in db.iter().enumerate() {
            exact_idx.add(id, v.clone()).unwrap();
            rq_idx.add(id, v.clone()).unwrap();
            rq_plus10.add(id, v.clone()).unwrap();
        }
        println!("  Index build:     {:.2}s", t_build.elapsed().as_secs_f64());

        let ground_truth: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| {
                exact_idx
                    .search(q, k)
                    .unwrap()
                    .into_iter()
                    .map(|r| r.id)
                    .collect()
            })
            .collect();

        println!();
        run_search("FlatF32 (exact)", &exact_idx, &queries, &ground_truth, k);
        run_search("RaBitQ 1-bit", &rq_idx, &queries, &ground_truth, k);
        run_search("RaBitQ+ rerank×10", &rq_plus10, &queries, &ground_truth, k);

        println!();
        let f32_mb = exact_idx.memory_bytes() as f64 / 1e6;
        let rq_mb = rq_idx.memory_bytes() as f64 / 1e6;
        println!(
            "  Memory: FlatF32={:.1}MB  RaBitQ-codes={:.1}MB  compression={:.1}x",
            f32_mb,
            rq_mb,
            f32_mb / rq_mb
        );
        println!(
            "  Bytes/vec: f32={:.0}  binary-code={:.0}  (D={d} → {} u64 words)",
            exact_idx.memory_bytes() as f64 / n as f64,
            rq_idx.memory_bytes() as f64 / n as f64,
            (d + 63) / 64
        );
    }
}
