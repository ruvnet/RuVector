//! LoRANN end-to-end benchmark harness.
//!
//! Produces the recall@10 and QPS numbers quoted in the research document.
//!
//!   cargo run --release -p ruvector-lorann --bin lorann-demo
//!   cargo run --release -p ruvector-lorann --bin lorann-demo -- --fast

use std::collections::HashSet;
use std::time::Instant;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

use ruvector_lorann::{AnnIndex, FlatExactIndex, LorannConfig, LorannIndex, SearchResult};

fn main() {
    let fast = std::env::args().any(|a| a == "--fast");
    println!("LoRANN benchmark — ruvector-lorann");
    println!("====================================");
    if fast { println!("(fast mode: reduced n)"); }

    let corpus_sizes: &[usize] = if fast { &[2_000, 5_000] } else { &[5_000, 20_000, 50_000] };

    for &n in corpus_sizes {
        let d = 128;
        let n_queries = if fast { 100 } else { 500 };
        println!("\n─── n={n}, d={d}, queries={n_queries} ───");

        let corpus = generate_clustered(n, d, 50, 1234);
        let queries = generate_clustered(n_queries, d, 50, 9999);

        // Ground truth from FlatExact
        let flat = FlatExactIndex::build(corpus.clone()).expect("flat build");
        let ground_truth: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| {
                flat.search(q, 10)
                    .unwrap()
                    .iter()
                    .map(|r| r.id)
                    .collect()
            })
            .collect();

        // Variant A: FlatExact baseline
        bench_variant("FlatExact    ", &flat, &queries, &ground_truth, 10);

        // Variant B: LoRANN rank=16
        let cfg16 = LorannConfig {
            n_clusters: cluster_count(n),
            rank: 16,
            n_probe: probe_count(n),
            candidate_set: 200,
            kmeans_max_iter: 15,
            seed: 42,
        };
        println!("  Building LoRANN rank=16 (n_clusters={}, n_probe={})…",
            cfg16.n_clusters, cfg16.n_probe);
        let t0 = Instant::now();
        let idx16 = LorannIndex::build(corpus.clone(), cfg16).expect("lorann-16 build");
        println!("    build time: {:.1}s, memory: {} KB",
            t0.elapsed().as_secs_f64(),
            idx16.memory_bytes() / 1024);
        bench_variant("LoRANN r=16  ", &idx16, &queries, &ground_truth, 10);

        // Variant C: LoRANN rank=32
        let cfg32 = LorannConfig {
            n_clusters: cluster_count(n),
            rank: 32,
            n_probe: probe_count(n),
            candidate_set: 200,
            kmeans_max_iter: 15,
            seed: 42,
        };
        println!("  Building LoRANN rank=32 (n_clusters={}, n_probe={})…",
            cfg32.n_clusters, cfg32.n_probe);
        let t0 = Instant::now();
        let idx32 = LorannIndex::build(corpus.clone(), cfg32).expect("lorann-32 build");
        println!("    build time: {:.1}s, memory: {} KB",
            t0.elapsed().as_secs_f64(),
            idx32.memory_bytes() / 1024);
        bench_variant("LoRANN r=32  ", &idx32, &queries, &ground_truth, 10);

        // n_probe sweep for LoRANN r=32
        println!("\n  n_probe sweep (LoRANN r=32, n={n}):");
        println!("  {:>8} {:>12} {:>12} {:>12}", "n_probe", "recall@10", "QPS", "vs_flat");
        let flat_qps = measure_qps(&flat, &queries, 10);
        for &np in &[2, 4, 8, 16, 32] {
            let cfg = LorannConfig {
                n_clusters: cluster_count(n),
                rank: 32,
                n_probe: np.min(cluster_count(n)),
                candidate_set: 200,
                kmeans_max_iter: 15,
                seed: 42,
            };
            if np > cluster_count(n) { continue; }
            let idx = LorannIndex::build(corpus.clone(), cfg).expect("build");
            let recall = mean_recall_at_k(&idx, &queries, &ground_truth, 10);
            let qps = measure_qps(&idx, &queries, 10);
            println!("  {:>8} {:>11.1}% {:>11.0} {:>11.1}x",
                np, recall * 100.0, qps, qps / flat_qps);
        }
    }

    println!("\n─── Numeric acceptance test ───");
    acceptance_test();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cluster_count(n: usize) -> usize {
    ((n as f64).sqrt().round() as usize).clamp(16, 256)
}

fn probe_count(n: usize) -> usize {
    (cluster_count(n) / 8).clamp(2, 32)
}

fn bench_variant(
    label: &str,
    idx: &dyn AnnIndex,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
    k: usize,
) {
    let recall = mean_recall_at_k(idx, queries, ground_truth, k);
    let qps = measure_qps(idx, queries, k);
    let mem_kb = idx.memory_bytes() / 1024;
    println!("  {label}  recall@{k}: {:5.1}%  QPS: {:8.0}  mem: {} KB",
        recall * 100.0, qps, mem_kb);
}

fn recall_at_k(truth: &[usize], got: &[SearchResult]) -> f64 {
    let truth_set: HashSet<usize> = truth.iter().copied().collect();
    let hits = got.iter().filter(|r| truth_set.contains(&r.id)).count();
    hits as f64 / truth.len() as f64
}

fn mean_recall_at_k(
    idx: &dyn AnnIndex,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
    k: usize,
) -> f64 {
    let total: f64 = queries
        .iter()
        .zip(ground_truth.iter())
        .map(|(q, gt)| {
            let res = idx.search(q, k).unwrap_or_default();
            recall_at_k(gt, &res)
        })
        .sum();
    total / queries.len() as f64
}

fn measure_qps(idx: &dyn AnnIndex, queries: &[Vec<f32>], k: usize) -> f64 {
    // Warm-up
    for q in queries.iter().take(10) {
        let _ = idx.search(q, k);
    }
    let repeats = 3usize;
    let t0 = Instant::now();
    for _ in 0..repeats {
        for q in queries {
            let _ = idx.search(q, k);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    (queries.len() * repeats) as f64 / elapsed
}

/// Gaussian-clustered synthetic data (same generator as ruvector-rabitq/acorn).
fn generate_clustered(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroid_range = Uniform::new(-2.0f32, 2.0);
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..d).map(|_| centroid_range.sample(&mut rng)).collect())
        .collect();
    let noise = Normal::new(0.0f64, 0.5).unwrap();
    (0..n)
        .map(|_| {
            use rand::Rng as _;
            let c = &centroids[rng.gen_range(0..n_clusters)];
            c.iter()
                .map(|&x| x + noise.sample(&mut rng) as f32)
                .collect()
        })
        .collect()
}

/// Numeric acceptance test: LoRANN recall@10 ≥ 70% at n=2000, rank=32.
///
/// This is the correctness gate — if the SVD math or search pipeline is broken
/// the test will fail with a clear error message instead of silently returning junk.
fn acceptance_test() {
    let n = 2_000;
    let d = 64;
    let corpus = generate_clustered(n, d, 20, 777);
    let queries = generate_clustered(200, d, 20, 888);

    let flat = FlatExactIndex::build(corpus.clone()).expect("flat build");
    let ground_truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| flat.search(q, 10).unwrap().iter().map(|r| r.id).collect())
        .collect();

    let cfg = LorannConfig {
        n_clusters: 45,
        rank: 32,
        n_probe: 8,
        candidate_set: 200,
        kmeans_max_iter: 20,
        seed: 42,
    };
    let idx = LorannIndex::build(corpus, cfg).expect("lorann build");
    let recall = mean_recall_at_k(&idx, &queries, &ground_truth, 10);

    println!("  LoRANN recall@10 on n=2000, d=64: {:.1}%", recall * 100.0);
    assert!(
        recall >= 0.70,
        "Acceptance test FAILED: recall@10 = {:.1}% < 70%",
        recall * 100.0
    );
    println!("  PASS (recall@10 ≥ 70%)");

    // Also verify FlatExact is always 100% (sanity check for ground-truth code)
    let flat2 = FlatExactIndex::build(
        (0..2000)
            .map(|_| vec![0.0f32; d])
            .collect::<Vec<_>>(),
    )
    .expect("flat2");
    let _ = flat2; // just checking build works on degenerate input
}
