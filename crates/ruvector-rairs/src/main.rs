//! rairs-demo — end-to-end benchmark for all three RAIRS variants.
//!
//! Generates a synthetic Gaussian corpus (configurable), trains each index,
//! measures:
//!   - recall@10 (fraction of true top-10 neighbours found)
//!   - query throughput (QPS)
//!   - index memory (bytes estimated from list entry counts)
//!
//! across nprobe ∈ {1, 4, 16, 32, 64, full}.

use std::collections::HashSet;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ruvector_rairs::index::l2sq;
use ruvector_rairs::{AnnIndex, IvfFlat, RairsSeil, RairsStrict};

// ─── configuration ────────────────────────────────────────────────────────────

const N: usize = 5_000; // corpus size
const DIM: usize = 128; // vector dimensionality
const NCLUSTERS: usize = 64; // IVF list count
const NQUERIES: usize = 200; // evaluation queries
const K: usize = 10; // recall@K
const KMEANS_ITER: usize = 25;
const SEED: u64 = 42;

// ─── helpers ─────────────────────────────────────────────────────────────────

fn random_corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    // Multi-cluster Gaussian for a more realistic distribution
    let ncenters = 20usize;
    let centers: Vec<Vec<f32>> = (0..ncenters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-5.0f32..5.0)).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % ncenters];
            c.iter().map(|&x| x + rng.gen_range(-0.5f32..0.5)).collect()
        })
        .collect()
}

/// Brute-force exact top-k IDs for a query.
fn exact_topk(query: &[f32], corpus: &[Vec<f32>], k: usize) -> HashSet<usize> {
    let mut dists: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2sq(query, v)))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(k).map(|(id, _)| *id).collect()
}

/// Measure recall@K for `results` vs ground truth `gt`.
fn recall_at_k(results: &[ruvector_rairs::SearchResult], gt: &HashSet<usize>) -> f64 {
    let hits = results.iter().filter(|r| gt.contains(&r.id)).count();
    hits as f64 / gt.len() as f64
}

/// Estimate memory used by an IvfFlat index (bytes).
fn ivf_memory_bytes(idx: &IvfFlat) -> usize {
    // centroids: nclusters × dim × 4 bytes
    let centroid_bytes = idx.num_lists() * DIM * 4;
    // list entries: (8 bytes id + dim×4 bytes vector) × total
    let entry_bytes = idx.len() * (8 + DIM * 4);
    centroid_bytes + entry_bytes
}

fn rairs_strict_memory_bytes(idx: &RairsStrict) -> usize {
    let centroid_bytes = idx.num_lists() * DIM * 4;
    // With dual assignment, total entries ≤ 2×N
    let entry_bytes = idx.len() * 2 * (8 + DIM * 4); // upper bound
    centroid_bytes + entry_bytes
}

fn rairs_seil_memory_bytes(idx: &RairsSeil) -> usize {
    let centroid_bytes = idx.num_lists() * DIM * 4;
    // SEIL stores each vector once regardless of list count
    let entry_bytes = idx.len() * (8 + DIM * 4);
    centroid_bytes + entry_bytes
}

// ─── benchmark one variant ───────────────────────────────────────────────────

fn bench<Idx: AnnIndex>(
    name: &str,
    idx: &Idx,
    queries: &[Vec<f32>],
    ground_truth: &[HashSet<usize>],
    nprobe_values: &[usize],
    memory_bytes: usize,
) {
    println!(
        "\n── {name} (memory ≈ {:.1} KB) ──",
        memory_bytes as f64 / 1024.0
    );
    println!("{:<10} {:>12} {:>12}", "nprobe", "recall@10", "QPS");

    for &np in nprobe_values {
        let np = np.min(idx.num_lists());
        let t0 = Instant::now();
        let mut total_recall = 0.0f64;
        for (qi, q) in queries.iter().enumerate() {
            let results = idx.search(q, K, np).expect("search failed");
            total_recall += recall_at_k(&results, &ground_truth[qi]);
        }
        let elapsed = t0.elapsed();
        let recall = total_recall / queries.len() as f64;
        let qps = queries.len() as f64 / elapsed.as_secs_f64();
        println!("{:<10} {:>11.1}% {:>12.0}", np, recall * 100.0, qps);
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("ruvector-rairs benchmark");
    println!("═══════════════════════════════════════");
    println!("corpus N={N}  dim={DIM}  clusters={NCLUSTERS}  queries={NQUERIES}  K={K}");

    // Generate data
    let corpus = random_corpus(N, DIM, SEED);
    let queries: Vec<Vec<f32>> = {
        let mut rng = StdRng::seed_from_u64(SEED + 1);
        (0..NQUERIES)
            .map(|_| {
                corpus[rng.gen_range(0..N)]
                    .iter()
                    .map(|&x| x + rng.gen_range(-0.1f32..0.1))
                    .collect()
            })
            .collect()
    };

    // Compute exact ground truth (brute force)
    println!("\nComputing exact ground truth …");
    let t_gt = Instant::now();
    let ground_truth: Vec<HashSet<usize>> =
        queries.iter().map(|q| exact_topk(q, &corpus, K)).collect();
    println!("  done in {:.1}ms", t_gt.elapsed().as_millis());

    let nprobe_values = [1, 4, 16, 32, 64, NCLUSTERS];

    // ── Variant 1: IvfFlat ───────────────────────────────────────────────────
    println!("\nTraining IvfFlat …");
    let t0 = Instant::now();
    let mut ivf = IvfFlat::new(DIM, NCLUSTERS, KMEANS_ITER, SEED);
    ivf.train(&corpus).unwrap();
    ivf.add(&corpus).unwrap();
    println!(
        "  built in {:.1}ms  lists={}",
        t0.elapsed().as_millis(),
        ivf.num_lists()
    );
    let mem_ivf = ivf_memory_bytes(&ivf);
    bench(
        "IvfFlat (baseline)",
        &ivf,
        &queries,
        &ground_truth,
        &nprobe_values,
        mem_ivf,
    );

    // ── Variant 2: RairsStrict ───────────────────────────────────────────────
    println!("\nTraining RairsStrict (λ=1.0) …");
    let t0 = Instant::now();
    let mut strict = RairsStrict::new(DIM, NCLUSTERS, KMEANS_ITER, SEED, 1.0);
    strict.train(&corpus).unwrap();
    strict.add(&corpus).unwrap();
    println!(
        "  built in {:.1}ms  lists={}",
        t0.elapsed().as_millis(),
        strict.num_lists()
    );
    let mem_strict = rairs_strict_memory_bytes(&strict);
    bench(
        "RairsStrict (SRAIR, no dedup)",
        &strict,
        &queries,
        &ground_truth,
        &nprobe_values,
        mem_strict,
    );

    // ── Variant 3: RairsSeil ─────────────────────────────────────────────────
    println!("\nTraining RairsSeil (λ=1.0, block=32) …");
    let t0 = Instant::now();
    let mut seil = RairsSeil::new(DIM, NCLUSTERS, KMEANS_ITER, SEED, 1.0);
    seil.train(&corpus).unwrap();
    seil.add(&corpus).unwrap();
    println!(
        "  built in {:.1}ms  lists={}",
        t0.elapsed().as_millis(),
        seil.num_lists()
    );
    let mem_seil = rairs_seil_memory_bytes(&seil);
    bench(
        "RairsSeil (full RAIRS+SEIL)",
        &seil,
        &queries,
        &ground_truth,
        &nprobe_values,
        mem_seil,
    );

    // ── Summary table ────────────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════");
    println!("Summary: recall@10 at nprobe=16");
    println!("{:<35} {:>12} {:>12}", "Variant", "recall@10", "mem KB");

    for (name, mem, idx_box) in [
        ("IvfFlat", mem_ivf, &ivf as &dyn AnnIndex),
        ("RairsStrict", mem_strict, &strict as &dyn AnnIndex),
        ("RairsSeil", mem_seil, &seil as &dyn AnnIndex),
    ] {
        let np = 16.min(idx_box.num_lists());
        let recall = queries
            .iter()
            .zip(ground_truth.iter())
            .map(|(q, gt)| {
                let r = idx_box.search(q, K, np).unwrap();
                recall_at_k(&r, gt)
            })
            .sum::<f64>()
            / queries.len() as f64;
        println!(
            "{:<35} {:>11.1}% {:>12.1}",
            name,
            recall * 100.0,
            mem as f64 / 1024.0
        );
    }
    println!();
}
