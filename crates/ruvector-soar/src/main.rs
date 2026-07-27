//! `soar-demo` — runs three IVF variants (Single, Spillover, SOAR) on a
//! synthetic clustered dataset and prints recall@10 and mean residual
//! correlation for each. Output is the source of the numbers in the
//! research doc and gist.

use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_soar::{brute_force_topk, recall, Assignment, IvfIndex};
use std::time::Instant;

fn make_dataset(n: usize, dim: usize, n_clusters: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    // Anisotropic clusters: each cluster has a random long axis with 4×
    // the variance of the orthogonal directions. This mimics real
    // embedding distributions and is the regime where SOAR's
    // anti-correlated coverage wins over plain spillover.
    let anchors: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-3.0..3.0_f32)).collect())
        .collect();
    let long_axes: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let raw: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0_f32)).collect();
            let n: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            raw.iter().map(|x| x / n.max(1e-6)).collect()
        })
        .collect();

    let db: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let ci = i % n_clusters;
            let a = &anchors[ci];
            let axis = &long_axes[ci];
            // base isotropic noise + anisotropic kick along the long axis
            let mut v: Vec<f32> = (0..dim)
                .map(|d| a[d] + rng.gen_range(-0.6..0.6_f32))
                .collect();
            let kick = rng.gen_range(-2.4..2.4_f32);
            for d in 0..dim {
                v[d] += kick * axis[d];
            }
            v
        })
        .collect();

    // Queries: uniform over the embedding range. NNs frequently cross
    // cluster boundaries — this is the hard regime for plain IVF.
    let queries: Vec<Vec<f32>> = (0..200)
        .map(|_| {
            (0..dim)
                .map(|_| rng.gen_range(-4.0..4.0_f32))
                .collect()
        })
        .collect();

    (db, queries)
}

fn evaluate(
    label: &str,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    truths: &[Vec<(u32, f32)>],
    n_centroids: usize,
    n_probe: usize,
    assignment: Assignment,
) {
    let t0 = Instant::now();
    let idx = IvfIndex::build(db.to_vec(), n_centroids, assignment, 0xC0FFEE).unwrap();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let posting = idx.posting_entries();

    let t0 = Instant::now();
    let mut total_recall = 0.0_f32;
    for (q, gt) in queries.iter().zip(truths.iter()) {
        let res = idx.search(q, 10, n_probe);
        total_recall += recall(&res, gt);
    }
    let avg_recall = total_recall / queries.len() as f32;
    let q_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / queries.len() as f64;

    let corr = idx
        .mean_residual_correlation()
        .map(|c| format!("{:>+6.3}", c))
        .unwrap_or_else(|| "    -- ".into());

    println!(
        "  {label:<22} | recall@10 = {avg_recall:.4} | postings = {posting:>7} | build = {build_ms:>7.1} ms | query = {q_us:>6.1} µs | corr = {corr}",
    );
}

fn main() {
    println!("ruvector-soar demo — synthetic clustered f32 vectors\n");

    // (N, dim, n_centroids, n_probe). Aggressive low n_probe — this is the
    // regime where boundary spillover matters most.
    for &(n, dim, k_centroids, n_probe) in &[
        (10_000usize, 32usize, 128usize, 1usize),
        (10_000, 32, 128, 2),
        (20_000, 64, 256, 2),
        (20_000, 64, 256, 4),
    ] {
        let (db, queries) = make_dataset(n, dim, k_centroids, 0xDEADBEEF + n as u64);
        let truths: Vec<Vec<(u32, f32)>> = queries
            .iter()
            .map(|q| brute_force_topk(&db, q, 10))
            .collect();

        println!(
            "Dataset: N={n} D={dim} centroids={k_centroids} n_probe={n_probe} queries={}",
            queries.len()
        );
        evaluate("Single (1x)", &db, &queries, &truths, k_centroids, n_probe, Assignment::Single);
        evaluate("Spillover (2x)", &db, &queries, &truths, k_centroids, n_probe, Assignment::Spillover);
        evaluate(
            "SOAR (lambda=1.5)",
            &db,
            &queries,
            &truths,
            k_centroids,
            n_probe,
            Assignment::Soar { lambda: 1.5 },
        );
        evaluate(
            "SOAR (lambda=4.0)",
            &db,
            &queries,
            &truths,
            k_centroids,
            n_probe,
            Assignment::Soar { lambda: 4.0 },
        );
        println!();
    }
}
