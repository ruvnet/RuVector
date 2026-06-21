//! Deterministic dataset generation for benchmarks and tests.
//!
//! Uses a fixed seed so every run produces the same dataset. No external
//! files, no network access, no OS entropy. Pure Rust RNG.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Generate N unit-normalized random vectors of dimension D.
pub fn random_unit_vectors(n: usize, dims: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).expect("valid normal distribution");
    let mut out = Vec::with_capacity(n * dims);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dims).map(|_| normal.sample(&mut rng)).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut v {
                *x /= norm;
            }
        }
        out.extend_from_slice(&v);
    }
    out
}

/// Generate a clustered dataset: n_clusters × n_per_cluster Gaussian blobs.
///
/// Each cluster center is a random unit vector; points are sampled from a
/// Gaussian of `std_dev` around the center and then re-normalized to the
/// unit sphere (preserving the cluster structure).
///
/// Returns: (flat_vectors, cluster_assignments)
pub fn clustered_unit_vectors(
    n_clusters: usize,
    n_per_cluster: usize,
    dims: usize,
    std_dev: f32,
    seed: u64,
) -> (Vec<f32>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal_center = Normal::new(0.0f32, 1.0).expect("normal");
    let noise = Normal::new(0.0f32, std_dev).expect("noise normal");

    // Generate cluster centers as random unit vectors.
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let mut v: Vec<f32> = (0..dims).map(|_| normal_center.sample(&mut rng)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect();

    let n = n_clusters * n_per_cluster;
    let mut flat = Vec::with_capacity(n * dims);
    let mut assignments = Vec::with_capacity(n);

    for (ci, center) in centers.iter().enumerate() {
        for _ in 0..n_per_cluster {
            let mut v: Vec<f32> = center.iter().map(|&c| c + noise.sample(&mut rng)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            flat.extend_from_slice(&v);
            assignments.push(ci);
        }
    }

    (flat, assignments)
}

/// Generate N random query vectors using a different seed.
pub fn random_queries(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let flat = random_unit_vectors(n, dims, seed);
    flat.chunks_exact(dims).map(|c| c.to_vec()).collect()
}

/// Generate N queries that are each near one of the clusters in the dataset.
///
/// For each query, we pick a random cluster and sample a point near its center.
/// This ensures each query has well-defined nearest neighbors.
pub fn clustered_queries(
    n: usize,
    dims: usize,
    dataset: &[f32],
    n_per_cluster: usize,
    std_dev: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0f32, std_dev * 0.5).expect("noise normal");
    let n_clusters = dataset.len() / dims / n_per_cluster;
    let cluster_pick = Uniform::new(0usize, n_clusters);

    (0..n)
        .map(|_| {
            // Pick a random cluster center (first vector of that cluster).
            let ci = cluster_pick.sample(&mut rng);
            let center = &dataset[ci * n_per_cluster * dims..(ci * n_per_cluster + 1) * dims];
            let mut v: Vec<f32> = center.iter().map(|&c| c + noise.sample(&mut rng)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect()
}

/// Brute-force ground truth: for each query, return indices of its k nearest
/// neighbors from the dataset.
pub fn ground_truth(dataset: &[f32], queries: &[Vec<f32>], dims: usize, k: usize) -> Vec<Vec<u32>> {
    let n = dataset.len() / dims;
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(u32, f32)> = (0..n)
                .map(|i| {
                    let v = &dataset[i * dims..(i + 1) * dims];
                    let d: f32 = v.iter().zip(q.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i as u32, d)
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            dists.truncate(k);
            dists.into_iter().map(|(idx, _)| idx).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_are_unit_normalized() {
        let data = random_unit_vectors(10, 8, 42);
        for chunk in data.chunks_exact(8) {
            let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");
        }
    }

    #[test]
    fn ground_truth_returns_k_results() {
        let data = random_unit_vectors(50, 8, 1);
        let queries = random_queries(5, 8, 99);
        let gt = ground_truth(&data, &queries, 8, 5);
        for (qi, nn) in gt.iter().enumerate() {
            assert_eq!(nn.len(), 5, "query {qi}");
        }
    }

    #[test]
    fn clustered_vectors_have_unit_norm() {
        let (data, assignments) = clustered_unit_vectors(4, 25, 8, 0.1, 7);
        assert_eq!(data.len(), 4 * 25 * 8);
        assert_eq!(assignments.len(), 100);
        for chunk in data.chunks_exact(8) {
            let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "norm = {norm}");
        }
    }
}
