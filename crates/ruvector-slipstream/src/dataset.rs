//! Deterministic dataset generation for Slipstream benchmarks.
//!
//! Two dataset shapes:
//! - **Streamed clustered**: vectors grouped by cluster, simulating spatial locality.
//! - **Shuffled clustered**: same clusters but random arrival order (breaks locality).
//!
//! Both use the same underlying point distribution so recall results are comparable.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Generate `n_clusters × per_cluster` vectors in a streamed (locality-preserving)
/// order: all vectors from cluster 0 first, then cluster 1, etc.
///
/// Returns `(vectors, cluster_assignments)`.
pub fn streamed_clustered(
    n_clusters: usize,
    per_cluster: usize,
    dims: usize,
    cluster_std: f32,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f64, cluster_std as f64).unwrap();

    // Generate cluster centroids.
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    let mut vectors = Vec::with_capacity(n_clusters * per_cluster);
    let mut assignments = Vec::with_capacity(n_clusters * per_cluster);

    for (c, centroid) in centroids.iter().enumerate() {
        for _ in 0..per_cluster {
            let v: Vec<f32> = centroid
                .iter()
                .map(|&x: &f32| {
                    let noise: f64 = normal.sample(&mut rng);
                    x + noise as f32
                })
                .collect();
            vectors.push(v);
            assignments.push(c);
        }
    }

    (vectors, assignments)
}

/// Same data as `streamed_clustered` but in a randomly shuffled arrival order.
/// Breaking the locality allows us to measure the benefit of warm-starting
/// in the streamed case vs. no benefit in the shuffled case.
pub fn shuffled_clustered(
    n_clusters: usize,
    per_cluster: usize,
    dims: usize,
    cluster_std: f32,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let (mut vecs, mut asgn) = streamed_clustered(n_clusters, per_cluster, dims, cluster_std, seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xABCD_EF01_2345_6789);

    // Fisher-Yates shuffle.
    let n = vecs.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        vecs.swap(i, j);
        asgn.swap(i, j);
    }
    (vecs, asgn)
}

/// Generate `n_queries` random query vectors drawn from cluster centroids
/// (same seed family as the dataset).
pub fn cluster_queries(
    n_queries: usize,
    dims: usize,
    n_clusters: usize,
    cluster_std: f32,
    dataset_seed: u64,
) -> Vec<Vec<f32>> {
    // Re-derive centroids from the same seed used in streamed_clustered.
    let mut rng = StdRng::seed_from_u64(dataset_seed);
    let normal = Normal::new(0.0f64, cluster_std as f64).unwrap();

    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    let mut qrng = StdRng::seed_from_u64(dataset_seed ^ 0x1111_2222_3333_4444);
    (0..n_queries)
        .map(|_| {
            let c = qrng.gen_range(0..n_clusters);
            centroids[c]
                .iter()
                .map(|&x: &f32| {
                    let noise: f64 = normal.sample(&mut qrng);
                    x + noise as f32
                })
                .collect()
        })
        .collect()
}

/// Brute-force ground truth: returns the indices of the k nearest vectors for each query.
pub fn ground_truth(dataset: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(f32, usize)> = dataset
                .iter()
                .enumerate()
                .map(|(i, v)| (l2_sq(q, v), i))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            dists.iter().take(k).map(|&(_, id)| id).collect()
        })
        .collect()
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_returns_correct_count() {
        let (vecs, asgn) = streamed_clustered(5, 100, 32, 0.1, 42);
        assert_eq!(vecs.len(), 500);
        assert_eq!(asgn.len(), 500);
    }

    #[test]
    fn streamed_has_locality_first_100_same_cluster() {
        let (_, asgn) = streamed_clustered(5, 100, 32, 0.1, 42);
        // All of the first 100 should be cluster 0.
        assert!(asgn[..100].iter().all(|&c| c == 0));
    }

    #[test]
    fn shuffled_breaks_locality() {
        let (_, asgn) = shuffled_clustered(5, 100, 32, 0.1, 42);
        // After shuffling, first 10 should NOT all be from cluster 0.
        let first_10_same = asgn[..10].windows(2).all(|w| w[0] == w[1]);
        assert!(!first_10_same, "shuffled should break per-cluster ordering");
    }

    #[test]
    fn ground_truth_nearest_is_self() {
        let (vecs, _) = streamed_clustered(2, 5, 4, 0.01, 1);
        let gt = ground_truth(&vecs, &vecs, 1);
        for (i, nn) in gt.iter().enumerate() {
            assert_eq!(nn[0], i, "nearest neighbour of self should be self");
        }
    }
}
