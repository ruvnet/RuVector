//! Synthetic vector dataset generation and exact k-NN ground truth.

use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

/// Generate `n` vectors of `dims` dimensions drawn from `k_clusters`
/// Gaussian clusters (σ = 0.3 per coordinate).
pub fn generate_clustered(n: usize, dims: usize, k_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);

    // Sample cluster centroids uniformly in [0, 1]^dims.
    let centroids: Vec<Vec<f32>> = (0..k_clusters)
        .map(|_| (0..dims).map(|_| rng.gen::<f32>()).collect())
        .collect();

    (0..n)
        .map(|_| {
            let c = rng.gen_range(0..k_clusters);
            centroids[c]
                .iter()
                .map(|&mean| {
                    // Box-Muller for Gaussian noise (σ = 0.3).
                    let u1: f32 = (1.0_f32 - rng.gen::<f32>()).max(1e-10);
                    let u2: f32 = rng.gen::<f32>();
                    let z: f32 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    (mean + 0.3 * z).clamp(0.0_f32, 1.0_f32)
                })
                .collect()
        })
        .collect()
}

/// Generate `n` uniformly-random query vectors.
pub fn generate_queries(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed ^ 0xDEAD_BEEF);
    (0..n)
        .map(|_| (0..dims).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

/// Compute exact k-NN ground truth by brute-force O(n·d) scan.
///
/// Returns, for each query, the indices of the k nearest dataset vectors
/// sorted by ascending squared distance.
pub fn brute_force_knn(dataset: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(f32, usize)> = dataset
                .iter()
                .enumerate()
                .map(|(i, v)| (sq_dist(v, q), i))
                .collect();
            // Partial sort: only guarantee top-k order (full sort is fine for PoC).
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            dists.truncate(k);
            dists.iter().map(|&(_, i)| i).collect()
        })
        .collect()
}

#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clustered_has_correct_shape() {
        let data = generate_clustered(200, 32, 5, 1);
        assert_eq!(data.len(), 200);
        assert!(data.iter().all(|v| v.len() == 32));
    }

    #[test]
    fn all_values_in_unit_cube() {
        let data = generate_clustered(100, 16, 4, 2);
        for v in &data {
            for &x in v {
                assert!((0.0..=1.0).contains(&x), "value {x} outside [0,1]");
            }
        }
    }

    #[test]
    fn brute_force_returns_sorted_neighbors() {
        let dataset: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32]).collect();
        let query = vec![10.0f32];
        let gt = brute_force_knn(&dataset, &[query], 5);
        assert_eq!(gt[0], vec![10, 9, 11, 8, 12]);
    }

    #[test]
    fn brute_force_k_capped_at_dataset_size() {
        let dataset: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32]).collect();
        let query = vec![2.0f32];
        let gt = brute_force_knn(&dataset, &[query], 10);
        assert_eq!(gt[0].len(), 5);
    }
}
