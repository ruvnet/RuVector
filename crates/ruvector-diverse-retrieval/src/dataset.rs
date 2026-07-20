//! Deterministic synthetic dataset generation for benchmarking.
//!
//! Provides two dataset variants:
//! - [`Dataset::generate_clustered`]: flat single-level Gaussian clusters.
//! - [`Dataset::generate_hierarchical`]: two-level hierarchy where each
//!   super-cluster contains several sub-clusters. This structure ensures the
//!   candidate pool for a query spans multiple sub-clusters, making the
//!   diversity difference between TopK and PartitionMMR clearly measurable.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// A flat collection of dense float vectors with optional ground-truth cluster labels.
pub struct Dataset {
    /// All stored vectors.
    pub vectors: Vec<Vec<f32>>,
    /// Vector dimensionality.
    pub dim: usize,
    /// Ground-truth cluster index per vector (parallel to `vectors`).
    pub labels: Vec<usize>,
    /// Number of distinct clusters.
    pub n_clusters: usize,
}

impl Dataset {
    /// Generate a synthetic clustered dataset with flat cluster structure.
    ///
    /// # Parameters
    /// - `n_clusters`: number of semantic clusters
    /// - `per_cluster`: vectors per cluster
    /// - `dim`: embedding dimension
    /// - `noise_std`: intra-cluster Gaussian spread
    /// - `spread`: inter-cluster spread (cluster centres sampled from `[-spread, spread]`)
    /// - `seed`: RNG seed for full reproducibility
    pub fn generate_clustered(
        n_clusters: usize,
        per_cluster: usize,
        dim: usize,
        noise_std: f32,
        spread: f32,
        seed: u64,
    ) -> Self {
        assert!(n_clusters > 0);
        assert!(per_cluster > 0);
        assert!(dim > 0);

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f64, noise_std as f64).expect("valid normal distribution");

        // Sample cluster centres uniformly in [-spread, spread]^dim
        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.gen_range(-spread..spread)).collect())
            .collect();

        let total = n_clusters * per_cluster;
        let mut vectors = Vec::with_capacity(total);
        let mut labels = Vec::with_capacity(total);

        for (cluster_id, center) in centers.iter().enumerate() {
            for _ in 0..per_cluster {
                let v: Vec<f32> = center
                    .iter()
                    .map(|&c| c + normal.sample(&mut rng) as f32)
                    .collect();
                vectors.push(v);
                labels.push(cluster_id);
            }
        }

        Self {
            vectors,
            dim,
            labels,
            n_clusters,
        }
    }

    /// Generate a two-level hierarchical dataset.
    ///
    /// Super-cluster centres are drawn from `[-super_spread, super_spread]^dim`.
    /// Sub-cluster centres are drawn from a Gaussian ball of radius `sub_spread`
    /// around each super-cluster centre.  Vectors are drawn from a Gaussian ball
    /// of radius `noise_std` around each sub-cluster centre.
    ///
    /// This structure ensures that a query near super-cluster S will have a
    /// candidate pool that spans all `n_sub_per_super` sub-clusters of S,
    /// making the partition-diversity contrast between TopK and PartitionMMR
    /// clearly measurable.
    ///
    /// # Label semantics
    /// `labels[i]` is the global sub-cluster index (not the super-cluster index).
    /// Global sub-cluster index = super_idx * n_sub_per_super + sub_idx.
    pub fn generate_hierarchical(
        n_super: usize,
        n_sub_per_super: usize,
        per_sub: usize,
        dim: usize,
        super_spread: f32,
        sub_spread: f32,
        noise_std: f32,
        seed: u64,
    ) -> Self {
        assert!(n_super > 0);
        assert!(n_sub_per_super > 0);
        assert!(per_sub > 0);
        assert!(dim > 0);

        let mut rng = StdRng::seed_from_u64(seed);
        let sub_noise = Normal::new(0.0f64, sub_spread as f64).expect("sub normal");
        let vec_noise = Normal::new(0.0f64, noise_std as f64).expect("vec normal");

        let n_clusters = n_super * n_sub_per_super;
        let total = n_clusters * per_sub;
        let mut vectors = Vec::with_capacity(total);
        let mut labels = Vec::with_capacity(total);

        for super_idx in 0..n_super {
            // Sample super-cluster centre uniformly in [-super_spread, super_spread]^dim.
            let super_center: Vec<f32> = (0..dim)
                .map(|_| rng.gen_range(-super_spread..super_spread))
                .collect();

            for sub_idx in 0..n_sub_per_super {
                let global_label = super_idx * n_sub_per_super + sub_idx;

                // Sub-cluster centre: super-centre + Gaussian noise with σ=sub_spread.
                let sub_center: Vec<f32> = super_center
                    .iter()
                    .map(|&c| c + sub_noise.sample(&mut rng) as f32)
                    .collect();

                // Each vector: sub-centre + Gaussian noise with σ=noise_std.
                for _ in 0..per_sub {
                    let v: Vec<f32> = sub_center
                        .iter()
                        .map(|&c| c + vec_noise.sample(&mut rng) as f32)
                        .collect();
                    vectors.push(v);
                    labels.push(global_label);
                }
            }
        }

        Self {
            vectors,
            dim,
            labels,
            n_clusters,
        }
    }

    /// Total number of vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// True when the dataset has no vectors.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Sample `n` query vectors from the dataset (copies of stored vectors).
    pub fn sample_queries(&self, n: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed ^ 0xDEAD_BEEF);
        (0..n)
            .map(|_| {
                let idx = rng.gen_range(0..self.vectors.len());
                self.vectors[idx].clone()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_size_matches() {
        let ds = Dataset::generate_clustered(5, 20, 16, 0.5, 5.0, 42);
        assert_eq!(ds.len(), 100);
        assert_eq!(ds.dim, 16);
        assert_eq!(ds.labels.len(), 100);
    }

    #[test]
    fn labels_in_range() {
        let ds = Dataset::generate_clustered(4, 10, 8, 1.0, 3.0, 7);
        for &l in &ds.labels {
            assert!(l < 4, "label {l} out of range");
        }
    }

    #[test]
    fn reproducible_with_same_seed() {
        let a = Dataset::generate_clustered(3, 5, 4, 0.5, 2.0, 99);
        let b = Dataset::generate_clustered(3, 5, 4, 0.5, 2.0, 99);
        assert_eq!(a.vectors, b.vectors);
    }

    #[test]
    fn sample_queries_length() {
        let ds = Dataset::generate_clustered(3, 10, 4, 0.5, 2.0, 1);
        let queries = ds.sample_queries(7, 1);
        assert_eq!(queries.len(), 7);
    }

    #[test]
    fn hierarchical_size_matches() {
        let ds = Dataset::generate_hierarchical(3, 4, 5, 8, 5.0, 1.0, 0.3, 42);
        assert_eq!(ds.len(), 3 * 4 * 5);
        assert_eq!(ds.n_clusters, 12);
    }

    #[test]
    fn hierarchical_labels_in_range() {
        let ds = Dataset::generate_hierarchical(2, 3, 4, 8, 5.0, 1.0, 0.3, 7);
        for &l in &ds.labels {
            assert!(l < 6, "label {l} out of range for 2*3=6 sub-clusters");
        }
    }

    #[test]
    fn hierarchical_reproducible() {
        let a = Dataset::generate_hierarchical(2, 2, 3, 4, 3.0, 0.5, 0.2, 42);
        let b = Dataset::generate_hierarchical(2, 2, 3, 4, 3.0, 0.5, 0.2, 42);
        assert_eq!(a.vectors, b.vectors);
    }
}
