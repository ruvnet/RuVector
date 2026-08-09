//! Deterministic synthetic dataset generator for semantic-cache benchmarks.
//!
//! Produces a "clustered query" workload that mimics agent memory access
//! patterns: a fixed number of semantic clusters, each queried multiple times
//! with small perturbations. The ground-truth top-k results are computed by
//! brute-force linear scan over a set of base vectors.

use crate::{dot, normalize};

/// Parameters for the synthetic dataset.
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Number of base vectors in the "database".
    pub n_vecs: usize,
    /// Number of unique semantic clusters (distinct intents).
    pub n_clusters: usize,
    /// Total queries to generate (with repetition across clusters).
    pub n_queries: usize,
    /// Embedding dimensionality.
    pub dims: usize,
    /// Std-dev of Gaussian noise added to each repeated query.
    pub noise_std: f32,
    /// Top-k to retrieve per query.
    pub k: usize,
    /// Seed for the LCG random number generator.
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            n_vecs: 10_000,
            n_clusters: 50,
            n_queries: 500,
            dims: 128,
            noise_std: 0.02,
            k: 10,
            seed: 0xC0DE_CAFE_BABE_7777,
        }
    }
}

/// The generated dataset.
pub struct Dataset {
    /// Base vectors (unit-normalized), shape [n_vecs × dims].
    pub base: Vec<Vec<f32>>,
    /// Query vectors (unit-normalized), shape [n_queries × dims].
    pub queries: Vec<Vec<f32>>,
    /// Cluster assignment for each query (0..n_clusters).
    pub query_cluster: Vec<usize>,
    /// Ground-truth top-k IDs for each query.
    pub ground_truth: Vec<Vec<u64>>,
    pub config: DatasetConfig,
}

/// Minimal LCG random number generator (no external crate).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x6C62272E07BB0142)
    }

    /// Returns next u64.
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Returns f32 in (-1, 1).
    fn next_f32(&mut self) -> f32 {
        let u = self.next_u64();
        (u as f32 / u64::MAX as f32) * 2.0 - 1.0
    }

    /// Box-Muller Gaussian sample (mean=0, std=1).
    fn next_gaussian(&mut self) -> f32 {
        // Use two uniform samples in (0,1].
        let u1 = (self.next_u64() as f32 / u64::MAX as f32).max(1e-10);
        let u2 = self.next_u64() as f32 / u64::MAX as f32;
        // Box-Muller approximation without trig: use ratio approximation.
        // For benchmarks, a simple stretched uniform is sufficient.
        (u1 - 0.5) * 3.46 * u2.sqrt()
    }

    fn next_usize(&mut self, limit: usize) -> usize {
        (self.next_u64() as usize) % limit
    }
}

fn random_unit_vec(rng: &mut Lcg, dims: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dims).map(|_| rng.next_f32()).collect();
    normalize(&mut v);
    v
}

impl Dataset {
    /// Generate the full dataset from `config`.
    pub fn generate(cfg: DatasetConfig) -> Self {
        let mut rng = Lcg::new(cfg.seed);

        // Base vectors.
        let base: Vec<Vec<f32>> = (0..cfg.n_vecs)
            .map(|_| random_unit_vec(&mut rng, cfg.dims))
            .collect();

        // Cluster centers.
        let centers: Vec<Vec<f32>> = (0..cfg.n_clusters)
            .map(|_| random_unit_vec(&mut rng, cfg.dims))
            .collect();

        // Queries: pick random cluster, add noise, normalize.
        let mut queries = Vec::with_capacity(cfg.n_queries);
        let mut query_cluster = Vec::with_capacity(cfg.n_queries);
        for _ in 0..cfg.n_queries {
            let c = rng.next_usize(cfg.n_clusters);
            let mut q: Vec<f32> = centers[c]
                .iter()
                .map(|&x| x + rng.next_gaussian() * cfg.noise_std)
                .collect();
            normalize(&mut q);
            queries.push(q);
            query_cluster.push(c);
        }

        // Ground truth: brute-force top-k for each query.
        let ground_truth: Vec<Vec<u64>> = queries
            .iter()
            .map(|q| brute_force_top_k(q, &base, cfg.k))
            .collect();

        Self {
            base,
            queries,
            query_cluster,
            ground_truth,
            config: cfg,
        }
    }
}

/// Brute-force top-k search by cosine similarity (dot product on unit vectors).
pub fn brute_force_top_k(query: &[f32], base: &[Vec<f32>], k: usize) -> Vec<u64> {
    let mut scores: Vec<(f32, u64)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (dot(query, v), i as u64))
        .collect();
    // Partial sort: find top-k.
    scores.select_nth_unstable_by(k.saturating_sub(1), |a, b| b.0.partial_cmp(&a.0).unwrap());
    scores[..k].iter().map(|(_, id)| *id).collect()
}

/// Recall@k: fraction of ground-truth IDs found in returned IDs.
pub fn recall_at_k(truth: &[u64], returned: &[u64]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let hits = returned.iter().filter(|id| truth.contains(id)).count();
    hits as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_shapes_are_correct() {
        let cfg = DatasetConfig {
            n_vecs: 200,
            n_clusters: 10,
            n_queries: 50,
            dims: 32,
            ..Default::default()
        };
        let ds = Dataset::generate(cfg.clone());
        assert_eq!(ds.base.len(), cfg.n_vecs);
        assert_eq!(ds.queries.len(), cfg.n_queries);
        assert_eq!(ds.ground_truth.len(), cfg.n_queries);
        assert_eq!(ds.ground_truth[0].len(), cfg.k);
    }

    #[test]
    fn ground_truth_recall_is_perfect() {
        let cfg = DatasetConfig {
            n_vecs: 200,
            n_clusters: 5,
            n_queries: 20,
            dims: 16,
            k: 5,
            ..Default::default()
        };
        let ds = Dataset::generate(cfg);
        // Brute force vs brute force must be 1.0.
        for (i, q) in ds.queries.iter().enumerate() {
            let returned = brute_force_top_k(q, &ds.base, ds.config.k);
            let r = recall_at_k(&ds.ground_truth[i], &returned);
            assert!((r - 1.0).abs() < 1e-5, "recall[{i}] = {r}");
        }
    }

    #[test]
    fn cluster_assignments_in_bounds() {
        let cfg = DatasetConfig {
            n_vecs: 100,
            n_clusters: 8,
            n_queries: 32,
            dims: 16,
            ..Default::default()
        };
        let ds = Dataset::generate(cfg.clone());
        for c in &ds.query_cluster {
            assert!(*c < cfg.n_clusters, "cluster {c} out of range");
        }
    }
}
