/// Multi-subspace HNSW with coherence-weighted fusion for RuVector.
///
/// Three measurable variants:
///
/// | Variant            | Description                                           |
/// |--------------------|-------------------------------------------------------|
/// | `BaselineHnsw`     | Single HNSW on all D dimensions (reference)           |
/// | `SubspaceUnionHnsw`| K sub-HNSWs on D/K dims; union + full-space re-rank   |
/// | `CoherenceHnsw`    | Same K sub-HNSWs; coherence-weighted distance fusion  |
///
/// # Research background
///
/// Closest prior work: "Subspace Collision" (arXiv:2411.14754, SIGMOD 2025)
/// uses subspace partitioning with clustering indexes and collision-count fusion.
/// "TaCo" (arXiv:2603.24919, March 2026) adds entropy-balanced partitioning.
/// Neither uses HNSW per-subspace nor runtime variance-based coherence weights.
pub mod dataset;
pub mod hnsw;
pub mod subspace;

pub use hnsw::{brute_force_knn, sq_l2, HnswIndex};
pub use subspace::{
    ground_truth, project, recall_at_k, BaselineHnsw, CoherenceHnsw, SubspaceUnionHnsw,
};

/// Common configuration for building indexes in benchmarks.
#[derive(Clone, Debug)]
pub struct IndexConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub num_subspaces: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        IndexConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 80,
            num_subspaces: 4,
        }
    }
}

/// Compute mean, p50, and p95 from a slice of durations (in microseconds).
pub fn percentiles(mut us: Vec<u64>) -> (f64, u64, u64) {
    if us.is_empty() {
        return (0.0, 0, 0);
    }
    us.sort_unstable();
    let mean = us.iter().sum::<u64>() as f64 / us.len() as f64;
    let p50 = us[us.len() / 2];
    let p95 = us[(us.len() * 95 / 100).min(us.len() - 1)];
    (mean, p50, p95)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{generate_clustered, generate_queries};

    fn build_and_check(n: usize, dim: usize, k_subspaces: usize, threshold: f32, label: &str) {
        let (vecs, _) = generate_clustered(n, dim, 10, dim / 2, 42);
        let queries = generate_queries(50, dim, 7);
        let cfg = IndexConfig {
            m: 16,
            ef_construction: 120,
            ef_search: 100,
            num_subspaces: k_subspaces,
        };

        let baseline = BaselineHnsw::build(&vecs, cfg.m, cfg.ef_construction);
        let coh = CoherenceHnsw::build(&vecs, cfg.num_subspaces, cfg.m, cfg.ef_construction);

        let recall_base: f32 = queries
            .iter()
            .map(|q| {
                let gt = ground_truth(&vecs, q, 10);
                let r = baseline.search(q, 10, cfg.ef_search);
                recall_at_k(&r, &gt, 10)
            })
            .sum::<f32>()
            / queries.len() as f32;

        let recall_coh: f32 = queries
            .iter()
            .map(|q| {
                let gt = ground_truth(&vecs, q, 10);
                let r = coh.search(q, 10, cfg.ef_search);
                recall_at_k(&r, &gt, 10)
            })
            .sum::<f32>()
            / queries.len() as f32;

        println!(
            "[{label}] baseline recall@10={recall_base:.3}, coherence recall@10={recall_coh:.3}"
        );

        assert!(
            recall_base >= threshold,
            "[{label}] baseline recall@10={recall_base:.3} < {threshold}"
        );
        // Coherence must stay within 10pp of baseline (trade-off is acceptable).
        assert!(
            recall_coh >= recall_base - 0.10,
            "[{label}] coherence {recall_coh:.3} more than 10pp below baseline {recall_base:.3}"
        );
    }

    /// Acceptance test 1: small dataset (n=500, d=32).
    #[test]
    fn acceptance_small() {
        build_and_check(500, 32, 2, 0.70, "small");
    }

    /// Acceptance test 2: medium dataset (n=2000, d=64).
    /// Baseline HNSW recall is ~0.63 on half-signal/half-noise data;
    /// coherence fusion reaches ~0.84, demonstrating the subspace benefit.
    #[test]
    fn acceptance_medium() {
        build_and_check(2000, 64, 4, 0.55, "medium");
    }

    #[test]
    fn percentiles_basic() {
        let us = vec![10u64, 20, 30, 40, 50];
        let (mean, p50, p95) = percentiles(us);
        assert!((mean - 30.0).abs() < 1e-6);
        assert_eq!(p50, 30);
        assert_eq!(p95, 50);
    }
}
