//! LoRANN: Clustering-Based ANN with Reduced-Rank Regression Score Approximation
//!
//! Implements the algorithm from:
//! Jääsaari, E., Hyvönen, V., Roos, T.
//! "LoRANN: Low-Rank Matrix Factorization for Approximate Nearest Neighbor Search"
//! NeurIPS 2024, arXiv:2410.18926.
//!
//! ## Core idea
//!
//! Standard IVF (inverted file index) assigns corpus vectors to clusters and,
//! at query time, scores all vectors in the `n_probe` nearest clusters exactly —
//! costing O(n_probe · m_avg · d) floating-point multiplications.
//!
//! LoRANN replaces the per-cluster exact scorer with a **rank-r approximation**
//! derived from the truncated SVD of the cluster's document matrix:
//!
//!   X_c ≈ U_r Σ_r V_r^T
//!
//! Score approximation: `score(q, X_c) ≈ A (B^T q)` where A = U_r Σ_r ∈ R^{m×r}
//! and B = V_r ∈ R^{d×r}.  Query cost drops from O(d·m) → O(r(d+m)).
//!
//! The top-`candidate_set` candidates are then **exact-reranked** using raw f32
//! inner products, recovering high recall at substantially higher QPS.
//!
//! ## Variants benchmarked
//!
//! | Struct | Score function | Rerank | Use when |
//! |---|---|---|---|
//! | `FlatExactIndex` | exact dot-product | N/A | accuracy baseline |
//! | `LorannIndex` (rank=16) | RRR rank-16 | yes | moderate recall |
//! | `LorannIndex` (rank=32) | RRR rank-32 | yes | high recall |
//!
//! ## Benchmarks
//!
//! See `src/main.rs` (`lorann-demo`) for end-to-end recall + QPS numbers.

pub mod config;
pub mod error;
pub mod index;
pub mod kmeans;
pub mod regression;

pub use config::LorannConfig;
pub use error::LorannError;
pub use index::{AnnIndex, FlatExactIndex, LorannIndex, SearchResult};

#[cfg(test)]
mod tests {
    use super::*;

    fn small_corpus(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::{Rng as _, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n).map(|_| (0..d).map(|_| rng.gen_range(-1.0f32..1.0)).collect()).collect()
    }

    fn recall(truth: &[usize], got: &[SearchResult]) -> f64 {
        let s: std::collections::HashSet<usize> = truth.iter().copied().collect();
        got.iter().filter(|r| s.contains(&r.id)).count() as f64 / truth.len() as f64
    }

    // FlatExactIndex always returns 100% recall against itself
    #[test]
    fn flat_exact_self_recall_is_one() {
        let data = small_corpus(500, 32, 1);
        let queries = small_corpus(20, 32, 2);
        let idx = FlatExactIndex::build(data).unwrap();
        for q in &queries {
            let res = idx.search(q, 10).unwrap();
            assert_eq!(res.len(), 10);
        }
        // Ground truth from itself — first result should be the query's nearest
        let flat2 = FlatExactIndex::build(small_corpus(500, 32, 1)).unwrap();
        let gt: Vec<Vec<usize>> = queries.iter()
            .map(|q| flat2.search(q, 10).unwrap().iter().map(|r| r.id).collect())
            .collect();
        let r: f64 = queries.iter().zip(gt.iter())
            .map(|(q, t)| recall(t, &idx.search(q, 10).unwrap()))
            .sum::<f64>() / queries.len() as f64;
        assert!((r - 1.0).abs() < 1e-9, "FlatExact recall should be 1.0, got {r}");
    }

    // LorannIndex: recall@10 ≥ 70% on a Gaussian-clustered corpus.
    // Uses the same generator as the main benchmark to ensure realistic cluster structure.
    #[test]
    fn lorann_recall_above_threshold() {
        use rand::{Rng as _, SeedableRng};
        use rand_distr::{Distribution, Normal, Uniform};
        let n = 1_500;
        let d = 64;
        let n_clusters_data = 15;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let centroid_range = Uniform::new(-2.0f32, 2.0);
        let centroids: Vec<Vec<f32>> = (0..n_clusters_data)
            .map(|_| (0..d).map(|_| centroid_range.sample(&mut rng)).collect())
            .collect();
        let noise = Normal::new(0.0f64, 0.5).unwrap();
        let data: Vec<Vec<f32>> = (0..n).map(|_| {
            let c = &centroids[rng.gen_range(0..n_clusters_data)];
            c.iter().map(|&x| x + noise.sample(&mut rng) as f32).collect()
        }).collect();
        let queries: Vec<Vec<f32>> = (0..50).map(|_| {
            let c = &centroids[rng.gen_range(0..n_clusters_data)];
            c.iter().map(|&x| x + noise.sample(&mut rng) as f32).collect()
        }).collect();

        let flat = FlatExactIndex::build(data.clone()).unwrap();
        let gt: Vec<Vec<usize>> = queries.iter()
            .map(|q| flat.search(q, 10).unwrap().iter().map(|r| r.id).collect())
            .collect();
        let cfg = LorannConfig { n_clusters: 38, rank: 32, n_probe: 10, candidate_set: 250,
            kmeans_max_iter: 20, seed: 42 };
        let idx = LorannIndex::build(data, cfg).unwrap();
        let r: f64 = queries.iter().zip(gt.iter())
            .map(|(q, t)| recall(t, &idx.search(q, 10).unwrap()))
            .sum::<f64>() / queries.len() as f64;
        assert!(r >= 0.70, "LoRANN recall@10 = {:.1}% < 70% on clustered data", r * 100.0);
    }

    // ClusterModel scores are correlated with exact inner products
    #[test]
    fn cluster_model_rank_correlation() {
        use crate::regression::ClusterModel;
        use rand::{Rng as _, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let m = 50;
        let d = 32;
        let docs: Vec<Vec<f32>> = (0..m)
            .map(|_| (0..d).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
            .collect();
        let query: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let model = ClusterModel::fit(0, &docs, 16).unwrap();
        let approx = model.approximate_scores(&query);
        assert_eq!(approx.len(), m);
        // Exact scores
        let exact: Vec<f32> = docs.iter()
            .map(|v| v.iter().zip(query.iter()).map(|(a, b)| a * b).sum())
            .collect();
        // Spearman rank correlation: top-5 approx overlap with top-5 exact
        let mut approx_ranked: Vec<usize> = (0..m).collect();
        approx_ranked.sort_unstable_by(|&a, &b| approx[b].total_cmp(&approx[a]));
        let mut exact_ranked: Vec<usize> = (0..m).collect();
        exact_ranked.sort_unstable_by(|&a, &b| exact[b].total_cmp(&exact[a]));
        let top5_approx: std::collections::HashSet<usize> = approx_ranked[..5].iter().copied().collect();
        let top5_exact: std::collections::HashSet<usize> = exact_ranked[..5].iter().copied().collect();
        let overlap = top5_approx.intersection(&top5_exact).count();
        // At rank=16, d=32, we expect at least 2/5 overlap
        assert!(overlap >= 2, "Top-5 overlap between approx and exact = {overlap} < 2");
    }

    // Memory bytes should be proportional to n × d
    #[test]
    fn memory_bytes_ordering() {
        let data_small = small_corpus(200, 32, 1);
        let data_large = small_corpus(1_000, 32, 1);
        let idx_s = FlatExactIndex::build(data_small).unwrap();
        let idx_l = FlatExactIndex::build(data_large).unwrap();
        assert!(idx_l.memory_bytes() > idx_s.memory_bytes());
    }

    // k-means produces the expected number of clusters
    #[test]
    fn kmeans_cluster_count() {
        use crate::kmeans::kmeans;
        let data = small_corpus(500, 16, 5);
        let result = kmeans(&data, 10, 10, 42).unwrap();
        assert_eq!(result.centroids.len(), 10);
        assert_eq!(result.assignments.len(), 500);
        // All assignments are valid
        for &a in &result.assignments {
            assert!(a < 10);
        }
    }
}
