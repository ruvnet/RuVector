//! High-level DABS index API.

use crate::dist::l2_sq;
use crate::error::DabsError;
use crate::graph::{search_dabs, search_fixed_ef, DabsGraph};

/// Statistics returned with every search.
#[derive(Debug, Clone)]
pub struct SearchStats {
    /// Number of full-dimension distance computations performed.
    pub dist_computations: usize,
}

/// How to run a search.
#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    /// Exhaustive O(n·D) scan — ground truth / baseline.
    Flat,
    /// Standard HNSW-style beam search with fixed expansion width.
    FixedEf { ef: usize },
    /// Distance Adaptive Beam Search (NeurIPS 2025).
    ///
    /// Terminates once `d(q, closest_unexplored) > (1 + gamma) * d(q, k_th_result)`.
    Dabs { gamma: f32 },
}

/// Index wrapping a greedy k-NN graph with both fixed-ef and DABS search paths.
pub struct DabsIndex {
    graph: DabsGraph,
}

impl DabsIndex {
    /// Build from a collection of equal-length f32 vectors.
    ///
    /// `m` — max neighbors per node (≥8 recommended; more = better recall, slower build).
    pub fn build(vectors: Vec<Vec<f32>>, m: usize) -> Result<Self, DabsError> {
        let graph = DabsGraph::build(vectors, m)?;
        Ok(Self { graph })
    }

    pub fn len(&self) -> usize { self.graph.len() }
    pub fn is_empty(&self) -> bool { self.graph.is_empty() }
    pub fn dim(&self) -> usize { self.graph.dim() }

    /// Search with the given mode. Returns `(sorted neighbor ids, stats)`.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        mode: SearchMode,
    ) -> Result<(Vec<u32>, SearchStats), DabsError> {
        let n = self.graph.len();
        if k > n {
            return Err(DabsError::KExceedsDataset { k, n });
        }
        if let SearchMode::Dabs { gamma } = mode {
            if gamma < 0.0 {
                return Err(DabsError::InvalidGamma { gamma });
            }
        }

        let (pairs, dist_computations) = match mode {
            SearchMode::Flat => {
                let mut dists: Vec<(u32, f32)> = (0..n)
                    .map(|i| (i as u32, l2_sq(query, self.graph.row(i))))
                    .collect();
                dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                dists.truncate(k);
                (dists, n)
            }
            SearchMode::FixedEf { ef } => search_fixed_ef(&self.graph, query, k, ef),
            SearchMode::Dabs { gamma } => search_dabs(&self.graph, query, k, gamma),
        };

        let ids: Vec<u32> = pairs.into_iter().map(|(id, _)| id).collect();
        Ok((ids, SearchStats { dist_computations }))
    }
}

/// Recall@k: fraction of ground-truth ids that appear in `results`.
pub fn recall_at_k(results: &[u32], ground_truth: &[u32]) -> f64 {
    if ground_truth.is_empty() { return 1.0; }
    let k = results.len().min(ground_truth.len());
    let hits = ground_truth[..k]
        .iter()
        .filter(|&&gt| results.contains(&gt))
        .count();
    hits as f64 / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0_f32, 1.0).unwrap();
        (0..n)
            .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    }

    #[test]
    fn flat_is_exact() {
        let data = uniform_data(100, 16);
        let idx = DabsIndex::build(data.clone(), 8).unwrap();
        let query = &data[0];
        let (ids, stats) = idx.search(query, 1, SearchMode::Flat).unwrap();
        assert_eq!(ids[0], 0, "nearest to a dataset vector is itself");
        assert_eq!(stats.dist_computations, 100);
    }

    #[test]
    fn fixed_ef_recall_high() {
        let data = uniform_data(500, 32);
        let idx = DabsIndex::build(data.clone(), 16).unwrap();
        let query = &data[7];
        let (gt, _) = idx.search(query, 10, SearchMode::Flat).unwrap();
        let (res, _) = idx.search(query, 10, SearchMode::FixedEf { ef: 64 }).unwrap();
        let r = recall_at_k(&res, &gt);
        assert!(r >= 0.7, "recall@10 should be ≥0.7 at ef=64, got {r:.3}");
    }

    #[test]
    fn dabs_recall_comparable_to_fixed_ef() {
        let data = uniform_data(500, 32);
        let idx = DabsIndex::build(data.clone(), 16).unwrap();
        let query = &data[42];
        let (gt, _) = idx.search(query, 10, SearchMode::Flat).unwrap();
        let (res_ef, stats_ef) = idx.search(query, 10, SearchMode::FixedEf { ef: 64 }).unwrap();
        let (res_dabs, stats_dabs) = idx.search(query, 10, SearchMode::Dabs { gamma: 0.5 }).unwrap();
        let r_ef = recall_at_k(&res_ef, &gt);
        let r_dabs = recall_at_k(&res_dabs, &gt);
        println!(
            "fixed_ef recall={r_ef:.3} ops={}, dabs recall={r_dabs:.3} ops={}",
            stats_ef.dist_computations, stats_dabs.dist_computations
        );
        // DABS recall within 15% of fixed-ef (tolerant for small n=500)
        assert!(r_dabs >= r_ef * 0.85, "dabs recall too low: {r_dabs:.3} vs ef {r_ef:.3}");
    }

    #[test]
    fn recall_at_k_perfect() {
        let r = recall_at_k(&[0, 1, 2], &[0, 1, 2]);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn recall_at_k_zero() {
        let r = recall_at_k(&[3, 4, 5], &[0, 1, 2]);
        assert_eq!(r, 0.0);
    }
}
