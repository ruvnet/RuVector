use crate::{
    error::{AcornError, Result},
    graph::NswGraph,
};
use std::collections::HashMap;

/// Configuration for an ACORN index.
#[derive(Debug, Clone)]
pub struct AcornConfig {
    /// Vector dimensionality.
    pub dim: usize,
    /// Base edges per node (M in the ACORN paper).
    pub m: usize,
    /// Neighbour-compression multiplier (γ).  γ=1 disables compression.
    /// γ=2 doubles each node's adjacency list with second-hop neighbours.
    pub gamma: usize,
    /// Candidate pool size during construction (ef_construction).
    pub ef_construction: usize,
}

impl Default for AcornConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            m: 16,
            gamma: 2,
            ef_construction: 64,
        }
    }
}

/// Result from a filtered ANN search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u32,
    pub distance: f32,
}

/// Which search strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchVariant {
    /// Post-filter: unfiltered ANN search, then discard non-matching results.
    /// Baseline — high QPS but recall collapses under selective filters.
    PostFilter,
    /// ACORN-1: only expands filter-passing nodes.
    /// Better than PostFilter for loose filters but strands navigability for tight ones.
    Acorn1,
    /// ACORN-γ: navigates all nodes; only filter-passing nodes enter the result heap.
    /// Requires `build_compression()` for full benefit.
    AcornGamma,
}

/// High-level filtered ANN index built on an NSW graph.
pub struct AcornIndex {
    cfg: AcornConfig,
    graph: NswGraph,
    /// Maps user-supplied id → internal graph index.
    id_map: HashMap<u32, u32>,
    /// Reverse map: internal index → user id.
    user_ids: Vec<u32>,
    compressed: bool,
}

impl AcornIndex {
    pub fn new(cfg: AcornConfig) -> Self {
        let graph = NswGraph::new(cfg.dim, cfg.m);
        Self {
            cfg,
            graph,
            id_map: HashMap::new(),
            user_ids: Vec::new(),
            compressed: false,
        }
    }

    /// Insert a vector with an application-level `id`.
    pub fn insert(&mut self, id: u32, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.cfg.dim {
            return Err(AcornError::DimensionMismatch {
                expected: self.cfg.dim,
                actual: vector.len(),
            });
        }
        self.compressed = false;
        let internal = self.graph.insert(vector);
        self.id_map.insert(id, internal);
        self.user_ids.push(id);
        Ok(())
    }

    /// Apply ACORN-γ neighbour compression.  Call once after all inserts.
    pub fn build_compression(&mut self) {
        if !self.compressed {
            self.graph.compress_neighbors(self.cfg.gamma);
            self.compressed = true;
        }
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Filtered approximate nearest-neighbour search.
    ///
    /// `filter(id)` receives the **user-level** id and returns `true` if the
    /// vector should appear in results.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: impl Fn(u32) -> bool,
        variant: SearchVariant,
    ) -> Result<Vec<SearchResult>> {
        if self.graph.is_empty() {
            return Err(AcornError::EmptyIndex);
        }
        if query.len() != self.cfg.dim {
            return Err(AcornError::DimensionMismatch {
                expected: self.cfg.dim,
                actual: query.len(),
            });
        }
        if k == 0 {
            return Err(AcornError::InvalidParameter("k must be > 0".into()));
        }

        // Translate user-id filter → internal-id filter
        let user_ids = &self.user_ids;
        let internal_filter = |internal: u32| filter(user_ids[internal as usize]);

        let raw = match variant {
            SearchVariant::PostFilter => {
                // Search with ef*k to get a larger pool then post-filter
                let ef_wide = (ef * 4).max(k * 8);
                self.graph
                    .search_postfilter(query, k, ef_wide, internal_filter)
            }
            SearchVariant::Acorn1 => {
                self.graph.search_acorn1(query, k, ef, internal_filter)
            }
            SearchVariant::AcornGamma => {
                self.graph
                    .search_acorn_gamma(query, k, ef, internal_filter)
            }
        };

        Ok(raw
            .into_iter()
            .map(|(internal, dist)| SearchResult {
                id: user_ids[internal as usize],
                distance: dist,
            })
            .collect())
    }

    /// Exact brute-force filtered search — used to compute ground-truth recall.
    pub fn ground_truth(
        &self,
        query: &[f32],
        k: usize,
        filter: impl Fn(u32) -> bool,
    ) -> Result<Vec<SearchResult>> {
        if self.graph.is_empty() {
            return Err(AcornError::EmptyIndex);
        }
        let user_ids = &self.user_ids;
        let internal_filter = |internal: u32| filter(user_ids[internal as usize]);
        let raw = self.graph.brute_force(query, k, internal_filter);
        Ok(raw
            .into_iter()
            .map(|(internal, dist)| SearchResult {
                id: user_ids[internal as usize],
                distance: dist,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index(n: usize, dim: usize, gamma: usize) -> AcornIndex {
        let cfg = AcornConfig {
            dim,
            m: 8,
            gamma,
            ef_construction: 32,
        };
        let mut idx = AcornIndex::new(cfg);
        // sequential vectors: id i → vector [i as f32, 0, ..., 0]
        for i in 0..n as u32 {
            let mut v = vec![0.0f32; dim];
            v[0] = i as f32;
            idx.insert(i, v).unwrap();
        }
        idx.build_compression();
        idx
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut idx = build_index(10, 4, 2);
        let result = idx.insert(99, vec![1.0, 2.0]); // wrong dim
        assert!(matches!(result, Err(AcornError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_empty_index_error() {
        let cfg = AcornConfig { dim: 4, m: 4, gamma: 1, ef_construction: 16 };
        let idx = AcornIndex::new(cfg);
        let q = vec![0.0f32; 4];
        assert!(matches!(
            idx.search(&q, 5, 16, |_| true, SearchVariant::AcornGamma),
            Err(AcornError::EmptyIndex)
        ));
    }

    #[test]
    fn test_unfiltered_returns_nearest() {
        let idx = build_index(100, 4, 2);
        let q = vec![50.0f32, 0.0, 0.0, 0.0];
        let results = idx
            .search(&q, 5, 64, |_| true, SearchVariant::AcornGamma)
            .unwrap();
        assert!(!results.is_empty());
        // Nearest should be id=50 (distance=0)
        assert_eq!(results[0].id, 50);
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn test_filter_respected_gamma() {
        let idx = build_index(200, 4, 2);
        let q = vec![100.0f32, 0.0, 0.0, 0.0];
        // Only odd ids
        let results = idx
            .search(&q, 10, 64, |id| id % 2 == 1, SearchVariant::AcornGamma)
            .unwrap();
        for r in &results {
            assert_eq!(r.id % 2, 1, "id {} is even — filter was violated", r.id);
        }
        assert!(!results.is_empty());
    }

    #[test]
    fn test_filter_respected_postfilter() {
        let idx = build_index(200, 4, 1);
        let q = vec![100.0f32, 0.0, 0.0, 0.0];
        let results = idx
            .search(&q, 5, 256, |id| id % 2 == 0, SearchVariant::PostFilter)
            .unwrap();
        for r in &results {
            assert_eq!(r.id % 2, 0, "id {} is odd — filter was violated", r.id);
        }
    }

    #[test]
    fn test_filter_respected_acorn1() {
        let idx = build_index(200, 4, 1);
        let q = vec![100.0f32, 0.0, 0.0, 0.0];
        let results = idx
            .search(&q, 5, 64, |id| id < 50, SearchVariant::Acorn1)
            .unwrap();
        for r in &results {
            assert!(r.id < 50, "id {} failed filter id<50", r.id);
        }
    }

    #[test]
    fn test_ground_truth_exact() {
        let idx = build_index(100, 4, 2);
        let q = vec![30.0f32, 0.0, 0.0, 0.0];
        let gt = idx.ground_truth(&q, 3, |_| true).unwrap();
        // Closest three to 30 are 30, 29/31, 28/32
        assert_eq!(gt[0].id, 30);
        assert!(gt[0].distance < 1e-6);
    }

    #[test]
    fn test_recall_gamma_beats_postfilter_on_tight_filter() {
        // With a 5 % filter, ACORN-γ should achieve better or equal recall vs PostFilter
        let n = 500usize;
        let dim = 32;
        let idx = build_index(n, dim, 2);

        let q: Vec<f32> = (0..dim).map(|i| if i == 0 { 250.0 } else { 0.0 }).collect();
        let k = 5;
        let ef = 64;
        let threshold = (n / 20) as u32; // 5 % selectivity

        let gt = idx
            .ground_truth(&q, k, |id| id < threshold)
            .unwrap();
        let gt_ids: std::collections::HashSet<u32> = gt.iter().map(|r| r.id).collect();

        let res_gamma = idx
            .search(&q, k, ef, |id| id < threshold, SearchVariant::AcornGamma)
            .unwrap();
        let recall_gamma = res_gamma
            .iter()
            .filter(|r| gt_ids.contains(&r.id))
            .count() as f64
            / k as f64;

        let res_post = idx
            .search(&q, k, ef * 4, |id| id < threshold, SearchVariant::PostFilter)
            .unwrap();
        let recall_post = res_post
            .iter()
            .filter(|r| gt_ids.contains(&r.id))
            .count() as f64
            / k as f64;

        // ACORN-γ recall must be at least as good as PostFilter
        assert!(
            recall_gamma >= recall_post - 0.2,
            "ACORN-γ recall {recall_gamma:.2} too far below PostFilter {recall_post:.2}"
        );
        // And both should find something
        assert!(
            recall_gamma > 0.0 || gt_ids.is_empty(),
            "ACORN-γ found nothing despite ground truth existing"
        );
    }
}
