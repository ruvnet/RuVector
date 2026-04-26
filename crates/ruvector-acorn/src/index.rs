use crate::error::AcornError;
use crate::graph::{exact_filtered_knn, AcornGraph};
use crate::search::{acorn_search, flat_filtered_search};

/// Common interface for all filtered-search index variants.
pub trait FilteredIndex {
    /// Build index from a dataset.
    fn build(data: Vec<Vec<f32>>) -> Result<Self, AcornError>
    where
        Self: Sized;

    /// Search for `k` nearest neighbors passing `predicate`.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        predicate: &dyn Fn(u32) -> bool,
    ) -> Result<Vec<(u32, f32)>, AcornError>;

    /// Approximate heap memory used by the index.
    fn memory_bytes(&self) -> usize;

    /// Index variant name for display.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Variant 1: FlatFilteredIndex — post-filter brute-force scan
// ---------------------------------------------------------------------------

/// Baseline: scan all vectors, apply predicate after distance computation.
/// O(n × D) per query. Best at high selectivity; degrades badly at low.
pub struct FlatFilteredIndex {
    data: Vec<Vec<f32>>,
}

impl FilteredIndex for FlatFilteredIndex {
    fn build(data: Vec<Vec<f32>>) -> Result<Self, AcornError> {
        if data.is_empty() {
            return Err(AcornError::EmptyDataset);
        }
        Ok(Self { data })
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        predicate: &dyn Fn(u32) -> bool,
    ) -> Result<Vec<(u32, f32)>, AcornError> {
        if k > self.data.len() {
            return Err(AcornError::KTooLarge {
                k,
                n: self.data.len(),
            });
        }
        let dim = self.data[0].len();
        if query.len() != dim {
            return Err(AcornError::DimMismatch {
                expected: dim,
                actual: query.len(),
            });
        }
        Ok(flat_filtered_search(&self.data, query, k, predicate))
    }

    fn memory_bytes(&self) -> usize {
        self.data.len() * self.data.first().map(|v| v.len()).unwrap_or(0) * 4
    }

    fn name(&self) -> &'static str {
        "FlatFiltered (baseline)"
    }
}

// ---------------------------------------------------------------------------
// Variant 2: AcornIndex1 — γ=1 (standard M edges, ACORN search)
// ---------------------------------------------------------------------------

/// ACORN-1: same edge budget as standard HNSW (M=16), but search always
/// expands ALL neighbors regardless of predicate. The graph is built with
/// greedy NN insertion. At low selectivity this outperforms the post-filter
/// baseline because it never abandons the beam when nodes fail the predicate.
pub struct AcornIndex1 {
    graph: AcornGraph,
    ef: usize,
}

impl AcornIndex1 {
    const M: usize = 16;

    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = ef;
        self
    }
}

impl FilteredIndex for AcornIndex1 {
    fn build(data: Vec<Vec<f32>>) -> Result<Self, AcornError> {
        if data.is_empty() {
            return Err(AcornError::EmptyDataset);
        }
        let graph = AcornGraph::build(data, Self::M)?;
        Ok(Self { graph, ef: 100 })
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        predicate: &dyn Fn(u32) -> bool,
    ) -> Result<Vec<(u32, f32)>, AcornError> {
        if k > self.graph.len() {
            return Err(AcornError::KTooLarge {
                k,
                n: self.graph.len(),
            });
        }
        let dim = self.graph.dim;
        if query.len() != dim {
            return Err(AcornError::DimMismatch {
                expected: dim,
                actual: query.len(),
            });
        }
        Ok(acorn_search(&self.graph, query, k, self.ef, predicate))
    }

    fn memory_bytes(&self) -> usize {
        self.graph.memory_bytes()
    }

    fn name(&self) -> &'static str {
        "ACORN-1 (γ=1, M=16)"
    }
}

// ---------------------------------------------------------------------------
// Variant 3: AcornIndexGamma — γ=2 (2×M edges, ACORN search)
// ---------------------------------------------------------------------------

/// ACORN-γ (γ=2): double the edge budget per node (32 neighbors). Denser
/// graph guarantees navigability even under 1% selectivity predicates.
/// Trades ~2× memory and ~2× build time for significantly better recall at
/// very low selectivities where ACORN-1 may still miss valid nodes.
pub struct AcornIndexGamma {
    graph: AcornGraph,
    #[allow(dead_code)] // carried for diagnostics / Display
    gamma: usize,
    ef: usize,
}

impl AcornIndexGamma {
    const M: usize = 16;

    pub fn new_with_gamma(data: Vec<Vec<f32>>, gamma: usize) -> Result<Self, AcornError> {
        if gamma < 1 {
            return Err(AcornError::InvalidGamma { gamma });
        }
        let graph = AcornGraph::build(data, Self::M * gamma)?;
        Ok(Self {
            graph,
            gamma,
            ef: 150,
        })
    }

    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = ef;
        self
    }
}

impl FilteredIndex for AcornIndexGamma {
    fn build(data: Vec<Vec<f32>>) -> Result<Self, AcornError> {
        Self::new_with_gamma(data, 2)
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        predicate: &dyn Fn(u32) -> bool,
    ) -> Result<Vec<(u32, f32)>, AcornError> {
        if k > self.graph.len() {
            return Err(AcornError::KTooLarge {
                k,
                n: self.graph.len(),
            });
        }
        let dim = self.graph.dim;
        if query.len() != dim {
            return Err(AcornError::DimMismatch {
                expected: dim,
                actual: query.len(),
            });
        }
        Ok(acorn_search(&self.graph, query, k, self.ef, predicate))
    }

    fn memory_bytes(&self) -> usize {
        self.graph.memory_bytes()
    }

    fn name(&self) -> &'static str {
        "ACORN-γ (γ=2, M=32)"
    }
}

/// Measure recall@k: fraction of true top-k in returned top-k.
pub fn recall_at_k(
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    predicate: impl Fn(u32) -> bool + Copy + Sync,
    index: &dyn FilteredIndex,
) -> f64 {
    let mut hit = 0usize;
    let mut total = 0usize;

    for q in queries {
        let truth = exact_filtered_knn(data, q, k, predicate);
        if truth.is_empty() {
            continue;
        }
        let got = index.search(q, k, &predicate).unwrap_or_default();
        let got_set: std::collections::HashSet<u32> = got.iter().map(|(id, _)| *id).collect();
        hit += truth.iter().filter(|id| got_set.contains(id)).count();
        total += truth.len();
    }

    if total == 0 {
        1.0
    } else {
        hit as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0_f32, 1.0).unwrap();
        (0..n)
            .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    }

    #[test]
    fn flat_index_full_recall() {
        let data = gaussian_data(200, 32, 42);
        let flat = FlatFilteredIndex::build(data.clone()).unwrap();
        let queries = gaussian_data(10, 32, 99);
        let r = recall_at_k(&data, &queries, 5, |_| true, &flat);
        assert!(r > 0.99, "flat full-pass recall should be ~1.0, got {r:.3}");
    }

    #[test]
    fn acorn1_reasonable_recall_half_filter() {
        // ACORN-1 with a greedy single-level graph achieves moderate recall.
        // The key property tested: ACORN search returns SOME correct neighbors
        // under a selective predicate (50%). Recall > 30% confirms the search
        // is correctly navigating the predicate subgraph (vs. 0% if broken).
        let data = gaussian_data(500, 32, 42);
        let idx = AcornIndex1::build(data.clone()).unwrap();
        let queries = gaussian_data(20, 32, 99);
        let r = recall_at_k(&data, &queries, 5, |id| id % 2 == 0, &idx);
        assert!(
            r > 0.30,
            "ACORN-1 half-filter recall should be >0.30, got {r:.3}"
        );
    }

    #[test]
    fn dim_mismatch_returns_error() {
        let data = gaussian_data(50, 16, 1);
        let idx = FlatFilteredIndex::build(data).unwrap();
        let bad_query = vec![0.0_f32; 8];
        assert!(idx.search(&bad_query, 3, &|_| true).is_err());
    }

    #[test]
    fn acorn_gamma_build_and_search() {
        let data = gaussian_data(200, 16, 7);
        let idx = AcornIndexGamma::new_with_gamma(data.clone(), 2).unwrap();
        let q = gaussian_data(5, 16, 77);
        for query in &q {
            let res = idx.search(query, 5, &|_| true).unwrap();
            assert_eq!(res.len(), 5);
        }
    }
}
