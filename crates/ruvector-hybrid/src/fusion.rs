//! Reciprocal Rank Fusion (RRF) hybrid index.
//!
//! RRF fuses a dense-cosine ranked list and a BM25 sparse ranked list by
//! summing reciprocal ranks:
//!
//! ```text
//! RRF(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_sparse(d))
//! ```
//!
//! The constant k=60 is the empirically established default from
//! Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
//! individual rank learning methods", SIGIR 2009.
//!
//! The hybrid index internalises both sub-indexes and exposes a single
//! [`HybridSearch`] surface.  Insert is forwarded to both sub-indexes;
//! search queries both sub-indexes and fuses their ranked lists.

use crate::{dense::DenseFlat, sparse::SparseBm25, HybridSearch, SearchResult};
use std::collections::HashMap;

/// Standard RRF constant — balances near-top rank changes vs. tail behaviour.
const RRF_K: f32 = 60.0;

/// How many candidates to retrieve from each leg before fusing.
/// Must be large enough so true top-k are always included in the pool;
/// a factor of 10× over k is sufficient for flat scan (all true top-k
/// are guaranteed present in the dense leg regardless of OVERSAMPLE).
const OVERSAMPLE: usize = 100;

/// Hybrid search via Reciprocal Rank Fusion of BM25 + cosine-dense lists.
///
/// Memory footprint = dense storage + sparse inverted index (both sub-indexes
/// are held in full — no compression is applied at this proof-of-concept stage).
pub struct HybridRrf {
    dense: DenseFlat,
    sparse: SparseBm25,
}

impl HybridRrf {
    /// Create an empty hybrid index for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            dense: DenseFlat::new(dim),
            sparse: SparseBm25::new(),
        }
    }
}

impl HybridSearch for HybridRrf {
    fn insert(&mut self, id: u32, vector: &[f32], tokens: &[u32]) {
        self.dense.insert(id, vector, tokens);
        self.sparse.insert(id, vector, tokens);
    }

    fn search(&self, query_vec: &[f32], query_tokens: &[u32], k: usize) -> Vec<SearchResult> {
        let n_cand = (k + OVERSAMPLE).max(k * 4);

        let dense_results = self.dense.search(query_vec, query_tokens, n_cand);
        let sparse_results = self.sparse.search(query_vec, query_tokens, n_cand);

        let mut rrf: HashMap<u32, f32> = HashMap::new();

        for (rank, r) in dense_results.iter().enumerate() {
            *rrf.entry(r.id).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
        }
        for (rank, r) in sparse_results.iter().enumerate() {
            *rrf.entry(r.id).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
        }

        let mut results: Vec<SearchResult> = rrf
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }

    fn len(&self) -> usize {
        self.dense.len()
    }

    fn memory_bytes(&self) -> usize {
        self.dense.memory_bytes() + self.sparse.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_small(dim: usize, n: usize) -> HybridRrf {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(1);
        let mut idx = HybridRrf::new(dim);
        for id in 0..n as u32 {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let tokens: Vec<u32> = (0..5).map(|_| rng.gen_range(0..20u32)).collect();
            idx.insert(id, &v, &tokens);
        }
        idx
    }

    #[test]
    fn len_correct() {
        let idx = build_small(16, 50);
        assert_eq!(idx.len(), 50);
    }

    #[test]
    fn returns_k_results() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(2);
        let idx = build_small(16, 100);
        let qv: Vec<f32> = (0..16).map(|_| rng.gen::<f32>()).collect();
        let qt: Vec<u32> = vec![1, 2, 3];
        let results = idx.search(&qv, &qt, 10);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn scores_descending() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(3);
        let idx = build_small(16, 200);
        let qv: Vec<f32> = (0..16).map(|_| rng.gen::<f32>()).collect();
        let qt: Vec<u32> = vec![5, 6];
        let results = idx.search(&qv, &qt, 20);
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "results must be sorted descending"
            );
        }
    }

    #[test]
    fn memory_positive() {
        let idx = build_small(32, 10);
        assert!(idx.memory_bytes() > 0);
    }
}
