//! # ruvector-hybrid
//!
//! Hybrid BM25 sparse + dense ANN vector search with Reciprocal Rank Fusion.
//!
//! Three search variants satisfy [`HybridSearch`]:
//! - [`DenseFlat`]: exact cosine flat scan over all stored vectors
//! - [`SparseBm25`]: BM25 keyword retrieval via an inverted index
//! - [`HybridRrf`]: Reciprocal Rank Fusion combining both signals
//!
//! # Design rationale
//!
//! Production RAG systems universally benefit from combining dense semantic
//! similarity with sparse keyword matching.  Dense-only retrieval misses
//! exact-match queries; sparse-only retrieval misses paraphrase and
//! synonymy.  RRF is the standard rank-based fusion approach used by
//! Vespa, Elasticsearch, and Qdrant Hybrid because it is parameter-free
//! and robust across very different score scales.
//!
//! See `docs/adr/ADR-194-hybrid-sparse-dense-search.md` and
//! `docs/research/nightly/2026-05-30-hybrid-sparse-dense-search/README.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod dataset;
pub mod dense;
pub mod fusion;
pub mod sparse;

pub use dataset::{generate, generate_structured, DatasetConfig, Document, Query};
pub use dense::DenseFlat;
pub use fusion::HybridRrf;
pub use sparse::SparseBm25;

/// A single search result returned by any search variant.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Document identifier (caller-assigned at insert time).
    pub id: u32,
    /// Score — higher is better.  Not comparable across variants.
    pub score: f32,
}

/// Core trait implemented by all three search variants.
pub trait HybridSearch {
    /// Insert a document with its dense embedding and sparse token IDs.
    ///
    /// Token IDs are caller-computed (e.g. from a vocab map); the index
    /// never sees raw text.
    fn insert(&mut self, id: u32, vector: &[f32], tokens: &[u32]);

    /// Return the top-`k` results for the given query.
    ///
    /// Both `query_vec` and `query_tokens` may be empty for variants that
    /// ignore one modality.
    fn search(&self, query_vec: &[f32], query_tokens: &[u32], k: usize) -> Vec<SearchResult>;

    /// Number of indexed documents.
    fn len(&self) -> usize;

    /// True when no documents have been indexed.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated heap memory used by the index in bytes.
    fn memory_bytes(&self) -> usize;
}

/// Compute recall@k: fraction of `ground_truth` IDs that appear in `results`.
pub fn recall_at_k(results: &[SearchResult], ground_truth: &[u32]) -> f32 {
    let k = ground_truth.len();
    if k == 0 {
        return 1.0;
    }
    let found = results
        .iter()
        .take(k)
        .filter(|r| ground_truth.contains(&r.id))
        .count();
    found as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_at_k_perfect() {
        let results = vec![
            SearchResult { id: 0, score: 1.0 },
            SearchResult { id: 1, score: 0.9 },
        ];
        assert_eq!(recall_at_k(&results, &[0, 1]), 1.0);
    }

    #[test]
    fn recall_at_k_zero() {
        let results = vec![SearchResult { id: 99, score: 1.0 }];
        assert_eq!(recall_at_k(&results, &[0, 1]), 0.0);
    }
}
