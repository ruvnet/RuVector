//! # ruvector-hybrid — Hybrid Sparse-Dense Search (BM25 + ANN + RRF)
//!
//! Three search backends unified under common traits:
//! - [`Bm25Index`] — Robertson BM25 lexical sparse retrieval
//! - [`FlatDenseIndex`] — exact cosine ANN (flat exhaustive scan)
//! - [`RrfHybridIndex`] — Reciprocal Rank Fusion combining both
//!
//! ## Design
//!
//! All backends implement either [`SparseSearch`], [`DenseSearch`], or
//! [`HybridSearch`].  A [`Document`] carries both textual tokens and a dense
//! embedding vector.  [`recall_at_k`] measures result quality against any
//! ground-truth set.
//!
//! See `docs/adr/ADR-256-hybrid-sparse-dense-search.md` for rationale and
//! `docs/research/nightly/2026-06-17-hybrid-sparse-dense/` for benchmarks.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bm25;
pub mod dense;
pub mod fusion;

pub use bm25::Bm25Index;
pub use dense::FlatDenseIndex;
pub use fusion::{RrfHybridIndex, RsfHybridIndex, ScoreFusionIndex};

/// A document carrying both tokenised text and a dense embedding.
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique document identifier (0-based, dense).
    pub id: usize,
    /// Pre-tokenised text tokens (caller controls tokenisation).
    pub tokens: Vec<String>,
    /// Dense embedding vector (any dimensionality; must match query dimension).
    pub vector: Vec<f32>,
}

/// A single ranked search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Document identifier matching [`Document::id`].
    pub id: usize,
    /// Relevance score — higher is better; scale is backend-specific.
    pub score: f32,
}

/// Lexical sparse search over tokenised text fields.
pub trait SparseSearch {
    /// Return at most `k` results ranked by BM25 score.
    fn search(&self, tokens: &[&str], k: usize) -> Vec<SearchResult>;
}

/// Approximate-nearest-neighbour search over dense embedding vectors.
pub trait DenseSearch {
    /// Return at most `k` results ranked by cosine similarity.
    fn search(&self, vector: &[f32], k: usize) -> Vec<SearchResult>;
}

/// Hybrid search combining sparse and dense signals.
pub trait HybridSearch {
    /// Return at most `k` results fused from both sparse and dense backends.
    fn search(&self, tokens: &[&str], vector: &[f32], k: usize) -> Vec<SearchResult>;
}

/// Recall@k: fraction of ground-truth items present in `returned`.
///
/// Returns 0.0 when `ground_truth` is empty.
pub fn recall_at_k(returned: &[SearchResult], ground_truth: &[usize]) -> f32 {
    if ground_truth.is_empty() {
        return 0.0;
    }
    let gt: std::collections::HashSet<usize> = ground_truth.iter().cloned().collect();
    let hits = returned.iter().filter(|r| gt.contains(&r.id)).count();
    hits as f32 / ground_truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_k_full() {
        let returned = vec![
            SearchResult { id: 0, score: 1.0 },
            SearchResult { id: 1, score: 0.9 },
        ];
        assert_eq!(recall_at_k(&returned, &[0, 1]), 1.0);
    }

    #[test]
    fn test_recall_at_k_partial() {
        let returned = vec![SearchResult { id: 0, score: 1.0 }];
        assert_eq!(recall_at_k(&returned, &[0, 1]), 0.5);
    }

    #[test]
    fn test_recall_at_k_empty_gt() {
        assert_eq!(recall_at_k(&[], &[]), 0.0);
    }
}
