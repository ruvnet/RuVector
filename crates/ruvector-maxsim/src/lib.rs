//! # ruvector-maxsim — ColBERT-style multi-vector MaxSim late interaction search
//!
//! Standard single-vector nearest-neighbor search compresses every document
//! into one embedding, losing information when a document covers multiple
//! topics. **Late interaction** (Khattab & Zaharia, ColBERT 2020) instead
//! stores *K token vectors* per document and scores a query as
//!
//! ```text
//! score(Q, D) = Σ_{q ∈ Q}  max_{d ∈ D}  cosine(q, d)
//! ```
//!
//! This preserves facet-level structure: a document about "Rust" AND "memory
//! safety" is discoverable by queries mentioning either topic alone.
//!
//! ## Index variants
//!
//! | Type | Recall | Latency | Notes |
//! |------|--------|---------|-------|
//! | [`FlatMaxSim`]   | 100% (oracle) | O(N·Td·Tq·D) | ground truth baseline |
//! | [`BucketMaxSim`] | ≈85-95%       | O(M·Td·Tq·D) | centroid pre-filter    |
//! | [`HnswMaxSim`]   | ≈80-90%       | sublinear     | PLAID-style NSW graph  |
//!
//! ## Quick start
//!
//! ```rust
//! use ruvector_maxsim::{FlatMaxSim, MultiVecIndex};
//! use ruvector_maxsim::types::{DocId, MultiVecDoc, MultiVecQuery};
//!
//! let mut idx = FlatMaxSim::new(4);
//! idx.add(MultiVecDoc {
//!     id: DocId(1),
//!     vecs: vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
//! }).unwrap();
//! let q = MultiVecQuery { vecs: vec![vec![1.0, 0.0, 0.0, 0.0]] };
//! let results = idx.search(&q, 1).unwrap();
//! assert_eq!(results[0].doc_id, DocId(1));
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bucket;
pub mod error;
pub mod flat;
pub mod graph;
pub mod hnsw;
pub mod score;
pub mod types;

pub use bucket::BucketMaxSim;
pub use error::MaxSimError;
pub use flat::FlatMaxSim;
pub use graph::GraphMaxSim;
pub use hnsw::HnswMaxSim;
pub use types::{DocId, MultiVecDoc, MultiVecQuery, RunStats, SearchResult};

/// The unified trait all MaxSim index variants implement.
///
/// A conformant implementation must:
/// 1. Reject vectors of incorrect dimension with [`MaxSimError::DimensionMismatch`].
/// 2. Score documents with the MaxSim kernel (or an approximation thereof).
/// 3. Return results in descending score order.
pub trait MultiVecIndex {
    /// Add a multi-vector document to the index.
    fn add(&mut self, doc: MultiVecDoc) -> Result<(), MaxSimError>;

    /// Search for the top-k documents most similar to the query.
    fn search(&self, query: &MultiVecQuery, k: usize) -> Result<Vec<SearchResult>, MaxSimError>;

    /// Number of documents indexed.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Embedding dimension.
    fn dims(&self) -> usize;
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn make_doc(id: u64, vecs: Vec<Vec<f32>>) -> MultiVecDoc {
        MultiVecDoc {
            id: DocId(id),
            vecs,
        }
    }

    fn make_query(vecs: Vec<Vec<f32>>) -> MultiVecQuery {
        MultiVecQuery { vecs }
    }

    /// The multi-token advantage: a document with two orthogonal topic vectors
    /// should beat a single-vector document for a query spanning both topics.
    fn multi_token_advantage<I: MultiVecIndex>(mut idx: I) {
        // Doc 1: covers topic A and topic B
        idx.add(make_doc(1, vec![vec![1.0, 0.0], vec![0.0, 1.0]]))
            .unwrap();
        // Doc 2: covers only topic A
        idx.add(make_doc(2, vec![vec![1.0, 0.0]])).unwrap();
        // Query asks about both topics
        let q = make_query(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let res = idx.search(&q, 2).unwrap();
        assert_eq!(res[0].doc_id, DocId(1), "multi-token doc should rank first");
        assert!(
            res[0].score > res[1].score,
            "score gap: {:.3} vs {:.3}",
            res[0].score,
            res[1].score
        );
    }

    #[test]
    fn flat_multi_token_advantage() {
        multi_token_advantage(FlatMaxSim::new(2));
    }

    #[test]
    fn bucket_multi_token_advantage() {
        multi_token_advantage(BucketMaxSim::new(2, 8));
    }

    #[test]
    fn hnsw_multi_token_advantage() {
        multi_token_advantage(HnswMaxSim::new(2, 16));
    }

    /// Dimension mismatch should return an error, not panic.
    fn rejects_bad_dims<I: MultiVecIndex>(mut idx: I) {
        idx.add(make_doc(1, vec![vec![1.0, 0.0]])).unwrap();
        let bad = make_doc(2, vec![vec![1.0, 0.0, 0.0]]);
        assert!(idx.add(bad).is_err(), "should reject wrong dimension");
    }

    #[test]
    fn flat_rejects_bad_dims() {
        rejects_bad_dims(FlatMaxSim::new(2));
    }

    #[test]
    fn bucket_rejects_bad_dims() {
        rejects_bad_dims(BucketMaxSim::new(2, 4));
    }

    #[test]
    fn hnsw_rejects_bad_dims() {
        rejects_bad_dims(HnswMaxSim::new(2, 8));
    }
}
