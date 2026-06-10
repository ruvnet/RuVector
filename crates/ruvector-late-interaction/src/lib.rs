/// Late interaction multi-vector (MaxSim / ColBERT-style) retrieval for RuVector.
///
/// Three variants with a common trait:
///   - `BruteForceIndex`  — exact O(N·T_d·T_q·D) scan, ground-truth baseline
///   - `PlaidLiteIndex`   — centroid pre-filter (PLAID-style), then full MaxSim on shortlist
///   - `CompressedIndex`  — SQ8 quantized tokens, int8 dot products
pub mod brute;
pub mod compressed;
pub mod dataset;
pub mod maxsim;
pub mod plaid;

use std::collections::HashSet;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LiError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("empty corpus")]
    EmptyCorpus,
    #[error("index not built: call build() first")]
    NotBuilt,
    #[error("k must be > 0")]
    InvalidK,
}

pub type Result<T> = std::result::Result<T, LiError>;

/// A document with one embedding per token (num_tokens × dim).
#[derive(Debug, Clone)]
pub struct MultiVecDoc {
    pub id: u64,
    /// L2-normalised token embeddings, shape [num_tokens][dim].
    pub tokens: Vec<Vec<f32>>,
}

impl MultiVecDoc {
    pub fn new(id: u64, tokens: Vec<Vec<f32>>) -> Self {
        Self { id, tokens }
    }
}

/// A query with one embedding per token.
#[derive(Debug, Clone)]
pub struct MultiVecQuery {
    /// L2-normalised token embeddings, shape [num_query_tokens][dim].
    pub tokens: Vec<Vec<f32>>,
}

impl MultiVecQuery {
    pub fn new(tokens: Vec<Vec<f32>>) -> Self {
        Self { tokens }
    }
}

/// A document paired with its retrieval score.
#[derive(Debug, Clone)]
pub struct ScoredDoc {
    pub id: u64,
    pub score: f32,
}

/// Core trait for late-interaction (MaxSim) indexes.
pub trait MaxSimIndex {
    fn insert(&mut self, doc: MultiVecDoc) -> Result<()>;
    fn build(&mut self) -> Result<()>;
    fn query(&self, q: &MultiVecQuery, top_k: usize) -> Result<Vec<ScoredDoc>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn dim(&self) -> usize;
    fn name(&self) -> &'static str;
    /// Estimated heap bytes for the stored token matrix.
    fn memory_bytes(&self) -> usize;
}

/// Recall\@k: fraction of ground-truth top-k IDs present in `results`.
pub fn recall_at_k(results: &[ScoredDoc], ground_truth: &[ScoredDoc], k: usize) -> f32 {
    let k = k.min(results.len()).min(ground_truth.len());
    if k == 0 {
        return 0.0;
    }
    let gt: HashSet<u64> = ground_truth.iter().take(k).map(|d| d.id).collect();
    let hits = results
        .iter()
        .take(k)
        .filter(|d| gt.contains(&d.id))
        .count();
    hits as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{brute::BruteForceIndex, compressed::CompressedIndex, plaid::PlaidLiteIndex};
    use dataset::DatasetGen;

    fn build_index<I: MaxSimIndex>(mut idx: I, docs: &[MultiVecDoc]) -> I {
        for d in docs {
            idx.insert(d.clone()).unwrap();
        }
        idx.build().unwrap();
        idx
    }

    #[test]
    fn brute_force_top1_is_self() {
        let gen = DatasetGen::new(42, 64);
        let docs = gen.random_docs(20, 8);
        let queries = gen.random_queries(5, 4);
        let idx = build_index(BruteForceIndex::new(64), &docs);
        for (qi, q) in queries.iter().enumerate() {
            let results = idx.query(q, 1).unwrap();
            assert_eq!(results.len(), 1, "query {qi} returned no results");
        }
    }

    #[test]
    fn compressed_recall_against_brute() {
        let gen = DatasetGen::new(7, 32);
        let docs = gen.random_docs(200, 8);
        let queries = gen.random_queries(10, 4);

        let bf = build_index(BruteForceIndex::new(32), &docs);
        let cmp = build_index(CompressedIndex::new(32), &docs);

        let mut total_recall = 0.0f32;
        for q in &queries {
            let gt = bf.query(q, 10).unwrap();
            let res = cmp.query(q, 10).unwrap();
            total_recall += recall_at_k(&res, &gt, 10);
        }
        let avg = total_recall / queries.len() as f32;
        assert!(
            avg >= 0.70,
            "SQ8 compressed recall@10 too low: {avg:.3} (expected ≥ 0.70)"
        );
    }

    #[test]
    fn plaid_recall_against_brute() {
        let gen = DatasetGen::new(13, 32);
        let docs = gen.random_docs(400, 8);
        let queries = gen.random_queries(10, 4);

        let bf = build_index(BruteForceIndex::new(32), &docs);
        let plaid = build_index(PlaidLiteIndex::new(32, 32, 4), &docs);

        let mut total_recall = 0.0f32;
        for q in &queries {
            let gt = bf.query(q, 10).unwrap();
            let res = plaid.query(q, 10).unwrap();
            total_recall += recall_at_k(&res, &gt, 10);
        }
        let avg = total_recall / queries.len() as f32;
        assert!(
            avg >= 0.65,
            "PLAID-lite recall@10 too low: {avg:.3} (expected ≥ 0.65)"
        );
    }

    #[test]
    fn recall_at_k_perfect() {
        let gt: Vec<ScoredDoc> = (0..10)
            .map(|i| ScoredDoc {
                id: i,
                score: 10.0 - i as f32,
            })
            .collect();
        assert!((recall_at_k(&gt, &gt, 10) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn memory_bytes_brute_matches_formula() {
        let gen = DatasetGen::new(99, 64);
        let docs = gen.random_docs(100, 16);
        let mut idx = BruteForceIndex::new(64);
        for d in &docs {
            idx.insert(d.clone()).unwrap();
        }
        idx.build().unwrap();
        // 100 docs × 16 tokens × 64 dims × 4 bytes = 4_096_000 bytes
        let expected = 100 * 16 * 64 * 4;
        assert_eq!(idx.memory_bytes(), expected);
    }
}
