//! ColBERT-style Multi-Vector Retrieval
//!
//! Implements late interaction retrieval where each document and query is
//! represented by multiple vectors (one per token or patch). Scoring uses
//! MaxSim: for each query token, find the maximum similarity across all
//! document tokens, then sum these maxima.
//!
//! # Scoring Variants
//!
//! - **MaxSim** (ColBERT default): sum of per-query-token max similarities
//! - **AvgSim**: average similarity across all query-doc token pairs
//! - **SumMax**: sum of per-document-token max similarities (inverse direction)
//!
//! # Example
//!
//! ```
//! use ruvector_core::advanced_features::multi_vector::*;
//! use ruvector_core::types::DistanceMetric;
//!
//! let config = MultiVectorConfig {
//!     metric: DistanceMetric::Cosine,
//!     scoring: ScoringVariant::MaxSim,
//! };
//! let mut index = MultiVectorIndex::new(config);
//! index.insert("doc1".into(), vec![vec![1.0, 0.0], vec![0.0, 1.0]], None).unwrap();
//! let results = index.search(&[vec![1.0, 0.0]], 10).unwrap();
//! assert_eq!(results[0].id, "doc1");
//! ```

use crate::error::{Result, RuvectorError};
use crate::types::{DistanceMetric, SearchResult, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single document entry containing multiple token embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorEntry {
    /// Unique document identifier.
    pub doc_id: VectorId,
    /// One embedding vector per token or patch.
    pub token_embeddings: Vec<Vec<f32>>,
    /// Precomputed L2 norms for each token embedding (used for cosine similarity).
    pub norms: Vec<f32>,
    /// Optional metadata associated with the document.
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Late-interaction scoring variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoringVariant {
    /// ColBERT default: for each query token, take the max similarity across
    /// all document tokens, then sum over query tokens.
    MaxSim,
    /// Average pairwise similarity across all query-document token pairs.
    AvgSim,
    /// For each *document* token, take the max similarity across all query
    /// tokens, then sum over document tokens.
    SumMax,
}

/// Configuration for the multi-vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorConfig {
    /// Distance metric used for token-level similarity.
    pub metric: DistanceMetric,
    /// Scoring variant for aggregating token-level similarities.
    pub scoring: ScoringVariant,
}

impl Default for MultiVectorConfig {
    fn default() -> Self {
        Self {
            metric: DistanceMetric::Cosine,
            scoring: ScoringVariant::MaxSim,
        }
    }
}

/// ColBERT-style multi-vector index supporting late interaction scoring.
///
/// Each document is stored as a set of token embeddings. At query time, every
/// query token is compared against every document token and the results are
/// aggregated according to the configured [`ScoringVariant`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorIndex {
    /// Index configuration.
    pub config: MultiVectorConfig,
    /// All stored document entries keyed by document ID.
    entries: HashMap<VectorId, MultiVectorEntry>,
}

impl MultiVectorIndex {
    /// Create a new empty multi-vector index.
    pub fn new(config: MultiVectorConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
        }
    }

    /// Insert a document represented by multiple token embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error if `embeddings` is empty or if any embedding has a
    /// different dimension than the first.
    pub fn insert(
        &mut self,
        doc_id: VectorId,
        embeddings: Vec<Vec<f32>>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<()> {
        if embeddings.is_empty() {
            return Err(RuvectorError::InvalidParameter(
                "Token embeddings cannot be empty".into(),
            ));
        }

        let dim = embeddings[0].len();
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != dim {
                return Err(RuvectorError::DimensionMismatch {
                    expected: dim,
                    actual: emb.len(),
                });
            }
            if emb.is_empty() {
                return Err(RuvectorError::InvalidParameter(
                    format!("Embedding at index {} has zero dimensions", i),
                ));
            }
        }

        let norms = embeddings.iter().map(|e| compute_norm(e)).collect();

        self.entries.insert(
            doc_id.clone(),
            MultiVectorEntry {
                doc_id,
                token_embeddings: embeddings,
                norms,
                metadata,
            },
        );

        Ok(())
    }

    /// Remove a document from the index.
    pub fn remove(&mut self, doc_id: &str) -> Option<MultiVectorEntry> {
        self.entries.remove(doc_id)
    }

    /// Return the number of documents in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Search the index using multi-vector query embeddings.
    ///
    /// Each element of `query_embeddings` represents one query token. The
    /// aggregated late-interaction score is computed for every document and the
    /// top-k results are returned in descending order of score.
    ///
    /// # Errors
    ///
    /// Returns an error if `query_embeddings` is empty.
    pub fn search(
        &self,
        query_embeddings: &[Vec<f32>],
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        if query_embeddings.is_empty() {
            return Err(RuvectorError::InvalidParameter(
                "Query embeddings cannot be empty".into(),
            ));
        }

        let query_norms: Vec<f32> = query_embeddings.iter().map(|q| compute_norm(q)).collect();

        let mut scored: Vec<(VectorId, f32)> = self
            .entries
            .values()
            .map(|entry| {
                let score = self.compute_score(
                    query_embeddings,
                    &query_norms,
                    &entry.token_embeddings,
                    &entry.norms,
                );
                (entry.doc_id.clone(), score)
            })
            .collect();

        // Sort descending by score (higher is more similar).
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored
            .into_iter()
            .map(|(id, score)| {
                let metadata = self.entries.get(&id).and_then(|e| e.metadata.clone());
                SearchResult {
                    id,
                    score,
                    vector: None,
                    metadata,
                }
            })
            .collect())
    }

    /// Search with a specific scoring variant, overriding the index default.
    pub fn search_with_scoring(
        &self,
        query_embeddings: &[Vec<f32>],
        top_k: usize,
        scoring: ScoringVariant,
    ) -> Result<Vec<SearchResult>> {
        let original = self.config.scoring;
        // We use a temporary clone to avoid mutating self.
        let mut temp = self.clone();
        temp.config.scoring = scoring;
        let results = temp.search(query_embeddings, top_k);
        // Restore is unnecessary since temp is dropped, but keeps intent clear.
        let _ = original;
        results
    }

    /// Compute the aggregated late-interaction score between query and document.
    fn compute_score(
        &self,
        query_embeddings: &[Vec<f32>],
        query_norms: &[f32],
        doc_embeddings: &[Vec<f32>],
        doc_norms: &[f32],
    ) -> f32 {
        match self.config.scoring {
            ScoringVariant::MaxSim => {
                self.maxsim(query_embeddings, query_norms, doc_embeddings, doc_norms)
            }
            ScoringVariant::AvgSim => {
                self.avgsim(query_embeddings, query_norms, doc_embeddings, doc_norms)
            }
            ScoringVariant::SumMax => {
                self.summax(query_embeddings, query_norms, doc_embeddings, doc_norms)
            }
        }
    }

    /// MaxSim: for each query token, find max similarity across doc tokens, sum the maxes.
    fn maxsim(
        &self,
        query_embeddings: &[Vec<f32>],
        query_norms: &[f32],
        doc_embeddings: &[Vec<f32>],
        doc_norms: &[f32],
    ) -> f32 {
        query_embeddings
            .iter()
            .enumerate()
            .map(|(qi, q)| {
                doc_embeddings
                    .iter()
                    .enumerate()
                    .map(|(di, d)| {
                        self.token_similarity(q, query_norms[qi], d, doc_norms[di])
                    })
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum()
    }

    /// AvgSim: average similarity across all query-document token pairs.
    fn avgsim(
        &self,
        query_embeddings: &[Vec<f32>],
        query_norms: &[f32],
        doc_embeddings: &[Vec<f32>],
        doc_norms: &[f32],
    ) -> f32 {
        let total_pairs = (query_embeddings.len() * doc_embeddings.len()) as f32;
        if total_pairs == 0.0 {
            return 0.0;
        }
        let sum: f32 = query_embeddings
            .iter()
            .enumerate()
            .flat_map(|(qi, q)| {
                doc_embeddings
                    .iter()
                    .enumerate()
                    .map(move |(di, d)| {
                        self.token_similarity(q, query_norms[qi], d, doc_norms[di])
                    })
            })
            .sum();
        sum / total_pairs
    }

    /// SumMax: for each doc token, find max similarity across query tokens, sum the maxes.
    fn summax(
        &self,
        query_embeddings: &[Vec<f32>],
        query_norms: &[f32],
        doc_embeddings: &[Vec<f32>],
        doc_norms: &[f32],
    ) -> f32 {
        doc_embeddings
            .iter()
            .enumerate()
            .map(|(di, d)| {
                query_embeddings
                    .iter()
                    .enumerate()
                    .map(|(qi, q)| {
                        self.token_similarity(q, query_norms[qi], d, doc_norms[di])
                    })
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum()
    }

    /// Compute token-level similarity using precomputed norms.
    #[inline]
    fn token_similarity(&self, a: &[f32], norm_a: f32, b: &[f32], norm_b: f32) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        match self.config.metric {
            DistanceMetric::Cosine => {
                let denom = norm_a * norm_b;
                if denom < f32::EPSILON {
                    0.0
                } else {
                    dot / denom
                }
            }
            DistanceMetric::DotProduct => dot,
            // For Euclidean and Manhattan we convert to a similarity-like score.
            DistanceMetric::Euclidean => {
                let dist_sq: f32 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum();
                1.0 / (1.0 + dist_sq.sqrt())
            }
            DistanceMetric::Manhattan => {
                let dist: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
                1.0 / (1.0 + dist)
            }
        }
    }
}

/// Compute the L2 norm of a vector.
#[inline]
fn compute_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_index() -> MultiVectorIndex {
        MultiVectorIndex::new(MultiVectorConfig::default())
    }

    #[test]
    fn test_insert_and_len() {
        let mut index = default_index();
        assert!(index.is_empty());
        index
            .insert("d1".into(), vec![vec![1.0, 0.0], vec![0.0, 1.0]], None)
            .unwrap();
        assert_eq!(index.len(), 1);
        index
            .insert("d2".into(), vec![vec![0.5, 0.5]], None)
            .unwrap();
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_insert_empty_embeddings_error() {
        let mut index = default_index();
        let res = index.insert("d1".into(), vec![], None);
        assert!(res.is_err());
    }

    #[test]
    fn test_insert_dimension_mismatch_error() {
        let mut index = default_index();
        let res = index.insert("d1".into(), vec![vec![1.0, 0.0], vec![1.0]], None);
        assert!(res.is_err());
    }

    #[test]
    fn test_maxsim_search_basic() {
        let mut index = default_index();
        // doc1: token embeddings pointing in x and y directions
        index
            .insert("doc1".into(), vec![vec![1.0, 0.0], vec![0.0, 1.0]], None)
            .unwrap();
        // doc2: token embedding pointing in x direction only
        index
            .insert("doc2".into(), vec![vec![1.0, 0.0]], None)
            .unwrap();

        // Query with a single token in x direction
        let results = index.search(&[vec![1.0, 0.0]], 10).unwrap();
        assert_eq!(results.len(), 2);
        // Both docs should have cosine similarity 1.0 with the query token
        // for their x-direction embedding. But doc1 and doc2 both max at 1.0.
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_maxsim_multi_query_tokens() {
        let mut index = default_index();
        // doc1 covers both x and y directions
        index
            .insert("doc1".into(), vec![vec![1.0, 0.0], vec![0.0, 1.0]], None)
            .unwrap();
        // doc2 covers only x direction
        index
            .insert("doc2".into(), vec![vec![1.0, 0.0]], None)
            .unwrap();

        // Query with two tokens: x and y directions
        let results = index.search(&[vec![1.0, 0.0], vec![0.0, 1.0]], 10).unwrap();
        // doc1: maxsim = max(cos(q1,d1), cos(q1,d2)) + max(cos(q2,d1), cos(q2,d2))
        //      = max(1.0, 0.0) + max(0.0, 1.0) = 2.0
        // doc2: maxsim = max(1.0) + max(0.0) = 1.0
        assert_eq!(results[0].id, "doc1");
        assert!((results[0].score - 2.0).abs() < 1e-5);
        assert_eq!(results[1].id, "doc2");
        assert!((results[1].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_avgsim_scoring() {
        let config = MultiVectorConfig {
            metric: DistanceMetric::Cosine,
            scoring: ScoringVariant::AvgSim,
        };
        let mut index = MultiVectorIndex::new(config);
        index
            .insert("doc1".into(), vec![vec![1.0, 0.0], vec![0.0, 1.0]], None)
            .unwrap();

        // Single query token [1,0]: avg of cos([1,0],[1,0]) and cos([1,0],[0,1])
        // = (1.0 + 0.0) / 2 = 0.5
        let results = index.search(&[vec![1.0, 0.0]], 10).unwrap();
        assert!((results[0].score - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_summax_scoring() {
        let config = MultiVectorConfig {
            metric: DistanceMetric::Cosine,
            scoring: ScoringVariant::SumMax,
        };
        let mut index = MultiVectorIndex::new(config);
        // doc1: two tokens
        index
            .insert("doc1".into(), vec![vec![1.0, 0.0], vec![0.0, 1.0]], None)
            .unwrap();

        // Query: single token [1,0]
        // SumMax: for each doc token, max sim across query tokens
        // doc_token [1,0] -> max over query = cos([1,0],[1,0]) = 1.0
        // doc_token [0,1] -> max over query = cos([0,1],[1,0]) = 0.0
        // SumMax = 1.0 + 0.0 = 1.0
        let results = index.search(&[vec![1.0, 0.0]], 10).unwrap();
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product_metric() {
        let config = MultiVectorConfig {
            metric: DistanceMetric::DotProduct,
            scoring: ScoringVariant::MaxSim,
        };
        let mut index = MultiVectorIndex::new(config);
        index
            .insert("doc1".into(), vec![vec![2.0, 0.0], vec![0.0, 3.0]], None)
            .unwrap();

        // Query token [1,0]: dot products are 2.0 and 0.0 -> max = 2.0
        let results = index.search(&[vec![1.0, 0.0]], 10).unwrap();
        assert!((results[0].score - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_search_empty_query_error() {
        let index = default_index();
        let res = index.search(&[], 10);
        assert!(res.is_err());
    }

    #[test]
    fn test_search_empty_index() {
        let index = default_index();
        let results = index.search(&[vec![1.0, 0.0]], 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_top_k_truncation() {
        let mut index = default_index();
        for i in 0..10 {
            let val = (i as f32) / 10.0;
            index
                .insert(format!("d{}", i), vec![vec![val, 1.0 - val]], None)
                .unwrap();
        }
        let results = index.search(&[vec![1.0, 0.0]], 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_remove_document() {
        let mut index = default_index();
        index
            .insert("doc1".into(), vec![vec![1.0, 0.0]], None)
            .unwrap();
        assert_eq!(index.len(), 1);
        let removed = index.remove("doc1");
        assert!(removed.is_some());
        assert!(index.is_empty());
    }

    #[test]
    fn test_metadata_preserved() {
        let mut index = default_index();
        let mut meta = HashMap::new();
        meta.insert("source".into(), serde_json::json!("colbert"));
        index
            .insert("doc1".into(), vec![vec![1.0, 0.0]], Some(meta))
            .unwrap();
        let results = index.search(&[vec![1.0, 0.0]], 10).unwrap();
        let result_meta = results[0].metadata.as_ref().unwrap();
        assert_eq!(result_meta.get("source").unwrap(), "colbert");
    }

    #[test]
    fn test_search_with_scoring_override() {
        let mut index = default_index(); // Default is MaxSim
        index
            .insert("doc1".into(), vec![vec![1.0, 0.0], vec![0.0, 1.0]], None)
            .unwrap();

        // Override to AvgSim
        let results = index
            .search_with_scoring(&[vec![1.0, 0.0]], 10, ScoringVariant::AvgSim)
            .unwrap();
        // AvgSim of [1,0] against {[1,0],[0,1]} = (1.0 + 0.0)/2 = 0.5
        assert!((results[0].score - 0.5).abs() < 1e-5);
    }
}
