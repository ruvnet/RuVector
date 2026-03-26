//! Matryoshka Representation Learning Support
//!
//! Implements adaptive-dimension embedding search inspired by Matryoshka
//! Representation Learning (MRL). Full-dimensional embeddings are stored once,
//! but searches can be performed at any prefix dimension—smaller prefixes run
//! faster while larger ones are more accurate.
//!
//! # Two-Phase Funnel Search
//!
//! The flagship feature is [`MatryoshkaIndex::funnel_search`], which:
//! 1. Filters candidates at a low dimension (fast, coarse)
//! 2. Reranks the survivors at full dimension (slower, precise)
//!
//! This typically yields the same recall as full-dimension search at a fraction
//! of the cost.
//!
//! # Example
//!
//! ```
//! use ruvector_core::advanced_features::matryoshka::*;
//! use ruvector_core::types::DistanceMetric;
//!
//! let config = MatryoshkaConfig {
//!     full_dim: 8,
//!     supported_dims: vec![2, 4, 8],
//!     metric: DistanceMetric::Cosine,
//! };
//! let mut index = MatryoshkaIndex::new(config).unwrap();
//! index.insert("v1".into(), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None).unwrap();
//! let results = index.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4, 10).unwrap();
//! assert_eq!(results[0].id, "v1");
//! ```

use crate::error::{Result, RuvectorError};
use crate::types::{DistanceMetric, SearchResult, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for a Matryoshka embedding index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatryoshkaConfig {
    /// The full (maximum) embedding dimension.
    pub full_dim: usize,
    /// Supported truncation dimensions, sorted ascending.
    /// Each must be <= `full_dim`. The last element should equal `full_dim`.
    pub supported_dims: Vec<usize>,
    /// Distance metric for similarity computation.
    pub metric: DistanceMetric,
}

impl Default for MatryoshkaConfig {
    fn default() -> Self {
        Self {
            full_dim: 768,
            supported_dims: vec![64, 128, 256, 512, 768],
            metric: DistanceMetric::Cosine,
        }
    }
}

/// Configuration for the multi-phase funnel search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunnelConfig {
    /// Dimension used for the coarse filtering phase.
    pub filter_dim: usize,
    /// Multiplier applied to `top_k` to determine how many candidates
    /// survive the coarse phase. E.g., 4.0 means 4x top_k candidates.
    pub candidate_multiplier: f32,
}

impl Default for FunnelConfig {
    fn default() -> Self {
        Self {
            filter_dim: 64,
            candidate_multiplier: 4.0,
        }
    }
}

/// Entry stored in the Matryoshka index.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MatryoshkaEntry {
    id: VectorId,
    /// Full-dimensional embedding.
    embedding: Vec<f32>,
    /// Precomputed L2 norm of the full embedding.
    full_norm: f32,
    /// Optional metadata.
    metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Matryoshka embedding index supporting adaptive-dimension search.
///
/// Stores embeddings at full dimensionality but can search at any prefix
/// dimension for a speed-accuracy trade-off.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatryoshkaIndex {
    /// Index configuration.
    pub config: MatryoshkaConfig,
    /// Stored entries.
    entries: Vec<MatryoshkaEntry>,
    /// Map from vector ID to index in `entries`.
    id_map: HashMap<VectorId, usize>,
}

impl MatryoshkaIndex {
    /// Create a new Matryoshka index.
    ///
    /// # Errors
    ///
    /// Returns an error if `supported_dims` is empty, any dimension is zero,
    /// or any dimension exceeds `full_dim`.
    pub fn new(mut config: MatryoshkaConfig) -> Result<Self> {
        if config.supported_dims.is_empty() {
            return Err(RuvectorError::InvalidParameter(
                "supported_dims must not be empty".into(),
            ));
        }
        config.supported_dims.sort_unstable();
        config.supported_dims.dedup();

        for &d in &config.supported_dims {
            if d == 0 {
                return Err(RuvectorError::InvalidParameter(
                    "Dimensions must be > 0".into(),
                ));
            }
            if d > config.full_dim {
                return Err(RuvectorError::InvalidParameter(format!(
                    "Supported dimension {} exceeds full_dim {}",
                    d, config.full_dim
                )));
            }
        }

        Ok(Self {
            config,
            entries: Vec::new(),
            id_map: HashMap::new(),
        })
    }

    /// Insert a full-dimensional embedding into the index.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding dimension does not match `full_dim`.
    pub fn insert(
        &mut self,
        id: VectorId,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<()> {
        if embedding.len() != self.config.full_dim {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.config.full_dim,
                actual: embedding.len(),
            });
        }

        let full_norm = compute_norm(&embedding);

        if let Some(&existing_idx) = self.id_map.get(&id) {
            self.entries[existing_idx] = MatryoshkaEntry {
                id,
                embedding,
                full_norm,
                metadata,
            };
        } else {
            let idx = self.entries.len();
            self.entries.push(MatryoshkaEntry {
                id: id.clone(),
                embedding,
                full_norm,
                metadata,
            });
            self.id_map.insert(id, idx);
        }

        Ok(())
    }

    /// Return the number of stored vectors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Search at a specific dimension by truncating embeddings to the first
    /// `dim` components.
    ///
    /// # Arguments
    ///
    /// * `query` - Full-dimensional (or at least `dim`-dimensional) query vector.
    /// * `dim` - The truncation dimension to use for search.
    /// * `top_k` - Number of results to return.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim` exceeds the query length or `full_dim`.
    pub fn search(
        &self,
        query: &[f32],
        dim: usize,
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        if dim == 0 {
            return Err(RuvectorError::InvalidParameter(
                "Search dimension must be > 0".into(),
            ));
        }
        if dim > self.config.full_dim {
            return Err(RuvectorError::InvalidParameter(format!(
                "Search dimension {} exceeds full_dim {}",
                dim, self.config.full_dim
            )));
        }
        if query.len() < dim {
            return Err(RuvectorError::DimensionMismatch {
                expected: dim,
                actual: query.len(),
            });
        }

        let query_prefix = &query[..dim];
        let query_norm = compute_norm(query_prefix);

        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let doc_prefix = &entry.embedding[..dim];
                let doc_norm = compute_norm(doc_prefix);
                let sim = similarity(query_prefix, query_norm, doc_prefix, doc_norm, self.config.metric);
                (idx, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored
            .into_iter()
            .map(|(idx, score)| {
                let entry = &self.entries[idx];
                SearchResult {
                    id: entry.id.clone(),
                    score,
                    vector: None,
                    metadata: entry.metadata.clone(),
                }
            })
            .collect())
    }

    /// Two-phase funnel search: coarse filter at low dimension, rerank at full dimension.
    ///
    /// 1. Search at `funnel_config.filter_dim` for `candidate_multiplier * top_k` candidates.
    /// 2. Rerank those candidates at `full_dim`.
    /// 3. Return the top `top_k`.
    ///
    /// # Errors
    ///
    /// Returns an error if the query is shorter than `full_dim`.
    pub fn funnel_search(
        &self,
        query: &[f32],
        top_k: usize,
        funnel_config: &FunnelConfig,
    ) -> Result<Vec<SearchResult>> {
        if query.len() < self.config.full_dim {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.config.full_dim,
                actual: query.len(),
            });
        }

        let filter_dim = funnel_config.filter_dim.min(self.config.full_dim);
        let num_candidates = ((top_k as f32) * funnel_config.candidate_multiplier).ceil() as usize;
        let num_candidates = num_candidates.max(top_k);

        // Phase 1: coarse search at low dimension.
        let coarse_results = self.search(query, filter_dim, num_candidates)?;

        // Phase 2: rerank at full dimension.
        let query_full = &query[..self.config.full_dim];
        let query_full_norm = compute_norm(query_full);

        let mut reranked: Vec<(VectorId, f32, Option<HashMap<String, serde_json::Value>>)> =
            coarse_results
                .into_iter()
                .filter_map(|r| {
                    let idx = self.id_map.get(&r.id)?;
                    let entry = &self.entries[*idx];
                    let sim = similarity(
                        query_full,
                        query_full_norm,
                        &entry.embedding,
                        entry.full_norm,
                        self.config.metric,
                    );
                    Some((entry.id.clone(), sim, entry.metadata.clone()))
                })
                .collect();

        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(top_k);

        Ok(reranked
            .into_iter()
            .map(|(id, score, metadata)| SearchResult {
                id,
                score,
                vector: None,
                metadata,
            })
            .collect())
    }

    /// Multi-stage cascade search through multiple dimensions.
    ///
    /// Searches through dimensions in ascending order, progressively narrowing
    /// candidates. At each stage, the candidate set is reduced by the
    /// `reduction_factor`.
    pub fn cascade_search(
        &self,
        query: &[f32],
        top_k: usize,
        dims: &[usize],
        reduction_factor: f32,
    ) -> Result<Vec<SearchResult>> {
        if dims.is_empty() {
            return Err(RuvectorError::InvalidParameter(
                "Dimension cascade must not be empty".into(),
            ));
        }
        if query.len() < self.config.full_dim {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.config.full_dim,
                actual: query.len(),
            });
        }

        // Start with all candidates at the lowest dimension.
        let mut candidate_indices: Vec<usize> = (0..self.entries.len()).collect();

        for &dim in dims {
            let dim = dim.min(self.config.full_dim);
            let query_prefix = &query[..dim];
            let query_norm = compute_norm(query_prefix);

            let mut scored: Vec<(usize, f32)> = candidate_indices
                .iter()
                .map(|&idx| {
                    let entry = &self.entries[idx];
                    let doc_prefix = &entry.embedding[..dim];
                    let doc_norm = compute_norm(doc_prefix);
                    let sim = similarity(
                        query_prefix,
                        query_norm,
                        doc_prefix,
                        doc_norm,
                        self.config.metric,
                    );
                    (idx, sim)
                })
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let keep = ((candidate_indices.len() as f32) / reduction_factor)
                .ceil()
                .max(top_k as f32) as usize;
            scored.truncate(keep);
            candidate_indices = scored.into_iter().map(|(idx, _)| idx).collect();
        }

        // Final scoring at the last dimension in the cascade.
        let last_dim = dims.last().copied().unwrap_or(self.config.full_dim);
        let last_dim = last_dim.min(self.config.full_dim);
        let query_prefix = &query[..last_dim];
        let query_norm = compute_norm(query_prefix);

        let mut final_scored: Vec<(usize, f32)> = candidate_indices
            .iter()
            .map(|&idx| {
                let entry = &self.entries[idx];
                let doc_prefix = &entry.embedding[..last_dim];
                let doc_norm = compute_norm(doc_prefix);
                let sim = similarity(
                    query_prefix,
                    query_norm,
                    doc_prefix,
                    doc_norm,
                    self.config.metric,
                );
                (idx, sim)
            })
            .collect();

        final_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        final_scored.truncate(top_k);

        Ok(final_scored
            .into_iter()
            .map(|(idx, score)| {
                let entry = &self.entries[idx];
                SearchResult {
                    id: entry.id.clone(),
                    score,
                    vector: None,
                    metadata: entry.metadata.clone(),
                }
            })
            .collect())
    }
}

/// Compute the L2 norm of a vector slice.
#[inline]
fn compute_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute similarity between two vectors using the given metric and precomputed norms.
#[inline]
fn similarity(a: &[f32], norm_a: f32, b: &[f32], norm_b: f32, metric: DistanceMetric) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    match metric {
        DistanceMetric::Cosine => {
            let denom = norm_a * norm_b;
            if denom < f32::EPSILON {
                0.0
            } else {
                dot / denom
            }
        }
        DistanceMetric::DotProduct => dot,
        DistanceMetric::Euclidean => {
            let dist_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
            1.0 / (1.0 + dist_sq.sqrt())
        }
        DistanceMetric::Manhattan => {
            let dist: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
            1.0 / (1.0 + dist)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(full_dim: usize, dims: Vec<usize>) -> MatryoshkaConfig {
        MatryoshkaConfig {
            full_dim,
            supported_dims: dims,
            metric: DistanceMetric::Cosine,
        }
    }

    fn make_index(full_dim: usize) -> MatryoshkaIndex {
        let dims: Vec<usize> = (1..=full_dim).filter(|d| d.is_power_of_two() || *d == full_dim).collect();
        MatryoshkaIndex::new(make_config(full_dim, dims)).unwrap()
    }

    #[test]
    fn test_insert_and_len() {
        let mut index = make_index(4);
        assert!(index.is_empty());
        index.insert("v1".into(), vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_insert_wrong_dimension_error() {
        let mut index = make_index(4);
        let res = index.insert("v1".into(), vec![1.0, 0.0], None);
        assert!(res.is_err());
    }

    #[test]
    fn test_search_at_full_dim() {
        let mut index = make_index(4);
        index.insert("v1".into(), vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
        index.insert("v2".into(), vec![0.0, 1.0, 0.0, 0.0], None).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 4, 10).unwrap();
        assert_eq!(results[0].id, "v1");
        assert!((results[0].score - 1.0).abs() < 1e-5);
        // v2 is orthogonal, score should be ~0
        assert!(results[1].score.abs() < 1e-5);
    }

    #[test]
    fn test_search_at_truncated_dim() {
        let mut index = make_index(4);
        // Vectors differ only in the last two components
        index.insert("v1".into(), vec![1.0, 0.0, 1.0, 0.0], None).unwrap();
        index.insert("v2".into(), vec![1.0, 0.0, 0.0, 1.0], None).unwrap();

        // At dim=2, both truncate to [1.0, 0.0] — identical scores
        let results = index.search(&[1.0, 0.0, 0.5, 0.5], 2, 10).unwrap();
        assert!((results[0].score - results[1].score).abs() < 1e-5);

        // At dim=4, they should differ
        let results = index.search(&[1.0, 0.0, 1.0, 0.0], 4, 10).unwrap();
        assert_eq!(results[0].id, "v1");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_funnel_search() {
        let mut index = make_index(8);
        // Insert vectors that share the same first 2 dims but differ later
        index
            .insert("best".into(), vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        index
            .insert("good".into(), vec![1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        index
            .insert("bad".into(), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None)
            .unwrap();

        let query = vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let funnel = FunnelConfig {
            filter_dim: 2,
            candidate_multiplier: 2.0,
        };
        let results = index.funnel_search(&query, 2, &funnel).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "best");
    }

    #[test]
    fn test_funnel_search_finds_correct_top_k() {
        let mut index = make_index(4);
        for i in 0..20 {
            let angle = (i as f32) * std::f32::consts::PI / 20.0;
            index
                .insert(
                    format!("v{}", i),
                    vec![angle.cos(), angle.sin(), 0.0, 0.0],
                    None,
                )
                .unwrap();
        }

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let funnel = FunnelConfig {
            filter_dim: 2,
            candidate_multiplier: 4.0,
        };
        let results = index.funnel_search(&query, 3, &funnel).unwrap();
        assert_eq!(results.len(), 3);
        // The closest vector should be v0 (angle=0, cos=1, sin=0)
        assert_eq!(results[0].id, "v0");
    }

    #[test]
    fn test_cascade_search() {
        let mut index = make_index(8);
        index
            .insert("a".into(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        index
            .insert("b".into(), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        index
            .insert("c".into(), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None)
            .unwrap();

        let query = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let results = index.cascade_search(&query, 2, &[2, 4, 8], 1.5).unwrap();
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_search_dim_exceeds_full_dim_error() {
        let index = make_index(4);
        let res = index.search(&[1.0, 0.0, 0.0, 0.0], 8, 10);
        assert!(res.is_err());
    }

    #[test]
    fn test_search_empty_index() {
        let index = make_index(4);
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 4, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_upsert_overwrites() {
        let mut index = make_index(4);
        index.insert("v1".into(), vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
        index.insert("v1".into(), vec![0.0, 1.0, 0.0, 0.0], None).unwrap();
        assert_eq!(index.len(), 1);
        let results = index.search(&[0.0, 1.0, 0.0, 0.0], 4, 10).unwrap();
        assert_eq!(results[0].id, "v1");
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_config_validation_empty_dims() {
        let res = MatryoshkaIndex::new(MatryoshkaConfig {
            full_dim: 4,
            supported_dims: vec![],
            metric: DistanceMetric::Cosine,
        });
        assert!(res.is_err());
    }

    #[test]
    fn test_config_validation_dim_exceeds_full() {
        let res = MatryoshkaIndex::new(MatryoshkaConfig {
            full_dim: 4,
            supported_dims: vec![2, 8],
            metric: DistanceMetric::Cosine,
        });
        assert!(res.is_err());
    }

    #[test]
    fn test_dot_product_metric() {
        let config = MatryoshkaConfig {
            full_dim: 4,
            supported_dims: vec![2, 4],
            metric: DistanceMetric::DotProduct,
        };
        let mut index = MatryoshkaIndex::new(config).unwrap();
        index.insert("v1".into(), vec![2.0, 0.0, 0.0, 0.0], None).unwrap();
        let results = index.search(&[3.0, 0.0, 0.0, 0.0], 4, 10).unwrap();
        assert!((results[0].score - 6.0).abs() < 1e-5);
    }
}
