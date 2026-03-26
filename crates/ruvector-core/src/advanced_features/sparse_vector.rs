//! Sparse Vector Index with Reciprocal Rank Fusion (RRF)
//!
//! Provides a production-quality sparse vector index suitable for SPLADE-style
//! learned sparse representations and hybrid retrieval pipelines.
//!
//! ## Features
//!
//! - **SparseVector**: Compressed sparse representation (sorted indices + values)
//! - **SparseIndex**: Inverted index with posting lists for sub-linear search
//! - **SPLADE-compatible scoring**: Dot-product between sparse query and documents
//! - **Reciprocal Rank Fusion (RRF)**: Combine dense + sparse rankings
//! - **Multiple fusion strategies**: RRF, Linear Combination, DBSF
//! - **Batch operations**: Insert and search across multiple vectors/queries
//! - **WASM-compatible**: No system-level dependencies

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::VectorId;

// ---------------------------------------------------------------------------
// SparseVector
// ---------------------------------------------------------------------------

/// A sparse vector stored as parallel sorted arrays of indices and values.
///
/// Indices are kept in ascending order so that set-intersection style
/// operations (dot product, merge) run in O(min(|a|, |b|)) time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    /// Dimension indices (sorted ascending, unique).
    pub indices: Vec<u32>,
    /// Corresponding non-zero values.
    pub values: Vec<f32>,
}

impl SparseVector {
    /// Create a new sparse vector from unsorted index/value pairs.
    ///
    /// Duplicate indices are summed. Zero-valued entries are dropped.
    pub fn new(mut pairs: Vec<(u32, f32)>) -> Self {
        // Aggregate duplicates via a temporary map.
        let mut map: HashMap<u32, f32> = HashMap::with_capacity(pairs.len());
        for (idx, val) in pairs.drain(..) {
            *map.entry(idx).or_insert(0.0) += val;
        }

        let mut entries: Vec<(u32, f32)> = map
            .into_iter()
            .filter(|(_, v)| *v != 0.0)
            .collect();
        entries.sort_unstable_by_key(|(idx, _)| *idx);

        let (indices, values) = entries.into_iter().unzip();
        Self { indices, values }
    }

    /// Create from pre-sorted, deduplicated index/value slices (unchecked).
    ///
    /// Caller must guarantee that `indices` is sorted ascending with no
    /// duplicates and that `indices.len() == values.len()`.
    pub fn from_sorted(indices: Vec<u32>, values: Vec<f32>) -> Self {
        debug_assert_eq!(indices.len(), values.len());
        Self { indices, values }
    }

    /// Number of non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` when the vector has no non-zero entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Dot product between two sparse vectors.
    ///
    /// Uses a merge-intersection over sorted indices — O(|a| + |b|).
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let (mut i, mut j) = (0usize, 0usize);
        let mut sum = 0.0f32;
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Equal => {
                    sum += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        sum
    }

    /// L2 (Euclidean) norm of the sparse vector.
    pub fn l2_norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }
}

// ---------------------------------------------------------------------------
// PostingEntry & SparseIndex
// ---------------------------------------------------------------------------

/// A single entry in a posting list: (document id, weight in that dimension).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingEntry {
    pub doc_id: VectorId,
    pub weight: f32,
}

/// Inverted index over sparse vectors.
///
/// Maps each active dimension to a posting list of `(doc_id, weight)` pairs.
/// Supports SPLADE-style dot-product scoring and multiple rank-fusion
/// strategies for hybrid dense+sparse retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseIndex {
    /// dimension -> posting list
    postings: HashMap<u32, Vec<PostingEntry>>,
    /// doc_id -> sparse vector (kept for reconstruction / re-scoring)
    docs: HashMap<VectorId, SparseVector>,
    /// Total number of indexed documents.
    doc_count: usize,
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseIndex {
    /// Create an empty sparse index.
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            docs: HashMap::new(),
            doc_count: 0,
        }
    }

    /// Number of indexed documents.
    #[inline]
    pub fn len(&self) -> usize {
        self.doc_count
    }

    /// Returns `true` when no documents have been indexed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// Insert a single document into the index.
    pub fn insert(&mut self, doc_id: VectorId, vector: SparseVector) {
        // Remove old postings if the doc already exists.
        if let Some(old) = self.docs.remove(&doc_id) {
            for idx in &old.indices {
                if let Some(list) = self.postings.get_mut(idx) {
                    list.retain(|e| e.doc_id != doc_id);
                }
            }
            self.doc_count -= 1;
        }

        // Add new postings.
        for (pos, &dim) in vector.indices.iter().enumerate() {
            self.postings
                .entry(dim)
                .or_default()
                .push(PostingEntry {
                    doc_id: doc_id.clone(),
                    weight: vector.values[pos],
                });
        }

        self.docs.insert(doc_id, vector);
        self.doc_count += 1;
    }

    /// Insert a batch of documents.
    pub fn insert_batch(&mut self, documents: Vec<(VectorId, SparseVector)>) {
        for (id, vec) in documents {
            self.insert(id, vec);
        }
    }

    /// Remove a document from the index. Returns `true` if it existed.
    pub fn remove(&mut self, doc_id: &VectorId) -> bool {
        if let Some(old) = self.docs.remove(doc_id) {
            for idx in &old.indices {
                if let Some(list) = self.postings.get_mut(idx) {
                    list.retain(|e| e.doc_id != *doc_id);
                }
            }
            self.doc_count -= 1;
            true
        } else {
            false
        }
    }

    /// Retrieve the stored sparse vector for a document.
    pub fn get(&self, doc_id: &VectorId) -> Option<&SparseVector> {
        self.docs.get(doc_id)
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Score all documents against a sparse query via dot product (SPLADE
    /// compatible) and return the top-k results sorted descending by score.
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<ScoredDoc> {
        let mut accum: HashMap<&VectorId, f32> = HashMap::new();

        for (pos, &dim) in query.indices.iter().enumerate() {
            let q_weight = query.values[pos];
            if let Some(list) = self.postings.get(&dim) {
                for entry in list {
                    *accum.entry(&entry.doc_id).or_insert(0.0) += q_weight * entry.weight;
                }
            }
        }

        let mut results: Vec<ScoredDoc> = accum
            .into_iter()
            .map(|(id, score)| ScoredDoc {
                id: id.clone(),
                score,
            })
            .collect();

        // Sort descending by score, ties broken by id for determinism.
        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });

        results.truncate(k);
        results
    }

    /// Batch search: run multiple queries and return results for each.
    pub fn search_batch(
        &self,
        queries: &[SparseVector],
        k: usize,
    ) -> Vec<Vec<ScoredDoc>> {
        queries.iter().map(|q| self.search(q, k)).collect()
    }
}

// ---------------------------------------------------------------------------
// ScoredDoc
// ---------------------------------------------------------------------------

/// A document id with an associated relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredDoc {
    pub id: VectorId,
    pub score: f32,
}

// ---------------------------------------------------------------------------
// Rank Fusion
// ---------------------------------------------------------------------------

/// Strategy for combining ranked lists from different retrieval systems.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion. `k` controls rank-pressure (default 60).
    RRF { k: f32 },
    /// Weighted linear combination of normalised scores.
    Linear { dense_weight: f32, sparse_weight: f32 },
    /// Distribution-Based Score Fusion: normalise each list to N(0,1) then
    /// combine with equal weight.
    DBSF,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        FusionStrategy::RRF { k: 60.0 }
    }
}

/// Configuration for hybrid rank fusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// The fusion strategy to apply.
    pub strategy: FusionStrategy,
    /// Maximum number of results to return after fusion.
    pub top_k: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            strategy: FusionStrategy::default(),
            top_k: 10,
        }
    }
}

/// Fuse two ranked result lists (e.g., dense and sparse) into a single
/// ranking using the configured [`FusionStrategy`].
///
/// Both input lists must be sorted descending by score.
pub fn fuse_rankings(
    dense: &[ScoredDoc],
    sparse: &[ScoredDoc],
    config: &FusionConfig,
) -> Vec<ScoredDoc> {
    match config.strategy {
        FusionStrategy::RRF { k } => fuse_rrf(dense, sparse, k, config.top_k),
        FusionStrategy::Linear {
            dense_weight,
            sparse_weight,
        } => fuse_linear(dense, sparse, dense_weight, sparse_weight, config.top_k),
        FusionStrategy::DBSF => fuse_dbsf(dense, sparse, config.top_k),
    }
}

// -- RRF -------------------------------------------------------------------

/// Reciprocal Rank Fusion: score(d) = sum_over_lists 1 / (k + rank(d)).
fn fuse_rrf(
    dense: &[ScoredDoc],
    sparse: &[ScoredDoc],
    k: f32,
    top_k: usize,
) -> Vec<ScoredDoc> {
    let mut scores: HashMap<VectorId, f32> = HashMap::new();

    for (rank, doc) in dense.iter().enumerate() {
        *scores.entry(doc.id.clone()).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
    }
    for (rank, doc) in sparse.iter().enumerate() {
        *scores.entry(doc.id.clone()).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
    }

    collect_top_k(scores, top_k)
}

// -- Linear ----------------------------------------------------------------

/// Normalise scores to [0, 1] via min-max then combine with weights.
fn fuse_linear(
    dense: &[ScoredDoc],
    sparse: &[ScoredDoc],
    dw: f32,
    sw: f32,
    top_k: usize,
) -> Vec<ScoredDoc> {
    let norm_dense = min_max_normalize(dense);
    let norm_sparse = min_max_normalize(sparse);

    let mut scores: HashMap<VectorId, f32> = HashMap::new();

    for (id, s) in &norm_dense {
        *scores.entry(id.clone()).or_insert(0.0) += dw * s;
    }
    for (id, s) in &norm_sparse {
        *scores.entry(id.clone()).or_insert(0.0) += sw * s;
    }

    collect_top_k(scores, top_k)
}

// -- DBSF ------------------------------------------------------------------

/// Distribution-Based Score Fusion: z-score normalise, then average.
fn fuse_dbsf(
    dense: &[ScoredDoc],
    sparse: &[ScoredDoc],
    top_k: usize,
) -> Vec<ScoredDoc> {
    let z_dense = z_score_normalize(dense);
    let z_sparse = z_score_normalize(sparse);

    let mut scores: HashMap<VectorId, f32> = HashMap::new();

    for (id, s) in &z_dense {
        *scores.entry(id.clone()).or_insert(0.0) += s;
    }
    for (id, s) in &z_sparse {
        *scores.entry(id.clone()).or_insert(0.0) += s;
    }

    collect_top_k(scores, top_k)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn collect_top_k(scores: HashMap<VectorId, f32>, top_k: usize) -> Vec<ScoredDoc> {
    let mut results: Vec<ScoredDoc> = scores
        .into_iter()
        .map(|(id, score)| ScoredDoc { id, score })
        .collect();

    results.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    results.truncate(top_k);
    results
}

fn min_max_normalize(docs: &[ScoredDoc]) -> Vec<(VectorId, f32)> {
    if docs.is_empty() {
        return Vec::new();
    }
    let min = docs.iter().map(|d| d.score).fold(f32::INFINITY, f32::min);
    let max = docs
        .iter()
        .map(|d| d.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    docs.iter()
        .map(|d| {
            let norm = if range > 0.0 {
                (d.score - min) / range
            } else {
                1.0
            };
            (d.id.clone(), norm)
        })
        .collect()
}

fn z_score_normalize(docs: &[ScoredDoc]) -> Vec<(VectorId, f32)> {
    if docs.is_empty() {
        return Vec::new();
    }
    let n = docs.len() as f32;
    let mean = docs.iter().map(|d| d.score).sum::<f32>() / n;
    let variance = docs.iter().map(|d| (d.score - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();

    docs.iter()
        .map(|d| {
            let z = if std > 0.0 {
                (d.score - mean) / std
            } else {
                0.0
            };
            (d.id.clone(), z)
        })
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- SparseVector tests -------------------------------------------------

    #[test]
    fn test_sparse_vector_new_sorts_and_deduplicates() {
        let sv = SparseVector::new(vec![(5, 1.0), (2, 3.0), (5, 2.0), (0, 0.5)]);
        assert_eq!(sv.indices, vec![0, 2, 5]);
        assert_eq!(sv.values, vec![0.5, 3.0, 3.0]); // 1.0 + 2.0 = 3.0 for idx 5
    }

    #[test]
    fn test_sparse_vector_dot_product() {
        let a = SparseVector::from_sorted(vec![0, 2, 5], vec![1.0, 2.0, 3.0]);
        let b = SparseVector::from_sorted(vec![2, 5, 8], vec![4.0, 5.0, 6.0]);
        // overlap at 2: 2*4=8, at 5: 3*5=15 => 23
        assert!((a.dot(&b) - 23.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_dot_no_overlap() {
        let a = SparseVector::from_sorted(vec![0, 1], vec![1.0, 2.0]);
        let b = SparseVector::from_sorted(vec![3, 4], vec![5.0, 6.0]);
        assert!((a.dot(&b)).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_empty() {
        let empty = SparseVector::new(vec![]);
        assert!(empty.is_empty());
        assert_eq!(empty.nnz(), 0);
        assert!((empty.l2_norm()).abs() < 1e-6);
    }

    // -- SparseIndex insert & search ----------------------------------------

    #[test]
    fn test_index_insert_and_search() {
        let mut idx = SparseIndex::new();
        idx.insert(
            "d1".into(),
            SparseVector::from_sorted(vec![0, 2, 5], vec![1.0, 2.0, 3.0]),
        );
        idx.insert(
            "d2".into(),
            SparseVector::from_sorted(vec![2, 5, 8], vec![4.0, 5.0, 6.0]),
        );
        idx.insert(
            "d3".into(),
            SparseVector::from_sorted(vec![0, 8], vec![0.5, 1.0]),
        );
        assert_eq!(idx.len(), 3);

        let query = SparseVector::from_sorted(vec![2, 5], vec![1.0, 1.0]);
        let results = idx.search(&query, 2);

        assert_eq!(results.len(), 2);
        // d2 should rank first: 4*1 + 5*1 = 9 vs d1: 2*1 + 3*1 = 5
        assert_eq!(results[0].id, "d2");
        assert!((results[0].score - 9.0).abs() < 1e-6);
        assert_eq!(results[1].id, "d1");
        assert!((results[1].score - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_index_empty_search() {
        let idx = SparseIndex::new();
        let query = SparseVector::from_sorted(vec![0], vec![1.0]);
        let results = idx.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_index_single_result() {
        let mut idx = SparseIndex::new();
        idx.insert(
            "only".into(),
            SparseVector::from_sorted(vec![7], vec![2.0]),
        );
        let query = SparseVector::from_sorted(vec![7], vec![3.0]);
        let results = idx.search(&query, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "only");
        assert!((results[0].score - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_index_remove() {
        let mut idx = SparseIndex::new();
        idx.insert(
            "d1".into(),
            SparseVector::from_sorted(vec![0], vec![1.0]),
        );
        assert_eq!(idx.len(), 1);
        assert!(idx.remove(&"d1".into()));
        assert_eq!(idx.len(), 0);
        assert!(!idx.remove(&"d1".into()));
    }

    #[test]
    fn test_index_upsert_replaces_old_postings() {
        let mut idx = SparseIndex::new();
        idx.insert(
            "d1".into(),
            SparseVector::from_sorted(vec![0, 1], vec![1.0, 2.0]),
        );
        // Re-insert same id with different dimensions.
        idx.insert(
            "d1".into(),
            SparseVector::from_sorted(vec![3], vec![5.0]),
        );
        assert_eq!(idx.len(), 1);

        // Old dimensions should not match.
        let q_old = SparseVector::from_sorted(vec![0], vec![1.0]);
        assert!(idx.search(&q_old, 5).is_empty());

        // New dimension should match.
        let q_new = SparseVector::from_sorted(vec![3], vec![1.0]);
        let res = idx.search(&q_new, 5);
        assert_eq!(res.len(), 1);
        assert!((res[0].score - 5.0).abs() < 1e-6);
    }

    // -- Rank Fusion tests --------------------------------------------------

    #[test]
    fn test_rrf_fusion_basic() {
        // Two lists with overlapping documents.
        let dense = vec![
            ScoredDoc { id: "a".into(), score: 10.0 },
            ScoredDoc { id: "b".into(), score: 8.0 },
            ScoredDoc { id: "c".into(), score: 6.0 },
        ];
        let sparse = vec![
            ScoredDoc { id: "b".into(), score: 9.0 },
            ScoredDoc { id: "d".into(), score: 7.0 },
            ScoredDoc { id: "a".into(), score: 5.0 },
        ];

        let config = FusionConfig {
            strategy: FusionStrategy::RRF { k: 60.0 },
            top_k: 4,
        };
        let fused = fuse_rankings(&dense, &sparse, &config);

        // "b" appears at dense rank 2 and sparse rank 1 => should score highest.
        assert_eq!(fused[0].id, "b");
        // "a" appears at dense rank 1 and sparse rank 3 => also high.
        assert_eq!(fused[1].id, "a");
        assert_eq!(fused.len(), 4);
    }

    #[test]
    fn test_rrf_with_disjoint_lists() {
        let dense = vec![
            ScoredDoc { id: "x".into(), score: 5.0 },
        ];
        let sparse = vec![
            ScoredDoc { id: "y".into(), score: 5.0 },
        ];

        let config = FusionConfig {
            strategy: FusionStrategy::RRF { k: 60.0 },
            top_k: 10,
        };
        let fused = fuse_rankings(&dense, &sparse, &config);
        assert_eq!(fused.len(), 2);
        // Both at rank 1 in their list => same RRF score; tie broken by id.
        assert_eq!(fused[0].id, "x");
        assert_eq!(fused[1].id, "y");
        assert!((fused[0].score - fused[1].score).abs() < 1e-6);
    }

    #[test]
    fn test_linear_fusion() {
        let dense = vec![
            ScoredDoc { id: "a".into(), score: 10.0 },
            ScoredDoc { id: "b".into(), score: 5.0 },
        ];
        let sparse = vec![
            ScoredDoc { id: "b".into(), score: 10.0 },
            ScoredDoc { id: "a".into(), score: 5.0 },
        ];

        let config = FusionConfig {
            strategy: FusionStrategy::Linear {
                dense_weight: 0.5,
                sparse_weight: 0.5,
            },
            top_k: 2,
        };
        let fused = fuse_rankings(&dense, &sparse, &config);

        // Both a and b appear in both lists. After min-max, each has
        // one normalised 1.0 and one 0.0 => combined 0.5 each.
        assert_eq!(fused.len(), 2);
        assert!((fused[0].score - fused[1].score).abs() < 1e-6);
    }

    #[test]
    fn test_dbsf_fusion() {
        let dense = vec![
            ScoredDoc { id: "a".into(), score: 10.0 },
            ScoredDoc { id: "b".into(), score: 8.0 },
        ];
        let sparse = vec![
            ScoredDoc { id: "a".into(), score: 6.0 },
            ScoredDoc { id: "c".into(), score: 4.0 },
        ];

        let config = FusionConfig {
            strategy: FusionStrategy::DBSF,
            top_k: 3,
        };
        let fused = fuse_rankings(&dense, &sparse, &config);
        assert_eq!(fused.len(), 3);
        // "a" appears in both z-normalised lists, should rank highest.
        assert_eq!(fused[0].id, "a");
    }

    #[test]
    fn test_fusion_empty_inputs() {
        let config = FusionConfig::default();
        let fused = fuse_rankings(&[], &[], &config);
        assert!(fused.is_empty());

        let single = vec![ScoredDoc { id: "x".into(), score: 1.0 }];
        let fused2 = fuse_rankings(&single, &[], &config);
        assert_eq!(fused2.len(), 1);
        assert_eq!(fused2[0].id, "x");
    }

    #[test]
    fn test_batch_search() {
        let mut idx = SparseIndex::new();
        idx.insert(
            "d1".into(),
            SparseVector::from_sorted(vec![0, 1], vec![1.0, 2.0]),
        );
        idx.insert(
            "d2".into(),
            SparseVector::from_sorted(vec![1, 2], vec![3.0, 4.0]),
        );

        let queries = vec![
            SparseVector::from_sorted(vec![0], vec![1.0]),
            SparseVector::from_sorted(vec![2], vec![1.0]),
        ];

        let results = idx.search_batch(&queries, 5);
        assert_eq!(results.len(), 2);
        // First query: only d1 has dim 0.
        assert_eq!(results[0].len(), 1);
        assert_eq!(results[0][0].id, "d1");
        // Second query: only d2 has dim 2.
        assert_eq!(results[1].len(), 1);
        assert_eq!(results[1][0].id, "d2");
    }

    #[test]
    fn test_rrf_top_k_truncation() {
        let dense: Vec<ScoredDoc> = (0..20)
            .map(|i| ScoredDoc {
                id: format!("d{}", i),
                score: 20.0 - i as f32,
            })
            .collect();
        let sparse: Vec<ScoredDoc> = (0..20)
            .rev()
            .map(|i| ScoredDoc {
                id: format!("d{}", i),
                score: i as f32 + 1.0,
            })
            .collect();

        let config = FusionConfig {
            strategy: FusionStrategy::RRF { k: 60.0 },
            top_k: 5,
        };
        let fused = fuse_rankings(&dense, &sparse, &config);
        assert_eq!(fused.len(), 5);
    }
}
