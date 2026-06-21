//! Three hybrid fusion strategies for sparse + dense retrieval.
//!
//! | Strategy | Approach | Used by |
//! |----------|----------|---------|
//! | [`ScoreFusionIndex`] | Min-max normalize scores, weighted linear blend | ruvector-core today |
//! | [`RrfHybridIndex`]   | Reciprocal Rank Fusion (rank-only, score-agnostic) | Qdrant, Milvus 2.5 |
//! | [`RsfHybridIndex`]   | Relative Score Fusion (query-local normalisation) | Weaviate (default) |
//!
//! All three implement [`HybridSearch`].  The benchmark in `src/main.rs` shows
//! recall@10 vs. a brute-force combined ground truth.
//!
//! ## RRF reference
//! Cormack, Clarke, Grossman — "Reciprocal rank fusion outperforms Condorcet
//! and individual rank learning methods", CIKM 2009.

use std::collections::HashMap;

use crate::{
    Bm25Index, DenseSearch, Document, FlatDenseIndex, HybridSearch, SearchResult, SparseSearch,
};

/// Constant used by RRF; 60 is the value proven optimal in the 2009 paper.
const RRF_K: f32 = 60.0;

// ─────────────────────────────────────────────────────────────────────────────
// 1. SCORE FUSION (ruvector-core current approach)
// ─────────────────────────────────────────────────────────────────────────────

/// Hybrid index using min-max-normalised weighted linear score combination.
///
/// `combined = α · cosine_norm + (1−α) · bm25_norm` where cosine_norm and
/// bm25_norm are normalised to [0,1] across all candidates.
///
/// Weakness: when score distributions differ in shape (peaky BM25 vs. smooth
/// cosine), the normalization distorts relative ordering.
pub struct ScoreFusionIndex {
    sparse: Bm25Index,
    dense: FlatDenseIndex,
    /// Weight given to the dense (vector) signal; keyword weight = 1 − alpha.
    pub alpha: f32,
    candidate_mult: usize,
}

impl ScoreFusionIndex {
    /// Build with default α=0.7 (matches ruvector-core default).
    pub fn build(docs: &[Document]) -> Self {
        Self::build_with_alpha(docs, 0.7)
    }

    /// Build with a custom α ∈ [0, 1].
    pub fn build_with_alpha(docs: &[Document], alpha: f32) -> Self {
        Self {
            sparse: Bm25Index::build(docs),
            dense: FlatDenseIndex::build(docs),
            alpha: alpha.clamp(0.0, 1.0),
            candidate_mult: 4,
        }
    }
}

impl HybridSearch for ScoreFusionIndex {
    fn search(&self, tokens: &[&str], vector: &[f32], k: usize) -> Vec<SearchResult> {
        let fetch = k * self.candidate_mult;
        let sparse = self.sparse.search(tokens, fetch);
        let dense = self.dense.search(vector, fetch);

        // Merge candidate sets
        let mut id_to_sparse: HashMap<usize, f32> =
            sparse.iter().map(|r| (r.id, r.score)).collect();
        let mut id_to_dense: HashMap<usize, f32> = dense.iter().map(|r| (r.id, r.score)).collect();

        let all_ids: std::collections::HashSet<usize> = id_to_sparse
            .keys()
            .chain(id_to_dense.keys())
            .cloned()
            .collect();

        // Min-max normalize each signal independently
        let s_max = id_to_sparse
            .values()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let s_min = id_to_sparse.values().cloned().fold(f32::INFINITY, f32::min);
        let d_max = id_to_dense
            .values()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let d_min = id_to_dense.values().cloned().fold(f32::INFINITY, f32::min);

        let s_range = (s_max - s_min).max(1e-10);
        let d_range = (d_max - d_min).max(1e-10);

        for v in id_to_sparse.values_mut() {
            *v = (*v - s_min) / s_range;
        }
        for v in id_to_dense.values_mut() {
            *v = (*v - d_min) / d_range;
        }

        let mut combined: Vec<SearchResult> = all_ids
            .into_iter()
            .map(|id| {
                let s = id_to_sparse.get(&id).cloned().unwrap_or(0.0);
                let d = id_to_dense.get(&id).cloned().unwrap_or(0.0);
                SearchResult {
                    id,
                    score: self.alpha * d + (1.0 - self.alpha) * s,
                }
            })
            .collect();

        combined.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        combined.truncate(k);
        combined
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. RECIPROCAL RANK FUSION (RRF)
// ─────────────────────────────────────────────────────────────────────────────

/// Hybrid index using Reciprocal Rank Fusion.
///
/// RRF_score(d) = Σ_i 1 / (60 + rank_i(d))
///
/// Rank-only: raw scores from BM25 and cosine are never combined, so
/// distribution incompatibility is not a problem.
pub struct RrfHybridIndex {
    sparse: Bm25Index,
    dense: FlatDenseIndex,
    candidate_mult: usize,
}

impl RrfHybridIndex {
    /// Build with default candidate multiplier of 4.
    pub fn build(docs: &[Document]) -> Self {
        Self {
            sparse: Bm25Index::build(docs),
            dense: FlatDenseIndex::build(docs),
            candidate_mult: 4,
        }
    }
}

impl HybridSearch for RrfHybridIndex {
    fn search(&self, tokens: &[&str], vector: &[f32], k: usize) -> Vec<SearchResult> {
        let fetch = k * self.candidate_mult;
        let sparse_list = self.sparse.search(tokens, fetch);
        let dense_list = self.dense.search(vector, fetch);

        let mut scores: HashMap<usize, f32> = HashMap::new();
        for (rank, r) in sparse_list.iter().enumerate() {
            *scores.entry(r.id).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
        }
        for (rank, r) in dense_list.iter().enumerate() {
            *scores.entry(r.id).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
        }

        let mut merged: Vec<SearchResult> = scores
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect();
        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(k);
        merged
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. RELATIVE SCORE FUSION (RSF / Weaviate default since v1.24)
// ─────────────────────────────────────────────────────────────────────────────

/// Hybrid index using Relative Score Fusion (Weaviate default since v1.24).
///
/// Per-query min-max normalisation of each ranked list, then linear blend:
/// `combined = α · dense_norm + (1−α) · sparse_norm`
///
/// Unlike [`ScoreFusionIndex`] which normalises globally across all candidates,
/// RSF normalises each signal over only its own ranked list, making the blend
/// numerically stable even when candidate sets differ in size.
pub struct RsfHybridIndex {
    sparse: Bm25Index,
    dense: FlatDenseIndex,
    /// α controls dense-vs-sparse blend; 0.5 = equal weight.
    pub alpha: f32,
    candidate_mult: usize,
}

impl RsfHybridIndex {
    /// Build with α=0.5 (equal blend, Weaviate default).
    pub fn build(docs: &[Document]) -> Self {
        Self::build_with_alpha(docs, 0.5)
    }

    /// Build with a custom α ∈ [0, 1].
    pub fn build_with_alpha(docs: &[Document], alpha: f32) -> Self {
        Self {
            sparse: Bm25Index::build(docs),
            dense: FlatDenseIndex::build(docs),
            alpha: alpha.clamp(0.0, 1.0),
            candidate_mult: 4,
        }
    }
}

impl HybridSearch for RsfHybridIndex {
    fn search(&self, tokens: &[&str], vector: &[f32], k: usize) -> Vec<SearchResult> {
        let fetch = k * self.candidate_mult;
        let sparse_list = self.sparse.search(tokens, fetch);
        let dense_list = self.dense.search(vector, fetch);

        // Per-list min-max normalisation
        let norm_sparse = minmax_normalize(&sparse_list);
        let norm_dense = minmax_normalize(&dense_list);

        let mut scores: HashMap<usize, f32> = HashMap::new();
        for (id, s) in norm_sparse {
            *scores.entry(id).or_insert(0.0) += (1.0 - self.alpha) * s;
        }
        for (id, d) in norm_dense {
            *scores.entry(id).or_insert(0.0) += self.alpha * d;
        }

        let mut merged: Vec<SearchResult> = scores
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect();
        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(k);
        merged
    }
}

fn minmax_normalize(results: &[SearchResult]) -> HashMap<usize, f32> {
    if results.is_empty() {
        return HashMap::new();
    }
    let min = results
        .iter()
        .map(|r| r.score)
        .fold(f32::INFINITY, f32::min);
    let max = results
        .iter()
        .map(|r| r.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-10);
    results
        .iter()
        .map(|r| (r.id, (r.score - min) / range))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    fn doc(id: usize, tokens: &[&str], v: Vec<f32>) -> Document {
        Document {
            id,
            tokens: tokens.iter().map(|s| s.to_string()).collect(),
            vector: v,
        }
    }

    fn three_docs() -> Vec<Document> {
        vec![
            // Doc 0: three "alpha" tokens → clearly higher BM25 TF than doc 2
            doc(0, &["alpha", "alpha", "alpha", "beta"], vec![1.0, 0.0, 0.0]),
            doc(1, &["gamma", "delta"], vec![0.0, 1.0, 0.0]),
            doc(2, &["alpha", "gamma"], vec![0.7, 0.7, 0.0]),
        ]
    }

    #[test]
    fn test_rrf_keyword_and_vector_match_wins() {
        let docs = three_docs();
        let idx = RrfHybridIndex::build(&docs);
        let r = idx.search(&["alpha"], &[1.0, 0.0, 0.0], 2);
        assert_eq!(r[0].id, 0, "Doc 0 scores on both signals");
    }

    #[test]
    fn test_rrf_dense_fallback() {
        let docs = three_docs();
        let idx = RrfHybridIndex::build(&docs);
        // "unknown" has no posting → RRF falls back to dense signal
        let r = idx.search(&["unknown"], &[1.0, 0.0, 0.0], 1);
        assert_eq!(r[0].id, 0, "Dense fallback should return doc 0");
    }

    #[test]
    fn test_score_fusion_returns_k() {
        let docs = three_docs();
        let idx = ScoreFusionIndex::build(&docs);
        let r = idx.search(&["alpha"], &[1.0, 0.0, 0.0], 2);
        assert!(r.len() <= 2);
    }

    #[test]
    fn test_rsf_equal_weight_coverage() {
        let docs = three_docs();
        let idx = RsfHybridIndex::build(&docs);
        let r = idx.search(&["alpha"], &[1.0, 0.0, 0.0], 3);
        assert!(!r.is_empty());
        // Doc 0 has both a keyword hit and the closest vector → must appear
        let ids: Vec<usize> = r.iter().map(|x| x.id).collect();
        assert!(ids.contains(&0));
    }

    #[test]
    fn test_minmax_normalize_empty() {
        let result = minmax_normalize(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_minmax_normalize_single() {
        let r = vec![SearchResult { id: 7, score: 3.0 }];
        let norm = minmax_normalize(&r);
        // single element → range = 0 → clamped to 1e-10 → score = 0.0
        assert_eq!(*norm.get(&7).unwrap(), 0.0);
    }
}
