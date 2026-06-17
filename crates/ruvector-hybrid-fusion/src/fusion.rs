//! Three hybrid fusion strategies for combining sparse (BM25) and dense (cosine) results.
//!
//! - [`rrf_fuse`]: Reciprocal Rank Fusion (k=60), the standard ensemble baseline.
//! - [`linear_fuse`]: Min-max normalised linear combination with fixed α=0.5.
//! - [`coherence_fuse`]: Per-query adaptive α derived from score-concentration ratio.
//!
//! # Coherence-adaptive weighting
//!
//! For each query we estimate how "confident" each leg is by comparing the
//! top-1 score to the mean of the top-k scores (concentration ratio).  A leg
//! with a high concentration ratio has a clearer winner and should receive more
//! weight. This implements the DAT (Dynamic Alpha Tuning, arXiv 2503.23013)
//! principle using a lightweight, embedding-free proxy signal.

use std::collections::{HashMap, HashSet};

/// Standard Reciprocal Rank Fusion.
///
/// `rrf_k` is the rank-smoothing constant (typical: 60).
pub fn rrf_fuse(
    sparse: &[(usize, f32)],
    dense: &[(usize, f32)],
    rrf_k: f32,
    top_n: usize,
) -> Vec<(usize, f32)> {
    let mut scores: HashMap<usize, f32> = HashMap::new();

    for (rank, &(id, _)) in sparse.iter().enumerate() {
        *scores.entry(id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32 + 1.0);
    }
    for (rank, &(id, _)) in dense.iter().enumerate() {
        *scores.entry(id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32 + 1.0);
    }

    let mut result: Vec<(usize, f32)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result.truncate(top_n);
    result
}

/// Fixed-weight linear combination with min-max normalisation, α=0.5.
pub fn linear_fuse(
    sparse: &[(usize, f32)],
    dense: &[(usize, f32)],
    top_n: usize,
) -> Vec<(usize, f32)> {
    let s_norm = minmax_normalise(sparse);
    let d_norm = minmax_normalise(dense);
    linear_combine(&s_norm, &d_norm, 0.5, top_n)
}

/// Per-query coherence-adaptive linear combination.
///
/// α is computed from the *concentration ratio* of each leg:
///   concentration(leg) = top1_score_normalised / mean_top_k_scores_normalised
/// A leg with higher concentration (clear winner at top) receives more weight.
///
///   α_dense = conc_dense / (conc_sparse + conc_dense)
///   final_score = (1 - α) · sparse_norm + α · dense_norm
///
/// This implements the DAT principle (arXiv 2503.23013) using a lightweight
/// proxy signal instead of a learned model.  In practice this formulation
/// consistently outperforms RRF on keyword-heavy queries while matching RRF
/// on hybrid queries.  See research doc for nuanced query-type breakdown.
pub fn coherence_fuse(
    sparse: &[(usize, f32)],
    dense: &[(usize, f32)],
    top_n: usize,
) -> Vec<(usize, f32)> {
    let s_norm = minmax_normalise(sparse);
    let d_norm = minmax_normalise(dense);
    let alpha = compute_alpha(&s_norm, &d_norm);
    linear_combine(&s_norm, &d_norm, alpha, top_n)
}

// ── internal helpers ──────────────────────────────────────────────────────────

/// Min-max normalise scores to [0, 1].  Returns empty vec if input is empty.
fn minmax_normalise(results: &[(usize, f32)]) -> Vec<(usize, f32)> {
    if results.is_empty() {
        return Vec::new();
    }
    let max = results.iter().map(|r| r.1).fold(f32::NEG_INFINITY, f32::max);
    let min = results.iter().map(|r| r.1).fold(f32::INFINITY, f32::min);
    let span = (max - min).max(1e-9);
    results
        .iter()
        .map(|&(id, s)| (id, (s - min) / span))
        .collect()
}

/// Compute per-query alpha for coherence fusion.
fn compute_alpha(sparse_norm: &[(usize, f32)], dense_norm: &[(usize, f32)]) -> f32 {
    let conc_s = concentration(sparse_norm);
    let conc_d = concentration(dense_norm);
    let total = conc_s + conc_d;
    if total < 1e-9 {
        0.5
    } else {
        conc_d / total
    }
}

/// Concentration ratio: top-1 score / mean-of-list score (both normalised to [0,1]).
/// Returns 0 when the list is empty.
fn concentration(results: &[(usize, f32)]) -> f32 {
    if results.is_empty() {
        return 0.0;
    }
    let top1 = results[0].1;
    let mean = results.iter().map(|r| r.1).sum::<f32>() / results.len() as f32;
    top1 / (mean + 1e-9)
}

/// Merge two normalised result lists with weight `alpha` on the dense leg.
fn linear_combine(
    sparse: &[(usize, f32)],
    dense: &[(usize, f32)],
    alpha: f32,
    top_n: usize,
) -> Vec<(usize, f32)> {
    let mut all_ids: HashSet<usize> = HashSet::new();
    all_ids.extend(sparse.iter().map(|r| r.0));
    all_ids.extend(dense.iter().map(|r| r.0));

    let s_map: HashMap<usize, f32> = sparse.iter().copied().collect();
    let d_map: HashMap<usize, f32> = dense.iter().copied().collect();

    let mut combined: Vec<(usize, f32)> = all_ids
        .into_iter()
        .map(|id| {
            let s = s_map.get(&id).copied().unwrap_or(0.0);
            let d = d_map.get(&id).copied().unwrap_or(0.0);
            (id, (1.0 - alpha) * s + alpha * d)
        })
        .collect();

    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    combined.truncate(top_n);
    combined
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sparse() -> Vec<(usize, f32)> {
        vec![(0, 5.0), (1, 3.0), (2, 1.0)]
    }

    fn make_dense() -> Vec<(usize, f32)> {
        vec![(3, 0.9), (4, 0.8), (0, 0.7)]
    }

    #[test]
    fn rrf_fuse_union_of_both() {
        let result = rrf_fuse(&make_sparse(), &make_dense(), 60.0, 10);
        let ids: Vec<usize> = result.iter().map(|r| r.0).collect();
        // Doc 0 appears in both lists and should score highest
        assert_eq!(ids[0], 0, "doc 0 in both lists should top RRF");
        // All 5 unique docs must appear
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn rrf_scores_positive() {
        for &(_, s) in &rrf_fuse(&make_sparse(), &make_dense(), 60.0, 10) {
            assert!(s > 0.0);
        }
    }

    #[test]
    fn linear_fuse_bounded() {
        let result = linear_fuse(&make_sparse(), &make_dense(), 10);
        for &(_, s) in &result {
            assert!(s >= 0.0 - 1e-6 && s <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn coherence_fuse_returns_results() {
        let result = coherence_fuse(&make_sparse(), &make_dense(), 3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn coherence_fuse_empty_sparse() {
        // When one leg is empty, alpha → 0 or 1 gracefully
        let result = coherence_fuse(&[], &make_dense(), 3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn minmax_single_element() {
        let r = minmax_normalise(&[(7, 42.0)]);
        assert!((r[0].1 - 0.0).abs() < 1e-6); // single element → 0.0 (max-min=0)
    }

    #[test]
    fn concentration_empty() {
        assert!((concentration(&[]) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn concentration_uniform() {
        // All equal → conc ≈ 1.0
        let r = vec![(0, 0.5), (1, 0.5), (2, 0.5)];
        let c = concentration(&r);
        assert!((c - 1.0).abs() < 1e-3, "uniform conc should be ≈1.0, got {}", c);
    }
}
