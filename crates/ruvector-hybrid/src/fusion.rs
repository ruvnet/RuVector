// Fusion strategies for combining dense and sparse retrieval results.
//
// Three strategies are implemented:
//
// 1. RRF (Reciprocal Rank Fusion, Cormack et al. 2009):
//    score(d) = Σ_r 1 / (k_rrf + rank_r(d))
//    k_rrf = 60 is the canonical default.
//    RRF is parameter-free, robust, and the dominant choice in 2026 production systems.
//
// 2. Linear score interpolation:
//    score(d) = α·dense_norm(d) + (1-α)·sparse_norm(d)
//    Scores are normalised to [0,1] by dividing by the max score in each list.
//    Requires tuning α; α=0.5 is a strong default for balanced queries.
//
// 3. Max-of-signals fusion:
//    score(d) = max(dense_norm(d), sparse_norm(d))
//    Useful when either signal alone can identify the best match.

use crate::Scored;
use std::collections::HashMap;

/// k constant for RRF, canonical value = 60.
pub const RRF_K: f32 = 60.0;

/// Reciprocal Rank Fusion.
/// `dense` and `sparse` are already-ranked lists (best first).
/// `candidate_k` may be larger than k — the caller fetches more candidates for fusion.
pub fn rrf(dense: &[Scored], sparse: &[Scored], k: usize) -> Vec<Scored> {
    let mut acc: HashMap<u32, f32> = HashMap::new();

    for (rank, s) in dense.iter().enumerate() {
        *acc.entry(s.id).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
    }
    for (rank, s) in sparse.iter().enumerate() {
        *acc.entry(s.id).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
    }

    sorted_top_k(acc, k)
}

/// Linear score interpolation with per-list max-normalisation.
/// `alpha`: weight of dense signal (0.0 = sparse-only, 1.0 = dense-only).
pub fn linear(dense: &[Scored], sparse: &[Scored], k: usize, alpha: f32) -> Vec<Scored> {
    let dense_max = dense
        .first()
        .map(|s| s.score.abs())
        .unwrap_or(1.0)
        .max(1e-9);
    let sparse_max = sparse
        .first()
        .map(|s| s.score.abs())
        .unwrap_or(1.0)
        .max(1e-9);

    let mut acc: HashMap<u32, f32> = HashMap::new();

    for s in dense {
        *acc.entry(s.id).or_insert(0.0) += alpha * (s.score / dense_max);
    }
    for s in sparse {
        *acc.entry(s.id).or_insert(0.0) += (1.0 - alpha) * (s.score / sparse_max);
    }

    sorted_top_k(acc, k)
}

/// Max-of-signals fusion.
pub fn max_signal(dense: &[Scored], sparse: &[Scored], k: usize) -> Vec<Scored> {
    let dense_max = dense
        .first()
        .map(|s| s.score.abs())
        .unwrap_or(1.0)
        .max(1e-9);
    let sparse_max = sparse
        .first()
        .map(|s| s.score.abs())
        .unwrap_or(1.0)
        .max(1e-9);

    let mut acc: HashMap<u32, f32> = HashMap::new();

    for s in dense {
        let entry = acc.entry(s.id).or_insert(0.0);
        *entry = entry.max(s.score / dense_max);
    }
    for s in sparse {
        let entry = acc.entry(s.id).or_insert(0.0);
        *entry = entry.max(s.score / sparse_max);
    }

    sorted_top_k(acc, k)
}

fn sorted_top_k(acc: HashMap<u32, f32>, k: usize) -> Vec<Scored> {
    let mut v: Vec<Scored> = acc
        .into_iter()
        .map(|(id, score)| Scored::new(id, score))
        .collect();
    v.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    v.truncate(k);
    v
}

/// Oracle fusion: used to generate ground-truth labels for recall@k evaluation.
/// Scores each document by α*normalised_dense + (1-α)*normalised_sparse using ALL
/// documents (not just top candidates), so recall is meaningful.
pub fn oracle_top_k(
    all_dense: &[Scored],
    all_sparse: &[Scored],
    k: usize,
    alpha: f32,
) -> Vec<Scored> {
    linear(all_dense, all_sparse, k, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(id: u32, score: f32) -> Scored {
        Scored::new(id, score)
    }

    #[test]
    fn rrf_combines_both_lists() {
        // Doc 5 appears in both lists at rank 0 — should win.
        let dense = vec![s(5, 0.9), s(1, 0.8), s(2, 0.7)];
        let sparse = vec![s(5, 1.0), s(3, 0.6)];
        let fused = rrf(&dense, &sparse, 3);
        assert_eq!(fused[0].id, 5, "doc 5 should rank first after RRF");
    }

    #[test]
    fn rrf_parameter_free_symmetry() {
        let a = vec![s(1, 1.0), s(2, 0.5)];
        let b = vec![s(2, 1.0), s(1, 0.5)];
        // Doc 1 top in a, doc 2 top in b: scores should be close (not identical due to rank).
        let fused = rrf(&a, &b, 2);
        // Both docs should appear.
        assert!(fused.iter().any(|r| r.id == 1));
        assert!(fused.iter().any(|r| r.id == 2));
    }

    #[test]
    fn linear_alpha_1_is_pure_dense() {
        let dense = vec![s(1, 0.9), s(2, 0.8)];
        let sparse = vec![s(99, 1.0), s(98, 0.9)]; // completely different docs
        let fused = linear(&dense, &sparse, 2, 1.0);
        // alpha=1.0: only dense matters
        assert!(
            fused.iter().all(|r| r.id == 1 || r.id == 2),
            "alpha=1.0 should return dense-only results"
        );
    }

    #[test]
    fn linear_alpha_0_is_pure_sparse() {
        let dense = vec![s(1, 0.9), s(2, 0.8)];
        let sparse = vec![s(99, 1.0), s(98, 0.9)];
        let fused = linear(&dense, &sparse, 2, 0.0);
        assert!(
            fused.iter().all(|r| r.id == 99 || r.id == 98),
            "alpha=0.0 should return sparse-only results"
        );
    }

    #[test]
    fn max_signal_takes_best() {
        let dense = vec![s(10, 1.0)];
        let sparse = vec![s(20, 1.0)];
        let fused = max_signal(&dense, &sparse, 2);
        // Both get max-normalised score 1.0, both should appear.
        assert_eq!(fused.len(), 2);
    }
}
