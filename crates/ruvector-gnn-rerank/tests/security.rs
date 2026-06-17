//! Adversarial input hardening for the reranker (productionizes #479, security step).
//!
//! Guarantees: malformed / poisoned candidate sets are rejected fail-fast with a
//! typed error instead of panicking or producing a silently-corrupted ranking
//! (the poisoned-first-stage / MemoryGraft threat model). No input should ever
//! panic any reranker variant.

use ruvector_gnn_rerank::{
    Candidate, CandidateReranker, ExactL2Reranker, GnnDiffusionReranker, GnnMincutReranker,
    NoisyScoreReranker, RerankerError,
};

fn cand(id: u32, vector: Vec<f32>, noisy_score: f32) -> Candidate {
    Candidate { id, vector, noisy_score }
}

/// Run an input through every variant; return whether ALL returned Ok.
fn all_variants(query: &[f32], cands: &[Candidate], k: usize) -> Vec<Result<(), RerankerError>> {
    let g = GnnDiffusionReranker::default();
    let m = GnnMincutReranker::default();
    vec![
        NoisyScoreReranker.rerank(query, cands, k).map(|_| ()),
        g.rerank(query, cands, k).map(|_| ()),
        m.rerank(query, cands, k).map(|_| ()),
        ExactL2Reranker.rerank(query, cands, k).map(|_| ()),
    ]
}

#[test]
fn rejects_nan_score() {
    let cands = vec![cand(0, vec![1.0, 0.0], f32::NAN), cand(1, vec![0.0, 1.0], 0.5)];
    for r in all_variants(&[1.0, 0.0], &cands, 1) {
        assert!(matches!(r, Err(RerankerError::NonFinite { .. })), "NaN score must be rejected, got {r:?}");
    }
}

#[test]
fn rejects_inf_score() {
    let cands = vec![cand(0, vec![1.0, 0.0], f32::INFINITY), cand(1, vec![0.0, 1.0], 0.5)];
    assert!(matches!(
        GnnDiffusionReranker::default().rerank(&[1.0, 0.0], &cands, 1),
        Err(RerankerError::NonFinite { .. })
    ));
}

#[test]
fn rejects_nan_in_vector() {
    let cands = vec![cand(0, vec![f32::NAN, 0.0], 0.9), cand(1, vec![0.0, 1.0], 0.5)];
    assert!(matches!(
        GnnDiffusionReranker::default().rerank(&[1.0, 0.0], &cands, 1),
        Err(RerankerError::NonFinite { .. })
    ));
}

#[test]
fn rejects_candidate_dimension_mismatch() {
    let cands = vec![cand(0, vec![1.0, 0.0, 0.0], 0.9), cand(1, vec![0.0, 1.0], 0.5)];
    for r in all_variants(&[1.0, 0.0, 0.0], &cands, 1) {
        assert!(matches!(r, Err(RerankerError::DimMismatch { .. })), "dim mismatch must be rejected, got {r:?}");
    }
}

#[test]
fn rejects_empty_and_k_too_large() {
    assert!(matches!(GnnDiffusionReranker::default().rerank(&[1.0], &[], 1), Err(RerankerError::Empty)));
    let cands = vec![cand(0, vec![1.0], 0.5)];
    assert!(matches!(GnnDiffusionReranker::default().rerank(&[1.0], &cands, 5), Err(RerankerError::KTooLarge { .. })));
}

#[test]
fn degenerate_inputs_do_not_panic() {
    // k=0 → empty result; single candidate; all-identical vectors (zero/degenerate
    // cosine); k == n. None of these may panic.
    let one = vec![cand(0, vec![1.0, 2.0], 0.5)];
    assert_eq!(GnnDiffusionReranker::default().rerank(&[1.0, 2.0], &one, 0).unwrap().len(), 0);
    assert_eq!(GnnDiffusionReranker::default().rerank(&[1.0, 2.0], &one, 1).unwrap().len(), 1);

    let identical: Vec<Candidate> = (0..5).map(|i| cand(i, vec![0.0, 0.0, 0.0], 0.1 * i as f32)).collect();
    for r in all_variants(&[0.0, 0.0, 0.0], &identical, 5) {
        assert!(r.is_ok(), "all-identical/zero vectors must not error or panic, got {r:?}");
    }
}
