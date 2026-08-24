//! MaxSim scoring: the core late-interaction kernel.
//!
//! For a query Q = {q_1, …, q_n} and document D = {d_1, …, d_m} the score is
//!
//!   score(Q, D) = Σ_{i=1}^{n}  max_{j=1}^{m}  cosine(q_i, d_j)
//!
//! This sums, over every query token, the *best-matching* document token.
//! Unlike averaging into a single vector, late interaction preserves the
//! multi-facet structure: a document about "Rust" AND "memory safety" scores
//! highly for either topic independently.

use crate::types::Embedding;

/// Cosine similarity in [-1, 1] between two vectors of equal length.
/// Returns 0.0 when either vector is zero-magnitude.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch in cosine");
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only seam counting calls to [`norm`], which `maxsim` uses
    /// exclusively for the hoisted per-query-token norm. Thread-local (not a
    /// shared atomic) so parallel test threads never pollute each other's
    /// count; a given test's `maxsim` call runs synchronously on its own
    /// thread, so this is race-free without any lock.
    static QUERY_NORM_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// L2 norm of a vector.
#[inline]
fn norm(v: &[f32]) -> f32 {
    #[cfg(test)]
    QUERY_NORM_CALLS.with(|c| c.set(c.get() + 1));
    let mut acc = 0.0_f32;
    for &x in v.iter() {
        acc += x * x;
    }
    acc.sqrt()
}

/// Cosine similarity with the left vector's norm supplied by the caller.
///
/// Accumulates in the same order as [`cosine`] and combines the same two
/// factors into the same denominator, so a caller that passes `norm(a)` gets
/// bit-identical results to calling [`cosine`] directly — but only when `a`
/// and `b` have equal length: `norm_a` was computed over the full `a`, so a
/// truncated zip over `a` and `b` here would combine a full-length norm with
/// a truncated dot product. Callers with mismatched lengths must use
/// [`cosine`] (or [`cosine_truncating`]) instead.
#[inline]
fn cosine_with_lhs_norm(a: &[f32], b: &[f32], norm_a: f32) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch in cosine");
    let mut dot = 0.0_f32;
    let mut nb = 0.0_f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        nb += bi * bi;
    }
    let denom = norm_a * nb.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

/// Fused cosine without [`cosine`]'s equal-length precondition: both norms
/// and the dot product accumulate inside the same `zip`, so mismatched
/// lengths truncate to the shorter vector rather than panicking in debug
/// builds. This is [`maxsim`]'s fallback for query/document token pairs of
/// differing dimension, where the hoisted `norm(q)` (taken over the full
/// query) would otherwise be combined with a dot product truncated to the
/// document's length.
#[inline]
fn cosine_truncating(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

/// MaxSim score between a multi-vector query and a multi-vector document.
///
/// Time: O(|query_vecs| * |doc_vecs| * D).
///
/// Each query token's norm is computed once and reused across every document
/// token of matching dimension, rather than being recomputed inside each
/// pairwise cosine. The inner loop therefore does two multiply-adds per
/// dimension instead of three. Pairs whose query and document tokens have
/// different lengths fall back to [`cosine_truncating`], since the hoisted
/// query norm is only bit-identical to per-pair cosine when both operands
/// are the same length.
pub fn maxsim(query_vecs: &[Embedding], doc_vecs: &[Embedding]) -> f32 {
    query_vecs
        .iter()
        .map(|q| {
            let norm_q = norm(q);
            doc_vecs
                .iter()
                .map(|d| {
                    if q.len() == d.len() {
                        cosine_with_lhs_norm(q, d, norm_q)
                    } else {
                        cosine_truncating(q, d)
                    }
                })
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// Dot product (assumes pre-normalised vectors for speed; use `cosine` otherwise).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// L2-normalise a vector in place.
pub fn l2_norm(v: &mut [f32]) {
    let len = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if len > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![1.0_f32, 0.0, 0.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn maxsim_single_query_single_doc() {
        let q = vec![vec![1.0_f32, 0.0, 0.0]];
        let d = vec![vec![1.0_f32, 0.0, 0.0]];
        assert!((maxsim(&q, &d) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn maxsim_picks_best_doc_token() {
        // Query = one token in X direction.
        // Doc has two tokens: X and Y. MaxSim should pick X (cosine=1).
        let q = vec![vec![1.0_f32, 0.0]];
        let d = vec![
            vec![1.0_f32, 0.0], // cos=1
            vec![0.0_f32, 1.0], // cos=0
        ];
        let s = maxsim(&q, &d);
        assert!((s - 1.0).abs() < 1e-5, "expected ~1.0, got {s}");
    }

    #[test]
    fn maxsim_multi_query_sums() {
        // Two orthogonal query tokens. Doc has two matching doc tokens.
        let q = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let d = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let s = maxsim(&q, &d);
        // Each query token matches exactly one doc token → sum = 2.0
        assert!((s - 2.0).abs() < 1e-5, "expected ~2.0, got {s}");
    }

    /// MaxSim recomputed with the naive, un-hoisted formulation: every
    /// pairwise score goes through the public [`cosine`], recomputing the
    /// query token's norm on each call instead of once per query token.
    ///
    /// This is the property that matters after `cosine` went back to a
    /// single fused pass: [`maxsim`]'s hoist (`norm(q)` once, then
    /// [`cosine_with_lhs_norm`] per document token) must still agree with
    /// calling the real `cosine` per pair. The two sides run genuinely
    /// different code — one composes `norm` + `cosine_with_lhs_norm`, the
    /// other calls fused `cosine` — so the comparison still guards a real
    /// invariant instead of comparing a function with a hand-copy of itself.
    fn maxsim_recomputing_query_norm(q: &[Embedding], d: &[Embedding]) -> f32 {
        q.iter()
            .map(|qv| {
                d.iter()
                    .map(|dv| cosine(qv, dv))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum()
    }

    fn vecs(count: usize, dim: usize, seed: u32) -> Vec<Embedding> {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (0..count)
            .map(|_| (0..dim).map(|_| next()).collect())
            .collect()
    }

    /// Hoisting the query norm must be exact, not merely close.
    ///
    /// The norm is accumulated in the same order and multiplied into the same
    /// denominator either way, so any difference at all would mean the
    /// refactor changed the arithmetic. Compared bitwise for that reason.
    #[test]
    fn hoisting_query_norm_is_bit_exact() {
        for dim in [1usize, 3, 8, 16, 33, 128, 384] {
            for (nq, nd) in [(1usize, 1usize), (1, 7), (5, 1), (4, 9)] {
                let q = vecs(nq, dim, dim as u32);
                let d = vecs(nd, dim, dim as u32 + 17);
                let got = maxsim(&q, &d);
                let want = maxsim_recomputing_query_norm(&q, &d);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "dim={dim} nq={nq} nd={nd}: {got} vs {want}"
                );
            }
        }
    }

    /// A zero-magnitude query token must still take the degenerate branch.
    #[test]
    fn zero_query_token_scores_zero() {
        let q = vec![vec![0.0_f32; 8]];
        let d = vecs(4, 8, 3);
        assert_eq!(maxsim(&q, &d), 0.0);
        assert_eq!(maxsim(&q, &d), maxsim_recomputing_query_norm(&q, &d));
    }

    /// A zero-magnitude document token must not poison the max.
    #[test]
    fn zero_doc_token_does_not_poison_max() {
        let q = vecs(2, 8, 5);
        let mut d = vecs(3, 8, 11);
        d.push(vec![0.0_f32; 8]);
        assert_eq!(
            maxsim(&q, &d).to_bits(),
            maxsim_recomputing_query_norm(&q, &d).to_bits()
        );
    }

    /// The hoist must retain its shape, not just its output: `maxsim` must
    /// compute each query token's norm exactly once, no matter how many
    /// document tokens it is scored against. More than one document token is
    /// used deliberately — a formulation that recomputes the query norm once
    /// per document token would tie with the hoisted one when `nd == 1`.
    #[test]
    fn hoist_computes_query_norm_once_per_token_regardless_of_doc_count() {
        let q = vecs(3, 8, 51);
        let d = vecs(5, 8, 59);
        QUERY_NORM_CALLS.with(|c| c.set(0));
        let _ = maxsim(&q, &d);
        let calls = QUERY_NORM_CALLS.with(|c| c.get());
        assert_eq!(
            calls,
            q.len(),
            "expected exactly one query-norm computation per query token \
             regardless of document count, got {calls} for {} query tokens \
             against {} document tokens",
            q.len(),
            d.len()
        );
    }

    /// A document containing ONLY a zero-magnitude token must score 0.0 and
    /// finite, not NaN silently masked by the max-fold's NaN-ignoring
    /// semantics.
    #[test]
    fn zero_only_document_scores_zero_and_finite() {
        let q = vecs(2, 8, 41);
        let d = vec![vec![0.0_f32; 8]];
        let got = maxsim(&q, &d);
        assert_eq!(got, 0.0);
        assert!(got.is_finite());
        assert_eq!(got, maxsim_recomputing_query_norm(&q, &d));
    }

    /// Every document token being zero-magnitude (not just one among
    /// otherwise-normal tokens) must still resolve to 0.0 and finite.
    #[test]
    fn all_zero_document_tokens_score_zero_and_finite() {
        let q = vecs(2, 8, 43);
        let d = vec![vec![0.0_f32; 8]; 4];
        let got = maxsim(&q, &d);
        assert_eq!(got, 0.0);
        assert!(got.is_finite());
        assert_eq!(got, maxsim_recomputing_query_norm(&q, &d));
    }

    /// A query token longer than the document token it is scored against
    /// must fall back to the pre-hoist, truncating-zip cosine rather than
    /// combining a norm taken over the full query with a dot product
    /// truncated to the document's length. q=[3,4,12] vs d=[3,4]: the
    /// truncating base semantics give dot=25, na=25 (truncated to 2 terms),
    /// nb=25, so 25/(5*5)=1.0 exactly. The hoisted fast path would instead
    /// divide by norm(q)=13, giving 25/(13*5)=0.3846....
    #[test]
    fn mismatched_query_longer_scores_via_truncating_fallback() {
        let q = vec![vec![3.0_f32, 4.0, 12.0]];
        let d = vec![vec![3.0_f32, 4.0]];
        let got = maxsim(&q, &d);
        assert_eq!(got, 1.0, "expected exact 1.0, got {got}");
    }

    /// The opposite mismatch direction — document token longer than the
    /// query token — must also route through the truncating fallback and
    /// agree with calling it directly on the same pair.
    #[test]
    fn mismatched_document_longer_matches_truncating_cosine() {
        let q = vec![vec![3.0_f32, 4.0]];
        let d = vec![vec![3.0_f32, 4.0, 12.0]];
        let got = maxsim(&q, &d);
        let want = cosine_truncating(&q[0], &d[0]);
        assert_eq!(got, want);
    }
}
