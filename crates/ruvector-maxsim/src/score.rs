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

/// MaxSim score between a multi-vector query and a multi-vector document.
///
/// Time: O(|query_vecs| * |doc_vecs| * D).
pub fn maxsim(query_vecs: &[Embedding], doc_vecs: &[Embedding]) -> f32 {
    query_vecs
        .iter()
        .map(|q| {
            doc_vecs
                .iter()
                .map(|d| cosine(q, d))
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
}
