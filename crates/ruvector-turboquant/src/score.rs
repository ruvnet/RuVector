//! Metric scoring over Turbo4 blobs — no f32 reconstruction, ever.
//!
//! Every supported metric decomposes over the level dot product:
//!
//! ```text
//! dot(a,b) ≈ αₐ·α_b · Σ L[aᵢ]·L[bᵢ]          (int kernel: · I8_UNIT²)
//! L2²(a,b) = αₐ²Sₐ + α_b²S_b − 2·dot(a,b)
//! cos(a,b) = dot(a,b) / (αₐ√D · α_b√D)       (α cancels — exact norms)
//! ```
//!
//! Three tiers, in decreasing speed / increasing fidelity:
//! 1. [`symmetric_distance`] — code×code (graph construction, 4-bit both sides);
//! 2. [`asymmetric_distance`] — int8-query×code (traversal, 8-bit query);
//! 3. [`rescore`] — exact f32 rotated query×code (final re-ranking).
//!
//! Distance conventions match `ruvector-core`'s f32 `DistanceFn`: Euclidean
//! returns the (positive) root, Cosine returns `1 − cos` clamped at 0,
//! DotProduct returns `−dot` clamped at 0 (hnsw_rs asserts non-negativity).

use crate::codec::{split_code, split_query, Turbo4Query};
use crate::simd::{dot_i8_nibble, dot_nibble_nibble};
use crate::tables::I8_UNIT;

/// Distance metrics Turbo4 can score directly. (Manhattan does not decompose
/// over the dot product and is rejected at index construction.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Euclidean,
    Cosine,
    DotProduct,
}

#[inline]
fn finish(metric: Metric, dot: f32, norm_sq_a: f32, norm_sq_b: f32) -> f32 {
    match metric {
        Metric::Euclidean => (norm_sq_a + norm_sq_b - 2.0 * dot).max(0.0).sqrt(),
        Metric::Cosine => {
            let denom = (norm_sq_a * norm_sq_b).sqrt();
            if denom > 0.0 {
                (1.0 - dot / denom).max(0.0)
            } else {
                1.0
            }
        }
        Metric::DotProduct => (-dot).max(0.0),
    }
}

/// RaBitQ-style length renormalization (the bias fix Qdrant's production
/// Turbo4 borrows; see ADR-296 refinements). The reconstruction `α·L[cᵢ]`
/// has norm `α√S`, while the true norm is exactly `α√D` (standardized
/// coordinates satisfy `Σzᵢ² = D`). Scaling the level dot by `√(D/S)` per
/// side makes the estimate use norm-exact directions, removing the
/// magnitude component of Lloyd-Max's inner-product bias at zero storage
/// and near-zero compute cost.
#[inline]
fn renorm(dim: usize, s: f32) -> f32 {
    if s > 0.0 {
        (dim as f32 / s).sqrt()
    } else {
        0.0
    }
}

/// Code×code distance (both sides 4-bit). Used for HNSW graph construction
/// and neighbor-to-neighbor evaluation.
#[inline]
pub fn symmetric_distance(metric: Metric, a: &[u8], b: &[u8], dim: usize) -> f32 {
    let (na, alpha_a, s_a) = split_code(a, dim);
    let (nb, alpha_b, s_b) = split_code(b, dim);
    let dot_int = dot_nibble_nibble(na, nb, dim) as f32;
    let dot = dot_int * I8_UNIT * I8_UNIT * alpha_a * alpha_b * renorm(dim, s_a) * renorm(dim, s_b);
    // Norms are exact under renormalization: ‖v‖² = α²·D.
    let d = dim as f32;
    finish(metric, dot, alpha_a * alpha_a * d, alpha_b * alpha_b * d)
}

/// Query×code distance (8-bit query, 4-bit code). Used during traversal.
#[inline]
pub fn asymmetric_distance(metric: Metric, query: &[u8], code: &[u8], dim: usize) -> f32 {
    let (q_i8, qscale, q_norm_sq) = split_query(query, dim);
    let (nc, alpha, s) = split_code(code, dim);
    let dot_int = dot_i8_nibble(nc, q_i8, dim) as f32;
    let dot = dot_int * qscale * I8_UNIT * alpha * renorm(dim, s);
    finish(metric, dot, q_norm_sq, alpha * alpha * dim as f32)
}

/// Exact rescore: f32 rotated query × code. No integer error — only the
/// 4-bit code error remains. Used to re-rank the top `k·rescore_multiplier`
/// traversal candidates.
pub fn rescore(metric: Metric, query: &Turbo4Query, code: &[u8], dim: usize) -> f32 {
    let (nc, alpha, s) = split_code(code, dim);
    // SIMD f32×nibble kernel on the int8 level grid; the grid rounding
    // (≤ I8_UNIT/2) is ~1 % of the code error, so this tier stays the
    // highest-fidelity scorer.
    let dot_lvl = crate::simd::dot_f32_nibble(nc, &query.rotated, dim) * I8_UNIT;
    let dot = dot_lvl * alpha * renorm(dim, s);
    finish(metric, dot, query.norm_sq, alpha * alpha * dim as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::Turbo4Codec;
    use crate::rotation::SplitMix64;

    fn gauss_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64(seed);
        let mut out = Vec::with_capacity(dim);
        while out.len() < dim {
            let u1 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let r = (-2.0 * u1.max(1e-12).ln()).sqrt();
            let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
            out.push((r * c) as f32);
            if out.len() < dim {
                out.push((r * s) as f32);
            }
        }
        out
    }

    fn exact(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum();
        let nb: f32 = b.iter().map(|x| x * x).sum();
        finish(metric, dot, na, nb)
    }

    #[test]
    fn all_tiers_approximate_exact_distance() {
        let dim = 512;
        let codec = Turbo4Codec::new(dim, 42).unwrap();
        for seed in 0..8u64 {
            let a = gauss_vec(dim, seed * 2 + 1);
            let b = gauss_vec(dim, seed * 2 + 2);
            let ca = codec.encode(&a).unwrap();
            let cb = codec.encode(&b).unwrap();
            let qa = codec.encode_query(&a).unwrap();

            for metric in [Metric::Euclidean, Metric::Cosine] {
                let truth = exact(metric, &a, &b);
                let sym = symmetric_distance(metric, &ca, &cb, dim);
                let asym = asymmetric_distance(metric, &qa.blob, &cb, dim);
                let resc = rescore(metric, &qa, &cb, dim);
                let tol = match metric {
                    Metric::Euclidean => 0.06 * truth.max(1.0),
                    _ => 0.05,
                };
                assert!(
                    (sym - truth).abs() < 2.0 * tol,
                    "{metric:?} sym {sym} vs {truth}"
                );
                assert!(
                    (asym - truth).abs() < tol,
                    "{metric:?} asym {asym} vs {truth}"
                );
                assert!(
                    (resc - truth).abs() < tol,
                    "{metric:?} rescore {resc} vs {truth}"
                );
                // Fidelity ordering on average is rescore ≤ asym ≤ sym error;
                // enforce only that rescore is at least as close as symmetric.
                assert!(
                    (resc - truth).abs() <= (sym - truth).abs() + tol,
                    "{metric:?} rescore worse than symmetric"
                );
            }
        }
    }

    /// The renormalized estimator must be (near-)unbiased: across many
    /// pairs the *mean signed* relative L2 error stays near zero. Without
    /// renormalization Lloyd-Max reconstructions are systematically short
    /// (‖r‖ = α√S < α√D), which skews distances one direction — the bias
    /// Qdrant corrects with RaBitQ-style renormalization.
    #[test]
    fn renormalized_estimator_is_unbiased() {
        let dim = 512;
        let codec = Turbo4Codec::new(dim, 42).unwrap();
        let mut sum_rel = 0.0f64;
        let n = 30;
        for seed in 0..n {
            let a = gauss_vec(dim, 1000 + seed);
            let b = gauss_vec(dim, 2000 + seed);
            let qa = codec.encode_query(&a).unwrap();
            let cb = codec.encode(&b).unwrap();
            let truth = exact(Metric::Euclidean, &a, &b);
            let est = rescore(Metric::Euclidean, &qa, &cb, dim);
            sum_rel += ((est - truth) / truth) as f64;
        }
        let mean_rel = sum_rel / n as f64;
        assert!(
            mean_rel.abs() < 0.01,
            "mean signed relative L2 error {mean_rel:.4} indicates estimator bias"
        );
    }

    #[test]
    fn self_distance_is_near_zero() {
        let dim = 256;
        let codec = Turbo4Codec::new(dim, 7).unwrap();
        let v = gauss_vec(dim, 5);
        let c = codec.encode(&v).unwrap();
        let q = codec.encode_query(&v).unwrap();
        let d = rescore(Metric::Euclidean, &q, &c, dim);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(d < 0.15 * norm, "self distance {d} vs norm {norm}");
        let cos = rescore(Metric::Cosine, &q, &c, dim);
        assert!(cos < 0.02, "self cosine distance {cos}");
    }

    #[test]
    fn ranking_agreement_with_exact() {
        // Turbo4 asymmetric scores must rank neighbors like exact f32 does.
        let dim = 128;
        let n = 200;
        let codec = Turbo4Codec::new(dim, 42).unwrap();
        let base: Vec<Vec<f32>> = (0..n).map(|i| gauss_vec(dim, 100 + i as u64)).collect();
        let codes: Vec<Vec<u8>> = base.iter().map(|v| codec.encode(v).unwrap()).collect();
        let query = gauss_vec(dim, 999);
        let qq = codec.encode_query(&query).unwrap();

        let mut ex: Vec<(usize, f32)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| (i, exact(Metric::Euclidean, &query, v)))
            .collect();
        ex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let exact_top10: std::collections::HashSet<usize> =
            ex[..10].iter().map(|(i, _)| *i).collect();

        let mut ap: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .map(|(i, c)| (i, rescore(Metric::Euclidean, &qq, c, dim)))
            .collect();
        ap.sort_by(|a, b| a.1.total_cmp(&b.1));
        let approx_top10: Vec<usize> = ap[..10].iter().map(|(i, _)| *i).collect();

        let hits = approx_top10
            .iter()
            .filter(|i| exact_top10.contains(i))
            .count();
        assert!(hits >= 8, "recall@10 {hits}/10 too low for flat rescore");
    }
}
