//! Distance calculations for the router's vector metrics.
//!
//! The default path is scalar. The loops below are manually unrolled eight
//! elements at a time, which helps the autovectorizer but does not emit vector
//! instructions on its own; nothing in this module has ever called SimSIMD,
//! despite what this file's header used to say.
//!
//! With the `lattice-simd` feature, all four metrics route through
//! `lattice-embed`'s runtime-dispatched kernels (AVX-512F, AVX2, NEON, wasm32
//! SIMD128, each with its own scalar fallback).
//!
//! The two paths do not return bit-identical values: SIMD kernels reduce in a
//! different order than the scalar loops below, so results can differ by
//! floating-point rounding. Each routed function documents the bound this
//! module's tests enforce: relative error `1e-4`, with an absolute floor of
//! `1e-5` for results near zero. Every metric keeps the sign and
//! similarity-to-distance conversion this module already defined, and the
//! degenerate-input branches are unchanged.

use crate::error::{Result, VectorDbError};
use crate::types::DistanceMetric;

/// Calculate distance between two vectors using specified metric
pub fn calculate_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> Result<f32> {
    if a.len() != b.len() {
        return Err(VectorDbError::InvalidDimensions {
            expected: a.len(),
            actual: b.len(),
        });
    }

    match metric {
        DistanceMetric::Euclidean => Ok(euclidean_distance(a, b)),
        DistanceMetric::Cosine => Ok(cosine_similarity(a, b)),
        DistanceMetric::DotProduct => Ok(dot_product(a, b)),
        DistanceMetric::Manhattan => Ok(manhattan_distance(a, b)),
    }
}

/// Euclidean distance (L2).
///
/// With `lattice-simd` enabled, the result is computed by a SIMD kernel that
/// reduces in a different order than the scalar loop below, so it may differ
/// from the scalar result by bounded floating-point rounding: relative error
/// up to `1e-4`, with an absolute floor of `1e-5` for results near zero.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "lattice-simd")]
    {
        // Equal lengths only. The scalar path below indexes `b` by `a`'s
        // length and panics on a short `b`, where lattice returns f32::MAX;
        // routing only equal lengths keeps enabling the feature from turning
        // a panic into a silent value.
        if a.len() == b.len() {
            return lattice_embed::simd::euclidean_distance(a, b);
        }
    }

    euclidean_distance_scalar(a, b)
}

#[inline]
fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    // Process in chunks for better SIMD utilization
    let len = a.len();
    let mut i = 0;

    // Main loop - process 8 elements at a time for AVX2
    while i + 8 <= len {
        for j in 0..8 {
            let diff = a[i + j] - b[i + j];
            sum += diff * diff;
        }
        i += 8;
    }

    // Handle remaining elements
    while i < len {
        let diff = a[i] - b[i];
        sum += diff * diff;
        i += 1;
    }

    sum.sqrt()
}

/// Cosine distance.
/// Returns 1 - cosine_similarity to convert similarity to distance
///
/// With `lattice-simd` enabled, the result is computed by a SIMD kernel that
/// reduces in a different order than the scalar loop below, so it may differ
/// from the scalar result by bounded floating-point rounding: relative error
/// up to `1e-4`, with an absolute floor of `1e-5` for results near zero.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "lattice-simd")]
    {
        // lattice returns 0.0 for a zero-magnitude operand, which lands on the
        // same 1.0 this function's own zero check returns, so the degenerate
        // case needs no special handling here.
        if a.len() == b.len() {
            return 1.0 - lattice_embed::simd::cosine_similarity(a, b);
        }
    }

    cosine_similarity_scalar(a, b)
}

#[inline]
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    let len = a.len();
    let mut i = 0;

    // Process in chunks
    while i + 8 <= len {
        for j in 0..8 {
            let ai = a[i + j];
            let bi = b[i + j];
            dot += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }
        i += 8;
    }

    // Handle remaining
    while i < len {
        let ai = a[i];
        let bi = b[i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
        i += 1;
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance
    }

    // Convert similarity to distance
    1.0 - (dot / (norm_a * norm_b))
}

/// Dot product, negated so that a larger similarity is a smaller distance.
///
/// With `lattice-simd` enabled, the result is computed by a SIMD kernel that
/// reduces in a different order than the scalar loop below, so it may differ
/// from the scalar result by bounded floating-point rounding: relative error
/// up to `1e-4`, with an absolute floor of `1e-5` for results near zero.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "lattice-simd")]
    {
        // The negation stays here rather than in the backend, so both paths
        // agree on the similarity-to-distance convention.
        if a.len() == b.len() {
            return -lattice_embed::simd::dot_product(a, b);
        }
    }

    dot_product_scalar(a, b)
}

#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    let len = a.len();
    let mut i = 0;

    // Process in chunks
    while i + 8 <= len {
        for j in 0..8 {
            sum += a[i + j] * b[i + j];
        }
        i += 8;
    }

    // Handle remaining
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    -sum // Negate to convert similarity to distance
}

/// Manhattan distance (L1).
///
/// With `lattice-simd` enabled, the result is computed by a SIMD kernel that
/// reduces in a different order than the scalar loop below, so it may differ
/// from the scalar result by bounded floating-point rounding: relative error
/// up to `1e-4`, with an absolute floor of `1e-5` for results near zero.
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "lattice-simd")]
    {
        // Equal lengths only, for the same reason as `euclidean_distance`: the
        // scalar path indexes `b` by `a`'s length and panics on a short `b`,
        // where lattice returns f32::MAX.
        if a.len() == b.len() {
            return lattice_embed::simd::manhattan_distance(a, b);
        }
    }

    manhattan_distance_scalar(a, b)
}

#[inline]
fn manhattan_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    let len = a.len();
    let mut i = 0;

    // Process in chunks
    while i + 8 <= len {
        for j in 0..8 {
            sum += (a[i + j] - b[i + j]).abs();
        }
        i += 8;
    }

    // Handle remaining
    while i < len {
        sum += (a[i] - b[i]).abs();
        i += 1;
    }

    sum
}

/// Batch distance calculation for multiple queries
pub fn batch_distance(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    vectors
        .par_iter()
        .map(|v| calculate_distance(query, v, metric))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.196).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 0.01); // Same vectors = distance 0
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = dot_product(&a, &b);
        assert!((dot - (-32.0)).abs() < 0.01); // Negated
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = manhattan_distance(&a, &b);
        assert!((dist - 9.0).abs() < 0.01);
    }

    fn pair(dim: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (
            (0..dim).map(|_| next()).collect(),
            (0..dim).map(|_| next()).collect(),
        )
    }

    fn reference_euclidean(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = f64::from(*x) - f64::from(*y);
            acc += d * d;
        }
        acc.sqrt() as f32
    }

    fn reference_cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += f64::from(*x) * f64::from(*y);
            na += f64::from(*x) * f64::from(*x);
            nb += f64::from(*y) * f64::from(*y);
        }
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        (1.0 - dot / (na.sqrt() * nb.sqrt())) as f32
    }

    fn reference_dot(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            acc += f64::from(*x) * f64::from(*y);
        }
        -acc as f32
    }

    fn reference_manhattan(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            acc += (f64::from(*x) - f64::from(*y)).abs();
        }
        acc as f32
    }

    /// Bound the parity test enforces between the compiled backend and the
    /// f64 reference: relative error, with an absolute floor for results
    /// near zero. This must match the numbers stated in this module's doc
    /// comments.
    const REL_TOL: f32 = 1e-4;
    const ABS_FLOOR: f32 = 1e-5;

    fn assert_within_tolerance(name: &str, dim: usize, seed: u32, got: f32, want: f32) {
        let tol = ABS_FLOOR.max(REL_TOL * want.abs());
        assert!(
            (got - want).abs() <= tol,
            "{name} dim={dim} seed={seed}: got={got} want={want} diff={} tol={tol}",
            (got - want).abs()
        );
    }

    /// Whichever backend is compiled must agree with an independent f64
    /// reference within the bound documented on each routed function.
    ///
    /// Dimensions straddle the manual 8-wide chunk boundary, the 4/8/16-lane
    /// widths a SIMD backend uses, and include large (384, 1536) and odd
    /// (385) lengths, so both remainder paths and realistic embedding sizes
    /// are exercised rather than assumed. Sign and the similarity-to-distance
    /// conversion are part of what is compared, not just magnitude.
    #[test]
    fn backends_match_reference() {
        for dim in [
            1usize, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 384, 385, 768, 1536,
        ] {
            for seed in 0..4u32 {
                let (a, b) = pair(dim, seed);

                for (name, got, want) in [
                    (
                        "euclidean",
                        euclidean_distance(&a, &b),
                        reference_euclidean(&a, &b),
                    ),
                    (
                        "cosine",
                        cosine_similarity(&a, &b),
                        reference_cosine(&a, &b),
                    ),
                    ("dot", dot_product(&a, &b), reference_dot(&a, &b)),
                    (
                        "manhattan",
                        manhattan_distance(&a, &b),
                        reference_manhattan(&a, &b),
                    ),
                ] {
                    assert_within_tolerance(name, dim, seed, got, want);
                }

                // Directly enforces the bound the rustdoc on each routed
                // function promises against the scalar path, not just against
                // the f64 reference: two implementations each within `tol` of
                // a third can differ from each other by up to `2 * tol`, so
                // the vs-reference assertions above don't cover this. Under
                // the default build this compares a function with itself
                // (diff 0); under `lattice-simd` it is the real check.
                for (name, got, want) in [
                    (
                        "euclidean/routed_vs_scalar",
                        euclidean_distance(&a, &b),
                        euclidean_distance_scalar(&a, &b),
                    ),
                    (
                        "cosine/routed_vs_scalar",
                        cosine_similarity(&a, &b),
                        cosine_similarity_scalar(&a, &b),
                    ),
                    (
                        "dot/routed_vs_scalar",
                        dot_product(&a, &b),
                        dot_product_scalar(&a, &b),
                    ),
                    (
                        "manhattan/routed_vs_scalar",
                        manhattan_distance(&a, &b),
                        manhattan_distance_scalar(&a, &b),
                    ),
                ] {
                    assert_within_tolerance(name, dim, seed, got, want);
                }
            }
        }
    }

    /// Public functions must agree with this module's own private scalar
    /// implementation on NaN, infinities, signed zero, and empty slices —
    /// whichever backend (`lattice-simd` or scalar) is compiled. This is not
    /// a tolerance comparison: exceptional-value propagation should be exact
    /// or explicitly NaN in both paths, so any disagreement is a real bug in
    /// the routed backend, not rounding.
    #[test]
    fn exceptional_inputs_match_scalar() {
        fn assert_matches_scalar(name: &str, got: f32, scalar: f32) {
            let matches = if scalar.is_nan() {
                got.is_nan()
            } else {
                got.to_bits() == scalar.to_bits()
            };
            assert!(
                matches,
                "{name}: routed={got:?} (0x{:08x}) scalar={scalar:?} (0x{:08x}) disagree on an exceptional input",
                got.to_bits(),
                scalar.to_bits()
            );
        }

        let base: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut nan_v = base.clone();
        nan_v[10] = f32::NAN;
        let mut pos_inf_v = base.clone();
        pos_inf_v[10] = f32::INFINITY;
        let mut neg_inf_v = base.clone();
        neg_inf_v[10] = f32::NEG_INFINITY;
        let mut neg_zero_v = base.clone();
        neg_zero_v[10] = -0.0;
        let mut pos_zero_v = base.clone();
        pos_zero_v[10] = 0.0;

        let cases: [(&str, &[f32], &[f32]); 8] = [
            ("nan_in_a", &nan_v, &base),
            ("nan_in_b", &base, &nan_v),
            ("pos_inf_in_a", &pos_inf_v, &base),
            ("neg_inf_in_a", &neg_inf_v, &base),
            ("pos_inf_vs_pos_inf", &pos_inf_v, &pos_inf_v),
            ("pos_inf_vs_neg_inf", &pos_inf_v, &neg_inf_v),
            ("signed_zero", &neg_zero_v, &pos_zero_v),
            ("empty", &[], &[]),
        ];

        for (name, a, b) in cases {
            assert_matches_scalar(
                &format!("euclidean/{name}"),
                euclidean_distance(a, b),
                euclidean_distance_scalar(a, b),
            );
            assert_matches_scalar(
                &format!("cosine/{name}"),
                cosine_similarity(a, b),
                cosine_similarity_scalar(a, b),
            );
            assert_matches_scalar(
                &format!("dot/{name}"),
                dot_product(a, b),
                dot_product_scalar(a, b),
            );
            assert_matches_scalar(
                &format!("manhattan/{name}"),
                manhattan_distance(a, b),
                manhattan_distance_scalar(a, b),
            );
        }
    }

    /// Length mismatch, called directly on each public function rather than
    /// through `calculate_distance`: a longer `b` is silently truncated to
    /// `a`'s length (documented on the scalar loops), and a shorter `b`
    /// panics because the scalar loop indexes `b` by `a`'s length. Mismatched
    /// lengths never route through `lattice-simd` (it requires equal
    /// lengths), so this behaviour is identical in both build configurations.
    #[test]
    fn length_mismatch_direct_call_matches_documented_behaviour() {
        let long = vec![1.0f32, 2.0, 3.0, 4.0];
        let short = vec![1.0f32, 2.0];

        for (name, f) in [
            ("euclidean", euclidean_distance as fn(&[f32], &[f32]) -> f32),
            ("cosine", cosine_similarity as fn(&[f32], &[f32]) -> f32),
            ("dot", dot_product as fn(&[f32], &[f32]) -> f32),
            ("manhattan", manhattan_distance as fn(&[f32], &[f32]) -> f32),
        ] {
            // `b` longer than `a`: the loop only walks `a`'s length, so `b`
            // is silently truncated to `a`'s first elements and the result
            // matches comparing `a` against that prefix of `b`.
            let truncated = f(&short, &long);
            let expected = f(&short, &long[..short.len()]);
            assert_eq!(
                truncated, expected,
                "{name}: longer b should be truncated to a's length"
            );

            // `b` shorter than `a`: the scalar loop indexes `b` by `a`'s
            // length and panics out of bounds.
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&long, &short)));
            assert!(
                result.is_err(),
                "{name}: a longer than b should panic on out-of-bounds indexing"
            );
        }
    }

    /// A zero-magnitude operand must give maximum cosine distance, not NaN.
    #[test]
    fn cosine_zero_vector_is_max_distance() {
        let zero = vec![0.0f32; 64];
        let (v, _) = pair(64, 21);
        assert_eq!(cosine_similarity(&zero, &v), 1.0);
        assert_eq!(cosine_similarity(&v, &zero), 1.0);
        assert_eq!(cosine_similarity(&zero, &zero), 1.0);
    }

    /// Dimension mismatch is rejected before any metric runs.
    #[test]
    fn calculate_distance_rejects_mismatched_dimensions() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0];
        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Cosine,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            assert!(calculate_distance(&a, &b, metric).is_err(), "{metric:?}");
        }
    }
}
