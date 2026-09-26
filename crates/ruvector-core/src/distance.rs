//! SIMD-optimized distance metrics
//!
//! For Euclidean, cosine, and dot, three mutually exclusive backends are selected at
//! compile time:
//!
//! - `lattice-simd`: `lattice-embed`'s kernels. Covers wasm32 (`simd128`) as well as
//!   x86_64 and aarch64, so it is the only backend that vectorizes on wasm.
//! - `simd` on non-wasm: SimSIMD. Its call sites are excluded on wasm32 (the simsimd
//!   dependency itself still resolves there), so scalar is used on wasm32 unless
//!   `lattice-simd` is enabled.
//! - otherwise: the portable scalar path.
//!
//! `lattice-simd` takes precedence where both are enabled. The scalar path stays the
//! reference implementation that the backends are checked against.
//!
//! Manhattan is not part of this split: it uses [`crate::simd_intrinsics`]'s
//! x86_64/aarch64 dispatch by default, regardless of the `simd` feature.
//!
//! ## Call sites
//!
//! The generic [`distance`] function and its four metric-specific adapters below back
//! the generic distance path (used by, e.g., [`crate::index::flat`]'s `FlatIndex`).
//! `crate::index::hnsw`'s HNSW index does not go through this module: it dispatches its
//! own kernels directly via `crate::simd_intrinsics` for every metric, so enabling
//! `lattice-simd` does not change HNSW's distance evaluations.

use crate::error::{Result, RuvectorError};
use crate::types::DistanceMetric;

/// Calculate distance between two vectors using the specified metric
#[inline]
pub fn distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> Result<f32> {
    if a.len() != b.len() {
        return Err(RuvectorError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    match metric {
        DistanceMetric::Euclidean => Ok(euclidean_distance(a, b)),
        DistanceMetric::Cosine => Ok(cosine_distance(a, b)),
        DistanceMetric::DotProduct => Ok(dot_product_distance(a, b)),
        DistanceMetric::Manhattan => Ok(manhattan_distance(a, b)),
    }
}

/// Euclidean (L2) distance
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "lattice-simd")]
    {
        // Already sqrt-ed, matching this function's contract.
        lattice_embed::simd::euclidean_distance(a, b)
    }
    #[cfg(all(
        not(feature = "lattice-simd"),
        feature = "simd",
        not(target_arch = "wasm32")
    ))]
    {
        (simsimd::SpatialSimilarity::sqeuclidean(a, b)
            .expect("SimSIMD euclidean failed")
            .sqrt()) as f32
    }
    #[cfg(all(
        not(feature = "lattice-simd"),
        any(not(feature = "simd"), target_arch = "wasm32")
    ))]
    {
        // Unrolled scalar fallback for WASM — 4x unroll for ILP
        let len = a.len();
        let chunks = len / 4;
        let mut sum = 0.0f32;
        for i in 0..chunks {
            let idx = i * 4;
            let d0 = a[idx] - b[idx];
            let d1 = a[idx + 1] - b[idx + 1];
            let d2 = a[idx + 2] - b[idx + 2];
            let d3 = a[idx + 3] - b[idx + 3];
            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }
        for i in (chunks * 4)..len {
            let d = a[i] - b[i];
            sum += d * d;
        }
        sum.sqrt()
    }
}

/// Cosine distance (1 - cosine_similarity)
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "lattice-simd")]
    {
        // lattice returns similarity; this function's contract is 1 - similarity.
        // Its kernels return 0.0 when either norm is exactly zero, so a zero vector
        // yields 1.0 here, matching the scalar path below.
        1.0 - lattice_embed::simd::cosine_similarity(a, b)
    }
    #[cfg(all(
        not(feature = "lattice-simd"),
        feature = "simd",
        not(target_arch = "wasm32")
    ))]
    {
        simsimd::SpatialSimilarity::cosine(a, b).expect("SimSIMD cosine failed") as f32
    }
    #[cfg(all(
        not(feature = "lattice-simd"),
        any(not(feature = "simd"), target_arch = "wasm32")
    ))]
    {
        // Single-pass cosine fallback for WASM — avoids 3x iteration overhead
        let (mut dot, mut norm_a_sq, mut norm_b_sq) = (0.0f32, 0.0f32, 0.0f32);
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            dot += ai * bi;
            norm_a_sq += ai * ai;
            norm_b_sq += bi * bi;
        }
        let denom = norm_a_sq.sqrt() * norm_b_sq.sqrt();
        if denom > 1e-8 {
            1.0 - (dot / denom)
        } else {
            1.0
        }
    }
}

/// Dot product distance (negative for maximization)
#[inline]
pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "lattice-simd")]
    {
        // Negated, matching this function's maximization contract.
        -lattice_embed::simd::dot_product(a, b)
    }
    #[cfg(all(
        not(feature = "lattice-simd"),
        feature = "simd",
        not(target_arch = "wasm32")
    ))]
    {
        let dot = simsimd::SpatialSimilarity::dot(a, b).expect("SimSIMD dot product failed");
        (-dot) as f32
    }
    #[cfg(all(
        not(feature = "lattice-simd"),
        any(not(feature = "simd"), target_arch = "wasm32")
    ))]
    {
        // Pure Rust fallback for WASM
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        -dot
    }
}

/// Manhattan (L1) distance — delegates to SIMD when available
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "lattice-simd")]
    {
        lattice_embed::simd::manhattan_distance(a, b)
    }
    #[cfg(not(feature = "lattice-simd"))]
    {
        // `simd_intrinsics` dispatches x86_64 and aarch64 and falls through to
        // scalar everywhere else, wasm32 included.
        crate::simd_intrinsics::manhattan_distance_simd(a, b)
    }
}

/// Batch distance calculation optimized with Rayon (native) or sequential (WASM)
pub fn batch_distances(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
) -> Result<Vec<f32>> {
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    {
        use rayon::prelude::*;
        vectors
            .par_iter()
            .map(|v| distance(query, v, metric))
            .collect()
    }
    #[cfg(any(not(feature = "parallel"), target_arch = "wasm32"))]
    {
        // Sequential fallback for WASM
        vectors.iter().map(|v| distance(query, v, metric)).collect()
    }
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
    fn test_cosine_distance() {
        // Test with identical vectors (should have distance ~0)
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let dist = cosine_distance(&a, &b);
        assert!(
            dist < 0.01,
            "Identical vectors should have ~0 distance, got {}",
            dist
        );

        // Test with opposite vectors (should have high distance)
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!(
            dist > 1.5,
            "Opposite vectors should have high distance, got {}",
            dist
        );
    }

    #[test]
    fn test_dot_product_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = dot_product_distance(&a, &b);
        assert!((dist + 32.0).abs() < 0.01); // -(4 + 10 + 18) = -32
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = manhattan_distance(&a, &b);
        assert!((dist - 9.0).abs() < 0.01); // |1-4| + |2-5| + |3-6| = 9
    }

    /// Reference implementations, deliberately naive and backend-independent.
    /// Whichever backend is compiled in must agree with these.
    mod reference {
        pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt()
        }

        pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na == 0.0 || nb == 0.0 {
                1.0
            } else {
                1.0 - dot / (na * nb)
            }
        }

        pub fn dot(a: &[f32], b: &[f32]) -> f32 {
            -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
        }

        pub fn manhattan(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>()
        }
    }

    /// Deterministic pseudo-random vectors, no dev-dependency needed.
    fn vecs(dim: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
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

    /// The active backend must agree with the scalar reference on every metric.
    ///
    /// This is what catches an adapter mistake: dropping the `1.0 -` on cosine or the
    /// negation on dot product still compiles and still passes the loose
    /// single-case assertions above, but fails here.
    #[test]
    fn test_backend_matches_scalar_reference() {
        // Dimensions straddling the SIMD lane widths (4/8/16) and their remainders,
        // so tail handling is exercised rather than assumed.
        for dim in [1usize, 3, 4, 7, 8, 15, 16, 17, 31, 64, 127, 384, 768] {
            for seed in 0..4u32 {
                let (a, b) = vecs(dim, seed);

                let got = euclidean_distance(&a, &b);
                let want = reference::euclidean(&a, &b);
                assert!(
                    (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                    "euclidean mismatch at dim={dim} seed={seed}: got {got}, want {want}"
                );

                let got = cosine_distance(&a, &b);
                let want = reference::cosine(&a, &b);
                assert!(
                    (got - want).abs() <= 1e-4,
                    "cosine mismatch at dim={dim} seed={seed}: got {got}, want {want}"
                );

                let got = dot_product_distance(&a, &b);
                let want = reference::dot(&a, &b);
                assert!(
                    (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                    "dot mismatch at dim={dim} seed={seed}: got {got}, want {want}"
                );

                let got = manhattan_distance(&a, &b);
                let want = reference::manhattan(&a, &b);
                assert!(
                    (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                    "manhattan mismatch at dim={dim} seed={seed}: got {got}, want {want}"
                );
            }
        }
    }

    /// A zero vector must not produce NaN, and cosine distance must saturate at 1.0.
    #[test]
    fn test_zero_vector_is_not_nan() {
        let zero = vec![0.0f32; 8];
        let other = vec![1.0f32; 8];

        let d = cosine_distance(&zero, &other);
        assert!(d.is_finite(), "cosine distance went non-finite: {d}");
        assert!(
            (d - 1.0).abs() < 1e-6,
            "zero vector should give cosine distance 1.0, got {d}"
        );

        assert!(euclidean_distance(&zero, &other).is_finite());
        assert!(dot_product_distance(&zero, &other).is_finite());
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = distance(&a, &b, DistanceMetric::Euclidean);
        assert!(result.is_err());
    }

    /// Recomputes what each adapter's contract says the `lattice-simd` backend must
    /// produce, straight from `lattice_embed`'s kernels. This is the seam that pins
    /// backend *selection* (not just arithmetic): the loose `test_backend_matches_scalar_reference`
    /// tolerance above passes even if an adapter silently fell back to the scalar path,
    /// but a bit-exact comparison against this function does not, since the scalar path's
    /// summation order and rounding differ from the lattice kernels'.
    #[cfg(feature = "lattice-simd")]
    fn lattice_kernel_result(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
        match metric {
            DistanceMetric::Euclidean => lattice_embed::simd::euclidean_distance(a, b),
            DistanceMetric::Cosine => 1.0 - lattice_embed::simd::cosine_similarity(a, b),
            DistanceMetric::DotProduct => -lattice_embed::simd::dot_product(a, b),
            DistanceMetric::Manhattan => lattice_embed::simd::manhattan_distance(a, b),
        }
    }

    /// Under `lattice-simd`, every public adapter must dispatch to the lattice kernels
    /// bit-for-bit. If any one of the four adapters is edited to fall through to its
    /// scalar or SimSIMD branch instead, this test fails even though the value stays
    /// numerically close, because the two implementations round differently.
    #[cfg(feature = "lattice-simd")]
    #[test]
    fn test_backend_selection_uses_lattice_simd() {
        for dim in [1usize, 3, 4, 7, 8, 15, 16, 17, 31, 64, 127, 384, 768] {
            for seed in 0..4u32 {
                let (a, b) = vecs(dim, seed);

                for metric in [
                    DistanceMetric::Euclidean,
                    DistanceMetric::Cosine,
                    DistanceMetric::DotProduct,
                    DistanceMetric::Manhattan,
                ] {
                    let got = distance(&a, &b, metric).unwrap();
                    let want = lattice_kernel_result(&a, &b, metric);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{metric:?} did not dispatch to the lattice-simd kernel at dim={dim} seed={seed}: got {got}, want {want}"
                    );
                }
            }
        }
    }

    /// Documents an intentional divergence: when the product of the two vector norms
    /// (`norm_a_sq.sqrt() * norm_b_sq.sqrt()`) is strictly between 0 and 1e-8, the scalar
    /// path's `denom > 1e-8` guard saturates cosine distance at 1.0, while the
    /// `lattice-simd` kernel only short-circuits on an *exactly* zero norm and otherwise
    /// computes the real cosine similarity. This is a deliberate contract difference
    /// between the two backends at the sub-1e-8 boundary, not a bug.
    ///
    /// The scalar assertion below calls the compiled production `cosine_distance` itself,
    /// under the same `cfg` as its scalar branch (`distance.rs:103-121`), rather than a
    /// hand-copied mirror — so a regression in that branch (e.g. dropping the saturation
    /// guard) fails this test instead of leaving a separately-maintained copy green.
    #[test]
    fn test_tiny_norm_cosine_divergence_is_intentional() {
        #[cfg(all(
            not(feature = "lattice-simd"),
            any(not(feature = "simd"), target_arch = "wasm32")
        ))]
        {
            // Each vector's norm is ~1e-9 (nonzero), so their product — the scalar
            // path's denom — is ~1e-18, well under its 1e-8 guard threshold.
            let a = vec![1e-9f32, 0.0, 0.0, 0.0];
            let b = vec![1e-9f32, 0.0, 0.0, 0.0];
            let scalar = cosine_distance(&a, &b);
            assert!(
                (scalar - 1.0).abs() < 1e-6,
                "scalar path should saturate at 1.0 below its 1e-8 denom guard, got {scalar}"
            );
        }

        #[cfg(feature = "lattice-simd")]
        {
            // These vectors are parallel, so the true cosine distance is ~0. The lattice
            // kernel does not apply the scalar path's 1e-8 guard, so it should report
            // that, diverging from the scalar path's saturation-at-1.0 behavior above.
            let a = vec![1e-9f32, 0.0, 0.0, 0.0];
            let b = vec![1e-9f32, 0.0, 0.0, 0.0];
            let lattice = cosine_distance(&a, &b);
            assert!(
                lattice.is_finite() && lattice < 0.5,
                "lattice path should not saturate at 1.0 for a tiny nonzero norm, got {lattice}"
            );
        }
    }
}
