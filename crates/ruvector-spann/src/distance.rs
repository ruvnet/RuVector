//! Distance computation for SPANN partition index.
//!
//! Two backends compute the same quantities. The default is the scalar code
//! that has always been here. With the `lattice-simd` feature the inner
//! products come from `lattice-embed`'s runtime-dispatched SIMD kernels
//! (AVX-512 / AVX2 / NEON / wasm32 SIMD128, with its own scalar fallback).
//!
//! The backend split covers the inner products only. Length handling and the
//! small-norm guard live outside it, so both backends take the same branches
//! and differ only in how the sums are accumulated.

#[cfg(all(test, feature = "lattice-simd"))]
thread_local! {
    static LATTICE_L2_WITNESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static LATTICE_DOT_WITNESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Routes to `lattice_embed`'s L2 kernel and records that the call returned.
///
/// The witness store lives inside this wrapper, not at the call site in
/// `l2_squared`, so that reverting the call site's expression to the scalar
/// fallback (while leaving this wrapper and its store untouched) stops the
/// wrapper from being invoked at all and the witness cannot fire.
#[cfg(feature = "lattice-simd")]
#[inline]
fn l2_lattice(a: &[f32], b: &[f32]) -> f32 {
    let result = lattice_embed::simd::squared_euclidean_distance(a, b);
    #[cfg(test)]
    LATTICE_L2_WITNESS.with(|w| w.set(true));
    result
}

/// Compute L2 squared distance between two f32 slices.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "lattice-simd")]
    {
        // Only the equal-length case is routed. lattice returns f32::MAX for a
        // length mismatch where the scalar path below truncates to the shorter
        // slice, so guarding here keeps the two backends from disagreeing on
        // an input the debug assertion already calls a caller bug.
        if a.len() == b.len() {
            return l2_lattice(a, b);
        }
    }

    l2_squared_scalar(a, b)
}

#[inline]
fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Compute cosine similarity (in [0, 2] range, lower = more similar).
/// Returns 1 - dot(a, b) / (|a| * |b|), scaled to [0, 2].
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (dot, norm_sq_a, norm_sq_b) = inner_products(a, b);
    let norm_a = norm_sq_a.sqrt();
    let norm_b = norm_sq_b.sqrt();
    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 1.0;
    }
    1.0 - dot / (norm_a * norm_b)
}

/// Routes to `lattice_embed`'s dot-product kernel for all three inner
/// products and records that the calls returned. See `l2_lattice` for why
/// the store lives in this wrapper rather than at the `inner_products` call
/// site.
#[cfg(feature = "lattice-simd")]
#[inline]
fn dot_lattice(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    use lattice_embed::simd::dot_product;
    let result = (dot_product(a, b), dot_product(a, a), dot_product(b, b));
    #[cfg(test)]
    LATTICE_DOT_WITNESS.with(|w| w.set(true));
    result
}

/// Returns `(dot(a, b), dot(a, a), dot(b, b))`.
///
/// `lattice_embed::simd::cosine_similarity` is deliberately not used here: it
/// applies its own zero-norm rule, while this module's contract is a 1e-9
/// threshold that returns 1.0. Composing the distance from three inner
/// products keeps that threshold the single place either backend decides it.
#[inline]
fn inner_products(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    #[cfg(feature = "lattice-simd")]
    {
        // lattice's dot_product returns 0.0 on a length mismatch, which would
        // read as a zero norm and short-circuit to 1.0. Route equal lengths only.
        if a.len() == b.len() {
            return dot_lattice(a, b);
        }
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_sq_a: f32 = a.iter().map(|x| x * x).sum();
    let norm_sq_b: f32 = b.iter().map(|x| x * x).sum();
    (dot, norm_sq_a, norm_sq_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_zero_self() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!(l2_squared(&v, &v) < 1e-9);
    }

    #[test]
    fn l2_known() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        assert!((l2_squared(&a, &b) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_identical() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!(cosine_distance(&v, &v) < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    fn reference_l2_squared(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = f64::from(*x) - f64::from(*y);
            acc += d * d;
        }
        acc as f32
    }

    fn reference_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += f64::from(*x) * f64::from(*y);
            na += f64::from(*x) * f64::from(*x);
            nb += f64::from(*y) * f64::from(*y);
        }
        let (na, nb) = (na.sqrt() as f32, nb.sqrt() as f32);
        if na < 1e-9 || nb < 1e-9 {
            return 1.0;
        }
        1.0 - (dot as f32) / (na * nb)
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

    /// Whichever backend is compiled must agree with an f64 reference.
    ///
    /// Dimensions straddle the 4/8/16-lane widths and the unrolled chunk sizes
    /// a SIMD backend uses, so remainder handling is exercised rather than
    /// assumed. Running it under both feature settings holds a default build
    /// and a `lattice-simd` build to one reference.
    #[test]
    fn backend_matches_reference() {
        for dim in [
            1usize, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 384, 768,
        ] {
            for seed in 0..4u32 {
                let (a, b) = pair(dim, seed);

                let got = l2_squared(&a, &b);
                let want = reference_l2_squared(&a, &b);
                assert!(
                    (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                    "l2_squared dim={dim} seed={seed}: {got} vs {want}"
                );

                let got = cosine_distance(&a, &b);
                let want = reference_cosine_distance(&a, &b);
                assert!(
                    (got - want).abs() <= 1e-4,
                    "cosine_distance dim={dim} seed={seed}: {got} vs {want}"
                );
            }
        }
    }

    /// A zero vector must take the small-norm branch, not produce a NaN.
    #[test]
    fn cosine_zero_vector_is_not_nan() {
        let zero = vec![0.0f32; 64];
        let (v, _) = pair(64, 9);
        assert_eq!(cosine_distance(&zero, &v), 1.0);
        assert_eq!(cosine_distance(&v, &zero), 1.0);
        assert_eq!(cosine_distance(&zero, &zero), 1.0);
    }

    /// Guards against a silent reversion of the `lattice-simd` routing back
    /// to the scalar loops. `backend_matches_reference` above cannot catch
    /// that: it would still pass if the routed calls were replaced by the
    /// scalar functions, since both agree with the f64 reference within
    /// tolerance, and a host without an accelerated path falls through to a
    /// `lattice_embed` scalar loop that can equal RuVector's own scalar sum
    /// bit-for-bit — a bit-inequality assertion would reject that valid
    /// route. This test instead witnesses that the `lattice_embed` call
    /// itself returned, independent of what bits it produced.
    #[cfg(feature = "lattice-simd")]
    #[test]
    fn lattice_backend_is_actually_called() {
        LATTICE_L2_WITNESS.with(|w| w.set(false));
        LATTICE_DOT_WITNESS.with(|w| w.set(false));

        let (a, b) = pair(768, 7);
        let _ = l2_squared(&a, &b);
        let _ = inner_products(&a, &b);

        assert!(
            LATTICE_L2_WITNESS.with(|w| w.get()),
            "l2_squared did not call lattice_embed::simd::squared_euclidean_distance; \
             the routing at distance.rs may have reverted to scalar"
        );
        assert!(
            LATTICE_DOT_WITNESS.with(|w| w.get()),
            "inner_products did not call lattice_embed::simd::dot_product; \
             the routing at distance.rs may have reverted to scalar"
        );
    }
}
