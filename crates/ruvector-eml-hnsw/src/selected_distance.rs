//! Selected-dimension cosine distance — the runtime "fast path" derived from
//! [`EmlDistanceModel::selected_dims`].
//!
//! Architecture: EML runs offline to pick which dimensions discriminate on the
//! caller's data distribution. At search time we just take plain cosine over
//! those dims. No EML tree evaluation per call.
//!
//! This is the path PR #353's author recommended after finding the per-call
//! EML tree was 2.1× slower than baseline. That recommendation was never
//! shipped as callable code; this file ships it.

use crate::cosine_decomp::{cosine_distance_f32, EmlDistanceModel};
use simsimd::SpatialSimilarity;

/// SimSIMD-backed cosine distance over full-dim vectors.
///
/// Returns `1 - cosine_similarity`, clamped to `[0.0, 2.0]`. Falls back to the
/// scalar reference implementation if SimSIMD returns `None` (e.g. on an
/// unsupported CPU, or mismatched lengths).
///
/// This is the kernel used by `EmlHnsw::search_with_rerank` when the metric
/// is cosine — the reduced-dim HNSW produces a candidate set, and this runs
/// once per candidate at full dim. SimSIMD gives us the AVX/NEON/SVE path
/// without building our own intrinsics.
#[inline]
pub fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    match <f32 as SpatialSimilarity>::cosine(a, b) {
        Some(d) => (d as f32).clamp(0.0, 2.0),
        None => cosine_distance_f32(a, b),
    }
}


/// Plain cosine distance computed over a chosen subset of dimensions.
///
/// `dims` must contain valid indices into `a` and `b`. The function L2-renorms
/// over the projected subspace, so the result is a true cosine distance on the
/// reduced representation and stays in `[0.0, 2.0]`.
#[inline]
pub fn cosine_distance_selected(a: &[f32], b: &[f32], dims: &[usize]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if dims.is_empty() {
        return 1.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for &i in dims {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-30 {
        return 1.0;
    }
    let similarity = dot / denom;
    (1.0 - similarity).clamp(0.0, 2.0) as f32
}

/// Squared-Euclidean proxy computed over a subset of dimensions.
///
/// Monotonic in full Euclidean distance over the same subset, so ranking is
/// preserved. Use when the underlying metric is L2 and you want the absolute
/// cheapest kernel (no sqrt, no norm accumulation).
#[inline]
pub fn sq_euclidean_selected(a: &[f32], b: &[f32], dims: &[usize]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f64;
    for &i in dims {
        let d = (a[i] - b[i]) as f64;
        s += d * d;
    }
    s as f32
}

/// Project a full-dimension vector onto the subset of dimensions.
///
/// Output length equals `dims.len()`.
#[inline]
pub fn project_vector(full: &[f32], dims: &[usize]) -> Vec<f32> {
    dims.iter().map(|&i| full[i]).collect()
}

/// Project a batch of vectors in-place-style (allocates a new Vec).
pub fn project_batch<V: AsRef<[f32]>>(full: &[V], dims: &[usize]) -> Vec<Vec<f32>> {
    full.iter().map(|v| project_vector(v.as_ref(), dims)).collect()
}

impl EmlDistanceModel {
    /// Runtime "fast path": plain cosine over the selected dimensions.
    ///
    /// This bypasses the EML tree — the tree was the offline teacher that
    /// discovered which dimensions discriminate. At search time there is no
    /// reason to pay for tree evaluation; the selected indices ARE the fast
    /// path.
    ///
    /// Falls back to full cosine if the model hasn't been trained.
    pub fn selected_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if !self.is_trained() {
            return crate::cosine_decomp::cosine_distance_f32(a, b);
        }
        cosine_distance_selected(a, b, self.selected_dims())
    }

    /// Project a full-dim vector to the model's selected subspace.
    ///
    /// Returns `None` if the model has not been trained.
    pub fn project(&self, full: &[f32]) -> Option<Vec<f32>> {
        if !self.is_trained() {
            return None;
        }
        Some(project_vector(full, self.selected_dims()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_distance_selected_identical() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let d = cosine_distance_selected(&a, &a, &[0, 2, 4]);
        assert!(d.abs() < 1e-6, "identical → 0.0, got {d}");
    }

    #[test]
    fn cosine_distance_selected_orthogonal_subspace() {
        // In the chosen subspace the two vectors become orthogonal.
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        let d = cosine_distance_selected(&a, &b, &[0, 1]);
        assert!((d - 1.0).abs() < 1e-6, "orthogonal → 1.0, got {d}");
    }

    #[test]
    fn cosine_distance_selected_opposite_subspace() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![-1.0f32, -2.0, 4.0];
        // On dims [0,1] they point in exactly opposite directions.
        let d = cosine_distance_selected(&a, &b, &[0, 1]);
        assert!((d - 2.0).abs() < 1e-4, "opposite → 2.0, got {d}");
    }

    #[test]
    fn project_vector_basic() {
        let v = vec![10.0f32, 20.0, 30.0, 40.0];
        let p = project_vector(&v, &[0, 3]);
        assert_eq!(p, vec![10.0, 40.0]);
    }

    #[test]
    fn project_vector_empty_dims() {
        let v = vec![1.0f32, 2.0];
        let p = project_vector(&v, &[]);
        assert!(p.is_empty());
    }

    #[test]
    fn empty_dims_returns_orthogonal() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let d = cosine_distance_selected(&a, &b, &[]);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sq_euclidean_selected_monotonic() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        let c = vec![2.0f32, 0.0, 0.0];
        let ab = sq_euclidean_selected(&a, &b, &[0]);
        let ac = sq_euclidean_selected(&a, &c, &[0]);
        assert!(ac > ab);
    }

    fn random_vec(seed: u64, dim: usize) -> Vec<f32> {
        let mut s = seed;
        (0..dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn cosine_distance_simd_matches_scalar_dim64() {
        for seed in 0..5 {
            let a = random_vec(seed, 64);
            let b = random_vec(seed + 100, 64);
            let scalar = crate::cosine_decomp::cosine_distance_f32(&a, &b);
            let simd = cosine_distance_simd(&a, &b);
            assert!((simd - scalar).abs() < 1e-4, "dim64 seed={} scalar={} simd={}", seed, scalar, simd);
        }
    }

    #[test]
    fn cosine_distance_simd_matches_scalar_dim128() {
        for seed in 0..5 {
            let a = random_vec(seed * 7 + 11, 128);
            let b = random_vec(seed * 7 + 12, 128);
            let scalar = crate::cosine_decomp::cosine_distance_f32(&a, &b);
            let simd = cosine_distance_simd(&a, &b);
            assert!((simd - scalar).abs() < 1e-4, "dim128 seed={} scalar={} simd={}", seed, scalar, simd);
        }
    }

    #[test]
    fn cosine_distance_simd_matches_scalar_dim384() {
        for seed in 0..5 {
            let a = random_vec(seed * 13 + 3, 384);
            let b = random_vec(seed * 13 + 4, 384);
            let scalar = crate::cosine_decomp::cosine_distance_f32(&a, &b);
            let simd = cosine_distance_simd(&a, &b);
            assert!((simd - scalar).abs() < 1e-4, "dim384 seed={} scalar={} simd={}", seed, scalar, simd);
        }
    }

    #[test]
    fn cosine_distance_simd_self_is_zero() {
        let a = random_vec(99, 128);
        let d = cosine_distance_simd(&a, &a);
        assert!(d.abs() < 1e-5, "self-distance ~0, got {d}");
    }
}
