//! SIMD-accelerated vector distance functions for kernel vector stores.
//!
//! This module provides optimized distance computations using:
//! - NEON instructions on ARM64
//! - AVX2/SSE instructions on x86_64
//! - Scalar fallback for other platforms
//!
//! Performance targets (ADR-087):
//! - Vector distance: SIMD acceleration with packed f32
//! - Process 4/8 floats per cycle depending on platform
//!
//! # Example
//!
//! ```
//! use ruvix_vecgraph::simd_distance::{cosine_similarity, euclidean_distance_squared};
//!
//! let a = [1.0f32, 2.0, 3.0, 4.0];
//! let b = [4.0f32, 3.0, 2.0, 1.0];
//!
//! let cosine = cosine_similarity(&a, &b);
//! let euclidean = euclidean_distance_squared(&a, &b);
//! ```

#[cfg(feature = "alloc")]
extern crate alloc;

/// SIMD capabilities detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdCapabilities {
    /// AVX2 is available (x86_64).
    pub avx2: bool,
    /// AVX-512 is available (x86_64).
    pub avx512: bool,
    /// NEON is available (ARM64).
    pub neon: bool,
    /// FMA (fused multiply-add) is available.
    pub fma: bool,
}

impl SimdCapabilities {
    /// Detects SIMD capabilities at runtime.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            avx2: cfg!(all(target_arch = "x86_64", target_feature = "avx2")),
            avx512: cfg!(all(target_arch = "x86_64", target_feature = "avx512f")),
            neon: cfg!(all(target_arch = "aarch64", target_feature = "neon")),
            fma: cfg!(target_feature = "fma"),
        }
    }

    /// Returns the optimal vector lane width for this platform.
    #[must_use]
    pub const fn lane_width(&self) -> usize {
        if self.avx512 {
            16 // 512 bits / 32 bits per f32
        } else if self.avx2 {
            8 // 256 bits / 32 bits per f32
        } else if self.neon {
            4 // 128 bits / 32 bits per f32
        } else {
            1 // Scalar fallback
        }
    }

    /// Returns true if any SIMD acceleration is available.
    #[must_use]
    pub const fn has_simd(&self) -> bool {
        self.avx2 || self.avx512 || self.neon
    }
}

impl Default for SimdCapabilities {
    fn default() -> Self {
        Self::detect()
    }
}

/// Computes cosine similarity between two vectors.
///
/// Returns a value in [-1, 1] where 1 means identical direction.
///
/// # Panics
///
/// Panics if vectors have different lengths.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        cosine_similarity_neon(a, b)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        cosine_similarity_avx2(a, b)
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        cosine_similarity_scalar(a, b)
    }
}

/// Computes squared Euclidean distance between two vectors.
///
/// Returns sum of squared differences (no sqrt for performance).
///
/// # Panics
///
/// Panics if vectors have different lengths.
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        euclidean_distance_squared_neon(a, b)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        euclidean_distance_squared_avx2(a, b)
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        euclidean_distance_squared_scalar(a, b)
    }
}

/// Computes dot product between two vectors.
///
/// # Panics
///
/// Panics if vectors have different lengths.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        dot_product_neon(a, b)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        dot_product_avx2(a, b)
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        dot_product_scalar(a, b)
    }
}

/// Computes L2 norm (magnitude) of a vector.
#[inline]
pub fn l2_norm(a: &[f32]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        l2_norm_neon(a)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        l2_norm_avx2(a)
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        l2_norm_scalar(a)
    }
}

// ============================================================================
// ARM64 NEON Implementations
// ============================================================================

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn cosine_similarity_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;

    unsafe {
        let mut dot_sum = vdupq_n_f32(0.0);
        let mut norm_a_sum = vdupq_n_f32(0.0);
        let mut norm_b_sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));

            // Fused multiply-add: dot += a * b
            dot_sum = vfmaq_f32(dot_sum, va, vb);
            // norm_a += a * a
            norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
            // norm_b += b * b
            norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);
        }

        // Horizontal sum
        let dot = vaddvq_f32(dot_sum);
        let norm_a = vaddvq_f32(norm_a_sum);
        let norm_b = vaddvq_f32(norm_b_sum);

        // Handle remaining elements
        let mut dot_tail = 0.0f32;
        let mut norm_a_tail = 0.0f32;
        let mut norm_b_tail = 0.0f32;

        for i in (chunks * 4)..n {
            let ai = a[i];
            let bi = b[i];
            dot_tail += ai * bi;
            norm_a_tail += ai * ai;
            norm_b_tail += bi * bi;
        }

        let total_dot = dot + dot_tail;
        let total_norm_a = (norm_a + norm_a_tail).sqrt();
        let total_norm_b = (norm_b + norm_b_tail).sqrt();

        if total_norm_a == 0.0 || total_norm_b == 0.0 {
            0.0
        } else {
            total_dot / (total_norm_a * total_norm_b)
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn euclidean_distance_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;

    unsafe {
        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));

            // diff = a - b
            let diff = vsubq_f32(va, vb);
            // sum += diff * diff
            sum = vfmaq_f32(sum, diff, diff);
        }

        // Horizontal sum
        let mut total = vaddvq_f32(sum);

        // Handle remaining elements
        for i in (chunks * 4)..n {
            let diff = a[i] - b[i];
            total += diff * diff;
        }

        total
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;

    unsafe {
        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            sum = vfmaq_f32(sum, va, vb);
        }

        let mut total = vaddvq_f32(sum);

        for i in (chunks * 4)..n {
            total += a[i] * b[i];
        }

        total
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn l2_norm_neon(a: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;

    unsafe {
        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            sum = vfmaq_f32(sum, va, va);
        }

        let mut total = vaddvq_f32(sum);

        for i in (chunks * 4)..n {
            total += a[i] * a[i];
        }

        total.sqrt()
    }
}

// ============================================================================
// x86_64 AVX2 Implementations
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;

    unsafe {
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

            // FMA: dot += a * b
            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
            norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
        }

        // Horizontal sum (AVX2)
        let dot = horizontal_sum_avx2(dot_sum);
        let norm_a = horizontal_sum_avx2(norm_a_sum);
        let norm_b = horizontal_sum_avx2(norm_b_sum);

        // Handle remaining elements
        let mut dot_tail = 0.0f32;
        let mut norm_a_tail = 0.0f32;
        let mut norm_b_tail = 0.0f32;

        for i in (chunks * 8)..n {
            let ai = a[i];
            let bi = b[i];
            dot_tail += ai * bi;
            norm_a_tail += ai * ai;
            norm_b_tail += bi * bi;
        }

        let total_dot = dot + dot_tail;
        let total_norm_a = (norm_a + norm_a_tail).sqrt();
        let total_norm_b = (norm_b + norm_b_tail).sqrt();

        if total_norm_a == 0.0 || total_norm_b == 0.0 {
            0.0
        } else {
            total_dot / (total_norm_a * total_norm_b)
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn euclidean_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;

    unsafe {
        let mut sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let mut total = horizontal_sum_avx2(sum);

        for i in (chunks * 8)..n {
            let diff = a[i] - b[i];
            total += diff * diff;
        }

        total
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;

    unsafe {
        let mut sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        let mut total = horizontal_sum_avx2(sum);

        for i in (chunks * 8)..n {
            total += a[i] * b[i];
        }

        total
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn l2_norm_avx2(a: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;

    unsafe {
        let mut sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            sum = _mm256_fmadd_ps(va, va, sum);
        }

        let mut total = horizontal_sum_avx2(sum);

        for i in (chunks * 8)..n {
            total += a[i] * a[i];
        }

        total.sqrt()
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn horizontal_sum_avx2(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;

    // Add high 128 bits to low 128 bits
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(high, low);

    // Horizontal add within 128-bit
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);

    _mm_cvtss_f32(sums2)
}

// ============================================================================
// Scalar Fallback Implementations
// ============================================================================

/// Scalar cosine similarity (portable fallback).
#[inline]
pub fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        let ai = a[i];
        let bi = b[i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Scalar squared Euclidean distance (portable fallback).
#[inline]
pub fn euclidean_distance_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

/// Scalar dot product (portable fallback).
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar L2 norm (portable fallback).
#[inline]
pub fn l2_norm_scalar(a: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in a {
        sum += x * x;
    }
    sum.sqrt()
}

// ============================================================================
// Batch Distance Computation
// ============================================================================

/// Result of a batch distance computation.
#[derive(Debug, Clone)]
pub struct DistanceResult {
    /// Index of the vector in the batch.
    pub index: usize,
    /// Distance value.
    pub distance: f32,
}

/// Computes distances from a query vector to all vectors in a batch.
///
/// Returns results sorted by distance (closest first).
///
/// # Arguments
///
/// * `query` - The query vector
/// * `batch` - Iterator over (index, vector) pairs
/// * `k` - Maximum number of results to return
///
/// # Panics
///
/// Panics if any vector in batch has different length than query.
#[cfg(feature = "alloc")]
pub fn batch_cosine_distances<'a, I>(
    query: &[f32],
    batch: I,
    k: usize,
) -> alloc::vec::Vec<DistanceResult>
where
    I: Iterator<Item = (usize, &'a [f32])>,
{
    use alloc::vec::Vec;

    let mut results: Vec<DistanceResult> = batch
        .map(|(index, vector)| DistanceResult {
            index,
            distance: 1.0 - cosine_similarity(query, vector), // Convert to distance
        })
        .collect();

    // Partial sort for top-k
    if results.len() > k {
        results.select_nth_unstable_by(k, |a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        results.truncate(k);
    }

    results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec::Vec;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [-1.0f32, 0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_zero() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let dist = euclidean_distance_squared(&a, &b);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_known() {
        let a = [0.0f32, 0.0, 0.0, 0.0];
        let b = [3.0f32, 4.0, 0.0, 0.0];
        let dist = euclidean_distance_squared(&a, &b);
        assert!((dist - 25.0).abs() < 1e-6); // 3^2 + 4^2 = 25
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [2.0f32, 3.0, 4.0, 5.0];
        let dot = dot_product(&a, &b);
        assert!((dot - 40.0).abs() < 1e-6); // 2+6+12+20 = 40
    }

    #[test]
    fn test_l2_norm() {
        let a = [3.0f32, 4.0, 0.0, 0.0];
        let norm = l2_norm(&a);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_large_vector() {
        // Test with 768 dimensions (common embedding size)
        let a: Vec<f32> = (0..768).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..768).map(|i| (i as f32) * 0.02).collect();

        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.99); // Same direction, just different magnitude

        let dot = dot_product(&a, &b);
        assert!(dot > 0.0);
    }

    #[test]
    fn test_scalar_matches_simd() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| ((i + 1) as f32) * 0.1).collect();

        let scalar_sim = cosine_similarity_scalar(&a, &b);
        let simd_sim = cosine_similarity(&a, &b);

        assert!((scalar_sim - simd_sim).abs() < 1e-5);

        let scalar_dist = euclidean_distance_squared_scalar(&a, &b);
        let simd_dist = euclidean_distance_squared(&a, &b);

        assert!((scalar_dist - simd_dist).abs() < 1e-4);
    }
}
