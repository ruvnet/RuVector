//! SIMD-optimized distance metrics
//! Uses SimSIMD when available (native), falls back to pure Rust for WASM

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
    #[cfg(all(feature = "simd", not(target_arch = "wasm32")))]
    {
        (simsimd::SpatialSimilarity::sqeuclidean(a, b)
            .expect("SimSIMD euclidean failed")
            .sqrt()) as f32
    }
    #[cfg(any(not(feature = "simd"), target_arch = "wasm32"))]
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
    #[cfg(all(feature = "simd", not(target_arch = "wasm32")))]
    {
        simsimd::SpatialSimilarity::cosine(a, b).expect("SimSIMD cosine failed") as f32
    }
    #[cfg(any(not(feature = "simd"), target_arch = "wasm32"))]
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
    #[cfg(all(feature = "simd", not(target_arch = "wasm32")))]
    {
        let dot = simsimd::SpatialSimilarity::dot(a, b).expect("SimSIMD dot product failed");
        (-dot) as f32
    }
    #[cfg(any(not(feature = "simd"), target_arch = "wasm32"))]
    {
        // Pure Rust fallback for WASM
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        -dot
    }
}

/// Manhattan (L1) distance — delegates to SIMD when available
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    crate::simd_intrinsics::manhattan_distance_simd(a, b)
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

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = distance(&a, &b, DistanceMetric::Euclidean);
        assert!(result.is_err());
    }
}
