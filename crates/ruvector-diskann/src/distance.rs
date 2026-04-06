//! SIMD-friendly distance computations

/// L2 (squared Euclidean) distance — no sqrt for speed
/// Uses 4 accumulators to exploit ILP and auto-vectorization
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut i = 0;

    // 16-wide loop with 4 accumulators for maximum ILP
    while i + 16 <= len {
        for j in (0..4).step_by(1) {
            let off = i + j * 4;
            let d0 = a[off] - b[off];
            let d1 = a[off + 1] - b[off + 1];
            let d2 = a[off + 2] - b[off + 2];
            let d3 = a[off + 3] - b[off + 3];
            s0 += d0 * d0;
            s1 += d1 * d1;
            s2 += d2 * d2;
            s3 += d3 * d3;
        }
        i += 16;
    }
    // Handle remainder
    while i < len {
        let d = a[i] - b[i];
        s0 += d * d;
        i += 1;
    }
    s0 + s1 + s2 + s3
}

/// Inner product distance (negated for min-heap compatibility)
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    let len = a.len();
    let mut i = 0;

    while i + 8 <= len {
        let mut block = 0.0f32;
        for j in 0..8 {
            block += a[i + j] * b[i + j];
        }
        sum += block;
        i += 8;
    }
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    -sum // negate so smaller = more similar
}

/// Asymmetric distance using PQ lookup tables
/// table[subspace][code] = precomputed distance from query subvector to centroid
#[inline]
pub fn pq_asymmetric_distance(codes: &[u8], table: &[Vec<f32>]) -> f32 {
    let mut dist = 0.0f32;
    for (i, &code) in codes.iter().enumerate() {
        dist += table[i][code as usize];
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_squared() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((l2_squared(&a, &b) - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_identical() {
        let a = vec![1.0; 128];
        assert!(l2_squared(&a, &a) < 1e-10);
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // dot = 4 + 10 + 18 = 32, negated = -32
        assert!((inner_product(&a, &b) - (-32.0)).abs() < 1e-6);
    }
}
