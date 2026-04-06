//! SIMD-friendly distance computations

/// L2 (squared Euclidean) distance — no sqrt for speed
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    let len = a.len();
    let mut i = 0;

    // Process 8 elements at a time for auto-vectorization
    while i + 8 <= len {
        let mut block = 0.0f32;
        for j in 0..8 {
            let d = a[i + j] - b[i + j];
            block += d * d;
        }
        sum += block;
        i += 8;
    }
    while i < len {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
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
