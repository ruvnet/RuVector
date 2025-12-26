//! SIMD consistency tests - verify SIMD and scalar implementations match
//!
//! These tests ensure that optimized SIMD code paths produce the same results
//! as the scalar fallback implementations.

use ruvector_postgres::distance::{scalar, simd};

#[cfg(test)]
mod simd_consistency {
    use super::*;

    const EPSILON: f32 = 1e-5;

    // ========================================================================
    // Euclidean Distance Consistency
    // ========================================================================

    #[test]
    fn test_euclidean_scalar_vs_simd_small() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let scalar_result = scalar::euclidean_distance(&a, &b);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let simd_result = simd::euclidean_distance_avx2_wrapper(&a, &b);
                assert!(
                    (scalar_result - simd_result).abs() < EPSILON,
                    "AVX2: scalar={}, simd={}",
                    scalar_result,
                    simd_result
                );
            }

            if is_x86_feature_detected!("avx512f") {
                let simd_result = simd::euclidean_distance_avx512_wrapper(&a, &b);
                assert!(
                    (scalar_result - simd_result).abs() < EPSILON,
                    "AVX512: scalar={}, simd={}",
                    scalar_result,
                    simd_result
                );
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let simd_result = simd::euclidean_distance_neon_wrapper(&a, &b);
            assert!((scalar_result - simd_result).abs() < EPSILON);
        }
    }

    #[test]
    fn test_euclidean_scalar_vs_simd_various_sizes() {
        // Test different sizes to exercise SIMD remainder handling
        for size in [1, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256] {
            let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.1).collect();

            let scalar_result = scalar::euclidean_distance(&a, &b);

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    let simd_result = simd::euclidean_distance_avx2_wrapper(&a, &b);
                    assert!(
                        (scalar_result - simd_result).abs() < EPSILON,
                        "Size {}: AVX2 mismatch",
                        size
                    );
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                let simd_result = simd::euclidean_distance_neon_wrapper(&a, &b);
                assert!(
                    (scalar_result - simd_result).abs() < EPSILON,
                    "Size {}: NEON mismatch",
                    size
                );
            }
        }
    }

    #[test]
    fn test_euclidean_scalar_vs_simd_negative() {
        let a = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let scalar_result = scalar::euclidean_distance(&a, &b);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let simd_result = simd::euclidean_distance_avx2_wrapper(&a, &b);
                assert!((scalar_result - simd_result).abs() < EPSILON);
            }
        }
    }

    // ========================================================================
    // Cosine Distance Consistency
    // ========================================================================

    #[test]
    fn test_cosine_scalar_vs_simd_small() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];

        let scalar_result = scalar::cosine_distance(&a, &b);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let simd_result = simd::cosine_distance_avx2_wrapper(&a, &b);
                assert!((scalar_result - simd_result).abs() < EPSILON);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let simd_result = simd::cosine_distance_neon_wrapper(&a, &b);
            assert!((scalar_result - simd_result).abs() < EPSILON);
        }
    }

    #[test]
    fn test_cosine_scalar_vs_simd_various_sizes() {
        for size in [8, 16, 32, 64, 128, 256] {
            let a: Vec<f32> = (0..size).map(|i| (i % 10) as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| ((i + 5) % 10) as f32).collect();

            // Skip if zero vectors
            if a.iter().all(|&x| x == 0.0) || b.iter().all(|&x| x == 0.0) {
                continue;
            }

            let scalar_result = scalar::cosine_distance(&a, &b);

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    let simd_result = simd::cosine_distance_avx2_wrapper(&a, &b);
                    assert!(
                        (scalar_result - simd_result).abs() < 1e-4,
                        "Size {}: scalar={}, simd={}",
                        size,
                        scalar_result,
                        simd_result
                    );
                }
            }
        }
    }

    #[test]
    fn test_cosine_scalar_vs_simd_normalized() {
        // Test with pre-normalized vectors
        let a = vec![0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let scalar_result = scalar::cosine_distance(&a, &b);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let simd_result = simd::cosine_distance_avx2_wrapper(&a, &b);
                assert!((scalar_result - simd_result).abs() < EPSILON);
            }
        }
    }

    // ========================================================================
    // Inner Product Consistency
    // ========================================================================

    #[test]
    fn test_inner_product_scalar_vs_simd_small() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let scalar_result = scalar::inner_product_distance(&a, &b);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let simd_result = simd::inner_product_avx2_wrapper(&a, &b);
                assert!((scalar_result - simd_result).abs() < EPSILON);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let simd_result = simd::inner_product_neon_wrapper(&a, &b);
            assert!((scalar_result - simd_result).abs() < EPSILON);
        }
    }

    #[test]
    fn test_inner_product_scalar_vs_simd_various_sizes() {
        for size in [4, 8, 16, 32, 64, 128] {
            let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.1).collect();

            let scalar_result = scalar::inner_product_distance(&a, &b);

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    let simd_result = simd::inner_product_avx2_wrapper(&a, &b);
                    assert!(
                        (scalar_result - simd_result).abs() < 1e-4,
                        "Size {}: mismatch",
                        size
                    );
                }
            }
        }
    }

    // ========================================================================
    // Manhattan Distance Consistency
    // ========================================================================

    #[test]
    fn test_manhattan_scalar_vs_simd_small() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let scalar_result = scalar::manhattan_distance(&a, &b);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let simd_result = simd::manhattan_distance_avx2_wrapper(&a, &b);
                assert!((scalar_result - simd_result).abs() < EPSILON);
            }
        }
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_zero_vectors() {
        let a = vec![0.0; 32];
        let b = vec![0.0; 32];

        let scalar_euclidean = scalar::euclidean_distance(&a, &b);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let simd_euclidean = simd::euclidean_distance_avx2_wrapper(&a, &b);
                assert!((scalar_euclidean - simd_euclidean).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_small_values() {
        let a: Vec<f32> = (0..64).map(|_| 1e-6).collect();
        let b: Vec<f32> = (0..64).map(|_| 1e-6).collect();

        let scalar_result = scalar::euclidean_distance(&a, &b);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let simd_result = simd::euclidean_distance_avx2_wrapper(&a, &b);
                assert!((scalar_result - simd_result).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_large_values() {
        let a: Vec<f32> = (0..64).map(|_| 1e6).collect();
        let b: Vec<f32> = (0..64).map(|_| 9e5).collect();

        let scalar_result = scalar::euclidean_distance(&a, &b);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let simd_result = simd::euclidean_distance_avx2_wrapper(&a, &b);
                // Allow larger epsilon for large values
                assert!((scalar_result - simd_result).abs() < 1.0);
            }
        }
    }

    // ========================================================================
    // Random Data Tests
    // ========================================================================

    #[test]
    fn test_random_data_consistency() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let size = rng.gen_range(8..256);
            let a: Vec<f32> = (0..size).map(|_| rng.gen_range(-100.0..100.0)).collect();
            let b: Vec<f32> = (0..size).map(|_| rng.gen_range(-100.0..100.0)).collect();

            let scalar_euclidean = scalar::euclidean_distance(&a, &b);
            let scalar_manhattan = scalar::manhattan_distance(&a, &b);

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    let simd_euclidean = simd::euclidean_distance_avx2_wrapper(&a, &b);
                    let simd_manhattan = simd::manhattan_distance_avx2_wrapper(&a, &b);

                    assert!(
                        (scalar_euclidean - simd_euclidean).abs() < 1e-3,
                        "Euclidean mismatch at size {}",
                        size
                    );
                    assert!(
                        (scalar_manhattan - simd_manhattan).abs() < 1e-3,
                        "Manhattan mismatch at size {}",
                        size
                    );
                }
            }
        }
    }
}
