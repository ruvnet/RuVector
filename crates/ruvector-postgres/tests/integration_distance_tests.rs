//! pgrx integration tests for distance functions and operators
//!
//! These tests run inside a PostgreSQL instance and test the full SQL interface
//!
//! Run with: `cargo pgrx test`

#![cfg(feature = "pg_test")]

#[pgrx::pg_schema]
mod integration_tests {
    use pgrx::prelude::*;
    use ruvector_postgres::operators::*;
    use ruvector_postgres::types::RuVector;

    // ========================================================================
    // L2 Distance Tests
    // ========================================================================

    #[pg_test]
    fn test_l2_distance_basic() {
        let a = RuVector::from_slice(&[0.0, 0.0, 0.0]);
        let b = RuVector::from_slice(&[3.0, 4.0, 0.0]);
        let dist = ruvector_l2_distance(a, b);
        assert!((dist - 5.0).abs() < 1e-5, "Expected 5.0, got {}", dist);
    }

    #[pg_test]
    fn test_l2_distance_same_vector() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let dist = ruvector_l2_distance(a.clone(), a.clone());
        assert!(dist.abs() < 1e-6, "Distance to self should be ~0");
    }

    #[pg_test]
    fn test_l2_distance_negative_values() {
        let a = RuVector::from_slice(&[-1.0, -2.0, -3.0]);
        let b = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let dist = ruvector_l2_distance(a, b);
        // sqrt(4 + 16 + 36) = sqrt(56) ≈ 7.48
        assert!((dist - 7.483).abs() < 0.01);
    }

    #[pg_test]
    fn test_l2_distance_operator() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[4.0, 5.0, 6.0]);

        let func_result = ruvector_l2_distance(a.clone(), b.clone());
        let op_result = ruvector_l2_dist_op(a, b);

        assert!((func_result - op_result).abs() < 1e-10);
    }

    #[pg_test]
    fn test_l2_distance_large_vectors() {
        let size = 1024;
        let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = vec![0.0; size];

        let a = RuVector::from_slice(&a_data);
        let b = RuVector::from_slice(&b_data);

        let dist = ruvector_l2_distance(a, b);
        assert!(dist > 0.0 && dist.is_finite());
    }

    // ========================================================================
    // Cosine Distance Tests
    // ========================================================================

    #[pg_test]
    fn test_cosine_distance_same_direction() {
        let a = RuVector::from_slice(&[1.0, 0.0, 0.0]);
        let b = RuVector::from_slice(&[2.0, 0.0, 0.0]); // Same direction, different magnitude

        let dist = ruvector_cosine_distance(a, b);
        assert!(dist.abs() < 1e-5, "Same direction should have distance ~0");
    }

    #[pg_test]
    fn test_cosine_distance_opposite_direction() {
        let a = RuVector::from_slice(&[1.0, 0.0, 0.0]);
        let b = RuVector::from_slice(&[-1.0, 0.0, 0.0]);

        let dist = ruvector_cosine_distance(a, b);
        assert!(
            (dist - 2.0).abs() < 1e-5,
            "Opposite direction should have distance ~2"
        );
    }

    #[pg_test]
    fn test_cosine_distance_orthogonal() {
        let a = RuVector::from_slice(&[1.0, 0.0, 0.0]);
        let b = RuVector::from_slice(&[0.0, 1.0, 0.0]);

        let dist = ruvector_cosine_distance(a, b);
        assert!(
            (dist - 1.0).abs() < 1e-5,
            "Orthogonal vectors should have distance ~1"
        );
    }

    #[pg_test]
    fn test_cosine_distance_operator() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[4.0, 5.0, 6.0]);

        let func_result = ruvector_cosine_distance(a.clone(), b.clone());
        let op_result = ruvector_cosine_dist_op(a, b);

        assert!((func_result - op_result).abs() < 1e-10);
    }

    #[pg_test]
    fn test_cosine_distance_normalized() {
        // Pre-normalized vectors
        let a = RuVector::from_slice(&[0.6, 0.8, 0.0]);
        let b = RuVector::from_slice(&[0.0, 1.0, 0.0]);

        let dist = ruvector_cosine_distance(a, b);
        assert!(dist >= 0.0 && dist <= 2.0);
    }

    // ========================================================================
    // Inner Product Tests
    // ========================================================================

    #[pg_test]
    fn test_inner_product_basic() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[4.0, 5.0, 6.0]);

        let dist = ruvector_ip_distance(a, b);
        // -(1*4 + 2*5 + 3*6) = -32
        assert!((dist - (-32.0)).abs() < 1e-5);
    }

    #[pg_test]
    fn test_inner_product_orthogonal() {
        let a = RuVector::from_slice(&[1.0, 0.0, 0.0]);
        let b = RuVector::from_slice(&[0.0, 1.0, 0.0]);

        let dist = ruvector_ip_distance(a, b);
        assert!(dist.abs() < 1e-6, "Orthogonal vectors should have IP ~0");
    }

    #[pg_test]
    fn test_inner_product_operator() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[2.0, 3.0, 4.0]);

        let func_result = ruvector_ip_distance(a.clone(), b.clone());
        let op_result = ruvector_neg_ip_op(a, b);

        assert!((func_result - op_result).abs() < 1e-10);
    }

    #[pg_test]
    fn test_inner_product_negative() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[-1.0, -2.0, -3.0]);

        let dist = ruvector_ip_distance(a, b);
        // -(1*-1 + 2*-2 + 3*-3) = -(-14) = 14
        assert!((dist - 14.0).abs() < 1e-5);
    }

    // ========================================================================
    // L1 (Manhattan) Distance Tests
    // ========================================================================

    #[pg_test]
    fn test_l1_distance_basic() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[4.0, 6.0, 8.0]);

        let dist = ruvector_l1_distance(a, b);
        // |4-1| + |6-2| + |8-3| = 3 + 4 + 5 = 12
        assert!((dist - 12.0).abs() < 1e-5);
    }

    #[pg_test]
    fn test_l1_distance_same_vector() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);

        let dist = ruvector_l1_distance(a.clone(), a.clone());
        assert!(dist.abs() < 1e-6);
    }

    #[pg_test]
    fn test_l1_distance_negative() {
        let a = RuVector::from_slice(&[-5.0, 10.0, -3.0]);
        let b = RuVector::from_slice(&[2.0, 5.0, 1.0]);

        let dist = ruvector_l1_distance(a, b);
        // |2-(-5)| + |5-10| + |1-(-3)| = 7 + 5 + 4 = 16
        assert!((dist - 16.0).abs() < 1e-5);
    }

    #[pg_test]
    fn test_l1_distance_operator() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[3.0, 4.0, 5.0]);

        let func_result = ruvector_l1_distance(a.clone(), b.clone());
        let op_result = ruvector_l1_dist_op(a, b);

        assert!((func_result - op_result).abs() < 1e-10);
    }

    // ========================================================================
    // SIMD Consistency Tests (various vector sizes)
    // ========================================================================

    #[pg_test]
    fn test_simd_sizes_l2() {
        // Test various sizes to exercise SIMD paths and remainders
        for size in [1, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128] {
            let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

            let a = RuVector::from_slice(&a_data);
            let b = RuVector::from_slice(&b_data);

            let dist = ruvector_l2_distance(a, b);
            assert!(
                dist.is_finite() && dist > 0.0,
                "L2 distance failed for size {}",
                size
            );
        }
    }

    #[pg_test]
    fn test_simd_sizes_cosine() {
        for size in [8, 16, 32, 64, 128] {
            let a_data: Vec<f32> = (0..size).map(|i| (i % 10) as f32).collect();
            let b_data: Vec<f32> = (0..size).map(|i| ((i + 5) % 10) as f32).collect();

            let a = RuVector::from_slice(&a_data);
            let b = RuVector::from_slice(&b_data);

            let dist = ruvector_cosine_distance(a, b);
            assert!(dist.is_finite(), "Cosine distance failed for size {}", size);
        }
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[pg_test]
    #[should_panic(expected = "Cannot compute distance between vectors of different dimensions")]
    fn test_l2_dimension_mismatch() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[1.0, 2.0]);
        let _ = ruvector_l2_distance(a, b);
    }

    #[pg_test]
    #[should_panic(expected = "Cannot compute distance between vectors of different dimensions")]
    fn test_cosine_dimension_mismatch() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b = RuVector::from_slice(&[1.0, 2.0]);
        let _ = ruvector_cosine_distance(a, b);
    }

    #[pg_test]
    #[should_panic(expected = "Cannot compute distance between vectors of different dimensions")]
    fn test_ip_dimension_mismatch() {
        let a = RuVector::from_slice(&[1.0]);
        let b = RuVector::from_slice(&[1.0, 2.0]);
        let _ = ruvector_ip_distance(a, b);
    }

    // ========================================================================
    // Zero Vector Edge Cases
    // ========================================================================

    #[pg_test]
    fn test_zero_vectors_l2() {
        let a = RuVector::zeros(10);
        let b = RuVector::zeros(10);

        let dist = ruvector_l2_distance(a, b);
        assert!(dist.abs() < 1e-6);
    }

    #[pg_test]
    fn test_zero_vector_one_side_l2() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::zeros(3);

        let dist = ruvector_l2_distance(a.clone(), b);
        let expected = a.norm();
        assert!((dist - expected).abs() < 1e-5);
    }

    #[pg_test]
    fn test_zero_vectors_cosine() {
        let a = RuVector::zeros(5);
        let b = RuVector::zeros(5);

        let dist = ruvector_cosine_distance(a, b);
        // Zero vectors are undefined for cosine, should handle gracefully
        assert!(dist.is_finite() || dist.abs() <= 1.0);
    }

    // ========================================================================
    // Symmetry Tests
    // ========================================================================

    #[pg_test]
    fn test_l2_symmetry() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = RuVector::from_slice(&[5.0, 4.0, 3.0, 2.0, 1.0]);

        let d1 = ruvector_l2_distance(a.clone(), b.clone());
        let d2 = ruvector_l2_distance(b, a);

        assert!((d1 - d2).abs() < 1e-6, "L2 distance should be symmetric");
    }

    #[pg_test]
    fn test_cosine_symmetry() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b = RuVector::from_slice(&[4.0, 3.0, 2.0, 1.0]);

        let d1 = ruvector_cosine_distance(a.clone(), b.clone());
        let d2 = ruvector_cosine_distance(b, a);

        assert!(
            (d1 - d2).abs() < 1e-6,
            "Cosine distance should be symmetric"
        );
    }

    #[pg_test]
    fn test_l1_symmetry() {
        let a = RuVector::from_slice(&[10.0, 20.0, 30.0]);
        let b = RuVector::from_slice(&[5.0, 15.0, 25.0]);

        let d1 = ruvector_l1_distance(a.clone(), b.clone());
        let d2 = ruvector_l1_distance(b, a);

        assert!((d1 - d2).abs() < 1e-6, "L1 distance should be symmetric");
    }
}
