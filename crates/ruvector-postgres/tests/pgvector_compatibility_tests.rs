//! Regression tests for pgvector compatibility
//!
//! These tests ensure that ruvector produces the same results as pgvector
//! for identical operations, ensuring drop-in replacement compatibility.
//!
//! Run with: `cargo pgrx test`

#![cfg(feature = "pg_test")]

#[pgrx::pg_schema]
mod pgvector_compat_tests {
    use pgrx::prelude::*;
    use ruvector_postgres::operators::*;
    use ruvector_postgres::types::RuVector;

    // ========================================================================
    // Distance Calculation Compatibility
    // ========================================================================

    /// Test vectors known from pgvector documentation
    #[pg_test]
    fn test_pgvector_example_l2() {
        // Example from pgvector docs: SELECT '[1,2,3]' <-> '[3,2,1]';
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[3.0, 2.0, 1.0]);

        let dist = ruvector_l2_distance(a, b);

        // Expected: sqrt((3-1)^2 + (2-2)^2 + (1-3)^2) = sqrt(8) ≈ 2.828
        let expected = 2.828427;
        assert!(
            (dist - expected).abs() < 0.001,
            "L2 distance doesn't match pgvector: expected {}, got {}",
            expected,
            dist
        );
    }

    #[pg_test]
    fn test_pgvector_example_cosine() {
        // Example: SELECT '[1,2,3]' <=> '[3,2,1]';
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[3.0, 2.0, 1.0]);

        let dist = ruvector_cosine_distance(a, b);

        // 1 - (1*3 + 2*2 + 3*1) / (sqrt(14) * sqrt(14))
        // = 1 - 10/14 ≈ 0.2857
        let expected = 0.2857;
        assert!((dist - expected).abs() < 0.01);
    }

    #[pg_test]
    fn test_pgvector_example_inner_product() {
        // Example: SELECT '[1,2,3]' <#> '[3,2,1]';
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[3.0, 2.0, 1.0]);

        let dist = ruvector_ip_distance(a, b);

        // -(1*3 + 2*2 + 3*1) = -10
        let expected = -10.0;
        assert!((dist - expected).abs() < 0.001);
    }

    // ========================================================================
    // Operator Symbol Compatibility
    // ========================================================================

    #[pg_test]
    fn test_operator_symbols_match_pgvector() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[4.0, 5.0, 6.0]);

        // <-> for L2
        let l2 = ruvector_l2_dist_op(a.clone(), b.clone());
        assert!(l2 > 0.0);

        // <=> for cosine
        let cosine = ruvector_cosine_dist_op(a.clone(), b.clone());
        assert!(cosine >= 0.0 && cosine <= 2.0);

        // <#> for inner product
        let ip = ruvector_neg_ip_op(a.clone(), b.clone());
        assert!(ip.is_finite());
    }

    // ========================================================================
    // Array Conversion Compatibility
    // ========================================================================

    #[pg_test]
    fn test_array_to_vector_conversion() {
        use ruvector_postgres::types::vector::{ruvector_from_array, ruvector_to_array};

        let arr = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vec = ruvector_from_array(arr.clone());

        assert_eq!(vec.dimensions(), 5);

        let back = ruvector_to_array(vec);
        assert_eq!(back, arr);
    }

    #[pg_test]
    fn test_vector_dimensions_function() {
        use ruvector_postgres::types::vector::ruvector_dims;

        let v = RuVector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ruvector_dims(v), 4);
    }

    #[pg_test]
    fn test_vector_norm_function() {
        use ruvector_postgres::types::vector::ruvector_norm;

        let v = RuVector::from_slice(&[3.0, 4.0]);
        let norm = ruvector_norm(v);
        assert!((norm - 5.0).abs() < 1e-5);
    }

    #[pg_test]
    fn test_vector_normalize_function() {
        use ruvector_postgres::types::vector::{ruvector_norm, ruvector_normalize};

        let v = RuVector::from_slice(&[3.0, 4.0, 0.0]);
        let normalized = ruvector_normalize(v);
        let norm = ruvector_norm(normalized);

        assert!((norm - 1.0).abs() < 1e-5);
    }

    // ========================================================================
    // Index Behavior Compatibility (Nearest Neighbor)
    // ========================================================================

    #[pg_test]
    fn test_nearest_neighbor_order_l2() {
        // Test that ordering by L2 distance works as expected
        let query = RuVector::from_slice(&[1.0, 1.0, 1.0]);

        let candidates = vec![
            RuVector::from_slice(&[1.0, 1.0, 1.0]), // dist = 0
            RuVector::from_slice(&[2.0, 2.0, 2.0]), // dist = sqrt(3) ≈ 1.73
            RuVector::from_slice(&[0.0, 0.0, 0.0]), // dist = sqrt(3) ≈ 1.73
            RuVector::from_slice(&[5.0, 5.0, 5.0]), // dist = sqrt(48) ≈ 6.93
        ];

        let mut distances: Vec<_> = candidates
            .iter()
            .map(|c| ruvector_l2_distance(query.clone(), c.clone()))
            .collect();

        // Check first one is closest (distance 0)
        assert!(distances[0] < distances[1]);
        assert!(distances[0] < distances[2]);
        assert!(distances[0] < distances[3]);

        // Check last one is farthest
        assert!(distances[3] > distances[0]);
        assert!(distances[3] > distances[1]);
        assert!(distances[3] > distances[2]);
    }

    #[pg_test]
    fn test_nearest_neighbor_order_cosine() {
        let query = RuVector::from_slice(&[1.0, 0.0, 0.0]);

        let candidates = vec![
            RuVector::from_slice(&[1.0, 0.0, 0.0]), // same direction, dist = 0
            RuVector::from_slice(&[0.5, 0.5, 0.0]), // 45 degrees
            RuVector::from_slice(&[0.0, 1.0, 0.0]), // 90 degrees, dist = 1
            RuVector::from_slice(&[-1.0, 0.0, 0.0]), // opposite, dist = 2
        ];

        let distances: Vec<_> = candidates
            .iter()
            .map(|c| ruvector_cosine_distance(query.clone(), c.clone()))
            .collect();

        // Check ordering: same direction < angled < orthogonal < opposite
        assert!(distances[0] < distances[1]);
        assert!(distances[1] < distances[2]);
        assert!(distances[2] < distances[3]);
    }

    // ========================================================================
    // Precision Compatibility Tests
    // ========================================================================

    #[pg_test]
    fn test_precision_matches_pgvector() {
        // pgvector uses f32, so we should match that precision
        let a = RuVector::from_slice(&[0.123456789, 0.987654321]);
        let b = RuVector::from_slice(&[0.111111111, 0.999999999]);

        let dist = ruvector_l2_distance(a, b);

        // Should be computed as f32, not f64
        assert!(dist.is_finite());

        // Verify it's actually using f32 precision
        let a_f32 = [0.123456789f32, 0.987654321f32];
        let b_f32 = [0.111111111f32, 0.999999999f32];
        let expected = ((a_f32[0] - b_f32[0]).powi(2) + (a_f32[1] - b_f32[1]).powi(2)).sqrt();

        assert!((dist - expected).abs() < 1e-6);
    }

    // ========================================================================
    // Edge Cases pgvector Handles
    // ========================================================================

    #[pg_test]
    fn test_single_dimension_vector() {
        let a = RuVector::from_slice(&[5.0]);
        let b = RuVector::from_slice(&[3.0]);

        let dist = ruvector_l2_distance(a, b);
        assert!((dist - 2.0).abs() < 1e-5);
    }

    #[pg_test]
    fn test_high_dimensional_vector() {
        // pgvector supports up to 16000 dimensions
        let size = 2000;
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = vec![0.0; size];

        let va = RuVector::from_slice(&a);
        let vb = RuVector::from_slice(&b);

        let dist = ruvector_l2_distance(va, vb);
        assert!(dist > 0.0 && dist.is_finite());
    }

    #[pg_test]
    fn test_vector_with_zeros() {
        let a = RuVector::from_slice(&[1.0, 0.0, 2.0, 0.0, 3.0]);
        let b = RuVector::from_slice(&[0.0, 1.0, 0.0, 2.0, 0.0]);

        let dist = ruvector_l2_distance(a, b);
        // sqrt(1 + 1 + 4 + 4 + 9) = sqrt(19) ≈ 4.359
        assert!((dist - 4.359).abs() < 0.01);
    }

    // ========================================================================
    // Text Format Compatibility
    // ========================================================================

    #[pg_test]
    fn test_text_format_parsing() {
        // pgvector accepts: [1,2,3] and [1.0, 2.0, 3.0]
        let v1: RuVector = "[1,2,3]".parse().unwrap();
        let v2: RuVector = "[1.0, 2.0, 3.0]".parse().unwrap();
        let v3: RuVector = "[1.0,2.0,3.0]".parse().unwrap();

        assert_eq!(v1, v2);
        assert_eq!(v2, v3);
        assert_eq!(v1.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[pg_test]
    fn test_text_format_whitespace() {
        // pgvector is flexible with whitespace
        let v1: RuVector = "[ 1 , 2 , 3 ]".parse().unwrap();
        let v2: RuVector = "[1,2,3]".parse().unwrap();

        assert_eq!(v1, v2);
    }

    // ========================================================================
    // Known pgvector Results (Regression Tests)
    // ========================================================================

    #[pg_test]
    fn test_known_result_1() {
        // From pgvector test suite
        let a = RuVector::from_slice(&[1.0, 1.0, 1.0]);
        let b = RuVector::from_slice(&[2.0, 2.0, 2.0]);

        let dist = ruvector_l2_distance(a, b);
        assert!((dist - 1.732).abs() < 0.01); // sqrt(3)
    }

    #[pg_test]
    fn test_known_result_2() {
        // Unit vectors at different angles
        let a = RuVector::from_slice(&[1.0, 0.0]);
        let b = RuVector::from_slice(&[0.0, 1.0]);

        let cosine_dist = ruvector_cosine_distance(a.clone(), b.clone());
        assert!((cosine_dist - 1.0).abs() < 0.01);

        let l2_dist = ruvector_l2_distance(a, b);
        assert!((l2_dist - 1.414).abs() < 0.01); // sqrt(2)
    }

    #[pg_test]
    fn test_known_result_3() {
        // Negative values
        let a = RuVector::from_slice(&[-1.0, -1.0, -1.0]);
        let b = RuVector::from_slice(&[1.0, 1.0, 1.0]);

        let dist = ruvector_l2_distance(a, b);
        assert!((dist - 3.464).abs() < 0.01); // sqrt(12)
    }
}
