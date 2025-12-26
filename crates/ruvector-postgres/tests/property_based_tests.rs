//! Property-based tests using proptest
//!
//! These tests generate random inputs and verify mathematical properties
//! that should always hold true, helping catch edge cases and numerical issues.

use proptest::prelude::*;
use ruvector_postgres::distance::{
    cosine_distance, euclidean_distance, inner_product_distance, manhattan_distance,
};
use ruvector_postgres::types::RuVector;

// ============================================================================
// Property: Distance Functions
// ============================================================================

proptest! {
    /// L2 distance should always be non-negative
    #[test]
    fn prop_l2_distance_non_negative(
        v1 in prop::collection::vec(-1000.0f32..1000.0f32, 1..100),
        v2 in prop::collection::vec(-1000.0f32..1000.0f32, 1..100)
    ) {
        if v1.len() == v2.len() {
            let dist = euclidean_distance(&v1, &v2);
            prop_assert!(dist >= 0.0, "L2 distance must be non-negative, got {}", dist);
            prop_assert!(dist.is_finite(), "L2 distance must be finite");
        }
    }

    /// L2 distance is symmetric: d(a,b) = d(b,a)
    #[test]
    fn prop_l2_distance_symmetric(
        v1 in prop::collection::vec(-100.0f32..100.0f32, 1..50),
        v2 in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        if v1.len() == v2.len() {
            let d1 = euclidean_distance(&v1, &v2);
            let d2 = euclidean_distance(&v2, &v1);
            prop_assert!((d1 - d2).abs() < 1e-5, "L2 distance must be symmetric");
        }
    }

    /// L2 distance from vector to itself is zero
    #[test]
    fn prop_l2_distance_self_is_zero(
        v in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        let dist = euclidean_distance(&v, &v);
        prop_assert!(dist.abs() < 1e-5, "Distance to self must be ~0, got {}", dist);
    }

    /// Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    #[test]
    fn prop_l2_triangle_inequality(
        v1 in prop::collection::vec(-100.0f32..100.0f32, 1..30),
        v2 in prop::collection::vec(-100.0f32..100.0f32, 1..30),
        v3 in prop::collection::vec(-100.0f32..100.0f32, 1..30)
    ) {
        if v1.len() == v2.len() && v2.len() == v3.len() {
            let d_ac = euclidean_distance(&v1, &v3);
            let d_ab = euclidean_distance(&v1, &v2);
            let d_bc = euclidean_distance(&v2, &v3);

            prop_assert!(
                d_ac <= d_ab + d_bc + 1e-4, // Small epsilon for floating point
                "Triangle inequality violated: {} > {} + {}", d_ac, d_ab, d_bc
            );
        }
    }

    /// Manhattan distance should always be non-negative
    #[test]
    fn prop_l1_distance_non_negative(
        v1 in prop::collection::vec(-1000.0f32..1000.0f32, 1..100),
        v2 in prop::collection::vec(-1000.0f32..1000.0f32, 1..100)
    ) {
        if v1.len() == v2.len() {
            let dist = manhattan_distance(&v1, &v2);
            prop_assert!(dist >= 0.0, "L1 distance must be non-negative");
            prop_assert!(dist.is_finite(), "L1 distance must be finite");
        }
    }

    /// Manhattan distance is symmetric
    #[test]
    fn prop_l1_distance_symmetric(
        v1 in prop::collection::vec(-100.0f32..100.0f32, 1..50),
        v2 in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        if v1.len() == v2.len() {
            let d1 = manhattan_distance(&v1, &v2);
            let d2 = manhattan_distance(&v2, &v1);
            prop_assert!((d1 - d2).abs() < 1e-5);
        }
    }

    /// Cosine distance should be in range [0, 2]
    #[test]
    fn prop_cosine_distance_range(
        v1 in prop::collection::vec(-100.0f32..100.0f32, 1..50),
        v2 in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        if v1.len() == v2.len() && v1.iter().any(|&x| x != 0.0) && v2.iter().any(|&x| x != 0.0) {
            let dist = cosine_distance(&v1, &v2);
            if dist.is_finite() {
                prop_assert!(dist >= -0.001, "Cosine distance should be >= 0, got {}", dist);
                prop_assert!(dist <= 2.001, "Cosine distance should be <= 2, got {}", dist);
            }
        }
    }

    /// Cosine distance is symmetric
    #[test]
    fn prop_cosine_distance_symmetric(
        v1 in prop::collection::vec(-100.0f32..100.0f32, 1..50),
        v2 in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        if v1.len() == v2.len() && v1.iter().any(|&x| x != 0.0) && v2.iter().any(|&x| x != 0.0) {
            let d1 = cosine_distance(&v1, &v2);
            let d2 = cosine_distance(&v2, &v1);
            if d1.is_finite() && d2.is_finite() {
                prop_assert!((d1 - d2).abs() < 1e-4);
            }
        }
    }
}

// ============================================================================
// Property: Vector Operations
// ============================================================================

proptest! {
    /// Normalization produces unit vectors
    #[test]
    fn prop_normalize_produces_unit_vector(
        data in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        // Skip zero vectors
        if data.iter().any(|&x| x != 0.0) {
            let v = RuVector::from_slice(&data);
            let normalized = v.normalize();
            let norm = normalized.norm();
            prop_assert!(
                (norm - 1.0).abs() < 1e-5,
                "Normalized vector should have norm ~1.0, got {}",
                norm
            );
        }
    }

    /// Adding zero vector doesn't change the vector
    #[test]
    fn prop_add_zero_identity(
        data in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        let v = RuVector::from_slice(&data);
        let zero = RuVector::zeros(data.len());
        let result = v.add(&zero);

        for (a, b) in data.iter().zip(result.as_slice().iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }

    /// Subtraction is inverse of addition: (a + b) - b = a
    #[test]
    fn prop_sub_inverse_of_add(
        v1 in prop::collection::vec(-100.0f32..100.0f32, 1..50),
        v2 in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        if v1.len() == v2.len() {
            let a = RuVector::from_slice(&v1);
            let b = RuVector::from_slice(&v2);

            let sum = a.add(&b);
            let result = sum.sub(&b);

            for (original, recovered) in v1.iter().zip(result.as_slice().iter()) {
                prop_assert!((original - recovered).abs() < 1e-4);
            }
        }
    }

    /// Scalar multiplication by 1 is identity
    #[test]
    fn prop_mul_scalar_identity(
        data in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        let v = RuVector::from_slice(&data);
        let result = v.mul_scalar(1.0);

        for (a, b) in data.iter().zip(result.as_slice().iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }

    /// Scalar multiplication by 0 produces zero vector
    #[test]
    fn prop_mul_scalar_zero(
        data in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        let v = RuVector::from_slice(&data);
        let result = v.mul_scalar(0.0);

        for &val in result.as_slice() {
            prop_assert_eq!(val, 0.0);
        }
    }

    /// Scalar multiplication is associative: (a * b) * c = a * (b * c)
    #[test]
    fn prop_mul_scalar_associative(
        data in prop::collection::vec(-10.0f32..10.0f32, 1..30),
        scalar1 in -10.0f32..10.0f32,
        scalar2 in -10.0f32..10.0f32
    ) {
        let v = RuVector::from_slice(&data);

        let r1 = v.mul_scalar(scalar1).mul_scalar(scalar2);
        let r2 = v.mul_scalar(scalar1 * scalar2);

        for (a, b) in r1.as_slice().iter().zip(r2.as_slice().iter()) {
            prop_assert!((a - b).abs() < 1e-4);
        }
    }

    /// Dot product is commutative: a · b = b · a
    #[test]
    fn prop_dot_commutative(
        v1 in prop::collection::vec(-100.0f32..100.0f32, 1..50),
        v2 in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        if v1.len() == v2.len() {
            let a = RuVector::from_slice(&v1);
            let b = RuVector::from_slice(&v2);

            let dot1 = a.dot(&b);
            let dot2 = b.dot(&a);

            prop_assert!((dot1 - dot2).abs() < 1e-4);
        }
    }

    /// Dot product with zero vector is zero
    #[test]
    fn prop_dot_with_zero(
        data in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        let v = RuVector::from_slice(&data);
        let zero = RuVector::zeros(data.len());

        let result = v.dot(&zero);
        prop_assert!(result.abs() < 1e-6);
    }

    /// Norm squared equals dot product with self
    #[test]
    fn prop_norm_squared_equals_self_dot(
        data in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        let v = RuVector::from_slice(&data);
        let norm_squared = v.norm() * v.norm();
        let dot_self = v.dot(&v);

        prop_assert!((norm_squared - dot_self).abs() < 1e-3);
    }
}

// ============================================================================
// Property: Serialization (Varlena Round-trip)
// NOTE: prop_varlena_roundtrip removed - requires PostgreSQL runtime (pgrx)
// Use `cargo pgrx test` for varlena property tests
// ============================================================================

proptest! {
    /// String parsing and display round-trip (for reasonable values)
    #[test]
    fn prop_string_roundtrip(
        data in prop::collection::vec(-1000.0f32..1000.0f32, 1..20)
    ) {
        let v1 = RuVector::from_slice(&data);
        let s = v1.to_string();

        if let Ok(v2) = s.parse::<RuVector>() {
            prop_assert_eq!(v1.dimensions(), v2.dimensions());

            for (a, b) in v1.as_slice().iter().zip(v2.as_slice().iter()) {
                // Allow some floating point precision loss in string conversion
                prop_assert!((a - b).abs() < 1e-4 || (a.abs() < 1e-6 && b.abs() < 1e-6));
            }
        }
    }
}

// ============================================================================
// Property: Numerical Stability
// ============================================================================

proptest! {
    /// Operations on very small values don't produce NaN/Inf
    #[test]
    fn prop_small_values_stable(
        data in prop::collection::vec(-1e-6f32..1e-6f32, 1..50)
    ) {
        let v = RuVector::from_slice(&data);

        let norm = v.norm();
        prop_assert!(norm.is_finite());

        // Only normalize if not too close to zero
        if data.iter().map(|x| x * x).sum::<f32>() > 1e-12 {
            let normalized = v.normalize();
            for &val in normalized.as_slice() {
                prop_assert!(val.is_finite());
            }
        }
    }

    /// Operations on large values don't overflow
    #[test]
    fn prop_large_values_no_overflow(
        data in prop::collection::vec(-1000.0f32..1000.0f32, 1..30)
    ) {
        let v1 = RuVector::from_slice(&data);
        let v2 = RuVector::from_slice(&data);

        let sum = v1.add(&v2);
        for &val in sum.as_slice() {
            prop_assert!(val.is_finite());
        }

        let diff = v1.sub(&v2);
        for &val in diff.as_slice() {
            prop_assert!(val.is_finite());
        }
    }

    /// Dot product doesn't overflow with reasonable inputs
    #[test]
    fn prop_dot_no_overflow(
        v1 in prop::collection::vec(-100.0f32..100.0f32, 1..100),
        v2 in prop::collection::vec(-100.0f32..100.0f32, 1..100)
    ) {
        if v1.len() == v2.len() {
            let a = RuVector::from_slice(&v1);
            let b = RuVector::from_slice(&v2);
            let dot = a.dot(&b);
            prop_assert!(dot.is_finite());
        }
    }
}

// ============================================================================
// Property: Edge Cases
// ============================================================================

proptest! {
    /// Single-element vectors work correctly
    #[test]
    fn prop_single_element_vector(
        val in -1000.0f32..1000.0f32
    ) {
        let v = RuVector::from_slice(&[val]);
        prop_assert_eq!(v.dimensions(), 1);
        prop_assert_eq!(v.as_slice()[0], val);

        let norm = v.norm();
        prop_assert!((norm - val.abs()).abs() < 1e-5);
    }

    /// Empty vectors handle operations gracefully
    #[test]
    fn prop_empty_vector_operations(_seed in 0u32..1000) {
        let v = RuVector::from_slice(&[]);

        prop_assert_eq!(v.dimensions(), 0);
        prop_assert_eq!(v.norm(), 0.0);

        let normalized = v.normalize();
        prop_assert_eq!(normalized.dimensions(), 0);
    }
}
