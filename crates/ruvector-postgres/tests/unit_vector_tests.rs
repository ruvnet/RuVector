//! Comprehensive unit tests for RuVector type
//!
//! Tests cover:
//! - Vector creation and initialization
//! - Serialization/deserialization (varlena roundtrips)
//! - Vector operations (arithmetic, normalization)
//! - Distance calculations
//! - Edge cases and error conditions
//! - Memory layout and alignment

use ruvector_postgres::types::RuVector;

#[cfg(test)]
mod ruvector_unit_tests {
    use super::*;

    // ========================================================================
    // Construction and Initialization Tests
    // ========================================================================

    #[test]
    fn test_from_slice_basic() {
        let v = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.dimensions(), 3);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_slice_empty() {
        let v = RuVector::from_slice(&[]);
        assert_eq!(v.dimensions(), 0);
        let empty: &[f32] = &[];
        assert_eq!(v.as_slice(), empty);
    }

    #[test]
    fn test_from_slice_single_element() {
        let v = RuVector::from_slice(&[42.0]);
        assert_eq!(v.dimensions(), 1);
        assert_eq!(v.as_slice(), &[42.0]);
    }

    #[test]
    fn test_zeros() {
        let v = RuVector::zeros(5);
        assert_eq!(v.dimensions(), 5);
        assert_eq!(v.as_slice(), &[0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_zeros_large() {
        let v = RuVector::zeros(1000);
        assert_eq!(v.dimensions(), 1000);
        assert!(v.as_slice().iter().all(|&x| x == 0.0));
    }

    // ========================================================================
    // Varlena Serialization Tests (Round-trip)
    // NOTE: Removed - requires PostgreSQL runtime (pgrx)
    // Use `cargo pgrx test` for varlena serialization tests
    // ========================================================================

    // ========================================================================
    // Vector Operations Tests
    // ========================================================================

    #[test]
    fn test_norm_basic() {
        let v = RuVector::from_slice(&[3.0, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm_zero_vector() {
        let v = RuVector::zeros(10);
        assert_eq!(v.norm(), 0.0);
    }

    #[test]
    fn test_norm_unit_vectors() {
        let v1 = RuVector::from_slice(&[1.0, 0.0, 0.0]);
        let v2 = RuVector::from_slice(&[0.0, 1.0, 0.0]);
        let v3 = RuVector::from_slice(&[0.0, 0.0, 1.0]);

        assert!((v1.norm() - 1.0).abs() < 1e-6);
        assert!((v2.norm() - 1.0).abs() < 1e-6);
        assert!((v3.norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_basic() {
        let v = RuVector::from_slice(&[3.0, 4.0]);
        let n = v.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-6);
        assert!((n.as_slice()[0] - 0.6).abs() < 1e-6);
        assert!((n.as_slice()[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = RuVector::zeros(3);
        let n = v.normalize();
        assert_eq!(n.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_already_normalized() {
        let v = RuVector::from_slice(&[1.0, 0.0, 0.0]);
        let n = v.normalize();
        assert_eq!(n.as_slice(), &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_add_basic() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[4.0, 5.0, 6.0]);
        let c = a.add(&b);
        assert_eq!(c.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_zero() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::zeros(3);
        let c = a.add(&b);
        assert_eq!(c.as_slice(), a.as_slice());
    }

    #[test]
    fn test_sub_basic() {
        let a = RuVector::from_slice(&[5.0, 7.0, 9.0]);
        let b = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let c = a.sub(&b);
        assert_eq!(c.as_slice(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sub_self() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let c = a.sub(&a);
        assert_eq!(c.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mul_scalar_basic() {
        let v = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let scaled = v.mul_scalar(2.0);
        assert_eq!(scaled.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_mul_scalar_zero() {
        let v = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let scaled = v.mul_scalar(0.0);
        assert_eq!(scaled.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mul_scalar_negative() {
        let v = RuVector::from_slice(&[1.0, -2.0, 3.0]);
        let scaled = v.mul_scalar(-1.0);
        assert_eq!(scaled.as_slice(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_dot_product_basic() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = RuVector::from_slice(&[1.0, 0.0, 0.0]);
        let b = RuVector::from_slice(&[0.0, 1.0, 0.0]);
        assert_eq!(a.dot(&b), 0.0);
    }

    #[test]
    fn test_dot_product_zero_vector() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::zeros(3);
        assert_eq!(a.dot(&b), 0.0);
    }

    // ========================================================================
    // String Parsing Tests
    // ========================================================================

    #[test]
    fn test_parse_basic() {
        let v: RuVector = "[1.0, 2.0, 3.0]".parse().unwrap();
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_no_spaces() {
        let v: RuVector = "[1,2,3]".parse().unwrap();
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_extra_spaces() {
        let v: RuVector = "[  1.0  ,  2.0  ,  3.0  ]".parse().unwrap();
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_negative() {
        let v: RuVector = "[-1.5, 2.3, -4.7]".parse().unwrap();
        assert_eq!(v.as_slice(), &[-1.5, 2.3, -4.7]);
    }

    #[test]
    fn test_parse_scientific_notation() {
        let v: RuVector = "[1e-3, 2.5e2, -3.14e-1]".parse().unwrap();
        assert_eq!(v.dimensions(), 3);
        assert!((v.as_slice()[0] - 0.001).abs() < 1e-10);
        assert!((v.as_slice()[1] - 250.0).abs() < 1e-6);
        assert!((v.as_slice()[2] - (-0.314)).abs() < 1e-6);
    }

    #[test]
    fn test_parse_empty() {
        let v: RuVector = "[]".parse().unwrap();
        assert_eq!(v.dimensions(), 0);
    }

    #[test]
    fn test_parse_invalid_format() {
        assert!("not a vector".parse::<RuVector>().is_err());
        assert!("1,2,3".parse::<RuVector>().is_err()); // Missing brackets
        assert!("[1,2,3".parse::<RuVector>().is_err()); // Missing closing bracket
        assert!("1,2,3]".parse::<RuVector>().is_err()); // Missing opening bracket
    }

    #[test]
    fn test_parse_invalid_numbers() {
        assert!("[1.0, abc, 3.0]".parse::<RuVector>().is_err());
        assert!("[1.0, , 3.0]".parse::<RuVector>().is_err());
    }

    #[test]
    fn test_parse_nan_rejected() {
        assert!("[1.0, nan, 3.0]".parse::<RuVector>().is_err());
        assert!("[NaN, 2.0]".parse::<RuVector>().is_err());
    }

    #[test]
    fn test_parse_infinity_rejected() {
        assert!("[1.0, inf, 3.0]".parse::<RuVector>().is_err());
        assert!("[1.0, infinity, 3.0]".parse::<RuVector>().is_err());
        assert!("[-inf, 2.0]".parse::<RuVector>().is_err());
    }

    // ========================================================================
    // Display/Format Tests
    // ========================================================================

    #[test]
    fn test_display_basic() {
        let v = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.to_string(), "[1,2,3]");
    }

    #[test]
    fn test_display_decimals() {
        let v = RuVector::from_slice(&[1.5, 2.3, 3.7]);
        assert_eq!(v.to_string(), "[1.5,2.3,3.7]");
    }

    #[test]
    fn test_display_negative() {
        let v = RuVector::from_slice(&[-1.0, 2.0, -3.0]);
        assert_eq!(v.to_string(), "[-1,2,-3]");
    }

    #[test]
    fn test_display_empty() {
        let v = RuVector::from_slice(&[]);
        assert_eq!(v.to_string(), "[]");
    }

    #[test]
    fn test_display_parse_roundtrip() {
        let original = RuVector::from_slice(&[1.5, -2.3, 4.7, 0.0]);
        let s = original.to_string();
        let parsed: RuVector = s.parse().unwrap();
        assert_eq!(original, parsed);
    }

    // ========================================================================
    // Memory and Metadata Tests
    // ========================================================================

    #[test]
    fn test_data_memory_size() {
        let v = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        // Header (4 bytes: 2 dims + 2 padding) + 3 * 4 bytes = 16 bytes
        assert_eq!(v.data_memory_size(), 16);
    }

    #[test]
    fn test_data_memory_size_empty() {
        let v = RuVector::from_slice(&[]);
        // Header only: 4 bytes
        assert_eq!(v.data_memory_size(), 4);
    }

    #[test]
    fn test_data_memory_size_large() {
        let v = RuVector::zeros(1000);
        // Header (4 bytes) + 1000 * 4 bytes = 4004 bytes
        assert_eq!(v.data_memory_size(), 4004);
    }

    #[test]
    fn test_dimensions_accessor() {
        let v = RuVector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(v.dimensions(), 5);
    }

    #[test]
    fn test_into_vec() {
        let v = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let vec = v.into_vec();
        assert_eq!(vec, vec![1.0, 2.0, 3.0]);
    }

    // ========================================================================
    // Equality Tests
    // ========================================================================

    #[test]
    fn test_equality_same_vectors() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_equality_different_values() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[1.0, 2.0, 4.0]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_equality_different_dimensions() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[1.0, 2.0]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_equality_empty_vectors() {
        let a = RuVector::from_slice(&[]);
        let b = RuVector::from_slice(&[]);
        assert_eq!(a, b);
    }

    // ========================================================================
    // Clone Tests
    // ========================================================================

    #[test]
    fn test_clone_basic() {
        let v1 = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let v2 = v1.clone();
        assert_eq!(v1, v2);
        assert_eq!(v2.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_clone_independence() {
        let v1 = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let mut v2 = v1.clone();

        // Modify v2
        v2.as_mut_slice()[0] = 99.0;

        // v1 should be unchanged
        assert_eq!(v1.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(v2.as_slice(), &[99.0, 2.0, 3.0]);
    }

    // ========================================================================
    // Edge Cases and Boundary Tests
    // ========================================================================

    #[test]
    fn test_large_dimension_vector() {
        let size = 10000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let v = RuVector::from_slice(&data);
        assert_eq!(v.dimensions(), size);
        assert_eq!(v.as_slice().len(), size);
    }

    #[test]
    fn test_various_dimension_sizes() {
        // Test power-of-2 and non-power-of-2 sizes for SIMD edge cases
        for size in [
            1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024,
        ] {
            let v = RuVector::zeros(size);
            assert_eq!(v.dimensions(), size);
            assert_eq!(v.as_slice().len(), size);
        }
    }

    #[test]
    fn test_all_same_values() {
        let v = RuVector::from_slice(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        assert!(v.as_slice().iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_alternating_signs() {
        let data: Vec<f32> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let v = RuVector::from_slice(&data);
        for (i, &val) in v.as_slice().iter().enumerate() {
            let expected = if i % 2 == 0 { 1.0 } else { -1.0 };
            assert_eq!(val, expected);
        }
    }
}
