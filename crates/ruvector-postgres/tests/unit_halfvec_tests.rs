//! Unit tests for HalfVec (half-precision f16) type
//!
//! Tests half-precision vector storage and conversions

use half::f16;
use ruvector_postgres::types::HalfVec;

#[cfg(test)]
mod halfvec_tests {
    use super::*;

    // ========================================================================
    // Construction Tests
    // ========================================================================

    #[test]
    fn test_from_f32_basic() {
        let data = [1.0, 2.0, 3.0];
        let hv = HalfVec::from_f32(&data);

        assert_eq!(hv.dimensions(), 3);
    }

    #[test]
    fn test_from_f32_precision_loss() {
        // f16 has less precision than f32
        let original = [1.23456789, 9.87654321];
        let hv = HalfVec::from_f32(&original);

        let recovered = hv.to_f32();

        // Should be close but not exact due to f16 precision
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.01);
        }
    }

    #[test]
    fn test_from_f32_empty() {
        let data: [f32; 0] = [];
        let hv = HalfVec::from_f32(&data);
        assert_eq!(hv.dimensions(), 0);
    }

    #[test]
    fn test_from_f32_large() {
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let hv = HalfVec::from_f32(&data);

        assert_eq!(hv.dimensions(), size);
    }

    // ========================================================================
    // Conversion Tests
    // ========================================================================

    #[test]
    fn test_f32_roundtrip_simple() {
        let original = [1.0, 2.0, 3.0, 4.0, 5.0];
        let hv = HalfVec::from_f32(&original);
        let recovered = hv.to_f32();

        assert_eq!(recovered.len(), 5);
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.001);
        }
    }

    #[test]
    fn test_f32_roundtrip_negative() {
        let original = [-1.5, 2.3, -4.7, 0.0, -0.001];
        let hv = HalfVec::from_f32(&original);
        let recovered = hv.to_f32();

        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.01);
        }
    }

    #[test]
    fn test_f32_roundtrip_extreme_values() {
        // Test values near f16 limits
        let original = [0.00001, 100.0, -100.0, 0.5];
        let hv = HalfVec::from_f32(&original);
        let recovered = hv.to_f32();

        for (orig, rec) in original.iter().zip(recovered.iter()) {
            // Relative error for extreme values
            let rel_error = if orig.abs() > 0.0 {
                ((orig - rec) / orig).abs()
            } else {
                (orig - rec).abs()
            };
            assert!(rel_error < 0.01 || (orig - rec).abs() < 0.01);
        }
    }

    // ========================================================================
    // Memory Efficiency Tests
    // ========================================================================

    #[test]
    fn test_memory_size() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let hv = HalfVec::from_f32(&data);

        // HalfVec should use ~50% of the memory of RuVector
        // Data portion: 100 elements * 2 bytes = 200 bytes
        // Plus header (4 bytes for dims/padding)
        let data_size = hv.memory_size();
        assert!(data_size >= 200 && data_size <= 220);
    }

    #[test]
    fn test_memory_savings() {
        use ruvector_postgres::types::RuVector;

        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let rv = RuVector::from_slice(&data);
        let hv = HalfVec::from_f32(&data);

        let rv_size = rv.data_memory_size();
        let hv_size = hv.memory_size();

        // HalfVec should be approximately half the size
        // (Header is the same size, so not exactly half)
        let ratio = hv_size as f64 / rv_size as f64;
        assert!(ratio < 0.60 && ratio > 0.40);
    }

    // ========================================================================
    // Accuracy Tests
    // ========================================================================

    #[test]
    fn test_integer_values_exact() {
        // Small integers should be represented exactly in f16
        let integers = [0.0, 1.0, 2.0, 3.0, 10.0, 100.0, -50.0];
        let hv = HalfVec::from_f32(&integers);
        let recovered = hv.to_f32();

        for (orig, rec) in integers.iter().zip(recovered.iter()) {
            if orig.abs() < 1000.0 {
                assert_eq!(*orig, *rec, "Integer {} should be exact", orig);
            }
        }
    }

    #[test]
    fn test_zero_preservation() {
        let zeros = [0.0, -0.0, 0.0, -0.0];
        let hv = HalfVec::from_f32(&zeros);
        let recovered = hv.to_f32();

        for rec in recovered.iter() {
            assert_eq!(*rec, 0.0);
        }
    }

    #[test]
    fn test_sign_preservation() {
        let values = [1.0, -1.0, 2.5, -2.5, 0.1, -0.1];
        let hv = HalfVec::from_f32(&values);
        let recovered = hv.to_f32();

        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert_eq!(
                orig.signum(),
                rec.signum(),
                "Sign should be preserved for {}",
                orig
            );
        }
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_single_element() {
        let data = [42.0];
        let hv = HalfVec::from_f32(&data);

        assert_eq!(hv.dimensions(), 1);
        let recovered = hv.to_f32();
        assert_eq!(recovered.len(), 1);
        assert!((recovered[0] - 42.0).abs() < 0.1);
    }

    #[test]
    fn test_power_of_two_sizes() {
        // Test sizes that align with SIMD boundaries
        for size in [8, 16, 32, 64, 128, 256, 512, 1024] {
            let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
            let hv = HalfVec::from_f32(&data);

            assert_eq!(hv.dimensions(), size);
            let recovered = hv.to_f32();
            assert_eq!(recovered.len(), size);
        }
    }

    #[test]
    fn test_non_power_of_two_sizes() {
        // Test sizes that don't align with SIMD boundaries
        for size in [7, 15, 31, 63, 127, 255] {
            let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
            let hv = HalfVec::from_f32(&data);

            assert_eq!(hv.dimensions(), size);
        }
    }

    // ========================================================================
    // Numerical Range Tests
    // ========================================================================

    #[test]
    fn test_small_values() {
        // Test values near f16's minimum normal value
        let small = [0.0001, 0.001, 0.01, 0.1];
        let hv = HalfVec::from_f32(&small);
        let recovered = hv.to_f32();

        for (orig, rec) in small.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.001 || (orig - rec) / orig < 0.1);
        }
    }

    #[test]
    fn test_large_values() {
        // Test values approaching f16's maximum
        let large = [100.0, 500.0, 1000.0];
        let hv = HalfVec::from_f32(&large);
        let recovered = hv.to_f32();

        for (orig, rec) in large.iter().zip(recovered.iter()) {
            let rel_error = ((orig - rec) / orig).abs();
            assert!(
                rel_error < 0.01,
                "Large value {} -> {}, error {}",
                orig,
                rec,
                rel_error
            );
        }
    }

    #[test]
    fn test_mixed_magnitude() {
        // Test vectors with widely varying magnitudes
        let mixed = [0.001, 1.0, 100.0, 0.01, 10.0];
        let hv = HalfVec::from_f32(&mixed);
        let recovered = hv.to_f32();

        for (orig, rec) in mixed.iter().zip(recovered.iter()) {
            let abs_error = (orig - rec).abs();
            let rel_error = if orig.abs() > 0.0 {
                abs_error / orig.abs()
            } else {
                abs_error
            };
            assert!(rel_error < 0.05 || abs_error < 0.01);
        }
    }

    // ========================================================================
    // Clone and Equality Tests
    // ========================================================================

    #[test]
    fn test_clone() {
        let data = [1.0, 2.0, 3.0];
        let hv1 = HalfVec::from_f32(&data);
        let hv2 = hv1; // Copy (since HalfVec is Copy)

        assert_eq!(hv1.dimensions(), hv2.dimensions());
        assert_eq!(hv1.to_f32(), hv2.to_f32());
    }

    // ========================================================================
    // Stress Tests
    // ========================================================================

    #[test]
    fn test_large_batch_conversion() {
        let num_vectors = 1000;
        let dim = 128;

        for i in 0..num_vectors {
            let data: Vec<f32> = (0..dim).map(|j| ((i * dim + j) as f32) * 0.001).collect();

            let hv = HalfVec::from_f32(&data);
            assert_eq!(hv.dimensions(), dim);

            let recovered = hv.to_f32();
            assert_eq!(recovered.len(), dim);
        }
    }

    #[test]
    fn test_alternating_pattern() {
        let size = 100;
        let data: Vec<f32> = (0..size)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let hv = HalfVec::from_f32(&data);
        let recovered = hv.to_f32();

        for (i, rec) in recovered.iter().enumerate() {
            let expected = if i % 2 == 0 { 1.0 } else { -1.0 };
            assert_eq!(*rec, expected);
        }
    }
}
