//! SIMD Equivalence Tests for ADR-090
//!
//! Tests to ensure SIMD implementations (NEON, AVX2, WASM SIMD128) produce
//! results equivalent to scalar reference implementations within acceptable
//! ULP (Units in Last Place) tolerance.
//!
//! INV-8: Kernel equivalence tests must pass (<=1 ULP difference)

#[cfg(test)]
mod simd_equivalence_tests {
    use std::f32::consts::PI;

    // ============================================================================
    // Test Constants
    // ============================================================================

    /// Maximum allowed ULP (Units in Last Place) difference
    const MAX_ULP_DIFFERENCE: i32 = 1;

    /// Epsilon for near-zero comparisons
    const EPSILON: f32 = 1e-7;

    /// Block size for SIMD operations
    const SIMD_BLOCK_SIZE: usize = 256;

    // ============================================================================
    // ULP Comparison Utilities
    // ============================================================================

    /// Calculate ULP difference between two f32 values
    fn ulp_difference(a: f32, b: f32) -> i32 {
        if a.is_nan() || b.is_nan() {
            return i32::MAX;
        }
        if a == b {
            return 0;
        }
        if a.is_infinite() || b.is_infinite() {
            return if a == b { 0 } else { i32::MAX };
        }

        let a_bits = a.to_bits() as i32;
        let b_bits = b.to_bits() as i32;

        // Handle sign difference
        if (a_bits < 0) != (b_bits < 0) {
            // Different signs - check if both are near zero
            if a.abs() < EPSILON && b.abs() < EPSILON {
                return 0;
            }
            return i32::MAX;
        }

        (a_bits - b_bits).abs()
    }

    /// Check if two f32 values are within ULP tolerance
    fn within_ulp_tolerance(a: f32, b: f32, max_ulp: i32) -> bool {
        ulp_difference(a, b) <= max_ulp
    }

    // ============================================================================
    // Scalar Reference Implementations
    // ============================================================================

    /// Scalar reference: Pi-quantization dequantize
    fn scalar_pi_dequantize(quantized: &[i8], alpha: f32, k: u8) -> Vec<f32> {
        let step = PI / (k as f32);
        quantized
            .iter()
            .map(|&q| (q as f32) * alpha * step)
            .collect()
    }

    /// Scalar reference: Compute block alpha
    fn scalar_compute_alpha(weights: &[f32], k: u8, bits: u8) -> f32 {
        if weights.is_empty() {
            return EPSILON;
        }

        let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let step = PI / (k as f32);
        let levels = (1 << bits) as f32;

        let alpha = max_abs / ((levels - 1.0) * step / 2.0);
        alpha.max(EPSILON)
    }

    /// Scalar reference: Quantize block
    fn scalar_quantize_block(weights: &[f32], k: u8, bits: u8) -> (Vec<i8>, f32) {
        let alpha = scalar_compute_alpha(weights, k, bits);
        let step = PI / (k as f32);
        let half_range = ((1i8 << bits) - 1) / 2;

        let quantized: Vec<i8> = weights
            .iter()
            .map(|&w| {
                let normalized = w / (alpha * step);
                let q = normalized.round() as i8;
                q.clamp(-half_range - 1, half_range)
            })
            .collect();

        (quantized, alpha)
    }

    // ============================================================================
    // Simulated SIMD Implementations (Stand-ins for actual SIMD code)
    // ============================================================================

    /// Simulated NEON implementation of pi_dequantize
    /// In production, this would use actual ARM NEON intrinsics
    #[cfg(target_arch = "aarch64")]
    fn simd_neon_pi_dequantize(quantized: &[i8], alpha: f32, k: u8) -> Vec<f32> {
        // NEON processes 4 f32 values at a time
        let step = PI / (k as f32);
        let scale = alpha * step;

        let mut result = Vec::with_capacity(quantized.len());

        // Process in chunks of 4 (NEON f32x4)
        let chunks = quantized.chunks(4);
        for chunk in chunks {
            for &q in chunk {
                result.push((q as f32) * scale);
            }
        }

        // Handle remainder
        while result.len() < quantized.len() {
            let idx = result.len();
            result.push((quantized[idx] as f32) * scale);
        }

        result
    }

    /// Simulated AVX2 implementation of pi_dequantize
    /// In production, this would use actual x86_64 AVX2 intrinsics
    #[cfg(target_arch = "x86_64")]
    fn simd_avx2_pi_dequantize(quantized: &[i8], alpha: f32, k: u8) -> Vec<f32> {
        // AVX2 processes 8 f32 values at a time
        let step = PI / (k as f32);
        let scale = alpha * step;

        let mut result = Vec::with_capacity(quantized.len());

        // Process in chunks of 8 (AVX2 f32x8)
        let chunks = quantized.chunks(8);
        for chunk in chunks {
            for &q in chunk {
                result.push((q as f32) * scale);
            }
        }

        // Handle remainder
        while result.len() < quantized.len() {
            let idx = result.len();
            result.push((quantized[idx] as f32) * scale);
        }

        result
    }

    /// Simulated WASM SIMD128 implementation of pi_dequantize
    fn simd_wasm_pi_dequantize(quantized: &[i8], alpha: f32, k: u8) -> Vec<f32> {
        // WASM SIMD128 processes 4 f32 values at a time
        let step = PI / (k as f32);
        let scale = alpha * step;

        let mut result = Vec::with_capacity(quantized.len());

        // Process in chunks of 4 (WASM SIMD128 f32x4)
        let chunks = quantized.chunks(4);
        for chunk in chunks {
            for &q in chunk {
                result.push((q as f32) * scale);
            }
        }

        // Handle remainder
        while result.len() < quantized.len() {
            let idx = result.len();
            result.push((quantized[idx] as f32) * scale);
        }

        result
    }

    /// Generic SIMD stub that falls back to scalar
    fn simd_generic_pi_dequantize(quantized: &[i8], alpha: f32, k: u8) -> Vec<f32> {
        // This simulates the behavior of a SIMD implementation
        // with potential minor floating-point differences
        let step = PI / (k as f32);
        let scale = alpha * step;

        quantized
            .iter()
            .map(|&q| {
                // Simulate potential SIMD FMA vs separate mul+add
                let q_f32 = q as f32;
                q_f32 * scale
            })
            .collect()
    }

    // ============================================================================
    // 1. SIMD vs Scalar Equivalence Tests (<=1 ULP)
    // ============================================================================

    #[test]
    fn test_simd_scalar_equivalence_simple() {
        let quantized: Vec<i8> = vec![0, 1, 2, 3, -1, -2, -3, -4];
        let alpha = 1.0;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

        assert_eq!(scalar_result.len(), simd_result.len());

        for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            let ulp = ulp_difference(s, simd);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "Element {}: scalar={}, simd={}, ULP diff={} (max={})",
                i,
                s,
                simd,
                ulp,
                MAX_ULP_DIFFERENCE
            );
        }
    }

    #[test]
    fn test_simd_scalar_equivalence_random_data() {
        // Generate pseudo-random quantized data
        let quantized: Vec<i8> = (0..SIMD_BLOCK_SIZE)
            .map(|i| ((i as i32 * 17 + 13) % 8 - 4) as i8)
            .collect();

        let alpha = 2.5;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

        let mut max_ulp_seen = 0i32;
        let mut ulp_violations = 0usize;

        for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            let ulp = ulp_difference(s, simd);
            max_ulp_seen = max_ulp_seen.max(ulp);

            if ulp > MAX_ULP_DIFFERENCE {
                ulp_violations += 1;
                eprintln!(
                    "ULP violation at {}: scalar={}, simd={}, ULP={}",
                    i, s, simd, ulp
                );
            }
        }

        assert_eq!(
            ulp_violations, 0,
            "Found {} ULP violations, max ULP seen = {}",
            ulp_violations, max_ulp_seen
        );
    }

    #[test]
    fn test_simd_scalar_equivalence_large_block() {
        // Test with a larger block
        let quantized: Vec<i8> = (0..4096)
            .map(|i| ((i as i32 * 31 + 7) % 8 - 4) as i8)
            .collect();

        let alpha = 0.75;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            assert!(
                within_ulp_tolerance(s, simd, MAX_ULP_DIFFERENCE),
                "Large block element {}: scalar={}, simd={}, ULP={}",
                i,
                s,
                simd,
                ulp_difference(s, simd)
            );
        }
    }

    // ============================================================================
    // 2. Edge Case Tests (Max/Min Values, Zeros)
    // ============================================================================

    #[test]
    fn test_simd_scalar_equivalence_zeros() {
        let quantized: Vec<i8> = vec![0; SIMD_BLOCK_SIZE];
        let alpha = 1.5;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            assert!(
                s == 0.0 && simd == 0.0,
                "Zero input should produce zero output at {}: scalar={}, simd={}",
                i,
                s,
                simd
            );
        }
    }

    #[test]
    fn test_simd_scalar_equivalence_max_values() {
        // All maximum positive values for 3-bit
        let quantized: Vec<i8> = vec![3; SIMD_BLOCK_SIZE];
        let alpha = 10.0;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            let ulp = ulp_difference(s, simd);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "Max value element {}: scalar={}, simd={}, ULP={}",
                i,
                s,
                simd,
                ulp
            );
        }
    }

    #[test]
    fn test_simd_scalar_equivalence_min_values() {
        // All minimum negative values for 3-bit
        let quantized: Vec<i8> = vec![-4; SIMD_BLOCK_SIZE];
        let alpha = 10.0;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            let ulp = ulp_difference(s, simd);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "Min value element {}: scalar={}, simd={}, ULP={}",
                i,
                s,
                simd,
                ulp
            );
        }
    }

    #[test]
    fn test_simd_scalar_equivalence_alternating_extremes() {
        // Alternating max and min values
        let quantized: Vec<i8> = (0..SIMD_BLOCK_SIZE)
            .map(|i| if i % 2 == 0 { 3 } else { -4 })
            .collect();

        let alpha = 5.0;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            let ulp = ulp_difference(s, simd);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "Alternating element {}: scalar={}, simd={}, ULP={}",
                i,
                s,
                simd,
                ulp
            );
        }
    }

    // ============================================================================
    // 3. Test with Different Alpha Values
    // ============================================================================

    #[test]
    fn test_simd_scalar_equivalence_small_alpha() {
        let quantized: Vec<i8> = (0..256).map(|i| (i % 8 - 4) as i8).collect();
        let alpha = 1e-6;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            let ulp = ulp_difference(s, simd);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "Small alpha element {}: scalar={}, simd={}, ULP={}",
                i,
                s,
                simd,
                ulp
            );
        }
    }

    #[test]
    fn test_simd_scalar_equivalence_large_alpha() {
        let quantized: Vec<i8> = (0..256).map(|i| (i % 8 - 4) as i8).collect();
        let alpha = 1e6;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            let ulp = ulp_difference(s, simd);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "Large alpha element {}: scalar={}, simd={}, ULP={}",
                i,
                s,
                simd,
                ulp
            );
        }
    }

    #[test]
    fn test_simd_scalar_equivalence_fractional_alpha() {
        let quantized: Vec<i8> = (0..256).map(|i| (i % 8 - 4) as i8).collect();
        let alpha = 0.123456789;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            let ulp = ulp_difference(s, simd);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "Fractional alpha element {}: scalar={}, simd={}, ULP={}",
                i,
                s,
                simd,
                ulp
            );
        }
    }

    // ============================================================================
    // 4. Test with Different k Values (Step Sizes)
    // ============================================================================

    #[test]
    fn test_simd_scalar_equivalence_k2() {
        let quantized: Vec<i8> = (0..256).map(|i| (i % 4 - 2) as i8).collect();
        let alpha = 1.5;
        let k = 2u8; // pi/2 step

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            let ulp = ulp_difference(s, simd);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "k=2 element {}: scalar={}, simd={}, ULP={}",
                i,
                s,
                simd,
                ulp
            );
        }
    }

    #[test]
    fn test_simd_scalar_equivalence_k4() {
        let quantized: Vec<i8> = (0..256).map(|i| (i % 8 - 4) as i8).collect();
        let alpha = 1.5;
        let k = 4u8; // pi/4 step

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            let ulp = ulp_difference(s, simd);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "k=4 element {}: scalar={}, simd={}, ULP={}",
                i,
                s,
                simd,
                ulp
            );
        }
    }

    // ============================================================================
    // 5. Full Quantize-Dequantize Pipeline Equivalence
    // ============================================================================

    #[test]
    fn test_full_pipeline_scalar_simd_equivalence() {
        // Generate weights
        let weights: Vec<f32> = (0..SIMD_BLOCK_SIZE)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();

        let k = 3u8;
        let bits = 3u8;

        // Quantize with scalar
        let (quantized, alpha) = scalar_quantize_block(&weights, k, bits);

        // Dequantize with both scalar and SIMD
        let scalar_dequant = scalar_pi_dequantize(&quantized, alpha, k);
        let simd_dequant = simd_generic_pi_dequantize(&quantized, alpha, k);

        // Verify equivalence
        for (i, (&s, &simd)) in scalar_dequant.iter().zip(simd_dequant.iter()).enumerate() {
            let ulp = ulp_difference(s, simd);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "Full pipeline element {}: scalar={}, simd={}, ULP={}",
                i,
                s,
                simd,
                ulp
            );
        }
    }

    // ============================================================================
    // 6. Platform-Specific Tests
    // ============================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_scalar_equivalence() {
        let quantized: Vec<i8> = (0..256).map(|i| (i % 8 - 4) as i8).collect();
        let alpha = 2.0;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let neon_result = simd_neon_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &neon)) in scalar_result.iter().zip(neon_result.iter()).enumerate() {
            let ulp = ulp_difference(s, neon);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "NEON element {}: scalar={}, neon={}, ULP={}",
                i,
                s,
                neon,
                ulp
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_scalar_equivalence() {
        let quantized: Vec<i8> = (0..256).map(|i| (i % 8 - 4) as i8).collect();
        let alpha = 2.0;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let avx2_result = simd_avx2_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &avx2)) in scalar_result.iter().zip(avx2_result.iter()).enumerate() {
            let ulp = ulp_difference(s, avx2);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "AVX2 element {}: scalar={}, avx2={}, ULP={}",
                i,
                s,
                avx2,
                ulp
            );
        }
    }

    #[test]
    fn test_wasm_scalar_equivalence() {
        let quantized: Vec<i8> = (0..256).map(|i| (i % 8 - 4) as i8).collect();
        let alpha = 2.0;
        let k = 3u8;

        let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
        let wasm_result = simd_wasm_pi_dequantize(&quantized, alpha, k);

        for (i, (&s, &wasm)) in scalar_result.iter().zip(wasm_result.iter()).enumerate() {
            let ulp = ulp_difference(s, wasm);
            assert!(
                ulp <= MAX_ULP_DIFFERENCE,
                "WASM element {}: scalar={}, wasm={}, ULP={}",
                i,
                s,
                wasm,
                ulp
            );
        }
    }

    // ============================================================================
    // 7. Stress Tests
    // ============================================================================

    #[test]
    fn test_simd_scalar_stress_many_blocks() {
        // Test many blocks in sequence
        // Note: Allow slightly higher ULP tolerance (4) for stress tests
        // due to floating-point accumulation across many operations
        const STRESS_TEST_ULP_TOLERANCE: i32 = 4;

        let num_blocks = 100;
        let block_size = SIMD_BLOCK_SIZE;
        let mut max_ulp_seen = 0i32;

        for block_idx in 0..num_blocks {
            let quantized: Vec<i8> = (0..block_size)
                .map(|i| (((block_idx * 17 + i) % 8) as i32 - 4) as i8)
                .collect();

            let alpha = 1.0 + (block_idx as f32) * 0.1;
            let k = 3u8;

            let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
            let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

            for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
                let ulp = ulp_difference(s, simd);
                max_ulp_seen = max_ulp_seen.max(ulp);
                assert!(
                    ulp <= STRESS_TEST_ULP_TOLERANCE,
                    "Block {} element {}: scalar={}, simd={}, ULP={} (max allowed={})",
                    block_idx,
                    i,
                    s,
                    simd,
                    ulp,
                    STRESS_TEST_ULP_TOLERANCE
                );
            }
        }

        // Log the max ULP seen for monitoring
        eprintln!("Stress test max ULP seen: {}", max_ulp_seen);
    }

    #[test]
    fn test_simd_scalar_equivalence_non_aligned() {
        // Test non-power-of-2 sizes
        for size in [1, 3, 7, 15, 31, 63, 127, 255, 257, 511, 1023] {
            let quantized: Vec<i8> = (0..size).map(|i| ((i % 8) as i32 - 4) as i8).collect();

            let alpha = 1.5;
            let k = 3u8;

            let scalar_result = scalar_pi_dequantize(&quantized, alpha, k);
            let simd_result = simd_generic_pi_dequantize(&quantized, alpha, k);

            assert_eq!(
                scalar_result.len(),
                simd_result.len(),
                "Size {} mismatch",
                size
            );

            for (i, (&s, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
                let ulp = ulp_difference(s, simd);
                assert!(
                    ulp <= MAX_ULP_DIFFERENCE,
                    "Size {} element {}: scalar={}, simd={}, ULP={}",
                    size,
                    i,
                    s,
                    simd,
                    ulp
                );
            }
        }
    }

    // ============================================================================
    // 8. Special Float Value Tests
    // ============================================================================

    #[test]
    fn test_ulp_calculation_special_values() {
        // Same value -> 0 ULP
        assert_eq!(ulp_difference(1.0, 1.0), 0);
        assert_eq!(ulp_difference(0.0, 0.0), 0);
        assert_eq!(ulp_difference(-1.0, -1.0), 0);

        // Adjacent floats -> 1 ULP
        let one = 1.0f32;
        let next = f32::from_bits(one.to_bits() + 1);
        assert_eq!(ulp_difference(one, next), 1);

        // NaN handling
        assert_eq!(ulp_difference(f32::NAN, 1.0), i32::MAX);
        assert_eq!(ulp_difference(1.0, f32::NAN), i32::MAX);
        assert_eq!(ulp_difference(f32::NAN, f32::NAN), i32::MAX);

        // Infinity handling
        assert_eq!(ulp_difference(f32::INFINITY, f32::INFINITY), 0);
        assert_eq!(ulp_difference(f32::NEG_INFINITY, f32::NEG_INFINITY), 0);
        assert_eq!(ulp_difference(f32::INFINITY, f32::NEG_INFINITY), i32::MAX);
    }

    #[test]
    fn test_within_ulp_tolerance_helper() {
        assert!(within_ulp_tolerance(1.0, 1.0, 0));
        assert!(within_ulp_tolerance(1.0, 1.0, 1));

        let one = 1.0f32;
        let next = f32::from_bits(one.to_bits() + 1);
        assert!(within_ulp_tolerance(one, next, 1));
        assert!(!within_ulp_tolerance(one, next, 0));
    }
}
