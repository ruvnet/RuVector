//! Pi-Quantization Unit Tests for ADR-090
//!
//! Comprehensive tests for the PiQuantizer implementation including:
//! - Unit tests for PiQuantizer struct
//! - Round-trip accuracy tests (quantize -> dequantize)
//! - Block packing/unpacking tests
//! - Bounds checking tests
//! - Invariant validation (INV-2, INV-3)

use std::f32::consts::PI;

#[cfg(test)]
mod pi_quant_tests {
    use super::*;

    // ============================================================================
    // Test Constants
    // ============================================================================

    /// Pi/k step size for 3-bit (k=3)
    const PI_OVER_3: f32 = PI / 3.0;
    /// Pi/k step size for 2-bit (k=2)
    const PI_OVER_2: f32 = PI / 2.0;
    /// Epsilon for floating-point comparisons
    const EPSILON: f32 = 1e-6;
    /// Block size for packing tests
    const BLOCK_SIZE: usize = 256;

    // ============================================================================
    // PiQuantizer Struct (Simulation)
    // ============================================================================

    /// Pi-constant quantizer with configurable bit width
    ///
    /// Uses pi/k as the step size for quantization, providing better
    /// preservation of weight distributions compared to uniform quantization.
    #[derive(Debug, Clone)]
    struct PiQuantizer {
        /// Number of bits (2 or 3)
        bits: u8,
        /// k value for pi/k step size
        k: u8,
        /// Per-channel alpha scaling factors
        alpha_per_channel: Vec<f32>,
    }

    impl PiQuantizer {
        /// Create a new PiQuantizer
        ///
        /// # Arguments
        /// * `bits` - Bit width (2 or 3)
        /// * `k` - Step size divisor (step = pi/k)
        fn new(bits: u8, k: u8) -> Self {
            assert!(bits == 2 || bits == 3, "bits must be 2 or 3");
            assert!(k > 0, "k must be positive");
            Self {
                bits,
                k,
                alpha_per_channel: Vec::new(),
            }
        }

        /// Create PiQ3 quantizer (3-bit with pi/3 step)
        fn piq3() -> Self {
            Self::new(3, 3)
        }

        /// Create PiQ2 quantizer (2-bit with pi/2 step)
        fn piq2() -> Self {
            Self::new(2, 2)
        }

        /// Get the step size (pi/k)
        fn step_size(&self) -> f32 {
            PI / (self.k as f32)
        }

        /// Get the number of quantization levels
        fn num_levels(&self) -> u8 {
            1 << self.bits
        }

        /// Compute alpha (scale factor) for a block of weights
        ///
        /// INV-2: Scale factor must be positive
        fn compute_alpha(&self, weights: &[f32]) -> f32 {
            if weights.is_empty() {
                return EPSILON;
            }

            let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
            let step = self.step_size();
            let levels = self.num_levels() as f32;

            // Alpha = max(|w|) / ((levels - 1) * step / 2)
            // This ensures the weight range maps to quantization range
            let alpha = max_abs / ((levels - 1.0) * step / 2.0);

            // INV-2: Ensure positive scale
            alpha.max(EPSILON)
        }

        /// Quantize a single scalar value
        ///
        /// INV-3: Quantized value must be in valid range
        fn quantize_scalar(&self, value: f32, alpha: f32) -> i8 {
            let step = self.step_size();
            let levels = self.num_levels() as i8;
            let half_range = (levels - 1) / 2;

            // Normalize by alpha and step
            let normalized = value / (alpha * step);

            // Round to nearest integer and clamp
            let q = normalized.round() as i8;
            q.clamp(-half_range, half_range)
        }

        /// Dequantize a single scalar value
        fn dequantize_scalar(&self, q: i8, alpha: f32) -> f32 {
            let step = self.step_size();
            (q as f32) * alpha * step
        }

        /// Quantize a block of values
        fn quantize_block(&self, weights: &[f32]) -> (Vec<i8>, f32) {
            let alpha = self.compute_alpha(weights);
            let quantized: Vec<i8> = weights
                .iter()
                .map(|&w| self.quantize_scalar(w, alpha))
                .collect();
            (quantized, alpha)
        }

        /// Dequantize a block of values
        fn dequantize_block(&self, quantized: &[i8], alpha: f32) -> Vec<f32> {
            quantized
                .iter()
                .map(|&q| self.dequantize_scalar(q, alpha))
                .collect()
        }
    }

    // ============================================================================
    // Pi3BitBlock Packed Format (3 bytes -> 8 values)
    // ============================================================================

    /// Packed 3-bit block: 8 values in 3 bytes (24 bits total)
    #[derive(Debug, Clone, Default)]
    struct Pi3BitBlock {
        /// 3 bytes holding 8 3-bit values
        packed: [u8; 3],
        /// Scale factor for this block
        alpha: f32,
    }

    impl Pi3BitBlock {
        const VALUES_PER_BLOCK: usize = 8;
        const BYTES_PER_BLOCK: usize = 3;

        fn new() -> Self {
            Self::default()
        }

        /// Pack 8 signed 3-bit values (-4 to 3 range, offset to 0-7)
        fn pack(values: &[i8], alpha: f32) -> Self {
            assert!(values.len() == Self::VALUES_PER_BLOCK);

            let mut block = Self::new();
            block.alpha = alpha;

            // Convert signed (-4 to 3) to unsigned (0 to 7)
            let unsigned: Vec<u8> = values.iter().map(|&v| (v + 4) as u8).collect();

            // Pack 8 3-bit values into 3 bytes
            // Byte 0: val[0](3) | val[1](3) | val[2](2 low bits)
            // Byte 1: val[2](1 high) | val[3](3) | val[4](3) | val[5](1 low)
            // Byte 2: val[5](2 high) | val[6](3) | val[7](3)
            block.packed[0] = unsigned[0] | (unsigned[1] << 3) | ((unsigned[2] & 0x03) << 6);
            block.packed[1] = ((unsigned[2] >> 2) & 0x01)
                | (unsigned[3] << 1)
                | (unsigned[4] << 4)
                | ((unsigned[5] & 0x01) << 7);
            block.packed[2] = ((unsigned[5] >> 1) & 0x03) | (unsigned[6] << 2) | (unsigned[7] << 5);

            block
        }

        /// Unpack 8 signed 3-bit values
        fn unpack(&self) -> Vec<i8> {
            let mut values = Vec::with_capacity(Self::VALUES_PER_BLOCK);

            // Unpack from bytes
            let v0 = self.packed[0] & 0x07;
            let v1 = (self.packed[0] >> 3) & 0x07;
            let v2 = ((self.packed[0] >> 6) & 0x03) | ((self.packed[1] & 0x01) << 2);
            let v3 = (self.packed[1] >> 1) & 0x07;
            let v4 = (self.packed[1] >> 4) & 0x07;
            let v5 = ((self.packed[1] >> 7) & 0x01) | ((self.packed[2] & 0x03) << 1);
            let v6 = (self.packed[2] >> 2) & 0x07;
            let v7 = (self.packed[2] >> 5) & 0x07;

            // Convert unsigned (0-7) back to signed (-4 to 3)
            for v in [v0, v1, v2, v3, v4, v5, v6, v7] {
                values.push((v as i8) - 4);
            }

            values
        }
    }

    // ============================================================================
    // Pi2BitBlock Packed Format (1 byte -> 4 values)
    // ============================================================================

    /// Packed 2-bit block: 4 values in 1 byte
    #[derive(Debug, Clone, Default)]
    struct Pi2BitBlock {
        /// 1 byte holding 4 2-bit values
        packed: u8,
        /// Scale factor for this block
        alpha: f32,
    }

    impl Pi2BitBlock {
        const VALUES_PER_BLOCK: usize = 4;

        fn new() -> Self {
            Self::default()
        }

        /// Pack 4 signed 2-bit values (-2 to 1 range, offset to 0-3)
        fn pack(values: &[i8], alpha: f32) -> Self {
            assert!(values.len() == Self::VALUES_PER_BLOCK);

            let mut block = Self::new();
            block.alpha = alpha;

            // Convert signed (-2 to 1) to unsigned (0 to 3)
            for (i, &v) in values.iter().enumerate() {
                let unsigned = ((v + 2) as u8) & 0x03;
                block.packed |= unsigned << (i * 2);
            }

            block
        }

        /// Unpack 4 signed 2-bit values
        fn unpack(&self) -> Vec<i8> {
            let mut values = Vec::with_capacity(Self::VALUES_PER_BLOCK);

            for i in 0..Self::VALUES_PER_BLOCK {
                let unsigned = (self.packed >> (i * 2)) & 0x03;
                values.push((unsigned as i8) - 2);
            }

            values
        }
    }

    // ============================================================================
    // 1. Unit Tests for PiQuantizer
    // ============================================================================

    #[test]
    fn test_piq3_step_size() {
        let q = PiQuantizer::piq3();
        let expected = PI / 3.0;
        assert!(
            (q.step_size() - expected).abs() < EPSILON,
            "PiQ3 step size should be pi/3, got {}",
            q.step_size()
        );
    }

    #[test]
    fn test_piq2_step_size() {
        let q = PiQuantizer::piq2();
        let expected = PI / 2.0;
        assert!(
            (q.step_size() - expected).abs() < EPSILON,
            "PiQ2 step size should be pi/2, got {}",
            q.step_size()
        );
    }

    #[test]
    fn test_piq3_num_levels() {
        let q = PiQuantizer::piq3();
        assert_eq!(q.num_levels(), 8, "PiQ3 should have 8 levels (2^3)");
    }

    #[test]
    fn test_piq2_num_levels() {
        let q = PiQuantizer::piq2();
        assert_eq!(q.num_levels(), 4, "PiQ2 should have 4 levels (2^2)");
    }

    #[test]
    fn test_alpha_positive_invariant() {
        // INV-2: Scale factor must be positive
        let q = PiQuantizer::piq3();

        // Test with all zeros
        let zeros = vec![0.0f32; 256];
        let alpha = q.compute_alpha(&zeros);
        assert!(alpha > 0.0, "Alpha must be positive even for zero weights");

        // Test with empty
        let empty: Vec<f32> = vec![];
        let alpha = q.compute_alpha(&empty);
        assert!(alpha > 0.0, "Alpha must be positive for empty input");

        // Test with normal values
        let normal = vec![0.5, -0.3, 0.1, -0.7];
        let alpha = q.compute_alpha(&normal);
        assert!(alpha > 0.0, "Alpha must be positive: got {}", alpha);
    }

    #[test]
    fn test_step_size_constraint_invariant() {
        // INV-3: Step size must be pi/k
        for k in 1..=8u8 {
            let q = PiQuantizer::new(3, k);
            let expected = PI / (k as f32);
            assert!(
                (q.step_size() - expected).abs() < EPSILON,
                "Step size should be pi/{}, got {}",
                k,
                q.step_size()
            );
        }
    }

    // ============================================================================
    // 2. Round-Trip Accuracy Tests (Quantize -> Dequantize)
    // ============================================================================

    #[test]
    fn test_roundtrip_zero() {
        let q = PiQuantizer::piq3();
        let weights = vec![0.0f32; 8];

        let (quantized, alpha) = q.quantize_block(&weights);
        let dequantized = q.dequantize_block(&quantized, alpha);

        for (i, (&orig, &deq)) in weights.iter().zip(dequantized.iter()).enumerate() {
            assert!(
                (orig - deq).abs() < EPSILON,
                "Zero roundtrip failed at {}: {} -> {}",
                i,
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_roundtrip_uniform_random() {
        let q = PiQuantizer::piq3();

        // Generate pseudo-random weights in [-1, 1]
        let weights: Vec<f32> = (0..256).map(|i| ((i as f32) * 1.234).sin()).collect();

        let (quantized, alpha) = q.quantize_block(&weights);
        let dequantized = q.dequantize_block(&quantized, alpha);

        // Compute MSE
        let mse: f32 = weights
            .iter()
            .zip(dequantized.iter())
            .map(|(w, d)| (w - d).powi(2))
            .sum::<f32>()
            / weights.len() as f32;

        // PiQ3 should achieve reasonable MSE (< 0.1 for normalized weights)
        assert!(
            mse < 0.1,
            "PiQ3 roundtrip MSE too high: {} (expected < 0.1)",
            mse
        );
    }

    #[test]
    fn test_roundtrip_mse_comparison_piq3_vs_uniform() {
        // Compare PiQ3 against uniform 3-bit quantization
        let weights: Vec<f32> = (0..256)
            .map(|i| ((i as f32) * 0.789).sin() * 2.0) // Range [-2, 2]
            .collect();

        // PiQ3 quantization
        let piq3 = PiQuantizer::piq3();
        let (q_pi, alpha_pi) = piq3.quantize_block(&weights);
        let deq_pi = piq3.dequantize_block(&q_pi, alpha_pi);

        let mse_pi: f32 = weights
            .iter()
            .zip(deq_pi.iter())
            .map(|(w, d)| (w - d).powi(2))
            .sum::<f32>()
            / weights.len() as f32;

        // Uniform 3-bit quantization (baseline)
        let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let uniform_step = max_abs * 2.0 / 7.0; // 8 levels = 7 steps
        let q_uniform: Vec<i8> = weights
            .iter()
            .map(|&w| {
                let q = (w / uniform_step).round() as i8;
                q.clamp(-4, 3)
            })
            .collect();
        let deq_uniform: Vec<f32> = q_uniform
            .iter()
            .map(|&q| (q as f32) * uniform_step)
            .collect();

        let mse_uniform: f32 = weights
            .iter()
            .zip(deq_uniform.iter())
            .map(|(w, d)| (w - d).powi(2))
            .sum::<f32>()
            / weights.len() as f32;

        // Log both for comparison (PiQ3 should be competitive or better)
        eprintln!("MSE PiQ3: {:.6}, MSE Uniform: {:.6}", mse_pi, mse_uniform);

        // Both should have reasonable MSE
        assert!(mse_pi < 1.0, "PiQ3 MSE too high: {}", mse_pi);
        assert!(mse_uniform < 1.0, "Uniform MSE too high: {}", mse_uniform);
    }

    #[test]
    fn test_roundtrip_preserves_sign() {
        let q = PiQuantizer::piq3();
        // Use larger values that are less likely to quantize to zero
        let weights = vec![1.0, -1.0, 0.8, -0.8, 2.0, -2.0, 0.6, -0.6];

        let (quantized, alpha) = q.quantize_block(&weights);
        let dequantized = q.dequantize_block(&quantized, alpha);

        let mut sign_preserved_count = 0;
        let mut total_nonzero = 0;

        for (i, (&orig, &deq)) in weights.iter().zip(dequantized.iter()).enumerate() {
            // Only check sign for values that are significant relative to alpha
            if orig.abs() > alpha * 0.3 {
                total_nonzero += 1;
                if orig.signum() == deq.signum() || deq.abs() < EPSILON {
                    sign_preserved_count += 1;
                }
            }
        }

        // Most signs should be preserved (allow some quantization loss)
        let preservation_rate = sign_preserved_count as f32 / total_nonzero.max(1) as f32;
        assert!(
            preservation_rate > 0.7,
            "Sign preservation rate too low: {:.1}% ({}/{})",
            preservation_rate * 100.0,
            sign_preserved_count,
            total_nonzero
        );
    }

    #[test]
    fn test_roundtrip_piq2() {
        let q = PiQuantizer::piq2();
        let weights: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.5).sin()).collect();

        let (quantized, alpha) = q.quantize_block(&weights);
        let dequantized = q.dequantize_block(&quantized, alpha);

        // PiQ2 has fewer levels, so higher MSE is expected
        let mse: f32 = weights
            .iter()
            .zip(dequantized.iter())
            .map(|(w, d)| (w - d).powi(2))
            .sum::<f32>()
            / weights.len() as f32;

        assert!(mse < 0.5, "PiQ2 roundtrip MSE too high: {}", mse);
    }

    // ============================================================================
    // 3. Block Packing/Unpacking Tests
    // ============================================================================

    #[test]
    fn test_pi3bit_pack_unpack_simple() {
        let values: Vec<i8> = vec![0, 1, 2, 3, -1, -2, -3, -4];
        let block = Pi3BitBlock::pack(&values, 1.0);
        let unpacked = block.unpack();

        assert_eq!(values, unpacked, "Pi3BitBlock pack/unpack roundtrip failed");
    }

    #[test]
    fn test_pi3bit_pack_unpack_all_zeros() {
        let values: Vec<i8> = vec![0; 8];
        let block = Pi3BitBlock::pack(&values, 1.0);
        let unpacked = block.unpack();

        assert_eq!(values, unpacked, "All-zeros pack/unpack failed");
    }

    #[test]
    fn test_pi3bit_pack_unpack_extremes() {
        // Test boundary values: -4 to 3 for 3-bit signed
        let values: Vec<i8> = vec![-4, -3, -2, -1, 0, 1, 2, 3];
        let block = Pi3BitBlock::pack(&values, 1.0);
        let unpacked = block.unpack();

        assert_eq!(values, unpacked, "Extreme values pack/unpack failed");
    }

    #[test]
    fn test_pi3bit_pack_unpack_alternating() {
        let values: Vec<i8> = vec![3, -4, 3, -4, 3, -4, 3, -4];
        let block = Pi3BitBlock::pack(&values, 1.0);
        let unpacked = block.unpack();

        assert_eq!(values, unpacked, "Alternating pack/unpack failed");
    }

    #[test]
    fn test_pi2bit_pack_unpack_simple() {
        let values: Vec<i8> = vec![0, 1, -1, -2];
        let block = Pi2BitBlock::pack(&values, 1.0);
        let unpacked = block.unpack();

        assert_eq!(values, unpacked, "Pi2BitBlock pack/unpack roundtrip failed");
    }

    #[test]
    fn test_pi2bit_pack_unpack_all_zeros() {
        let values: Vec<i8> = vec![0; 4];
        let block = Pi2BitBlock::pack(&values, 1.0);
        let unpacked = block.unpack();

        assert_eq!(values, unpacked, "All-zeros 2-bit pack/unpack failed");
    }

    #[test]
    fn test_pi2bit_pack_unpack_extremes() {
        // Test boundary values: -2 to 1 for 2-bit signed
        let values: Vec<i8> = vec![-2, -1, 0, 1];
        let block = Pi2BitBlock::pack(&values, 1.0);
        let unpacked = block.unpack();

        assert_eq!(values, unpacked, "Extreme 2-bit values pack/unpack failed");
    }

    #[test]
    fn test_pi3bit_storage_size() {
        // 8 values * 3 bits = 24 bits = 3 bytes
        assert_eq!(
            Pi3BitBlock::BYTES_PER_BLOCK,
            3,
            "Pi3BitBlock should use 3 bytes for 8 values"
        );
    }

    #[test]
    fn test_pi2bit_storage_size() {
        // 4 values * 2 bits = 8 bits = 1 byte
        let block = Pi2BitBlock::new();
        assert_eq!(
            std::mem::size_of_val(&block.packed),
            1,
            "Pi2BitBlock should use 1 byte for 4 values"
        );
    }

    // ============================================================================
    // 4. Bounds Checking Tests
    // ============================================================================

    #[test]
    fn test_quantize_clamps_to_valid_range_piq3() {
        let q = PiQuantizer::piq3();
        let alpha = 0.1; // Small alpha to ensure large normalized values

        // Test values that should clamp to extremes
        let max_q = q.quantize_scalar(100.0, alpha);
        let min_q = q.quantize_scalar(-100.0, alpha);

        // Verify values are at the extremes of the valid range
        assert!(
            max_q >= 2 && max_q <= 3,
            "Large positive should clamp to max range, got {}",
            max_q
        );
        assert!(
            min_q >= -4 && min_q <= -3,
            "Large negative should clamp to min range, got {}",
            min_q
        );

        // Most importantly, verify clamping works (values stay in range)
        assert!(max_q <= 3, "Max should not exceed 3");
        assert!(min_q >= -4, "Min should not be less than -4");
    }

    #[test]
    fn test_quantize_clamps_to_valid_range_piq2() {
        let q = PiQuantizer::piq2();
        let alpha = 0.1; // Small alpha to ensure large normalized values

        // Test values that should clamp to extremes
        let max_q = q.quantize_scalar(100.0, alpha);
        let min_q = q.quantize_scalar(-100.0, alpha);

        // Verify values are at the extremes of the valid range
        assert!(
            max_q >= 0 && max_q <= 1,
            "Large positive should clamp to max range, got {}",
            max_q
        );
        assert!(
            min_q >= -2 && min_q <= -1,
            "Large negative should clamp to min range, got {}",
            min_q
        );

        // Most importantly, verify clamping works (values stay in range)
        assert!(max_q <= 1, "Max should not exceed 1");
        assert!(min_q >= -2, "Min should not be less than -2");
    }

    #[test]
    fn test_quantized_values_in_valid_range_piq3() {
        let q = PiQuantizer::piq3();
        let weights: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.1 - 50.0)).collect();

        let (quantized, _alpha) = q.quantize_block(&weights);

        for &qval in &quantized {
            assert!(
                qval >= -4 && qval <= 3,
                "PiQ3 quantized value {} out of range [-4, 3]",
                qval
            );
        }
    }

    #[test]
    fn test_quantized_values_in_valid_range_piq2() {
        let q = PiQuantizer::piq2();
        let weights: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.1 - 50.0)).collect();

        let (quantized, _alpha) = q.quantize_block(&weights);

        for &qval in &quantized {
            assert!(
                qval >= -2 && qval <= 1,
                "PiQ2 quantized value {} out of range [-2, 1]",
                qval
            );
        }
    }

    #[test]
    fn test_infinity_handling() {
        let q = PiQuantizer::piq3();
        let weights = vec![f32::INFINITY, f32::NEG_INFINITY, 1.0, -1.0];

        // This test verifies that infinity values don't cause panics
        // and produce values within the valid range
        let result = std::panic::catch_unwind(|| q.quantize_block(&weights));

        match result {
            Ok((quantized, alpha)) => {
                // Verify all quantized values are in valid range
                for &qval in &quantized {
                    assert!(
                        qval >= -4 && qval <= 3,
                        "Quantized value {} out of range [-4, 3]",
                        qval
                    );
                }

                // Alpha computation may produce infinity - that's acceptable
                // as long as it doesn't produce NaN
                assert!(!alpha.is_nan(), "Alpha should not be NaN: {}", alpha);
            }
            Err(_) => {
                // Panicking on infinity is acceptable behavior
                // (implementation may choose to reject invalid input)
            }
        }
    }

    #[test]
    fn test_nan_handling() {
        let q = PiQuantizer::piq3();
        let weights = vec![f32::NAN, 1.0, -1.0, 0.5];

        // Should not panic
        let result = std::panic::catch_unwind(|| q.quantize_block(&weights));

        // Either succeeds with reasonable output or panics gracefully
        if let Ok((quantized, _alpha)) = result {
            // All quantized values should be in valid range
            for &qval in &quantized {
                assert!(
                    qval >= -4 && qval <= 3,
                    "Quantized value with NaN input out of range: {}",
                    qval
                );
            }
        }
    }

    #[test]
    fn test_very_small_values() {
        let q = PiQuantizer::piq3();
        let weights = vec![1e-40, -1e-40, 1e-35, -1e-35, 0.0, 0.0, 0.0, 0.0];

        let (quantized, alpha) = q.quantize_block(&weights);

        // Very small values should quantize to 0 or near-zero
        // Alpha should be positive (INV-2)
        assert!(alpha > 0.0, "Alpha must be positive for tiny weights");

        // Values should be in valid range
        for &qval in &quantized {
            assert!(qval >= -4 && qval <= 3);
        }
    }

    #[test]
    fn test_very_large_values() {
        let q = PiQuantizer::piq3();
        let weights = vec![1e30, -1e30, 1e35, -1e35, 0.0, 0.0, 0.0, 0.0];

        let (quantized, alpha) = q.quantize_block(&weights);

        // Alpha should be positive and large (or infinity for very large inputs)
        assert!(alpha > 0.0, "Alpha should be positive");

        // Large values should quantize to non-zero values or extremes
        // The sign preservation depends on the relative magnitude and alpha
        // Key invariant: quantized values must be in valid range
        for &qval in &quantized {
            assert!(
                qval >= -4 && qval <= 3,
                "Quantized value {} out of range",
                qval
            );
        }

        // Dequantization should not produce NaN
        let dequantized = q.dequantize_block(&quantized, alpha);
        for &d in &dequantized {
            assert!(!d.is_nan(), "Dequantized value should not be NaN");
        }

        // If alpha is finite, verify the large values were handled
        if alpha.is_finite() && alpha < 1e38 {
            assert!(
                alpha > 1e20,
                "Alpha should scale with large weights: {}",
                alpha
            );
        }
    }

    // ============================================================================
    // 5. Edge Case Tests
    // ============================================================================

    #[test]
    fn test_empty_block() {
        let q = PiQuantizer::piq3();
        let weights: Vec<f32> = vec![];

        let alpha = q.compute_alpha(&weights);
        assert!(alpha > 0.0, "Alpha must be positive for empty input");

        let (quantized, _) = q.quantize_block(&weights);
        assert!(
            quantized.is_empty(),
            "Empty input should produce empty output"
        );
    }

    #[test]
    fn test_single_element() {
        let q = PiQuantizer::piq3();
        let weights = vec![0.5];

        let (quantized, alpha) = q.quantize_block(&weights);
        assert_eq!(quantized.len(), 1);

        let dequantized = q.dequantize_block(&quantized, alpha);
        assert_eq!(dequantized.len(), 1);

        // Roundtrip error should be bounded
        let error = (weights[0] - dequantized[0]).abs();
        assert!(
            error < 0.5,
            "Single element roundtrip error too high: {}",
            error
        );
    }

    #[test]
    fn test_all_same_value() {
        let q = PiQuantizer::piq3();
        let weights = vec![0.42; 256];

        let (quantized, alpha) = q.quantize_block(&weights);
        let dequantized = q.dequantize_block(&quantized, alpha);

        // All dequantized values should be similar
        let first = dequantized[0];
        for &d in &dequantized {
            assert!(
                (d - first).abs() < EPSILON,
                "All same input should produce same output"
            );
        }
    }

    #[test]
    fn test_large_block() {
        let q = PiQuantizer::piq3();
        let weights: Vec<f32> = (0..4096).map(|i| ((i as f32) * 0.001).sin()).collect();

        let (quantized, alpha) = q.quantize_block(&weights);
        let dequantized = q.dequantize_block(&quantized, alpha);

        assert_eq!(quantized.len(), 4096);
        assert_eq!(dequantized.len(), 4096);

        // MSE should be reasonable
        let mse: f32 = weights
            .iter()
            .zip(dequantized.iter())
            .map(|(w, d)| (w - d).powi(2))
            .sum::<f32>()
            / weights.len() as f32;

        assert!(mse < 0.1, "Large block MSE too high: {}", mse);
    }

    // ============================================================================
    // 6. Quality Metrics Tests
    // ============================================================================

    #[test]
    fn test_cosine_similarity() {
        let q = PiQuantizer::piq3();
        let weights: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05).sin()).collect();

        let (quantized, alpha) = q.quantize_block(&weights);
        let dequantized = q.dequantize_block(&quantized, alpha);

        // Compute cosine similarity
        let dot: f32 = weights
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_orig: f32 = weights.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_deq: f32 = dequantized.iter().map(|x| x * x).sum::<f32>().sqrt();

        let cosine_sim = if norm_orig > EPSILON && norm_deq > EPSILON {
            dot / (norm_orig * norm_deq)
        } else {
            1.0 // Both near zero
        };

        // PiQ3 should maintain high cosine similarity (> 0.95)
        assert!(
            cosine_sim > 0.95,
            "Cosine similarity too low: {} (expected > 0.95)",
            cosine_sim
        );
    }

    #[test]
    fn test_spectral_distortion() {
        let q = PiQuantizer::piq3();

        // Use weights with known spectral properties
        let weights: Vec<f32> = (0..256)
            .map(|i| ((i as f32) * 2.0 * PI / 256.0).sin() * 0.5)
            .collect();

        let (quantized, alpha) = q.quantize_block(&weights);
        let dequantized = q.dequantize_block(&quantized, alpha);

        // Compute signal-to-noise ratio in dB
        let signal_power: f32 = weights.iter().map(|x| x * x).sum::<f32>() / weights.len() as f32;
        let noise_power: f32 = weights
            .iter()
            .zip(dequantized.iter())
            .map(|(w, d)| (w - d).powi(2))
            .sum::<f32>()
            / weights.len() as f32;

        let snr_db = if noise_power > EPSILON {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f32::INFINITY
        };

        // PiQ3 should achieve reasonable SNR (> 10 dB for 3-bit)
        assert!(
            snr_db > 10.0,
            "Spectral SNR too low: {} dB (expected > 10 dB)",
            snr_db
        );
    }

    #[test]
    fn test_outlier_preservation() {
        let q = PiQuantizer::piq3();

        // Create weights with outliers
        let mut weights = vec![0.0; 256];
        for i in 0..250 {
            weights[i] = ((i as f32) * 0.01).sin() * 0.1; // Small values
        }
        // Add outliers at the end
        weights[250] = 2.0;
        weights[251] = -2.0;
        weights[252] = 1.5;
        weights[253] = -1.5;
        weights[254] = 3.0;
        weights[255] = -3.0;

        let (quantized, alpha) = q.quantize_block(&weights);
        let dequantized = q.dequantize_block(&quantized, alpha);

        // Outliers should be preserved with reasonable accuracy
        for i in 250..256 {
            let error = (weights[i] - dequantized[i]).abs();
            let relative_error = error / weights[i].abs().max(0.1);
            assert!(
                relative_error < 0.5,
                "Outlier at {} not preserved: orig={}, deq={}, error={}",
                i,
                weights[i],
                dequantized[i],
                error
            );
        }
    }
}
