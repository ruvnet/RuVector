//! WASM Bindings for Pi-Quantization (ADR-090 Phase 4)
//!
//! This module provides WebAssembly bindings for Pi-constant quantization,
//! enabling ultra-low-bit weight compression directly in web browsers.
//!
//! ## Features
//!
//! - **Pi-Quantization**: Uses pi-based step sizes instead of uniform grids
//! - **WASM SIMD128**: Hardware-accelerated dequantization kernel
//! - **Quality Metrics**: MSE and spectral distortion computation
//! - **JSON Serialization**: Configuration export/import
//!
//! ## Pi-Quantization Formula
//!
//! ```text
//! w_q = round(w / (alpha * pi / k)) * (alpha * pi / k)
//! ```
//!
//! Where:
//! - `alpha` is a learnable per-channel scale factor
//! - `k` is a small integer (2, 3, 4, or 5) controlling step granularity
//! - `pi` provides mathematically favorable quantization boundaries
//!
//! ## Quick Start (JavaScript)
//!
//! ```javascript
//! import { PiQuantWasm } from 'ruvllm-wasm';
//!
//! // Create a 3-bit pi-quantizer with k=4
//! const quantizer = new PiQuantWasm(3, 4);
//!
//! // Quantize weights
//! const weights = new Float32Array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]);
//! const packed = quantizer.quantize(weights);
//!
//! // Dequantize
//! const reconstructed = quantizer.dequantize(packed);
//!
//! // Compute quality metrics
//! const mse = quantizer.computeMse(weights, packed);
//! const spectral = quantizer.spectralDistortion(weights, packed);
//!
//! console.log(`MSE: ${mse}, Spectral Distortion: ${spectral} dB`);
//! ```

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

// ============================================================================
// Constants
// ============================================================================

/// Valid k values for pi-quantization step size
const VALID_K_VALUES: [u8; 4] = [2, 3, 4, 5];

/// Number of weights per 3-bit block
const PI3_BLOCK_WEIGHTS: usize = 8;

/// Number of bytes per 3-bit block (8 weights * 3 bits = 24 bits = 3 bytes)
const PI3_BLOCK_BYTES: usize = 3;

/// Number of weights per 2-bit block
const PI2_BLOCK_WEIGHTS: usize = 4;

/// Number of bytes per 2-bit block
const PI2_BLOCK_BYTES: usize = 1;

// ============================================================================
// Configuration Structure (for JSON serialization)
// ============================================================================

/// Serializable configuration for PiQuantWasm
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PiQuantConfig {
    bits: u8,
    k: u8,
    alpha: f32,
}

// ============================================================================
// PiQuantWasm - Main WASM Binding
// ============================================================================

/// WASM-bindgen wrapper for Pi-constant quantization.
///
/// Provides browser-compatible ultra-low-bit weight compression using
/// pi-based step sizes for improved precision.
#[wasm_bindgen]
pub struct PiQuantWasm {
    /// Number of quantization bits (2 or 3)
    bits: u8,

    /// Divisor for pi step size (2, 3, 4, or 5)
    k: u8,

    /// Per-channel scale factor (must be positive)
    alpha: f32,

    /// Precomputed half-range for clamping: 2^(bits-1)
    half_range: i8,

    /// Precomputed base step: pi / k
    base_step: f32,
}

#[wasm_bindgen]
impl PiQuantWasm {
    /// Create a new Pi-quantizer with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `bits` - Number of quantization bits (must be 2 or 3)
    /// * `k` - Divisor for pi step size (must be 2, 3, 4, or 5)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `bits` is not 2 or 3
    /// - `k` is not in {2, 3, 4, 5}
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const quantizer = new PiQuantWasm(3, 4); // 3-bit with k=4
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(bits: u8, k: u8) -> Self {
        // Validate bits
        if bits != 2 && bits != 3 {
            panic!("Bits must be 2 or 3, got {}", bits);
        }

        // Validate k
        if !VALID_K_VALUES.contains(&k) {
            panic!("K value must be in {{2, 3, 4, 5}}, got {}", k);
        }

        let half_range = 1i8 << (bits - 1);
        let base_step = PI / (k as f32);

        Self {
            bits,
            k,
            alpha: 1.0, // Default scale
            half_range,
            base_step,
        }
    }

    /// Get the effective bits per weight including scale overhead.
    ///
    /// This accounts for per-block scale metadata storage.
    #[wasm_bindgen(getter, js_name = bitsPerWeight)]
    pub fn bits_per_weight(&self) -> f32 {
        match self.bits {
            3 => 3.0625, // 3 bits + scale overhead
            2 => 2.0625, // 2 bits + scale overhead
            _ => self.bits as f32,
        }
    }

    /// Get the current step size (alpha * pi / k).
    #[wasm_bindgen(getter, js_name = stepSize)]
    pub fn step_size(&self) -> f32 {
        self.alpha * self.base_step
    }

    /// Get the number of quantization bits.
    #[wasm_bindgen(getter)]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Get the k divisor value.
    #[wasm_bindgen(getter)]
    pub fn k(&self) -> u8 {
        self.k
    }

    /// Get the alpha scale factor.
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Set the alpha scale factor.
    ///
    /// # Arguments
    ///
    /// * `value` - New alpha value (must be positive)
    ///
    /// # Panics
    ///
    /// Panics if value is not positive or not finite.
    #[wasm_bindgen(setter)]
    pub fn set_alpha(&mut self, value: f32) {
        if value <= 0.0 || !value.is_finite() {
            panic!("Alpha must be positive and finite, got {}", value);
        }
        self.alpha = value;
    }

    /// Calibrate alpha from a set of weights.
    ///
    /// Sets alpha to minimize clipping based on the maximum absolute value.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight values to calibrate from
    #[wasm_bindgen(js_name = calibrateAlpha)]
    pub fn calibrate_alpha(&mut self, weights: &[f32]) {
        if weights.is_empty() {
            return;
        }

        let max_abs = weights
            .iter()
            .map(|w| w.abs())
            .fold(0.0f32, |a, b| a.max(b));

        let half = self.half_range as f32;
        let divisor = (half - 0.5) * self.base_step;

        // Ensure positivity
        self.alpha = (max_abs / divisor).max(1e-8);
    }

    /// Quantize f32 weights to packed format.
    ///
    /// For 3-bit: packs 8 weights into 3 bytes
    /// For 2-bit: packs 4 weights into 1 byte
    ///
    /// # Arguments
    ///
    /// * `weights` - Input f32 weight values
    ///
    /// # Returns
    ///
    /// Packed quantized data as byte array
    pub fn quantize(&self, weights: &[f32]) -> Vec<u8> {
        let step = self.step_size();
        let inv_step = if step > 1e-10 { 1.0 / step } else { 0.0 };
        let half = self.half_range as i32;

        match self.bits {
            3 => self.quantize_3bit(weights, inv_step, half),
            2 => self.quantize_2bit(weights, inv_step, half),
            _ => Vec::new(),
        }
    }

    /// Dequantize packed data back to f32 weights.
    ///
    /// Uses WASM SIMD128 when available for acceleration.
    ///
    /// # Arguments
    ///
    /// * `packed` - Packed quantized data
    ///
    /// # Returns
    ///
    /// Dequantized f32 weight values
    pub fn dequantize(&self, packed: &[u8]) -> Vec<f32> {
        let scale = self.step_size();

        match self.bits {
            3 => self.dequantize_3bit(packed, scale),
            2 => self.dequantize_2bit(packed, scale),
            _ => Vec::new(),
        }
    }

    /// Compute Mean Squared Error between original and quantized weights.
    ///
    /// # Arguments
    ///
    /// * `original` - Original f32 weights
    /// * `quantized` - Packed quantized data
    ///
    /// # Returns
    ///
    /// MSE value (lower is better)
    #[wasm_bindgen(js_name = computeMse)]
    pub fn compute_mse(&self, original: &[f32], quantized: &[u8]) -> f32 {
        let reconstructed = self.dequantize(quantized);

        if original.len() != reconstructed.len() || original.is_empty() {
            return 0.0;
        }

        let sum: f64 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(&o, &q)| {
                let diff = (o - q) as f64;
                diff * diff
            })
            .sum();

        (sum / original.len() as f64) as f32
    }

    /// Compute spectral distortion in dB.
    ///
    /// Formula: 10 * log10(MSE / signal_power)
    ///
    /// # Arguments
    ///
    /// * `original` - Original f32 weights
    /// * `quantized` - Packed quantized data
    ///
    /// # Returns
    ///
    /// Spectral distortion in decibels (more negative is better)
    #[wasm_bindgen(js_name = spectralDistortion)]
    pub fn spectral_distortion(&self, original: &[f32], quantized: &[u8]) -> f32 {
        let reconstructed = self.dequantize(quantized);

        if original.len() != reconstructed.len() || original.is_empty() {
            return f32::NEG_INFINITY;
        }

        let signal_power: f64 = original.iter().map(|&x| (x as f64).powi(2)).sum();
        if signal_power == 0.0 {
            return 0.0;
        }

        let mse = self.compute_mse(original, quantized) as f64;
        let avg_power = signal_power / original.len() as f64;

        (10.0 * (mse / avg_power).log10()) as f32
    }

    /// Export configuration to JSON string.
    ///
    /// # Returns
    ///
    /// JSON representation of the quantizer configuration
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        let config = PiQuantConfig {
            bits: self.bits,
            k: self.k,
            alpha: self.alpha,
        };

        serde_json::to_string(&config).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create a PiQuantWasm from JSON configuration.
    ///
    /// # Arguments
    ///
    /// * `json` - JSON string containing configuration
    ///
    /// # Returns
    ///
    /// New PiQuantWasm instance
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(json: &str) -> Result<PiQuantWasm, JsValue> {
        let config: PiQuantConfig =
            serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let mut quantizer = PiQuantWasm::new(config.bits, config.k);
        quantizer.alpha = config.alpha;

        Ok(quantizer)
    }

    // ========================================================================
    // Internal Methods: 3-bit Quantization
    // ========================================================================

    /// Quantize to 3-bit packed format (8 weights -> 3 bytes)
    fn quantize_3bit(&self, weights: &[f32], inv_step: f32, half: i32) -> Vec<u8> {
        let num_groups = (weights.len() + PI3_BLOCK_WEIGHTS - 1) / PI3_BLOCK_WEIGHTS;
        let mut packed = vec![0u8; num_groups * PI3_BLOCK_BYTES];

        for group in 0..num_groups {
            let val_offset = group * PI3_BLOCK_WEIGHTS;
            let byte_offset = group * PI3_BLOCK_BYTES;

            let mut combined: u32 = 0;

            for i in 0..PI3_BLOCK_WEIGHTS {
                let idx = val_offset + i;
                let w = if idx < weights.len() {
                    weights[idx]
                } else {
                    0.0
                };

                // Quantize: round(w / step)
                let q = (w * inv_step).round() as i32;

                // Clamp to valid range: [-half, half-1] = [-4, 3] for 3-bit
                let q_clamped = q.clamp(-half, half - 1);

                // Convert to unsigned 3-bit: add offset
                let unsigned = ((q_clamped + half) as u32) & 0x7;

                combined |= unsigned << (i * 3);
            }

            packed[byte_offset] = (combined & 0xFF) as u8;
            packed[byte_offset + 1] = ((combined >> 8) & 0xFF) as u8;
            packed[byte_offset + 2] = ((combined >> 16) & 0xFF) as u8;
        }

        packed
    }

    /// Dequantize 3-bit packed format using WASM SIMD128 when available
    fn dequantize_3bit(&self, packed: &[u8], scale: f32) -> Vec<f32> {
        if packed.len() % PI3_BLOCK_BYTES != 0 {
            return Vec::new();
        }

        let num_groups = packed.len() / PI3_BLOCK_BYTES;
        let mut output = vec![0.0f32; num_groups * PI3_BLOCK_WEIGHTS];

        // Use SIMD kernel on wasm32 target
        #[cfg(target_arch = "wasm32")]
        {
            // SAFETY: We're running on wasm32, so SIMD128 is available
            unsafe {
                self.dequantize_3bit_simd128(packed, scale, &mut output, num_groups);
            }
            return output;
        }

        // Scalar fallback for non-wasm32 targets
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.dequantize_3bit_scalar(packed, scale, &mut output, num_groups);
            output
        }
    }

    /// WASM SIMD128-accelerated 3-bit dequantization kernel
    #[cfg(target_arch = "wasm32")]
    unsafe fn dequantize_3bit_simd128(
        &self,
        packed: &[u8],
        scale: f32,
        output: &mut [f32],
        num_groups: usize,
    ) {
        // Sign-extension LUT: maps 3-bit unsigned [0-7] to signed [-4, +3]
        // Index i -> (i - 4) as i8
        let sign_lut = i8x16(-4, -3, -2, -1, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0);

        // Broadcast scale to f32x4 vector
        let scale_vec = f32x4_splat(scale);

        for group in 0..num_groups {
            let byte_offset = group * PI3_BLOCK_BYTES;
            let out_offset = group * PI3_BLOCK_WEIGHTS;

            // Load and combine 3 bytes into 24-bit value
            let b0 = *packed.get_unchecked(byte_offset) as u32;
            let b1 = *packed.get_unchecked(byte_offset + 1) as u32;
            let b2 = *packed.get_unchecked(byte_offset + 2) as u32;
            let combined = b0 | (b1 << 8) | (b2 << 16);

            // Extract 8 x 3-bit indices into array
            let mut indices = [0u8; 16];
            for i in 0..8 {
                indices[i] = ((combined >> (i * 3)) & 0x7) as u8;
            }

            // Load indices into v128
            let idx_vec = v128_load(indices.as_ptr() as *const v128);

            // LUT lookup: i8x16_swizzle maps each index to signed value
            let signed_i8 = i8x16_swizzle(sign_lut, idx_vec);

            // Process low 4 values (indices 0-3)
            let signed_i16_lo = i16x8_extend_low_i8x16(signed_i8);
            let signed_i32_lo_lo = i32x4_extend_low_i16x8(signed_i16_lo);
            let signed_i32_lo_hi = i32x4_extend_high_i16x8(signed_i16_lo);

            // Convert to f32 and multiply by scale
            let f32_0 = f32x4_mul(f32x4_convert_i32x4(signed_i32_lo_lo), scale_vec);
            let f32_1 = f32x4_mul(f32x4_convert_i32x4(signed_i32_lo_hi), scale_vec);

            // Store results
            v128_store(output.as_mut_ptr().add(out_offset) as *mut v128, f32_0);
            v128_store(output.as_mut_ptr().add(out_offset + 4) as *mut v128, f32_1);
        }
    }

    /// Scalar 3-bit dequantization (fallback)
    #[cfg(not(target_arch = "wasm32"))]
    fn dequantize_3bit_scalar(
        &self,
        packed: &[u8],
        scale: f32,
        output: &mut [f32],
        num_groups: usize,
    ) {
        let half = self.half_range as i32;

        for group in 0..num_groups {
            let byte_offset = group * PI3_BLOCK_BYTES;
            let out_offset = group * PI3_BLOCK_WEIGHTS;

            let b0 = packed[byte_offset] as u32;
            let b1 = packed[byte_offset + 1] as u32;
            let b2 = packed[byte_offset + 2] as u32;
            let combined = b0 | (b1 << 8) | (b2 << 16);

            for i in 0..PI3_BLOCK_WEIGHTS {
                let unsigned = ((combined >> (i * 3)) & 0x7) as i32;
                let signed = unsigned - half;
                output[out_offset + i] = (signed as f32) * scale;
            }
        }
    }

    // ========================================================================
    // Internal Methods: 2-bit Quantization
    // ========================================================================

    /// Quantize to 2-bit packed format (4 weights -> 1 byte)
    fn quantize_2bit(&self, weights: &[f32], inv_step: f32, half: i32) -> Vec<u8> {
        let num_blocks = (weights.len() + PI2_BLOCK_WEIGHTS - 1) / PI2_BLOCK_WEIGHTS;
        let mut packed = vec![0u8; num_blocks * PI2_BLOCK_BYTES];

        for block in 0..num_blocks {
            let val_offset = block * PI2_BLOCK_WEIGHTS;

            let mut byte: u8 = 0;

            for i in 0..PI2_BLOCK_WEIGHTS {
                let idx = val_offset + i;
                let w = if idx < weights.len() {
                    weights[idx]
                } else {
                    0.0
                };

                // Quantize: round(w / step)
                let q = (w * inv_step).round() as i32;

                // Clamp to valid range: [-half, half-1] = [-2, 1] for 2-bit
                let q_clamped = q.clamp(-half, half - 1);

                // Convert to unsigned 2-bit: add offset
                let unsigned = ((q_clamped + half) as u8) & 0x3;

                byte |= unsigned << (i * 2);
            }

            packed[block] = byte;
        }

        packed
    }

    /// Dequantize 2-bit packed format
    fn dequantize_2bit(&self, packed: &[u8], scale: f32) -> Vec<f32> {
        let num_blocks = packed.len();
        let mut output = vec![0.0f32; num_blocks * PI2_BLOCK_WEIGHTS];
        let half = self.half_range as i32;

        for block in 0..num_blocks {
            let byte = packed[block];
            let out_offset = block * PI2_BLOCK_WEIGHTS;

            for i in 0..PI2_BLOCK_WEIGHTS {
                let unsigned = ((byte >> (i * 2)) & 0x3) as i32;
                let signed = unsigned - half;
                output[out_offset + i] = (signed as f32) * scale;
            }
        }

        output
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_new_valid_3bit() {
        let q = PiQuantWasm::new(3, 4);
        assert_eq!(q.bits(), 3);
        assert_eq!(q.k(), 4);
        assert!((q.alpha() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_new_valid_2bit() {
        let q = PiQuantWasm::new(2, 4);
        assert_eq!(q.bits(), 2);
        assert_eq!(q.k(), 4);
    }

    #[test]
    #[should_panic(expected = "Bits must be 2 or 3")]
    fn test_new_invalid_bits() {
        PiQuantWasm::new(4, 4);
    }

    #[test]
    #[should_panic(expected = "K value must be")]
    fn test_new_invalid_k() {
        PiQuantWasm::new(3, 1);
    }

    #[test]
    fn test_step_size() {
        let q = PiQuantWasm::new(3, 4);
        let expected = PI / 4.0;
        assert!((q.step_size() - expected).abs() < EPSILON);
    }

    #[test]
    fn test_bits_per_weight() {
        let q3 = PiQuantWasm::new(3, 4);
        assert!((q3.bits_per_weight() - 3.0625).abs() < EPSILON);

        let q2 = PiQuantWasm::new(2, 4);
        assert!((q2.bits_per_weight() - 2.0625).abs() < EPSILON);
    }

    #[test]
    fn test_set_alpha() {
        let mut q = PiQuantWasm::new(3, 4);
        q.set_alpha(2.0);
        assert!((q.alpha() - 2.0).abs() < EPSILON);
    }

    #[test]
    #[should_panic(expected = "Alpha must be positive")]
    fn test_set_alpha_negative() {
        let mut q = PiQuantWasm::new(3, 4);
        q.set_alpha(-1.0);
    }

    #[test]
    fn test_calibrate_alpha() {
        let mut q = PiQuantWasm::new(3, 4);
        let weights = vec![0.1, -0.5, 0.3, -0.8, 0.2, 0.4, -0.6, 0.7];
        q.calibrate_alpha(&weights);
        assert!(q.alpha() > 0.0);
    }

    #[test]
    fn test_quantize_dequantize_3bit_roundtrip() {
        let q = PiQuantWasm::new(3, 4);
        let weights = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75];

        let packed = q.quantize(&weights);
        assert_eq!(packed.len(), 3); // 8 weights -> 3 bytes

        let reconstructed = q.dequantize(&packed);
        assert_eq!(reconstructed.len(), 8);

        // Check MSE is reasonable
        let mse = q.compute_mse(&weights, &packed);
        assert!(mse < 0.5, "MSE too high: {}", mse);
    }

    #[test]
    fn test_quantize_dequantize_2bit_roundtrip() {
        let q = PiQuantWasm::new(2, 4);
        let weights = vec![0.0, 0.5, -0.5, 1.0];

        let packed = q.quantize(&weights);
        assert_eq!(packed.len(), 1); // 4 weights -> 1 byte

        let reconstructed = q.dequantize(&packed);
        assert_eq!(reconstructed.len(), 4);
    }

    #[test]
    fn test_compute_mse_identical() {
        let q = PiQuantWasm::new(3, 4);
        let step = q.step_size();

        // Use values that quantize exactly
        let weights: Vec<f32> = (-4..=3).map(|v| (v as f32) * step).collect();
        let packed = q.quantize(&weights);

        let mse = q.compute_mse(&weights, &packed);
        assert!(mse < EPSILON, "MSE should be near zero: {}", mse);
    }

    #[test]
    fn test_spectral_distortion() {
        let q = PiQuantWasm::new(3, 4);
        let weights = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let packed = q.quantize(&weights);

        let sd = q.spectral_distortion(&weights, &packed);
        assert!(sd.is_finite(), "Spectral distortion should be finite");
    }

    #[test]
    fn test_json_roundtrip() {
        let mut q = PiQuantWasm::new(3, 4);
        q.set_alpha(2.5);

        let json = q.to_json().unwrap();
        let q2 = PiQuantWasm::from_json(&json).unwrap();

        assert_eq!(q.bits(), q2.bits());
        assert_eq!(q.k(), q2.k());
        assert!((q.alpha() - q2.alpha()).abs() < EPSILON);
    }

    #[test]
    fn test_quantize_empty() {
        let q = PiQuantWasm::new(3, 4);
        let packed = q.quantize(&[]);
        assert!(packed.is_empty());
    }

    #[test]
    fn test_dequantize_empty() {
        let q = PiQuantWasm::new(3, 4);
        let output = q.dequantize(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_quantize_partial_block() {
        let q = PiQuantWasm::new(3, 4);
        // 5 weights < 8 (one full block), should pad
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let packed = q.quantize(&weights);
        assert_eq!(packed.len(), 3); // Padded to full block

        let reconstructed = q.dequantize(&packed);
        assert_eq!(reconstructed.len(), 8); // Full block output

        // First 5 values should be close to original
        for i in 0..5 {
            let diff = (weights[i] - reconstructed[i]).abs();
            assert!(diff < 1.0, "Value {} differs too much: {}", i, diff);
        }
    }

    #[test]
    fn test_all_k_values() {
        for k in VALID_K_VALUES {
            let q = PiQuantWasm::new(3, k);
            let expected_step = PI / (k as f32);
            assert!(
                (q.step_size() - expected_step).abs() < EPSILON,
                "k={}: step mismatch",
                k
            );
        }
    }

    #[test]
    fn test_clamping_3bit() {
        let q = PiQuantWasm::new(3, 4);
        let step = q.step_size();

        // Values well outside the range [-4, 3] * step
        let weights = vec![
            step * 10.0,
            step * -10.0,
            step * 5.0,
            step * -6.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];

        let packed = q.quantize(&weights);
        let reconstructed = q.dequantize(&packed);

        // Should be clamped to max/min
        assert!(
            (reconstructed[0] - step * 3.0).abs() < EPSILON,
            "Should clamp to max"
        );
        assert!(
            (reconstructed[1] - step * -4.0).abs() < EPSILON,
            "Should clamp to min"
        );
    }
}
