//! Differentiable Quantization Module
//!
//! This module provides the `DifferentiableQuantizer` trait and implementations
//! for quantization-aware training (QAT). The quantizers support both forward
//! (quantization) and backward (gradient) passes.
//!
//! ## Architecture
//!
//! ```text
//! DifferentiableQuantizer (trait)
//! +-- forward(): Quantize weights to integer representation
//! +-- backward(): Compute gradients via STE
//! +-- dequantize(): Convert quantized values back to float
//!
//! Implementations:
//! +-- UniformQuantizer: Standard uniform quantization
//! +-- PiQuantDifferentiable: Pi-constant scaled quantization (ADR-090)
//! ```
//!
//! ## System Invariants
//!
//! - **INV-1**: STE gradient flow (no zero-gradient regions except clipping)
//! - **INV-2**: Scale positivity (alpha > 0 for Pi-quant)
//! - **INV-3**: Step size constraint (step = alpha * pi / k)

use super::config::{QatConfig, QuantGranularity, SteVariant};
use std::f32::consts::PI;

// ============================================================================
// DifferentiableQuantizer Trait
// ============================================================================

/// Trait for differentiable quantization operations
///
/// Implementations must support:
/// - Forward pass: Convert FP32 weights to quantized representation
/// - Backward pass: Compute gradients through the quantization operation
/// - Dequantization: Convert quantized values back to FP32
///
/// # Example
///
/// ```rust,ignore
/// let quantizer = PiQuantDifferentiable::new(3, 4); // 3-bit, k=4
///
/// // Forward pass
/// let (quantized, dequantized) = quantizer.forward(&weights);
///
/// // Backward pass (during training)
/// let grad_w = quantizer.backward(&weights, &dequantized, &grad_out);
/// ```
pub trait DifferentiableQuantizer: Send + Sync {
    /// Forward quantization pass
    ///
    /// # Arguments
    ///
    /// * `w` - Input weights (FP32)
    ///
    /// # Returns
    ///
    /// Tuple of (quantized_indices, dequantized_values)
    /// - quantized_indices: Integer quantization levels (as i8)
    /// - dequantized_values: Values after quant -> dequant (for loss computation)
    fn forward(&self, w: &[f32]) -> (Vec<i8>, Vec<f32>);

    /// Backward pass: compute gradient w.r.t. latent weights
    ///
    /// # Arguments
    ///
    /// * `w` - Latent weights (FP32)
    /// * `q` - Dequantized values (FP32)
    /// * `grad_out` - Upstream gradient (dL/dq)
    ///
    /// # Returns
    ///
    /// Gradient w.r.t. latent weights (dL/dw)
    fn backward(&self, w: &[f32], q: &[f32], grad_out: &[f32]) -> Vec<f32>;

    /// Dequantize integer values back to FP32
    ///
    /// # Arguments
    ///
    /// * `q_int` - Quantized integer values
    ///
    /// # Returns
    ///
    /// Dequantized FP32 values
    fn dequantize(&self, q_int: &[i8]) -> Vec<f32>;

    /// Get the current scale value
    fn scale(&self) -> f32;

    /// Update the scale (for learned step size)
    fn set_scale(&mut self, scale: f32);

    /// Get bits per weight
    fn bits(&self) -> u8;

    /// Get number of quantization levels
    fn num_levels(&self) -> usize {
        1 << self.bits()
    }

    /// Get the STE variant being used
    fn ste_variant(&self) -> &SteVariant;
}

// ============================================================================
// Uniform Quantizer
// ============================================================================

/// Standard uniform quantization with learnable scale
///
/// Quantizes weights to uniformly spaced levels:
/// q = clamp(round(w / scale), -half, half-1)
/// dequant = q * scale
#[derive(Debug, Clone)]
pub struct UniformQuantizer {
    /// Bits for quantization
    bits: u8,
    /// Scale factor (learnable)
    scale: f32,
    /// STE variant for gradient computation
    ste_variant: SteVariant,
    /// Use symmetric quantization (no zero-point)
    symmetric: bool,
}

impl UniformQuantizer {
    /// Create a new uniform quantizer
    pub fn new(bits: u8, ste_variant: SteVariant) -> Self {
        Self {
            bits,
            scale: 1.0,
            ste_variant,
            symmetric: true,
        }
    }

    /// Create from QAT config
    pub fn from_config(config: &QatConfig) -> Self {
        Self {
            bits: config.bits,
            scale: 1.0,
            ste_variant: config.ste_variant.clone(),
            symmetric: config.symmetric,
        }
    }

    /// Initialize scale from weight statistics
    pub fn init_scale_from_weights(&mut self, weights: &[f32]) {
        // Find max absolute value
        let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);

        // Set scale so that max_abs maps to max quantization level
        let half = (1 << self.bits) / 2;
        self.scale = if max_abs > 0.0 {
            max_abs / (half as f32 - 0.5) // Slight margin
        } else {
            1.0
        };
    }

    /// Get the quantization range
    fn get_range(&self) -> (i8, i8) {
        let half = (1i8 << (self.bits - 1)) as i8;
        (-half, half - 1)
    }
}

impl DifferentiableQuantizer for UniformQuantizer {
    fn forward(&self, w: &[f32]) -> (Vec<i8>, Vec<f32>) {
        let (min_q, max_q) = self.get_range();
        let mut q_int = Vec::with_capacity(w.len());
        let mut q_float = Vec::with_capacity(w.len());

        for &weight in w {
            // Quantize
            let q = (weight / self.scale).round();
            let q_clamped = q.clamp(min_q as f32, max_q as f32) as i8;

            // Dequantize
            let dequant = q_clamped as f32 * self.scale;

            q_int.push(q_clamped);
            q_float.push(dequant);
        }

        (q_int, q_float)
    }

    fn backward(&self, w: &[f32], q: &[f32], grad_out: &[f32]) -> Vec<f32> {
        let mut grad_w = vec![0.0f32; w.len()];
        self.ste_variant.backward_batch(w, q, grad_out, &mut grad_w);
        grad_w
    }

    fn dequantize(&self, q_int: &[i8]) -> Vec<f32> {
        q_int.iter().map(|&q| q as f32 * self.scale).collect()
    }

    fn scale(&self) -> f32 {
        self.scale
    }

    fn set_scale(&mut self, scale: f32) {
        // INV-2: Ensure scale is positive
        assert!(scale > 0.0, "Scale must be positive (INV-2)");
        self.scale = scale;
    }

    fn bits(&self) -> u8 {
        self.bits
    }

    fn ste_variant(&self) -> &SteVariant {
        &self.ste_variant
    }
}

// ============================================================================
// Pi-Quantization Differentiable
// ============================================================================

/// Pi-constant scaled quantization for QAT (ADR-090)
///
/// Uses pi-scaled step sizes instead of uniform grids:
/// step = alpha * pi / k
///
/// This provides better representation of periodic/trigonometric weight patterns
/// and can achieve ~0.5 effective bit improvement over uniform quantization.
///
/// ## System Invariants
///
/// - **INV-2**: alpha (scale) is always positive
/// - **INV-3**: step = alpha * pi / k where k in {2, 3, 4, 5}
#[derive(Debug, Clone)]
pub struct PiQuantDifferentiable {
    /// Bits for quantization
    bits: u8,
    /// Pi-constant divisor (step = alpha * pi / k)
    k: u8,
    /// Per-channel scales (learnable)
    alpha: Vec<f32>,
    /// Number of channels (1 for per-tensor)
    num_channels: usize,
    /// STE variant for gradient computation
    ste_variant: SteVariant,
    /// Quantization granularity
    granularity: QuantGranularity,
}

impl PiQuantDifferentiable {
    /// Create a new Pi-quantizer
    ///
    /// # Arguments
    ///
    /// * `bits` - Number of bits (2, 3, 4, 5)
    /// * `k` - Pi divisor (step = alpha * pi / k)
    pub fn new(bits: u8, k: u8) -> Self {
        assert!(matches!(bits, 2 | 3 | 4 | 5), "Bits must be 2, 3, 4, or 5");
        assert!(
            matches!(k, 2 | 3 | 4 | 5),
            "k must be 2, 3, 4, or 5 (INV-3)"
        );

        Self {
            bits,
            k,
            alpha: vec![1.0], // Single scale initially
            num_channels: 1,
            ste_variant: SteVariant::LearnedStepSize,
            granularity: QuantGranularity::PerTensor,
        }
    }

    /// Create from QAT config
    pub fn from_config(config: &QatConfig) -> Self {
        let k = config.pi_k.expect("Pi-quant requires pi_k to be set");

        Self {
            bits: config.bits,
            k,
            alpha: vec![1.0],
            num_channels: 1,
            ste_variant: config.ste_variant.clone(),
            granularity: config.granularity.clone(),
        }
    }

    /// Create for 3-bit Pi-quantization (PiQ3)
    pub fn piq3() -> Self {
        Self::new(3, 4) // step = alpha * pi/4
    }

    /// Create for 2-bit Pi-quantization (PiQ2)
    pub fn piq2() -> Self {
        Self::new(2, 3) // step = alpha * pi/3
    }

    /// Initialize scales for per-channel quantization
    pub fn init_per_channel(&mut self, num_channels: usize) {
        self.num_channels = num_channels;
        self.alpha = vec![1.0; num_channels];
        self.granularity = QuantGranularity::PerChannel;
    }

    /// Initialize scales from weight statistics
    pub fn init_scale_from_weights(&mut self, weights: &[f32], channel_size: Option<usize>) {
        match (channel_size, &self.granularity) {
            (Some(ch_size), QuantGranularity::PerChannel) => {
                // Per-channel initialization
                let num_channels = weights.len() / ch_size;
                self.alpha = Vec::with_capacity(num_channels);

                for c in 0..num_channels {
                    let start = c * ch_size;
                    let end = start + ch_size;
                    let channel_weights = &weights[start..end];

                    let max_abs = channel_weights
                        .iter()
                        .map(|w| w.abs())
                        .fold(0.0f32, f32::max);
                    let step = self.step_size(0); // Use default step for calculation
                    let half = (1 << self.bits) / 2;

                    let alpha = if max_abs > 0.0 && step > 0.0 {
                        max_abs / (step * (half as f32 - 0.5))
                    } else {
                        1.0
                    };

                    // INV-2: Ensure positive
                    self.alpha.push(alpha.max(1e-6));
                }

                self.num_channels = num_channels;
            }
            _ => {
                // Per-tensor initialization
                let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
                let step = PI / (self.k as f32);
                let half = (1 << self.bits) / 2;

                let alpha = if max_abs > 0.0 {
                    max_abs / (step * (half as f32 - 0.5))
                } else {
                    1.0
                };

                // INV-2: Ensure positive
                self.alpha = vec![alpha.max(1e-6)];
                self.num_channels = 1;
            }
        }
    }

    /// Get the step size for a channel
    ///
    /// INV-3: step = alpha * pi / k
    #[inline]
    pub fn step_size(&self, channel: usize) -> f32 {
        let alpha = self.alpha.get(channel).copied().unwrap_or(self.alpha[0]);
        alpha * PI / (self.k as f32)
    }

    /// Get the quantization range
    fn get_range(&self) -> (i8, i8) {
        let half = 1i8 << (self.bits - 1);
        (-half, half - 1)
    }

    /// Quantize a single value
    #[inline]
    fn quantize_scalar(&self, w: f32, channel: usize) -> (i8, f32) {
        let step = self.step_size(channel);
        let (min_q, max_q) = self.get_range();

        // Quantize
        let q = (w / step).round();
        let q_clamped = q.clamp(min_q as f32, max_q as f32) as i8;

        // Dequantize
        let dequant = q_clamped as f32 * step;

        (q_clamped, dequant)
    }

    /// Get all alpha values
    pub fn alphas(&self) -> &[f32] {
        &self.alpha
    }

    /// Set alpha for a channel
    pub fn set_alpha(&mut self, channel: usize, alpha: f32) {
        // INV-2: Ensure positive
        assert!(alpha > 0.0, "Alpha must be positive (INV-2)");
        if channel < self.alpha.len() {
            self.alpha[channel] = alpha;
        }
    }

    /// Get k value
    pub fn k(&self) -> u8 {
        self.k
    }

    /// Compute scale gradient for learned step size
    ///
    /// For Pi-quant: dalpha = sum(grad_out * q_int * pi / k)
    pub fn compute_alpha_grad(&self, weights: &[f32], grad_out: &[f32], channel: usize) -> f32 {
        let step = self.step_size(channel);
        let (min_q, max_q) = self.get_range();

        let mut grad_alpha = 0.0f32;
        let pi_over_k = PI / (self.k as f32);

        for (&w, &g) in weights.iter().zip(grad_out.iter()) {
            let q_int = (w / step).round().clamp(min_q as f32, max_q as f32);
            grad_alpha += g * q_int * pi_over_k;
        }

        // Normalize
        let normalizer = (weights.len() as f32 * self.num_levels() as f32).sqrt();
        grad_alpha / normalizer
    }
}

impl DifferentiableQuantizer for PiQuantDifferentiable {
    fn forward(&self, w: &[f32]) -> (Vec<i8>, Vec<f32>) {
        let mut q_int = Vec::with_capacity(w.len());
        let mut q_float = Vec::with_capacity(w.len());

        // Determine channel size for per-channel quantization
        let channel_size = match &self.granularity {
            QuantGranularity::PerChannel if self.num_channels > 1 => w.len() / self.num_channels,
            _ => w.len(),
        };

        for (i, &weight) in w.iter().enumerate() {
            let channel = if self.num_channels > 1 {
                i / channel_size
            } else {
                0
            };
            let (q, dequant) = self.quantize_scalar(weight, channel);
            q_int.push(q);
            q_float.push(dequant);
        }

        (q_int, q_float)
    }

    fn backward(&self, w: &[f32], q: &[f32], grad_out: &[f32]) -> Vec<f32> {
        let mut grad_w = vec![0.0f32; w.len()];
        self.ste_variant.backward_batch(w, q, grad_out, &mut grad_w);
        grad_w
    }

    fn dequantize(&self, q_int: &[i8]) -> Vec<f32> {
        let channel_size = if self.num_channels > 1 {
            q_int.len() / self.num_channels
        } else {
            q_int.len()
        };

        q_int
            .iter()
            .enumerate()
            .map(|(i, &q)| {
                let channel = if self.num_channels > 1 {
                    i / channel_size
                } else {
                    0
                };
                q as f32 * self.step_size(channel)
            })
            .collect()
    }

    fn scale(&self) -> f32 {
        // Return first alpha for compatibility
        self.alpha[0]
    }

    fn set_scale(&mut self, scale: f32) {
        // INV-2: Ensure positive
        assert!(scale > 0.0, "Scale must be positive (INV-2)");
        self.alpha[0] = scale;
    }

    fn bits(&self) -> u8 {
        self.bits
    }

    fn ste_variant(&self) -> &SteVariant {
        &self.ste_variant
    }
}

// ============================================================================
// Factory Function
// ============================================================================

/// Create a differentiable quantizer from QAT config
pub fn create_quantizer(config: &QatConfig) -> Box<dyn DifferentiableQuantizer> {
    if config.is_pi_quant() {
        Box::new(PiQuantDifferentiable::from_config(config))
    } else {
        Box::new(UniformQuantizer::from_config(config))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_quantizer_basic() {
        let quantizer = UniformQuantizer::new(4, SteVariant::Standard);

        let weights = vec![0.5, -0.5, 0.25, -0.25, 1.0, -1.0];
        let (q_int, q_float) = quantizer.forward(&weights);

        // Check that we got quantized values
        assert_eq!(q_int.len(), weights.len());
        assert_eq!(q_float.len(), weights.len());

        // Dequantized values should be close to original
        for (orig, dequant) in weights.iter().zip(q_float.iter()) {
            assert!((orig - dequant).abs() < 0.5, "Quantization error too large");
        }
    }

    #[test]
    fn test_uniform_quantizer_backward() {
        let quantizer = UniformQuantizer::new(4, SteVariant::Standard);

        let weights = vec![0.5, -0.5, 0.25, -0.25];
        let (_, q_float) = quantizer.forward(&weights);
        let grad_out = vec![1.0, 1.0, 1.0, 1.0];

        let grad_w = quantizer.backward(&weights, &q_float, &grad_out);

        // Standard STE: gradient should pass through unchanged
        assert_eq!(grad_w, grad_out);
    }

    #[test]
    fn test_uniform_quantizer_scale_init() {
        let mut quantizer = UniformQuantizer::new(4, SteVariant::Standard);

        let weights = vec![0.0; 100]
            .iter()
            .enumerate()
            .map(|(i, _)| (i as f32 - 50.0) / 10.0)
            .collect::<Vec<_>>();

        quantizer.init_scale_from_weights(&weights);

        // Scale should be set based on max value
        assert!(quantizer.scale() > 0.0);

        // Quantize and check range
        let (q_int, _) = quantizer.forward(&weights);
        let (min_q, max_q) = quantizer.get_range();

        for q in q_int {
            assert!(q >= min_q && q <= max_q, "Quantized value out of range");
        }
    }

    #[test]
    fn test_pi_quant_basic() {
        let quantizer = PiQuantDifferentiable::piq3();

        let weights = vec![0.5, -0.5, 0.25, -0.25, 1.0, -1.0];
        let (q_int, q_float) = quantizer.forward(&weights);

        assert_eq!(q_int.len(), weights.len());
        assert_eq!(q_float.len(), weights.len());
    }

    #[test]
    fn test_pi_quant_step_size() {
        // PiQ3: step = alpha * pi/4
        let quantizer = PiQuantDifferentiable::piq3();
        let expected_step = PI / 4.0; // alpha = 1.0 initially
        assert!((quantizer.step_size(0) - expected_step).abs() < 1e-6);

        // PiQ2: step = alpha * pi/3
        let quantizer = PiQuantDifferentiable::piq2();
        let expected_step = PI / 3.0;
        assert!((quantizer.step_size(0) - expected_step).abs() < 1e-6);
    }

    #[test]
    fn test_pi_quant_inv2_positive_scale() {
        let mut quantizer = PiQuantDifferentiable::piq3();

        // Setting positive scale should work
        quantizer.set_scale(0.5);
        assert_eq!(quantizer.scale(), 0.5);

        quantizer.set_alpha(0, 2.0);
        assert_eq!(quantizer.alphas()[0], 2.0);
    }

    #[test]
    #[should_panic(expected = "Alpha must be positive")]
    fn test_pi_quant_inv2_reject_negative() {
        let mut quantizer = PiQuantDifferentiable::piq3();
        quantizer.set_alpha(0, -1.0); // Should panic
    }

    #[test]
    fn test_pi_quant_inv3_step_constraint() {
        // INV-3: step = alpha * pi / k
        for k in [2u8, 3, 4, 5] {
            let quantizer = PiQuantDifferentiable::new(3, k);
            let alpha = 1.5;
            let mut q = quantizer;
            q.set_alpha(0, alpha);

            let expected = alpha * PI / (k as f32);
            let actual = q.step_size(0);
            assert!(
                (actual - expected).abs() < 1e-6,
                "INV-3 violation: step {} != alpha*pi/k {} for k={}",
                actual,
                expected,
                k
            );
        }
    }

    #[test]
    fn test_pi_quant_backward() {
        let quantizer = PiQuantDifferentiable::piq3();

        let weights = vec![0.5, -0.5, 0.25, -0.25];
        let (_, q_float) = quantizer.forward(&weights);
        let grad_out = vec![1.0, 1.0, 1.0, 1.0];

        let grad_w = quantizer.backward(&weights, &q_float, &grad_out);

        // Gradient should be computed via STE
        assert_eq!(grad_w.len(), weights.len());

        // With LearnedStepSize STE, gradient should pass through
        for g in grad_w {
            assert!((g - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_pi_quant_per_channel() {
        let mut quantizer = PiQuantDifferentiable::piq3();
        quantizer.init_per_channel(4);

        assert_eq!(quantizer.alphas().len(), 4);
        assert_eq!(quantizer.num_channels, 4);

        // Set different scales per channel
        for c in 0..4 {
            quantizer.set_alpha(c, (c + 1) as f32);
        }

        // Check step sizes are different
        for c in 0..4 {
            let expected_step = (c + 1) as f32 * PI / 4.0;
            assert!((quantizer.step_size(c) - expected_step).abs() < 1e-6);
        }
    }

    #[test]
    fn test_pi_quant_scale_init() {
        let mut quantizer = PiQuantDifferentiable::piq3();

        let weights: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 10.0).collect();

        quantizer.init_scale_from_weights(&weights, None);

        // Alpha should be positive (INV-2)
        assert!(quantizer.alphas()[0] > 0.0);
    }

    #[test]
    fn test_pi_quant_dequantize() {
        let quantizer = PiQuantDifferentiable::piq3();

        let q_int: Vec<i8> = vec![-4, -2, 0, 2, 3];
        let dequant = quantizer.dequantize(&q_int);

        let step = quantizer.step_size(0);
        for (q, d) in q_int.iter().zip(dequant.iter()) {
            let expected = *q as f32 * step;
            assert!((d - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_alpha_grad_computation() {
        let quantizer = PiQuantDifferentiable::piq3();

        let weights = vec![0.5, -0.5, 0.25, -0.25];
        let grad_out = vec![1.0, 1.0, 1.0, 1.0];

        let grad_alpha = quantizer.compute_alpha_grad(&weights, &grad_out, 0);

        // Gradient should be finite
        assert!(grad_alpha.is_finite());
    }

    #[test]
    fn test_create_quantizer_uniform() {
        let config = QatConfig::default().with_bits(4);
        let quantizer = create_quantizer(&config);

        assert_eq!(quantizer.bits(), 4);
        assert!(!config.is_pi_quant());
    }

    #[test]
    fn test_create_quantizer_pi() {
        let config = QatConfig::piq3();
        let quantizer = create_quantizer(&config);

        assert_eq!(quantizer.bits(), 3);
        assert!(config.is_pi_quant());
    }

    #[test]
    fn test_quantization_roundtrip() {
        let quantizer = UniformQuantizer::new(4, SteVariant::Standard);

        let weights = vec![0.5, -0.5, 0.0, 1.0, -1.0];
        let (q_int, _) = quantizer.forward(&weights);
        let dequant = quantizer.dequantize(&q_int);

        // Dequantized should be close to quantized values from forward
        let (_, q_float) = quantizer.forward(&weights);
        for (a, b) in dequant.iter().zip(q_float.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ewgs_backward() {
        let quantizer = UniformQuantizer::new(4, SteVariant::Ewgs { lambda: 0.1 });

        let weights = vec![0.5, 0.5, 0.5, 0.5];
        let (_, q_float) = quantizer.forward(&weights);
        let grad_out = vec![1.0, 1.0, 1.0, 1.0];

        let grad_w = quantizer.backward(&weights, &q_float, &grad_out);

        // EWGS should scale gradients based on quantization error
        for g in grad_w {
            assert!(g >= 1.0, "EWGS gradient should be >= 1.0");
        }
    }

    #[test]
    fn test_clipped_backward() {
        let quantizer = UniformQuantizer::new(4, SteVariant::Clipped { clip_val: 0.5 });

        let weights = vec![0.3, -0.3, 0.8, -0.8]; // Last two outside clip range
        let (_, q_float) = quantizer.forward(&weights);
        let grad_out = vec![1.0, 1.0, 1.0, 1.0];

        let grad_w = quantizer.backward(&weights, &q_float, &grad_out);

        // Inside clip range
        assert_eq!(grad_w[0], 1.0);
        assert_eq!(grad_w[1], 1.0);

        // Outside clip range
        assert_eq!(grad_w[2], 0.0);
        assert_eq!(grad_w[3], 0.0);
    }

    #[test]
    fn test_num_levels() {
        assert_eq!(
            UniformQuantizer::new(2, SteVariant::Standard).num_levels(),
            4
        );
        assert_eq!(
            UniformQuantizer::new(3, SteVariant::Standard).num_levels(),
            8
        );
        assert_eq!(
            UniformQuantizer::new(4, SteVariant::Standard).num_levels(),
            16
        );
        assert_eq!(PiQuantDifferentiable::piq3().num_levels(), 8);
        assert_eq!(PiQuantDifferentiable::piq2().num_levels(), 4);
    }
}
