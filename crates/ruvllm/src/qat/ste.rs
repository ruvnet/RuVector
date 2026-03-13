//! Straight-Through Estimator (STE) Module
//!
//! This module implements the backward pass for quantization-aware training.
//! During training, the forward pass uses discrete quantized values, but gradients
//! must flow through the non-differentiable quantization operation.
//!
//! ## System Invariant
//!
//! **INV-1: STE Gradient Flow** - Gradient passes through quantization via STE;
//! no zero-gradient regions except explicit clipping.
//!
//! ## STE Variants
//!
//! | Variant | Backward Formula | Use Case |
//! |---------|------------------|----------|
//! | Standard | grad_out | Default, identity pass-through |
//! | Clipped | grad_out if |w| <= c, else 0 | Prevents gradient explosion |
//! | LearnedStepSize | grad_out (scale grad separate) | Adaptive step learning |
//! | EWGS | grad_out * (1 + lambda * |w-q|) | Better convergence |
//!
//! ## References
//!
//! - Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons"
//! - Esser et al., "Learned Step Size Quantization" (LSQ)
//! - Lee et al., "Element-Wise Gradient Scaling" (EWGS)

use super::config::SteVariant;

// ============================================================================
// STE Backward Implementation
// ============================================================================

impl SteVariant {
    /// Compute the backward pass gradient for a single weight
    ///
    /// # Arguments
    ///
    /// * `w` - Latent (full-precision) weight value
    /// * `q` - Quantized weight value (after forward quantization)
    /// * `grad_out` - Upstream gradient (dL/dq)
    ///
    /// # Returns
    ///
    /// Gradient with respect to latent weight (dL/dw)
    ///
    /// # System Invariant
    ///
    /// INV-1: Non-zero gradient everywhere except explicit clipping boundaries
    #[inline]
    pub fn backward(&self, w: f32, q: f32, grad_out: f32) -> f32 {
        match self {
            // Standard STE: Identity pass-through
            // dL/dw = dL/dq (gradient flows unchanged)
            SteVariant::Standard => grad_out,

            // Clipped STE: Zero gradient outside clip range
            // dL/dw = dL/dq * 1{|w| <= clip_val}
            SteVariant::Clipped { clip_val } => {
                if w.abs() <= *clip_val {
                    grad_out
                } else {
                    0.0
                }
            }

            // Learned Step Size: Identity for weight gradient
            // Scale gradient is computed separately via compute_scale_grad
            SteVariant::LearnedStepSize => grad_out,

            // EWGS: Gradient scaled by quantization error
            // dL/dw = dL/dq * (1 + lambda * |w - q|)
            // This gives stronger gradient signal for weights far from quantization points
            SteVariant::Ewgs { lambda } => grad_out * (1.0 + lambda * (w - q).abs()),
        }
    }

    /// Compute backward pass for a batch of weights
    ///
    /// # Arguments
    ///
    /// * `weights` - Latent weight values
    /// * `quantized` - Quantized weight values
    /// * `grad_out` - Upstream gradients
    /// * `grad_w` - Output buffer for weight gradients
    #[inline]
    pub fn backward_batch(
        &self,
        weights: &[f32],
        quantized: &[f32],
        grad_out: &[f32],
        grad_w: &mut [f32],
    ) {
        debug_assert_eq!(weights.len(), quantized.len());
        debug_assert_eq!(weights.len(), grad_out.len());
        debug_assert_eq!(weights.len(), grad_w.len());

        for i in 0..weights.len() {
            grad_w[i] = self.backward(weights[i], quantized[i], grad_out[i]);
        }
    }

    /// Compute scale gradient for Learned Step Size quantization
    ///
    /// For LSQ, the scale (step size) is learned alongside weights.
    /// The gradient w.r.t. scale s is:
    ///
    /// dL/ds = sum_i (dL/dq_i * (q_int_i - grad_clip))
    ///
    /// where q_int_i is the integer quantization index.
    ///
    /// # Arguments
    ///
    /// * `weights` - Latent weight values
    /// * `scale` - Current scale value
    /// * `grad_out` - Upstream gradients
    /// * `num_levels` - Number of quantization levels (2^bits)
    ///
    /// # Returns
    ///
    /// Gradient with respect to scale
    pub fn compute_scale_grad(
        weights: &[f32],
        scale: f32,
        grad_out: &[f32],
        num_levels: usize,
    ) -> f32 {
        if scale == 0.0 {
            return 0.0;
        }

        let half = (num_levels / 2) as f32;
        let min_q = -half;
        let max_q = half - 1.0;

        let mut grad_scale = 0.0f32;

        for (&w, &g) in weights.iter().zip(grad_out.iter()) {
            // Integer quantization index (clamped)
            let q_int = (w / scale).round().clamp(min_q, max_q);

            // Gradient contribution: g * (q_int - w/s)
            // This pushes scale toward values that minimize quantization error
            grad_scale += g * q_int;
        }

        // Normalize by sqrt(n * num_levels) per LSQ paper
        let normalizer = (weights.len() as f32 * num_levels as f32).sqrt();
        grad_scale / normalizer
    }
}

// ============================================================================
// SIMD-Optimized Backward Pass (Future optimization)
// ============================================================================

/// SIMD-optimized STE backward for Standard variant
#[cfg(target_arch = "aarch64")]
pub mod simd {
    /// NEON-accelerated backward pass (identity, no-op for Standard STE)
    #[inline]
    pub unsafe fn backward_standard_neon(grad_out: &[f32], grad_w: &mut [f32]) {
        // For Standard STE, just copy
        grad_w.copy_from_slice(grad_out);
    }

    /// NEON-accelerated EWGS backward pass
    #[inline]
    pub unsafe fn backward_ewgs_neon(
        weights: &[f32],
        quantized: &[f32],
        grad_out: &[f32],
        grad_w: &mut [f32],
        lambda: f32,
    ) {
        use std::arch::aarch64::*;

        let n = weights.len();
        let lambda_v = vdupq_n_f32(lambda);
        let one_v = vdupq_n_f32(1.0);

        let mut i = 0;
        while i + 4 <= n {
            // Load vectors
            let w = vld1q_f32(weights.as_ptr().add(i));
            let q = vld1q_f32(quantized.as_ptr().add(i));
            let g = vld1q_f32(grad_out.as_ptr().add(i));

            // |w - q|
            let diff = vabsq_f32(vsubq_f32(w, q));

            // 1 + lambda * |w - q|
            let scale = vaddq_f32(one_v, vmulq_f32(lambda_v, diff));

            // g * scale
            let result = vmulq_f32(g, scale);

            vst1q_f32(grad_w.as_mut_ptr().add(i), result);
            i += 4;
        }

        // Handle remainder
        while i < n {
            grad_w[i] = grad_out[i] * (1.0 + lambda * (weights[i] - quantized[i]).abs());
            i += 1;
        }
    }

    /// NEON-accelerated Clipped STE backward pass
    #[inline]
    pub unsafe fn backward_clipped_neon(
        weights: &[f32],
        grad_out: &[f32],
        grad_w: &mut [f32],
        clip_val: f32,
    ) {
        use std::arch::aarch64::*;

        let n = weights.len();
        let clip_v = vdupq_n_f32(clip_val);
        let neg_clip_v = vnegq_f32(clip_v);
        let zero_v = vdupq_n_f32(0.0);

        let mut i = 0;
        while i + 4 <= n {
            let w = vld1q_f32(weights.as_ptr().add(i));
            let g = vld1q_f32(grad_out.as_ptr().add(i));

            // Create mask: w <= clip_val && w >= -clip_val
            let le_clip = vcleq_f32(w, clip_v);
            let ge_neg_clip = vcgeq_f32(w, neg_clip_v);
            let mask = vandq_u32(le_clip, ge_neg_clip);

            // Select g if in range, else 0
            let result = vbslq_f32(mask, g, zero_v);

            vst1q_f32(grad_w.as_mut_ptr().add(i), result);
            i += 4;
        }

        // Handle remainder
        while i < n {
            grad_w[i] = if weights[i].abs() <= clip_val {
                grad_out[i]
            } else {
                0.0
            };
            i += 1;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_ste_backward() {
        let ste = SteVariant::Standard;

        // Standard STE should pass gradient through unchanged
        assert_eq!(ste.backward(0.5, 0.4, 1.0), 1.0);
        assert_eq!(ste.backward(-0.3, -0.2, 0.5), 0.5);
        assert_eq!(ste.backward(1.0, 1.0, -0.5), -0.5);

        // Zero gradient should remain zero
        assert_eq!(ste.backward(0.5, 0.4, 0.0), 0.0);
    }

    #[test]
    fn test_clipped_ste_backward() {
        let ste = SteVariant::Clipped { clip_val: 1.0 };

        // Inside clip range: gradient passes through
        assert_eq!(ste.backward(0.5, 0.4, 1.0), 1.0);
        assert_eq!(ste.backward(-0.5, -0.4, 0.5), 0.5);
        assert_eq!(ste.backward(1.0, 1.0, 0.3), 0.3);

        // Outside clip range: gradient is zero
        assert_eq!(ste.backward(1.5, 1.0, 1.0), 0.0);
        assert_eq!(ste.backward(-1.5, -1.0, 0.5), 0.0);

        // Edge case: exactly at boundary
        assert_eq!(ste.backward(1.0, 1.0, 1.0), 1.0);
        assert_eq!(ste.backward(-1.0, -1.0, 1.0), 1.0);
    }

    #[test]
    fn test_learned_step_size_backward() {
        let ste = SteVariant::LearnedStepSize;

        // LSQ weight gradient is identity (same as Standard)
        assert_eq!(ste.backward(0.5, 0.4, 1.0), 1.0);
        assert_eq!(ste.backward(-0.3, -0.2, 0.5), 0.5);
    }

    #[test]
    fn test_ewgs_backward() {
        let ste = SteVariant::Ewgs { lambda: 0.1 };

        // When w == q, gradient is unchanged
        let grad = ste.backward(0.5, 0.5, 1.0);
        assert!((grad - 1.0).abs() < 1e-6);

        // When w != q, gradient is scaled up
        let grad = ste.backward(0.5, 0.3, 1.0);
        // Expected: 1.0 * (1 + 0.1 * |0.5 - 0.3|) = 1.0 * 1.02 = 1.02
        assert!((grad - 1.02).abs() < 1e-6);

        // Larger error -> larger gradient
        let grad_small_error = ste.backward(0.5, 0.4, 1.0);
        let grad_large_error = ste.backward(0.5, 0.1, 1.0);
        assert!(grad_large_error > grad_small_error);
    }

    #[test]
    fn test_ewgs_gradient_scaling() {
        let ste = SteVariant::Ewgs { lambda: 1.0 };

        // With lambda=1.0, gradient scaling is more aggressive
        let grad = ste.backward(1.0, 0.0, 1.0);
        // Expected: 1.0 * (1 + 1.0 * 1.0) = 2.0
        assert!((grad - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_batch() {
        let ste = SteVariant::Standard;

        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let quantized = vec![0.0, 0.25, 0.25, 0.5];
        let grad_out = vec![1.0, 2.0, 3.0, 4.0];
        let mut grad_w = vec![0.0; 4];

        ste.backward_batch(&weights, &quantized, &grad_out, &mut grad_w);

        assert_eq!(grad_w, grad_out);
    }

    #[test]
    fn test_backward_batch_ewgs() {
        let ste = SteVariant::Ewgs { lambda: 0.5 };

        let weights = vec![0.5, 0.5, 0.5, 0.5];
        let quantized = vec![0.5, 0.4, 0.3, 0.2];
        let grad_out = vec![1.0, 1.0, 1.0, 1.0];
        let mut grad_w = vec![0.0; 4];

        ste.backward_batch(&weights, &quantized, &grad_out, &mut grad_w);

        // First element: w == q, so grad should be 1.0
        assert!((grad_w[0] - 1.0).abs() < 1e-6);

        // Other elements should have larger gradients
        for i in 1..4 {
            assert!(grad_w[i] > 1.0);
        }

        // Gradients should increase with error
        assert!(grad_w[1] < grad_w[2]);
        assert!(grad_w[2] < grad_w[3]);
    }

    #[test]
    fn test_compute_scale_grad() {
        let weights = vec![0.5, -0.5, 0.25, -0.25];
        let scale = 0.25; // So weights quantize to 2, -2, 1, -1
        let grad_out = vec![1.0, 1.0, 1.0, 1.0];
        let num_levels = 16; // 4-bit

        let grad_scale = SteVariant::compute_scale_grad(&weights, scale, &grad_out, num_levels);

        // Scale gradient should be well-defined
        assert!(grad_scale.is_finite());
    }

    #[test]
    fn test_scale_grad_zero_scale() {
        let weights = vec![0.5, -0.5];
        let scale = 0.0;
        let grad_out = vec![1.0, 1.0];
        let num_levels = 16;

        let grad_scale = SteVariant::compute_scale_grad(&weights, scale, &grad_out, num_levels);
        assert_eq!(grad_scale, 0.0);
    }

    #[test]
    fn test_inv1_no_zero_regions_standard() {
        // INV-1: Standard STE should never produce zero gradient (except when grad_out is zero)
        let ste = SteVariant::Standard;

        for w in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            for q in [-1.0, 0.0, 1.0] {
                let grad = ste.backward(w, q, 1.0);
                assert_eq!(grad, 1.0, "Standard STE should always pass through");
            }
        }
    }

    #[test]
    fn test_inv1_clipped_only_outside() {
        // INV-1: Clipped STE only produces zero gradient outside explicit range
        let ste = SteVariant::Clipped { clip_val: 1.0 };

        // Inside range: always passes gradient
        for w in [-0.9, -0.5, 0.0, 0.5, 0.9] {
            let grad = ste.backward(w, 0.0, 1.0);
            assert_eq!(grad, 1.0, "Clipped STE should pass gradient inside range");
        }

        // Outside range: zero gradient (explicit clipping)
        for w in [-2.0, -1.5, 1.5, 2.0] {
            let grad = ste.backward(w, 0.0, 1.0);
            assert_eq!(grad, 0.0, "Clipped STE should zero gradient outside range");
        }
    }

    #[test]
    fn test_gradient_correctness_vs_reference() {
        // Reference implementations (from PyTorch / JAX literature)

        // Standard STE
        let ste_std = SteVariant::Standard;
        assert_eq!(ste_std.backward(0.7, 0.5, 0.3), 0.3);

        // Clipped STE with clip_val=1
        let ste_clip = SteVariant::Clipped { clip_val: 1.0 };
        assert_eq!(ste_clip.backward(0.7, 0.5, 0.3), 0.3); // Inside
        assert_eq!(ste_clip.backward(1.2, 1.0, 0.3), 0.0); // Outside

        // EWGS with lambda=0.1
        let ste_ewgs = SteVariant::Ewgs { lambda: 0.1 };
        let expected = 0.3_f32 * (1.0_f32 + 0.1_f32 * (0.7_f32 - 0.5_f32).abs());
        let actual = ste_ewgs.backward(0.7, 0.5, 0.3);
        assert!(
            (actual - expected).abs() < 1e-6,
            "EWGS mismatch: {} vs {}",
            actual,
            expected
        );
    }

    #[test]
    fn test_requires_scale_grad() {
        assert!(!SteVariant::Standard.requires_scale_grad());
        assert!(!SteVariant::Clipped { clip_val: 1.0 }.requires_scale_grad());
        assert!(SteVariant::LearnedStepSize.requires_scale_grad());
        assert!(!SteVariant::Ewgs { lambda: 0.1 }.requires_scale_grad());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_simd_ewgs_equivalence() {
        use super::simd::backward_ewgs_neon;

        let weights: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let quantized: Vec<f32> = weights.iter().map(|w| (w * 4.0).round() / 4.0).collect();
        let grad_out: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();

        // Scalar reference
        let ste = SteVariant::Ewgs { lambda: 0.1 };
        let mut grad_scalar = vec![0.0f32; 100];
        ste.backward_batch(&weights, &quantized, &grad_out, &mut grad_scalar);

        // SIMD implementation
        let mut grad_simd = vec![0.0f32; 100];
        unsafe {
            backward_ewgs_neon(&weights, &quantized, &grad_out, &mut grad_simd, 0.1);
        }

        // Compare (should be within 1 ULP per INV-8)
        for i in 0..100 {
            let diff = (grad_scalar[i] - grad_simd[i]).abs();
            let ulp = f32::EPSILON * grad_scalar[i].abs().max(1.0);
            assert!(
                diff <= ulp,
                "SIMD mismatch at {}: {} vs {} (diff {})",
                i,
                grad_scalar[i],
                grad_simd[i],
                diff
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_simd_clipped_equivalence() {
        use super::simd::backward_clipped_neon;

        let weights: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 25.0).collect();
        let grad_out: Vec<f32> = vec![1.0; 100];

        // Scalar reference
        let ste = SteVariant::Clipped { clip_val: 1.0 };
        let mut grad_scalar = vec![0.0f32; 100];
        let quantized = vec![0.0f32; 100]; // Dummy, not used for clipped
        ste.backward_batch(&weights, &quantized, &grad_out, &mut grad_scalar);

        // SIMD implementation
        let mut grad_simd = vec![0.0f32; 100];
        unsafe {
            backward_clipped_neon(&weights, &grad_out, &mut grad_simd, 1.0);
        }

        // Compare
        for i in 0..100 {
            assert_eq!(
                grad_scalar[i], grad_simd[i],
                "Clipped SIMD mismatch at {}: {} vs {}",
                i, grad_scalar[i], grad_simd[i]
            );
        }
    }
}
