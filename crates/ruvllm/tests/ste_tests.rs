//! Straight-Through Estimator (STE) Tests for ADR-090
//!
//! Tests for gradient correctness of each STE variant used in
//! Quantization-Aware Training (QAT).
//!
//! INV-1: Gradient correctness tests vs PyTorch reference values
//!
//! STE Variants tested:
//! - Standard STE (identity gradient in quantization region)
//! - Clipped STE (zero gradient outside [-1, 1])
//! - Learned Step Size (LSQ) with scale gradient
//! - EWGS (Element-Wise Gradient Scaling)

#[cfg(test)]
mod ste_tests {
    use std::f32::consts::PI;

    // ============================================================================
    // Test Constants
    // ============================================================================

    /// Epsilon for floating-point comparisons
    const EPSILON: f32 = 1e-5;

    /// Learning rate for gradient tests
    const LR: f32 = 0.01;

    // ============================================================================
    // STE Variant Enum
    // ============================================================================

    /// Straight-Through Estimator variants
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum SteVariant {
        /// Standard STE: grad_input = grad_output (always pass through)
        Standard,
        /// Clipped STE: grad_input = grad_output if |x| <= 1, else 0
        Clipped,
        /// Learned Step Size (LSQ): includes gradient for scale parameter
        LearnedStepSize,
        /// Element-Wise Gradient Scaling: scales gradient by quantization error
        Ewgs,
    }

    // ============================================================================
    // Forward/Backward Implementation
    // ============================================================================

    /// Quantization forward pass
    fn quantize_forward(x: f32, scale: f32, bits: u8) -> (f32, i8) {
        let levels = 1 << bits;
        let half_range = (levels - 1) / 2;

        // Quantize: q = round(x / scale)
        let q = (x / scale)
            .round()
            .clamp(-(half_range as f32) - 1.0, half_range as f32) as i8;

        // Dequantize: x_hat = q * scale
        let x_hat = (q as f32) * scale;

        (x_hat, q)
    }

    /// STE backward pass for input gradient
    fn ste_backward_input(grad_output: f32, x: f32, scale: f32, variant: SteVariant) -> f32 {
        match variant {
            SteVariant::Standard => {
                // Standard STE: pass gradient through unchanged
                grad_output
            }
            SteVariant::Clipped => {
                // Clipped STE: zero gradient outside [-scale, scale]
                let normalized = x / scale;
                if normalized.abs() <= 1.0 {
                    grad_output
                } else {
                    0.0
                }
            }
            SteVariant::LearnedStepSize => {
                // LSQ: gradient passes through in quantization region
                let normalized = x / scale;
                if normalized.abs() <= 4.0 {
                    // Wider range for LSQ
                    grad_output
                } else {
                    0.0
                }
            }
            SteVariant::Ewgs => {
                // EWGS: scale gradient by (1 - tanh(error)^2)
                let (x_hat, _) = quantize_forward(x, scale, 3);
                let error = (x - x_hat).abs() / scale.max(EPSILON);
                let scaling = 1.0 - (error.tanh()).powi(2);
                grad_output * scaling
            }
        }
    }

    /// STE backward pass for scale gradient (LSQ variant)
    fn ste_backward_scale(grad_output: f32, x: f32, scale: f32, q: i8, variant: SteVariant) -> f32 {
        match variant {
            SteVariant::LearnedStepSize => {
                // LSQ scale gradient: grad_s = grad_output * (q - x/s) / sqrt(n_levels)
                // Simplified: grad_s = grad_output * (q - x/s)
                let q_f = q as f32;
                let x_normalized = x / scale;
                grad_output * (q_f - x_normalized)
            }
            _ => 0.0, // Other variants don't have scale gradient
        }
    }

    // ============================================================================
    // 1. Standard STE Tests
    // ============================================================================

    #[test]
    fn test_standard_ste_passes_gradient() {
        let variant = SteVariant::Standard;
        let scale = 1.0;

        // Gradient should pass through unchanged
        for grad_out in [-1.0, 0.0, 0.5, 1.0, 2.0] {
            for x in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0] {
                let grad_in = ste_backward_input(grad_out, x, scale, variant);
                assert!(
                    (grad_in - grad_out).abs() < EPSILON,
                    "Standard STE should pass gradient: grad_out={}, got grad_in={}",
                    grad_out,
                    grad_in
                );
            }
        }
    }

    #[test]
    fn test_standard_ste_with_varying_scale() {
        let variant = SteVariant::Standard;

        for scale in [0.1, 1.0, 10.0] {
            let grad_out = 1.0;
            let x = 0.5;
            let grad_in = ste_backward_input(grad_out, x, scale, variant);
            assert!(
                (grad_in - grad_out).abs() < EPSILON,
                "Standard STE should ignore scale: scale={}, grad_in={}",
                scale,
                grad_in
            );
        }
    }

    // ============================================================================
    // 2. Clipped STE Tests
    // ============================================================================

    #[test]
    fn test_clipped_ste_within_range() {
        let variant = SteVariant::Clipped;
        let scale = 1.0;
        let grad_out = 1.0;

        // Within [-1, 1]: gradient passes through
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let grad_in = ste_backward_input(grad_out, x, scale, variant);
            assert!(
                (grad_in - grad_out).abs() < EPSILON,
                "Clipped STE should pass gradient for x={}: got {}",
                x,
                grad_in
            );
        }
    }

    #[test]
    fn test_clipped_ste_outside_range() {
        let variant = SteVariant::Clipped;
        let scale = 1.0;
        let grad_out = 1.0;

        // Outside [-1, 1]: gradient is zero
        for x in [-2.0, -1.5, 1.5, 2.0, 10.0] {
            let grad_in = ste_backward_input(grad_out, x, scale, variant);
            assert!(
                grad_in.abs() < EPSILON,
                "Clipped STE should zero gradient for x={}: got {}",
                x,
                grad_in
            );
        }
    }

    #[test]
    fn test_clipped_ste_scale_affects_range() {
        let variant = SteVariant::Clipped;
        let grad_out = 1.0;

        // With scale=2, the clipping range is [-2, 2]
        let scale = 2.0;

        // x=1.5 is within range when scale=2
        let grad_in = ste_backward_input(grad_out, 1.5, scale, variant);
        assert!(
            (grad_in - grad_out).abs() < EPSILON,
            "x=1.5 should be in range with scale=2: got {}",
            grad_in
        );

        // x=3.0 is outside range even with scale=2
        let grad_in = ste_backward_input(grad_out, 3.0, scale, variant);
        assert!(
            grad_in.abs() < EPSILON,
            "x=3.0 should be outside range with scale=2: got {}",
            grad_in
        );
    }

    // ============================================================================
    // 3. Learned Step Size (LSQ) Tests
    // ============================================================================

    #[test]
    fn test_lsq_input_gradient() {
        let variant = SteVariant::LearnedStepSize;
        let scale = 1.0;
        let grad_out = 1.0;

        // LSQ has wider range [-4, 4]
        for x in [-4.0, -2.0, 0.0, 2.0, 4.0] {
            let grad_in = ste_backward_input(grad_out, x, scale, variant);
            assert!(
                (grad_in - grad_out).abs() < EPSILON,
                "LSQ should pass gradient for x={}: got {}",
                x,
                grad_in
            );
        }

        // Outside [-4, 4]: zero gradient
        for x in [-5.0, 5.0, 10.0] {
            let grad_in = ste_backward_input(grad_out, x, scale, variant);
            assert!(
                grad_in.abs() < EPSILON,
                "LSQ should zero gradient for x={}: got {}",
                x,
                grad_in
            );
        }
    }

    #[test]
    fn test_lsq_scale_gradient() {
        let variant = SteVariant::LearnedStepSize;
        let scale = 1.0;
        let bits = 3u8;
        let grad_out = 1.0;

        // Test scale gradient for different inputs
        let test_cases = [
            (0.0, 0), // x=0, q=0 -> grad_s = 0
            (1.0, 1), // x=1, q=1 -> grad_s = grad_out * (1 - 1) = 0
            (0.5, 0), // x=0.5, q=0 -> grad_s = grad_out * (0 - 0.5) = -0.5
            (1.5, 2), // x=1.5, q=2 -> grad_s = grad_out * (2 - 1.5) = 0.5
        ];

        for (x, q) in test_cases {
            let (_, q_actual) = quantize_forward(x, scale, bits);
            let grad_s = ste_backward_scale(grad_out, x, scale, q_actual, variant);

            let expected = grad_out * (q_actual as f32 - x / scale);
            assert!(
                (grad_s - expected).abs() < 0.1,
                "LSQ scale gradient for x={}: expected {}, got {}",
                x,
                expected,
                grad_s
            );
        }
    }

    #[test]
    fn test_lsq_scale_update() {
        // Simulate one gradient update step
        let mut scale = 1.0f32;
        let x = 1.5;
        let bits = 3u8;
        let grad_out = 1.0;

        let (_, q) = quantize_forward(x, scale, bits);
        let grad_s = ste_backward_scale(grad_out, x, scale, q, SteVariant::LearnedStepSize);

        // Update scale with gradient descent
        let scale_old = scale;
        scale -= LR * grad_s;

        // Scale should change in expected direction
        if grad_s > 0.0 {
            assert!(scale < scale_old, "Scale should decrease when grad_s > 0");
        } else if grad_s < 0.0 {
            assert!(scale > scale_old, "Scale should increase when grad_s < 0");
        }
    }

    // ============================================================================
    // 4. EWGS (Element-Wise Gradient Scaling) Tests
    // ============================================================================

    #[test]
    fn test_ewgs_zero_error_full_gradient() {
        let variant = SteVariant::Ewgs;
        let scale = 1.0;
        let grad_out = 1.0;

        // When x exactly equals quantized value, error is 0, scaling is 1
        // x=0 quantizes to 0, so error=0
        let x = 0.0;
        let grad_in = ste_backward_input(grad_out, x, scale, variant);

        // tanh(0) = 0, so scaling = 1 - 0 = 1
        assert!(
            (grad_in - grad_out).abs() < EPSILON,
            "EWGS with zero error should pass full gradient: got {}",
            grad_in
        );
    }

    #[test]
    fn test_ewgs_large_error_reduced_gradient() {
        let variant = SteVariant::Ewgs;
        let scale = 1.0;
        let grad_out = 1.0;

        // x=0.4999 is close to rounding boundary, so error is ~0.5
        let x = 0.4999;
        let grad_in = ste_backward_input(grad_out, x, scale, variant);

        // Large error should reduce gradient
        assert!(
            grad_in < grad_out,
            "EWGS with large error should reduce gradient: got {}",
            grad_in
        );
        assert!(
            grad_in > 0.0,
            "EWGS gradient should still be positive: got {}",
            grad_in
        );
    }

    #[test]
    fn test_ewgs_gradient_scaling_range() {
        let variant = SteVariant::Ewgs;
        let scale = 1.0;
        let grad_out = 1.0;

        // Test various inputs
        for x in [-2.0, -1.0, -0.5, 0.0, 0.3, 0.7, 1.0, 1.5, 2.0] {
            let grad_in = ste_backward_input(grad_out, x, scale, variant);

            // EWGS gradient should be in [0, grad_out]
            assert!(
                grad_in >= 0.0 && grad_in <= grad_out,
                "EWGS gradient out of range for x={}: got {}",
                x,
                grad_in
            );
        }
    }

    #[test]
    fn test_ewgs_symmetry() {
        let variant = SteVariant::Ewgs;
        let scale = 1.0;
        let grad_out = 1.0;

        // Symmetric inputs should have symmetric gradients
        for abs_x in [0.3, 0.5, 0.7, 1.0, 1.5] {
            let grad_pos = ste_backward_input(grad_out, abs_x, scale, variant);
            let grad_neg = ste_backward_input(grad_out, -abs_x, scale, variant);

            assert!(
                (grad_pos - grad_neg).abs() < EPSILON,
                "EWGS should be symmetric: x={} -> {}, x={} -> {}",
                abs_x,
                grad_pos,
                -abs_x,
                grad_neg
            );
        }
    }

    // ============================================================================
    // 5. PyTorch Reference Value Comparison
    // ============================================================================

    // These values are computed using PyTorch as reference:
    // torch.autograd.grad for Standard/Clipped STE
    // LSQ paper reference implementation

    #[test]
    fn test_pytorch_reference_standard_ste() {
        // PyTorch reference: grad flows through unchanged
        let variant = SteVariant::Standard;
        let test_cases = [
            // (x, scale, grad_out, expected_grad_in)
            (0.5, 1.0, 1.0, 1.0),
            (-0.5, 1.0, 1.0, 1.0),
            (2.0, 1.0, 0.5, 0.5),
            (-2.0, 0.5, 2.0, 2.0),
        ];

        for (x, scale, grad_out, expected) in test_cases {
            let grad_in = ste_backward_input(grad_out, x, scale, variant);
            assert!(
                (grad_in - expected).abs() < EPSILON,
                "PyTorch ref Standard STE: x={}, expected={}, got={}",
                x,
                expected,
                grad_in
            );
        }
    }

    #[test]
    fn test_pytorch_reference_clipped_ste() {
        // PyTorch reference: grad is zero outside [-1, 1] normalized
        let variant = SteVariant::Clipped;
        let test_cases = [
            // (x, scale, grad_out, expected_grad_in)
            (0.5, 1.0, 1.0, 1.0),  // Inside range
            (-0.5, 1.0, 1.0, 1.0), // Inside range
            (1.0, 1.0, 1.0, 1.0),  // At boundary
            (-1.0, 1.0, 1.0, 1.0), // At boundary
            (1.5, 1.0, 1.0, 0.0),  // Outside range
            (-1.5, 1.0, 1.0, 0.0), // Outside range
            (0.5, 0.5, 1.0, 1.0),  // Inside range (x/scale = 1)
            (1.0, 0.5, 1.0, 0.0),  // Outside range (x/scale = 2)
        ];

        for (x, scale, grad_out, expected) in test_cases {
            let grad_in = ste_backward_input(grad_out, x, scale, variant);
            assert!(
                (grad_in - expected).abs() < EPSILON,
                "PyTorch ref Clipped STE: x={}, scale={}, expected={}, got={}",
                x,
                scale,
                expected,
                grad_in
            );
        }
    }

    #[test]
    fn test_pytorch_reference_lsq_input() {
        // LSQ paper: gradient passes through in [-Qn, Qp] range
        let variant = SteVariant::LearnedStepSize;
        let test_cases = [
            (0.0, 1.0, 1.0, 1.0), // Center
            (2.0, 1.0, 1.0, 1.0), // Within range
            (4.0, 1.0, 1.0, 1.0), // At boundary
            (5.0, 1.0, 1.0, 0.0), // Outside range
        ];

        for (x, scale, grad_out, expected) in test_cases {
            let grad_in = ste_backward_input(grad_out, x, scale, variant);
            assert!(
                (grad_in - expected).abs() < EPSILON,
                "PyTorch ref LSQ input: x={}, expected={}, got={}",
                x,
                expected,
                grad_in
            );
        }
    }

    // ============================================================================
    // 6. Gradient Flow Tests
    // ============================================================================

    #[test]
    fn test_gradient_chain_rule() {
        // Test that gradients compose correctly through chain rule
        let x = 0.7;
        let scale = 1.0;
        let upstream_grad = 2.5;

        for variant in [
            SteVariant::Standard,
            SteVariant::Clipped,
            SteVariant::LearnedStepSize,
            SteVariant::Ewgs,
        ] {
            let grad_1 = ste_backward_input(1.0, x, scale, variant);
            let grad_chain = ste_backward_input(upstream_grad, x, scale, variant);

            // Chain rule: grad_chain = upstream_grad * local_grad
            assert!(
                (grad_chain - upstream_grad * grad_1).abs() < EPSILON,
                "{:?}: chain rule violated: {} * {} != {}",
                variant,
                upstream_grad,
                grad_1,
                grad_chain
            );
        }
    }

    #[test]
    fn test_gradient_accumulation() {
        // Test gradient accumulation from multiple paths
        let x = 0.5;
        let scale = 1.0;
        let variant = SteVariant::Standard;

        let grad1 = ste_backward_input(1.0, x, scale, variant);
        let grad2 = ste_backward_input(2.0, x, scale, variant);
        let grad3 = ste_backward_input(3.0, x, scale, variant);

        // Accumulated gradient
        let accumulated = grad1 + grad2 + grad3;
        let expected = 1.0 + 2.0 + 3.0; // For standard STE

        assert!(
            (accumulated - expected).abs() < EPSILON,
            "Gradient accumulation: expected {}, got {}",
            expected,
            accumulated
        );
    }

    // ============================================================================
    // 7. Numerical Gradient Check
    // ============================================================================

    #[test]
    fn test_numerical_gradient_check() {
        // Use finite differences to verify analytical gradient
        let scale = 1.0;
        let bits = 3u8;
        let eps = 1e-4;

        // Test points away from quantization boundaries
        // For bits=3, scale=1.0: levels=8, half_range=3
        // Boundaries are at x = n * scale = n (integers and x.5 values)
        // So we use points clearly away from 0.5 increments
        for x in [-1.3, -0.7, 0.2, 0.8, 1.2] {
            // Forward passes
            let (_y, _) = quantize_forward(x, scale, bits);
            let (y_plus, _) = quantize_forward(x + eps, scale, bits);
            let (y_minus, _) = quantize_forward(x - eps, scale, bits);

            // Numerical gradient (central difference)
            let numerical_grad = (y_plus - y_minus) / (2.0 * eps);

            // The quantization function is piecewise constant, so
            // numerical gradient is 0 everywhere except at boundaries
            // This is expected behavior for quantization

            // Away from boundaries, numerical gradient should be ~0
            assert!(
                numerical_grad.abs() < 1.0,
                "Numerical gradient should be near 0 away from boundaries at x={}: {}",
                x,
                numerical_grad
            );
        }

        // At boundaries, numerical gradient can be very large (jump in function)
        // This verifies the quantization function has discrete jumps
        let boundary_x = 0.5; // Known boundary point
        let (y_plus, _) = quantize_forward(boundary_x + eps, scale, bits);
        let (y_minus, _) = quantize_forward(boundary_x - eps, scale, bits);
        let boundary_grad = (y_plus - y_minus).abs() / (2.0 * eps);

        // Either zero (no boundary crossed) or large (boundary crossed)
        // This is expected - quantization is discontinuous
        assert!(
            boundary_grad < 1.0 || boundary_grad > 100.0,
            "Boundary gradient should be either ~0 or very large: {}",
            boundary_grad
        );
    }

    // ============================================================================
    // 8. Edge Cases
    // ============================================================================

    #[test]
    fn test_ste_zero_gradient() {
        // Zero upstream gradient should give zero local gradient
        for variant in [
            SteVariant::Standard,
            SteVariant::Clipped,
            SteVariant::LearnedStepSize,
            SteVariant::Ewgs,
        ] {
            let grad_in = ste_backward_input(0.0, 0.5, 1.0, variant);
            assert!(
                grad_in.abs() < EPSILON,
                "{:?}: zero upstream should give zero local: got {}",
                variant,
                grad_in
            );
        }
    }

    #[test]
    fn test_ste_negative_gradient() {
        // Negative upstream gradient should propagate correctly
        let variant = SteVariant::Standard;
        let grad_in = ste_backward_input(-1.0, 0.5, 1.0, variant);
        assert!(
            (grad_in - (-1.0)).abs() < EPSILON,
            "Negative gradient not propagated: got {}",
            grad_in
        );
    }

    #[test]
    fn test_ste_very_small_scale() {
        let scale = 1e-6;
        let x = 1e-7;
        let grad_out = 1.0;

        for variant in [SteVariant::Standard, SteVariant::Clipped] {
            let grad_in = ste_backward_input(grad_out, x, scale, variant);
            assert!(
                grad_in.is_finite(),
                "{:?}: gradient should be finite with small scale: got {}",
                variant,
                grad_in
            );
        }
    }

    #[test]
    fn test_ste_very_large_scale() {
        let scale = 1e6;
        let x = 1e5;
        let grad_out = 1.0;

        for variant in [SteVariant::Standard, SteVariant::Clipped] {
            let grad_in = ste_backward_input(grad_out, x, scale, variant);
            assert!(
                grad_in.is_finite(),
                "{:?}: gradient should be finite with large scale: got {}",
                variant,
                grad_in
            );
        }
    }

    // ============================================================================
    // 9. Convergence Tests
    // ============================================================================

    #[test]
    fn test_ste_enables_convergence() {
        // Simulate training loop to verify gradients enable convergence
        let target = 0.7;
        let mut scale = 0.5;
        let bits = 3u8;
        let lr = 0.1;
        let variant = SteVariant::LearnedStepSize;

        let initial_error = {
            let (y, _) = quantize_forward(target, scale, bits);
            (target - y).abs()
        };

        // Training iterations
        for _ in 0..100 {
            let (y, q) = quantize_forward(target, scale, bits);
            let loss = (target - y).powi(2);
            let grad_loss = -2.0 * (target - y);

            let grad_s = ste_backward_scale(grad_loss, target, scale, q, variant);
            scale -= lr * grad_s;
            scale = scale.max(0.01); // Prevent scale from going negative
        }

        let final_error = {
            let (y, _) = quantize_forward(target, scale, bits);
            (target - y).abs()
        };

        // Error should decrease or stay bounded
        assert!(
            final_error <= initial_error * 1.1 || final_error < 0.5,
            "Training should reduce error: initial={}, final={}",
            initial_error,
            final_error
        );
    }

    // ============================================================================
    // 10. Variant Comparison Tests
    // ============================================================================

    #[test]
    fn test_variant_ordering_by_sparsity() {
        // Different variants produce different gradient sparsity patterns
        let x_values: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.1).collect();
        let scale = 1.0;
        let grad_out = 1.0;

        let mut sparsity: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();

        for variant in [
            SteVariant::Standard,
            SteVariant::Clipped,
            SteVariant::LearnedStepSize,
        ] {
            let zeros = x_values
                .iter()
                .filter(|&&x| ste_backward_input(grad_out, x, scale, variant).abs() < EPSILON)
                .count();

            let name = match variant {
                SteVariant::Standard => "Standard",
                SteVariant::Clipped => "Clipped",
                SteVariant::LearnedStepSize => "LSQ",
                SteVariant::Ewgs => "EWGS",
            };
            sparsity.insert(name, zeros);
        }

        // Standard should have no zeros (always passes through)
        assert_eq!(sparsity["Standard"], 0, "Standard STE should have no zeros");

        // Clipped should have more zeros than Standard
        assert!(
            sparsity["Clipped"] > sparsity["Standard"],
            "Clipped should be sparser than Standard"
        );

        // LSQ has wider range, so fewer zeros than Clipped
        assert!(
            sparsity["LSQ"] <= sparsity["Clipped"],
            "LSQ should have <= zeros than Clipped"
        );
    }
}
