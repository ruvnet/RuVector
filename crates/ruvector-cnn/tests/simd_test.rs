//! Tests for SIMD-accelerated operations
//!
//! Tests cover:
//! - SIMD vs scalar equivalence
//! - dot_product accuracy
//! - Edge cases (empty, misaligned, remainder handling)

use ruvector_cnn::simd;
use ruvector_cnn::simd::scalar;

// ============================================================================
// Dot Product Tests
// ============================================================================

#[test]
fn test_dot_product_simd_vs_scalar() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    let simd_result = simd::dot_product_simd(&a, &b);
    let scalar_result = scalar::dot_product_scalar(&a, &b);

    assert!(
        (simd_result - scalar_result).abs() < 1e-5,
        "SIMD: {}, Scalar: {}",
        simd_result,
        scalar_result
    );
}

#[test]
fn test_dot_product_large_vector() {
    // Large vector to exercise SIMD loop (512 elements)
    let size = 512;
    let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..size).map(|i| ((size - i) as f32) * 0.01).collect();

    let simd_result = simd::dot_product_simd(&a, &b);
    let scalar_result = scalar::dot_product_scalar(&a, &b);

    assert!(
        (simd_result - scalar_result).abs() < 1.0, // Allow larger epsilon for accumulated error
        "SIMD: {}, Scalar: {}",
        simd_result,
        scalar_result
    );
}

#[test]
fn test_dot_product_various_sizes() {
    // Test sizes that exercise different SIMD code paths
    for size in [
        1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 100, 128, 255, 256,
    ] {
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..size).map(|i| ((size - i) as f32) * 0.1).collect();

        let simd_result = simd::dot_product_simd(&a, &b);
        let scalar_result = scalar::dot_product_scalar(&a, &b);

        let abs_diff = (simd_result - scalar_result).abs();
        let rel_error = if scalar_result.abs() > 1e-10 {
            abs_diff / scalar_result.abs()
        } else {
            abs_diff
        };

        assert!(
            rel_error < 1e-4 || abs_diff < 1e-4,
            "Size {}: SIMD={}, Scalar={}, diff={}",
            size,
            simd_result,
            scalar_result,
            abs_diff
        );
    }
}

#[test]
fn test_dot_product_empty() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];

    let result = simd::dot_product_simd(&a, &b);
    assert_eq!(result, 0.0);
}

#[test]
fn test_dot_product_single_element() {
    let a = vec![3.0];
    let b = vec![4.0];

    let result = simd::dot_product_simd(&a, &b);
    assert!((result - 12.0).abs() < 1e-6);
}

#[test]
fn test_dot_product_negative_values() {
    let a = vec![-1.0, -2.0, 3.0, 4.0];
    let b = vec![2.0, -3.0, -4.0, 5.0];

    // (-1*2) + (-2*-3) + (3*-4) + (4*5) = -2 + 6 - 12 + 20 = 12
    let result = simd::dot_product_simd(&a, &b);
    assert!((result - 12.0).abs() < 1e-5);
}

#[test]
fn test_dot_product_known_value() {
    // Simple known calculation
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    // 1+2+3+4+5+6+7+8 = 36
    let result = simd::dot_product_simd(&a, &b);
    assert!((result - 36.0).abs() < 1e-5);
}

#[test]
fn test_dot_product_large_small_values() {
    // Test numerical precision with large and small values
    let a = vec![1e6, 1e-6, 1e6, 1e-6];
    let b = vec![1e-6, 1e6, 1e-6, 1e6];

    let simd_result = simd::dot_product_simd(&a, &b);
    let scalar_result = scalar::dot_product_scalar(&a, &b);

    assert!(
        (simd_result - scalar_result).abs() < 1.0,
        "SIMD: {}, Scalar: {}",
        simd_result,
        scalar_result
    );
}

// ============================================================================
// ReLU Tests
// ============================================================================

#[test]
fn test_relu_simd_vs_scalar() {
    let input: Vec<f32> = (-16..16).map(|i| i as f32 * 0.5).collect();
    let mut simd_output = vec![0.0; input.len()];
    let mut scalar_output = vec![0.0; input.len()];

    simd::relu_simd(&input, &mut simd_output);
    scalar::relu_scalar(&input, &mut scalar_output);

    for (i, (&s, &r)) in simd_output.iter().zip(scalar_output.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-6,
            "Index {}: SIMD={}, Scalar={}",
            i,
            s,
            r
        );
    }
}

#[test]
fn test_relu_all_negative() {
    let input = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
    let mut output = vec![0.0; 8];

    simd::relu_simd(&input, &mut output);

    for &val in &output {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_relu_all_positive() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0; 8];

    simd::relu_simd(&input, &mut output);

    assert_eq!(output, input);
}

#[test]
fn test_relu_mixed() {
    let input = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
    let mut output = vec![0.0; 8];

    simd::relu_simd(&input, &mut output);

    assert_eq!(output, vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0]);
}

#[test]
fn test_relu_large_batch() {
    let size = 1024;
    let input: Vec<f32> = (0..size).map(|i| (i as f32) - 512.0).collect();
    let mut simd_output = vec![0.0; size];
    let mut scalar_output = vec![0.0; size];

    simd::relu_simd(&input, &mut simd_output);
    scalar::relu_scalar(&input, &mut scalar_output);

    for (i, (&s, &r)) in simd_output.iter().zip(scalar_output.iter()).enumerate() {
        assert_eq!(s, r, "Mismatch at index {}", i);
    }
}

// ============================================================================
// ReLU6 Tests
// ============================================================================

#[test]
fn test_relu6_simd_vs_scalar() {
    let input: Vec<f32> = (-8..16).map(|i| i as f32 * 0.5).collect();
    let mut simd_output = vec![0.0; input.len()];
    let mut scalar_output = vec![0.0; input.len()];

    simd::relu6_simd(&input, &mut simd_output);
    scalar::relu6_scalar(&input, &mut scalar_output);

    for (i, (&s, &r)) in simd_output.iter().zip(scalar_output.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-6,
            "Index {}: SIMD={}, Scalar={}",
            i,
            s,
            r
        );
    }
}

#[test]
fn test_relu6_clamps() {
    let input = vec![-1.0, 2.0, 7.0, 4.0, -5.0, 10.0, 3.0, 8.0];
    let mut output = vec![0.0; 8];

    simd::relu6_simd(&input, &mut output);

    assert_eq!(output, vec![0.0, 2.0, 6.0, 4.0, 0.0, 6.0, 3.0, 6.0]);
}

#[test]
fn test_relu6_boundary() {
    let input = vec![0.0, 6.0, -0.001, 6.001];
    let mut output = vec![0.0; 4];

    simd::relu6_simd(&input, &mut output);

    assert!(output[0].abs() < 1e-6); // 0 -> 0
    assert!((output[1] - 6.0).abs() < 1e-6); // 6 -> 6
    assert!(output[2].abs() < 1e-6); // -0.001 -> 0
    assert!((output[3] - 6.0).abs() < 1e-6); // 6.001 -> 6
}

// ============================================================================
// Batch Normalization Tests
// ============================================================================

#[test]
fn test_batch_norm_simd_vs_scalar() {
    let channels = 4;
    let spatial = 16;
    let input: Vec<f32> = (0..channels * spatial).map(|i| (i as f32) * 0.1).collect();
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];
    let mean = vec![0.0; channels];
    let var = vec![1.0; channels];
    let epsilon = 1e-5;

    let mut simd_output = vec![0.0; input.len()];
    let mut scalar_output = vec![0.0; input.len()];

    simd::batch_norm_simd(
        &input,
        &mut simd_output,
        &gamma,
        &beta,
        &mean,
        &var,
        epsilon,
        channels,
    );
    scalar::batch_norm_scalar(
        &input,
        &mut scalar_output,
        &gamma,
        &beta,
        &mean,
        &var,
        epsilon,
        channels,
    );

    for (i, (&s, &r)) in simd_output.iter().zip(scalar_output.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-4,
            "Index {}: SIMD={}, Scalar={}",
            i,
            s,
            r
        );
    }
}

#[test]
fn test_batch_norm_identity() {
    // With gamma=1, beta=0, mean=0, var=1: output should equal input
    let channels = 8;
    let spatial = 4;
    let input: Vec<f32> = (0..channels * spatial).map(|i| (i as f32) * 0.1).collect();
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];
    let mean = vec![0.0; channels];
    let var = vec![1.0; channels];

    let mut output = vec![0.0; input.len()];

    simd::batch_norm_simd(
        &input,
        &mut output,
        &gamma,
        &beta,
        &mean,
        &var,
        1e-5,
        channels,
    );

    for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
        assert!(
            (inp - out).abs() < 0.01,
            "Index {}: input={}, output={}",
            i,
            inp,
            out
        );
    }
}

#[test]
fn test_batch_norm_normalization() {
    // Test that batch norm actually normalizes with given stats
    let channels = 2;
    let input = vec![
        5.0, 10.0, // mean of ch0=5, mean of ch1=10
        5.0, 10.0,
    ];
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    let mean = vec![5.0, 10.0];
    let var = vec![1.0, 1.0];

    let mut output = vec![0.0; 4];

    simd::batch_norm_simd(
        &input,
        &mut output,
        &gamma,
        &beta,
        &mean,
        &var,
        1e-5,
        channels,
    );

    // (5 - 5) / sqrt(1 + eps) = 0
    // (10 - 10) / sqrt(1 + eps) = 0
    for &val in &output {
        assert!(val.abs() < 0.01, "Expected ~0, got {}", val);
    }
}

// ============================================================================
// 3x3 Convolution Tests
// ============================================================================

#[test]
fn test_conv_3x3_simd_vs_scalar() {
    let in_h = 8;
    let in_w = 8;
    let in_c = 3;
    let out_c = 4;
    let stride = 1;
    let padding = 1;

    let input: Vec<f32> = (0..in_h * in_w * in_c).map(|i| (i as f32) * 0.01).collect();
    let kernel: Vec<f32> = (0..out_c * 3 * 3 * in_c)
        .map(|i| (i as f32) * 0.001)
        .collect();

    let out_h = (in_h + 2 * padding - 3) / stride + 1;
    let out_w = (in_w + 2 * padding - 3) / stride + 1;

    let mut simd_output = vec![0.0; out_h * out_w * out_c];
    let mut scalar_output = vec![0.0; out_h * out_w * out_c];

    simd::conv_3x3_simd(
        &input,
        &kernel,
        &mut simd_output,
        in_h,
        in_w,
        in_c,
        out_c,
        stride,
        padding,
    );
    scalar::conv_3x3_scalar(
        &input,
        &kernel,
        &mut scalar_output,
        in_h,
        in_w,
        in_c,
        out_c,
        stride,
        padding,
    );

    for (i, (&s, &r)) in simd_output.iter().zip(scalar_output.iter()).enumerate() {
        assert!((s - r).abs() < 0.1, "Index {}: SIMD={}, Scalar={}", i, s, r);
    }
}

// ============================================================================
// Depthwise 3x3 Convolution Tests
// ============================================================================

#[test]
fn test_depthwise_conv_3x3_simd_vs_scalar() {
    let h = 8;
    let w = 8;
    let c = 4;
    let stride = 1;
    let padding = 1;

    let input: Vec<f32> = (0..h * w * c).map(|i| (i as f32) * 0.01).collect();
    let kernel: Vec<f32> = (0..c * 9).map(|i| (i as f32) * 0.01).collect();

    let out_h = (h + 2 * padding - 3) / stride + 1;
    let out_w = (w + 2 * padding - 3) / stride + 1;

    let mut simd_output = vec![0.0; out_h * out_w * c];
    let mut scalar_output = vec![0.0; out_h * out_w * c];

    simd::depthwise_conv_3x3_simd(&input, &kernel, &mut simd_output, h, w, c, stride, padding);
    scalar::depthwise_conv_3x3_scalar(
        &input,
        &kernel,
        &mut scalar_output,
        h,
        w,
        c,
        stride,
        padding,
    );

    for (i, (&s, &r)) in simd_output.iter().zip(scalar_output.iter()).enumerate() {
        assert!((s - r).abs() < 0.1, "Index {}: SIMD={}, Scalar={}", i, s, r);
    }
}

// ============================================================================
// Global Average Pooling Tests
// ============================================================================

#[test]
fn test_global_avg_pool_simd_vs_scalar() {
    let h = 4;
    let w = 4;
    let c = 8;

    let input: Vec<f32> = (0..h * w * c).map(|i| (i as f32) * 0.1).collect();

    let mut simd_output = vec![0.0; c];
    let mut scalar_output = vec![0.0; c];

    simd::global_avg_pool_simd(&input, &mut simd_output, h, w, c);
    scalar::global_avg_pool_scalar(&input, &mut scalar_output, h, w, c);

    for (i, (&s, &r)) in simd_output.iter().zip(scalar_output.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-4,
            "Channel {}: SIMD={}, Scalar={}",
            i,
            s,
            r
        );
    }
}

#[test]
fn test_global_avg_pool_uniform_input() {
    let h = 4;
    let w = 4;
    let c = 4;

    // All values = 2.0, average should be 2.0
    let input = vec![2.0; h * w * c];
    let mut output = vec![0.0; c];

    simd::global_avg_pool_simd(&input, &mut output, h, w, c);

    for &val in &output {
        assert!((val - 2.0).abs() < 1e-5);
    }
}

// ============================================================================
// Max Pooling 2x2 Tests
// ============================================================================

#[test]
fn test_max_pool_2x2_simd_vs_scalar() {
    let h = 8;
    let w = 8;
    let c = 4;
    let stride = 2;

    let input: Vec<f32> = (0..h * w * c).map(|i| (i as f32) * 0.1).collect();

    let out_h = h / stride;
    let out_w = w / stride;

    let mut simd_output = vec![0.0; out_h * out_w * c];
    let mut scalar_output = vec![0.0; out_h * out_w * c];

    simd::max_pool_2x2_simd(&input, &mut simd_output, h, w, c, stride);
    scalar::max_pool_2x2_scalar(&input, &mut scalar_output, h, w, c, stride);

    for (i, (&s, &r)) in simd_output.iter().zip(scalar_output.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-5,
            "Index {}: SIMD={}, Scalar={}",
            i,
            s,
            r
        );
    }
}

#[test]
fn test_max_pool_2x2_finds_max() {
    // 4x4 input, 1 channel
    let input = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let mut output = vec![0.0; 4];

    simd::max_pool_2x2_simd(&input, &mut output, 4, 4, 1, 2);

    // 2x2 windows:
    // [1,2,5,6] -> 6
    // [3,4,7,8] -> 8
    // [9,10,13,14] -> 14
    // [11,12,15,16] -> 16
    assert_eq!(output[0], 6.0);
    assert_eq!(output[1], 8.0);
    assert_eq!(output[2], 14.0);
    assert_eq!(output[3], 16.0);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_simd_empty_input() {
    let empty: Vec<f32> = vec![];
    let mut output: Vec<f32> = vec![];

    // These should not panic
    simd::relu_simd(&empty, &mut output);
    simd::relu6_simd(&empty, &mut output);
}

#[test]
fn test_simd_single_element() {
    let input = vec![5.0];
    let mut output = vec![0.0];

    simd::relu_simd(&input, &mut output);
    assert_eq!(output[0], 5.0);

    let input_neg = vec![-5.0];
    simd::relu_simd(&input_neg, &mut output);
    assert_eq!(output[0], 0.0);
}

#[test]
fn test_simd_remainder_handling() {
    // Test sizes that don't align with SIMD width (not multiple of 8)
    for size in [3, 7, 9, 15, 17, 25, 33] {
        let input: Vec<f32> = (0..size)
            .map(|i| (i as f32) - (size as f32 / 2.0))
            .collect();
        let mut simd_output = vec![0.0; size];
        let mut scalar_output = vec![0.0; size];

        simd::relu_simd(&input, &mut simd_output);
        scalar::relu_scalar(&input, &mut scalar_output);

        for (i, (&s, &r)) in simd_output.iter().zip(scalar_output.iter()).enumerate() {
            assert_eq!(s, r, "Size {}, index {}: SIMD={}, Scalar={}", size, i, s, r);
        }
    }
}

// ============================================================================
// Scalar Function Tests (for reference)
// ============================================================================

#[test]
fn test_scalar_swish() {
    let input = vec![0.0, 1.0, -1.0, 2.0];
    let mut output = vec![0.0; 4];

    scalar::swish_scalar(&input, &mut output);

    // swish(0) = 0
    assert!(output[0].abs() < 1e-6);
    // swish(1) = 1 * sigmoid(1) ~ 0.731
    assert!((output[1] - 0.731).abs() < 0.01);
    // swish(-1) = -1 * sigmoid(-1) ~ -0.268
    assert!((output[2] - (-0.268)).abs() < 0.01);
}

#[test]
fn test_scalar_hard_swish() {
    let input = vec![-4.0, -3.0, 0.0, 3.0, 4.0];
    let mut output = vec![0.0; 5];

    scalar::hard_swish_scalar(&input, &mut output);

    assert!(output[0].abs() < 1e-5); // -4 -> 0
    assert!(output[1].abs() < 1e-5); // -3 -> 0
    assert!(output[2].abs() < 1e-5); // 0 -> 0
    assert!((output[3] - 3.0).abs() < 1e-5); // 3 -> 3
}

#[test]
fn test_scalar_sigmoid() {
    let input = vec![0.0, 10.0, -10.0];
    let mut output = vec![0.0; 3];

    scalar::sigmoid_scalar(&input, &mut output);

    assert!((output[0] - 0.5).abs() < 1e-5); // sigmoid(0) = 0.5
    assert!((output[1] - 1.0).abs() < 0.001); // sigmoid(10) ~ 1.0
    assert!(output[2] < 0.001); // sigmoid(-10) ~ 0.0
}

// ============================================================================
// Platform Detection Tests
// ============================================================================

#[test]
fn test_simd_feature_detection() {
    // This test verifies the code compiles and runs on any platform
    let a = vec![1.0f32; 16];
    let b = vec![2.0f32; 16];

    // Should use optimal SIMD path for current platform
    let result = simd::dot_product_simd(&a, &b);

    assert!((result - 32.0).abs() < 1e-5);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_avx2_detection() {
    // On x86_64, check if AVX2 is detected (informational)
    let has_avx2 = is_x86_feature_detected!("avx2");
    println!("AVX2 available: {}", has_avx2);

    // Test should pass regardless of AVX2 availability
    let a = vec![1.0f32; 32];
    let b = vec![1.0f32; 32];
    let result = simd::dot_product_simd(&a, &b);
    assert!((result - 32.0).abs() < 1e-5);
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_neon_available() {
    // NEON is always available on aarch64
    let a = vec![1.0f32; 32];
    let b = vec![1.0f32; 32];
    let result = simd::dot_product_simd(&a, &b);
    assert!((result - 32.0).abs() < 1e-5);
}

// ============================================================================
// Numerical Stability Edge Cases
// ============================================================================

#[test]
fn test_dot_product_inf_handling() {
    let a = vec![f32::INFINITY, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let b = vec![1.0f32; 8];

    let result = simd::dot_product_simd(&a, &b);

    assert!(result.is_infinite() && result > 0.0);
}

#[test]
fn test_dot_product_nan_propagation() {
    let a = vec![f32::NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let b = vec![1.0f32; 8];

    let result = simd::dot_product_simd(&a, &b);

    assert!(result.is_nan());
}

#[test]
fn test_activation_with_special_values() {
    let input = vec![
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
        1.0,
        -1.0,
        6.0,
        100.0,
    ];
    let mut output = vec![0.0; 8];

    simd::relu_simd(&input, &mut output);

    assert!(output[0].is_infinite() && output[0] > 0.0); // inf stays inf
    assert_eq!(output[1], 0.0); // -inf becomes 0
    // NaN handling depends on backend: AVX2 `_mm256_max_ps(NaN, 0)` returns
    // the second operand (0.0) per Intel's unordered-comparison semantics,
    // while a scalar `f32::max` propagates NaN. Both behaviors are
    // legitimate ReLU implementations, so accept either.
    assert!(
        output[2].is_nan() || output[2] == 0.0,
        "expected NaN or 0.0 for ReLU(NaN), got {}",
        output[2]
    );
    assert_eq!(output[3], 0.0);
    assert_eq!(output[4], 1.0);
    assert_eq!(output[5], 0.0);
    assert_eq!(output[6], 6.0);
    assert_eq!(output[7], 100.0);
}
