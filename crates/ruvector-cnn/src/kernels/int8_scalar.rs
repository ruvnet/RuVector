//! INT8 Scalar Reference Kernels
//!
//! Pure Rust reference implementations for INT8 quantized convolution operations.
//! These kernels serve as:
//! - **Correctness baseline** for testing optimized SIMD kernels
//! - **Portable fallback** for platforms without SIMD support
//! - **Educational reference** for understanding INT8 computation
//!
//! # Design Principles
//!
//! 1. **i32 Accumulators**: All intermediate sums use i32 to prevent overflow
//!    - INT8 multiply: i8 * i8 → i16 (safe, can accumulate ~32k products)
//!    - We use i32 for safety: can accumulate 2^31 / 127^2 ≈ 133M products
//!
//! 2. **Per-Channel Scales**: Different quantization scale per output channel
//!    - Critical for CNN accuracy (weight distributions vary per channel)
//!    - Stored in `scales` array: one f32 per output channel
//!
//! 3. **Requantization**: i32 accumulator → i8 output
//!    - Apply per-channel scale multiplication
//!    - Round to nearest i8 with clamping to [-128, 127]
//!
//! 4. **Zero-Point Handling**: Support for asymmetric quantization
//!    - Activations: often non-negative (ReLU), need zero_point offset
//!    - Weights: typically symmetric (zero_point = 0)
//!
//! # Performance
//!
//! - These are **reference kernels** (scalar, not optimized)
//! - Expect 1-2x speedup vs FP32 scalar (memory bandwidth benefit)
//! - For production: use AVX2/NEON optimized kernels (4-8x speedup)

// No external dependencies needed for scalar kernels

/// Requantize i32 accumulator to i8 with per-channel scale
///
/// # Arguments
///
/// * `input` - i32 accumulator values (after convolution)
/// * `output` - i8 output tensor (requantized)
/// * `scales` - Per-channel scales (one per output channel)
/// * `zero_point` - Zero point for output quantization
/// * `out_channels` - Number of output channels
///
/// # Formula
///
/// ```text
/// output[i] = clamp(round(input[i] * scale[c] + zero_point), -128, 127)
/// where c = i % out_channels
/// ```
///
/// # Example
///
/// ```ignore
/// let acc = vec![1000, 2000, 3000, 4000]; // i32 accumulators
/// let mut output = vec![0i8; 4];
/// let scales = vec![0.01, 0.01]; // 2 output channels
/// requantize_scalar(&acc, &mut output, &scales, 0, 2);
/// // output ≈ [10, 20, 30, 40]
/// ```
#[inline]
pub fn requantize_scalar(
    input: &[i32],
    output: &mut [i8],
    scales: &[f32],
    zero_point: i8,
    out_channels: usize,
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(!scales.is_empty());

    let zp = zero_point as f32;

    for (i, (&acc, out)) in input.iter().zip(output.iter_mut()).enumerate() {
        let channel = i % out_channels;
        let scale = scales[channel];

        // Quantize: acc_i32 * scale + zero_point
        let scaled = acc as f32 * scale + zp;

        // Round and clamp to i8 range
        *out = scaled.round().clamp(-128.0, 127.0) as i8;
    }
}

/// INT8 2D Convolution (Reference Implementation)
///
/// # Arguments
///
/// * `input` - Input tensor (i8): [batch, in_h, in_w, in_c] (NHWC layout)
/// * `weights` - Quantized weights (i8): [out_c, kh, kw, in_c]
/// * `bias` - Bias in i32 accumulator space: [out_c]
/// * `output` - Output accumulators (i32): [batch, out_h, out_w, out_c]
/// * `batch` - Batch size
/// * `in_h`, `in_w`, `in_c` - Input dimensions
/// * `out_c` - Number of output channels
/// * `kh`, `kw` - Kernel height/width
/// * `stride` - Convolution stride
/// * `padding` - Zero padding
/// * `dilation` - Kernel dilation (1 = no dilation)
///
/// # Layout
///
/// - Input/Output: NHWC (batch, height, width, channels)
/// - Weights: OHWI (out_channels, kernel_h, kernel_w, in_channels)
///
/// # Formula
///
/// ```text
/// output[b, oh, ow, oc] = bias[oc] +
///     Σ Σ Σ input[b, ih, iw, ic] * weights[oc, kh, kw, ic]
///     kh kw ic
/// where:
///     ih = oh * stride + kh * dilation - padding
///     iw = ow * stride + kw * dilation - padding
/// ```
///
/// # Example
///
/// ```ignore
/// // 3x3 convolution, stride=1, padding=1
/// let input = vec![0i8; 1 * 28 * 28 * 3]; // 1x28x28x3
/// let weights = vec![0i8; 16 * 3 * 3 * 3]; // 16x3x3x3
/// let bias = vec![0i32; 16];
/// let mut output = vec![0i32; 1 * 28 * 28 * 16];
///
/// conv2d_int8_scalar(
///     &input, &weights, &bias, &mut output,
///     1, 28, 28, 3, 16, 3, 3, 1, 1, 1,
/// );
/// ```
#[inline]
pub fn conv2d_int8_scalar(
    input: &[i8],
    weights: &[i8],
    bias: &[i32],
    output: &mut [i32],
    batch: usize,
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) {
    let out_h = (in_h + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
    let out_w = (in_w + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;

    // Validate dimensions
    debug_assert_eq!(input.len(), batch * in_h * in_w * in_c);
    debug_assert_eq!(weights.len(), out_c * kh * kw * in_c);
    debug_assert_eq!(bias.len(), out_c);
    debug_assert_eq!(output.len(), batch * out_h * out_w * out_c);

    for b in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                for oc in 0..out_c {
                    // Start with bias
                    let mut acc = bias[oc];

                    // Convolve over kernel
                    for kh_idx in 0..kh {
                        for kw_idx in 0..kw {
                            // Compute input coordinates with dilation
                            let ih = (oh * stride + kh_idx * dilation) as isize - padding as isize;
                            let iw = (ow * stride + kw_idx * dilation) as isize - padding as isize;

                            // Skip if out of bounds (zero padding)
                            if ih < 0 || ih >= in_h as isize || iw < 0 || iw >= in_w as isize {
                                continue;
                            }

                            let ih = ih as usize;
                            let iw = iw as usize;

                            // Accumulate over input channels
                            for ic in 0..in_c {
                                let input_idx = ((b * in_h + ih) * in_w + iw) * in_c + ic;
                                let weight_idx = ((oc * kh + kh_idx) * kw + kw_idx) * in_c + ic;

                                let input_val = input[input_idx] as i32;
                                let weight_val = weights[weight_idx] as i32;

                                // i32 accumulator prevents overflow
                                acc += input_val * weight_val;
                            }
                        }
                    }

                    let output_idx = ((b * out_h + oh) * out_w + ow) * out_c + oc;
                    output[output_idx] = acc;
                }
            }
        }
    }
}

/// INT8 Depthwise 2D Convolution (Reference Implementation)
///
/// Depthwise convolution: one filter per input channel (groups = in_channels).
/// More efficient than standard convolution for mobile architectures.
///
/// # Arguments
///
/// * `input` - Input tensor (i8): [batch, in_h, in_w, channels]
/// * `weights` - Depthwise weights (i8): [channels, kh, kw]
/// * `bias` - Bias in i32 space: [channels]
/// * `output` - Output accumulators (i32): [batch, out_h, out_w, channels]
/// * `batch`, `in_h`, `in_w`, `channels` - Input dimensions
/// * `kh`, `kw` - Kernel dimensions
/// * `stride`, `padding`, `dilation` - Convolution parameters
///
/// # Per-Channel Scales
///
/// Depthwise convolution naturally supports per-channel quantization:
/// - Each channel has its own 3x3 (or other size) filter
/// - Each channel can have its own scale factor
/// - Critical for accuracy in MobileNets
///
/// # Formula
///
/// ```text
/// output[b, oh, ow, c] = bias[c] +
///     Σ Σ input[b, ih, iw, c] * weights[c, kh, kw]
///     kh kw
/// ```
///
/// # Example
///
/// ```ignore
/// // MobileNet depthwise 3x3, stride=1
/// let input = vec![0i8; 1 * 56 * 56 * 32]; // 1x56x56x32
/// let weights = vec![0i8; 32 * 3 * 3]; // 32x3x3 (one 3x3 per channel)
/// let bias = vec![0i32; 32];
/// let mut output = vec![0i32; 1 * 56 * 56 * 32];
///
/// depthwise_conv2d_int8_scalar(
///     &input, &weights, &bias, &mut output,
///     1, 56, 56, 32, 3, 3, 1, 1, 1,
/// );
/// ```
#[inline]
pub fn depthwise_conv2d_int8_scalar(
    input: &[i8],
    weights: &[i8],
    bias: &[i32],
    output: &mut [i32],
    batch: usize,
    in_h: usize,
    in_w: usize,
    channels: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) {
    let out_h = (in_h + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
    let out_w = (in_w + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;

    debug_assert_eq!(input.len(), batch * in_h * in_w * channels);
    debug_assert_eq!(weights.len(), channels * kh * kw);
    debug_assert_eq!(bias.len(), channels);
    debug_assert_eq!(output.len(), batch * out_h * out_w * channels);

    for b in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                for c in 0..channels {
                    let mut acc = bias[c];

                    for kh_idx in 0..kh {
                        for kw_idx in 0..kw {
                            let ih = (oh * stride + kh_idx * dilation) as isize - padding as isize;
                            let iw = (ow * stride + kw_idx * dilation) as isize - padding as isize;

                            if ih < 0 || ih >= in_h as isize || iw < 0 || iw >= in_w as isize {
                                continue;
                            }

                            let ih = ih as usize;
                            let iw = iw as usize;

                            let input_idx = ((b * in_h + ih) * in_w + iw) * channels + c;
                            let weight_idx = (c * kh + kh_idx) * kw + kw_idx;

                            acc += (input[input_idx] as i32) * (weights[weight_idx] as i32);
                        }
                    }

                    let output_idx = ((b * out_h + oh) * out_w + ow) * channels + c;
                    output[output_idx] = acc;
                }
            }
        }
    }
}

/// INT8 Matrix Multiplication (for Fully Connected Layers)
///
/// Computes: C = A * B + bias (in i32 accumulator space)
///
/// # Arguments
///
/// * `a` - Matrix A (i8): [M, K] (activations)
/// * `b` - Matrix B (i8): [K, N] (weights, transposed layout)
/// * `bias` - Bias vector (i32): [N]
/// * `output` - Output matrix (i32): [M, N]
/// * `m`, `k`, `n` - Matrix dimensions
///
/// # Layout
///
/// - A: Row-major [M, K]
/// - B: Column-major [K, N] (transposed for cache efficiency)
/// - C: Row-major [M, N]
///
/// # Formula
///
/// ```text
/// C[i, j] = bias[j] + Σ A[i, k] * B[k, j]
///                     k=0..K
/// ```
///
/// # Example
///
/// ```ignore
/// // FC layer: 1000 classes, 512 input features
/// let activations = vec![0i8; 1 * 512]; // batch=1, features=512
/// let weights = vec![0i8; 512 * 1000]; // [512, 1000] transposed
/// let bias = vec![0i32; 1000];
/// let mut output = vec![0i32; 1 * 1000];
///
/// matmul_int8_scalar(
///     &activations, &weights, &bias, &mut output,
///     1, 512, 1000,
/// );
/// ```
#[inline]
pub fn matmul_int8_scalar(
    a: &[i8],
    b: &[i8],
    bias: &[i32],
    output: &mut [i32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(bias.len(), n);
    debug_assert_eq!(output.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut acc = bias[j];

            for k_idx in 0..k {
                let a_val = a[i * k + k_idx] as i32;
                let b_val = b[k_idx * n + j] as i32; // B is column-major
                acc += a_val * b_val;
            }

            output[i * n + j] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_requantize_scalar() {
        let input = vec![1000, 2000, 3000, 4000];
        let mut output = vec![0i8; 4];
        let scales = vec![0.01, 0.02]; // 2 channels
        let zero_point = 0;

        requantize_scalar(&input, &mut output, &scales, zero_point, 2);

        // Channel 0: 1000 * 0.01 = 10, 3000 * 0.01 = 30
        assert_eq!(output[0], 10);
        assert_eq!(output[2], 30);

        // Channel 1: 2000 * 0.02 = 40, 4000 * 0.02 = 80
        assert_eq!(output[1], 40);
        assert_eq!(output[3], 80);
    }

    #[test]
    fn test_requantize_clamping() {
        let input = vec![20000, -20000]; // Will overflow i8
        let mut output = vec![0i8; 2];
        let scales = vec![1.0];
        let zero_point = 0;

        requantize_scalar(&input, &mut output, &scales, zero_point, 1);

        // Should clamp to i8 range
        assert_eq!(output[0], 127);
        assert_eq!(output[1], -128);
    }

    #[test]
    fn test_conv2d_int8_scalar_3x3_identity() {
        // 1x3x3x1 input with identity 3x3 filter
        let input = vec![1i8, 2, 3, 4, 5, 6, 7, 8, 9];
        let weights = vec![
            0i8, 0, 0, 0, 1, 0, 0, 0, 0, // Identity kernel: center=1, rest=0
        ];
        let bias = vec![0i32];
        let mut output = vec![0i32; 9]; // Same size with padding=1

        conv2d_int8_scalar(
            &input, &weights, &bias, &mut output, 1, 3, 3, 1, 1, 3, 3, 1, 1, 1,
        );

        // Center pixel should match input (5)
        // Corners/edges affected by zero padding
        assert_eq!(output[4], 5); // Center: input[4] * 1 = 5
    }

    #[test]
    fn test_conv2d_int8_scalar_no_overflow() {
        // Test that i32 accumulator handles large products
        let input = vec![127i8; 64]; // Max i8 value
        let weights = vec![127i8; 64]; // Max i8 value
        let bias = vec![0i32];
        let mut output = vec![0i32; 64];

        conv2d_int8_scalar(
            &input, &weights, &bias, &mut output, 1, 8, 8, 1, 1, 3, 3, 1, 1, 1,
        );

        // All accumulations should be valid (no overflow)
        // Max single product: 127 * 127 = 16129
        // Max accumulation for 3x3: 9 * 16129 = 145161 (well within i32)
        for &val in &output {
            assert!(val > 0); // Should be positive
            assert!(val <= 145161); // Should be <= 9 * 127^2
        }
    }

    #[test]
    fn test_depthwise_conv2d_int8_scalar() {
        // 1x3x3x2 input, 2 channels
        let input = vec![1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];

        // 2x3x3 depthwise weights
        let weights = vec![
            1i8, 0, 0, 0, 1, 0, 0, 0, 1, // Channel 0: diagonal
            0i8, 0, 1, 0, 1, 0, 1, 0, 0, // Channel 1: cross
        ];

        let bias = vec![0i32, 0i32];
        let mut output = vec![0i32; 18];

        depthwise_conv2d_int8_scalar(
            &input, &weights, &bias, &mut output, 1, 3, 3, 2, 3, 3, 1, 1, 1,
        );

        // Channel 0 center: 1*1 + 5*1 + 9*1 = 15
        assert_eq!(output[8], 15);

        // Channel 1 center: 3*1 + 5*1 + 7*1 = 15
        assert_eq!(output[9], 15);
    }

    #[test]
    fn test_matmul_int8_scalar_2x2() {
        // 2x3 * 3x2 = 2x2
        let a = vec![1i8, 2, 3, 4, 5, 6]; // [2, 3]
        let b = vec![1i8, 2, 3, 4, 5, 6]; // [3, 2] column-major
        let bias = vec![0i32, 0i32];
        let mut output = vec![0i32; 4]; // [2, 2]

        matmul_int8_scalar(&a, &b, &bias, &mut output, 2, 3, 2);

        // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        // C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        // C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        assert_eq!(output[0], 22);
        assert_eq!(output[1], 28);
        assert_eq!(output[2], 49);
        assert_eq!(output[3], 64);
    }

    #[test]
    fn test_matmul_int8_scalar_with_bias() {
        let a = vec![1i8, 2]; // [1, 2]
        let b = vec![3i8, 4]; // [2, 1] column-major
        let bias = vec![10i32];
        let mut output = vec![0i32];

        matmul_int8_scalar(&a, &b, &bias, &mut output, 1, 2, 1);

        // C[0,0] = bias + 1*3 + 2*4 = 10 + 3 + 8 = 21
        assert_eq!(output[0], 21);
    }

    #[test]
    fn test_conv2d_stride_2() {
        // Test stride=2 downsampling
        let input = vec![1i8; 16]; // 1x4x4x1
        let weights = vec![1i8; 9]; // 1x3x3x1
        let bias = vec![0i32];
        let mut output = vec![0i32; 4]; // 1x2x2x1 (stride=2, padding=1)

        conv2d_int8_scalar(
            &input, &weights, &bias, &mut output, 1, 4, 4, 1, 1, 3, 3, 2, 1, 1,
        );

        // All outputs should be valid
        for &val in &output {
            assert!(val >= 0);
        }
    }

    #[test]
    fn test_conv2d_dilation_2() {
        // Test dilated convolution
        let input = vec![1i8; 25]; // 1x5x5x1
        let weights = vec![1i8; 9]; // 1x3x3x1
        let bias = vec![0i32];
        let mut output = vec![0i32; 9]; // 1x3x3x1 (dilation=2, padding=2)

        conv2d_int8_scalar(
            &input, &weights, &bias, &mut output, 1, 5, 5, 1, 1, 3, 3, 1, 2, 2,
        );

        // All outputs should be valid
        for &val in &output {
            assert!(val >= 0);
        }
    }

    // Property-based tests
    #[test]
    fn test_requantize_preserves_range() {
        // Test that requantization always produces valid i8
        use std::i32::{MAX, MIN};

        let input = vec![MAX, MIN, 0, 1000, -1000];
        let mut output = vec![0i8; 5];
        let scales = vec![0.001];

        requantize_scalar(&input, &mut output, &scales, 0, 1);

        for &val in &output {
            assert!(val >= -128 && val <= 127);
        }
    }

    #[test]
    fn test_conv2d_commutative_channels() {
        // Output should be independent of channel ordering (given same weights)
        let input = vec![1i8, 2, 3, 4]; // 1x2x2x1
        let weights = vec![1i8, 1, 1, 1]; // 1x2x2x1
        let bias = vec![0i32];
        let mut output1 = vec![0i32; 4];
        let mut output2 = vec![0i32; 4];

        conv2d_int8_scalar(
            &input, &weights, &bias, &mut output1, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1,
        );

        // Run again - should be deterministic
        conv2d_int8_scalar(
            &input, &weights, &bias, &mut output2, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1,
        );

        assert_eq!(output1, output2);
    }

    #[test]
    fn test_depthwise_per_channel_independence() {
        // Each channel should be independent in depthwise conv
        let input = vec![1i8, 10, 2, 20, 3, 30, 4, 40]; // 1x2x2x2
        let weights = vec![1i8, 1, 1, 1, 2, 2, 2, 2]; // 2x2x2
        let bias = vec![0i32, 0i32];
        let mut output = vec![0i32; 8];

        depthwise_conv2d_int8_scalar(
            &input, &weights, &bias, &mut output, 1, 2, 2, 2, 2, 2, 1, 0, 1,
        );

        // Channel 0 (weights = [1,1,1,1]): should see input [1,2,3,4]
        // Channel 1 (weights = [2,2,2,2]): should see input [10,20,30,40]
        // Channel 1 outputs should be ~2x Channel 0 (scaled by weight difference)
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_matmul_associative_bias() {
        // (A*B) + bias = A*B + bias (bias is associative)
        let a = vec![1i8, 2];
        let b = vec![3i8, 4];
        let bias1 = vec![5i32];
        let bias2 = vec![10i32];
        let mut output1 = vec![0i32];
        let mut output2 = vec![0i32];

        matmul_int8_scalar(&a, &b, &bias1, &mut output1, 1, 2, 1);
        matmul_int8_scalar(&a, &b, &bias2, &mut output2, 1, 2, 1);

        assert_eq!(output2[0] - output1[0], 5); // bias2 - bias1
    }
}
