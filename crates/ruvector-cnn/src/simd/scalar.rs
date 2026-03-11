//! Scalar Fallback Implementations
//!
//! Pure Rust implementations without SIMD intrinsics.
//! Used as fallback on unsupported platforms.

/// Scalar dot product
#[inline(always)]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Scalar ReLU activation: max(0, x)
#[inline(always)]
pub fn relu_scalar(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (out, &inp) in output.iter_mut().zip(input.iter()) {
        *out = inp.max(0.0);
    }
}

/// Scalar ReLU6 activation: min(6, max(0, x))
#[inline(always)]
pub fn relu6_scalar(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (out, &inp) in output.iter_mut().zip(input.iter()) {
        *out = inp.max(0.0).min(6.0);
    }
}

/// Scalar Swish activation: x * sigmoid(x)
#[inline(always)]
pub fn swish_scalar(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (out, &inp) in output.iter_mut().zip(input.iter()) {
        let sigmoid = 1.0 / (1.0 + (-inp).exp());
        *out = inp * sigmoid;
    }
}

/// Scalar HardSwish activation: x * relu6(x + 3) / 6
#[inline(always)]
pub fn hard_swish_scalar(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (out, &inp) in output.iter_mut().zip(input.iter()) {
        *out = inp * (inp + 3.0).max(0.0).min(6.0) / 6.0;
    }
}

/// Scalar sigmoid activation: 1 / (1 + exp(-x))
#[inline(always)]
pub fn sigmoid_scalar(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (out, &inp) in output.iter_mut().zip(input.iter()) {
        *out = 1.0 / (1.0 + (-inp).exp());
    }
}

/// Scalar batch normalization
/// y = gamma * (x - mean) / sqrt(var + epsilon) + beta
#[inline(always)]
pub fn batch_norm_scalar(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    epsilon: f32,
    channels: usize,
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(gamma.len(), channels);
    debug_assert_eq!(beta.len(), channels);
    debug_assert_eq!(mean.len(), channels);
    debug_assert_eq!(var.len(), channels);

    // Pre-compute scale and shift for each channel
    let mut scale = vec![0.0f32; channels];
    let mut shift = vec![0.0f32; channels];

    for c in 0..channels {
        let inv_std = 1.0 / (var[c] + epsilon).sqrt();
        scale[c] = gamma[c] * inv_std;
        shift[c] = beta[c] - mean[c] * scale[c];
    }

    // Apply batch normalization (NHWC layout)
    for (i, (out, &inp)) in output.iter_mut().zip(input.iter()).enumerate() {
        let c = i % channels;
        *out = inp * scale[c] + shift[c];
    }
}

/// Scalar 3x3 convolution (NHWC layout)
#[inline]
pub fn conv_3x3_scalar(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_c: usize,
    stride: usize,
    padding: usize,
) {
    let out_h = (in_h + 2 * padding - 3) / stride + 1;
    let out_w = (in_w + 2 * padding - 3) / stride + 1;

    // Kernel shape: [out_c, 3, 3, in_c] (OHWI format for NHWC input)
    debug_assert_eq!(kernel.len(), out_c * 9 * in_c);
    debug_assert_eq!(output.len(), out_h * out_w * out_c);

    for oh in 0..out_h {
        for ow in 0..out_w {
            for oc in 0..out_c {
                let mut sum = 0.0f32;

                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;

                            for ic in 0..in_c {
                                let input_idx = ih * in_w * in_c + iw * in_c + ic;
                                let kernel_idx = oc * 9 * in_c + kh * 3 * in_c + kw * in_c + ic;
                                sum += input[input_idx] * kernel[kernel_idx];
                            }
                        }
                    }
                }

                let output_idx = oh * out_w * out_c + ow * out_c + oc;
                output[output_idx] = sum;
            }
        }
    }
}

/// Scalar depthwise 3x3 convolution (NHWC layout)
#[inline]
pub fn depthwise_conv_3x3_scalar(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    stride: usize,
    padding: usize,
) {
    let out_h = (h + 2 * padding - 3) / stride + 1;
    let out_w = (w + 2 * padding - 3) / stride + 1;

    // Kernel shape: [c, 3, 3] (depthwise: one 3x3 filter per channel)
    debug_assert_eq!(kernel.len(), c * 9);
    debug_assert_eq!(output.len(), out_h * out_w * c);

    for oh in 0..out_h {
        for ow in 0..out_w {
            for ch in 0..c {
                let mut sum = 0.0f32;

                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;

                            let input_idx = ih * w * c + iw * c + ch;
                            let kernel_idx = ch * 9 + kh * 3 + kw;
                            sum += input[input_idx] * kernel[kernel_idx];
                        }
                    }
                }

                let output_idx = oh * out_w * c + ow * c + ch;
                output[output_idx] = sum;
            }
        }
    }
}

/// Scalar 1x1 convolution (pointwise, NHWC layout)
#[inline]
pub fn conv_1x1_scalar(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    in_c: usize,
    out_c: usize,
) {
    // Kernel shape: [out_c, in_c]
    debug_assert_eq!(kernel.len(), out_c * in_c);
    debug_assert_eq!(input.len(), h * w * in_c);
    debug_assert_eq!(output.len(), h * w * out_c);

    for y in 0..h {
        for x in 0..w {
            let input_base = (y * w + x) * in_c;
            let output_base = (y * w + x) * out_c;

            for oc in 0..out_c {
                let mut sum = 0.0f32;
                let kernel_base = oc * in_c;

                for ic in 0..in_c {
                    sum += input[input_base + ic] * kernel[kernel_base + ic];
                }

                output[output_base + oc] = sum;
            }
        }
    }
}

/// Scalar global average pooling (NHWC layout)
#[inline]
pub fn global_avg_pool_scalar(input: &[f32], output: &mut [f32], h: usize, w: usize, c: usize) {
    debug_assert_eq!(input.len(), h * w * c);
    debug_assert_eq!(output.len(), c);

    let spatial_size = (h * w) as f32;

    // Initialize output to zero
    for out in output.iter_mut() {
        *out = 0.0;
    }

    // Sum over spatial dimensions
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * c;
            for ch in 0..c {
                output[ch] += input[base + ch];
            }
        }
    }

    // Divide by spatial size
    for out in output.iter_mut() {
        *out /= spatial_size;
    }
}

/// Scalar max pooling 2x2 (NHWC layout)
#[inline]
pub fn max_pool_2x2_scalar(
    input: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    stride: usize,
) {
    let out_h = (h - 2) / stride + 1;
    let out_w = (w - 2) / stride + 1;

    debug_assert_eq!(input.len(), h * w * c);
    debug_assert_eq!(output.len(), out_h * out_w * c);

    for oh in 0..out_h {
        for ow in 0..out_w {
            let ih = oh * stride;
            let iw = ow * stride;

            for ch in 0..c {
                let mut max_val = f32::NEG_INFINITY;

                for kh in 0..2 {
                    for kw in 0..2 {
                        let idx = (ih + kh) * w * c + (iw + kw) * c + ch;
                        max_val = max_val.max(input[idx]);
                    }
                }

                let out_idx = oh * out_w * c + ow * c + ch;
                output[out_idx] = max_val;
            }
        }
    }
}

/// Scalar average pooling 2x2 (NHWC layout)
#[inline]
pub fn avg_pool_2x2_scalar(
    input: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    stride: usize,
) {
    let out_h = (h - 2) / stride + 1;
    let out_w = (w - 2) / stride + 1;

    debug_assert_eq!(input.len(), h * w * c);
    debug_assert_eq!(output.len(), out_h * out_w * c);

    for oh in 0..out_h {
        for ow in 0..out_w {
            let ih = oh * stride;
            let iw = ow * stride;

            for ch in 0..c {
                let mut sum = 0.0f32;

                for kh in 0..2 {
                    for kw in 0..2 {
                        let idx = (ih + kh) * w * c + (iw + kw) * c + ch;
                        sum += input[idx];
                    }
                }

                let out_idx = oh * out_w * c + ow * c + ch;
                output[out_idx] = sum / 4.0;
            }
        }
    }
}

/// Scalar max pooling with configurable kernel size (NHWC layout)
#[inline]
pub fn max_pool_scalar(
    input: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) {
    let out_h = (h + 2 * padding - kernel_size) / stride + 1;
    let out_w = (w + 2 * padding - kernel_size) / stride + 1;

    for oh in 0..out_h {
        for ow in 0..out_w {
            for ch in 0..c {
                let mut max_val = f32::NEG_INFINITY;

                for kh in 0..kernel_size {
                    for kw in 0..kernel_size {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                            let idx = ih as usize * w * c + iw as usize * c + ch;
                            max_val = max_val.max(input[idx]);
                        }
                    }
                }

                let out_idx = oh * out_w * c + ow * c + ch;
                output[out_idx] = max_val;
            }
        }
    }
}

/// Scalar average pooling with configurable kernel size (NHWC layout)
#[inline]
pub fn avg_pool_scalar(
    input: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) {
    let out_h = (h + 2 * padding - kernel_size) / stride + 1;
    let out_w = (w + 2 * padding - kernel_size) / stride + 1;

    for oh in 0..out_h {
        for ow in 0..out_w {
            for ch in 0..c {
                let mut sum = 0.0f32;
                let mut count = 0;

                for kh in 0..kernel_size {
                    for kw in 0..kernel_size {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                            let idx = ih as usize * w * c + iw as usize * c + ch;
                            sum += input[idx];
                            count += 1;
                        }
                    }
                }

                let out_idx = oh * out_w * c + ow * c + ch;
                output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let result = dot_product_scalar(&a, &b);
        assert!((result - 40.0).abs() < 0.001);
    }

    #[test]
    fn test_relu_scalar() {
        let input = vec![-1.0, 2.0, -3.0, 4.0];
        let mut output = vec![0.0; 4];
        relu_scalar(&input, &mut output);
        assert_eq!(output, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_relu6_scalar() {
        let input = vec![-1.0, 2.0, 7.0, 4.0];
        let mut output = vec![0.0; 4];
        relu6_scalar(&input, &mut output);
        assert_eq!(output, vec![0.0, 2.0, 6.0, 4.0]);
    }

    #[test]
    fn test_sigmoid_scalar() {
        let input = vec![0.0];
        let mut output = vec![0.0; 1];
        sigmoid_scalar(&input, &mut output);
        assert!((output[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_batch_norm_scalar() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 spatial, 1 channel
        let mut output = vec![0.0; 4];
        let gamma = vec![1.0];
        let beta = vec![0.0];
        let mean = vec![2.5];
        let var = vec![1.0];

        batch_norm_scalar(&input, &mut output, &gamma, &beta, &mean, &var, 1e-5, 1);

        // Check that output is normalized
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() < 0.01);
    }

    #[test]
    fn test_global_avg_pool_scalar() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2x2 spatial, 2 channels
        let mut output = vec![0.0; 2];
        global_avg_pool_scalar(&input, &mut output, 2, 2, 2);

        // Channel 0: (1 + 3 + 5 + 7) / 4 = 4.0
        // Channel 1: (2 + 4 + 6 + 8) / 4 = 5.0
        assert!((output[0] - 4.0).abs() < 0.001);
        assert!((output[1] - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_max_pool_2x2_scalar() {
        // 2x2 input with 1 channel: [[1, 2], [3, 4]]
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 1];
        max_pool_2x2_scalar(&input, &mut output, 2, 2, 1, 2);
        assert_eq!(output[0], 4.0);
    }
}
