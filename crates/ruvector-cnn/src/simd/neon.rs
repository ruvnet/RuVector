//! NEON SIMD Implementations for ARM64/Apple Silicon
//!
//! Optimized implementations using NEON intrinsics:
//! - 128-bit registers (4 floats per iteration)
//! - FMA instructions for improved throughput
//! - Horizontal reduction via vaddvq_f32

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// Dot Product
// ============================================================================

/// NEON dot product with 4x unrolling
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // Use 4 accumulators for better ILP
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    // Process 16 floats at a time (4 x 4)
    let chunks = len / 16;
    let mut idx = 0usize;

    for _ in 0..chunks {
        let va0 = vld1q_f32(a_ptr.add(idx));
        let vb0 = vld1q_f32(b_ptr.add(idx));
        sum0 = vfmaq_f32(sum0, va0, vb0);

        let va1 = vld1q_f32(a_ptr.add(idx + 4));
        let vb1 = vld1q_f32(b_ptr.add(idx + 4));
        sum1 = vfmaq_f32(sum1, va1, vb1);

        let va2 = vld1q_f32(a_ptr.add(idx + 8));
        let vb2 = vld1q_f32(b_ptr.add(idx + 8));
        sum2 = vfmaq_f32(sum2, va2, vb2);

        let va3 = vld1q_f32(a_ptr.add(idx + 12));
        let vb3 = vld1q_f32(b_ptr.add(idx + 12));
        sum3 = vfmaq_f32(sum3, va3, vb3);

        idx += 16;
    }

    // Tree reduction
    let sum01 = vaddq_f32(sum0, sum1);
    let sum23 = vaddq_f32(sum2, sum3);
    let sum = vaddq_f32(sum01, sum23);

    // Process remaining 4-float chunks
    let remaining_start = chunks * 16;
    let remaining_chunks = (len - remaining_start) / 4;
    let mut final_sum = sum;

    idx = remaining_start;
    for _ in 0..remaining_chunks {
        let va = vld1q_f32(a_ptr.add(idx));
        let vb = vld1q_f32(b_ptr.add(idx));
        final_sum = vfmaq_f32(final_sum, va, vb);
        idx += 4;
    }

    // Horizontal sum
    let mut total = vaddvq_f32(final_sum);

    // Handle remainder
    let scalar_start = remaining_start + remaining_chunks * 4;
    for i in scalar_start..len {
        total += *a.get_unchecked(i) * *b.get_unchecked(i);
    }

    total
}

// ============================================================================
// ReLU Activation
// ============================================================================

/// NEON ReLU: max(0, x)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn relu_neon(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    let len = input.len();
    let zero = vdupq_n_f32(0.0);

    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let v = vld1q_f32(input.as_ptr().add(idx));
        let result = vmaxq_f32(v, zero);
        vst1q_f32(output.as_mut_ptr().add(idx), result);
    }

    // Handle remainder
    for i in (chunks * 4)..len {
        output[i] = input[i].max(0.0);
    }
}

/// NEON ReLU6: min(6, max(0, x))
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn relu6_neon(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    let len = input.len();
    let zero = vdupq_n_f32(0.0);
    let six = vdupq_n_f32(6.0);

    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let v = vld1q_f32(input.as_ptr().add(idx));
        let clamped_low = vmaxq_f32(v, zero);
        let clamped = vminq_f32(clamped_low, six);
        vst1q_f32(output.as_mut_ptr().add(idx), clamped);
    }

    // Handle remainder
    for i in (chunks * 4)..len {
        output[i] = input[i].max(0.0).min(6.0);
    }
}

// ============================================================================
// Batch Normalization
// ============================================================================

/// NEON batch normalization
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn batch_norm_neon(
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
    debug_assert!(input.len() % channels == 0);

    // Pre-compute scale and shift
    let mut scale = vec![0.0f32; channels];
    let mut shift = vec![0.0f32; channels];

    for c in 0..channels {
        let inv_std = 1.0 / (var[c] + epsilon).sqrt();
        scale[c] = gamma[c] * inv_std;
        shift[c] = beta[c] - mean[c] * scale[c];
    }

    let spatial_size = input.len() / channels;

    if channels >= 4 {
        let channel_chunks = channels / 4;

        for s in 0..spatial_size {
            let base = s * channels;

            for cc in 0..channel_chunks {
                let c = cc * 4;
                let idx = base + c;

                let v = vld1q_f32(input.as_ptr().add(idx));
                let scale_v = vld1q_f32(scale.as_ptr().add(c));
                let shift_v = vld1q_f32(shift.as_ptr().add(c));

                let result = vaddq_f32(vmulq_f32(v, scale_v), shift_v);
                vst1q_f32(output.as_mut_ptr().add(idx), result);
            }

            // Handle remaining channels
            for c in (channel_chunks * 4)..channels {
                let idx = base + c;
                output[idx] = input[idx] * scale[c] + shift[c];
            }
        }
    } else {
        // Scalar fallback
        for (i, (out, &inp)) in output.iter_mut().zip(input.iter()).enumerate() {
            let c = i % channels;
            *out = inp * scale[c] + shift[c];
        }
    }
}

// ============================================================================
// 3x3 Convolution
// ============================================================================

/// NEON 3x3 convolution (NHWC layout)
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn conv_3x3_neon(
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

    let out_c_chunks = out_c / 4;

    for oh in 0..out_h {
        for ow in 0..out_w {
            // Process 4 output channels at a time
            for occ in 0..out_c_chunks {
                let oc_base = occ * 4;
                let mut sum = vdupq_n_f32(0.0);

                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;

                            for ic in 0..in_c {
                                let input_idx = ih * in_w * in_c + iw * in_c + ic;
                                let input_val = vdupq_n_f32(input[input_idx]);

                                // Load 4 kernel values
                                let mut kernel_vals = [0.0f32; 4];
                                for i in 0..4 {
                                    let kernel_idx =
                                        (oc_base + i) * 9 * in_c + kh * 3 * in_c + kw * in_c + ic;
                                    kernel_vals[i] = kernel[kernel_idx];
                                }
                                let kernel_v = vld1q_f32(kernel_vals.as_ptr());

                                sum = vfmaq_f32(sum, input_val, kernel_v);
                            }
                        }
                    }
                }

                let output_idx = oh * out_w * out_c + ow * out_c + oc_base;
                vst1q_f32(output.as_mut_ptr().add(output_idx), sum);
            }

            // Handle remaining output channels
            for oc in (out_c_chunks * 4)..out_c {
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

                output[oh * out_w * out_c + ow * out_c + oc] = sum;
            }
        }
    }
}

// ============================================================================
// Depthwise Convolution
// ============================================================================

/// NEON depthwise 3x3 convolution
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn depthwise_conv_3x3_neon(
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

    let c_chunks = c / 4;

    for oh in 0..out_h {
        for ow in 0..out_w {
            // Process 4 channels at a time
            for cc in 0..c_chunks {
                let ch_base = cc * 4;
                let mut sum = vdupq_n_f32(0.0);

                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;

                            let input_idx = ih * w * c + iw * c + ch_base;
                            let input_v = vld1q_f32(input.as_ptr().add(input_idx));

                            // Load 4 kernel values
                            let mut kernel_vals = [0.0f32; 4];
                            for i in 0..4 {
                                kernel_vals[i] = kernel[(ch_base + i) * 9 + kh * 3 + kw];
                            }
                            let kernel_v = vld1q_f32(kernel_vals.as_ptr());

                            sum = vfmaq_f32(sum, input_v, kernel_v);
                        }
                    }
                }

                let output_idx = oh * out_w * c + ow * c + ch_base;
                vst1q_f32(output.as_mut_ptr().add(output_idx), sum);
            }

            // Handle remaining channels
            for ch in (c_chunks * 4)..c {
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

                output[oh * out_w * c + ow * c + ch] = sum;
            }
        }
    }
}

// ============================================================================
// Pooling Operations
// ============================================================================

/// NEON global average pooling
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn global_avg_pool_neon(
    input: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
) {
    let spatial_size = h * w;
    let inv_spatial = 1.0 / spatial_size as f32;
    let inv_spatial_v = vdupq_n_f32(inv_spatial);

    let c_chunks = c / 4;

    // Initialize output
    for i in 0..c_chunks {
        vst1q_f32(output.as_mut_ptr().add(i * 4), vdupq_n_f32(0.0));
    }
    for i in (c_chunks * 4)..c {
        output[i] = 0.0;
    }

    // Sum over spatial dimensions
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * c;

            for cc in 0..c_chunks {
                let ch_base = cc * 4;
                let input_v = vld1q_f32(input.as_ptr().add(base + ch_base));
                let out_v = vld1q_f32(output.as_ptr().add(ch_base));
                let sum_v = vaddq_f32(out_v, input_v);
                vst1q_f32(output.as_mut_ptr().add(ch_base), sum_v);
            }

            for ch in (c_chunks * 4)..c {
                output[ch] += input[base + ch];
            }
        }
    }

    // Multiply by 1/spatial_size
    for cc in 0..c_chunks {
        let ch_base = cc * 4;
        let sum_v = vld1q_f32(output.as_ptr().add(ch_base));
        let avg_v = vmulq_f32(sum_v, inv_spatial_v);
        vst1q_f32(output.as_mut_ptr().add(ch_base), avg_v);
    }
    for ch in (c_chunks * 4)..c {
        output[ch] *= inv_spatial;
    }
}

/// NEON max pooling 2x2
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn max_pool_2x2_neon(
    input: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    stride: usize,
) {
    let out_h = (h - 2) / stride + 1;
    let out_w = (w - 2) / stride + 1;

    let c_chunks = c / 4;

    for oh in 0..out_h {
        for ow in 0..out_w {
            let ih = oh * stride;
            let iw = ow * stride;

            // Process 4 channels at a time
            for cc in 0..c_chunks {
                let ch_base = cc * 4;

                let idx00 = ih * w * c + iw * c + ch_base;
                let idx01 = ih * w * c + (iw + 1) * c + ch_base;
                let idx10 = (ih + 1) * w * c + iw * c + ch_base;
                let idx11 = (ih + 1) * w * c + (iw + 1) * c + ch_base;

                let v00 = vld1q_f32(input.as_ptr().add(idx00));
                let v01 = vld1q_f32(input.as_ptr().add(idx01));
                let v10 = vld1q_f32(input.as_ptr().add(idx10));
                let v11 = vld1q_f32(input.as_ptr().add(idx11));

                let max01 = vmaxq_f32(v00, v01);
                let max23 = vmaxq_f32(v10, v11);
                let max_val = vmaxq_f32(max01, max23);

                let out_idx = oh * out_w * c + ow * c + ch_base;
                vst1q_f32(output.as_mut_ptr().add(out_idx), max_val);
            }

            // Handle remaining channels
            for ch in (c_chunks * 4)..c {
                let idx00 = ih * w * c + iw * c + ch;
                let idx01 = ih * w * c + (iw + 1) * c + ch;
                let idx10 = (ih + 1) * w * c + iw * c + ch;
                let idx11 = (ih + 1) * w * c + (iw + 1) * c + ch;

                let max_val = input[idx00]
                    .max(input[idx01])
                    .max(input[idx10])
                    .max(input[idx11]);

                output[oh * out_w * c + ow * c + ch] = max_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "aarch64")]
    use super::*;

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_dot_product_neon() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = unsafe { dot_product_neon(&a, &b) };
        let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();

        assert!((result - expected).abs() < 0.001);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_relu_neon() {
        let input = vec![-1.0, 2.0, -3.0, 4.0];
        let mut output = vec![0.0; 4];

        unsafe { relu_neon(&input, &mut output) };

        assert_eq!(output, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_relu6_neon() {
        let input = vec![-1.0, 2.0, 7.0, 4.0];
        let mut output = vec![0.0; 4];

        unsafe { relu6_neon(&input, &mut output) };

        assert_eq!(output, vec![0.0, 2.0, 6.0, 4.0]);
    }
}
