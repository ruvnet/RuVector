//! AVX2 and AVX-512 SIMD Implementations
//!
//! Provides x86_64-specific SIMD implementations for CNN operations.
//! Includes AVX2 (8 floats) and AVX-512 (16 floats) variants.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 dot product with FMA
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    let sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = 0.0f32;
    _mm_store_ss(&mut result, sum32);

    // Handle remainder
    for i in (chunks * 8)..a.len() {
        result += a[i] * b[i];
    }

    result
}

/// AVX2 dot product without FMA
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        let prod = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, prod);
    }

    // Horizontal sum
    let sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = 0.0f32;
    _mm_store_ss(&mut result, sum32);

    // Handle remainder
    for i in (chunks * 8)..a.len() {
        result += a[i] * b[i];
    }

    result
}

/// AVX-512 dot product
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut sum = _mm512_setzero_ps();
    let chunks = a.len() / 16;

    for i in 0..chunks {
        let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    let mut result = _mm512_reduce_add_ps(sum);

    // Handle remainder
    for i in (chunks * 16)..a.len() {
        result += a[i] * b[i];
    }

    result
}

/// AVX2 ReLU activation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn relu_avx2(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    let zero = _mm256_setzero_ps();
    let chunks = input.len() / 8;

    for i in 0..chunks {
        let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
        let result = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), result);
    }

    // Handle remainder
    for i in (chunks * 8)..input.len() {
        output[i] = input[i].max(0.0);
    }
}

/// AVX2 ReLU6 activation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn relu6_avx2(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    let zero = _mm256_setzero_ps();
    let six = _mm256_set1_ps(6.0);
    let chunks = input.len() / 8;

    for i in 0..chunks {
        let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
        let result = _mm256_min_ps(_mm256_max_ps(v, zero), six);
        _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), result);
    }

    // Handle remainder
    for i in (chunks * 8)..input.len() {
        output[i] = input[i].max(0.0).min(6.0);
    }
}

/// AVX2 batch normalization
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn batch_norm_avx2(
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

    // Pre-compute scale and shift for each channel
    let mut scale = vec![0.0f32; channels];
    let mut shift = vec![0.0f32; channels];

    for c in 0..channels {
        let inv_std = 1.0 / (var[c] + epsilon).sqrt();
        scale[c] = gamma[c] * inv_std;
        shift[c] = beta[c] - mean[c] * scale[c];
    }

    let spatial = input.len() / channels;

    // Process 8 spatial positions at a time if channels == 8
    if channels == 8 {
        let scale_v = _mm256_loadu_ps(scale.as_ptr());
        let shift_v = _mm256_loadu_ps(shift.as_ptr());

        for s in 0..spatial {
            let offset = s * channels;
            let v = _mm256_loadu_ps(input.as_ptr().add(offset));
            let result = _mm256_fmadd_ps(v, scale_v, shift_v);
            _mm256_storeu_ps(output.as_mut_ptr().add(offset), result);
        }
    } else {
        // Fallback for other channel counts
        for (i, (out, &inp)) in output.iter_mut().zip(input.iter()).enumerate() {
            let c = i % channels;
            *out = inp * scale[c] + shift[c];
        }
    }
}

/// AVX2 3x3 convolution with FMA (4x loop unrolling + multiple accumulators)
///
/// Processes 8 output channels simultaneously using AVX2 FMA instructions.
/// Uses 4x loop unrolling on input channels with 4 independent accumulators
/// for improved instruction-level parallelism (ILP).
///
/// Optimizations applied:
/// - 4x unrolled input channel loop (reduces loop overhead)
/// - 4 independent FMA accumulators (enables CPU pipelining)
/// - Hoisted bounds checks out of inner loop
///
/// Expected speedup: 1.3-1.5x over non-unrolled version.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn conv_3x3_avx2_fma(
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

    // Process 8 output channels at a time
    let out_c_chunks = out_c / 8;

    // Pre-compute input channel unrolling parameters
    let ic_chunks = in_c / 4;
    let ic_remainder_start = ic_chunks * 4;

    for oh in 0..out_h {
        for ow in 0..out_w {
            let out_spatial_idx = oh * out_w + ow;

            // Process 8 output channels at once
            for oc_chunk in 0..out_c_chunks {
                let oc_base = oc_chunk * 8;

                // Use 4 independent accumulators for ILP
                let mut sum0 = _mm256_setzero_ps();
                let mut sum1 = _mm256_setzero_ps();
                let mut sum2 = _mm256_setzero_ps();
                let mut sum3 = _mm256_setzero_ps();

                // Convolve over 3x3 kernel
                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;
                            let input_base = (ih * in_w + iw) * in_c;
                            let kernel_offset = kh * 3 + kw;

                            // 4x unrolled input channel loop
                            for ic_chunk_idx in 0..ic_chunks {
                                let ic_base = ic_chunk_idx * 4;

                                // Load 4 input values and broadcast each
                                let input_val0 =
                                    _mm256_set1_ps(*input.get_unchecked(input_base + ic_base));
                                let input_val1 =
                                    _mm256_set1_ps(*input.get_unchecked(input_base + ic_base + 1));
                                let input_val2 =
                                    _mm256_set1_ps(*input.get_unchecked(input_base + ic_base + 2));
                                let input_val3 =
                                    _mm256_set1_ps(*input.get_unchecked(input_base + ic_base + 3));

                                // Gather 8 kernel weights for each of the 4 input channels
                                let mut kv0 = [0.0f32; 8];
                                let mut kv1 = [0.0f32; 8];
                                let mut kv2 = [0.0f32; 8];
                                let mut kv3 = [0.0f32; 8];

                                for i in 0..8 {
                                    let oc_idx = oc_base + i;
                                    kv0[i] = *kernel.get_unchecked(
                                        (oc_idx * in_c + ic_base) * 9 + kernel_offset,
                                    );
                                    kv1[i] = *kernel.get_unchecked(
                                        (oc_idx * in_c + ic_base + 1) * 9 + kernel_offset,
                                    );
                                    kv2[i] = *kernel.get_unchecked(
                                        (oc_idx * in_c + ic_base + 2) * 9 + kernel_offset,
                                    );
                                    kv3[i] = *kernel.get_unchecked(
                                        (oc_idx * in_c + ic_base + 3) * 9 + kernel_offset,
                                    );
                                }

                                let kernel_v0 = _mm256_loadu_ps(kv0.as_ptr());
                                let kernel_v1 = _mm256_loadu_ps(kv1.as_ptr());
                                let kernel_v2 = _mm256_loadu_ps(kv2.as_ptr());
                                let kernel_v3 = _mm256_loadu_ps(kv3.as_ptr());

                                // FMA into 4 independent accumulators (better ILP)
                                sum0 = _mm256_fmadd_ps(input_val0, kernel_v0, sum0);
                                sum1 = _mm256_fmadd_ps(input_val1, kernel_v1, sum1);
                                sum2 = _mm256_fmadd_ps(input_val2, kernel_v2, sum2);
                                sum3 = _mm256_fmadd_ps(input_val3, kernel_v3, sum3);
                            }

                            // Handle remainder input channels (0-3 channels)
                            for ic in ic_remainder_start..in_c {
                                let input_val =
                                    _mm256_set1_ps(*input.get_unchecked(input_base + ic));

                                let mut kernel_vals = [0.0f32; 8];
                                for i in 0..8 {
                                    kernel_vals[i] = *kernel.get_unchecked(
                                        ((oc_base + i) * in_c + ic) * 9 + kernel_offset,
                                    );
                                }
                                let kernel_v = _mm256_loadu_ps(kernel_vals.as_ptr());

                                sum0 = _mm256_fmadd_ps(input_val, kernel_v, sum0);
                            }
                        }
                    }
                }

                // Combine 4 accumulators (tree reduction for better pipelining)
                let sum01 = _mm256_add_ps(sum0, sum1);
                let sum23 = _mm256_add_ps(sum2, sum3);
                let sum = _mm256_add_ps(sum01, sum23);

                // Store 8 output values
                let out_base = out_spatial_idx * out_c + oc_base;
                _mm256_storeu_ps(output.as_mut_ptr().add(out_base), sum);
            }

            // Handle remainder output channels with scalar (0-7 channels)
            for oc in (out_c_chunks * 8)..out_c {
                let mut sum = 0.0f32;

                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;

                            for ic in 0..in_c {
                                let input_idx = (ih * in_w + iw) * in_c + ic;
                                let kernel_idx = (oc * in_c + ic) * 9 + kh * 3 + kw;
                                sum += input[input_idx] * kernel[kernel_idx];
                            }
                        }
                    }
                }

                output[out_spatial_idx * out_c + oc] = sum;
            }
        }
    }
}

/// AVX2 3x3 convolution (without FMA, 4x loop unrolling + multiple accumulators)
///
/// Processes 8 output channels simultaneously using AVX2 instructions.
/// Uses 4x loop unrolling on input channels with 4 independent accumulators.
/// For CPUs without FMA support (uses mul + add instead).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn conv_3x3_avx2(
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

    let out_c_chunks = out_c / 8;
    let ic_chunks = in_c / 4;
    let ic_remainder_start = ic_chunks * 4;

    for oh in 0..out_h {
        for ow in 0..out_w {
            let out_spatial_idx = oh * out_w + ow;

            // Process 8 output channels at once
            for oc_chunk in 0..out_c_chunks {
                let oc_base = oc_chunk * 8;

                // 4 independent accumulators for ILP
                let mut sum0 = _mm256_setzero_ps();
                let mut sum1 = _mm256_setzero_ps();
                let mut sum2 = _mm256_setzero_ps();
                let mut sum3 = _mm256_setzero_ps();

                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;
                            let input_base = (ih * in_w + iw) * in_c;
                            let kernel_offset = kh * 3 + kw;

                            // 4x unrolled input channel loop
                            for ic_chunk_idx in 0..ic_chunks {
                                let ic_base = ic_chunk_idx * 4;

                                let input_val0 =
                                    _mm256_set1_ps(*input.get_unchecked(input_base + ic_base));
                                let input_val1 =
                                    _mm256_set1_ps(*input.get_unchecked(input_base + ic_base + 1));
                                let input_val2 =
                                    _mm256_set1_ps(*input.get_unchecked(input_base + ic_base + 2));
                                let input_val3 =
                                    _mm256_set1_ps(*input.get_unchecked(input_base + ic_base + 3));

                                let mut kv0 = [0.0f32; 8];
                                let mut kv1 = [0.0f32; 8];
                                let mut kv2 = [0.0f32; 8];
                                let mut kv3 = [0.0f32; 8];

                                for i in 0..8 {
                                    let oc_idx = oc_base + i;
                                    kv0[i] = *kernel.get_unchecked(
                                        (oc_idx * in_c + ic_base) * 9 + kernel_offset,
                                    );
                                    kv1[i] = *kernel.get_unchecked(
                                        (oc_idx * in_c + ic_base + 1) * 9 + kernel_offset,
                                    );
                                    kv2[i] = *kernel.get_unchecked(
                                        (oc_idx * in_c + ic_base + 2) * 9 + kernel_offset,
                                    );
                                    kv3[i] = *kernel.get_unchecked(
                                        (oc_idx * in_c + ic_base + 3) * 9 + kernel_offset,
                                    );
                                }

                                let kernel_v0 = _mm256_loadu_ps(kv0.as_ptr());
                                let kernel_v1 = _mm256_loadu_ps(kv1.as_ptr());
                                let kernel_v2 = _mm256_loadu_ps(kv2.as_ptr());
                                let kernel_v3 = _mm256_loadu_ps(kv3.as_ptr());

                                // mul + add (no FMA)
                                sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(input_val0, kernel_v0));
                                sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(input_val1, kernel_v1));
                                sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(input_val2, kernel_v2));
                                sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(input_val3, kernel_v3));
                            }

                            // Remainder input channels
                            for ic in ic_remainder_start..in_c {
                                let input_val =
                                    _mm256_set1_ps(*input.get_unchecked(input_base + ic));

                                let mut kernel_vals = [0.0f32; 8];
                                for i in 0..8 {
                                    kernel_vals[i] = *kernel.get_unchecked(
                                        ((oc_base + i) * in_c + ic) * 9 + kernel_offset,
                                    );
                                }
                                let kernel_v = _mm256_loadu_ps(kernel_vals.as_ptr());

                                sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(input_val, kernel_v));
                            }
                        }
                    }
                }

                // Combine accumulators
                let sum01 = _mm256_add_ps(sum0, sum1);
                let sum23 = _mm256_add_ps(sum2, sum3);
                let sum = _mm256_add_ps(sum01, sum23);

                let out_base = out_spatial_idx * out_c + oc_base;
                _mm256_storeu_ps(output.as_mut_ptr().add(out_base), sum);
            }

            // Remainder output channels
            for oc in (out_c_chunks * 8)..out_c {
                let mut sum = 0.0f32;
                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;
                            for ic in 0..in_c {
                                let input_idx = (ih * in_w + iw) * in_c + ic;
                                let kernel_idx = (oc * in_c + ic) * 9 + kh * 3 + kw;
                                sum += input[input_idx] * kernel[kernel_idx];
                            }
                        }
                    }
                }
                output[out_spatial_idx * out_c + oc] = sum;
            }
        }
    }
}

/// AVX2 depthwise 3x3 convolution (kernel position unrolling)
///
/// Processes 8 channels simultaneously. Each channel has its own 3x3 kernel.
/// Uses 3x3=9 kernel positions unrolled into 3 groups of 3 for better ILP.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn depthwise_conv_3x3_avx2(
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
    let c_chunks = c / 8;

    for oh in 0..out_h {
        for ow in 0..out_w {
            // Process 8 channels at a time
            for c_chunk in 0..c_chunks {
                let c_base = c_chunk * 8;

                // 3 accumulators for 3 rows (better ILP than single accumulator)
                let mut sum_row0 = _mm256_setzero_ps();
                let mut sum_row1 = _mm256_setzero_ps();
                let mut sum_row2 = _mm256_setzero_ps();

                // Pre-load kernel weights for this channel group (8 channels x 9 positions)
                let mut kernel_cache = [[_mm256_setzero_ps(); 3]; 3];
                for kh in 0..3 {
                    for kw in 0..3 {
                        let mut kvals = [0.0f32; 8];
                        for i in 0..8 {
                            kvals[i] = *kernel.get_unchecked((c_base + i) * 9 + kh * 3 + kw);
                        }
                        kernel_cache[kh][kw] = _mm256_loadu_ps(kvals.as_ptr());
                    }
                }

                // Process row 0 of kernel (kh=0)
                let ih0 = (oh * stride) as isize - padding as isize;
                if ih0 >= 0 && ih0 < h as isize {
                    let ih0 = ih0 as usize;
                    for kw in 0..3 {
                        let iw = (ow * stride + kw) as isize - padding as isize;
                        if iw >= 0 && iw < w as isize {
                            let input_base = (ih0 * w + iw as usize) * c + c_base;
                            let input_v = _mm256_loadu_ps(input.as_ptr().add(input_base));
                            sum_row0 = _mm256_add_ps(
                                sum_row0,
                                _mm256_mul_ps(input_v, kernel_cache[0][kw]),
                            );
                        }
                    }
                }

                // Process row 1 of kernel (kh=1)
                let ih1 = (oh * stride + 1) as isize - padding as isize;
                if ih1 >= 0 && ih1 < h as isize {
                    let ih1 = ih1 as usize;
                    for kw in 0..3 {
                        let iw = (ow * stride + kw) as isize - padding as isize;
                        if iw >= 0 && iw < w as isize {
                            let input_base = (ih1 * w + iw as usize) * c + c_base;
                            let input_v = _mm256_loadu_ps(input.as_ptr().add(input_base));
                            sum_row1 = _mm256_add_ps(
                                sum_row1,
                                _mm256_mul_ps(input_v, kernel_cache[1][kw]),
                            );
                        }
                    }
                }

                // Process row 2 of kernel (kh=2)
                let ih2 = (oh * stride + 2) as isize - padding as isize;
                if ih2 >= 0 && ih2 < h as isize {
                    let ih2 = ih2 as usize;
                    for kw in 0..3 {
                        let iw = (ow * stride + kw) as isize - padding as isize;
                        if iw >= 0 && iw < w as isize {
                            let input_base = (ih2 * w + iw as usize) * c + c_base;
                            let input_v = _mm256_loadu_ps(input.as_ptr().add(input_base));
                            sum_row2 = _mm256_add_ps(
                                sum_row2,
                                _mm256_mul_ps(input_v, kernel_cache[2][kw]),
                            );
                        }
                    }
                }

                // Combine row accumulators
                let sum = _mm256_add_ps(_mm256_add_ps(sum_row0, sum_row1), sum_row2);

                let out_base = (oh * out_w + ow) * c + c_base;
                _mm256_storeu_ps(output.as_mut_ptr().add(out_base), sum);
            }

            // Handle remainder channels
            for ch in (c_chunks * 8)..c {
                let mut sum = 0.0f32;
                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;
                            let input_idx = (ih * w + iw) * c + ch;
                            let kernel_idx = ch * 9 + kh * 3 + kw;
                            sum += input[input_idx] * kernel[kernel_idx];
                        }
                    }
                }
                output[(oh * out_w + ow) * c + ch] = sum;
            }
        }
    }
}

/// AVX2 global average pooling
///
/// Averages over H*W spatial dimensions, processing 8 channels at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn global_avg_pool_avx2(
    input: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
) {
    let spatial = h * w;
    let c_chunks = c / 8;
    let inv_spatial = _mm256_set1_ps(1.0 / spatial as f32);

    // Process 8 channels at a time
    for c_chunk in 0..c_chunks {
        let c_base = c_chunk * 8;
        let mut sum = _mm256_setzero_ps();

        // Sum over all spatial positions
        for s in 0..spatial {
            let input_base = s * c + c_base;
            let v = _mm256_loadu_ps(input.as_ptr().add(input_base));
            sum = _mm256_add_ps(sum, v);
        }

        // Divide by spatial size
        let avg = _mm256_mul_ps(sum, inv_spatial);
        _mm256_storeu_ps(output.as_mut_ptr().add(c_base), avg);
    }

    // Handle remainder channels
    let inv_spatial_scalar = 1.0 / spatial as f32;
    for ch in (c_chunks * 8)..c {
        let mut sum = 0.0f32;
        for s in 0..spatial {
            sum += input[s * c + ch];
        }
        output[ch] = sum * inv_spatial_scalar;
    }
}

/// AVX2 max pooling 2x2
///
/// Computes max over 2x2 windows, processing 8 channels at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn max_pool_2x2_avx2(
    input: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    stride: usize,
) {
    let out_h = (h - 2) / stride + 1;
    let out_w = (w - 2) / stride + 1;
    let c_chunks = c / 8;

    for oh in 0..out_h {
        for ow in 0..out_w {
            let ih = oh * stride;
            let iw = ow * stride;

            // Process 8 channels at a time
            for c_chunk in 0..c_chunks {
                let c_base = c_chunk * 8;

                // Load 4 values from 2x2 window for 8 channels each
                let idx00 = (ih * w + iw) * c + c_base;
                let idx01 = (ih * w + iw + 1) * c + c_base;
                let idx10 = ((ih + 1) * w + iw) * c + c_base;
                let idx11 = ((ih + 1) * w + iw + 1) * c + c_base;

                let v00 = _mm256_loadu_ps(input.as_ptr().add(idx00));
                let v01 = _mm256_loadu_ps(input.as_ptr().add(idx01));
                let v10 = _mm256_loadu_ps(input.as_ptr().add(idx10));
                let v11 = _mm256_loadu_ps(input.as_ptr().add(idx11));

                // Compute max of all 4 values
                let max01 = _mm256_max_ps(v00, v01);
                let max23 = _mm256_max_ps(v10, v11);
                let max_all = _mm256_max_ps(max01, max23);

                let out_base = (oh * out_w + ow) * c + c_base;
                _mm256_storeu_ps(output.as_mut_ptr().add(out_base), max_all);
            }

            // Handle remainder channels
            for ch in (c_chunks * 8)..c {
                let v00 = input[(ih * w + iw) * c + ch];
                let v01 = input[(ih * w + iw + 1) * c + ch];
                let v10 = input[((ih + 1) * w + iw) * c + ch];
                let v11 = input[((ih + 1) * w + iw + 1) * c + ch];
                output[(oh * out_w + ow) * c + ch] = v00.max(v01).max(v10).max(v11);
            }
        }
    }
}

// Non-x86_64 stubs to allow compilation
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot_product_avx2_fma(_a: &[f32], _b: &[f32]) -> f32 {
    0.0
}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot_product_avx2(_a: &[f32], _b: &[f32]) -> f32 {
    0.0
}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot_product_avx512(_a: &[f32], _b: &[f32]) -> f32 {
    0.0
}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn relu_avx2(_input: &[f32], _output: &mut [f32]) {}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn relu6_avx2(_input: &[f32], _output: &mut [f32]) {}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn batch_norm_avx2(
    _input: &[f32],
    _output: &mut [f32],
    _gamma: &[f32],
    _beta: &[f32],
    _mean: &[f32],
    _var: &[f32],
    _epsilon: f32,
    _channels: usize,
) {
}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn conv_3x3_avx2_fma(
    _input: &[f32],
    _kernel: &[f32],
    _output: &mut [f32],
    _in_h: usize,
    _in_w: usize,
    _in_c: usize,
    _out_c: usize,
    _stride: usize,
    _padding: usize,
) {
}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn conv_3x3_avx2(
    _input: &[f32],
    _kernel: &[f32],
    _output: &mut [f32],
    _in_h: usize,
    _in_w: usize,
    _in_c: usize,
    _out_c: usize,
    _stride: usize,
    _padding: usize,
) {
}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn depthwise_conv_3x3_avx2(
    _input: &[f32],
    _kernel: &[f32],
    _output: &mut [f32],
    _h: usize,
    _w: usize,
    _c: usize,
    _stride: usize,
    _padding: usize,
) {
}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn global_avg_pool_avx2(
    _input: &[f32],
    _output: &mut [f32],
    _h: usize,
    _w: usize,
    _c: usize,
) {
}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn max_pool_2x2_avx2(
    _input: &[f32],
    _output: &mut [f32],
    _h: usize,
    _w: usize,
    _c: usize,
    _stride: usize,
) {
}
