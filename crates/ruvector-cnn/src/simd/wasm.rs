//! WASM SIMD Implementations
//!
//! 128-bit SIMD operations for WebAssembly targets.
//! Uses wasm32 SIMD intrinsics when available.

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

// ============================================================================
// Dot Product
// ============================================================================

/// WASM SIMD dot product (4 floats per iteration)
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn dot_product_wasm(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let mut sum = f32x4_splat(0.0);

    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        unsafe {
            let va = v128_load(a[idx..].as_ptr() as *const v128);
            let vb = v128_load(b[idx..].as_ptr() as *const v128);
            let prod = f32x4_mul(va, vb);
            sum = f32x4_add(sum, prod);
        }
    }

    // Horizontal sum
    let mut total = f32x4_extract_lane::<0>(sum)
        + f32x4_extract_lane::<1>(sum)
        + f32x4_extract_lane::<2>(sum)
        + f32x4_extract_lane::<3>(sum);

    // Handle remainder
    for i in (chunks * 4)..len {
        total += a[i] * b[i];
    }

    total
}

// ============================================================================
// ReLU Activation
// ============================================================================

/// WASM SIMD ReLU: max(0, x)
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn relu_wasm(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    let len = input.len();
    let zero = f32x4_splat(0.0);

    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        unsafe {
            let v = v128_load(input[idx..].as_ptr() as *const v128);
            let result = f32x4_max(v, zero);
            v128_store(output[idx..].as_mut_ptr() as *mut v128, result);
        }
    }

    // Handle remainder
    for i in (chunks * 4)..len {
        output[i] = input[i].max(0.0);
    }
}

/// WASM SIMD ReLU6: min(6, max(0, x))
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn relu6_wasm(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    let len = input.len();
    let zero = f32x4_splat(0.0);
    let six = f32x4_splat(6.0);

    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        unsafe {
            let v = v128_load(input[idx..].as_ptr() as *const v128);
            let clamped_low = f32x4_max(v, zero);
            let clamped = f32x4_min(clamped_low, six);
            v128_store(output[idx..].as_mut_ptr() as *mut v128, clamped);
        }
    }

    // Handle remainder
    for i in (chunks * 4)..len {
        output[i] = input[i].max(0.0).min(6.0);
    }
}

// ============================================================================
// Batch Normalization
// ============================================================================

/// WASM SIMD batch normalization
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn batch_norm_wasm(
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

                unsafe {
                    let v = v128_load(input[idx..].as_ptr() as *const v128);
                    let scale_v = v128_load(scale[c..].as_ptr() as *const v128);
                    let shift_v = v128_load(shift[c..].as_ptr() as *const v128);

                    let result = f32x4_add(f32x4_mul(v, scale_v), shift_v);
                    v128_store(output[idx..].as_mut_ptr() as *mut v128, result);
                }
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

/// WASM SIMD 3x3 convolution (NHWC layout)
///
/// Processes 4 output channels simultaneously using WASM SIMD128.
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn conv_3x3_wasm(
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
            let out_spatial_idx = oh * out_w + ow;

            // Process 4 output channels at once
            for oc_chunk in 0..out_c_chunks {
                let oc_base = oc_chunk * 4;
                let mut sum = f32x4_splat(0.0);

                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;

                            for ic in 0..in_c {
                                let input_idx = (ih * in_w + iw) * in_c + ic;
                                let input_val = f32x4_splat(input[input_idx]);

                                // Gather 4 kernel weights
                                let mut kernel_vals = [0.0f32; 4];
                                for i in 0..4 {
                                    let k_idx = ((oc_base + i) * in_c + ic) * 9 + kh * 3 + kw;
                                    if k_idx < kernel.len() {
                                        kernel_vals[i] = kernel[k_idx];
                                    }
                                }
                                unsafe {
                                    let kernel_v = v128_load(kernel_vals.as_ptr() as *const v128);
                                    let prod = f32x4_mul(input_val, kernel_v);
                                    sum = f32x4_add(sum, prod);
                                }
                            }
                        }
                    }
                }

                let out_base = out_spatial_idx * out_c + oc_base;
                unsafe {
                    v128_store(output[out_base..].as_mut_ptr() as *mut v128, sum);
                }
            }

            // Remainder channels
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

// ============================================================================
// Depthwise Convolution
// ============================================================================

/// WASM depthwise 3x3 convolution
///
/// Processes 4 channels simultaneously using WASM SIMD128.
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn depthwise_conv_3x3_wasm(
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
            for c_chunk in 0..c_chunks {
                let c_base = c_chunk * 4;
                let mut sum = f32x4_splat(0.0);

                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;

                            // Load 4 input values
                            let input_base = (ih * w + iw) * c + c_base;

                            // Load 4 kernel weights
                            let mut kernel_vals = [0.0f32; 4];
                            for i in 0..4 {
                                kernel_vals[i] = kernel[(c_base + i) * 9 + kh * 3 + kw];
                            }

                            unsafe {
                                let input_v =
                                    v128_load(input[input_base..].as_ptr() as *const v128);
                                let kernel_v = v128_load(kernel_vals.as_ptr() as *const v128);

                                let prod = f32x4_mul(input_v, kernel_v);
                                sum = f32x4_add(sum, prod);
                            }
                        }
                    }
                }

                let out_base = (oh * out_w + ow) * c + c_base;
                unsafe {
                    v128_store(output[out_base..].as_mut_ptr() as *mut v128, sum);
                }
            }

            // Remainder channels
            for ch in (c_chunks * 4)..c {
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

// ============================================================================
// Pooling Operations
// ============================================================================

/// WASM global average pooling
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn global_avg_pool_wasm(input: &[f32], output: &mut [f32], h: usize, w: usize, c: usize) {
    let spatial_size = h * w;
    let inv_spatial = 1.0 / spatial_size as f32;
    let inv_spatial_v = f32x4_splat(inv_spatial);

    let c_chunks = c / 4;

    // Initialize output
    for i in 0..c_chunks {
        unsafe {
            v128_store(output[i * 4..].as_mut_ptr() as *mut v128, f32x4_splat(0.0));
        }
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
                unsafe {
                    let input_v = v128_load(input[base + ch_base..].as_ptr() as *const v128);
                    let out_v = v128_load(output[ch_base..].as_ptr() as *const v128);
                    let sum_v = f32x4_add(out_v, input_v);
                    v128_store(output[ch_base..].as_mut_ptr() as *mut v128, sum_v);
                }
            }

            for ch in (c_chunks * 4)..c {
                output[ch] += input[base + ch];
            }
        }
    }

    // Multiply by 1/spatial_size
    for cc in 0..c_chunks {
        let ch_base = cc * 4;
        unsafe {
            let sum_v = v128_load(output[ch_base..].as_ptr() as *const v128);
            let avg_v = f32x4_mul(sum_v, inv_spatial_v);
            v128_store(output[ch_base..].as_mut_ptr() as *mut v128, avg_v);
        }
    }
    for ch in (c_chunks * 4)..c {
        output[ch] *= inv_spatial;
    }
}

/// WASM max pooling 2x2
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn max_pool_2x2_wasm(
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

                unsafe {
                    let v00 = v128_load(input[idx00..].as_ptr() as *const v128);
                    let v01 = v128_load(input[idx01..].as_ptr() as *const v128);
                    let v10 = v128_load(input[idx10..].as_ptr() as *const v128);
                    let v11 = v128_load(input[idx11..].as_ptr() as *const v128);

                    let max01 = f32x4_max(v00, v01);
                    let max23 = f32x4_max(v10, v11);
                    let max_val = f32x4_max(max01, max23);

                    let out_idx = oh * out_w * c + ow * c + ch_base;
                    v128_store(output[out_idx..].as_mut_ptr() as *mut v128, max_val);
                }
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

// Stub implementations for non-wasm32 targets
#[cfg(not(target_arch = "wasm32"))]
pub fn dot_product_wasm(_a: &[f32], _b: &[f32]) -> f32 {
    unimplemented!("WASM SIMD not available on this architecture")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn relu_wasm(_input: &[f32], _output: &mut [f32]) {
    unimplemented!("WASM SIMD not available on this architecture")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn relu6_wasm(_input: &[f32], _output: &mut [f32]) {
    unimplemented!("WASM SIMD not available on this architecture")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn batch_norm_wasm(
    _input: &[f32],
    _output: &mut [f32],
    _gamma: &[f32],
    _beta: &[f32],
    _mean: &[f32],
    _var: &[f32],
    _epsilon: f32,
    _channels: usize,
) {
    unimplemented!("WASM SIMD not available on this architecture")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn conv_3x3_wasm(
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
    unimplemented!("WASM SIMD not available on this architecture")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn depthwise_conv_3x3_wasm(
    _input: &[f32],
    _kernel: &[f32],
    _output: &mut [f32],
    _h: usize,
    _w: usize,
    _c: usize,
    _stride: usize,
    _padding: usize,
) {
    unimplemented!("WASM SIMD not available on this architecture")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn global_avg_pool_wasm(_input: &[f32], _output: &mut [f32], _h: usize, _w: usize, _c: usize) {
    unimplemented!("WASM SIMD not available on this architecture")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn max_pool_2x2_wasm(
    _input: &[f32],
    _output: &mut [f32],
    _h: usize,
    _w: usize,
    _c: usize,
    _stride: usize,
) {
    unimplemented!("WASM SIMD not available on this architecture")
}
