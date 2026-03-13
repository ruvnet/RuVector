//! WASM SIMD128 INT8 Kernels for Quantized CNN Operations
//!
//! Implements INT8 convolution using WebAssembly SIMD128 instructions:
//! - `i8x16` operations for 16-element INT8 vectors
//! - Widening multiply-add for accumulation
//! - Efficient for edge/browser deployment
//!
//! Expected speedup: 1.5-2.5x over scalar on WASM runtimes with SIMD support

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

/// WASM SIMD128 INT8 dot product
///
/// Computes: sum(a[i] * b[i]) using WASM SIMD128 i8x16 operations
///
/// # Safety
/// - Requires WASM SIMD128 support
/// - Input slices must have equal length
#[cfg(target_arch = "wasm32")]
pub unsafe fn dot_product_int8_wasm(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 16; // Process 16 elements per iteration

    // Accumulator (4 x i32)
    let mut acc = i32x4_splat(0);

    for i in 0..chunks {
        let offset = i * 16;

        // Load 16 i8 values
        let va = v128_load(a.as_ptr().add(offset) as *const v128);
        let vb = v128_load(b.as_ptr().add(offset) as *const v128);

        // Extract low and high 8 bytes
        let va_low = i16x8_extend_low_i8x16(va);
        let va_high = i16x8_extend_high_i8x16(va);
        let vb_low = i16x8_extend_low_i8x16(vb);
        let vb_high = i16x8_extend_high_i8x16(vb);

        // Multiply i16 * i16 -> i16
        let prod_low = i16x8_mul(va_low, vb_low);
        let prod_high = i16x8_mul(va_high, vb_high);

        // Extend to i32 and accumulate
        let prod32_low_lo = i32x4_extend_low_i16x8(prod_low);
        let prod32_low_hi = i32x4_extend_high_i16x8(prod_low);
        let prod32_high_lo = i32x4_extend_low_i16x8(prod_high);
        let prod32_high_hi = i32x4_extend_high_i16x8(prod_high);

        acc = i32x4_add(acc, prod32_low_lo);
        acc = i32x4_add(acc, prod32_low_hi);
        acc = i32x4_add(acc, prod32_high_lo);
        acc = i32x4_add(acc, prod32_high_hi);
    }

    // Horizontal sum of 4 x i32
    let mut result = i32x4_extract_lane::<0>(acc)
        + i32x4_extract_lane::<1>(acc)
        + i32x4_extract_lane::<2>(acc)
        + i32x4_extract_lane::<3>(acc);

    // Handle remainder
    for i in (chunks * 16)..len {
        result += (a[i] as i32) * (b[i] as i32);
    }

    result
}

/// WASM SIMD128 INT8 2D convolution
///
/// Processes 4 output channels at a time using WASM SIMD128.
///
/// # Safety
/// Requires WASM SIMD128 support.
#[cfg(target_arch = "wasm32")]
pub unsafe fn conv2d_int8_wasm(
    input: &[u8],
    input_zero_point: i32,
    kernel: &[i8],
    bias_i32: &[i32],
    output: &mut [i32],
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
    let kernel_size = 3;

    // Pre-compute weight sums
    let mut weight_sums = vec![0i32; out_c];
    for oc in 0..out_c {
        let mut sum = 0i32;
        for ic in 0..in_c {
            for kh in 0..3 {
                for kw in 0..3 {
                    let idx = (oc * in_c + ic) * 9 + kh * 3 + kw;
                    sum += kernel[idx] as i32;
                }
            }
        }
        weight_sums[oc] = sum;
    }

    for oh in 0..out_h {
        for ow in 0..out_w {
            let out_spatial_idx = oh * out_w + ow;

            // Process 4 output channels
            for oc_chunk in 0..out_c_chunks {
                let oc_base = oc_chunk * 4;

                let mut acc = [0i32; 4];
                for i in 0..4 {
                    let oc = oc_base + i;
                    acc[i] = bias_i32[oc] - input_zero_point * weight_sums[oc];
                }

                // 3x3 convolution
                for kh in 0..kernel_size {
                    for kw in 0..kernel_size {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;

                            // Process input channels in groups of 16
                            let ic_chunks = in_c / 16;

                            for ic_chunk in 0..ic_chunks {
                                let ic_base = ic_chunk * 16;
                                let input_base = (ih * in_w + iw) * in_c + ic_base;

                                // Load 16 u8 inputs and convert to i8
                                let input_u8 =
                                    v128_load(input.as_ptr().add(input_base) as *const v128);
                                let offset = u8x16_splat(128);
                                let input_shifted = u8x16_sub(input_u8, offset);

                                for i in 0..4 {
                                    let oc = oc_base + i;

                                    // Load 16 i8 weights
                                    let mut w_buf = [0i8; 16];
                                    for j in 0..16 {
                                        let k_idx = (oc * in_c + ic_base + j) * 9 + kh * 3 + kw;
                                        w_buf[j] = kernel[k_idx];
                                    }
                                    let kernel_i8 = v128_load(w_buf.as_ptr() as *const v128);

                                    // Reinterpret input_shifted as i8x16
                                    let input_i8 = input_shifted;

                                    // Extend and multiply
                                    let input_low = i16x8_extend_low_i8x16(input_i8);
                                    let input_high = i16x8_extend_high_i8x16(input_i8);
                                    let kernel_low = i16x8_extend_low_i8x16(kernel_i8);
                                    let kernel_high = i16x8_extend_high_i8x16(kernel_i8);

                                    let prod_low = i16x8_mul(input_low, kernel_low);
                                    let prod_high = i16x8_mul(input_high, kernel_high);

                                    // Extend to i32 and sum
                                    let sum_ll = i32x4_extend_low_i16x8(prod_low);
                                    let sum_lh = i32x4_extend_high_i16x8(prod_low);
                                    let sum_hl = i32x4_extend_low_i16x8(prod_high);
                                    let sum_hh = i32x4_extend_high_i16x8(prod_high);

                                    let total1 = i32x4_add(sum_ll, sum_lh);
                                    let total2 = i32x4_add(sum_hl, sum_hh);
                                    let total = i32x4_add(total1, total2);

                                    // Horizontal sum
                                    let sum = i32x4_extract_lane::<0>(total)
                                        + i32x4_extract_lane::<1>(total)
                                        + i32x4_extract_lane::<2>(total)
                                        + i32x4_extract_lane::<3>(total);

                                    acc[i] += sum;
                                }
                            }

                            // Remainder input channels
                            for ic in (ic_chunks * 16)..in_c {
                                let input_idx = (ih * in_w + iw) * in_c + ic;
                                let input_val = input[input_idx] as i32;

                                for i in 0..4 {
                                    let oc = oc_base + i;
                                    let k_idx = (oc * in_c + ic) * 9 + kh * 3 + kw;
                                    let w_val = kernel[k_idx] as i32;
                                    acc[i] += input_val * w_val;
                                }
                            }
                        }
                    }
                }

                for i in 0..4 {
                    output[out_spatial_idx * out_c + oc_base + i] = acc[i];
                }
            }

            // Remainder output channels
            for oc in (out_c_chunks * 4)..out_c {
                let mut acc = bias_i32[oc] - input_zero_point * weight_sums[oc];

                for kh in 0..kernel_size {
                    for kw in 0..kernel_size {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;

                            for ic in 0..in_c {
                                let input_idx = (ih * in_w + iw) * in_c + ic;
                                let k_idx = (oc * in_c + ic) * 9 + kh * 3 + kw;
                                acc += (input[input_idx] as i32) * (kernel[k_idx] as i32);
                            }
                        }
                    }
                }

                output[out_spatial_idx * out_c + oc] = acc;
            }
        }
    }
}

/// WASM SIMD128 depthwise convolution
///
/// # Safety
/// Requires WASM SIMD128 support.
#[cfg(target_arch = "wasm32")]
pub unsafe fn depthwise_conv2d_int8_wasm(
    input: &[u8],
    input_zero_point: i32,
    kernel: &[i8],
    bias_i32: &[i32],
    output: &mut [i32],
    h: usize,
    w: usize,
    c: usize,
    stride: usize,
    padding: usize,
) {
    let out_h = (h + 2 * padding - 3) / stride + 1;
    let out_w = (w + 2 * padding - 3) / stride + 1;

    // Pre-compute weight sums
    let mut weight_sums = vec![0i32; c];
    for ch in 0..c {
        let mut sum = 0i32;
        for k in 0..9 {
            sum += kernel[ch * 9 + k] as i32;
        }
        weight_sums[ch] = sum;
    }

    for oh in 0..out_h {
        for ow in 0..out_w {
            for ch in 0..c {
                let mut acc = bias_i32[ch] - input_zero_point * weight_sums[ch];

                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;
                            let input_idx = (ih * w + iw) * c + ch;
                            let kernel_idx = ch * 9 + kh * 3 + kw;
                            acc += (input[input_idx] as i32) * (kernel[kernel_idx] as i32);
                        }
                    }
                }

                output[(oh * out_w + ow) * c + ch] = acc;
            }
        }
    }
}

/// WASM SIMD128 matrix multiplication
///
/// # Safety
/// Requires WASM SIMD128 support.
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_int8_wasm(
    a: &[i8],
    b: &[i8],
    output: &mut [i32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(output.len(), m * n);

    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];

        for col in 0..n {
            let mut sum = 0i32;

            // Process K in chunks of 16
            let k_chunks = k / 16;

            for k_chunk in 0..k_chunks {
                let k_base = k_chunk * 16;

                let va = v128_load(a_row.as_ptr().add(k_base) as *const v128);

                let mut w_buf = [0i8; 16];
                for j in 0..16 {
                    w_buf[j] = b[(k_base + j) * n + col];
                }
                let vb = v128_load(w_buf.as_ptr() as *const v128);

                let va_low = i16x8_extend_low_i8x16(va);
                let va_high = i16x8_extend_high_i8x16(va);
                let vb_low = i16x8_extend_low_i8x16(vb);
                let vb_high = i16x8_extend_high_i8x16(vb);

                let prod_low = i16x8_mul(va_low, vb_low);
                let prod_high = i16x8_mul(va_high, vb_high);

                let sum_ll = i32x4_extend_low_i16x8(prod_low);
                let sum_lh = i32x4_extend_high_i16x8(prod_low);
                let sum_hl = i32x4_extend_low_i16x8(prod_high);
                let sum_hh = i32x4_extend_high_i16x8(prod_high);

                let total1 = i32x4_add(sum_ll, sum_lh);
                let total2 = i32x4_add(sum_hl, sum_hh);
                let total = i32x4_add(total1, total2);

                sum += i32x4_extract_lane::<0>(total)
                    + i32x4_extract_lane::<1>(total)
                    + i32x4_extract_lane::<2>(total)
                    + i32x4_extract_lane::<3>(total);
            }

            // Remainder K
            for k_idx in (k_chunks * 16)..k {
                sum += (a_row[k_idx] as i32) * (b[k_idx * n + col] as i32);
            }

            output[row * n + col] = sum;
        }
    }
}

// Non-wasm32 stubs
#[cfg(not(target_arch = "wasm32"))]
pub unsafe fn dot_product_int8_wasm(_a: &[i8], _b: &[i8]) -> i32 {
    0
}

#[cfg(not(target_arch = "wasm32"))]
pub unsafe fn conv2d_int8_wasm(
    _input: &[u8],
    _input_zero_point: i32,
    _kernel: &[i8],
    _bias_i32: &[i32],
    _output: &mut [i32],
    _in_h: usize,
    _in_w: usize,
    _in_c: usize,
    _out_c: usize,
    _stride: usize,
    _padding: usize,
) {
}

#[cfg(not(target_arch = "wasm32"))]
pub unsafe fn depthwise_conv2d_int8_wasm(
    _input: &[u8],
    _input_zero_point: i32,
    _kernel: &[i8],
    _bias_i32: &[i32],
    _output: &mut [i32],
    _h: usize,
    _w: usize,
    _c: usize,
    _stride: usize,
    _padding: usize,
) {
}

#[cfg(not(target_arch = "wasm32"))]
pub unsafe fn matmul_int8_wasm(
    _a: &[i8],
    _b: &[i8],
    _output: &mut [i32],
    _m: usize,
    _k: usize,
    _n: usize,
) {
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_int8_wasm() {
        let a: Vec<i8> = (0..32).map(|i| (i % 64) as i8).collect();
        let b: Vec<i8> = (0..32).map(|i| ((i % 64) as i8)).collect();

        #[cfg(target_arch = "wasm32")]
        {
            unsafe {
                let result = dot_product_int8_wasm(&a, &b);

                let mut expected = 0i32;
                for i in 0..32 {
                    expected += (a[i] as i32) * (b[i] as i32);
                }

                assert_eq!(result, expected);
            }
        }
    }
}
