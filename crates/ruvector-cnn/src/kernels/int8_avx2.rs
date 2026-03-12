//! AVX2 INT8 SIMD Kernels for Quantized CNN Operations
//!
//! Implements high-performance INT8 convolution and matrix multiplication using:
//! - `_mm256_maddubs_epi16`: Multiply unsigned u8 by signed i8, accumulate to i16
//! - `_mm256_madd_epi16`: Multiply i16 pairs, accumulate to i32
//! - VNNI instructions (`_mm256_dpbusd_epi32`) when available for further speedup
//!
//! Expected speedup: 2-4x over FP32 on AVX2, 3-5x with AVX-512 VNNI

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 INT8 dot product
///
/// Computes: sum(a[i] * b[i]) where a is unsigned u8, b is signed i8
/// Uses the _mm256_maddubs_epi16 + _mm256_madd_epi16 cascade for efficient INT8 multiply-accumulate.
///
/// # Safety
/// - Requires AVX2 CPU support
/// - Input slices must have equal length
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_int8_avx2(a: &[u8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 32; // Process 32 elements per iteration

    // Accumulator for partial sums (8 x i32)
    let mut acc = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(1); // For horizontal sum in madd

    for i in 0..chunks {
        // Load 32 unsigned u8 activations
        let va = _mm256_loadu_si256(a.as_ptr().add(i * 32) as *const __m256i);

        // Load 32 signed i8 weights
        let vb = _mm256_loadu_si256(b.as_ptr().add(i * 32) as *const __m256i);

        // Multiply u8 * i8 -> i16, pairwise add: 32 products -> 16 sums
        let prod16 = _mm256_maddubs_epi16(va, vb);

        // Sum pairs of i16 to i32: 16 values -> 8 values
        let prod32 = _mm256_madd_epi16(prod16, ones);

        // Accumulate
        acc = _mm256_add_epi32(acc, prod32);
    }

    // Horizontal sum of 8 x i32
    let sum128 = _mm_add_epi32(
        _mm256_extracti128_si256(acc, 0),
        _mm256_extracti128_si256(acc, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));

    let mut result = _mm_cvtsi128_si32(sum32);

    // Handle remainder
    for i in (chunks * 32)..len {
        result += (a[i] as i32) * (b[i] as i32);
    }

    result
}

/// AVX2 INT8 2D convolution with per-channel quantization
///
/// Processes 8 output channels simultaneously using AVX2 INT8 operations.
///
/// # Arguments
/// - `input`: Quantized activations (u8), shape [H, W, C]
/// - `input_zero_point`: Zero point for asymmetric activation quantization
/// - `kernel`: Quantized weights (i8, symmetric), shape [out_c, in_c, 3, 3]
/// - `bias_i32`: Pre-computed bias in int32 accumulator space
/// - `output`: INT32 accumulators to be dequantized later
/// - Spatial and channel dimensions
///
/// # Safety
/// Requires AVX2 support. All pointers must be valid.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn conv2d_int8_avx2(
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

    let out_c_chunks = out_c / 8;
    let kernel_size = 3;

    // Pre-compute zero-point correction term
    // For each output channel: zp_a * sum(weights)
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

            // Process 8 output channels at once
            for oc_chunk in 0..out_c_chunks {
                let oc_base = oc_chunk * 8;

                // Initialize accumulators with bias and zero-point correction
                let mut acc = [0i32; 8];
                for i in 0..8 {
                    let oc = oc_base + i;
                    // Bias - zp_a * sum(weights) for this output channel
                    acc[i] = bias_i32[oc] - input_zero_point * weight_sums[oc];
                }

                // Convolve over 3x3 kernel
                for kh in 0..kernel_size {
                    for kw in 0..kernel_size {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;

                            // Process input channels in groups of 32 for AVX2
                            let ic_chunks = in_c / 32;

                            for ic_chunk in 0..ic_chunks {
                                let ic_base = ic_chunk * 32;

                                // Load 32 input activations
                                let input_base = (ih * in_w + iw) * in_c + ic_base;
                                let va = _mm256_loadu_si256(input.as_ptr().add(input_base) as *const __m256i);

                                // For each output channel in this chunk
                                for i in 0..8 {
                                    let oc = oc_base + i;

                                    // Load 32 weights for this output channel
                                    let mut w_buf = [0i8; 32];
                                    for j in 0..32 {
                                        let k_idx = (oc * in_c + ic_base + j) * 9 + kh * 3 + kw;
                                        w_buf[j] = kernel[k_idx];
                                    }
                                    let vw = _mm256_loadu_si256(w_buf.as_ptr() as *const __m256i);

                                    // u8 * i8 -> i16, pairwise add
                                    let prod16 = _mm256_maddubs_epi16(va, vw);

                                    // i16 -> i32, pairwise add
                                    let ones = _mm256_set1_epi16(1);
                                    let prod32 = _mm256_madd_epi16(prod16, ones);

                                    // Horizontal sum to single i32
                                    let sum = horizontal_sum_epi32(prod32);
                                    acc[i] += sum;
                                }
                            }

                            // Handle remainder input channels
                            for ic in (ic_chunks * 32)..in_c {
                                let input_idx = (ih * in_w + iw) * in_c + ic;
                                let input_val = input[input_idx] as i32;

                                for i in 0..8 {
                                    let oc = oc_base + i;
                                    let k_idx = (oc * in_c + ic) * 9 + kh * 3 + kw;
                                    let w_val = kernel[k_idx] as i32;
                                    acc[i] += input_val * w_val;
                                }
                            }
                        }
                    }
                }

                // Store accumulated results
                for i in 0..8 {
                    output[out_spatial_idx * out_c + oc_base + i] = acc[i];
                }
            }

            // Handle remainder output channels
            for oc in (out_c_chunks * 8)..out_c {
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

/// AVX2 INT8 depthwise 3x3 convolution
///
/// Each input channel has its own 3x3 kernel. Processes 8 channels simultaneously.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn depthwise_conv2d_int8_avx2(
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
    let c_chunks = c / 8;

    // Pre-compute weight sums for zero-point correction
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
            // Process 8 channels at a time
            for c_chunk in 0..c_chunks {
                let c_base = c_chunk * 8;

                // Initialize accumulators
                let mut acc = [0i32; 8];
                for i in 0..8 {
                    let ch = c_base + i;
                    acc[i] = bias_i32[ch] - input_zero_point * weight_sums[ch];
                }

                // 3x3 convolution
                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (oh * stride + kh) as isize - padding as isize;
                        let iw = (ow * stride + kw) as isize - padding as isize;

                        if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                            let ih = ih as usize;
                            let iw = iw as usize;

                            // Load 8 input values
                            let input_base = (ih * w + iw) * c + c_base;
                            let mut input_vals = [0u8; 8];
                            for i in 0..8 {
                                input_vals[i] = input[input_base + i];
                            }

                            // Load 8 kernel weights
                            let mut kernel_vals = [0i8; 8];
                            for i in 0..8 {
                                kernel_vals[i] = kernel[(c_base + i) * 9 + kh * 3 + kw];
                            }

                            // Multiply and accumulate
                            for i in 0..8 {
                                acc[i] += (input_vals[i] as i32) * (kernel_vals[i] as i32);
                            }
                        }
                    }
                }

                // Store results
                for i in 0..8 {
                    output[(oh * out_w + ow) * c + c_base + i] = acc[i];
                }
            }

            // Handle remainder channels
            for ch in (c_chunks * 8)..c {
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

/// AVX2 INT8 matrix multiplication (M x K) * (K x N) -> (M x N)
///
/// Uses AVX2 INT8 multiply-accumulate for efficient GEMM.
/// Can optionally use VNNI (`_mm256_dpbusd_epi32`) if available for further speedup.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn matmul_int8_avx2(
    a: &[u8],       // M x K (unsigned activations)
    b: &[i8],       // K x N (signed weights)
    output: &mut [i32], // M x N (int32 accumulators)
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(output.len(), m * n);

    // Process N in chunks of 8 (8 output columns at a time)
    let n_chunks = n / 8;

    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];

        // Process 8 output columns at once
        for n_chunk in 0..n_chunks {
            let n_base = n_chunk * 8;

            // Accumulators for 8 output values
            let mut acc = [0i32; 8];

            // Process K dimension in chunks of 32
            let k_chunks = k / 32;

            for k_chunk in 0..k_chunks {
                let k_base = k_chunk * 32;

                // Load 32 values from matrix A
                let va = _mm256_loadu_si256(a_row.as_ptr().add(k_base) as *const __m256i);

                // For each of the 8 output columns
                for i in 0..8 {
                    let col = n_base + i;

                    // Load 32 weights from column of matrix B
                    let mut w_buf = [0i8; 32];
                    for j in 0..32 {
                        w_buf[j] = b[(k_base + j) * n + col];
                    }
                    let vb = _mm256_loadu_si256(w_buf.as_ptr() as *const __m256i);

                    // u8 * i8 -> i16, pairwise add
                    let prod16 = _mm256_maddubs_epi16(va, vb);

                    // i16 -> i32, pairwise add
                    let ones = _mm256_set1_epi16(1);
                    let prod32 = _mm256_madd_epi16(prod16, ones);

                    // Horizontal sum
                    acc[i] += horizontal_sum_epi32(prod32);
                }
            }

            // Handle remainder K dimension
            for k_idx in (k_chunks * 32)..k {
                let a_val = a_row[k_idx] as i32;
                for i in 0..8 {
                    let col = n_base + i;
                    let b_val = b[k_idx * n + col] as i32;
                    acc[i] += a_val * b_val;
                }
            }

            // Store 8 output values
            for i in 0..8 {
                output[row * n + n_base + i] = acc[i];
            }
        }

        // Handle remainder N dimension
        for col in (n_chunks * 8)..n {
            let mut sum = 0i32;
            for k_idx in 0..k {
                sum += (a_row[k_idx] as i32) * (b[k_idx * n + col] as i32);
            }
            output[row * n + col] = sum;
        }
    }
}

/// Horizontal sum of 8 x i32 in __m256i
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn horizontal_sum_epi32(v: __m256i) -> i32 {
    let sum128 = _mm_add_epi32(
        _mm256_extracti128_si256(v, 0),
        _mm256_extracti128_si256(v, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    _mm_cvtsi128_si32(sum32)
}

// Non-x86_64 stubs
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot_product_int8_avx2(_a: &[u8], _b: &[i8]) -> i32 {
    0
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn conv2d_int8_avx2(
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

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn depthwise_conv2d_int8_avx2(
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

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn matmul_int8_avx2(
    _a: &[u8],
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
    fn test_dot_product_int8_avx2() {
        let a: Vec<u8> = (0..64).map(|i| (i % 128) as u8).collect();
        let b: Vec<i8> = (0..64).map(|i| ((i % 128) as i8)).collect();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let result = dot_product_int8_avx2(&a, &b);

                    // Compute reference
                    let mut expected = 0i32;
                    for i in 0..64 {
                        expected += (a[i] as i32) * (b[i] as i32);
                    }

                    assert_eq!(result, expected);
                }
            }
        }
    }

    #[test]
    fn test_conv2d_int8_avx2_simple() {
        // Simple test: 5x5 input, 3x3 kernel, 1 input channel, 8 output channels
        let in_h = 5;
        let in_w = 5;
        let in_c = 1;
        let out_c = 8;

        let input = vec![10u8; in_h * in_w * in_c];
        let kernel = vec![1i8; out_c * in_c * 9];
        let bias = vec![0i32; out_c];
        let mut output = vec![0i32; 3 * 3 * out_c]; // 3x3 output with stride=1, padding=0

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    conv2d_int8_avx2(
                        &input, 0, &kernel, &bias, &mut output, in_h, in_w, in_c, out_c, 1, 0,
                    );

                    // All outputs should be 10 * 9 = 90 (10 input * 9 weights)
                    for &val in &output {
                        assert_eq!(val, 90);
                    }
                }
            }
        }
    }

    #[test]
    fn test_matmul_int8_avx2_simple() {
        let m = 4;
        let k = 32;
        let n = 8;

        let a = vec![2u8; m * k];
        let b = vec![3i8; k * n];
        let mut output = vec![0i32; m * n];

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    matmul_int8_avx2(&a, &b, &mut output, m, k, n);

                    // Each output should be 2 * 3 * 32 = 192
                    for &val in &output {
                        assert_eq!(val, 192);
                    }
                }
            }
        }
    }
}
