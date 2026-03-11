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

/// AVX2 3x3 convolution with FMA
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
    // For now, use scalar fallback
    // Full SIMD implementation would be more complex
    crate::simd::scalar::conv_3x3_scalar(input, kernel, output, in_h, in_w, in_c, out_c, stride, padding);
}

/// AVX2 3x3 convolution
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
    crate::simd::scalar::conv_3x3_scalar(input, kernel, output, in_h, in_w, in_c, out_c, stride, padding);
}

/// AVX2 depthwise 3x3 convolution
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
    crate::simd::scalar::depthwise_conv_3x3_scalar(input, kernel, output, h, w, c, stride, padding);
}

/// AVX2 global average pooling
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn global_avg_pool_avx2(input: &[f32], output: &mut [f32], h: usize, w: usize, c: usize) {
    crate::simd::scalar::global_avg_pool_scalar(input, output, h, w, c);
}

/// AVX2 max pooling 2x2
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
    crate::simd::scalar::max_pool_2x2_scalar(input, output, h, w, c, stride);
}

// Non-x86_64 stubs to allow compilation
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot_product_avx2_fma(_a: &[f32], _b: &[f32]) -> f32 { 0.0 }
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot_product_avx2(_a: &[f32], _b: &[f32]) -> f32 { 0.0 }
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot_product_avx512(_a: &[f32], _b: &[f32]) -> f32 { 0.0 }
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn relu_avx2(_input: &[f32], _output: &mut [f32]) {}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn relu6_avx2(_input: &[f32], _output: &mut [f32]) {}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn batch_norm_avx2(_input: &[f32], _output: &mut [f32], _gamma: &[f32], _beta: &[f32], _mean: &[f32], _var: &[f32], _epsilon: f32, _channels: usize) {}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn conv_3x3_avx2_fma(_input: &[f32], _kernel: &[f32], _output: &mut [f32], _in_h: usize, _in_w: usize, _in_c: usize, _out_c: usize, _stride: usize, _padding: usize) {}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn conv_3x3_avx2(_input: &[f32], _kernel: &[f32], _output: &mut [f32], _in_h: usize, _in_w: usize, _in_c: usize, _out_c: usize, _stride: usize, _padding: usize) {}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn depthwise_conv_3x3_avx2(_input: &[f32], _kernel: &[f32], _output: &mut [f32], _h: usize, _w: usize, _c: usize, _stride: usize, _padding: usize) {}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn global_avg_pool_avx2(_input: &[f32], _output: &mut [f32], _h: usize, _w: usize, _c: usize) {}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn max_pool_2x2_avx2(_input: &[f32], _output: &mut [f32], _h: usize, _w: usize, _c: usize, _stride: usize) {}
