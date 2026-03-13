//! SIMD Backend Dispatch Module
//!
//! Provides architecture-specific SIMD implementations with automatic dispatch:
//! - AVX-512 for modern Intel/AMD (16 floats per iteration)
//! - AVX2 with FMA for Intel Haswell+ / AMD Zen+ (8 floats per iteration)
//! - NEON for ARM64/Apple Silicon (4 floats per iteration)
//! - WASM SIMD for WebAssembly (4 floats per iteration)
//! - Winograd F(2,3) for 2.25x faster 3x3 convolutions
//! - Scalar fallback for all other platforms

pub mod avx2;
pub mod quantize;
pub mod scalar;
pub mod winograd;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export the dispatch functions
pub use avx2::*;
pub use quantize::{
    dequantize_batch, dequantize_simd, pi_constants, quantize_batch, quantize_simd,
    PerChannelQuantParams, QuantParams, QuantizationType, QuantizedTensor,
};
pub use scalar::*;
pub use winograd::{
    conv_3x3_winograd, transform_filter, transform_input, transform_output, WinogradFilterCache,
};

/// SIMD-accelerated dot product with automatic architecture dispatch
#[inline(always)]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { avx2::dot_product_avx512(a, b) }
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::dot_product_avx2_fma(a, b) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { avx2::dot_product_avx2(a, b) }
        } else {
            scalar::dot_product_scalar(a, b)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::dot_product_neon(a, b) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        wasm::dot_product_wasm(a, b)
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        scalar::dot_product_scalar(a, b)
    }
}

/// SIMD-accelerated ReLU activation with automatic architecture dispatch
#[inline(always)]
pub fn relu_simd(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::relu_avx2(input, output) }
        } else {
            scalar::relu_scalar(input, output)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::relu_neon(input, output) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        wasm::relu_wasm(input, output)
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        scalar::relu_scalar(input, output)
    }
}

/// SIMD-accelerated ReLU6 activation with automatic architecture dispatch
#[inline(always)]
pub fn relu6_simd(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::relu6_avx2(input, output) }
        } else {
            scalar::relu6_scalar(input, output)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::relu6_neon(input, output) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        wasm::relu6_wasm(input, output)
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        scalar::relu6_scalar(input, output)
    }
}

/// SIMD-accelerated batch normalization with automatic architecture dispatch
#[inline(always)]
pub fn batch_norm_simd(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    epsilon: f32,
    channels: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::batch_norm_avx2(input, output, gamma, beta, mean, var, epsilon, channels)
            }
        } else {
            scalar::batch_norm_scalar(input, output, gamma, beta, mean, var, epsilon, channels)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::batch_norm_neon(input, output, gamma, beta, mean, var, epsilon, channels) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        wasm::batch_norm_wasm(input, output, gamma, beta, mean, var, epsilon, channels)
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        scalar::batch_norm_scalar(input, output, gamma, beta, mean, var, epsilon, channels)
    }
}

/// SIMD-accelerated 3x3 convolution with automatic architecture dispatch
#[inline(always)]
pub fn conv_3x3_simd(
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
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                avx2::conv_3x3_avx2_fma(
                    input, kernel, output, in_h, in_w, in_c, out_c, stride, padding,
                )
            }
        } else if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::conv_3x3_avx2(
                    input, kernel, output, in_h, in_w, in_c, out_c, stride, padding,
                )
            }
        } else {
            scalar::conv_3x3_scalar(
                input, kernel, output, in_h, in_w, in_c, out_c, stride, padding,
            )
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::conv_3x3_neon(
                input, kernel, output, in_h, in_w, in_c, out_c, stride, padding,
            )
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        wasm::conv_3x3_wasm(
            input, kernel, output, in_h, in_w, in_c, out_c, stride, padding,
        )
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        scalar::conv_3x3_scalar(
            input, kernel, output, in_h, in_w, in_c, out_c, stride, padding,
        )
    }
}

/// SIMD-accelerated depthwise 3x3 convolution
#[inline(always)]
pub fn depthwise_conv_3x3_simd(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    stride: usize,
    padding: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::depthwise_conv_3x3_avx2(input, kernel, output, h, w, c, stride, padding)
            }
        } else {
            scalar::depthwise_conv_3x3_scalar(input, kernel, output, h, w, c, stride, padding)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::depthwise_conv_3x3_neon(input, kernel, output, h, w, c, stride, padding) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        wasm::depthwise_conv_3x3_wasm(input, kernel, output, h, w, c, stride, padding)
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        scalar::depthwise_conv_3x3_scalar(input, kernel, output, h, w, c, stride, padding)
    }
}

/// SIMD-accelerated global average pooling
#[inline(always)]
pub fn global_avg_pool_simd(input: &[f32], output: &mut [f32], h: usize, w: usize, c: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::global_avg_pool_avx2(input, output, h, w, c) }
        } else {
            scalar::global_avg_pool_scalar(input, output, h, w, c)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::global_avg_pool_neon(input, output, h, w, c) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        wasm::global_avg_pool_wasm(input, output, h, w, c)
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        scalar::global_avg_pool_scalar(input, output, h, w, c)
    }
}

/// SIMD-accelerated max pooling 2x2
#[inline(always)]
pub fn max_pool_2x2_simd(
    input: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    stride: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::max_pool_2x2_avx2(input, output, h, w, c, stride) }
        } else {
            scalar::max_pool_2x2_scalar(input, output, h, w, c, stride)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::max_pool_2x2_neon(input, output, h, w, c, stride) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        wasm::max_pool_2x2_wasm(input, output, h, w, c, stride)
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        scalar::max_pool_2x2_scalar(input, output, h, w, c, stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = dot_product_simd(&a, &b);
        let expected = scalar::dot_product_scalar(&a, &b);

        assert!((result - expected).abs() < 0.001);
    }

    #[test]
    fn test_relu_simd() {
        let input = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
        let mut output = vec![0.0; 8];

        relu_simd(&input, &mut output);

        assert_eq!(output, vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0]);
    }

    #[test]
    fn test_relu6_simd() {
        let input = vec![-1.0, 2.0, 7.0, 4.0, -5.0, 10.0, 3.0, 8.0];
        let mut output = vec![0.0; 8];

        relu6_simd(&input, &mut output);

        assert_eq!(output, vec![0.0, 2.0, 6.0, 4.0, 0.0, 6.0, 3.0, 6.0]);
    }
}
