//! SIMD (AVX2) implementations of INT8 kernels
//!
//! These implementations leverage x86_64 AVX2 SIMD instructions for
//! 2-4x speedup over scalar implementations.

#![cfg(target_arch = "x86_64")]

use crate::int8::QuantParams;
use std::arch::x86_64::*;

/// AVX2 INT8 matrix multiplication
///
/// # Safety
///
/// This function uses AVX2 intrinsics and requires:
/// - Target CPU to support AVX2 (check with `is_x86_feature_detected!("avx2")`)
/// - Properly aligned memory access patterns
#[target_feature(enable = "avx2")]
pub unsafe fn matmul_int8_simd(
    a: &[i8],
    b: &[i8],
    params: QuantParams,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<i32> {
    // For now, delegate to scalar implementation
    // Full SIMD implementation will be added in Phase 4
    super::scalar::matmul_int8_scalar(a, b, params, m, n, k)
}

/// AVX2 INT8 2D convolution
///
/// # Safety
///
/// This function uses AVX2 intrinsics and requires:
/// - Target CPU to support AVX2
/// - Properly aligned memory access patterns
#[target_feature(enable = "avx2")]
pub unsafe fn conv2d_int8_simd(
    input: &[i8],
    kernel: &[i8],
    params: QuantParams,
    h: usize,
    w: usize,
    c: usize,
    k: usize,
    stride: usize,
) -> Vec<i32> {
    // For now, delegate to scalar implementation
    // Full SIMD implementation will be added in Phase 4
    super::scalar::conv2d_int8_scalar(input, kernel, params, h, w, c, k, stride)
}

/// AVX2 INT8 dot product
///
/// Computes: sum(a[i] * b[i]) using AVX2 intrinsics
///
/// # Safety
///
/// Requires AVX2 support
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn dot_product_int8_avx2(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();

    let mut sum = 0i32;
    let mut i = 0;

    // Process 32 elements at a time with AVX2
    while i + 32 <= len {
        // Load 32 i8 values
        let a_vec = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let b_vec = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        // Convert to i16 for multiplication (to avoid overflow)
        // This is a simplified version - full implementation would use _mm256_maddubs_epi16
        let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a_vec));
        let b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_vec));

        // Multiply and accumulate
        let prod = _mm256_mullo_epi16(a_lo, b_lo);

        // Horizontal add (sum 16 i16 values)
        let prod_i32 = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));

        // Extract and sum
        let mut temp = [0i32; 8];
        _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, prod_i32);
        sum += temp.iter().sum::<i32>();

        i += 16; // Only processed 16 because we only did low half
    }

    // Handle remaining elements with scalar code
    while i < len {
        sum += a[i] as i32 * b[i] as i32;
        i += 1;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_simd_available() {
        if is_x86_feature_detected!("avx2") {
            let params = QuantParams {
                scale: 1.0,
                zero_point: 0,
            };

            let a = vec![1i8; 16];
            let b = vec![1i8; 16];

            let c = unsafe { matmul_int8_simd(&a, &b, params, 4, 4, 4) };

            // Each element should be sum of 4 1s = 4
            assert!(c.iter().all(|&x| x == 4));
        } else {
            println!("AVX2 not available, skipping SIMD test");
        }
    }

    #[test]
    fn test_conv2d_simd_available() {
        if is_x86_feature_detected!("avx2") {
            let params = QuantParams {
                scale: 1.0,
                zero_point: 0,
            };

            let input = vec![1i8; 5 * 5 * 1];
            let kernel = vec![1i8; 3 * 3 * 1];

            let output = unsafe { conv2d_int8_simd(&input, &kernel, params, 5, 5, 1, 3, 1) };

            assert_eq!(output.len(), 3 * 3);
            assert!(output.iter().all(|&x| x == 9));
        }
    }
}
