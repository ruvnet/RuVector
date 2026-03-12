//! Scalar implementations of INT8 kernels
//!
//! These serve as reference implementations and fallbacks for non-x86_64 platforms.

use crate::int8::QuantParams;

/// Scalar INT8 matrix multiplication
///
/// Computes C = A × B where:
/// - A: m × k matrix (row-major)
/// - B: k × n matrix (row-major)
/// - C: m × n matrix (row-major)
pub fn matmul_int8_scalar(
    a: &[i8],
    b: &[i8],
    _params: QuantParams,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<i32> {
    let mut c = vec![0i32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i32;
            for p in 0..k {
                let a_val = a[i * k + p] as i32;
                let b_val = b[p * n + j] as i32;
                sum += a_val * b_val;
            }
            c[i * n + j] = sum;
        }
    }

    c
}

/// Scalar INT8 2D convolution
///
/// Computes convolution of input with kernel:
/// - input: [h, w, c] spatial feature map
/// - kernel: [k, k, c] convolution kernel
/// - output: [out_h, out_w] where out_h = (h - k) / stride + 1
pub fn conv2d_int8_scalar(
    input: &[i8],
    kernel: &[i8],
    _params: QuantParams,
    h: usize,
    w: usize,
    c: usize,
    k: usize,
    stride: usize,
) -> Vec<i32> {
    let out_h = (h - k) / stride + 1;
    let out_w = (w - k) / stride + 1;
    let mut output = vec![0i32; out_h * out_w];

    for oh in 0..out_h {
        for ow in 0..out_w {
            let mut sum = 0i32;

            for kh in 0..k {
                for kw in 0..k {
                    for ch in 0..c {
                        let ih = oh * stride + kh;
                        let iw = ow * stride + kw;

                        let input_idx = (ih * w + iw) * c + ch;
                        let kernel_idx = (kh * k + kw) * c + ch;

                        let input_val = input[input_idx] as i32;
                        let kernel_val = kernel[kernel_idx] as i32;

                        sum += input_val * kernel_val;
                    }
                }
            }

            output[oh * out_w + ow] = sum;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        let params = QuantParams {
            scale: 1.0,
            zero_point: 0,
        };

        // 4x4 identity matrix
        let mut a = vec![0i8; 16];
        for i in 0..4 {
            a[i * 4 + i] = 1;
        }

        // 4x4 matrix with all 1s
        let b = vec![1i8; 16];

        let c = matmul_int8_scalar(&a, &b, params, 4, 4, 4);

        // Result should have 1s where identity had 1s
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(c[i * 4 + j], 1);
            }
        }
    }

    #[test]
    fn test_conv2d_uniform() {
        let params = QuantParams {
            scale: 1.0,
            zero_point: 0,
        };

        // 5x5x1 input with all 1s
        let input = vec![1i8; 5 * 5 * 1];

        // 3x3x1 kernel with all 1s
        let kernel = vec![1i8; 3 * 3 * 1];

        let output = conv2d_int8_scalar(&input, &kernel, params, 5, 5, 1, 3, 1);

        // Output should be 3x3 with all 9s (sum of 3x3 kernel)
        assert_eq!(output.len(), 3 * 3);
        for &val in &output {
            assert_eq!(val, 9);
        }
    }

    #[test]
    fn test_matmul_zeros() {
        let params = QuantParams {
            scale: 1.0,
            zero_point: 0,
        };

        let a = vec![0i8; 8 * 8];
        let b = vec![1i8; 8 * 8];

        let c = matmul_int8_scalar(&a, &b, params, 8, 8, 8);

        assert!(c.iter().all(|&x| x == 0));
    }
}
