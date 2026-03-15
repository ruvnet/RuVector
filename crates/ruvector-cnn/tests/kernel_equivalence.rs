//! Kernel Equivalence Tests
//!
//! Validates that SIMD and scalar implementations produce identical results:
//! - SIMD vs scalar equivalence
//! - Random input fuzzing
//! - Edge case coverage

#[cfg(target_arch = "x86_64")]
use ruvector_cnn::int8::kernels::simd::{conv2d_int8_simd, matmul_int8_simd};

use ruvector_cnn::int8::kernels::scalar::{conv2d_int8_scalar, matmul_int8_scalar};
use ruvector_cnn::int8::QuantParams;

#[cfg(test)]
mod kernel_equivalence {
    use super::*;

    /// Compare two i32 vectors for equality with tolerance
    fn assert_vectors_equal(a: &[i32], b: &[i32], tolerance: i32, context: &str) {
        assert_eq!(
            a.len(),
            b.len(),
            "{}: Vector lengths differ: {} vs {}",
            context,
            a.len(),
            b.len()
        );

        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (va - vb).abs();
            assert!(
                diff <= tolerance,
                "{}: Element {} differs by {}: {} vs {}",
                context,
                i,
                diff,
                va,
                vb
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_matmul_simd_vs_scalar() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping SIMD test: AVX2 not available");
            return;
        }

        println!("\n=== MatMul SIMD vs Scalar Equivalence ===");

        let test_cases = vec![
            ("small_square", 16, 16, 16),
            ("medium_square", 64, 64, 64),
            ("tall_narrow", 128, 32, 64),
            ("wide_short", 32, 128, 64),
            ("large_square", 128, 128, 128),
        ];

        let params = QuantParams {
            scale: 0.01,
            zero_point: 0,
        };

        for (name, m, n, k) in test_cases {
            let mut rng = fastrand::Rng::with_seed(42);

            // Generate random INT8 matrices
            let a: Vec<i8> = (0..m * k).map(|_| rng.i8(..)).collect();
            let b: Vec<i8> = (0..k * n).map(|_| rng.i8(..)).collect();

            // Compute with scalar
            let result_scalar = matmul_int8_scalar(&a, &b, params, m, n, k);

            // Compute with SIMD
            let result_simd = unsafe { matmul_int8_simd(&a, &b, params, m, n, k) };

            // Compare results
            assert_vectors_equal(
                &result_scalar,
                &result_simd,
                0, // Exact match required
                &format!("matmul_{}", name),
            );

            println!("✓ {:<20} {}x{}x{} - SIMD matches scalar", name, m, n, k);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_conv2d_simd_vs_scalar() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping SIMD test: AVX2 not available");
            return;
        }

        println!("\n=== Conv2D SIMD vs Scalar Equivalence ===");

        let test_cases = vec![
            ("small_3x3", 28, 28, 16, 3, 1),
            ("medium_3x3", 56, 56, 32, 3, 1),
            ("large_3x3", 112, 112, 64, 3, 1),
            ("stride_2", 56, 56, 32, 3, 2),
            ("5x5_kernel", 56, 56, 32, 5, 1),
        ];

        let params = QuantParams {
            scale: 0.01,
            zero_point: 0,
        };

        for (name, h, w, c, k, stride) in test_cases {
            let mut rng = fastrand::Rng::with_seed(123);

            // Generate random input and kernel
            let input: Vec<i8> = (0..h * w * c).map(|_| rng.i8(..)).collect();
            let kernel: Vec<i8> = (0..k * k * c).map(|_| rng.i8(..)).collect();

            // Compute with scalar
            let result_scalar = conv2d_int8_scalar(&input, &kernel, params, h, w, c, k, stride);

            // Compute with SIMD
            let result_simd =
                unsafe { conv2d_int8_simd(&input, &kernel, params, h, w, c, k, stride) };

            // Compare results
            assert_vectors_equal(
                &result_scalar,
                &result_simd,
                0, // Exact match required
                &format!("conv2d_{}", name),
            );

            println!(
                "✓ {:<20} {}x{}x{} k={} s={} - SIMD matches scalar",
                name, h, w, c, k, stride
            );
        }
    }

    #[test]
    fn test_matmul_scalar_determinism() {
        println!("\n=== MatMul Scalar Determinism ===");

        let m = 64;
        let n = 64;
        let k = 64;

        let mut rng = fastrand::Rng::with_seed(456);
        let a: Vec<i8> = (0..m * k).map(|_| rng.i8(..)).collect();
        let b: Vec<i8> = (0..k * n).map(|_| rng.i8(..)).collect();

        let params = QuantParams {
            scale: 0.01,
            zero_point: 0,
        };

        // Compute multiple times
        let result1 = matmul_int8_scalar(&a, &b, params, m, n, k);
        let result2 = matmul_int8_scalar(&a, &b, params, m, n, k);
        let result3 = matmul_int8_scalar(&a, &b, params, m, n, k);

        // All results should be identical
        assert_eq!(result1, result2, "First and second runs differ");
        assert_eq!(result2, result3, "Second and third runs differ");

        println!("✓ Scalar matmul is deterministic across 3 runs");
    }

    #[test]
    fn test_conv2d_scalar_determinism() {
        println!("\n=== Conv2D Scalar Determinism ===");

        let h = 56;
        let w = 56;
        let c = 32;
        let k = 3;
        let stride = 1;

        let mut rng = fastrand::Rng::with_seed(789);
        let input: Vec<i8> = (0..h * w * c).map(|_| rng.i8(..)).collect();
        let kernel: Vec<i8> = (0..k * k * c).map(|_| rng.i8(..)).collect();

        let params = QuantParams {
            scale: 0.01,
            zero_point: 0,
        };

        // Compute multiple times
        let result1 = conv2d_int8_scalar(&input, &kernel, params, h, w, c, k, stride);
        let result2 = conv2d_int8_scalar(&input, &kernel, params, h, w, c, k, stride);
        let result3 = conv2d_int8_scalar(&input, &kernel, params, h, w, c, k, stride);

        // All results should be identical
        assert_eq!(result1, result2, "First and second runs differ");
        assert_eq!(result2, result3, "Second and third runs differ");

        println!("✓ Scalar conv2d is deterministic across 3 runs");
    }

    #[test]
    fn test_matmul_fuzz_random_inputs() {
        println!("\n=== MatMul Random Input Fuzzing ===");

        let num_tests = 20;
        let mut passed = 0;

        for seed in 0..num_tests {
            let mut rng = fastrand::Rng::with_seed(seed);

            // Random dimensions
            let m = rng.usize(8..128);
            let n = rng.usize(8..128);
            let k = rng.usize(8..128);

            // Random matrices
            let a: Vec<i8> = (0..m * k).map(|_| rng.i8(..)).collect();
            let b: Vec<i8> = (0..k * n).map(|_| rng.i8(..)).collect();

            // Random quantization params
            let scale = rng.f32() * 0.1 + 0.001; // [0.001, 0.101]
            let zero_point = rng.i8(-64..64);

            let params = QuantParams { scale, zero_point };

            // Should not panic
            let result = matmul_int8_scalar(&a, &b, params, m, n, k);

            // Verify output size
            assert_eq!(
                result.len(),
                m * n,
                "Output size mismatch for {}x{}x{}",
                m,
                n,
                k
            );

            passed += 1;
        }

        println!("✓ Passed {}/{} random fuzz tests", passed, num_tests);
    }

    #[test]
    fn test_conv2d_fuzz_random_inputs() {
        println!("\n=== Conv2D Random Input Fuzzing ===");

        let num_tests = 20;
        let mut passed = 0;

        for seed in 0..num_tests {
            let mut rng = fastrand::Rng::with_seed(seed);

            // Random dimensions
            let h = rng.usize(16..64);
            let w = rng.usize(16..64);
            let c = rng.usize(4..32);
            let k = rng.choice([3, 5, 7]).unwrap();
            let stride = rng.choice([1, 2]).unwrap();

            // Skip if output would be too small
            if h < k || w < k {
                continue;
            }

            // Random input and kernel
            let input: Vec<i8> = (0..h * w * c).map(|_| rng.i8(..)).collect();
            let kernel: Vec<i8> = (0..k * k * c).map(|_| rng.i8(..)).collect();

            // Random quantization params
            let scale = rng.f32() * 0.1 + 0.001;
            let zero_point = rng.i8(-64..64);

            let params = QuantParams { scale, zero_point };

            // Should not panic
            let result = conv2d_int8_scalar(&input, &kernel, params, h, w, c, k, stride);

            // Verify output size
            let expected_h = (h - k) / stride + 1;
            let expected_w = (w - k) / stride + 1;
            assert_eq!(
                result.len(),
                expected_h * expected_w,
                "Output size mismatch for {}x{}x{} k={} s={}",
                h,
                w,
                c,
                k,
                stride
            );

            passed += 1;
        }

        println!("✓ Passed {}/{} random fuzz tests", passed, num_tests);
    }

    #[test]
    fn test_matmul_edge_cases() {
        println!("\n=== MatMul Edge Cases ===");

        let params = QuantParams {
            scale: 0.01,
            zero_point: 0,
        };

        // Edge case 1: All zeros
        let a = vec![0i8; 16 * 16];
        let b = vec![0i8; 16 * 16];
        let result = matmul_int8_scalar(&a, &b, params, 16, 16, 16);
        assert!(result.iter().all(|&x| x == 0), "All zeros case failed");
        println!("✓ All zeros");

        // Edge case 2: Identity-like (ones on diagonal)
        let mut a = vec![0i8; 16 * 16];
        for i in 0..16 {
            a[i * 16 + i] = 1;
        }
        let b = vec![1i8; 16 * 16];
        let result = matmul_int8_scalar(&a, &b, params, 16, 16, 16);
        // Each row should sum to 16
        for i in 0..16 {
            let row_sum: i32 = result[i * 16..(i + 1) * 16].iter().sum();
            assert_eq!(row_sum, 16, "Identity-like case failed at row {}", i);
        }
        println!("✓ Identity-like");

        // Edge case 3: Max values
        let a = vec![127i8; 8 * 8];
        let b = vec![127i8; 8 * 8];
        let result = matmul_int8_scalar(&a, &b, params, 8, 8, 8);
        // Should not overflow (using i32 accumulator)
        let max_val = *result.iter().max().unwrap();
        assert!(max_val > 0, "Max values case produced non-positive result");
        println!("✓ Max values (no overflow)");

        // Edge case 4: Min values
        let a = vec![-128i8; 8 * 8];
        let b = vec![-128i8; 8 * 8];
        let result = matmul_int8_scalar(&a, &b, params, 8, 8, 8);
        // Product of negatives should be positive
        let min_val = *result.iter().min().unwrap();
        assert!(min_val > 0, "Min values case produced non-positive result");
        println!("✓ Min values");

        // Edge case 5: Mixed signs
        let a = vec![127i8; 8 * 8];
        let b = vec![-128i8; 8 * 8];
        let result = matmul_int8_scalar(&a, &b, params, 8, 8, 8);
        // Product of opposite signs should be negative
        let max_val = *result.iter().max().unwrap();
        assert!(max_val < 0, "Mixed signs case failed");
        println!("✓ Mixed signs");
    }

    #[test]
    fn test_conv2d_edge_cases() {
        println!("\n=== Conv2D Edge Cases ===");

        let params = QuantParams {
            scale: 0.01,
            zero_point: 0,
        };

        // Edge case 1: All zeros
        let input = vec![0i8; 28 * 28 * 3];
        let kernel = vec![0i8; 3 * 3 * 3];
        let result = conv2d_int8_scalar(&input, &kernel, params, 28, 28, 3, 3, 1);
        assert!(result.iter().all(|&x| x == 0), "All zeros case failed");
        println!("✓ All zeros");

        // Edge case 2: Uniform input and kernel
        let input = vec![1i8; 28 * 28 * 3];
        let kernel = vec![1i8; 3 * 3 * 3];
        let result = conv2d_int8_scalar(&input, &kernel, params, 28, 28, 3, 3, 1);
        // Each output should be 3*3*3 = 27
        assert!(
            result.iter().all(|&x| x == 27),
            "Uniform case failed: expected 27, got {:?}",
            &result[..5]
        );
        println!("✓ Uniform (all ones)");

        // Edge case 3: Sparse input (mostly zeros)
        let mut input = vec![0i8; 28 * 28 * 3];
        input[0] = 1;
        input[100] = 1;
        input[200] = 1;
        let kernel = vec![1i8; 3 * 3 * 3];
        let result = conv2d_int8_scalar(&input, &kernel, params, 28, 28, 3, 3, 1);
        // Most outputs should be 0 or small
        let non_zero_count = result.iter().filter(|&&x| x != 0).count();
        assert!(
            non_zero_count < result.len() / 2,
            "Sparse case failed: too many non-zeros"
        );
        println!("✓ Sparse input");

        // Edge case 4: Large stride
        let input = vec![1i8; 56 * 56 * 16];
        let kernel = vec![1i8; 3 * 3 * 16];
        let result = conv2d_int8_scalar(&input, &kernel, params, 56, 56, 16, 3, 2);
        let expected_size = ((56 - 3) / 2 + 1) * ((56 - 3) / 2 + 1);
        assert_eq!(
            result.len(),
            expected_size,
            "Large stride output size mismatch"
        );
        println!("✓ Large stride (s=2)");

        // Edge case 5: Minimal size
        let input = vec![1i8; 3 * 3 * 1];
        let kernel = vec![1i8; 3 * 3 * 1];
        let result = conv2d_int8_scalar(&input, &kernel, params, 3, 3, 1, 3, 1);
        assert_eq!(result.len(), 1, "Minimal size case failed");
        assert_eq!(result[0], 9, "Minimal size value mismatch");
        println!("✓ Minimal size (3x3 -> 1x1)");
    }

    #[test]
    fn test_matmul_boundary_dimensions() {
        println!("\n=== MatMul Boundary Dimensions ===");

        let params = QuantParams {
            scale: 0.01,
            zero_point: 0,
        };

        // Very tall and narrow
        let a = vec![1i8; 256 * 4];
        let b = vec![1i8; 4 * 8];
        let result = matmul_int8_scalar(&a, &b, params, 256, 8, 4);
        assert_eq!(result.len(), 256 * 8, "Tall-narrow dimension mismatch");
        println!("✓ Tall-narrow (256x4 × 4x8)");

        // Very wide and short
        let a = vec![1i8; 4 * 256];
        let b = vec![1i8; 256 * 8];
        let result = matmul_int8_scalar(&a, &b, params, 4, 8, 256);
        assert_eq!(result.len(), 4 * 8, "Wide-short dimension mismatch");
        println!("✓ Wide-short (4x256 × 256x8)");

        // Vector-vector (outer product)
        let a = vec![1i8; 16 * 1];
        let b = vec![1i8; 1 * 16];
        let result = matmul_int8_scalar(&a, &b, params, 16, 16, 1);
        assert_eq!(result.len(), 16 * 16, "Vector-vector dimension mismatch");
        println!("✓ Vector-vector outer product (16x1 × 1x16)");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_alignment_independence() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping SIMD alignment test: AVX2 not available");
            return;
        }

        println!("\n=== SIMD Alignment Independence ===");

        let params = QuantParams {
            scale: 0.01,
            zero_point: 0,
        };

        // Test matmul with various sizes (aligned and unaligned)
        let sizes: Vec<(usize, usize, usize)> = vec![
            (15, 15, 15), // Unaligned
            (16, 16, 16), // Aligned to 16
            (17, 17, 17), // Unaligned
            (31, 31, 31), // Unaligned
            (32, 32, 32), // Aligned to 32
            (33, 33, 33), // Unaligned
        ];

        for (m, n, k) in sizes {
            let mut rng = fastrand::Rng::with_seed(42 + m as u64);
            let a: Vec<i8> = (0..m * k).map(|_| rng.i8(..)).collect();
            let b: Vec<i8> = (0..k * n).map(|_| rng.i8(..)).collect();

            let result_scalar = matmul_int8_scalar(&a, &b, params, m, n, k);
            let result_simd = unsafe { matmul_int8_simd(&a, &b, params, m, n, k) };

            assert_vectors_equal(
                &result_scalar,
                &result_simd,
                0,
                &format!("alignment_{}x{}x{}", m, n, k),
            );

            println!("✓ {}x{}x{} - SIMD matches scalar", m, n, k);
        }
    }
}
