//! Hadamard Transform Tests for ADR-090
//!
//! Property-based tests for the Walsh-Hadamard transform used in
//! incoherence processing for ultra-low-bit quantization.
//!
//! Key properties tested:
//! - INV-4: H × H^T = n × I (orthogonality property)
//! - Invertibility: H^(-1) × H × x = x
//! - Sign flip correctness for random rotations

#[cfg(test)]
mod hadamard_tests {
    use std::f32::consts::SQRT_2;

    // ============================================================================
    // Test Constants
    // ============================================================================

    /// Epsilon for floating-point comparisons
    const EPSILON: f32 = 1e-5;

    /// Sizes to test (must be powers of 2)
    const TEST_SIZES: [usize; 6] = [2, 4, 8, 16, 32, 64];

    // ============================================================================
    // Hadamard Transform Implementation
    // ============================================================================

    /// Hadamard transform struct with O(n log n) implementation
    #[derive(Debug, Clone)]
    struct HadamardTransform {
        /// Size of the transform (must be power of 2)
        size: usize,
        /// Optional random sign flips for incoherence
        sign_flips: Option<Vec<i8>>,
    }

    impl HadamardTransform {
        /// Create a new Hadamard transform of given size
        fn new(size: usize) -> Self {
            assert!(size.is_power_of_two(), "Size must be power of 2");
            Self {
                size,
                sign_flips: None,
            }
        }

        /// Create with random sign flips for incoherence
        fn with_sign_flips(size: usize, seed: u64) -> Self {
            assert!(size.is_power_of_two(), "Size must be power of 2");

            // Simple PRNG for deterministic sign generation
            let mut state = seed;
            let signs: Vec<i8> = (0..size)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if (state >> 63) == 0 {
                        1
                    } else {
                        -1
                    }
                })
                .collect();

            Self {
                size,
                sign_flips: Some(signs),
            }
        }

        /// Forward Hadamard transform in-place using O(n log n) algorithm
        fn forward_inplace(&self, data: &mut [f32]) {
            assert_eq!(data.len(), self.size, "Data size must match transform size");

            // Apply sign flips before transform if present
            if let Some(ref signs) = self.sign_flips {
                for (i, &sign) in signs.iter().enumerate() {
                    data[i] *= sign as f32;
                }
            }

            // Fast Walsh-Hadamard Transform (Cooley-Tukey-like butterfly)
            let n = self.size;
            let mut h = 1;

            while h < n {
                for i in (0..n).step_by(h * 2) {
                    for j in i..(i + h) {
                        let x = data[j];
                        let y = data[j + h];
                        data[j] = x + y;
                        data[j + h] = x - y;
                    }
                }
                h *= 2;
            }

            // Normalize by sqrt(n) for orthonormal transform
            let scale = 1.0 / (n as f32).sqrt();
            for x in data.iter_mut() {
                *x *= scale;
            }
        }

        /// Inverse Hadamard transform in-place
        fn inverse_inplace(&self, data: &mut [f32]) {
            assert_eq!(data.len(), self.size, "Data size must match transform size");

            // The inverse of normalized Hadamard is itself (up to sign flips)
            let n = self.size;
            let mut h = 1;

            while h < n {
                for i in (0..n).step_by(h * 2) {
                    for j in i..(i + h) {
                        let x = data[j];
                        let y = data[j + h];
                        data[j] = x + y;
                        data[j + h] = x - y;
                    }
                }
                h *= 2;
            }

            // Normalize
            let scale = 1.0 / (n as f32).sqrt();
            for x in data.iter_mut() {
                *x *= scale;
            }

            // Apply inverse sign flips after transform if present
            if let Some(ref signs) = self.sign_flips {
                for (i, &sign) in signs.iter().enumerate() {
                    data[i] *= sign as f32;
                }
            }
        }

        /// Forward transform (returns new vector)
        fn forward(&self, data: &[f32]) -> Vec<f32> {
            let mut result = data.to_vec();
            self.forward_inplace(&mut result);
            result
        }

        /// Inverse transform (returns new vector)
        fn inverse(&self, data: &[f32]) -> Vec<f32> {
            let mut result = data.to_vec();
            self.inverse_inplace(&mut result);
            result
        }
    }

    // ============================================================================
    // Matrix Helpers for Property Testing
    // ============================================================================

    /// Generate the full Hadamard matrix (for small sizes only)
    /// Returns the normalized orthogonal Hadamard matrix where H × H^T = I
    fn hadamard_matrix(n: usize) -> Vec<Vec<f32>> {
        assert!(n.is_power_of_two());

        // First generate unnormalized matrix recursively
        fn hadamard_unnormalized(n: usize) -> Vec<Vec<f32>> {
            if n == 1 {
                return vec![vec![1.0]];
            }

            let half = hadamard_unnormalized(n / 2);
            let mut result = vec![vec![0.0; n]; n];

            for i in 0..n / 2 {
                for j in 0..n / 2 {
                    let val = half[i][j];
                    result[i][j] = val;
                    result[i][j + n / 2] = val;
                    result[i + n / 2][j] = val;
                    result[i + n / 2][j + n / 2] = -val;
                }
            }

            result
        }

        let mut result = hadamard_unnormalized(n);

        // Normalize once at the end by 1/sqrt(n) to make it orthogonal
        let scale = 1.0 / (n as f32).sqrt();
        for row in result.iter_mut() {
            for val in row.iter_mut() {
                *val *= scale;
            }
        }

        result
    }

    /// Matrix multiply A × B
    fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = a.len();
        let mut result = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        result
    }

    /// Transpose a matrix
    fn transpose(m: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = m.len();
        let mut result = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                result[j][i] = m[i][j];
            }
        }

        result
    }

    /// Check if matrix is identity (within tolerance)
    fn is_identity(m: &[Vec<f32>], tol: f32) -> bool {
        let n = m.len();
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (m[i][j] - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    // ============================================================================
    // 1. Property Test: H × H^T = I (Orthogonality)
    // ============================================================================

    #[test]
    fn test_hadamard_orthogonality_property() {
        // INV-4: H × H^T = I for normalized Hadamard
        for &n in &TEST_SIZES {
            let h = hadamard_matrix(n);
            let ht = transpose(&h);
            let hht = matmul(&h, &ht);

            assert!(
                is_identity(&hht, EPSILON),
                "H × H^T != I for n={}\nH×H^T:\n{:?}",
                n,
                hht
            );
        }
    }

    #[test]
    fn test_hadamard_hth_identity() {
        // Also verify H^T × H = I
        for &n in &TEST_SIZES {
            let h = hadamard_matrix(n);
            let ht = transpose(&h);
            let hth = matmul(&ht, &h);

            assert!(is_identity(&hth, EPSILON), "H^T × H != I for n={}", n);
        }
    }

    // ============================================================================
    // 2. Invertibility Test: H^(-1) × H × x = x
    // ============================================================================

    #[test]
    fn test_hadamard_invertibility_simple() {
        for &n in &TEST_SIZES {
            let transform = HadamardTransform::new(n);

            // Simple test vector
            let x: Vec<f32> = (0..n).map(|i| i as f32).collect();

            // Forward then inverse
            let hx = transform.forward(&x);
            let reconstructed = transform.inverse(&hx);

            for (i, (&orig, &rec)) in x.iter().zip(reconstructed.iter()).enumerate() {
                assert!(
                    (orig - rec).abs() < EPSILON,
                    "Invertibility failed at {} for n={}: orig={}, rec={}",
                    i,
                    n,
                    orig,
                    rec
                );
            }
        }
    }

    #[test]
    fn test_hadamard_invertibility_random() {
        for &n in &TEST_SIZES {
            let transform = HadamardTransform::new(n);

            // Pseudo-random test vector
            let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.234).sin()).collect();

            let hx = transform.forward(&x);
            let reconstructed = transform.inverse(&hx);

            for (i, (&orig, &rec)) in x.iter().zip(reconstructed.iter()).enumerate() {
                assert!(
                    (orig - rec).abs() < EPSILON,
                    "Random invertibility failed at {} for n={}: orig={}, rec={}",
                    i,
                    n,
                    orig,
                    rec
                );
            }
        }
    }

    #[test]
    fn test_hadamard_invertibility_zeros() {
        for &n in &TEST_SIZES {
            let transform = HadamardTransform::new(n);
            let x = vec![0.0f32; n];

            let hx = transform.forward(&x);
            let reconstructed = transform.inverse(&hx);

            for (i, (&orig, &rec)) in x.iter().zip(reconstructed.iter()).enumerate() {
                assert!(
                    rec.abs() < EPSILON,
                    "Zero invertibility failed at {} for n={}: rec={}",
                    i,
                    n,
                    rec
                );
            }
        }
    }

    #[test]
    fn test_hadamard_invertibility_ones() {
        for &n in &TEST_SIZES {
            let transform = HadamardTransform::new(n);
            let x = vec![1.0f32; n];

            let hx = transform.forward(&x);
            let reconstructed = transform.inverse(&hx);

            for (i, (&orig, &rec)) in x.iter().zip(reconstructed.iter()).enumerate() {
                assert!(
                    (orig - rec).abs() < EPSILON,
                    "Ones invertibility failed at {} for n={}: orig={}, rec={}",
                    i,
                    n,
                    orig,
                    rec
                );
            }
        }
    }

    // ============================================================================
    // 3. Sign Flip Correctness Tests
    // ============================================================================

    #[test]
    fn test_sign_flips_invertibility() {
        for &n in &TEST_SIZES {
            let transform = HadamardTransform::with_sign_flips(n, 12345);

            let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.5).cos()).collect();

            let hx = transform.forward(&x);
            let reconstructed = transform.inverse(&hx);

            for (i, (&orig, &rec)) in x.iter().zip(reconstructed.iter()).enumerate() {
                assert!(
                    (orig - rec).abs() < EPSILON,
                    "Sign flip invertibility failed at {} for n={}: orig={}, rec={}",
                    i,
                    n,
                    orig,
                    rec
                );
            }
        }
    }

    #[test]
    fn test_different_sign_flip_seeds() {
        let n = 32;

        for seed in [1, 42, 12345, 999999] {
            let transform = HadamardTransform::with_sign_flips(n, seed);
            let x: Vec<f32> = (0..n).map(|i| i as f32).collect();

            let hx = transform.forward(&x);
            let reconstructed = transform.inverse(&hx);

            let mse: f32 = x
                .iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / n as f32;

            assert!(
                mse < EPSILON * EPSILON,
                "Seed {} failed with MSE {}",
                seed,
                mse
            );
        }
    }

    #[test]
    fn test_sign_flips_values() {
        let transform = HadamardTransform::with_sign_flips(8, 42);

        if let Some(ref signs) = transform.sign_flips {
            // All signs should be +1 or -1
            for &sign in signs {
                assert!(sign == 1 || sign == -1, "Invalid sign value: {}", sign);
            }
        }
    }

    // ============================================================================
    // 4. Energy Preservation Test (Parseval's Theorem)
    // ============================================================================

    #[test]
    fn test_energy_preservation() {
        // Orthogonal transforms preserve energy: ||H×x||² = ||x||²
        for &n in &TEST_SIZES {
            let transform = HadamardTransform::new(n);
            let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();

            let energy_x: f32 = x.iter().map(|v| v * v).sum();
            let hx = transform.forward(&x);
            let energy_hx: f32 = hx.iter().map(|v| v * v).sum();

            assert!(
                (energy_x - energy_hx).abs() < EPSILON * energy_x.max(1.0),
                "Energy not preserved for n={}: input={}, output={}",
                n,
                energy_x,
                energy_hx
            );
        }
    }

    #[test]
    fn test_energy_preservation_with_sign_flips() {
        for &n in &TEST_SIZES {
            let transform = HadamardTransform::with_sign_flips(n, 9999);
            let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).cos()).collect();

            let energy_x: f32 = x.iter().map(|v| v * v).sum();
            let hx = transform.forward(&x);
            let energy_hx: f32 = hx.iter().map(|v| v * v).sum();

            assert!(
                (energy_x - energy_hx).abs() < EPSILON * energy_x.max(1.0),
                "Energy not preserved with sign flips for n={}: input={}, output={}",
                n,
                energy_x,
                energy_hx
            );
        }
    }

    // ============================================================================
    // 5. Linearity Test
    // ============================================================================

    #[test]
    fn test_hadamard_linearity() {
        // H(ax + by) = aH(x) + bH(y)
        for &n in &TEST_SIZES {
            let transform = HadamardTransform::new(n);

            let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let y: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
            let a = 2.5f32;
            let b = -1.3f32;

            // Compute H(ax + by)
            let ax_by: Vec<f32> = x
                .iter()
                .zip(y.iter())
                .map(|(xi, yi)| a * xi + b * yi)
                .collect();
            let h_ax_by = transform.forward(&ax_by);

            // Compute aH(x) + bH(y)
            let hx = transform.forward(&x);
            let hy = transform.forward(&y);
            let a_hx_b_hy: Vec<f32> = hx
                .iter()
                .zip(hy.iter())
                .map(|(hxi, hyi)| a * hxi + b * hyi)
                .collect();

            // Use relative epsilon for larger values due to accumulated float errors
            let linearity_epsilon = 1e-4;
            for (i, (&left, &right)) in h_ax_by.iter().zip(a_hx_b_hy.iter()).enumerate() {
                let diff = (left - right).abs();
                let max_val = left.abs().max(right.abs()).max(1.0);
                assert!(
                    diff < linearity_epsilon * max_val,
                    "Linearity failed at {} for n={}: H(ax+by)={}, aH(x)+bH(y)={}, diff={}",
                    i,
                    n,
                    left,
                    right,
                    diff
                );
            }
        }
    }

    // ============================================================================
    // 6. Known Value Tests
    // ============================================================================

    #[test]
    fn test_hadamard_2x2_known_values() {
        // H_2 = (1/sqrt(2)) * [[1, 1], [1, -1]]
        let h = hadamard_matrix(2);

        let expected_scale = 1.0 / SQRT_2;
        let expected = vec![
            vec![expected_scale, expected_scale],
            vec![expected_scale, -expected_scale],
        ];

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (h[i][j] - expected[i][j]).abs() < EPSILON,
                    "H_2[{}][{}] = {}, expected {}",
                    i,
                    j,
                    h[i][j],
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn test_hadamard_4x4_structure() {
        // H_4 = (1/2) * [[1,1,1,1], [1,-1,1,-1], [1,1,-1,-1], [1,-1,-1,1]]
        let h = hadamard_matrix(4);

        // All entries should be +/- 0.5
        let expected_abs = 0.5;
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (h[i][j].abs() - expected_abs).abs() < EPSILON,
                    "H_4[{}][{}] = {}, expected +/- {}",
                    i,
                    j,
                    h[i][j],
                    expected_abs
                );
            }
        }
    }

    #[test]
    fn test_transform_of_unit_vector() {
        // H × e_0 should give first column of H
        let n = 8;
        let transform = HadamardTransform::new(n);
        let h = hadamard_matrix(n);

        let mut e0 = vec![0.0f32; n];
        e0[0] = 1.0;

        let result = transform.forward(&e0);

        for i in 0..n {
            assert!(
                (result[i] - h[i][0]).abs() < EPSILON,
                "Transform of e_0 mismatch at {}: got {}, expected {}",
                i,
                result[i],
                h[i][0]
            );
        }
    }

    // ============================================================================
    // 7. Edge Case Tests
    // ============================================================================

    #[test]
    fn test_hadamard_size_1() {
        let transform = HadamardTransform::new(1);
        let x = vec![5.0f32];

        let hx = transform.forward(&x);
        assert_eq!(hx.len(), 1);
        assert!((hx[0] - 5.0).abs() < EPSILON, "H_1 should be identity");

        let reconstructed = transform.inverse(&hx);
        assert!((reconstructed[0] - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_hadamard_size_2() {
        let transform = HadamardTransform::new(2);
        let x = vec![1.0f32, 0.0];

        let hx = transform.forward(&x);
        // H_2 * [1, 0]^T = (1/sqrt(2)) * [1, 1]^T
        let expected = 1.0 / SQRT_2;
        assert!((hx[0] - expected).abs() < EPSILON);
        assert!((hx[1] - expected).abs() < EPSILON);
    }

    #[test]
    fn test_hadamard_large_values() {
        let transform = HadamardTransform::new(16);
        let x: Vec<f32> = vec![1e6; 16];

        let hx = transform.forward(&x);
        let reconstructed = transform.inverse(&hx);

        for (i, (&orig, &rec)) in x.iter().zip(reconstructed.iter()).enumerate() {
            let relative_error = (orig - rec).abs() / orig.abs().max(1.0);
            assert!(
                relative_error < EPSILON,
                "Large value invertibility failed at {}: orig={}, rec={}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_hadamard_small_values() {
        let transform = HadamardTransform::new(16);
        let x: Vec<f32> = vec![1e-6; 16];

        let hx = transform.forward(&x);
        let reconstructed = transform.inverse(&hx);

        for (i, (&orig, &rec)) in x.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-10,
                "Small value invertibility failed at {}: orig={}, rec={}",
                i,
                orig,
                rec
            );
        }
    }

    // ============================================================================
    // 8. Performance Hints Test
    // ============================================================================

    #[test]
    fn test_hadamard_inplace_modifies_input() {
        let transform = HadamardTransform::new(8);
        let original: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut data = original.clone();

        transform.forward_inplace(&mut data);

        // Data should be modified
        let different = data
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > EPSILON);
        assert!(different, "In-place transform should modify data");
    }

    #[test]
    fn test_hadamard_forward_returns_new_vector() {
        let transform = HadamardTransform::new(8);
        let original: Vec<f32> = (0..8).map(|i| i as f32).collect();

        let result = transform.forward(&original);

        // Original should be unchanged
        for (i, (&orig, &expected)) in original
            .iter()
            .zip((0..8).map(|i| i as f32).collect::<Vec<_>>().iter())
            .enumerate()
        {
            assert!(
                (orig - expected).abs() < EPSILON,
                "Original modified at {}: {} vs {}",
                i,
                orig,
                expected
            );
        }

        // Result should be different
        let different = result
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > EPSILON);
        assert!(different, "Transform result should differ from input");
    }

    // ============================================================================
    // 9. Incoherence Property Tests
    // ============================================================================

    #[test]
    fn test_hadamard_spreads_energy() {
        // Hadamard should spread concentrated energy
        let transform = HadamardTransform::new(32);

        // Concentrated input (spike)
        let mut x = vec![0.0f32; 32];
        x[0] = 32.0;

        let hx = transform.forward(&x);

        // After transform, energy should be spread
        // All components should have similar magnitude
        let max_component = hx.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let min_component = hx.iter().map(|v| v.abs()).fold(f32::MAX, f32::min);

        // For Hadamard of spike, all outputs should be equal (sqrt(n))
        assert!(
            (max_component - min_component).abs() < EPSILON,
            "Energy should be uniformly spread: max={}, min={}",
            max_component,
            min_component
        );
    }

    #[test]
    fn test_sign_flips_add_randomization() {
        let n = 32;
        let transform1 = HadamardTransform::with_sign_flips(n, 1);
        let transform2 = HadamardTransform::with_sign_flips(n, 2);

        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let hx1 = transform1.forward(&x);
        let hx2 = transform2.forward(&x);

        // Different seeds should produce different outputs
        let different = hx1
            .iter()
            .zip(hx2.iter())
            .any(|(a, b)| (a - b).abs() > EPSILON);
        assert!(
            different,
            "Different seeds should produce different outputs"
        );
    }
}
