//! Hadamard Transform for Incoherence Processing (ADR-090 Phase 3)
//!
//! This module implements the Walsh-Hadamard Transform for incoherence processing
//! in quantization pipelines. The Hadamard transform spreads quantization error
//! uniformly across all coefficients, reducing the impact of outliers.
//!
//! ## Mathematical Background
//!
//! The Walsh-Hadamard transform is defined recursively:
//! ```text
//! H_1 = [1]
//! H_2 = [1  1]
//!       [1 -1]
//! H_n = H_2 ⊗ H_{n/2}  (Kronecker product)
//! ```
//!
//! Key property: H × H^T = n × I (orthogonal, self-inverse up to scaling)
//!
//! ## Implementation Details
//!
//! - Uses in-place butterfly operations for O(n log n) complexity
//! - Supports randomized variant with sign flips for better incoherence
//! - SIMD-optimized for ARM NEON and x86 AVX2 architectures
//! - Normalized by 1/sqrt(n) for energy preservation
//!
//! ## References
//!
//! - ADR-090: Ultra-Low-Bit Quantization Design
//! - "Hadamard Transform" - Signal Processing Fundamentals

use crate::error::{Result, RuvLLMError};

// ============================================================================
// Constants
// ============================================================================

/// Maximum supported log dimension (2^20 = 1M elements)
pub const MAX_LOG_DIM: u32 = 20;

/// SIMD lane width for vectorized operations
#[cfg(target_arch = "aarch64")]
pub const SIMD_LANES: usize = 4; // NEON float32x4

#[cfg(target_arch = "x86_64")]
pub const SIMD_LANES: usize = 8; // AVX2 __m256

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub const SIMD_LANES: usize = 1; // Scalar fallback

// ============================================================================
// Hadamard Transform
// ============================================================================

/// Walsh-Hadamard Transform implementation with optional randomization
///
/// The transform is performed in-place using the butterfly algorithm,
/// achieving O(n log n) time complexity with O(1) auxiliary space.
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::quantize::hadamard::HadamardTransform;
///
/// // Create transform for 8-element vectors (log2(8) = 3)
/// let transform = HadamardTransform::new(3, Some(42));
///
/// let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// transform.forward_inplace(&mut data);
///
/// // Apply inverse to recover original
/// transform.inverse_inplace(&mut data);
/// ```
#[derive(Debug, Clone)]
pub struct HadamardTransform {
    /// Log2 of the dimension (dimension = 2^log_dim)
    log_dim: u32,
    /// Dimension (cached for convenience)
    dim: usize,
    /// Random signs for randomized Hadamard (+1 or -1)
    signs: Vec<i8>,
    /// Normalization factor (1/sqrt(n))
    norm_factor: f32,
    /// Whether this is a randomized transform
    randomized: bool,
}

impl HadamardTransform {
    /// Create a new Hadamard transform
    ///
    /// # Arguments
    ///
    /// * `log_dim` - Log2 of the dimension (dimension will be 2^log_dim)
    /// * `seed` - Optional seed for random sign generation (None for deterministic transform)
    ///
    /// # Returns
    ///
    /// A new `HadamardTransform` instance
    ///
    /// # Errors
    ///
    /// Returns an error if log_dim exceeds MAX_LOG_DIM
    pub fn new(log_dim: u32, seed: Option<u64>) -> Result<Self> {
        if log_dim > MAX_LOG_DIM {
            return Err(RuvLLMError::Quantization(format!(
                "Hadamard dimension 2^{} exceeds maximum supported 2^{}",
                log_dim, MAX_LOG_DIM
            )));
        }

        let dim = 1usize << log_dim;
        let norm_factor = 1.0 / (dim as f32).sqrt();

        let (signs, randomized) = match seed {
            Some(s) => {
                // Generate random signs using a simple LCG PRNG
                let mut rng_state = s;
                let signs: Vec<i8> = (0..dim)
                    .map(|_| {
                        rng_state = rng_state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        if (rng_state >> 63) & 1 == 0 {
                            1
                        } else {
                            -1
                        }
                    })
                    .collect();
                (signs, true)
            }
            None => {
                // Deterministic: all signs are +1
                (vec![1i8; dim], false)
            }
        };

        Ok(Self {
            log_dim,
            dim,
            signs,
            norm_factor,
            randomized,
        })
    }

    /// Create a deterministic (non-randomized) Hadamard transform
    pub fn deterministic(log_dim: u32) -> Result<Self> {
        Self::new(log_dim, None)
    }

    /// Create a randomized Hadamard transform with the given seed
    pub fn randomized(log_dim: u32, seed: u64) -> Result<Self> {
        Self::new(log_dim, Some(seed))
    }

    /// Get the dimension of the transform
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the log2 of the dimension
    #[inline]
    pub fn log_dim(&self) -> u32 {
        self.log_dim
    }

    /// Check if this is a randomized transform
    #[inline]
    pub fn is_randomized(&self) -> bool {
        self.randomized
    }

    /// Apply forward Hadamard transform in-place
    ///
    /// The transform is computed as:
    /// 1. Apply random sign flips (if randomized)
    /// 2. Apply Walsh-Hadamard butterfly operations
    /// 3. Normalize by 1/sqrt(n)
    ///
    /// # Arguments
    ///
    /// * `data` - Mutable slice of f32 values (must have length 2^log_dim)
    ///
    /// # Panics
    ///
    /// Panics if data length doesn't match the transform dimension
    pub fn forward_inplace(&self, data: &mut [f32]) {
        assert_eq!(
            data.len(),
            self.dim,
            "Data length {} must match transform dimension {}",
            data.len(),
            self.dim
        );

        // Step 1: Apply random sign flips
        if self.randomized {
            self.apply_signs(data);
        }

        // Step 2: Apply Walsh-Hadamard butterfly
        self.hadamard_butterfly(data);

        // Step 3: Normalize
        self.normalize(data);
    }

    /// Apply inverse Hadamard transform in-place
    ///
    /// For orthogonal Hadamard matrices, H^-1 = H/n, so the inverse
    /// is essentially the forward transform with adjusted normalization.
    /// With our sqrt(n) normalization on forward, inverse is the same operation.
    ///
    /// # Arguments
    ///
    /// * `data` - Mutable slice of f32 values (must have length 2^log_dim)
    pub fn inverse_inplace(&self, data: &mut [f32]) {
        assert_eq!(
            data.len(),
            self.dim,
            "Data length {} must match transform dimension {}",
            data.len(),
            self.dim
        );

        // Step 1: Apply Walsh-Hadamard butterfly (same as forward)
        self.hadamard_butterfly(data);

        // Step 2: Normalize (same as forward due to orthogonality)
        self.normalize(data);

        // Step 3: Undo sign flips (apply again since signs are +/-1)
        if self.randomized {
            self.apply_signs(data);
        }
    }

    /// Apply random sign flips to data
    #[inline]
    fn apply_signs(&self, data: &mut [f32]) {
        // Use SIMD when available
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.apply_signs_neon(data);
            return;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    self.apply_signs_avx2(data);
                }
                return;
            }
        }

        // Scalar fallback
        self.apply_signs_scalar(data);
    }

    /// Scalar sign application
    #[inline]
    fn apply_signs_scalar(&self, data: &mut [f32]) {
        for (d, &s) in data.iter_mut().zip(self.signs.iter()) {
            *d *= s as f32;
        }
    }

    /// NEON-optimized sign application (ARM64)
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn apply_signs_neon(&self, data: &mut [f32]) {
        use std::arch::aarch64::*;

        let n = data.len();
        let chunks = n / 4;
        let remainder = n % 4;

        let data_ptr = data.as_mut_ptr();

        for i in 0..chunks {
            let idx = i * 4;
            // Load 4 floats
            let v = vld1q_f32(data_ptr.add(idx));

            // Convert signs to floats
            let signs = [
                self.signs[idx] as f32,
                self.signs[idx + 1] as f32,
                self.signs[idx + 2] as f32,
                self.signs[idx + 3] as f32,
            ];
            let s = vld1q_f32(signs.as_ptr());

            // Multiply and store
            let result = vmulq_f32(v, s);
            vst1q_f32(data_ptr.add(idx), result);
        }

        // Handle remainder
        for i in (chunks * 4)..n {
            data[i] *= self.signs[i] as f32;
        }
    }

    /// AVX2-optimized sign application (x86_64)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn apply_signs_avx2(&self, data: &mut [f32]) {
        use std::arch::x86_64::*;

        let n = data.len();
        let chunks = n / 8;

        let data_ptr = data.as_mut_ptr();

        for i in 0..chunks {
            let idx = i * 8;
            // Load 8 floats
            let v = _mm256_loadu_ps(data_ptr.add(idx));

            // Convert signs to floats
            let signs: [f32; 8] = [
                self.signs[idx] as f32,
                self.signs[idx + 1] as f32,
                self.signs[idx + 2] as f32,
                self.signs[idx + 3] as f32,
                self.signs[idx + 4] as f32,
                self.signs[idx + 5] as f32,
                self.signs[idx + 6] as f32,
                self.signs[idx + 7] as f32,
            ];
            let s = _mm256_loadu_ps(signs.as_ptr());

            // Multiply and store
            let result = _mm256_mul_ps(v, s);
            _mm256_storeu_ps(data_ptr.add(idx), result);
        }

        // Handle remainder
        for i in (chunks * 8)..n {
            data[i] *= self.signs[i] as f32;
        }
    }

    /// Apply Walsh-Hadamard butterfly operations
    ///
    /// The butterfly operation for each level h:
    /// ```text
    /// for j in 0..n by 2h:
    ///     for k in 0..h:
    ///         a = data[j + k]
    ///         b = data[j + k + h]
    ///         data[j + k]     = a + b
    ///         data[j + k + h] = a - b
    /// ```
    fn hadamard_butterfly(&self, data: &mut [f32]) {
        let n = self.dim;

        // Try SIMD-optimized path first
        #[cfg(target_arch = "aarch64")]
        {
            if n >= 8 {
                unsafe {
                    self.hadamard_butterfly_neon(data);
                }
                return;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            if n >= 16 && is_x86_feature_detected!("avx2") {
                unsafe {
                    self.hadamard_butterfly_avx2(data);
                }
                return;
            }
        }

        // Scalar fallback
        self.hadamard_butterfly_scalar(data);
    }

    /// Scalar butterfly implementation
    fn hadamard_butterfly_scalar(&self, data: &mut [f32]) {
        let n = self.dim;
        let mut h = 1;

        while h < n {
            let mut j = 0;
            while j < n {
                for k in 0..h {
                    let a = data[j + k];
                    let b = data[j + k + h];
                    data[j + k] = a + b;
                    data[j + k + h] = a - b;
                }
                j += h * 2;
            }
            h *= 2;
        }
    }

    /// NEON-optimized butterfly (ARM64)
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn hadamard_butterfly_neon(&self, data: &mut [f32]) {
        use std::arch::aarch64::*;

        let n = self.dim;
        let mut h = 1;

        while h < n {
            if h >= 4 {
                // SIMD path for h >= 4
                let mut j = 0;
                while j < n {
                    let mut k = 0;
                    while k + 4 <= h {
                        let ptr_a = data.as_mut_ptr().add(j + k);
                        let ptr_b = data.as_mut_ptr().add(j + k + h);

                        let a = vld1q_f32(ptr_a);
                        let b = vld1q_f32(ptr_b);

                        let sum = vaddq_f32(a, b);
                        let diff = vsubq_f32(a, b);

                        vst1q_f32(ptr_a, sum);
                        vst1q_f32(ptr_b, diff);

                        k += 4;
                    }
                    // Handle remainder
                    while k < h {
                        let a = data[j + k];
                        let b = data[j + k + h];
                        data[j + k] = a + b;
                        data[j + k + h] = a - b;
                        k += 1;
                    }
                    j += h * 2;
                }
            } else {
                // Scalar path for h < 4
                let mut j = 0;
                while j < n {
                    for k in 0..h {
                        let a = data[j + k];
                        let b = data[j + k + h];
                        data[j + k] = a + b;
                        data[j + k + h] = a - b;
                    }
                    j += h * 2;
                }
            }
            h *= 2;
        }
    }

    /// AVX2-optimized butterfly (x86_64)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn hadamard_butterfly_avx2(&self, data: &mut [f32]) {
        use std::arch::x86_64::*;

        let n = self.dim;
        let mut h = 1;

        while h < n {
            if h >= 8 {
                // SIMD path for h >= 8
                let mut j = 0;
                while j < n {
                    let mut k = 0;
                    while k + 8 <= h {
                        let ptr_a = data.as_mut_ptr().add(j + k);
                        let ptr_b = data.as_mut_ptr().add(j + k + h);

                        let a = _mm256_loadu_ps(ptr_a);
                        let b = _mm256_loadu_ps(ptr_b);

                        let sum = _mm256_add_ps(a, b);
                        let diff = _mm256_sub_ps(a, b);

                        _mm256_storeu_ps(ptr_a, sum);
                        _mm256_storeu_ps(ptr_b, diff);

                        k += 8;
                    }
                    // Handle remainder
                    while k < h {
                        let a = data[j + k];
                        let b = data[j + k + h];
                        data[j + k] = a + b;
                        data[j + k + h] = a - b;
                        k += 1;
                    }
                    j += h * 2;
                }
            } else {
                // Scalar path for h < 8
                let mut j = 0;
                while j < n {
                    for k in 0..h {
                        let a = data[j + k];
                        let b = data[j + k + h];
                        data[j + k] = a + b;
                        data[j + k + h] = a - b;
                    }
                    j += h * 2;
                }
            }
            h *= 2;
        }
    }

    /// Normalize data by 1/sqrt(n)
    #[inline]
    fn normalize(&self, data: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.normalize_neon(data);
            return;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    self.normalize_avx2(data);
                }
                return;
            }
        }

        // Scalar fallback
        for d in data.iter_mut() {
            *d *= self.norm_factor;
        }
    }

    /// NEON-optimized normalization
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn normalize_neon(&self, data: &mut [f32]) {
        use std::arch::aarch64::*;

        let n = data.len();
        let chunks = n / 4;
        let norm = vdupq_n_f32(self.norm_factor);
        let data_ptr = data.as_mut_ptr();

        for i in 0..chunks {
            let idx = i * 4;
            let v = vld1q_f32(data_ptr.add(idx));
            let result = vmulq_f32(v, norm);
            vst1q_f32(data_ptr.add(idx), result);
        }

        // Handle remainder
        for i in (chunks * 4)..n {
            data[i] *= self.norm_factor;
        }
    }

    /// AVX2-optimized normalization
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn normalize_avx2(&self, data: &mut [f32]) {
        use std::arch::x86_64::*;

        let n = data.len();
        let chunks = n / 8;
        let norm = _mm256_set1_ps(self.norm_factor);
        let data_ptr = data.as_mut_ptr();

        for i in 0..chunks {
            let idx = i * 8;
            let v = _mm256_loadu_ps(data_ptr.add(idx));
            let result = _mm256_mul_ps(v, norm);
            _mm256_storeu_ps(data_ptr.add(idx), result);
        }

        // Handle remainder
        for i in (chunks * 8)..n {
            data[i] *= self.norm_factor;
        }
    }

    /// Verify the orthogonality property: H × H^T = n × I
    ///
    /// This is the INV-4 property test from ADR-090.
    /// For a properly implemented Hadamard matrix, applying the transform
    /// twice (with proper normalization) should return the original data.
    ///
    /// # Returns
    ///
    /// True if the property holds within numerical tolerance
    pub fn verify_orthogonality(&self, tolerance: f32) -> bool {
        // Generate test data
        let mut data: Vec<f32> = (0..self.dim)
            .map(|i| (i as f32 + 1.0) / self.dim as f32)
            .collect();
        let original = data.clone();

        // Apply forward then inverse
        self.forward_inplace(&mut data);
        self.inverse_inplace(&mut data);

        // Check if we recovered the original
        for (a, b) in data.iter().zip(original.iter()) {
            if (a - b).abs() > tolerance {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Apply Hadamard transform to multiple vectors in batch
///
/// More efficient than calling forward_inplace repeatedly due to better
/// cache utilization and potential for parallelization.
pub fn hadamard_batch_transform(
    transform: &HadamardTransform,
    data: &mut [f32],
    batch_size: usize,
) -> Result<()> {
    let dim = transform.dim();
    if data.len() != batch_size * dim {
        return Err(RuvLLMError::Quantization(format!(
            "Data length {} doesn't match batch_size {} * dim {}",
            data.len(),
            batch_size,
            dim
        )));
    }

    // Process each vector in the batch
    for i in 0..batch_size {
        let start = i * dim;
        let end = start + dim;
        transform.forward_inplace(&mut data[start..end]);
    }

    Ok(())
}

/// Apply inverse Hadamard transform to multiple vectors in batch
pub fn hadamard_batch_inverse(
    transform: &HadamardTransform,
    data: &mut [f32],
    batch_size: usize,
) -> Result<()> {
    let dim = transform.dim();
    if data.len() != batch_size * dim {
        return Err(RuvLLMError::Quantization(format!(
            "Data length {} doesn't match batch_size {} * dim {}",
            data.len(),
            batch_size,
            dim
        )));
    }

    for i in 0..batch_size {
        let start = i * dim;
        let end = start + dim;
        transform.inverse_inplace(&mut data[start..end]);
    }

    Ok(())
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute the next power of 2 greater than or equal to n
#[inline]
pub fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    1usize << (usize::BITS - (n - 1).leading_zeros())
}

/// Compute log2 of a power of 2
#[inline]
pub fn log2_exact(n: usize) -> Option<u32> {
    if n == 0 || (n & (n - 1)) != 0 {
        return None;
    }
    Some(n.trailing_zeros())
}

/// Pad data to the next power of 2 with zeros
pub fn pad_to_power_of_2(data: &[f32]) -> Vec<f32> {
    let target_len = next_power_of_2(data.len());
    let mut padded = data.to_vec();
    padded.resize(target_len, 0.0);
    padded
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_basic() {
        let transform = HadamardTransform::deterministic(3).unwrap();
        assert_eq!(transform.dim(), 8);
        assert!(!transform.is_randomized());
    }

    #[test]
    fn test_hadamard_roundtrip() {
        // Test that forward followed by inverse recovers original data
        let transform = HadamardTransform::deterministic(4).unwrap();

        let original: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut data = original.clone();

        transform.forward_inplace(&mut data);
        transform.inverse_inplace(&mut data);

        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "Roundtrip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_hadamard_randomized_roundtrip() {
        let transform = HadamardTransform::randomized(5, 12345).unwrap();

        let original: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 10.0).collect();
        let mut data = original.clone();

        transform.forward_inplace(&mut data);
        transform.inverse_inplace(&mut data);

        for (a, b) in data.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Randomized roundtrip failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_orthogonality_property_inv4() {
        // INV-4: H × H^T = n × I
        // This means applying H twice (with proper normalization) gives back original
        let transform = HadamardTransform::deterministic(6).unwrap();
        assert!(
            transform.verify_orthogonality(1e-5),
            "Orthogonality property (INV-4) violated"
        );
    }

    #[test]
    fn test_orthogonality_randomized() {
        let transform = HadamardTransform::randomized(6, 42).unwrap();
        assert!(
            transform.verify_orthogonality(1e-5),
            "Randomized transform orthogonality violated"
        );
    }

    #[test]
    fn test_hadamard_known_values() {
        // Test against known Hadamard transform result for dim=4
        // H_4 = [[1,1,1,1], [1,-1,1,-1], [1,1,-1,-1], [1,-1,-1,1]]
        // For input [1,0,0,0], output should be [0.5, 0.5, 0.5, 0.5]
        let transform = HadamardTransform::deterministic(2).unwrap();
        let mut data = vec![1.0, 0.0, 0.0, 0.0];

        transform.forward_inplace(&mut data);

        // After normalization by 1/sqrt(4) = 0.5
        for &v in &data {
            assert!((v - 0.5).abs() < 1e-5, "Expected 0.5, got {}", v);
        }
    }

    #[test]
    fn test_energy_preservation() {
        // Hadamard transform preserves L2 norm (Parseval's theorem)
        let transform = HadamardTransform::deterministic(4).unwrap();

        let original: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let original_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut data = original.clone();
        transform.forward_inplace(&mut data);
        let transformed_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (original_norm - transformed_norm).abs() < 1e-4,
            "Energy not preserved: {} vs {}",
            original_norm,
            transformed_norm
        );
    }

    #[test]
    fn test_batch_transform() {
        let transform = HadamardTransform::deterministic(3).unwrap();
        let batch_size = 4;

        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let original = data.clone();

        hadamard_batch_transform(&transform, &mut data, batch_size).unwrap();
        hadamard_batch_inverse(&transform, &mut data, batch_size).unwrap();

        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(16), 16);
        assert_eq!(next_power_of_2(17), 32);
    }

    #[test]
    fn test_log2_exact() {
        assert_eq!(log2_exact(1), Some(0));
        assert_eq!(log2_exact(2), Some(1));
        assert_eq!(log2_exact(4), Some(2));
        assert_eq!(log2_exact(1024), Some(10));
        assert_eq!(log2_exact(3), None);
        assert_eq!(log2_exact(0), None);
    }

    #[test]
    fn test_large_dimension() {
        // Test with larger dimension (256 elements)
        let transform = HadamardTransform::deterministic(8).unwrap();

        let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 100.0).collect();
        let mut data = original.clone();

        transform.forward_inplace(&mut data);
        transform.inverse_inplace(&mut data);

        let max_error: f32 = data
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a.max(b));

        assert!(
            max_error < 1e-4,
            "Large dimension roundtrip error too high: {}",
            max_error
        );
    }

    #[test]
    fn test_error_on_invalid_dimension() {
        let result = HadamardTransform::new(MAX_LOG_DIM + 1, None);
        assert!(result.is_err());
    }
}
