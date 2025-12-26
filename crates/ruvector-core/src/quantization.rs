//! Quantization techniques for memory compression

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Trait for quantized vector representations
pub trait QuantizedVector: Send + Sync {
    /// Quantize a full-precision vector
    fn quantize(vector: &[f32]) -> Self;

    /// Calculate distance to another quantized vector
    fn distance(&self, other: &Self) -> f32;

    /// Reconstruct approximate full-precision vector
    fn reconstruct(&self) -> Vec<f32>;
}

/// Scalar quantization to int8 (4x compression)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantized {
    /// Quantized values (int8)
    pub data: Vec<u8>,
    /// Minimum value for dequantization
    pub min: f32,
    /// Scale factor for dequantization
    pub scale: f32,
}

impl QuantizedVector for ScalarQuantized {
    fn quantize(vector: &[f32]) -> Self {
        let min = vector.iter().copied().fold(f32::INFINITY, f32::min);
        let max = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Handle edge case where all values are the same (scale = 0)
        let scale = if (max - min).abs() < f32::EPSILON {
            1.0 // Arbitrary non-zero scale when all values are identical
        } else {
            (max - min) / 255.0
        };

        let data = vector
            .iter()
            .map(|&v| ((v - min) / scale).round().clamp(0.0, 255.0) as u8)
            .collect();

        Self { data, min, scale }
    }

    fn distance(&self, other: &Self) -> f32 {
        // Fast int8 distance calculation
        // Use i32 to avoid overflow: max diff is 255, and 255*255=65025 fits in i32

        // Scale handling: We use the average of both scales for balanced comparison.
        // Using max(scale) would bias toward the vector with larger range,
        // while average provides a more symmetric distance metric.
        // This ensures distance(a, b) ≈ distance(b, a) in the reconstructed space.
        let avg_scale = (self.scale + other.scale) / 2.0;

        self.data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| {
                let diff = a as i32 - b as i32;
                (diff * diff) as f32
            })
            .sum::<f32>()
            .sqrt()
            * avg_scale
    }

    fn reconstruct(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&v| self.min + (v as f32) * self.scale)
            .collect()
    }
}

/// Product quantization (8-16x compression)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantized {
    /// Quantized codes (one per subspace)
    pub codes: Vec<u8>,
    /// Codebooks for each subspace
    pub codebooks: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantized {
    /// Train product quantization on a set of vectors
    pub fn train(
        vectors: &[Vec<f32>],
        num_subspaces: usize,
        codebook_size: usize,
        iterations: usize,
    ) -> Result<Self> {
        if vectors.is_empty() {
            return Err(crate::error::RuvectorError::InvalidInput(
                "Cannot train on empty vector set".into(),
            ));
        }
        if vectors[0].is_empty() {
            return Err(crate::error::RuvectorError::InvalidInput(
                "Cannot train on vectors with zero dimensions".into(),
            ));
        }
        if codebook_size > 256 {
            return Err(crate::error::RuvectorError::InvalidParameter(format!(
                "Codebook size {} exceeds u8 maximum of 256",
                codebook_size
            )));
        }
        let dimensions = vectors[0].len();
        let subspace_dim = dimensions / num_subspaces;

        let mut codebooks = Vec::with_capacity(num_subspaces);

        // Train codebook for each subspace using k-means
        for subspace_idx in 0..num_subspaces {
            let start = subspace_idx * subspace_dim;
            let end = start + subspace_dim;

            // Extract subspace vectors
            let subspace_vectors: Vec<Vec<f32>> =
                vectors.iter().map(|v| v[start..end].to_vec()).collect();

            // Run k-means
            let codebook = kmeans_clustering(&subspace_vectors, codebook_size, iterations);
            codebooks.push(codebook);
        }

        Ok(Self {
            codes: vec![],
            codebooks,
        })
    }

    /// Quantize a vector using trained codebooks
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let num_subspaces = self.codebooks.len();
        let subspace_dim = vector.len() / num_subspaces;

        let mut codes = Vec::with_capacity(num_subspaces);

        for (subspace_idx, codebook) in self.codebooks.iter().enumerate() {
            let start = subspace_idx * subspace_dim;
            let end = start + subspace_dim;
            let subvector = &vector[start..end];

            // Find nearest centroid
            let code = codebook
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let dist_a = euclidean_squared(subvector, a);
                    let dist_b = euclidean_squared(subvector, b);
                    dist_a.partial_cmp(&dist_b).unwrap()
                })
                .map(|(idx, _)| idx as u8)
                .unwrap_or(0);

            codes.push(code);
        }

        codes
    }
}

/// Binary quantization (32x compression)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryQuantized {
    /// Binary representation (1 bit per dimension, packed into bytes)
    pub bits: Vec<u8>,
    /// Number of dimensions
    pub dimensions: usize,
}

impl QuantizedVector for BinaryQuantized {
    fn quantize(vector: &[f32]) -> Self {
        let dimensions = vector.len();
        let num_bytes = (dimensions + 7) / 8;
        let mut bits = vec![0u8; num_bytes];

        for (i, &v) in vector.iter().enumerate() {
            if v > 0.0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                bits[byte_idx] |= 1 << bit_idx;
            }
        }

        Self { bits, dimensions }
    }

    fn distance(&self, other: &Self) -> f32 {
        // Hamming distance
        let mut distance = 0u32;

        for (&a, &b) in self.bits.iter().zip(&other.bits) {
            distance += (a ^ b).count_ones();
        }

        distance as f32
    }

    fn reconstruct(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dimensions);

        for i in 0..self.dimensions {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bit = (self.bits[byte_idx] >> bit_idx) & 1;
            result.push(if bit == 1 { 1.0 } else { -1.0 });
        }

        result
    }
}

// Helper functions

fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

fn kmeans_clustering(vectors: &[Vec<f32>], k: usize, iterations: usize) -> Vec<Vec<f32>> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let mut rng = thread_rng();

    // Initialize centroids randomly
    let mut centroids: Vec<Vec<f32>> = vectors.choose_multiple(&mut rng, k).cloned().collect();

    for _ in 0..iterations {
        // Assign vectors to nearest centroid
        let mut assignments = vec![Vec::new(); k];

        for vector in vectors {
            let nearest = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let dist_a = euclidean_squared(vector, a);
                    let dist_b = euclidean_squared(vector, b);
                    dist_a.partial_cmp(&dist_b).unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            assignments[nearest].push(vector.clone());
        }

        // Update centroids
        for (centroid, assigned) in centroids.iter_mut().zip(&assignments) {
            if !assigned.is_empty() {
                let dim = centroid.len();
                *centroid = vec![0.0; dim];

                for vector in assigned {
                    for (i, &v) in vector.iter().enumerate() {
                        centroid[i] += v;
                    }
                }

                let count = assigned.len() as f32;
                for v in centroid.iter_mut() {
                    *v /= count;
                }
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_quantization() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quantized = ScalarQuantized::quantize(&vector);
        let reconstructed = quantized.reconstruct();

        // Check approximate reconstruction
        for (orig, recon) in vector.iter().zip(&reconstructed) {
            assert!((orig - recon).abs() < 0.1);
        }
    }

    #[test]
    fn test_binary_quantization() {
        let vector = vec![1.0, -1.0, 2.0, -2.0, 0.5];
        let quantized = BinaryQuantized::quantize(&vector);

        assert_eq!(quantized.dimensions, 5);
        assert_eq!(quantized.bits.len(), 1); // 5 bits fit in 1 byte
    }

    #[test]
    fn test_binary_distance() {
        let v1 = vec![1.0, 1.0, 1.0, 1.0];
        let v2 = vec![1.0, 1.0, -1.0, -1.0];

        let q1 = BinaryQuantized::quantize(&v1);
        let q2 = BinaryQuantized::quantize(&v2);

        let dist = q1.distance(&q2);
        assert_eq!(dist, 2.0); // 2 bits differ
    }

    #[test]
    fn test_scalar_quantization_roundtrip() {
        // Test that quantize -> reconstruct produces values close to original
        let test_vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![-10.0, -5.0, 0.0, 5.0, 10.0],
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![100.0, 200.0, 300.0, 400.0, 500.0],
        ];

        for vector in test_vectors {
            let quantized = ScalarQuantized::quantize(&vector);
            let reconstructed = quantized.reconstruct();

            assert_eq!(vector.len(), reconstructed.len());

            for (orig, recon) in vector.iter().zip(reconstructed.iter()) {
                // With 8-bit quantization, max error is roughly (max-min)/255
                let max = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let min = vector.iter().copied().fold(f32::INFINITY, f32::min);
                let max_error = (max - min) / 255.0 * 2.0; // Allow 2x for rounding

                assert!(
                    (orig - recon).abs() < max_error,
                    "Roundtrip error too large: orig={}, recon={}, error={}",
                    orig,
                    recon,
                    (orig - recon).abs()
                );
            }
        }
    }

    #[test]
    fn test_scalar_distance_symmetry() {
        // Test that distance(a, b) == distance(b, a)
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let q1 = ScalarQuantized::quantize(&v1);
        let q2 = ScalarQuantized::quantize(&v2);

        let dist_ab = q1.distance(&q2);
        let dist_ba = q2.distance(&q1);

        // Distance should be symmetric (within floating point precision)
        assert!(
            (dist_ab - dist_ba).abs() < 0.01,
            "Distance is not symmetric: d(a,b)={}, d(b,a)={}",
            dist_ab,
            dist_ba
        );
    }

    #[test]
    fn test_scalar_distance_different_scales() {
        // Test distance calculation with vectors that have different scales
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // range: 4.0
        let v2 = vec![10.0, 20.0, 30.0, 40.0, 50.0]; // range: 40.0

        let q1 = ScalarQuantized::quantize(&v1);
        let q2 = ScalarQuantized::quantize(&v2);

        let dist_ab = q1.distance(&q2);
        let dist_ba = q2.distance(&q1);

        // With average scaling, symmetry should be maintained
        assert!(
            (dist_ab - dist_ba).abs() < 0.01,
            "Distance with different scales not symmetric: d(a,b)={}, d(b,a)={}",
            dist_ab,
            dist_ba
        );
    }

    #[test]
    fn test_scalar_quantization_edge_cases() {
        // Test with all same values
        let same_values = vec![5.0, 5.0, 5.0, 5.0];
        let quantized = ScalarQuantized::quantize(&same_values);
        let reconstructed = quantized.reconstruct();

        for (orig, recon) in same_values.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 0.01);
        }

        // Test with extreme ranges
        let extreme = vec![f32::MIN / 1e10, 0.0, f32::MAX / 1e10];
        let quantized = ScalarQuantized::quantize(&extreme);
        let reconstructed = quantized.reconstruct();

        assert_eq!(extreme.len(), reconstructed.len());
    }

    #[test]
    fn test_binary_distance_symmetry() {
        // Test that binary distance is symmetric
        let v1 = vec![1.0, -1.0, 1.0, -1.0];
        let v2 = vec![1.0, 1.0, -1.0, -1.0];

        let q1 = BinaryQuantized::quantize(&v1);
        let q2 = BinaryQuantized::quantize(&v2);

        let dist_ab = q1.distance(&q2);
        let dist_ba = q2.distance(&q1);

        assert_eq!(
            dist_ab, dist_ba,
            "Binary distance not symmetric: d(a,b)={}, d(b,a)={}",
            dist_ab, dist_ba
        );
    }
}
