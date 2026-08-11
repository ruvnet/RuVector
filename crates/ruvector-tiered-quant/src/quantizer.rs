//! Scalar and binary vector quantizers.

/// Scalar quantizer: maps each f32 dimension linearly to a u8 byte.
///
/// Per-dimension min/max are computed from the input vector only (self-contained).
/// This is "uniform scalar quantization" — each dimension independently.
pub struct ScalarQuantizer {
    dims: usize,
}

impl ScalarQuantizer {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }

    /// Quantize a single f32 vector to `(bytes, min_per_dim, scale_per_dim)`.
    ///
    /// `min[i]` is the dimension minimum; `scale[i] = (max[i] - min[i]) / 255.0`.
    /// Reconstruction: `f32_approx[i] = min[i] + bytes[i] as f32 * scale[i]`.
    pub fn quantize(&self, v: &[f32]) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        assert_eq!(v.len(), self.dims);
        let mut bytes = vec![0u8; self.dims];
        let mut min = vec![0.0f32; self.dims];
        let mut scale = vec![0.0f32; self.dims];

        for i in 0..self.dims {
            let mn = v[i];
            let mx = v[i];
            // Single-vector quantization: min == max → use a small epsilon.
            // In practice this is called across a corpus; per-vector SQ works
            // well for uniform data and serves as a fast approximation baseline.
            let range = mx - mn;
            let sc = if range.abs() < 1e-9 {
                1.0 / 255.0
            } else {
                range / 255.0
            };
            min[i] = mn;
            scale[i] = sc;
            bytes[i] = 0; // single-value → 0
        }

        // For a proper corpus-aware quantizer we need the per-dimension stats.
        // This variant uses a global min/max across the full vector instead,
        // which is a reasonable trade-off for a single-vector call.
        let global_min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let global_max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let global_range = global_max - global_min;
        let global_scale = if global_range < 1e-9 {
            1.0 / 255.0
        } else {
            global_range / 255.0
        };

        for i in 0..self.dims {
            min[i] = global_min;
            scale[i] = global_scale;
            let normalized = (v[i] - global_min) / global_range.max(1e-9);
            bytes[i] = (normalized * 255.0).clamp(0.0, 255.0) as u8;
        }

        (bytes, min, scale)
    }

    pub fn dims(&self) -> usize {
        self.dims
    }
}

/// Binary quantizer: maps each f32 dimension to a single bit.
///
/// Quantization rule: `bit = 1` if `v[i] > 0.0` else `bit = 0`.
/// Bits are packed into u64 words (LSB first).
pub struct BinaryQuantizer {
    dims: usize,
    words: usize,
}

impl BinaryQuantizer {
    pub fn new(dims: usize) -> Self {
        let words = dims.div_ceil(64);
        Self { dims, words }
    }

    /// Quantize a full-precision vector to a packed bit array.
    pub fn quantize(&self, v: &[f32]) -> Vec<u64> {
        assert_eq!(v.len(), self.dims);
        let mut packed = vec![0u64; self.words];
        for (i, &val) in v.iter().enumerate() {
            if val > 0.0 {
                packed[i / 64] |= 1u64 << (i % 64);
            }
        }
        packed
    }

    /// Compute Hamming distance between two packed bit arrays.
    pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn words(&self) -> usize {
        self.words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_quantizer_reconstructs_approximately() {
        let sq = ScalarQuantizer::new(4);
        let v = vec![0.1f32, 0.5, 0.9, -0.3];
        let (bytes, min, scale) = sq.quantize(&v);

        for i in 0..4 {
            let reconstructed = min[i] + bytes[i] as f32 * scale[i];
            let err = (v[i] - reconstructed).abs();
            assert!(err < 0.01, "dim {i}: reconstruction error {err} too large");
        }
    }

    #[test]
    fn binary_quantizer_hamming_zero_for_same() {
        let bq = BinaryQuantizer::new(128);
        let v: Vec<f32> = (0..128)
            .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        let p = bq.quantize(&v);
        assert_eq!(BinaryQuantizer::hamming(&p, &p), 0);
    }

    #[test]
    fn binary_quantizer_hamming_max_for_inverse() {
        let bq = BinaryQuantizer::new(64);
        let pos: Vec<f32> = vec![1.0f32; 64];
        let neg: Vec<f32> = vec![-1.0f32; 64];
        let pp = bq.quantize(&pos);
        let np = bq.quantize(&neg);
        assert_eq!(BinaryQuantizer::hamming(&pp, &np), 64);
    }
}
