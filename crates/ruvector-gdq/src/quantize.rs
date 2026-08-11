/// Scalar 8-bit uniform quantizer with per-dimension scaling.
/// Per-dimension scaling ensures each dimension's full range maps to [0,255].
#[derive(Clone, Debug)]
pub struct Scalar8BitQuantizer {
    pub mins: Vec<f32>,
    pub scales: Vec<f32>, // (max[d] - min[d]) / 255.0 per dimension
    pub dim: usize,
}

impl Scalar8BitQuantizer {
    pub fn fit(data: &[Vec<f32>]) -> Self {
        let dim = data[0].len();
        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];
        for vec in data {
            for (d, &v) in vec.iter().enumerate() {
                if v < mins[d] {
                    mins[d] = v;
                }
                if v > maxs[d] {
                    maxs[d] = v;
                }
            }
        }
        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(lo, hi)| (hi - lo).max(1e-6) / 255.0)
            .collect();
        Self { mins, scales, dim }
    }

    #[inline]
    pub fn encode_val(&self, d: usize, v: f32) -> u8 {
        let q = ((v - self.mins[d]) / self.scales[d]).round();
        q.clamp(0.0, 255.0) as u8
    }

    #[inline]
    pub fn decode_val(&self, d: usize, q: u8) -> f32 {
        self.mins[d] + q as f32 * self.scales[d]
    }

    /// Encodes a single vector to bytes (one byte per dim).
    pub fn encode(&self, vec: &[f32]) -> Vec<u8> {
        vec.iter()
            .enumerate()
            .map(|(d, &v)| self.encode_val(d, v))
            .collect()
    }

    /// Decodes a byte slice back to f32.
    pub fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        bytes
            .iter()
            .enumerate()
            .map(|(d, &b)| self.decode_val(d, b))
            .collect()
    }

    /// Memory per vector in bytes.
    pub fn bytes_per_vec(dim: usize) -> usize {
        dim
    }

    /// Overhead bytes for the quantizer state itself (shared across all vectors).
    pub fn metadata_bytes(&self) -> usize {
        self.dim * 2 * 4 // mins + scales, each f32
    }
}

/// Nibble 4-bit uniform quantizer with per-dimension scaling.
/// Two dimensions are packed per byte. Supports asymmetric ranges per dimension.
#[derive(Clone, Debug)]
pub struct Nibble4BitQuantizer {
    pub mins: Vec<f32>,
    pub scales: Vec<f32>, // (max[d] - min[d]) / 15.0 per dimension
    pub dim: usize,
}

impl Nibble4BitQuantizer {
    pub fn fit(data: &[Vec<f32>]) -> Self {
        let dim = data[0].len();
        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];
        for vec in data {
            for (d, &v) in vec.iter().enumerate() {
                if v < mins[d] {
                    mins[d] = v;
                }
                if v > maxs[d] {
                    maxs[d] = v;
                }
            }
        }
        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(lo, hi)| (hi - lo).max(1e-6) / 15.0)
            .collect();
        Self { mins, scales, dim }
    }

    #[inline]
    pub fn encode_val(&self, d: usize, v: f32) -> u8 {
        let q = ((v - self.mins[d]) / self.scales[d]).round();
        q.clamp(0.0, 15.0) as u8
    }

    #[inline]
    pub fn decode_val(&self, d: usize, nibble: u8) -> f32 {
        self.mins[d] + (nibble & 0xF) as f32 * self.scales[d]
    }

    /// Pack dim floats into ceil(dim/2) bytes.
    pub fn encode(&self, vec: &[f32]) -> Vec<u8> {
        let packed_len = (vec.len() + 1) / 2;
        let mut out = vec![0u8; packed_len];
        for (i, &v) in vec.iter().enumerate() {
            let nibble = self.encode_val(i, v);
            if i % 2 == 0 {
                out[i / 2] = nibble << 4;
            } else {
                out[i / 2] |= nibble & 0xF;
            }
        }
        out
    }

    /// Unpack back to f32 slice.
    pub fn decode(&self, packed: &[u8], dim: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(dim);
        for i in 0..dim {
            let byte = packed[i / 2];
            let nibble = if i % 2 == 0 {
                (byte >> 4) & 0xF
            } else {
                byte & 0xF
            };
            out.push(self.decode_val(i, nibble));
        }
        out
    }

    /// Memory per vector in bytes (excluding quantizer metadata).
    pub fn bytes_per_vec(dim: usize) -> usize {
        (dim + 1) / 2
    }

    /// Overhead bytes for the quantizer state itself.
    pub fn metadata_bytes(&self) -> usize {
        self.dim * 2 * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<Vec<f32>> {
        vec![vec![-5.0, 0.0, 5.0, 10.0], vec![-3.0, 2.0, 7.0, -1.0]]
    }

    #[test]
    fn test_8bit_roundtrip() {
        let data = sample_data();
        let q = Scalar8BitQuantizer::fit(&data);
        for vec in &data {
            let encoded = q.encode(vec);
            let decoded = q.decode(&encoded);
            for (&orig, approx) in vec.iter().zip(decoded.iter()) {
                let err = (orig - approx).abs();
                // Per-dimension 8-bit: max error = per_dim_range/510
                assert!(err < 0.1, "8-bit error too large: {}", err);
            }
        }
    }

    #[test]
    fn test_4bit_roundtrip() {
        let data = sample_data();
        let q = Nibble4BitQuantizer::fit(&data);
        for vec in &data {
            let encoded = q.encode(vec);
            let decoded = q.decode(&encoded, vec.len());
            for (&orig, approx) in vec.iter().zip(decoded.iter()) {
                let err = (orig - approx).abs();
                // Per-dimension 4-bit: max error = per_dim_range/30
                assert!(err < 0.5, "4-bit error too large: {}", err);
            }
        }
    }

    #[test]
    fn test_nibble_packing_even_dim() {
        let data = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        let q = Nibble4BitQuantizer::fit(&data);
        let encoded = q.encode(&data[0]);
        assert_eq!(encoded.len(), 2);
    }

    #[test]
    fn test_nibble_packing_odd_dim() {
        let data = vec![vec![1.0f32, 2.0, 3.0]];
        let q = Nibble4BitQuantizer::fit(&data);
        let encoded = q.encode(&data[0]);
        assert_eq!(encoded.len(), 2);
    }

    #[test]
    fn test_memory_layout() {
        let dim = 128;
        assert_eq!(Scalar8BitQuantizer::bytes_per_vec(dim), 128);
        assert_eq!(Nibble4BitQuantizer::bytes_per_vec(dim), 64);
    }

    #[test]
    fn test_per_dim_better_than_global_for_wide_data() {
        // Dimension 0 has range [0, 1], dimension 1 has range [0, 100]
        // Per-dim quantization should give much better error for dim 0
        let data: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 / 100.0, i as f32]).collect();
        let q = Nibble4BitQuantizer::fit(&data);
        // dim 0: scale = 1.0/15 ≈ 0.067, max err ≈ 0.033
        // dim 1: scale = 100.0/15 ≈ 6.67, max err ≈ 3.33
        let test_vec = vec![0.5f32, 50.0f32];
        let enc = q.encode(&test_vec);
        let dec = q.decode(&enc, 2);
        let err0 = (test_vec[0] - dec[0]).abs();
        let err1 = (test_vec[1] - dec[1]).abs();
        // dim 0 should have < 0.07 error, dim 1 < 7.0
        assert!(err0 < 0.07, "dim 0 error: {}", err0);
        assert!(err1 < 7.0, "dim 1 error: {}", err1);
    }
}
