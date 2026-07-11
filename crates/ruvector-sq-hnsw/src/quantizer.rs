//! Per-dimension scalar quantizer: 8-bit (256 levels) and 4-bit (16 levels).
//!
//! Training computes per-dimension [min, max] from a corpus.
//! Encoding maps each f32 to an integer in [0, 2^bits - 1].
//! Distance computation scales integer differences back to approximate f32 L2.

/// Scalar quantizer trained on a corpus of f32 vectors.
pub struct ScalarQuantizer {
    pub dims: usize,
    pub bits: u8,
    /// Per-dimension minimum (origin of the quantization range).
    pub min: Vec<f32>,
    /// Per-dimension range (max - min), floored to 1e-9 to avoid div-by-zero.
    pub scale: Vec<f32>,
}

impl ScalarQuantizer {
    /// Train from a slice of vectors.  All must have identical length.
    pub fn train(vectors: &[Vec<f32>], bits: u8) -> Self {
        assert!(!vectors.is_empty(), "need at least one training vector");
        assert!(bits == 4 || bits == 8, "bits must be 4 or 8");
        let dims = vectors[0].len();
        let mut min = vec![f32::INFINITY; dims];
        let mut max = vec![f32::NEG_INFINITY; dims];
        for v in vectors {
            for (d, &x) in v.iter().enumerate() {
                if x < min[d] {
                    min[d] = x;
                }
                if x > max[d] {
                    max[d] = x;
                }
            }
        }
        let scale = min
            .iter()
            .zip(max.iter())
            .map(|(mn, mx)| (mx - mn).max(1e-9))
            .collect();
        ScalarQuantizer {
            dims,
            bits,
            min,
            scale,
        }
    }

    /// Encode to 8-bit: one byte per dimension.
    pub fn encode8(&self, v: &[f32]) -> Vec<u8> {
        v.iter()
            .enumerate()
            .map(|(d, &x)| {
                let t = (x - self.min[d]) / self.scale[d];
                (t.clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect()
    }

    /// Encode to 4-bit: two dimensions packed per byte (high nibble = even dim).
    pub fn encode4(&self, v: &[f32]) -> Vec<u8> {
        let n_bytes = (self.dims + 1) / 2;
        let mut bytes = vec![0u8; n_bytes];
        for d in 0..self.dims {
            let t = (v[d] - self.min[d]) / self.scale[d];
            let q = (t.clamp(0.0, 1.0) * 15.0).round() as u8;
            if d % 2 == 0 {
                bytes[d / 2] |= q << 4;
            } else {
                bytes[d / 2] |= q & 0x0F;
            }
        }
        bytes
    }

    /// Approximate squared L2 distance between two 8-bit encoded vectors.
    /// Scales integer delta by the per-dimension range.
    #[inline]
    pub fn l2_sq8(&self, a: &[u8], b: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for d in 0..self.dims {
            let diff = (a[d] as f32 - b[d] as f32) * (self.scale[d] / 255.0);
            sum += diff * diff;
        }
        sum
    }

    /// Approximate squared L2 distance between two 4-bit encoded vectors.
    #[inline]
    pub fn l2_sq4(&self, a: &[u8], b: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        let n_bytes = (self.dims + 1) / 2;
        for i in 0..n_bytes {
            let ai0 = (a[i] >> 4) as f32;
            let bi0 = (b[i] >> 4) as f32;
            let d0 = i * 2;
            if d0 < self.dims {
                let diff = (ai0 - bi0) * (self.scale[d0] / 15.0);
                sum += diff * diff;
            }
            let ai1 = (a[i] & 0x0F) as f32;
            let bi1 = (b[i] & 0x0F) as f32;
            let d1 = d0 + 1;
            if d1 < self.dims {
                let diff = (ai1 - bi1) * (self.scale[d1] / 15.0);
                sum += diff * diff;
            }
        }
        sum
    }

    /// Bytes per encoded 8-bit vector.
    pub fn encoded_bytes8(&self) -> usize {
        self.dims
    }

    /// Bytes per encoded 4-bit vector.
    pub fn encoded_bytes4(&self) -> usize {
        (self.dims + 1) / 2
    }
}
