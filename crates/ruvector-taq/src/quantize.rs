//! Scalar quantization to 8-bit and 4-bit (nibble-packed) representations.
//!
//! SQ8: one u8 per dimension. 4x compression vs f32. Max error: scale/255 per dim.
//! SQ4: two dimensions per byte (nibble). 8x compression. Max error: scale/15 per dim.

#[derive(Clone, Debug)]
pub struct Sq8Params {
    pub mins: Vec<f32>,
    pub scales: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Sq4Params {
    pub mins: Vec<f32>,
    pub scales: Vec<f32>,
}

impl Sq8Params {
    /// Fit quantization parameters from a slice of training vectors.
    pub fn fit(vectors: &[Vec<f32>], dim: usize) -> Self {
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for v in vectors {
            for (d, &x) in v.iter().enumerate() {
                if x < mins[d] {
                    mins[d] = x;
                }
                if x > maxs[d] {
                    maxs[d] = x;
                }
            }
        }
        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(mn, mx)| {
                let range = mx - mn;
                if range < 1e-9 {
                    1.0
                } else {
                    range / 255.0
                }
            })
            .collect();
        Self { mins, scales }
    }

    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        v.iter()
            .zip(self.mins.iter())
            .zip(self.scales.iter())
            .map(|((x, mn), sc)| ((x - mn) / sc).clamp(0.0, 255.0).round() as u8)
            .collect()
    }

    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        codes
            .iter()
            .zip(self.mins.iter())
            .zip(self.scales.iter())
            .map(|((c, mn), sc)| mn + *c as f32 * sc)
            .collect()
    }

    /// Memory per stored vector in bytes (excludes params overhead).
    pub fn bytes_per_vec(&self) -> usize {
        self.mins.len()
    }
}

impl Sq4Params {
    pub fn fit(vectors: &[Vec<f32>], dim: usize) -> Self {
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for v in vectors {
            for (d, &x) in v.iter().enumerate() {
                if x < mins[d] {
                    mins[d] = x;
                }
                if x > maxs[d] {
                    maxs[d] = x;
                }
            }
        }
        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(mn, mx)| {
                let range = mx - mn;
                if range < 1e-9 {
                    1.0
                } else {
                    range / 15.0
                }
            })
            .collect();
        Self { mins, scales }
    }

    /// Encode to nibble-packed bytes: 2 dimensions per byte (high nibble first).
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        let dim = v.len();
        let byte_count = (dim + 1) / 2;
        let mut codes = vec![0u8; byte_count];
        for (i, (x, (mn, sc))) in v
            .iter()
            .zip(self.mins.iter().zip(self.scales.iter()))
            .enumerate()
        {
            let nib = ((x - mn) / sc).clamp(0.0, 15.0).round() as u8;
            if i % 2 == 0 {
                codes[i / 2] = nib << 4;
            } else {
                codes[i / 2] |= nib;
            }
        }
        codes
    }

    pub fn decode(&self, codes: &[u8], dim: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(dim);
        for i in 0..dim {
            let byte = codes[i / 2];
            let nib = if i % 2 == 0 {
                (byte >> 4) & 0x0F
            } else {
                byte & 0x0F
            };
            result.push(self.mins[i] + nib as f32 * self.scales[i]);
        }
        result
    }

    pub fn bytes_per_vec(&self, dim: usize) -> usize {
        (dim + 1) / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vecs() -> Vec<Vec<f32>> {
        vec![
            vec![0.0, 0.5, 1.0, -0.5],
            vec![-1.0, 0.0, 0.5, 1.0],
            vec![0.3, -0.3, 0.7, -0.7],
        ]
    }

    #[test]
    fn sq8_roundtrip_error_bounded() {
        let vecs = sample_vecs();
        let params = Sq8Params::fit(&vecs, 4);
        for v in &vecs {
            let encoded = params.encode(v);
            let decoded = params.decode(&encoded);
            for (orig, dec) in v.iter().zip(decoded.iter()) {
                let max_err = params.scales.iter().cloned().fold(0.0f32, f32::max);
                assert!(
                    (orig - dec).abs() <= max_err + 1e-5,
                    "SQ8 error too large: {} vs {} (max_scale={})",
                    orig,
                    dec,
                    max_err
                );
            }
        }
    }

    #[test]
    fn sq4_roundtrip_error_bounded() {
        let vecs = sample_vecs();
        let params = Sq4Params::fit(&vecs, 4);
        for v in &vecs {
            let encoded = params.encode(v);
            let decoded = params.decode(&encoded, 4);
            assert_eq!(decoded.len(), 4);
            for (orig, dec) in v.iter().zip(decoded.iter()) {
                let max_err = params.scales.iter().cloned().fold(0.0f32, f32::max);
                assert!(
                    (orig - dec).abs() <= max_err + 1e-5,
                    "SQ4 error too large: {} vs {} (max_scale={})",
                    orig,
                    dec,
                    max_err
                );
            }
        }
    }

    #[test]
    fn sq8_memory_is_dim_bytes() {
        let vecs = sample_vecs();
        let params = Sq8Params::fit(&vecs, 4);
        assert_eq!(params.bytes_per_vec(), 4);
    }

    #[test]
    fn sq4_memory_is_half_dim_bytes() {
        let vecs = sample_vecs();
        let params = Sq4Params::fit(&vecs, 4);
        assert_eq!(params.bytes_per_vec(4), 2); // 4 dims → 2 bytes
    }
}
