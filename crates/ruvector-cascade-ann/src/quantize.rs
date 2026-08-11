//! INT8 scalar quantization for cascade ANN.
//!
//! Global linear quantization: a single (min, scale) pair maps every f32
//! component to a u8 code.  Asymmetric distance computation keeps the query
//! in f32 and dequantizes stored codes on the fly, so no lookup table is needed.
//!
//! Memory: N * D bytes instead of N * D * 4 bytes — 4× reduction.

/// Quantization parameters shared across all stored vectors.
#[derive(Debug, Clone)]
pub struct QuantParams {
    pub min: f32,
    pub scale: f32, // (max - min) / 255
}

impl QuantParams {
    /// Learn min/max from the corpus.
    pub fn from_vecs(vecs: &[Vec<f32>]) -> Self {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for v in vecs {
            for &x in v {
                if x < lo {
                    lo = x;
                }
                if x > hi {
                    hi = x;
                }
            }
        }
        let range = (hi - lo).max(1e-9);
        QuantParams {
            min: lo,
            scale: range / 255.0,
        }
    }

    #[inline(always)]
    pub fn encode(&self, x: f32) -> u8 {
        ((x - self.min) / self.scale).round().clamp(0.0, 255.0) as u8
    }

    #[inline(always)]
    pub fn decode(&self, c: u8) -> f32 {
        self.min + c as f32 * self.scale
    }
}

/// Compressed vector corpus stored as row-major u8 matrix.
pub struct QuantizedCorpus {
    pub n: usize,
    pub dim: usize,
    pub data: Vec<u8>, // n * dim bytes
    pub params: QuantParams,
    /// Original f32 vectors kept for the exact verify pass.
    pub f32_data: Vec<Vec<f32>>,
}

impl QuantizedCorpus {
    /// Build from f32 vectors.
    pub fn build(vecs: &[Vec<f32>]) -> Self {
        let n = vecs.len();
        let dim = vecs[0].len();
        let params = QuantParams::from_vecs(vecs);
        let mut data = vec![0u8; n * dim];
        for (i, v) in vecs.iter().enumerate() {
            for (d, &x) in v.iter().enumerate() {
                data[i * dim + d] = params.encode(x);
            }
        }
        QuantizedCorpus {
            n,
            dim,
            data,
            params,
            f32_data: vecs.to_vec(),
        }
    }

    /// Asymmetric squared Euclidean distance: query f32 vs stored u8 (dequantized).
    #[inline]
    pub fn dist_sq_approx(&self, id: usize, query: &[f32]) -> f32 {
        let row = &self.data[id * self.dim..(id + 1) * self.dim];
        let mut acc = 0.0f32;
        for (q, &c) in query.iter().zip(row.iter()) {
            let diff = q - self.params.decode(c);
            acc += diff * diff;
        }
        acc
    }

    /// Exact f32 squared Euclidean distance using stored originals.
    #[inline]
    pub fn dist_sq_exact(&self, id: usize, query: &[f32]) -> f32 {
        crate::dist_sq(query, &self.f32_data[id])
    }

    /// Memory used by the u8 corpus alone (excludes the f32 copy).
    pub fn quantized_bytes(&self) -> usize {
        self.data.len()
    }

    /// Memory used by the f32 copy.
    pub fn f32_bytes(&self) -> usize {
        self.n * self.dim * 4
    }
}
