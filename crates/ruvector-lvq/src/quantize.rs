//! Single-level 8-bit Locally-Adaptive Vector Quantization.
//!
//! For each input vector `v ∈ R^d` we store:
//!   * `bias`  — minimum of `(v - mean(v))`
//!   * `scale` — `(max - min)` of the centered vector divided by 255
//!   * `mean`  — per-vector mean (kept so reconstruction matches the *original*
//!     vector, not just the centered one — this lets us reuse query-side
//!     dot products without subtracting the mean every search)
//!   * `code`  — `d` bytes; `code[j] = round((v[j] - mean - bias) / scale)`
//!
//! Decoding is `v[j] ≈ mean + bias + scale * code[j]`.
//!
//! Compared to a fixed-range global int8 quantizer, the per-vector scale
//! adapts to each vector's dynamic range — preserving precision for
//! low-magnitude vectors and avoiding saturation on outliers. This is the
//! key insight from the LVQ paper.

use serde::{Deserialize, Serialize};

use crate::error::LvqError;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Lvq8Stats {
    pub mean: f32,
    pub bias: f32,
    pub scale: f32,
}

impl Lvq8Stats {
    #[inline]
    pub fn decode_lane(&self, code: u8) -> f32 {
        self.mean + self.bias + self.scale * (code as f32)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lvq8Code {
    pub stats: Lvq8Stats,
    pub code: Vec<u8>,
}

impl Lvq8Code {
    pub fn dim(&self) -> usize {
        self.code.len()
    }

    /// Reconstruct the original vector with the unavoidable
    /// quantization error.
    pub fn decode(&self) -> Vec<f32> {
        self.code
            .iter()
            .map(|&c| self.stats.decode_lane(c))
            .collect()
    }

    /// Bytes written to disk for this code, including stats overhead.
    /// Useful for honest memory accounting.
    pub fn byte_size(&self) -> usize {
        self.code.len() + std::mem::size_of::<Lvq8Stats>()
    }
}

/// Stateless encoder / batch container for LVQ-8.
///
/// Holds a contiguous flat array of codes (`n * dim` bytes) plus a parallel
/// stats array — this is the layout you want for SIMD-friendly scans.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Lvq8 {
    pub dim: usize,
    pub stats: Vec<Lvq8Stats>,
    pub codes: Vec<u8>,
}

impl Lvq8 {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            stats: Vec::new(),
            codes: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.stats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }

    pub fn byte_size(&self) -> usize {
        self.codes.len() + self.stats.len() * std::mem::size_of::<Lvq8Stats>()
    }

    /// Encode a single vector and append it to the batch.
    pub fn push(&mut self, v: &[f32]) -> Result<(), LvqError> {
        if v.len() != self.dim {
            return Err(LvqError::DimMismatch {
                expected: self.dim,
                actual: v.len(),
            });
        }
        let (stats, code) = encode_one(v)?;
        self.stats.push(stats);
        self.codes.extend_from_slice(&code);
        Ok(())
    }

    /// Bulk-encode a row-major `n x dim` slice.
    pub fn extend_from_flat(&mut self, flat: &[f32]) -> Result<(), LvqError> {
        if flat.is_empty() {
            return Err(LvqError::Empty);
        }
        if flat.len() % self.dim != 0 {
            return Err(LvqError::DimMismatch {
                expected: self.dim,
                actual: flat.len() % self.dim,
            });
        }
        for chunk in flat.chunks_exact(self.dim) {
            self.push(chunk)?;
        }
        Ok(())
    }

    /// Borrow the i-th code row.
    #[inline]
    pub fn code_row(&self, i: usize) -> &[u8] {
        let off = i * self.dim;
        &self.codes[off..off + self.dim]
    }

    /// Borrow the i-th stats entry.
    #[inline]
    pub fn stats_at(&self, i: usize) -> Lvq8Stats {
        self.stats[i]
    }

    /// Materialize the i-th vector back to f32. Used for reranking.
    pub fn decode(&self, i: usize) -> Vec<f32> {
        let s = self.stats[i];
        self.code_row(i)
            .iter()
            .map(|&c| s.decode_lane(c))
            .collect()
    }

    /// Compute the residual `v - decode(i)` for the given original vector.
    /// Used to feed the second LVQ level.
    pub fn residual(&self, i: usize, v: &[f32]) -> Vec<f32> {
        let s = self.stats[i];
        let row = self.code_row(i);
        v.iter()
            .zip(row.iter())
            .map(|(x, &c)| x - s.decode_lane(c))
            .collect()
    }
}

/// Encode a single fp32 vector into LVQ-8 stats + codes.
pub fn encode_one(v: &[f32]) -> Result<(Lvq8Stats, Vec<u8>), LvqError> {
    if v.is_empty() {
        return Err(LvqError::Empty);
    }
    let mut sum = 0.0_f64;
    for (i, &x) in v.iter().enumerate() {
        if !x.is_finite() {
            return Err(LvqError::NonFinite(i));
        }
        sum += x as f64;
    }
    let mean = (sum / v.len() as f64) as f32;

    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &x in v {
        let c = x - mean;
        if c < lo {
            lo = c;
        }
        if c > hi {
            hi = c;
        }
    }
    // Degenerate (all-equal) vector: scale=0 and codes all zero. Decoder
    // returns mean+bias which equals each input.
    let range = hi - lo;
    let scale = if range > 0.0 { range / 255.0 } else { 0.0 };

    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let mut codes = Vec::with_capacity(v.len());
    for &x in v {
        let centered = x - mean - lo;
        let q = if scale > 0.0 {
            (centered * inv_scale).round().clamp(0.0, 255.0) as u8
        } else {
            0
        };
        codes.push(q);
    }

    Ok((
        Lvq8Stats {
            mean,
            bias: lo,
            scale,
        },
        codes,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_recovers_within_tolerance() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let (stats, code) = encode_one(&v).unwrap();
        let decoded: Vec<f32> = code.iter().map(|&c| stats.decode_lane(c)).collect();

        let max_err = v
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // 8-bit LVQ on a range-2 signal: half-step ~ 2/255 ≈ 7.84e-3.
        assert!(max_err < 1.0e-2, "max_err = {max_err}");
    }

    #[test]
    fn handles_constant_vector() {
        let v = vec![3.5_f32; 64];
        let (stats, code) = encode_one(&v).unwrap();
        assert_eq!(stats.scale, 0.0);
        for c in &code {
            assert_eq!(*c, 0);
        }
        let dec: Vec<f32> = code.iter().map(|&c| stats.decode_lane(c)).collect();
        for x in dec {
            assert!((x - 3.5).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_non_finite() {
        let v = vec![1.0, f32::NAN, 2.0];
        assert!(matches!(encode_one(&v), Err(LvqError::NonFinite(1))));
    }
}
