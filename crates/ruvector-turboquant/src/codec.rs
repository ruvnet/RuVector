//! Turbo4 encoder: rotated, standardized, 4-bit Lloyd-Max packed codes.
//!
//! ## Code blob layout (`code_len = D/2 + 8` bytes)
//!
//! ```text
//! [ D/2 packed nibbles | α: f32 LE | S: f32 LE ]
//! ```
//!
//! * byte `i` holds dim `i` in the **low** nibble and dim `i + D/2` in the
//!   **high** nibble — so SIMD unpacking yields two *contiguous* dimension
//!   runs (`0..D/2` and `D/2..D`) with no cross-lane shuffling;
//! * `α = ‖v‖₂ / √D` — the standardization factor (rotated coords are divided
//!   by α before table lookup, so they're ~N(0,1));
//! * `S = Σ level(cᵢ)²` — precomputed for the L2 decomposition.
//!
//! ## Query blob layout (`query_len = D + 8` bytes)
//!
//! ```text
//! [ D int8 codes | qscale: f32 LE | ‖q‖²: f32 LE ]
//! ```
//!
//! `q_i8[i] = round(q_rot[i] / qscale)`, `qscale = max|q_rot| / 127`.
//!
//! Blob lengths are structurally disjoint (`D/2+8` vs `D+8` for `D ≥ 2`), so a
//! scorer can tell the roles apart from slice lengths alone — this is what lets
//! `hnsw_rs::Distance<u8>::eval` run asymmetric scoring during traversal and
//! symmetric scoring during graph construction with one distance functor.
//!
//! The original f32 vector is **never stored** — decoding reconstructs an
//! approximation only, and only for tests/debugging.

use crate::rotation::Rotation;
use crate::tables::{level, quantize_coord, LEVELS_F32};
use crate::TurboQuantError;

/// Bytes of per-blob constants (α + S, or qscale + ‖q‖²).
pub const META_BYTES: usize = 8;

/// A prepared query: the persisted-format blob plus the exact rotated f32
/// coordinates for final rescoring.
pub struct Turbo4Query {
    /// `[D i8 | qscale | ‖q‖²]` — feed this to the traversal scorer.
    pub blob: Vec<u8>,
    /// Exact rotated query, for `rescore` (never persisted).
    pub rotated: Vec<f32>,
    /// Exact squared norm of the query.
    pub norm_sq: f32,
}

/// The Turbo4 codec for a fixed (dimension, rotation-seed) pair.
pub struct Turbo4Codec {
    dim: usize,
    rotation: Rotation,
}

impl Turbo4Codec {
    /// Build a codec. `dim` must be even and ≥ 2 (all practical embedding
    /// widths are; evenness keeps the two-run nibble layout exact).
    pub fn new(dim: usize, rotation_seed: u64) -> Result<Self, TurboQuantError> {
        if dim < 2 || dim % 2 != 0 {
            return Err(TurboQuantError::InvalidDimension(dim));
        }
        Ok(Self {
            dim,
            rotation: Rotation::new(dim, rotation_seed),
        })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Stored bytes per vector: `D/2` nibbles + 8 bytes of constants.
    #[inline]
    pub fn code_len(&self) -> usize {
        self.dim / 2 + META_BYTES
    }

    /// Query blob length: `D` int8 codes + 8 bytes of constants.
    #[inline]
    pub fn query_len(&self) -> usize {
        self.dim + META_BYTES
    }

    /// Encode a vector into its Turbo4 code blob.
    pub fn encode(&self, v: &[f32]) -> Result<Vec<u8>, TurboQuantError> {
        if v.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                actual: v.len(),
            });
        }
        let rotated = self.rotation.apply(v);
        let norm_sq: f32 = rotated.iter().map(|x| x * x).sum();
        let alpha = (norm_sq / self.dim as f32).sqrt();
        Ok(self.encode_rotated(&rotated, alpha))
    }

    /// Encode both planes from one rotation pass: the Turbo4 code and the
    /// 1-bit candidate-generation code (ADR-297 phase C). The bits blob
    /// shares this codec's rotation, so a single query prep serves both.
    pub fn encode_dual(&self, v: &[f32]) -> Result<(Vec<u8>, Vec<u8>), TurboQuantError> {
        if v.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                actual: v.len(),
            });
        }
        let rotated = self.rotation.apply(v);
        let norm_sq: f32 = rotated.iter().map(|x| x * x).sum();
        let alpha = (norm_sq / self.dim as f32).sqrt();
        let bits = crate::bits1::encode_bits(&rotated, alpha);
        let turbo4 = self.encode_rotated(&rotated, alpha);
        Ok((turbo4, bits))
    }

    /// Pack an already-rotated vector (with its standardization factor) into
    /// the Turbo4 blob — the shared tail of `encode` / `encode_dual`.
    fn encode_rotated(&self, rotated: &[f32], alpha: f32) -> Vec<u8> {
        let inv = if alpha > 0.0 { 1.0 / alpha } else { 0.0 };
        let half = self.dim / 2;
        let mut blob = vec![0u8; self.code_len()];
        let mut s = 0.0f32;
        for i in 0..half {
            let c_lo = quantize_coord(rotated[i] * inv);
            let c_hi = quantize_coord(rotated[i + half] * inv);
            s += level(c_lo) * level(c_lo) + level(c_hi) * level(c_hi);
            blob[i] = c_lo | (c_hi << 4);
        }
        blob[half..half + 4].copy_from_slice(&alpha.to_le_bytes());
        blob[half + 4..half + 8].copy_from_slice(&s.to_le_bytes());
        blob
    }

    /// Prepare a query for traversal + rescoring.
    pub fn encode_query(&self, q: &[f32]) -> Result<Turbo4Query, TurboQuantError> {
        if q.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                actual: q.len(),
            });
        }
        let rotated = self.rotation.apply(q);
        let norm_sq: f32 = rotated.iter().map(|x| x * x).sum();
        let qmax = rotated.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let qscale = if qmax > 0.0 { qmax / 127.0 } else { 0.0 };
        let inv = if qscale > 0.0 { 1.0 / qscale } else { 0.0 };

        let mut blob = vec![0u8; self.query_len()];
        for (i, &x) in rotated.iter().enumerate() {
            blob[i] = ((x * inv).round() as i8) as u8;
        }
        blob[self.dim..self.dim + 4].copy_from_slice(&qscale.to_le_bytes());
        blob[self.dim + 4..self.dim + 8].copy_from_slice(&norm_sq.to_le_bytes());
        Ok(Turbo4Query {
            blob,
            rotated,
            norm_sq,
        })
    }

    /// Reconstruct the *rotated-space* approximation from a code blob
    /// (tests/debugging only — the search path never reconstructs).
    pub fn decode_rotated(&self, blob: &[u8]) -> Vec<f32> {
        let (nibbles, alpha, _) = split_code(blob, self.dim);
        let half = self.dim / 2;
        let mut out = vec![0.0f32; self.dim];
        for i in 0..half {
            out[i] = LEVELS_F32[(nibbles[i] & 0x0F) as usize] * alpha;
            out[i + half] = LEVELS_F32[(nibbles[i] >> 4) as usize] * alpha;
        }
        out
    }

    /// Reconstruct the original-space approximation (inverse rotation applied).
    pub fn decode(&self, blob: &[u8]) -> Vec<f32> {
        self.rotation.apply_inverse(&self.decode_rotated(blob))
    }
}

/// Split a code blob into (packed nibbles, α, S). `blob.len()` must equal
/// `dim/2 + META_BYTES`.
#[inline]
pub fn split_code(blob: &[u8], dim: usize) -> (&[u8], f32, f32) {
    let half = dim / 2;
    assert_eq!(dim % 2, 0, "Turbo4 dimensions must be even");
    assert_eq!(blob.len(), half + META_BYTES, "invalid Turbo4 code length");
    let alpha = f32::from_le_bytes(blob[half..half + 4].try_into().unwrap());
    let s = f32::from_le_bytes(blob[half + 4..half + 8].try_into().unwrap());
    (&blob[..half], alpha, s)
}

/// Split a query blob into (int8 codes, qscale, ‖q‖²).
#[inline]
pub fn split_query(blob: &[u8], dim: usize) -> (&[u8], f32, f32) {
    assert_eq!(dim % 2, 0, "Turbo4 dimensions must be even");
    assert_eq!(blob.len(), dim + META_BYTES, "invalid Turbo4 query length");
    let qscale = f32::from_le_bytes(blob[dim..dim + 4].try_into().unwrap());
    let norm_sq = f32::from_le_bytes(blob[dim + 4..dim + 8].try_into().unwrap());
    (&blob[..dim], qscale, norm_sq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotation::SplitMix64;

    fn gauss_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64(seed);
        let mut out = Vec::with_capacity(dim);
        while out.len() < dim {
            let u1 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let r = (-2.0 * u1.max(1e-12).ln()).sqrt();
            let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
            out.push((r * c) as f32);
            if out.len() < dim {
                out.push((r * s) as f32);
            }
        }
        out
    }

    #[test]
    fn code_len_is_8x_compression() {
        let codec = Turbo4Codec::new(1536, 42).unwrap();
        assert_eq!(codec.code_len(), 768 + 8);
        // 6144 f32 bytes / 776 = 7.92x
        assert!(1536.0 * 4.0 / codec.code_len() as f32 > 7.5);
    }

    #[test]
    fn rejects_odd_or_tiny_dims() {
        assert!(Turbo4Codec::new(3, 42).is_err());
        assert!(Turbo4Codec::new(0, 42).is_err());
        assert!(Turbo4Codec::new(128, 42).is_ok());
    }

    #[test]
    fn roundtrip_error_is_bounded() {
        let dim = 256;
        let codec = Turbo4Codec::new(dim, 42).unwrap();
        let v = gauss_vec(dim, 3);
        let blob = codec.encode(&v).unwrap();
        let back = codec.decode(&blob);
        // Lloyd-Max 4-bit on N(0,1) has ~0.009 MSE per unit variance;
        // allow generous slack for rotation Gaussianization error.
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        let err_sq: f32 = v.iter().zip(&back).map(|(a, b)| (a - b) * (a - b)).sum();
        assert!(
            err_sq / norm_sq < 0.05,
            "relative sq error {}",
            err_sq / norm_sq
        );
    }

    #[test]
    fn zero_vector_is_safe() {
        let codec = Turbo4Codec::new(64, 42).unwrap();
        let blob = codec.encode(&vec![0.0; 64]).unwrap();
        let (_, alpha, _) = split_code(&blob, 64);
        assert_eq!(alpha, 0.0);
        assert!(codec.decode(&blob).iter().all(|&x| x == 0.0));
        let q = codec.encode_query(&vec![0.0; 64]).unwrap();
        assert_eq!(q.norm_sq, 0.0);
    }

    #[test]
    fn blob_lengths_are_disjoint() {
        for dim in [2usize, 64, 384, 1536] {
            let codec = Turbo4Codec::new(dim, 1).unwrap();
            assert_ne!(codec.code_len(), codec.query_len());
        }
    }

    #[test]
    fn encode_dual_matches_single_encoders() {
        let dim = 128;
        let codec = Turbo4Codec::new(dim, 42).unwrap();
        let v = gauss_vec(dim, 17);
        let (t4, bits) = codec.encode_dual(&v).unwrap();
        assert_eq!(t4, codec.encode(&v).unwrap());
        assert_eq!(bits.len(), crate::bits1::code1_len(dim));
    }

    #[test]
    fn encoding_is_deterministic() {
        let dim = 384;
        let v = gauss_vec(dim, 9);
        let c1 = Turbo4Codec::new(dim, 42).unwrap();
        let c2 = Turbo4Codec::new(dim, 42).unwrap();
        assert_eq!(c1.encode(&v).unwrap(), c2.encode(&v).unwrap());
    }
}
