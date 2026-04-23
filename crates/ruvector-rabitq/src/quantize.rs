//! Bit-packing and XNOR-popcount distance kernel.
//!
//! Each dimension is encoded as a single bit: 1 if the rotated value ≥ 0, else 0.
//! Bits are packed MSB-first into u64 words. Distance estimation uses XNOR-popcount
//! followed by the angular correction formula (see `BinaryCode::estimated_sq_distance`).

/// A packed binary code representing one vector (D bits).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BinaryCode {
    /// Packed u64 words (ceil(D/64) words).
    pub words: Vec<u64>,
    /// Original L2 norm before normalisation (needed for the IP estimator).
    pub norm: f32,
    /// Number of dimensions.
    pub dim: usize,
}

impl BinaryCode {
    /// Encode a (possibly rotated) vector into a binary code.
    ///
    /// `norm` should be the L2 norm of the *pre-rotation* vector so the estimator
    /// can rescale correctly.
    pub fn encode(rotated: &[f32], norm: f32) -> Self {
        let dim = rotated.len();
        let n_words = (dim + 63) / 64;
        let mut words = vec![0u64; n_words];
        for (i, &v) in rotated.iter().enumerate() {
            if v >= 0.0 {
                words[i / 64] |= 1u64 << (63 - (i % 64));
            }
        }
        Self { words, norm, dim }
    }

    /// XNOR-popcount agreement: number of matching bits between self and other.
    #[inline]
    pub fn xnor_popcount(&self, other: &Self) -> u32 {
        debug_assert_eq!(self.words.len(), other.words.len());
        self.words
            .iter()
            .zip(other.words.iter())
            .map(|(&a, &b)| (!(a ^ b)).count_ones())
            .sum()
    }

    /// Angular inner-product estimate (RaBitQ SIGMOD 2024).
    ///
    /// For normalized database vector x (original norm stored as `self.norm`) and
    /// normalized query q (original norm stored as `query_code.norm`):
    ///
    ///   E[B/D] = 1 − θ/π   where θ = arccos(<x̂, q̂>)
    ///   ⟹  est cos(θ) = cos(π · (1 − B/D))
    ///   ⟹  est <q, x> = ||q|| · ||x|| · cos(π · (1 − B/D))
    ///
    /// Returns estimated squared L2 via: ||q − x||² = ||q||² + ||x||² − 2<q, x>.
    ///
    /// This is the exact angular distance formula, not the small-angle approximation.
    #[inline]
    pub fn estimated_sq_distance(&self, query_code: &Self) -> f32 {
        use std::f32::consts::PI;
        let d = self.dim as f32;
        let agreement = self.xnor_popcount(query_code) as f32;
        // Angular estimator: cos(π·(1 − B/D)) gives correct IP for all angles.
        let est_cos = (PI * (1.0 - agreement / d)).cos();
        let est_ip = self.norm * query_code.norm * est_cos;
        let q_sq = query_code.norm * query_code.norm;
        q_sq + self.norm * self.norm - 2.0 * est_ip
    }
}

/// Pack bits from a boolean slice into u64 words (for testing/utilities).
pub fn pack_bits(bits: &[bool]) -> Vec<u64> {
    let n_words = (bits.len() + 63) / 64;
    let mut words = vec![0u64; n_words];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            words[i / 64] |= 1u64 << (63 - (i % 64));
        }
    }
    words
}

/// Unpack u64 words back into a bool slice of length `dim`.
pub fn unpack_bits(words: &[u64], dim: usize) -> Vec<bool> {
    (0..dim)
        .map(|i| words[i / 64] & (1u64 << (63 - (i % 64))) != 0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let bits: Vec<bool> = (0..130).map(|i| i % 3 == 0).collect();
        let words = pack_bits(&bits);
        let unpacked = unpack_bits(&words, 130);
        assert_eq!(bits, unpacked);
    }

    #[test]
    fn xnor_self_is_all_ones() {
        let v: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let code = BinaryCode::encode(&v, 1.0);
        let agreement = code.xnor_popcount(&code);
        assert_eq!(agreement, 64, "self-agreement should be D=64, got {agreement}");
    }

    #[test]
    fn xnor_opposite_is_zero() {
        let v: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let neg_v: Vec<f32> = v.iter().map(|&x| -x).collect();
        let code = BinaryCode::encode(&v, 1.0);
        let code_neg = BinaryCode::encode(&neg_v, 1.0);
        let agreement = code.xnor_popcount(&code_neg);
        assert_eq!(agreement, 0, "opposite vectors should have 0 agreement");
    }

    #[test]
    fn estimated_distance_self_is_near_zero() {
        // A unit vector against itself should estimate distance ≈ 0.
        let v: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0).sin()).collect();
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let unit: Vec<f32> = v.iter().map(|&x| x / norm).collect();
        let code = BinaryCode::encode(&unit, 1.0);
        let est = code.estimated_sq_distance(&code);
        // At D=128 the estimator has ~10% error; self-distance should still be small.
        assert!(est < 0.3, "self sq-distance estimate too large: {est}");
    }
}
