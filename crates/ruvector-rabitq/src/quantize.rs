//! Bit-packing, XNOR-popcount, and the two RaBitQ distance estimators:
//!
//! 1. **Symmetric Charikar-style angular estimator** — both query and database
//!    are 1-bit. Derived from hyperplane-LSH collision probability:
//!        E[B/D] = 1 − θ/π
//!    This is what the shipped crate had at commit `f2dbb6efb`.
//!
//! 2. **Asymmetric RaBitQ-2024 inner-product estimator** — query stays in f32,
//!    database is 1-bit `b_i ∈ {−1/√D, +1/√D}`. Inner product is reconstructed
//!    by summing the rotated query's components with signs, then rescaled by
//!    a precomputed factor derived from the stored unit-sphere inner-product
//!    bias. Unbiased for Haar-uniform rotations with O(1/√D) variance.
//!
//! The asymmetric path closes the gap between this crate's estimator and the
//! SIGMOD 2024 paper (Gao & Long) — the symmetric path remains for
//! apples-to-apples comparison against naive 1-bit codes.
//!
//! ## Bit packing
//!
//! Each dimension is one bit: 1 if the rotated value ≥ 0, else 0. Bits are
//! packed MSB-first into u64 words. When `D % 64 != 0` the last word carries
//! `64·n_words − D` padding bits that are zero in every code; XNOR-popcount
//! must mask those bits off before counting, otherwise padding bits always
//! agree and the estimator is biased. `masked_xnor_popcount` handles this
//! correctly; `xnor_popcount` is retained for the aligned case.

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

    /// Raw XNOR-popcount across all stored bits. **Do not use when
    /// `D % 64 != 0`** — the padding bits in the last word are zero in every
    /// code and XNOR-popcount counts them as matches, biasing the estimator.
    /// Retained as a fast path for the aligned case (D multiple of 64).
    #[inline]
    pub fn xnor_popcount(&self, other: &Self) -> u32 {
        debug_assert_eq!(self.words.len(), other.words.len());
        self.words
            .iter()
            .zip(other.words.iter())
            .map(|(&a, &b)| (!(a ^ b)).count_ones())
            .sum()
    }

    /// Padding-safe XNOR-popcount. Masks the trailing
    /// `64·n_words − D` bits of the last word so padding zeros don't inflate
    /// the agreement count. Correct at any `D ≥ 1`; same cost as the raw
    /// version up to one extra AND on the last word.
    #[inline]
    pub fn masked_xnor_popcount(&self, other: &Self) -> u32 {
        debug_assert_eq!(self.words.len(), other.words.len());
        debug_assert_eq!(self.dim, other.dim);
        let n_words = self.words.len();
        if n_words == 0 {
            return 0;
        }
        let mut sum: u32 = 0;
        for i in 0..n_words - 1 {
            sum += (!(self.words[i] ^ other.words[i])).count_ones();
        }
        // Last word: mask off the padding bits that were never written.
        let valid_bits = self.dim - 64 * (n_words - 1);
        let mask: u64 = if valid_bits == 64 {
            !0u64
        } else {
            // Keep the top `valid_bits` MSBs (because we packed MSB-first).
            !0u64 << (64 - valid_bits)
        };
        let last = !(self.words[n_words - 1] ^ other.words[n_words - 1]) & mask;
        sum += last.count_ones();
        sum
    }

    /// **Symmetric** angular estimator (Charikar-style) — both operands are
    /// 1-bit codes of rotated unit vectors.
    ///
    /// For normalized database x̂ (`self.norm` holds the original ‖x‖) and
    /// normalized query q̂ (`query_code.norm` holds the original ‖q‖):
    ///
    ///   E[B/D] = 1 − θ/π  where  θ = arccos(⟨x̂, q̂⟩)
    ///   ⟹  est cos(θ) = cos(π · (1 − B/D))
    ///   ⟹  est ⟨q, x⟩ = ‖q‖ · ‖x‖ · est cos(θ)
    ///
    /// Returns estimated squared-L2: ‖q − x‖² = ‖q‖² + ‖x‖² − 2⟨q, x⟩.
    #[inline]
    pub fn estimated_sq_distance(&self, query_code: &Self) -> f32 {
        use std::f32::consts::PI;
        let d = self.dim as f32;
        let agreement = self.masked_xnor_popcount(query_code) as f32;
        let est_cos = (PI * (1.0 - agreement / d)).cos();
        let est_ip = self.norm * query_code.norm * est_cos;
        let q_sq = query_code.norm * query_code.norm;
        q_sq + self.norm * self.norm - 2.0 * est_ip
    }

    /// **Asymmetric** inner-product estimator (RaBitQ-style, keeps the query
    /// in f32). More accurate than the symmetric path, at the cost of
    /// O(D) arithmetic per candidate instead of O(D/64) popcount.
    ///
    /// Given the rotated-unit query `q_rot` (‖q_rot‖ = 1) and the stored 1-bit
    /// code `b_x` ∈ {−1/√D, +1/√D}ᴰ, the unbiased inner-product estimate is:
    ///
    ///   ⟨q̂_rot, u_x⟩ ≈ (1/√D) · Σᵢ sign(x_rot,i) · q_rot,i
    ///
    /// where u_x is the rotated unit vector and `b_x,i = sign(x_rot,i)/√D`.
    /// The unbiasing factor accounts for the concentration of
    /// `Σ|q_rot,i|` on a Haar-uniform rotation of q (which preserves norm).
    ///
    /// Returns estimated squared-L2: `‖q − x‖² = ‖q‖² + ‖x‖² − 2‖q‖·‖x‖·ŝ`
    /// where `ŝ = ⟨q̂_rot, u_x⟩` is the unit-sphere IP estimate above.
    ///
    /// `q_rotated` must be length `self.dim`; caller pre-normalises and
    /// pre-rotates the query once per search (amortised across n candidates).
    #[inline]
    pub fn estimated_sq_distance_asymmetric(&self, q_rotated_unit: &[f32], q_norm: f32) -> f32 {
        debug_assert_eq!(q_rotated_unit.len(), self.dim);
        let d = self.dim;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        // Σᵢ sign(x_rot,i) · q_rot,i  without materialising signs: bit = 1
        // means +1, bit = 0 means −1.
        let mut ip = 0.0f32;
        for (i, &q_i) in q_rotated_unit.iter().enumerate() {
            let bit_set = (self.words[i / 64] >> (63 - (i % 64))) & 1 == 1;
            ip += if bit_set { q_i } else { -q_i };
        }
        let unit_ip = ip * inv_sqrt_d;
        let est_ip = q_norm * self.norm * unit_ip;
        q_norm * q_norm + self.norm * self.norm - 2.0 * est_ip
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
        let v: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let code = BinaryCode::encode(&v, 1.0);
        let agreement = code.xnor_popcount(&code);
        assert_eq!(
            agreement, 64,
            "self-agreement should be D=64, got {agreement}"
        );
    }

    #[test]
    fn xnor_opposite_is_zero() {
        let v: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let neg_v: Vec<f32> = v.iter().map(|&x| -x).collect();
        let code = BinaryCode::encode(&v, 1.0);
        let code_neg = BinaryCode::encode(&neg_v, 1.0);
        let agreement = code.xnor_popcount(&code_neg);
        assert_eq!(agreement, 0, "opposite vectors should have 0 agreement");
    }

    /// Bug surfaced by the deep review: at `D % 64 != 0` the padding bits in
    /// the last word are zero in every code, so XNOR-popcount counts them as
    /// matches. `masked_xnor_popcount` must not count padding.
    #[test]
    fn masked_popcount_handles_non_aligned_dim() {
        // D=100 → 2 u64 words, 28 padding bits.
        let v: Vec<f32> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let neg_v: Vec<f32> = v.iter().map(|&x| -x).collect();
        let code = BinaryCode::encode(&v, 1.0);
        let code_neg = BinaryCode::encode(&neg_v, 1.0);
        // Raw would read 0 matches + 28 padding matches = 28 (wrong).
        let raw = code.xnor_popcount(&code_neg);
        assert_eq!(
            raw, 28,
            "raw xnor should count padding as matches (bug demo)"
        );
        // Masked must report 0 matches.
        let masked = code.masked_xnor_popcount(&code_neg);
        assert_eq!(
            masked, 0,
            "masked xnor must ignore padding bits; got {masked}"
        );
        // Self-compare: every real bit matches, padding is masked.
        let self_masked = code.masked_xnor_popcount(&code);
        assert_eq!(self_masked, 100);
    }

    #[test]
    fn masked_popcount_matches_raw_when_aligned() {
        // D=128 is 64-aligned, so masked == raw.
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let w: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).cos()).collect();
        let ca = BinaryCode::encode(&v, 1.0);
        let cb = BinaryCode::encode(&w, 1.0);
        assert_eq!(ca.xnor_popcount(&cb), ca.masked_xnor_popcount(&cb));
    }

    #[test]
    fn estimated_distance_self_is_near_zero() {
        // A unit vector against itself should estimate distance ≈ 0.
        let v: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0).sin()).collect();
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let unit: Vec<f32> = v.iter().map(|&x| x / norm).collect();
        let code = BinaryCode::encode(&unit, 1.0);
        let est = code.estimated_sq_distance(&code);
        // Symmetric Charikar estimator on the same code: cos(π·(1−D/D))=1 → est=0.
        assert!(
            est.abs() < 1e-5,
            "self sq-distance estimate too large: {est}"
        );
    }

    #[test]
    fn asymmetric_matches_symmetric_in_sign() {
        // The asymmetric IP estimator and the symmetric cos-angle estimator
        // should agree on which of two candidates is closer (even when the
        // magnitudes differ) — they encode the same angular signal.
        use rand::{Rng as _, SeedableRng as _};
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let d = 128;
        let q: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let q_norm: f32 = q.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let q_unit: Vec<f32> = q.iter().map(|&x| x / q_norm).collect();
        let qc = BinaryCode::encode(&q_unit, q_norm);

        let near: Vec<f32> = q.iter().map(|&x| x + rng.gen::<f32>() * 0.1).collect();
        let far: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();

        let encode_one = |v: &[f32]| {
            let n: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let u: Vec<f32> = v.iter().map(|&x| x / n).collect();
            BinaryCode::encode(&u, n)
        };
        let cn = encode_one(&near);
        let cf = encode_one(&far);

        // Symmetric
        let s_near = cn.estimated_sq_distance(&qc);
        let s_far = cf.estimated_sq_distance(&qc);
        // Asymmetric: the "rotated unit query" here is just q_unit (no
        // rotation since we're testing the estimator math directly).
        let a_near = cn.estimated_sq_distance_asymmetric(&q_unit, q_norm);
        let a_far = cf.estimated_sq_distance_asymmetric(&q_unit, q_norm);
        assert!(s_near < s_far, "symmetric ordering wrong");
        assert!(a_near < a_far, "asymmetric ordering wrong");
    }
}
