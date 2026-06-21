//! RaBitQ-style binary quantization (Gao & Long, SIGMOD 2024).
//!
//! Improves on naive sign-bit binary quantization with:
//!
//! 1. **Centroid centering** — vectors are encoded relative to a single
//!    global centroid, so codes capture the residual geometry instead of
//!    the (often uninformative) absolute signs.
//! 2. **Deterministic pseudo-random rotation** — a seeded randomized
//!    Hadamard transform (sign flips + fast Walsh–Hadamard, repeated for
//!    [`DEFAULT_ROUNDS`] rounds, dims padded to a power of two) spreads
//!    energy uniformly across dimensions so 1 bit per dimension carries
//!    near-optimal information. The rotation is orthonormal and fully
//!    reproducible from the stored seed.
//! 3. **Per-vector correction scalars** — the residual norm and the
//!    dot-correction `<o_unit, s_unit>` from the RaBitQ estimator, which
//!    turn Hamming-style codes into an unbiased inner-product estimator.
//! 4. **Asymmetric distance estimation** — full-precision queries are
//!    compared against binary codes via the correction scalars, giving a
//!    far better candidate ranking than symmetric Hamming distance.
//!
//! Intended use is a two-stage search: scan codes with the estimator to
//! collect `oversample * k` candidates, then rescore those candidates with
//! exact f32 distances.

use alloc::vec;
use alloc::vec::Vec;

use crate::tier::TemperatureTier;
use crate::traits::Quantizer;

/// Default number of randomized-Hadamard rounds. Two rounds already mix
/// well; three gives near-Gaussian rotations at negligible cost.
pub const DEFAULT_ROUNDS: u8 = 3;

/// Number of bytes of per-vector correction scalars (norm + dot_corr).
pub const CORRECTION_BYTES: usize = 8;

/// RaBitQ quantizer parameters (shared across all encoded vectors).
#[derive(Clone, Debug)]
pub struct RabitqQuantizer {
    /// Original vector dimensionality.
    pub dim: usize,
    /// Dimensionality after padding to the next power of two.
    pub padded_dim: usize,
    /// Seed for the deterministic pseudo-random rotation.
    pub seed: u64,
    /// Number of randomized-Hadamard rounds.
    pub rounds: u8,
    /// Global centroid (length `dim`) subtracted before rotation.
    pub centroid: Vec<f32>,
}

/// A single encoded vector: 1-bit sign code plus correction scalars.
#[derive(Clone, Debug, PartialEq)]
pub struct RabitqCode {
    /// Sign bits of the rotated centered vector (`padded_dim` bits,
    /// dimension `d` maps to bit `d % 8` of byte `d / 8`).
    pub bits: Vec<u8>,
    /// Residual norm `||v - centroid||`.
    pub norm: f32,
    /// Dot correction `<o_unit, s_unit>` where `o_unit` is the unit
    /// rotated residual and `s_unit = signs / sqrt(padded_dim)`.
    pub dot_corr: f32,
}

impl RabitqCode {
    /// Total stored bytes for this code (bits + correction scalars).
    #[inline]
    pub fn stored_bytes(&self) -> usize {
        self.bits.len() + CORRECTION_BYTES
    }
}

/// A query prepared for asymmetric distance estimation (computed once
/// per query, reused across all codes).
#[derive(Clone, Debug)]
pub struct RabitqQuery {
    /// Rotated centered query, length `padded_dim`.
    pub rotated: Vec<f32>,
    /// Squared residual norm `||q - centroid||^2`.
    pub norm_sq: f32,
}

/// SplitMix64 mixer (same constants as the runtime's deterministic
/// leveling) — used to derive reproducible rotation sign flips.
#[inline]
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Smallest power of two `>= n` (and `>= 1`).
#[inline]
fn next_pow2(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

/// In-place unnormalized fast Walsh–Hadamard transform.
/// `v.len()` must be a power of two.
fn fwht(v: &mut [f32]) {
    let n = v.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = v[j];
                let y = v[j + h];
                v[j] = x + y;
                v[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

impl RabitqQuantizer {
    /// Train a RaBitQ quantizer over the given vectors: the centroid is
    /// the per-dimension mean. The rotation is derived from `seed`.
    ///
    /// # Panics
    ///
    /// Panics if `vectors` is empty or dimensionality is inconsistent.
    pub fn train(vectors: &[&[f32]], seed: u64) -> Self {
        assert!(!vectors.is_empty(), "need at least one training vector");
        let dim = vectors[0].len();
        assert!(dim > 0, "vector dimensionality must be > 0");

        let mut centroid = vec![0.0f64; dim];
        for v in vectors {
            assert_eq!(v.len(), dim, "dimension mismatch in training data");
            for (acc, &x) in centroid.iter_mut().zip(v.iter()) {
                *acc += x as f64;
            }
        }
        let inv_n = 1.0 / vectors.len() as f64;
        let centroid: Vec<f32> = centroid.iter().map(|&s| (s * inv_n) as f32).collect();

        Self::with_centroid(dim, centroid, seed, DEFAULT_ROUNDS)
    }

    /// Construct from explicit parameters (used by the QUANT_SEG codec).
    pub fn with_centroid(dim: usize, centroid: Vec<f32>, seed: u64, rounds: u8) -> Self {
        assert_eq!(centroid.len(), dim, "centroid length must equal dim");
        Self {
            dim,
            padded_dim: next_pow2(dim),
            seed,
            rounds: rounds.max(1),
            centroid,
        }
    }

    /// Deterministic sign flip for dimension `i` of rotation round `round`:
    /// returns `true` for negate.
    #[inline]
    fn sign_flip(&self, round: u8, i: usize) -> bool {
        // One SplitMix64 word covers 64 dimensions; counter-based so any
        // (round, word) is independently addressable and reproducible.
        let word = splitmix64(
            self.seed
                ^ (round as u64).wrapping_mul(0xA076_1D64_78BD_642F)
                ^ ((i as u64) / 64).wrapping_mul(0xE703_7ED1_A0B4_28DB),
        );
        (word >> (i % 64)) & 1 == 1
    }

    /// Apply the seeded orthonormal rotation: pad to `padded_dim`, then
    /// `rounds` of (sign flips, normalized Walsh–Hadamard).
    pub fn rotate(&self, v: &[f32]) -> Vec<f32> {
        debug_assert!(v.len() <= self.padded_dim);
        let mut buf = vec![0.0f32; self.padded_dim];
        buf[..v.len()].copy_from_slice(v);
        let scale = 1.0 / (self.padded_dim as f32).sqrt();
        for round in 0..self.rounds {
            for (i, x) in buf.iter_mut().enumerate() {
                if self.sign_flip(round, i) {
                    *x = -*x;
                }
            }
            fwht(&mut buf);
            for x in buf.iter_mut() {
                *x *= scale;
            }
        }
        buf
    }

    /// Inverse of [`Self::rotate`] (rounds in reverse: Hadamard, then
    /// sign flips — both are their own inverses).
    pub fn rotate_inverse(&self, v: &[f32]) -> Vec<f32> {
        debug_assert_eq!(v.len(), self.padded_dim);
        let mut buf = v.to_vec();
        let scale = 1.0 / (self.padded_dim as f32).sqrt();
        for round in (0..self.rounds).rev() {
            fwht(&mut buf);
            for x in buf.iter_mut() {
                *x *= scale;
            }
            for (i, x) in buf.iter_mut().enumerate() {
                if self.sign_flip(round, i) {
                    *x = -*x;
                }
            }
        }
        buf
    }

    /// Encode a vector: center, rotate, take sign bits, and compute the
    /// RaBitQ correction scalars.
    pub fn encode_code(&self, vector: &[f32]) -> RabitqCode {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");
        let centered: Vec<f32> = vector
            .iter()
            .zip(self.centroid.iter())
            .map(|(&x, &c)| x - c)
            .collect();
        let rotated = self.rotate(&centered);

        let mut norm_sq = 0.0f32;
        let mut abs_sum = 0.0f32;
        let mut bits = vec![0u8; self.padded_dim.div_ceil(8)];
        for (d, &x) in rotated.iter().enumerate() {
            norm_sq += x * x;
            abs_sum += x.abs();
            if x >= 0.0 {
                bits[d / 8] |= 1 << (d % 8);
            }
        }
        let norm = norm_sq.sqrt();
        // <o_unit, s_unit> = sum |r_i| / (||r|| * sqrt(D)). For a zero
        // residual (vector == centroid) the estimator multiplies by
        // norm = 0 anyway, so any positive placeholder is fine.
        let dot_corr = if norm > f32::EPSILON {
            (abs_sum / (norm * (self.padded_dim as f32).sqrt())).max(f32::EPSILON)
        } else {
            1.0
        };
        RabitqCode {
            bits,
            norm,
            dot_corr,
        }
    }

    /// Prepare a query for repeated asymmetric distance estimation.
    pub fn prepare_query(&self, query: &[f32]) -> RabitqQuery {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");
        let centered: Vec<f32> = query
            .iter()
            .zip(self.centroid.iter())
            .map(|(&x, &c)| x - c)
            .collect();
        let rotated = self.rotate(&centered);
        let norm_sq = rotated.iter().map(|&x| x * x).sum();
        RabitqQuery { rotated, norm_sq }
    }

    /// Estimate the squared L2 distance between the (full-precision)
    /// prepared query and an encoded vector.
    ///
    /// Uses the RaBitQ estimator: `<o_unit, x> ~ <s_unit, x> / <s_unit,
    /// o_unit>`, so `<v-c, q-c> ~ norm * (<s, rq> / sqrt(D)) / dot_corr`,
    /// and `||v-q||^2 = ||v-c||^2 + ||q-c||^2 - 2<v-c, q-c>` (rotation
    /// preserves norms and inner products).
    pub fn estimate_l2_sq(&self, query: &RabitqQuery, code: &RabitqCode) -> f32 {
        let mut signed_sum = 0.0f32;
        for (d, &x) in query.rotated.iter().enumerate() {
            if (code.bits[d / 8] >> (d % 8)) & 1 == 1 {
                signed_sum += x;
            } else {
                signed_sum -= x;
            }
        }
        let est_ip = code.norm * (signed_sum / (self.padded_dim as f32).sqrt()) / code.dot_corr;
        code.norm * code.norm + query.norm_sq - 2.0 * est_ip
    }

    /// Bytes stored per encoded vector (sign bits + correction scalars).
    #[inline]
    pub fn stored_bytes_per_vector(&self) -> usize {
        self.padded_dim.div_ceil(8) + CORRECTION_BYTES
    }

    /// Compression ratio versus raw f32 storage of the original vector.
    #[inline]
    pub fn compression_ratio(&self) -> f32 {
        (self.dim * 4) as f32 / self.stored_bytes_per_vector() as f32
    }

    /// Serialize a code to bytes: `[bits][norm: f32 LE][dot_corr: f32 LE]`.
    pub fn code_to_bytes(&self, code: &RabitqCode) -> Vec<u8> {
        let mut out = Vec::with_capacity(code.stored_bytes());
        out.extend_from_slice(&code.bits);
        out.extend_from_slice(&code.norm.to_le_bytes());
        out.extend_from_slice(&code.dot_corr.to_le_bytes());
        out
    }

    /// Deserialize a code produced by [`Self::code_to_bytes`].
    /// Returns `None` if `data` is too short (panic-free on bad input).
    pub fn code_from_bytes(&self, data: &[u8]) -> Option<RabitqCode> {
        let nbits = self.padded_dim.div_ceil(8);
        if data.len() < nbits + CORRECTION_BYTES {
            return None;
        }
        let bits = data[..nbits].to_vec();
        let norm = f32::from_le_bytes(data[nbits..nbits + 4].try_into().ok()?);
        let dot_corr = f32::from_le_bytes(data[nbits + 4..nbits + 8].try_into().ok()?);
        Some(RabitqCode {
            bits,
            norm,
            dot_corr,
        })
    }
}

impl Quantizer for RabitqQuantizer {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        self.code_to_bytes(&self.encode_code(vector))
    }

    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let code = match self.code_from_bytes(codes) {
            Some(c) => c,
            None => return vec![0.0; self.dim],
        };
        // Best rank-1 reconstruction: project onto the code direction,
        // r_hat = norm * dot_corr * s / sqrt(D), then invert the rotation
        // and re-add the centroid.
        let scale = code.norm * code.dot_corr / (self.padded_dim as f32).sqrt();
        let mut rotated = Vec::with_capacity(self.padded_dim);
        for d in 0..self.padded_dim {
            let sign = if (code.bits[d / 8] >> (d % 8)) & 1 == 1 {
                1.0
            } else {
                -1.0
            };
            rotated.push(sign * scale);
        }
        let residual = self.rotate_inverse(&rotated);
        residual
            .iter()
            .take(self.dim)
            .zip(self.centroid.iter())
            .map(|(&r, &c)| r + c)
            .collect()
    }

    fn tier(&self) -> TemperatureTier {
        TemperatureTier::Cold
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..dim)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) as f32) / (u32::MAX as f32) - 0.5
            })
            .collect()
    }

    fn make_quantizer(dim: usize, n: usize) -> (RabitqQuantizer, Vec<Vec<f32>>) {
        let data: Vec<Vec<f32>> = (0..n).map(|i| lcg_vector(dim, i as u64)).collect();
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        (RabitqQuantizer::train(&refs, 0xDEAD_BEEF), data)
    }

    fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    #[test]
    fn rotation_is_orthonormal_and_deterministic() {
        let (rq, data) = make_quantizer(100, 8); // non-pow2 dim -> padded 128
        assert_eq!(rq.padded_dim, 128);
        for v in &data {
            let r1 = rq.rotate(v);
            let r2 = rq.rotate(v);
            assert_eq!(r1, r2, "rotation must be deterministic");

            let norm_in: f32 = v.iter().map(|x| x * x).sum();
            let norm_out: f32 = r1.iter().map(|x| x * x).sum();
            assert!(
                (norm_in - norm_out).abs() < 1e-3 * norm_in.max(1.0),
                "rotation must preserve norms: {norm_in} vs {norm_out}"
            );

            // Inverse round-trips back to the padded input.
            let back = rq.rotate_inverse(&r1);
            for (d, (&orig, &rec)) in v.iter().zip(back.iter()).enumerate() {
                assert!(
                    (orig - rec).abs() < 1e-4,
                    "dim {d}: {orig} != {rec} after inverse rotation"
                );
            }
            for &pad in &back[v.len()..] {
                assert!(pad.abs() < 1e-4, "padding must invert to ~0");
            }
        }
    }

    #[test]
    fn rotation_preserves_inner_products() {
        let (rq, data) = make_quantizer(64, 4);
        let a = &data[0];
        let b = &data[1];
        let ip: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let ra = rq.rotate(a);
        let rb = rq.rotate(b);
        let rip: f32 = ra.iter().zip(rb.iter()).map(|(x, y)| x * y).sum();
        assert!((ip - rip).abs() < 1e-3, "ip {ip} vs rotated ip {rip}");
    }

    #[test]
    fn different_seeds_give_different_rotations() {
        let v = lcg_vector(32, 7);
        let a = RabitqQuantizer::with_centroid(32, vec![0.0; 32], 1, DEFAULT_ROUNDS);
        let b = RabitqQuantizer::with_centroid(32, vec![0.0; 32], 2, DEFAULT_ROUNDS);
        assert_ne!(a.rotate(&v), b.rotate(&v));
    }

    #[test]
    fn code_round_trip_bytes() {
        let (rq, data) = make_quantizer(48, 16);
        for v in &data {
            let code = rq.encode_code(v);
            let bytes = rq.code_to_bytes(&code);
            assert_eq!(bytes.len(), rq.stored_bytes_per_vector());
            let back = rq.code_from_bytes(&bytes).expect("decode");
            assert_eq!(back, code);
        }
        // Truncated input must be rejected, not panic.
        let code = rq.encode_code(&data[0]);
        let bytes = rq.code_to_bytes(&code);
        assert!(rq.code_from_bytes(&bytes[..bytes.len() - 1]).is_none());
        assert!(rq.code_from_bytes(&[]).is_none());
    }

    #[test]
    fn decode_reconstruction_beats_naive_sign_bits() {
        // The corrected reconstruction must be closer to the original than
        // the naive +-1 sign decode is (sanity that corrections help).
        let (rq, data) = make_quantizer(128, 64);
        let mut rabitq_err = 0.0f64;
        let mut naive_err = 0.0f64;
        for v in &data {
            let rec = rq.decode(&rq.encode(v));
            rabitq_err += l2_sq(v, &rec) as f64;

            let bits = crate::binary::encode_binary(v);
            let nrec = crate::binary::decode_binary(&bits, v.len());
            naive_err += l2_sq(v, &nrec) as f64;
        }
        assert!(
            rabitq_err < naive_err,
            "RaBitQ reconstruction error {rabitq_err} must beat naive {naive_err}"
        );
    }

    #[test]
    fn estimator_correlates_with_true_distances() {
        // Pearson correlation between estimated and true squared L2 over
        // many (query, vector) pairs must be strong.
        let dim = 128;
        let (rq, data) = make_quantizer(dim, 200);
        let codes: Vec<RabitqCode> = data.iter().map(|v| rq.encode_code(v)).collect();

        let mut est = Vec::new();
        let mut truth = Vec::new();
        for qi in 0..20u64 {
            let q = lcg_vector(dim, 5_000 + qi);
            let prepared = rq.prepare_query(&q);
            for (v, code) in data.iter().zip(codes.iter()) {
                est.push(rq.estimate_l2_sq(&prepared, code) as f64);
                truth.push(l2_sq(&q, v) as f64);
            }
        }

        let n = est.len() as f64;
        let me = est.iter().sum::<f64>() / n;
        let mt = truth.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut ve = 0.0;
        let mut vt = 0.0;
        for (&e, &t) in est.iter().zip(truth.iter()) {
            cov += (e - me) * (t - mt);
            ve += (e - me) * (e - me);
            vt += (t - mt) * (t - mt);
        }
        let corr = cov / (ve.sqrt() * vt.sqrt());
        #[cfg(feature = "std")]
        std::eprintln!("estimator/true distance correlation (128d): {corr:.4}");
        assert!(
            corr > 0.8,
            "estimator correlation {corr:.3} too weak (expected > 0.8)"
        );

        // The estimator must also be roughly unbiased: mean relative error
        // of estimated vs true distance stays small.
        let mean_rel: f64 = est
            .iter()
            .zip(truth.iter())
            .map(|(&e, &t)| ((e - t) / t.max(1e-9)).abs())
            .sum::<f64>()
            / n;
        #[cfg(feature = "std")]
        std::eprintln!("estimator mean relative distance error (128d): {mean_rel:.4}");
        assert!(
            mean_rel < 0.25,
            "mean relative error {mean_rel:.3} too large"
        );
    }

    #[test]
    fn compression_ratio_targets() {
        // Code-only payload is exactly 32x; with the 8 correction bytes the
        // total is ~21x at 128 dims and approaches 32x as dims grow.
        let rq128 = RabitqQuantizer::with_centroid(128, vec![0.0; 128], 1, DEFAULT_ROUNDS);
        assert_eq!(rq128.padded_dim, 128);
        assert_eq!((rq128.dim * 4) / (rq128.padded_dim / 8), 32);
        assert!(rq128.compression_ratio() >= 20.0);

        let rq1024 = RabitqQuantizer::with_centroid(1024, vec![0.0; 1024], 1, DEFAULT_ROUNDS);
        assert!(rq1024.compression_ratio() >= 30.0);
    }

    #[test]
    fn zero_residual_vector_is_safe() {
        let (rq, _) = make_quantizer(16, 4);
        let code = rq.encode_code(&rq.centroid.clone());
        assert!(code.norm <= 1e-6);
        let q = rq.prepare_query(&lcg_vector(16, 99));
        let est = rq.estimate_l2_sq(&q, &code);
        assert!(est.is_finite());
    }
}
