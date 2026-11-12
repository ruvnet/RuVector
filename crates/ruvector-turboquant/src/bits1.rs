//! 1-bit sign codes over the Turbo4 rotation (ADR-297 phase C).
//!
//! The cheap candidate-generation plane: `ceil(D/8)` bytes per vector plus
//! two f32 constants, scored by pure AND+POPCNT. Codes are derived from the
//! **same** rotated, standardized coordinates as Turbo4 nibbles — one
//! rotation pass yields both representations, and both stay bit-stable
//! across platforms.
//!
//! ## Code blob (`ceil(D/8) + 8` bytes)
//!
//! ```text
//! [ sign bits (LSB-first in LE u64 words) | α: f32 | c: f32 ]
//! ```
//!
//! `α = ‖v‖/√D` (as Turbo4), `c = mean|zᵢ|` — the per-vector MSE-optimal
//! 1-bit reconstruction scale, so `v ≈ α·c·s`, `sᵢ ∈ {±1}` (RaBitQ-style).
//!
//! ## Query: bit-plane decomposition
//!
//! The int8 query is biased to u8 and split into 8 bit-planes. For plane
//! `p`: `Σᵢ planeₚᵢ·sᵢ = 2·pop(planeₚ ∧ bits) − pop(planeₚ)`, so
//!
//! ```text
//! Σ q_u8·s = Σₚ 2ᵖ·(2·pop(planeₚ ∧ bits) − popₚ)
//! Σ q_i8·s = Σ q_u8·s − 128·(2·pop(bits) − D)
//! dot(q,v) ≈ qscale · α · c · Σ q_i8·s
//! ```
//!
//! Per candidate that is 9 masked-popcount passes over `D/64` words — with
//! ~4× less memory traffic than the Turbo4 nibble kernel, which is the point:
//! at scale, traversal is bandwidth-bound.

use crate::codec::META_BYTES;
use crate::score::Metric;
use crate::TurboQuantError;

/// Bytes of sign bits for `dim` dimensions, padded to whole u64 words so the
/// popcount kernel needs no tail handling.
#[inline]
pub fn bits_len(dim: usize) -> usize {
    dim.div_ceil(64) * 8
}

/// Full 1-bit code blob length.
#[inline]
pub fn code1_len(dim: usize) -> usize {
    bits_len(dim) + META_BYTES
}

/// Encode the 1-bit blob from rotated coordinates and the standardization
/// factor `alpha` (both already computed by the Turbo4 encoder — pass
/// `rotated` and `alpha` straight through; `rotated` must be `dim` long).
pub fn encode_bits(rotated: &[f32], alpha: f32) -> Vec<u8> {
    let dim = rotated.len();
    let n_words = dim.div_ceil(64);
    let mut words = vec![0u64; n_words];
    let mut abs_sum = 0.0f32;
    let inv = if alpha > 0.0 { 1.0 / alpha } else { 0.0 };
    for (i, &r) in rotated.iter().enumerate() {
        let z = r * inv;
        abs_sum += z.abs();
        if z >= 0.0 {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    let c = abs_sum / dim as f32;
    let mut blob = Vec::with_capacity(code1_len(dim));
    for w in &words {
        blob.extend_from_slice(&w.to_le_bytes());
    }
    blob.extend_from_slice(&alpha.to_le_bytes());
    blob.extend_from_slice(&c.to_le_bytes());
    blob
}

/// A query prepared for popcount scoring against 1-bit codes.
pub struct Bits1Query {
    /// 8 bit-planes of the biased (u8) query, each `bits_len(dim)` bytes as
    /// LE u64 words, plane 0 = LSB.
    planes: [Vec<u64>; 8],
    /// Per-plane popcount over real dims.
    plane_pops: [u32; 8],
    /// `qscale` from the int8 quantization (`q_i8 = round(q_rot/qscale)`).
    qscale: f32,
    /// Exact ‖q‖².
    pub norm_sq: f32,
    dim: usize,
}

impl Bits1Query {
    /// Build from the rotated f32 query (same input the Turbo4 query prep
    /// uses; `qscale`/`norm_sq` semantics identical to `encode_query`).
    pub fn new(rotated: &[f32]) -> Result<Self, TurboQuantError> {
        let dim = rotated.len();
        if dim < 2 {
            return Err(TurboQuantError::InvalidDimension(dim));
        }
        let norm_sq: f32 = rotated.iter().map(|x| x * x).sum();
        let qmax = rotated.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let qscale = if qmax > 0.0 { qmax / 127.0 } else { 0.0 };
        let inv = if qscale > 0.0 { 1.0 / qscale } else { 0.0 };

        let n_words = dim.div_ceil(64);
        let mut planes: [Vec<u64>; 8] = std::array::from_fn(|_| vec![0u64; n_words]);
        let mut plane_pops = [0u32; 8];
        for (i, &x) in rotated.iter().enumerate() {
            let q_u8 = ((x * inv).round() as i8 as i16 + 128) as u16 as u8;
            for (p, plane) in planes.iter_mut().enumerate() {
                if q_u8 >> p & 1 != 0 {
                    plane[i / 64] |= 1u64 << (i % 64);
                    plane_pops[p] += 1;
                }
            }
        }
        Ok(Self {
            planes,
            plane_pops,
            qscale,
            norm_sq,
            dim,
        })
    }

    /// Estimated distance to a 1-bit code blob under `metric`. Pure
    /// AND+POPCNT per plane; conventions match the Turbo4 scorers.
    pub fn distance_to(&self, metric: Metric, blob: &[u8]) -> f32 {
        let qblob = self.to_blob();
        query_blob_distance(metric, &qblob, blob, self.dim)
    }

    /// Serialize into the flat query-blob form consumed by
    /// [`query_blob_distance`] — this is what travels through
    /// `hnsw_rs::Distance<u8>::eval` in cascade mode.
    ///
    /// Layout: `[8 planes × bits_len | 8 × plane_pop u32 | qscale | ‖q‖²]`.
    pub fn to_blob(&self) -> Vec<u8> {
        let bl = bits_len(self.dim);
        let mut out = Vec::with_capacity(query1_len(self.dim));
        for plane in &self.planes {
            for w in plane {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        for p in &self.plane_pops {
            out.extend_from_slice(&p.to_le_bytes());
        }
        out.extend_from_slice(&self.qscale.to_le_bytes());
        out.extend_from_slice(&self.norm_sq.to_le_bytes());
        debug_assert_eq!(out.len(), 8 * bl + 40);
        out
    }
}

/// Query-blob length for `dim` (`8·bits_len + 40`).
#[inline]
pub fn query1_len(dim: usize) -> usize {
    8 * bits_len(dim) + 40
}

/// Score a serialized 1-bit query blob (see [`Bits1Query::to_blob`]) against
/// a 1-bit code blob, on raw slices — no allocation, callable from inside a
/// `Distance<u8>` functor.
pub fn query_blob_distance(metric: Metric, qblob: &[u8], code: &[u8], dim: usize) -> f32 {
    let n_words = dim.div_ceil(64);
    let bl = n_words * 8;
    assert!(dim >= 2, "1-bit query dimensions must be at least 2");
    assert_eq!(qblob.len(), 8 * bl + 40, "invalid 1-bit query length");
    assert_eq!(code.len(), bl + META_BYTES, "invalid 1-bit code length");

    let alpha = f32::from_le_bytes(code[bl..bl + 4].try_into().unwrap());
    let c = f32::from_le_bytes(code[bl + 4..bl + 8].try_into().unwrap());
    let qscale = f32::from_le_bytes(qblob[8 * bl + 32..8 * bl + 36].try_into().unwrap());
    let q_norm_sq = f32::from_le_bytes(qblob[8 * bl + 36..8 * bl + 40].try_into().unwrap());

    let word = |bytes: &[u8], k: usize| -> u64 {
        u64::from_le_bytes(bytes[k * 8..k * 8 + 8].try_into().unwrap())
    };

    let mut bits_pop = 0u32;
    for k in 0..n_words {
        bits_pop += word(code, k).count_ones();
    }
    let mut dot_u8 = 0i64;
    for p in 0..8 {
        let plane = &qblob[p * bl..(p + 1) * bl];
        let pop_p = u32::from_le_bytes(
            qblob[8 * bl + p * 4..8 * bl + p * 4 + 4]
                .try_into()
                .unwrap(),
        );
        let mut agree = 0u32;
        for k in 0..n_words {
            agree += (word(plane, k) & word(code, k)).count_ones();
        }
        dot_u8 += (1i64 << p) * (2 * agree as i64 - pop_p as i64);
    }
    let sum_s = 2 * bits_pop as i64 - dim as i64;
    let dot_i8 = dot_u8 - 128 * sum_s;
    let dot = qscale * alpha * c * dot_i8 as f32;

    let norm_sq_v = alpha * alpha * dim as f32; // exact
    match metric {
        Metric::Euclidean => (q_norm_sq + norm_sq_v - 2.0 * dot).max(0.0).sqrt(),
        Metric::Cosine => {
            let denom = (q_norm_sq * norm_sq_v).sqrt();
            if denom > 0.0 {
                (1.0 - dot / denom).max(0.0)
            } else {
                1.0
            }
        }
        Metric::DotProduct => (-dot).max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::Turbo4Codec;
    use crate::rotation::{Rotation, SplitMix64};

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

    /// Popcount path must equal the naive Σ q_i8·sᵢ sign-sum exactly.
    #[test]
    fn bitplane_dot_matches_naive_sign_sum() {
        for dim in [64usize, 100, 384, 1536] {
            let rot = Rotation::new(dim, 42);
            for seed in 0..4u64 {
                let v = gauss_vec(dim, seed + 1);
                let q = gauss_vec(dim, seed + 100);
                let rv = rot.apply(&v);
                let rq = rot.apply(&q);
                let norm_sq: f32 = rv.iter().map(|x| x * x).sum();
                let alpha = (norm_sq / dim as f32).sqrt();
                let blob = encode_bits(&rv, alpha);
                let query = Bits1Query::new(&rq).unwrap();

                // Naive: identical int8 quantization, direct sign sum.
                let qmax = rq.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                let qscale = if qmax > 0.0 { qmax / 127.0 } else { 0.0 };
                let inv_a = if alpha > 0.0 { 1.0 / alpha } else { 0.0 };
                let mut naive = 0i64;
                for i in 0..dim {
                    let qi = (rq[i] / qscale).round() as i8 as i64;
                    let s = if rv[i] * inv_a >= 0.0 { 1 } else { -1 };
                    naive += qi * s;
                }
                let n_words = dim.div_ceil(64);
                let alpha_read =
                    f32::from_le_bytes(blob[n_words * 8..n_words * 8 + 4].try_into().unwrap());
                let c =
                    f32::from_le_bytes(blob[n_words * 8 + 4..n_words * 8 + 8].try_into().unwrap());
                let expected_dot = qscale * alpha_read * c * naive as f32;
                // Recover the kernel's dot from the Euclidean output.
                let d = query.distance_to(Metric::Euclidean, &blob);
                let norm_sq_v = alpha_read * alpha_read * dim as f32;
                let kernel_dot = (query.norm_sq + norm_sq_v - d * d) / 2.0;
                assert!(
                    (kernel_dot - expected_dot).abs() <= 1e-2 * expected_dot.abs().max(1.0),
                    "dim {dim} seed {seed}: kernel dot {kernel_dot} vs naive {expected_dot}"
                );
            }
        }
    }

    /// 1-bit candidate generation: with 4x oversampling the 1-bit plane must
    /// retain the true top-10 well enough for Turbo4 rescoring to recover it.
    #[test]
    fn candidate_generation_recall_with_oversampling() {
        let dim = 128;
        let n = 300;
        let codec = Turbo4Codec::new(dim, 42).unwrap();
        let rot = Rotation::new(dim, 42);
        let base: Vec<Vec<f32>> = (0..n as u64).map(|i| gauss_vec(dim, 500 + i)).collect();
        let blobs: Vec<Vec<u8>> = base
            .iter()
            .map(|v| {
                let rv = rot.apply(v);
                let norm_sq: f32 = rv.iter().map(|x| x * x).sum();
                encode_bits(&rv, (norm_sq / dim as f32).sqrt())
            })
            .collect();
        let t4codes: Vec<Vec<u8>> = base.iter().map(|v| codec.encode(v).unwrap()).collect();

        let mut total_hits = 0usize;
        for qs in 0..10u64 {
            let q = gauss_vec(dim, 9000 + qs);
            let rq = rot.apply(&q);
            let bq = Bits1Query::new(&rq).unwrap();
            let tq = codec.encode_query(&q).unwrap();

            let l2 =
                |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
            let mut truth: Vec<(usize, f32)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2(&q, v)))
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1));
            let top10: std::collections::HashSet<usize> =
                truth[..10].iter().map(|(i, _)| *i).collect();

            // Stage 1: 1-bit candidates, 4x oversampled.
            let mut stage1: Vec<(usize, f32)> = blobs
                .iter()
                .enumerate()
                .map(|(i, b)| (i, bq.distance_to(Metric::Euclidean, b)))
                .collect();
            stage1.sort_by(|a, b| a.1.total_cmp(&b.1));
            // Stage 2: Turbo4 exact-LUT rescore of the top 40.
            let mut stage2: Vec<(usize, f32)> = stage1[..40]
                .iter()
                .map(|&(i, _)| {
                    (
                        i,
                        crate::score::rescore(Metric::Euclidean, &tq, &t4codes[i], dim),
                    )
                })
                .collect();
            stage2.sort_by(|a, b| a.1.total_cmp(&b.1));
            total_hits += stage2[..10]
                .iter()
                .filter(|(i, _)| top10.contains(i))
                .count();
        }
        let recall = total_hits as f32 / 100.0;
        assert!(
            recall >= 0.70,
            "1-bit cascade recall@10 {recall} below floor on Gaussian worst case"
        );
    }

    #[test]
    fn zero_vector_is_safe() {
        let dim = 64;
        let blob = encode_bits(&vec![0.0; dim], 0.0);
        let q = Bits1Query::new(&vec![0.0; dim]).unwrap();
        let d = q.distance_to(Metric::Euclidean, &blob);
        assert_eq!(d, 0.0);
        assert_eq!(q.distance_to(Metric::Cosine, &blob), 1.0);
    }

    #[test]
    fn blob_length_is_word_padded() {
        assert_eq!(code1_len(64), 8 + 8);
        assert_eq!(code1_len(100), 16 + 8);
        assert_eq!(code1_len(1536), 192 + 8);
    }
}
