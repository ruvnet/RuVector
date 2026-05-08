//! Asymmetric distance kernels for LVQ.
//!
//! Queries are kept in fp32. Database vectors are decoded on the fly while
//! computing the inner product or squared L2 — this keeps memory traffic
//! at one byte per dimension while preserving fp32 query precision.
//!
//! All kernels are written in straight-line scalar code. The compiler
//! auto-vectorises them on x86_64 (`-C target-cpu=native` produces AVX2
//! tight loops) and arm64 (NEON). We intentionally avoid platform-specific
//! intrinsics so the crate stays portable and reproducible.

use crate::quantize::Lvq8Stats;
use crate::two_level::Lvq8x8;

/// Squared L2 distance: `||q - decode(code, stats)||²`.
#[inline]
pub fn lvq8_l2sq(q: &[f32], code: &[u8], stats: Lvq8Stats) -> f32 {
    debug_assert_eq!(q.len(), code.len());
    let bias = stats.mean + stats.bias;
    let scale = stats.scale;
    let mut acc = 0.0_f32;
    for j in 0..q.len() {
        let recon = bias + scale * (code[j] as f32);
        let d = q[j] - recon;
        acc += d * d;
    }
    acc
}

/// Inner product: `<q, decode(code, stats)>`.
#[inline]
pub fn lvq8_dot(q: &[f32], code: &[u8], stats: Lvq8Stats) -> f32 {
    debug_assert_eq!(q.len(), code.len());
    let bias = stats.mean + stats.bias;
    let scale = stats.scale;
    let mut q_sum = 0.0_f32;
    let mut q_dot_code = 0.0_f32;
    for j in 0..q.len() {
        q_sum += q[j];
        q_dot_code += q[j] * (code[j] as f32);
    }
    bias * q_sum + scale * q_dot_code
}

/// Squared L2 distance against the two-level reconstruction:
/// `||q - (decode_primary + decode_residual)||²`.
#[inline]
pub fn lvq8x8_l2sq(q: &[f32], idx: usize, db: &Lvq8x8) -> f32 {
    let dim = db.dim();
    debug_assert_eq!(q.len(), dim);
    let p_stats = db.primary_stats(idx);
    let r_stats = db.residual_stats_at(idx);
    let p_row = db.primary_row(idx);
    let r_row = db.residual_row(idx);

    let p_bias = p_stats.mean + p_stats.bias;
    let p_scale = p_stats.scale;
    let r_bias = r_stats.mean + r_stats.bias;
    let r_scale = r_stats.scale;

    let mut acc = 0.0_f32;
    for j in 0..dim {
        let recon =
            p_bias + p_scale * (p_row[j] as f32) + r_bias + r_scale * (r_row[j] as f32);
        let d = q[j] - recon;
        acc += d * d;
    }
    acc
}

/// Squared L2 against the *primary only* level — used for fast prefiltering.
#[inline]
pub fn lvq8x8_l2sq_primary(q: &[f32], idx: usize, db: &Lvq8x8) -> f32 {
    let stats = db.primary_stats(idx);
    let row = db.primary_row(idx);
    lvq8_l2sq(q, row, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::encode_one;

    #[test]
    fn lvq8_l2sq_matches_decoded_reference() {
        let q: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).cos()).collect();
        let v: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin()).collect();
        let (stats, code) = encode_one(&v).unwrap();

        let approx = lvq8_l2sq(&q, &code, stats);
        let decoded: Vec<f32> = code.iter().map(|&c| stats.decode_lane(c)).collect();
        let reference: f32 = q
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!((approx - reference).abs() < 1e-3, "{approx} vs {reference}");
    }
}
