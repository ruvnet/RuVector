//! RaBitQ 1-bit encoding and asymmetric distance estimation.
//!
//! Each D-dimensional vector is binarised as sign(R × x), packed into
//! ceil(D/8) bytes (D bits). The precomputed norm ‖R × x‖₂ is stored
//! separately to enable the asymmetric estimator.
//!
//! ## Asymmetric distance estimate
//!
//! For query q (f32) and database code b (bits) with precomputed ‖x‖:
//!
//!   est_IP(q, x) = (‖q_rot‖ × norm_x / √D) × (2 × popcount(q_sign XNOR b) − D)
//!
//!   est_L2(q, x) = ‖q‖² + ‖x‖² − 2 × est_IP(q, x)
//!
//! where q_rot = R × q, q_sign = sign(q_rot), norm_x = ‖R × x‖.
//!
//! ## Batch batch estimation
//!
//! For the SymphonyQG co-located layout we call `batch_asym_dist` over
//! R neighbor codes stored contiguously. All R codes are read sequentially;
//! distances are accumulated using u64 popcount, matching the FastScan
//! spirit without requiring platform-specific SIMD intrinsics.

/// Number of bytes needed to pack `dim` bits.
#[inline(always)]
pub fn packed_bytes(dim: usize) -> usize {
    dim.div_ceil(8)
}

/// Encode a rotated vector (f32 slice) as 1-bit sign codes packed into bytes.
/// Returns (codes, norm) where norm = ‖x_rot‖₂.
pub fn encode(x_rot: &[f32]) -> (Vec<u8>, f32) {
    let dim = x_rot.len();
    let nbytes = packed_bytes(dim);
    let mut codes = vec![0u8; nbytes];
    for (i, &v) in x_rot.iter().enumerate() {
        if v >= 0.0 {
            codes[i / 8] |= 1 << (i % 8);
        }
    }
    let norm = x_rot.iter().map(|v| v * v).sum::<f32>().sqrt();
    (codes, norm)
}

/// Precomputed per-query data needed for asymmetric estimation.
pub struct QueryProjection {
    /// sign(q_rot) packed as bits, same layout as database codes.
    pub sign_bits: Vec<u8>,
    /// q_rot values (for the correction term).
    pub q_rot: Vec<f32>,
    /// ‖q_rot‖₂.
    pub q_norm: f32,
    /// Dimension.
    pub dim: usize,
}

impl QueryProjection {
    pub fn new(q_rot: Vec<f32>) -> Self {
        let dim = q_rot.len();
        let (sign_bits, q_norm) = encode(&q_rot);
        Self { sign_bits, q_rot, q_norm, dim }
    }
}

/// Asymmetric L2 distance estimate for a single database code.
///
/// Returns the estimated squared L2 distance ‖q − x‖².
#[inline]
pub fn asym_l2_dist(qp: &QueryProjection, code: &[u8], norm_x: f32, norm_q_sq: f32) -> f32 {
    let dim = qp.dim;
    let nbytes = packed_bytes(dim);

    // popcount(q_sign XNOR code) counts matching bits
    let mut matches = 0u32;
    let full_words = nbytes / 8;
    for i in 0..full_words {
        let a = u64::from_le_bytes(qp.sign_bits[i * 8..i * 8 + 8].try_into().unwrap());
        let b = u64::from_le_bytes(code[i * 8..i * 8 + 8].try_into().unwrap());
        matches += (!(a ^ b)).count_ones();
    }
    for i in full_words * 8..nbytes {
        matches += (!(qp.sign_bits[i] ^ code[i])).count_ones() as u32;
    }
    // Correct for padding bits beyond dim (they should not contribute)
    let pad_bits = nbytes * 8 - dim;
    // Bits past dim in the last byte are 0 in code and 0 in sign_bits (default), so xnor=1 → subtract
    matches = matches.saturating_sub(pad_bits as u32);

    // score ∈ [−D, D]: positive means aligned, negative means opposite
    let score = 2 * matches as i32 - dim as i32;
    let est_ip = (qp.q_norm * norm_x / (dim as f32).sqrt()) * score as f32;
    norm_q_sq + norm_x * norm_x - 2.0 * est_ip
}

/// Batch asymmetric L2 estimates for `n_neighbors` codes stored contiguously.
///
/// `codes_block` must be `n_neighbors × nbytes` bytes laid out sequentially.
/// `norms` must be `n_neighbors` floats.
///
/// Returns a `Vec<f32>` of length `n_neighbors` with estimated distances.
pub fn batch_asym_l2(
    qp: &QueryProjection,
    codes_block: &[u8],
    norms: &[f32],
    norm_q_sq: f32,
) -> Vec<f32> {
    let nbytes = packed_bytes(qp.dim);
    let n = norms.len();
    debug_assert_eq!(codes_block.len(), n * nbytes);

    let dim = qp.dim;
    let sqrt_d = (dim as f32).sqrt();
    let q_norm = qp.q_norm;

    norms
        .iter()
        .enumerate()
        .map(|(j, &norm_x)| {
            let code = &codes_block[j * nbytes..(j + 1) * nbytes];
            let mut matches = 0u32;
            let full_words = nbytes / 8;
            for i in 0..full_words {
                let a = u64::from_le_bytes(
                    qp.sign_bits[i * 8..i * 8 + 8].try_into().unwrap(),
                );
                let b = u64::from_le_bytes(code[i * 8..i * 8 + 8].try_into().unwrap());
                matches += (!(a ^ b)).count_ones();
            }
            for i in full_words * 8..nbytes {
                matches += (!(qp.sign_bits[i] ^ code[i])).count_ones() as u32;
            }
            let pad_bits = nbytes * 8 - dim;
            matches = matches.saturating_sub(pad_bits as u32);
            let score = 2 * matches as i32 - dim as i32;
            // Same operation order as asym_l2_dist to avoid IEEE 754 rounding divergence
            let est_ip = (q_norm * norm_x / sqrt_d) * score as f32;
            norm_q_sq + norm_x * norm_x - 2.0 * est_ip
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_signs() {
        let x = vec![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let (codes, _) = encode(&x);
        assert_eq!(codes.len(), 1);
        // bits 0,2,4,6 set (positive values), bits 1,3,5,7 clear
        assert_eq!(codes[0], 0b01010101u8);
    }

    #[test]
    fn asym_aligned_vectors_give_small_distance() {
        let dim = 64;
        let x: Vec<f32> = (0..dim).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let q = x.clone();
        let (code, norm_x) = encode(&x);
        let qp = QueryProjection::new(q.clone());
        let norm_q_sq = q.iter().map(|v| v * v).sum::<f32>();
        let dist = asym_l2_dist(&qp, &code, norm_x, norm_q_sq);
        // Aligned vectors → L2 = 0 (estimated)
        assert!(dist < 10.0, "dist={dist}");
    }

    #[test]
    fn batch_matches_single() {
        let dim = 128;
        let n = 8;
        let mut codes_block = vec![0u8; n * packed_bytes(dim)];
        let mut norms = vec![0.0f32; n];
        let q: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32 - 0.5).collect();
        let qp = QueryProjection::new(q.clone());
        let norm_q_sq = q.iter().map(|v| v * v).sum::<f32>();

        for j in 0..n {
            let x: Vec<f32> = (0..dim).map(|i| (i + j) as f32 / dim as f32 - 0.5).collect();
            let (c, norm) = encode(&x);
            let start = j * packed_bytes(dim);
            codes_block[start..start + packed_bytes(dim)].copy_from_slice(&c);
            norms[j] = norm;
        }

        let batch = batch_asym_l2(&qp, &codes_block, &norms, norm_q_sq);
        for j in 0..n {
            let code = &codes_block[j * packed_bytes(dim)..(j + 1) * packed_bytes(dim)];
            let single = asym_l2_dist(&qp, code, norms[j], norm_q_sq);
            assert!((batch[j] - single).abs() < 1e-6, "mismatch at {j}");
        }
    }
}
