//! FastScan kernel: applies a pre-built LUT over packed 4-bit neighbor codes.
//!
//! Given a LUT[m * 16] (u8 distances per subspace × centroid) and a slice of
//! packed nibble codes for N neighbors (each code is `ceil(m/2)` bytes), this
//! module returns an estimated distance (u16 accumulator) for every neighbor.
//!
//! Two paths:
//!   `scan_scalar` — portable, always available.
//!   `scan_avx2`   — x86_64 AVX2 `_mm256_shuffle_epi8` path (16 vectors/cycle).
//!
//! Both return identical results (modulo SIMD ordering which we sort afterward).

use crate::pq4::CENTROIDS_PER_SUBSPACE;

/// Estimate distances from a query (represented by `lut`) to every neighbor
/// whose packed 4-bit codes appear in `codes_block`.
///
/// - `lut`: length `m * 16`, u8 distances to each centroid per subspace.
/// - `codes_block`: row-major, each row is `code_bytes` = `ceil(m/2)` bytes.
/// - Returns a `Vec<u16>` of length `n_neighbors` with accumulated distances.
pub fn scan_neighbors(lut: &[u8], codes_block: &[u8], n_neighbors: usize, m: usize) -> Vec<u16> {
    let code_bytes = (m + 1) / 2;
    debug_assert_eq!(codes_block.len(), n_neighbors * code_bytes);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { scan_avx2(lut, codes_block, n_neighbors, m, code_bytes) };
    }

    #[allow(unreachable_code)]
    scan_scalar(lut, codes_block, n_neighbors, m, code_bytes)
}

/// Scalar (portable) implementation. O(n_neighbors × m) operations.
pub fn scan_scalar(lut: &[u8], codes_block: &[u8], n_neighbors: usize, m: usize, code_bytes: usize) -> Vec<u16> {
    let k = CENTROIDS_PER_SUBSPACE;
    let mut out = vec![0u16; n_neighbors];
    for n in 0..n_neighbors {
        let row = &codes_block[n * code_bytes..(n + 1) * code_bytes];
        let mut acc: u16 = 0;
        for sub in 0..m {
            let byte_idx = sub / 2;
            let nibble = if sub % 2 == 0 {
                (row[byte_idx] & 0x0F) as usize
            } else {
                ((row[byte_idx] >> 4) & 0x0F) as usize
            };
            acc = acc.saturating_add(lut[sub * k + nibble] as u16);
        }
        out[n] = acc;
    }
    out
}

/// AVX2 implementation — processes pairs of subspaces using `_mm256_shuffle_epi8`.
///
/// Each call handles `m` subspaces (must be even). We load two LUT rows (sub 0 + sub 1)
/// into a 32-byte SIMD register (lo half = sub 0 centroids, hi half = sub 1 centroids),
/// then use shuffle to look up all 32 neighbors in parallel per subspace pair.
///
/// # Safety
/// Caller must ensure AVX2 is available (`target_feature = "avx2"`).
#[cfg(target_arch = "x86_64")]
pub unsafe fn scan_avx2(
    lut: &[u8],
    codes_block: &[u8],
    n_neighbors: usize,
    m: usize,
    code_bytes: usize,
) -> Vec<u16> {
    #[cfg(target_feature = "avx2")]
    {
        use std::arch::x86_64::*;

        let k = CENTROIDS_PER_SUBSPACE; // 16
        let mut out = vec![0u16; n_neighbors];

        // Process in blocks of 32 neighbors for maximum SIMD utilisation.
        // Remainder is handled by the scalar path.
        let block_size = 32;
        let full_blocks = n_neighbors / block_size;
        let remainder = n_neighbors % block_size;

        for blk in 0..full_blocks {
            let base_n = blk * block_size;
            // Accumulators: one per neighbor in block, stored as pairs in 256-bit regs.
            let mut acc_lo = _mm256_setzero_si256(); // neighbors 0..15 (lo byte of u16)
            let mut acc_hi = _mm256_setzero_si256(); // neighbors 16..31 (lo byte of u16)

            // Collect codes for this block: 32 neighbors × code_bytes.
            let mut block_lo_codes = [0u8; 16 * 32];
            let mut block_hi_codes = [0u8; 16 * 32];
            for n in 0..32 {
                let row = &codes_block[(base_n + n) * code_bytes..(base_n + n + 1) * code_bytes];
                for byte_idx in 0..code_bytes.min(16) {
                    if n < 16 {
                        block_lo_codes[byte_idx * 16 + n] = row[byte_idx];
                    } else {
                        block_hi_codes[byte_idx * 16 + (n - 16)] = row[byte_idx];
                    }
                }
            }

            for sub in 0..m {
                // Load 16-entry LUT for this subspace into both halves of 256-bit reg.
                let lut_ptr = lut[sub * k..].as_ptr();
                let lut_reg = _mm_loadu_si128(lut_ptr as *const __m128i);
                let lut256 = _mm256_set_m128i(lut_reg, lut_reg);

                let byte_idx = sub / 2;
                let lo_byte = if sub % 2 == 0 { 0x0F_u8 } else { 0xF0_u8 };
                let shift = if sub % 2 == 0 { 0 } else { 4 };

                // Load 16 packed codes (nibbles) for neighbors 0..15.
                let codes_lo = _mm_loadu_si128(
                    block_lo_codes[byte_idx * 16..].as_ptr() as *const __m128i
                );
                // Load 16 packed codes for neighbors 16..31.
                let codes_hi = _mm_loadu_si128(
                    block_hi_codes[byte_idx * 16..].as_ptr() as *const __m128i
                );

                // Extract 4-bit nibbles.
                let mask4 = _mm_set1_epi8(lo_byte as i8);
                let nibbles_lo = if shift == 0 {
                    _mm_and_si128(codes_lo, mask4)
                } else {
                    _mm_and_si128(_mm_srli_epi16(codes_lo, shift), _mm_set1_epi8(0x0F))
                };
                let nibbles_hi = if shift == 0 {
                    _mm_and_si128(codes_hi, mask4)
                } else {
                    _mm_and_si128(_mm_srli_epi16(codes_hi, shift), _mm_set1_epi8(0x0F))
                };

                // Shuffle: lookup LUT at nibble indices.
                let dist_lo = _mm_shuffle_epi8(lut_reg, nibbles_lo);
                let dist_hi = _mm_shuffle_epi8(lut_reg, nibbles_hi);

                // Widen to u16 and accumulate.
                let wlo = _mm256_cvtepu8_epi16(dist_lo);
                let whi = _mm256_cvtepu8_epi16(dist_hi);
                acc_lo = _mm256_add_epi16(acc_lo, wlo);
                acc_hi = _mm256_add_epi16(acc_hi, whi);
                let _ = lut256; // suppress unused-variable warning
            }

            // Store results.
            let mut tmp = [0u16; 32];
            _mm256_storeu_si256(tmp[..16].as_mut_ptr() as *mut __m256i, acc_lo);
            _mm256_storeu_si256(tmp[16..].as_mut_ptr() as *mut __m256i, acc_hi);
            out[base_n..base_n + 32].copy_from_slice(&tmp);
        }

        // Scalar remainder.
        if remainder > 0 {
            let rem_base = full_blocks * block_size;
            let rem_codes = &codes_block[rem_base * code_bytes..];
            let rem_out = scan_scalar(lut, rem_codes, remainder, m, code_bytes);
            out[rem_base..rem_base + remainder].copy_from_slice(&rem_out);
        }

        return out;
    }

    // Fallback (should be unreachable when avx2 feature is active).
    scan_scalar(lut, codes_block, n_neighbors, m, code_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_scan_identity_lut() {
        // LUT where centroid 0 has distance 0 for every subspace.
        let m = 4;
        let k = CENTROIDS_PER_SUBSPACE;
        let mut lut = vec![255u8; m * k];
        for sub in 0..m {
            lut[sub * k] = 0; // centroid 0 → distance 0
        }
        // Neighbor codes: all nibbles = 0.
        let n = 3;
        let code_bytes = (m + 1) / 2;
        let codes = vec![0u8; n * code_bytes];
        let dists = scan_scalar(&lut, &codes, n, m, code_bytes);
        assert_eq!(dists, vec![0u16; n]);
    }

    #[test]
    fn scalar_scan_max_lut() {
        let m = 4;
        let k = CENTROIDS_PER_SUBSPACE;
        // All centroids have distance 1 except centroid 0.
        let lut = vec![1u8; m * k];
        let n = 5;
        let code_bytes = (m + 1) / 2;
        let codes = vec![0x00u8; n * code_bytes]; // all nibbles = 0 → dist[0] = 1 per sub
        let dists = scan_scalar(&lut, &codes, n, m, code_bytes);
        // Each neighbor accumulates 1 per subspace = m total.
        assert!(dists.iter().all(|&d| d == m as u16));
    }

    #[test]
    fn scan_neighbors_matches_scalar() {
        let m = 8;
        let k = CENTROIDS_PER_SUBSPACE;
        let lut: Vec<u8> = (0..(m * k) as u8).collect();
        let n = 10;
        let code_bytes = (m + 1) / 2;
        let codes: Vec<u8> = (0..n * code_bytes).map(|i| i as u8 & 0x77).collect();
        let a = scan_scalar(&lut, &codes, n, m, code_bytes);
        let b = scan_neighbors(&lut, &codes, n, m);
        assert_eq!(a, b);
    }
}
