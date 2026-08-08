//! Integer dot-product kernels over packed Turbo4 codes.
//!
//! Two primitives, each with a scalar oracle and an AVX2 path selected at
//! runtime (NEON / AVX-512 VNNI / WASM SIMD128 are phase-3 of ADR-296 and
//! plug into the same dispatch seam):
//!
//! * [`dot_i8_nibble`] — asymmetric: int8 query (`D` lanes, contiguous)
//!   × packed nibble code (`D/2` bytes, low nibbles = dims `0..D/2`,
//!   high nibbles = dims `D/2..D`).
//! * [`dot_nibble_nibble`] — symmetric: packed code × packed code.
//!
//! Nibbles index [`LEVELS_I8`], the 16-entry int8 Lloyd-Max grid that fits a
//! single `pshufb` register. Products are i8×i8 (≤ 16 129), accumulated
//! exactly in i32 — the kernels are bit-exact vs the scalar oracle, which the
//! tests enforce.

use crate::tables::LEVELS_I8;

/// Asymmetric dot: Σᵢ q[i] · L_i8[code[i]] with q as raw-i8 bytes.
/// `nibbles.len() == dim/2`, `q.len() == dim`.
#[inline]
pub fn dot_i8_nibble(nibbles: &[u8], q: &[u8], dim: usize) -> i32 {
    // These are safety preconditions for the unchecked AVX2 loads below, so
    // they must remain enforced in release builds. A debug-only assertion
    // would make this safe public function unsound for malformed slices.
    assert_eq!(dim % 2, 0, "Turbo4 dimensions must be even");
    assert_eq!(nibbles.len(), dim / 2, "invalid Turbo4 code length");
    assert!(q.len() >= dim, "invalid Turbo4 query length");
    #[cfg(target_arch = "x86_64")]
    {
        if dim >= 64 && is_x86_feature_detected!("avx2") {
            return unsafe { dot_i8_nibble_avx2(nibbles, q, dim) };
        }
    }
    dot_i8_nibble_scalar(nibbles, q, dim)
}

/// Symmetric dot: Σᵢ L_i8[a[i]] · L_i8[b[i]] over two packed codes.
#[inline]
pub fn dot_nibble_nibble(a: &[u8], b: &[u8], dim: usize) -> i32 {
    // Keep the unchecked AVX2 loads memory-safe for every caller, including
    // release builds and callers outside this crate.
    assert_eq!(dim % 2, 0, "Turbo4 dimensions must be even");
    assert_eq!(a.len(), dim / 2, "invalid left Turbo4 code length");
    assert_eq!(b.len(), dim / 2, "invalid right Turbo4 code length");
    #[cfg(target_arch = "x86_64")]
    {
        if dim >= 64 && is_x86_feature_detected!("avx2") {
            return unsafe { dot_nibble_nibble_avx2(a, b, dim) };
        }
    }
    dot_nibble_nibble_scalar(a, b, dim)
}

/// Rescore dot: Σᵢ q[i] · L_i8[code[i]] with an f32 query, returned in level
/// units (multiply by `I8_UNIT · α` for the physical dot). Uses the int8
/// level grid on every path so scalar and SIMD agree in semantics; the grid's
/// rounding (≤ `I8_UNIT/2` ≈ 0.011 per level) is ~1 % of the intrinsic 4-bit
/// code error, so the rescore tier remains the highest-fidelity scorer.
#[inline]
pub fn dot_f32_nibble(nibbles: &[u8], q: &[f32], dim: usize) -> f32 {
    // These checks guard raw-pointer vector loads in the AVX2/FMA path.
    assert_eq!(dim % 2, 0, "Turbo4 dimensions must be even");
    assert_eq!(nibbles.len(), dim / 2, "invalid Turbo4 code length");
    assert!(q.len() >= dim, "invalid Turbo4 query length");
    #[cfg(target_arch = "x86_64")]
    {
        if dim >= 64 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_nibble_avx2(nibbles, q, dim) };
        }
    }
    dot_f32_nibble_scalar(nibbles, q, dim)
}

pub(crate) fn dot_f32_nibble_scalar(nibbles: &[u8], q: &[f32], dim: usize) -> f32 {
    let half = dim / 2;
    let mut acc = 0.0f32;
    for i in 0..half {
        let lo = LEVELS_I8[(nibbles[i] & 0x0F) as usize] as f32;
        let hi = LEVELS_I8[(nibbles[i] >> 4) as usize] as f32;
        acc += q[i] * lo + q[i + half] * hi;
    }
    acc
}

pub(crate) fn dot_i8_nibble_scalar(nibbles: &[u8], q: &[u8], dim: usize) -> i32 {
    let half = dim / 2;
    let mut acc = 0i32;
    for i in 0..half {
        let lo = LEVELS_I8[(nibbles[i] & 0x0F) as usize] as i32;
        let hi = LEVELS_I8[(nibbles[i] >> 4) as usize] as i32;
        acc += (q[i] as i8 as i32) * lo + (q[i + half] as i8 as i32) * hi;
    }
    acc
}

pub(crate) fn dot_nibble_nibble_scalar(a: &[u8], b: &[u8], dim: usize) -> i32 {
    let half = dim / 2;
    let mut acc = 0i32;
    for i in 0..half {
        let al = LEVELS_I8[(a[i] & 0x0F) as usize] as i32;
        let ah = LEVELS_I8[(a[i] >> 4) as usize] as i32;
        let bl = LEVELS_I8[(b[i] & 0x0F) as usize] as i32;
        let bh = LEVELS_I8[(b[i] >> 4) as usize] as i32;
        acc += al * bl + ah * bh;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::LEVELS_I8;
    use std::arch::x86_64::*;

    /// The 16-entry level grid broadcast to both 128-bit lanes for
    /// `_mm256_shuffle_epi8` (which shuffles within lanes).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn level_table() -> __m256i {
        let t = _mm_loadu_si128(LEVELS_I8.as_ptr() as *const __m128i);
        _mm256_broadcastsi128_si256(t)
    }

    /// Multiply signed i8 lanes of `a` and `b`, accumulating i32 into `acc`.
    ///
    /// Uses the abs/sign + `maddubs` idiom (the pshufb-LUT kernel shape
    /// production engines converge on — ADR-296 refinements §3):
    /// `maddubs(|a|, sign(b, a))` computes exact `aᵢ·bᵢ` pairs in i16, then
    /// one `madd(1)` widens to i32. Replaces the previous four
    /// `cvtepi8_epi16` + two `madd` sequence (~35 % fewer µops).
    ///
    /// Operand contract: `a` may span the full i8 range including −128
    /// (`abs` feeds `maddubs`'s *unsigned* operand, where the 0x80 pattern
    /// reads as +128); `b` must lie in [−127, 127], because `sign(b, a)`
    /// negates `b` and −128 would wrap. Every Turbo4 call site puts the
    /// level table (±127 by construction) in `b`. Saturation-free: pair
    /// sums are ≤ 2·128·127 = 32 512 < i16::MAX.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn madd_i8(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        let a_abs = _mm256_abs_epi8(a);
        let b_signed = _mm256_sign_epi8(b, a);
        let pairs_i16 = _mm256_maddubs_epi16(a_abs, b_signed);
        let p = _mm256_madd_epi16(pairs_i16, _mm256_set1_epi16(1));
        _mm256_add_epi32(acc, p)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_i32(v: __m256i) -> i32 {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        _mm_cvtsi128_si32(s)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i8_nibble_avx2(nibbles: &[u8], q: &[u8], dim: usize) -> i32 {
        let half = dim / 2;
        let table = level_table();
        let mask = _mm256_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_si256();

        let chunks = half / 32;
        for c in 0..chunks {
            let i = c * 32;
            let packed = _mm256_loadu_si256(nibbles.as_ptr().add(i) as *const __m256i);
            // Low nibbles → levels for dims i..i+32 (contiguous run 1).
            let lo_idx = _mm256_and_si256(packed, mask);
            let lo_lev = _mm256_shuffle_epi8(table, lo_idx);
            let q_lo = _mm256_loadu_si256(q.as_ptr().add(i) as *const __m256i);
            acc = madd_i8(acc, q_lo, lo_lev);
            // High nibbles → levels for dims half+i..half+i+32 (run 2).
            let hi_idx = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
            let hi_lev = _mm256_shuffle_epi8(table, hi_idx);
            let q_hi = _mm256_loadu_si256(q.as_ptr().add(half + i) as *const __m256i);
            acc = madd_i8(acc, q_hi, hi_lev);
        }

        let mut total = hsum_i32(acc);
        // Scalar tail over remaining packed bytes.
        for i in chunks * 32..half {
            let lo = LEVELS_I8[(nibbles[i] & 0x0F) as usize] as i32;
            let hi = LEVELS_I8[(nibbles[i] >> 4) as usize] as i32;
            total += (q[i] as i8 as i32) * lo + (q[i + half] as i8 as i32) * hi;
        }
        total
    }

    /// Accumulate `q[k..k+8] · f32(levels_i8[k..k+8])` for the 32 i8 levels
    /// in `lev` starting at query offset `base`. Byte order out of `pshufb`
    /// is sequential, so query loads stay contiguous — no scrambling.
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn fmadd_levels(acc: __m256, lev: __m256i, q: &[f32], base: usize) -> __m256 {
        let lo128 = _mm256_castsi256_si128(lev);
        let hi128 = _mm256_extracti128_si256(lev, 1);
        let mut acc = acc;
        for (g, half) in [(0usize, lo128), (2usize, hi128)] {
            let g0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(half));
            let q0 = _mm256_loadu_ps(q.as_ptr().add(base + g * 8));
            acc = _mm256_fmadd_ps(g0, q0, acc);
            let g1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(half, 8)));
            let q1 = _mm256_loadu_ps(q.as_ptr().add(base + (g + 1) * 8));
            acc = _mm256_fmadd_ps(g1, q1, acc);
        }
        acc
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_f32_nibble_avx2(nibbles: &[u8], q: &[f32], dim: usize) -> f32 {
        let half = dim / 2;
        let table = level_table();
        let mask = _mm256_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_ps();

        let chunks = half / 32;
        for c in 0..chunks {
            let i = c * 32;
            let packed = _mm256_loadu_si256(nibbles.as_ptr().add(i) as *const __m256i);
            let lo_lev = _mm256_shuffle_epi8(table, _mm256_and_si256(packed, mask));
            acc = fmadd_levels(acc, lo_lev, q, i);
            let hi_lev =
                _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask));
            acc = fmadd_levels(acc, hi_lev, q, half + i);
        }

        // Horizontal sum of 8 f32 lanes.
        let hi = _mm256_extractf128_ps(acc, 1);
        let s = _mm_add_ps(_mm256_castps256_ps128(acc), hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        let mut total = _mm_cvtss_f32(s);

        for i in chunks * 32..half {
            let lo = LEVELS_I8[(nibbles[i] & 0x0F) as usize] as f32;
            let hi = LEVELS_I8[(nibbles[i] >> 4) as usize] as f32;
            total += q[i] * lo + q[i + half] * hi;
        }
        total
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_nibble_nibble_avx2(a: &[u8], b: &[u8], dim: usize) -> i32 {
        let half = dim / 2;
        let table = level_table();
        let mask = _mm256_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_si256();

        let chunks = half / 32;
        for c in 0..chunks {
            let i = c * 32;
            let pa = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let pb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let a_lo = _mm256_shuffle_epi8(table, _mm256_and_si256(pa, mask));
            let b_lo = _mm256_shuffle_epi8(table, _mm256_and_si256(pb, mask));
            acc = madd_i8(acc, a_lo, b_lo);
            let a_hi = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(pa, 4), mask));
            let b_hi = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(pb, 4), mask));
            acc = madd_i8(acc, a_hi, b_hi);
        }

        let mut total = hsum_i32(acc);
        for i in chunks * 32..half {
            let al = LEVELS_I8[(a[i] & 0x0F) as usize] as i32;
            let ah = LEVELS_I8[(a[i] >> 4) as usize] as i32;
            let bl = LEVELS_I8[(b[i] & 0x0F) as usize] as i32;
            let bh = LEVELS_I8[(b[i] >> 4) as usize] as i32;
            total += al * bl + ah * bh;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
use avx2::{dot_f32_nibble_avx2, dot_i8_nibble_avx2, dot_nibble_nibble_avx2};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotation::SplitMix64;

    fn random_code(half: usize, seed: u64) -> Vec<u8> {
        let mut rng = SplitMix64(seed);
        (0..half).map(|_| (rng.next_u64() & 0xFF) as u8).collect()
    }

    fn random_q(dim: usize, seed: u64) -> Vec<u8> {
        let mut rng = SplitMix64(seed);
        (0..dim).map(|_| (rng.next_u64() & 0xFF) as u8).collect()
    }

    #[test]
    fn simd_matches_scalar_oracle() {
        // Cover multiple sizes incl. non-multiples of 64 dims (tail path).
        for dim in [64usize, 128, 192, 384, 1536, 100, 70] {
            let dim = dim & !1; // even
            let half = dim / 2;
            for seed in 0..5u64 {
                let code_a = random_code(half, seed * 3 + 1);
                let code_b = random_code(half, seed * 3 + 2);
                let q = random_q(dim, seed * 3 + 3);
                assert_eq!(
                    dot_i8_nibble(&code_a, &q, dim),
                    dot_i8_nibble_scalar(&code_a, &q, dim),
                    "asym dim {dim} seed {seed}"
                );
                assert_eq!(
                    dot_nibble_nibble(&code_a, &code_b, dim),
                    dot_nibble_nibble_scalar(&code_a, &code_b, dim),
                    "sym dim {dim} seed {seed}"
                );
            }
        }
    }

    /// The f32 kernel is float math, so SIMD and scalar differ only by
    /// summation order — bound the relative error tightly.
    #[test]
    fn f32_kernel_matches_scalar_within_epsilon() {
        for dim in [64usize, 128, 384, 1536, 100] {
            let dim = dim & !1;
            let half = dim / 2;
            for seed in 0..4u64 {
                let code = random_code(half, seed * 5 + 1);
                let mut s = seed * 5 + 2;
                let q: Vec<f32> = (0..dim)
                    .map(|_| {
                        let mut st = s;
                        s = s.wrapping_add(1);
                        st = st.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(7);
                        ((st >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                    })
                    .collect();
                let fast = dot_f32_nibble(&code, &q, dim);
                let oracle = dot_f32_nibble_scalar(&code, &q, dim);
                let tol = 1e-3 * oracle.abs().max(1.0);
                assert!(
                    (fast - oracle).abs() <= tol,
                    "dim {dim} seed {seed}: {fast} vs {oracle}"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::identity_op, clippy::neg_multiply)] // literal per-dim products mirror the layout
    fn known_small_case() {
        // dim 4: dims 0,1 in low nibbles of bytes 0,1; dims 2,3 in high.
        // code: dim0=15 (level 127), dim1=0 (level -127), dim2=8 (6), dim3=7 (-6)
        let code = vec![0x8F, 0x70];
        let q = vec![1i8 as u8, 2i8 as u8, 3i8 as u8, (-4i8) as u8];
        let expect = 1 * 127 + 2 * -127 + 3 * 6 + -4 * -6;
        assert_eq!(dot_i8_nibble_scalar(&code, &q, 4), expect);
    }

    #[test]
    fn malformed_buffers_are_rejected_before_simd_dispatch() {
        let dim = 64;
        assert!(std::panic::catch_unwind(|| dot_i8_nibble(&[0; 31], &[0; 64], dim)).is_err());
        assert!(std::panic::catch_unwind(|| dot_nibble_nibble(&[0; 32], &[0; 31], dim)).is_err());
        assert!(std::panic::catch_unwind(|| dot_f32_nibble(&[0; 32], &[0.0; 63], dim)).is_err());
        assert!(std::panic::catch_unwind(|| dot_i8_nibble(&[0; 32], &[0; 65], 65)).is_err());
    }
}
