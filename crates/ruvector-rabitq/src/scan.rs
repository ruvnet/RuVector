//! SIMD-accelerated symmetric-scan agreement-count kernel.
//!
//! The symmetric RaBitQ scan reduces to a padding-safe XNOR-popcount between
//! a single query word-vector and every database word-vector. The scalar
//! version runs `u64::count_ones` once per word; this module adds an AVX2-era
//! fast path that (a) uses the hardware `popcnt` instruction directly via
//! `_popcnt64` and (b) unrolls the outer loop by 4 to hide its latency and
//! reduce branch mispredicts.
//!
//! The kernel only computes the agreement count — the cos-LUT lookup,
//! score arithmetic, and TopK heap management stay on the host loop in
//! `index.rs` because they are not SIMD-amenable (small LUT, scalar FP,
//! branchy heap). This file exposes:
//!
//!   * [`scan_scalar`] — portable fallback, identical math to the inline
//!     loop that lived in `index.rs` before this module existed.
//!   * `scan_avx2` — x86_64 AVX2+POPCNT variant (4 candidates/iter).
//!   * [`scan`] — runtime-dispatched entry point. Picks the best kernel
//!     once at process start via a `OnceLock<fn(...)>` cache.
//!
//! All three produce **bit-identical** `out_agree[]` arrays — the SIMD
//! path reorders work but not arithmetic. This is asserted by the unit
//! tests below.

use std::sync::OnceLock;

/// Function pointer signature used by the runtime dispatcher.
///
/// * `packed`    — flat row-major database codes, length `n * n_words`.
/// * `n_words`   — u64 words per candidate.
/// * `n`         — candidate count.
/// * `q_packed`  — query words, length `n_words`.
/// * `mask`      — last-word mask (`!0u64` when `dim % 64 == 0`).
/// * `out_agree` — output agreement counts, length `n`.
type ScanFn = fn(
    packed: &[u64],
    n_words: usize,
    n: usize,
    q_packed: &[u64],
    mask: u64,
    out_agree: &mut [u32],
);

static SCAN_IMPL: OnceLock<ScanFn> = OnceLock::new();

/// Portable scalar kernel. Matches the pre-SIMD inline loop byte-for-byte.
#[inline]
pub fn scan_scalar(
    packed: &[u64],
    n_words: usize,
    n: usize,
    q_packed: &[u64],
    mask: u64,
    out_agree: &mut [u32],
) {
    debug_assert_eq!(packed.len(), n * n_words);
    debug_assert_eq!(q_packed.len(), n_words);
    debug_assert_eq!(out_agree.len(), n);

    let aligned = mask == !0u64;
    if aligned && n_words == 2 {
        // D=128 hot path.
        let q0 = q_packed[0];
        let q1 = q_packed[1];
        for i in 0..n {
            let b = i * 2;
            // SAFETY: asserted above that packed.len() == n * n_words.
            let w0 = unsafe { *packed.get_unchecked(b) };
            let w1 = unsafe { *packed.get_unchecked(b + 1) };
            let a = (!(w0 ^ q0)).count_ones() + (!(w1 ^ q1)).count_ones();
            unsafe { *out_agree.get_unchecked_mut(i) = a };
        }
    } else if aligned {
        for i in 0..n {
            let base = i * n_words;
            let mut a: u32 = 0;
            for w in 0..n_words {
                let wi = unsafe { *packed.get_unchecked(base + w) };
                let qi = unsafe { *q_packed.get_unchecked(w) };
                a += (!(wi ^ qi)).count_ones();
            }
            unsafe { *out_agree.get_unchecked_mut(i) = a };
        }
    } else {
        // Unaligned last word needs the padding-zero mask.
        let last = n_words - 1;
        for i in 0..n {
            let base = i * n_words;
            let mut a: u32 = 0;
            for w in 0..last {
                let wi = unsafe { *packed.get_unchecked(base + w) };
                let qi = unsafe { *q_packed.get_unchecked(w) };
                a += (!(wi ^ qi)).count_ones();
            }
            let wi = unsafe { *packed.get_unchecked(base + last) };
            let qi = unsafe { *q_packed.get_unchecked(last) };
            a += (!(wi ^ qi) & mask).count_ones();
            unsafe { *out_agree.get_unchecked_mut(i) = a };
        }
    }
}

/// AVX2 + POPCNT kernel. Processes 4 candidates per outer iteration for the
/// D=128 (`n_words == 2`, aligned) fast path; falls back to a 4× unrolled
/// popcount loop for other shapes. The actual popcount uses the scalar
/// `popcnt` instruction — on AVX2-class hardware it is already 1/cycle per
/// port, so the win here is pipelining four independent candidates and
/// shrinking the loop overhead, not vectorisation of the popcount itself.
///
/// # Safety
///
/// Caller must guarantee `is_x86_feature_detected!("avx2")` and
/// `is_x86_feature_detected!("popcnt")`. The dispatcher in [`scan`] does
/// this; external callers should use [`scan`] instead.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
unsafe fn scan_avx2(
    packed: &[u64],
    n_words: usize,
    n: usize,
    q_packed: &[u64],
    mask: u64,
    out_agree: &mut [u32],
) {
    use core::arch::x86_64::_popcnt64;

    debug_assert_eq!(packed.len(), n * n_words);
    debug_assert_eq!(q_packed.len(), n_words);
    debug_assert_eq!(out_agree.len(), n);

    let aligned = mask == !0u64;
    let p = packed.as_ptr();
    let o = out_agree.as_mut_ptr();

    if aligned && n_words == 2 {
        // D=128: 2 words per candidate, 4 candidates per iter (= 8 popcnts).
        let q0 = q_packed[0] as i64;
        let q1 = q_packed[1] as i64;
        let n4 = n & !3usize;
        let mut i = 0usize;
        while i < n4 {
            // Load 8 words. LLVM emits aligned 256-bit loads if lucky.
            let b = p.add(i * 2);
            let w0 = *b as i64;
            let w1 = *b.add(1) as i64;
            let w2 = *b.add(2) as i64;
            let w3 = *b.add(3) as i64;
            let w4 = *b.add(4) as i64;
            let w5 = *b.add(5) as i64;
            let w6 = *b.add(6) as i64;
            let w7 = *b.add(7) as i64;
            // Eight independent popcnts — one per candidate-word — run in
            // parallel on ports 1/5. The ALU pipe is the bottleneck, not
            // the loads.
            let a0: i32 = _popcnt64(!(w0 ^ q0)) + _popcnt64(!(w1 ^ q1));
            let a1: i32 = _popcnt64(!(w2 ^ q0)) + _popcnt64(!(w3 ^ q1));
            let a2: i32 = _popcnt64(!(w4 ^ q0)) + _popcnt64(!(w5 ^ q1));
            let a3: i32 = _popcnt64(!(w6 ^ q0)) + _popcnt64(!(w7 ^ q1));
            *o.add(i) = a0 as u32;
            *o.add(i + 1) = a1 as u32;
            *o.add(i + 2) = a2 as u32;
            *o.add(i + 3) = a3 as u32;
            i += 4;
        }
        // Tail.
        while i < n {
            let b = p.add(i * 2);
            let a: i32 = _popcnt64(!((*b as i64) ^ q0)) + _popcnt64(!((*b.add(1) as i64) ^ q1));
            *o.add(i) = a as u32;
            i += 1;
        }
        return;
    }

    // General path: any dim, 4 candidates at a time. Each candidate runs an
    // inner word loop. The outer unroll still reduces loop overhead.
    let n4 = n & !3usize;
    let mut i = 0usize;
    if aligned {
        while i < n4 {
            let mut a0: i32 = 0;
            let mut a1: i32 = 0;
            let mut a2: i32 = 0;
            let mut a3: i32 = 0;
            for w in 0..n_words {
                let qi = *q_packed.get_unchecked(w) as i64;
                a0 += _popcnt64(!((*p.add(i * n_words + w) as i64) ^ qi));
                a1 += _popcnt64(!((*p.add((i + 1) * n_words + w) as i64) ^ qi));
                a2 += _popcnt64(!((*p.add((i + 2) * n_words + w) as i64) ^ qi));
                a3 += _popcnt64(!((*p.add((i + 3) * n_words + w) as i64) ^ qi));
            }
            *o.add(i) = a0 as u32;
            *o.add(i + 1) = a1 as u32;
            *o.add(i + 2) = a2 as u32;
            *o.add(i + 3) = a3 as u32;
            i += 4;
        }
        while i < n {
            let mut a: i32 = 0;
            for w in 0..n_words {
                let qi = *q_packed.get_unchecked(w) as i64;
                a += _popcnt64(!((*p.add(i * n_words + w) as i64) ^ qi));
            }
            *o.add(i) = a as u32;
            i += 1;
        }
    } else {
        let last = n_words - 1;
        let m = mask as i64;
        while i < n4 {
            let mut a0: i32 = 0;
            let mut a1: i32 = 0;
            let mut a2: i32 = 0;
            let mut a3: i32 = 0;
            for w in 0..last {
                let qi = *q_packed.get_unchecked(w) as i64;
                a0 += _popcnt64(!((*p.add(i * n_words + w) as i64) ^ qi));
                a1 += _popcnt64(!((*p.add((i + 1) * n_words + w) as i64) ^ qi));
                a2 += _popcnt64(!((*p.add((i + 2) * n_words + w) as i64) ^ qi));
                a3 += _popcnt64(!((*p.add((i + 3) * n_words + w) as i64) ^ qi));
            }
            let qi = *q_packed.get_unchecked(last) as i64;
            a0 += _popcnt64(!((*p.add(i * n_words + last) as i64) ^ qi) & m);
            a1 += _popcnt64(!((*p.add((i + 1) * n_words + last) as i64) ^ qi) & m);
            a2 += _popcnt64(!((*p.add((i + 2) * n_words + last) as i64) ^ qi) & m);
            a3 += _popcnt64(!((*p.add((i + 3) * n_words + last) as i64) ^ qi) & m);
            *o.add(i) = a0 as u32;
            *o.add(i + 1) = a1 as u32;
            *o.add(i + 2) = a2 as u32;
            *o.add(i + 3) = a3 as u32;
            i += 4;
        }
        while i < n {
            let mut a: i32 = 0;
            for w in 0..last {
                let qi = *q_packed.get_unchecked(w) as i64;
                a += _popcnt64(!((*p.add(i * n_words + w) as i64) ^ qi));
            }
            let qi = *q_packed.get_unchecked(last) as i64;
            a += _popcnt64(!((*p.add(i * n_words + last) as i64) ^ qi) & m);
            *o.add(i) = a as u32;
            i += 1;
        }
    }
}

/// Thin wrapper that adapts the `unsafe` AVX2 kernel to the safe `ScanFn`
/// signature for the dispatcher cache. Safe to call only when the CPU
/// supports AVX2+POPCNT — which the dispatcher checks.
#[cfg(target_arch = "x86_64")]
fn scan_avx2_dispatch(
    packed: &[u64],
    n_words: usize,
    n: usize,
    q_packed: &[u64],
    mask: u64,
    out_agree: &mut [u32],
) {
    // SAFETY: dispatcher only installs this fn pointer if both AVX2 and
    // POPCNT are detected at runtime.
    unsafe { scan_avx2(packed, n_words, n, q_packed, mask, out_agree) };
}

/// Runtime-dispatched entry point. First call installs the best available
/// kernel into a process-global `OnceLock`; subsequent calls dereference a
/// cached function pointer (one predictable indirect branch).
#[inline]
pub fn scan(
    packed: &[u64],
    n_words: usize,
    n: usize,
    q_packed: &[u64],
    mask: u64,
    out_agree: &mut [u32],
) {
    let f = SCAN_IMPL.get_or_init(select_impl);
    f(packed, n_words, n, q_packed, mask, out_agree);
}

fn select_impl() -> ScanFn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt") {
            return scan_avx2_dispatch;
        }
    }
    scan_scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_packed(dim: usize, n: usize, seed: u64) -> (Vec<u64>, Vec<u64>, u64) {
        // Xorshift64 — zero-dep deterministic PRNG so tests don't depend on
        // the `rand` crate's internal sequence.
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut step = || -> u64 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let n_words = (dim + 63) / 64;
        let mut packed = vec![0u64; n * n_words];
        for w in &mut packed {
            *w = step();
        }
        let mut q = vec![0u64; n_words];
        for w in &mut q {
            *w = step();
        }
        // Build the last-word mask.
        let valid_bits = dim - 64 * (n_words - 1);
        let mask = if valid_bits == 64 {
            !0u64
        } else {
            !0u64 << (64 - valid_bits)
        };
        // Zero the padding bits on every candidate+query so both kernels see
        // identical inputs — matches how the index pre-zeroes padding.
        for i in 0..n {
            let last = i * n_words + n_words - 1;
            packed[last] &= mask;
        }
        q[n_words - 1] &= mask;
        (packed, q, mask)
    }

    fn run_both(dim: usize, n: usize, seed: u64) {
        let (packed, q, mask) = random_packed(dim, n, seed);
        let n_words = (dim + 63) / 64;

        let mut out_scalar = vec![0u32; n];
        scan_scalar(&packed, n_words, n, &q, mask, &mut out_scalar);

        // Always exercise the dispatcher (which may pick AVX2 or scalar).
        let mut out_dispatch = vec![0u32; n];
        scan(&packed, n_words, n, &q, mask, &mut out_dispatch);
        assert_eq!(
            out_scalar, out_dispatch,
            "dispatcher output diverged from scalar at dim={dim} n={n}"
        );

        // Directly exercise AVX2 when the host supports it — otherwise the
        // test would silently run scalar-vs-scalar on CI boxes that lack it.
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt") {
            let mut out_avx2 = vec![0u32; n];
            unsafe {
                scan_avx2(&packed, n_words, n, &q, mask, &mut out_avx2);
            }
            assert_eq!(
                out_scalar, out_avx2,
                "AVX2 output diverged from scalar at dim={dim} n={n}"
            );
        }
    }

    #[test]
    fn scan_agree_matches_scalar_at_d128() {
        // 1000 candidates at the production dim.
        run_both(128, 1000, 0xA5A5_5A5A_1234_CAFE);
    }

    #[test]
    fn scan_agree_matches_scalar_at_d64_and_d192() {
        // D=64  → n_words=1, aligned.
        run_both(64, 777, 0x0123_4567_89AB_CDEF);
        // D=192 → n_words=3, aligned.
        run_both(192, 513, 0xFEDC_BA98_7654_3210);
        // D=100 → n_words=2, unaligned (last word masked).
        run_both(100, 641, 0xDEAD_BEEF_CAFE_F00D);
        // D=200 → n_words=4, unaligned.
        run_both(200, 333, 0x1357_9BDF_2468_ACE0);
        // Tail-handling: n not a multiple of 4.
        run_both(128, 1023, 0x4242_4242_4242_4242);
        run_both(128, 7, 0x9999_AAAA_BBBB_CCCC);
    }
}
