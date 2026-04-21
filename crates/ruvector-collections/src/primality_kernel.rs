// Deterministic Miller-Rabin kernel — ADR-151 (PIAL).
//
// `include!`d into two contexts (build.rs and src/primality.rs) which use
// different subsets of the symbols. Per-fn `#[allow(dead_code)]` keeps each
// context warning-clean; inner attributes (#![...]) aren't legal in
// included files.
//
// This file is intentionally context-free: no `use` of crate modules, no
// `pub use` re-exports, no doc-comments that would trip `#![warn(missing_docs)]`
// in dependents. It is `include!`d from BOTH `src/primality.rs` AND `build.rs`
// so the table generator and the runtime share one source of truth.
//
// Witness sets:
//   u32: {2, 7, 61}                                  Pomerance/Selfridge/Wagstaff
//   u64: {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}  Sinclair (2011)
//
// Both are deterministic over their full ranges. Pinned pseudoprime
// regressions live in `tests/primality_pseudoprimes.rs`.

#[inline]
#[allow(dead_code)]
fn mr_mulmod_u64(a: u64, b: u64, m: u64) -> u64 {
    // u128 product avoids overflow without allocation.
    ((a as u128).wrapping_mul(b as u128) % (m as u128)) as u64
}

#[inline]
#[allow(dead_code)]
fn mr_powmod_u64(mut base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut acc: u64 = 1;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mr_mulmod_u64(acc, base, m);
        }
        exp >>= 1;
        if exp > 0 {
            base = mr_mulmod_u64(base, base, m);
        }
    }
    acc
}

// Returns true iff `a` is a Miller-Rabin witness of compositeness for `n`.
// Caller guarantees: n is odd, n > 3, and a in [2, n-2]. n - 1 = d * 2^s
// with d odd (passed in pre-decomposed for speed).
#[inline]
#[allow(dead_code)]
fn mr_is_composite_witness(n: u64, d: u64, s: u32, a: u64) -> bool {
    let mut x = mr_powmod_u64(a, d, n);
    if x == 1 || x == n - 1 {
        return false;
    }
    for _ in 0..s.saturating_sub(1) {
        x = mr_mulmod_u64(x, x, n);
        if x == n - 1 {
            return false;
        }
    }
    true
}

#[inline]
#[allow(dead_code)]
fn mr_is_prime_u64(n: u64) -> bool {
    // Small-n fast path covers all of the ill-defined / edge cases the
    // Sinclair set assumes away (n < 9, even n, n ≤ largest witness).
    if n < 2 {
        return false;
    }
    // Cheap divisibility screen by the first few primes.
    const SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &p in &SMALL_PRIMES {
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }
    // n is now odd, > 37, and coprime to every Sinclair witness — so
    // every witness is a valid base in [2, n-2].
    let mut d = n - 1;
    let mut s: u32 = 0;
    while d & 1 == 0 {
        d >>= 1;
        s += 1;
    }
    for &a in &SMALL_PRIMES {
        if mr_is_composite_witness(n, d, s, a) {
            return false;
        }
    }
    true
}

#[inline]
#[allow(dead_code)]
fn mr_is_prime_u32(n: u32) -> bool {
    // Witnesses {2, 7, 61} are sufficient for all u32; reuse the u64
    // implementation which already screens small primes.
    mr_is_prime_u64(n as u64)
}

// Find the largest prime strictly less than `upper`. Returns 0 if none
// exists in u64 (i.e. upper <= 2). Used by build.rs and the general
// `prev_prime_u64` runtime path.
#[inline]
#[allow(dead_code)]
fn mr_prev_prime_u64(upper: u64) -> u64 {
    if upper <= 2 {
        return 0;
    }
    if upper == 3 {
        return 2;
    }
    // Walk downward through odd candidates.
    let mut n = upper - 1;
    if n & 1 == 0 {
        n -= 1;
    }
    loop {
        if mr_is_prime_u64(n) {
            return n;
        }
        if n <= 3 {
            return 2;
        }
        n -= 2;
    }
}

// Find the smallest prime strictly greater than `lower`. Returns 0 if
// `lower` >= largest u64 prime (u64::MAX - 58).
#[inline]
#[allow(dead_code)]
fn mr_next_prime_u64(lower: u64) -> u64 {
    if lower < 2 {
        return 2;
    }
    if lower < 3 {
        return 3;
    }
    let largest_u64_prime: u64 = u64::MAX - 58;
    if lower >= largest_u64_prime {
        return 0;
    }
    let mut n = lower + 1;
    if n & 1 == 0 {
        n += 1;
    }
    loop {
        if mr_is_prime_u64(n) {
            return n;
        }
        // Bounded: we proved above that some prime exists in (lower, u64::MAX].
        n += 2;
    }
}
