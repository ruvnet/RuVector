//! Deterministic Miller-Rabin primality plus tabled fast paths for the
//! power-of-two-aligned cases that dominate ruvector's hot paths.
//!
//! Designed for ADR-151 (PIAL — Prime-Indexed Acceleration Layer). Five
//! consumers (shard router, HNSW buckets, sparsifier strides, mincut LSH,
//! pi-brain witness chain) get one shared utility and zero new external
//! dependencies.
//!
//! # Determinism
//!
//! | Range | Witnesses | Result |
//! |-------|-----------|--------|
//! | `n < 2^32` | `{2, 7, 61}` (Pomerance/Selfridge/Wagstaff) | Deterministic |
//! | `n < 2^64` | `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}` (Sinclair, 2011) | Deterministic |
//! | `n < 2^128` | 40 random rounds (`unstable-u128` feature) | `Pr[err] < 2⁻⁸⁰` |
//!
//! Pinned-pseudoprime regressions in `tests/primality_pseudoprimes.rs`
//! protect the deterministic ranges from witness-set "optimizations".
//!
//! # Hot vs cold paths
//!
//! Three of PIAL's five sites request primes near *fixed* power-of-two
//! sizes. Those calls hit [`prev_prime_below_pow2`] / [`next_prime_above_pow2`]
//! — a single L1-cached load, ~1 ns. The two unpredictable sites (LSH
//! universe, witness ephemeral primes) use the general MR descent at
//! ~250 ns. Both are cold.
//!
//! Crucially the table is generated at build time from this very module's
//! [`is_prime_u64`], so MR remains the source of truth.

// Pull in the deterministic Miller-Rabin kernel that build.rs also uses.
// Same code, same answers — that's the whole point.
include!("primality_kernel.rs");

// Pull in the build-time-generated tables (PRIMES_BELOW_2K, PRIMES_ABOVE_2K).
include!(concat!(env!("OUT_DIR"), "/prime_tables.rs"));

/// Returns `true` iff `n` is prime. Deterministic for all `u32`.
///
/// Uses the Pomerance/Selfridge/Wagstaff witness set `{2, 7, 61}` via the
/// shared u64 path.
#[inline]
pub fn is_prime_u32(n: u32) -> bool {
    mr_is_prime_u32(n)
}

/// Returns `true` iff `n` is prime. Deterministic for all `u64`.
///
/// Uses Sinclair's 2011 witness set
/// `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}` — known to be sufficient
/// for the entire `u64` range. Allocation-free.
#[inline]
pub fn is_prime_u64(n: u64) -> bool {
    mr_is_prime_u64(n)
}

/// Largest prime strictly less than `2^k`, for `k ∈ [8, 64]`.
///
/// Single L1-cached table load (~1 ns). Use this whenever the caller knows
/// the size is a power of two — shard routers, HNSW bucket sizing,
/// sparsifier strides.
///
/// # Panics (debug only)
///
/// Debug-asserts `8 <= k <= 64`.
#[inline]
pub fn prev_prime_below_pow2(k: u32) -> u64 {
    debug_assert!((8..=64).contains(&k), "k out of table range [8, 64]");
    PRIMES_BELOW_2K[(k - 8) as usize]
}

/// Smallest prime strictly greater than `2^k`, for `k ∈ [8, 63]`.
///
/// Symmetric companion to [`prev_prime_below_pow2`]. The `k = 64` entry of
/// the underlying table is a sentinel (no `u64` prime exists greater than
/// `2^64`); callers must not request it.
///
/// # Panics (debug only)
///
/// Debug-asserts `8 <= k <= 63`.
#[inline]
pub fn next_prime_above_pow2(k: u32) -> u64 {
    debug_assert!(
        (8..=63).contains(&k),
        "k out of table range [8, 63]; PRIMES_ABOVE_2K[64] is a sentinel"
    );
    PRIMES_ABOVE_2K[(k - 8) as usize]
}

/// Largest prime strictly less than `n`. Returns `0` if no such `u64` prime
/// exists (i.e. `n <= 2`).
///
/// Routes power-of-two-aligned inputs (`n = 2^k`, `k ∈ [8, 64]`) to the
/// table; everything else falls through to a Miller-Rabin descent.
#[inline]
pub fn prev_prime_u64(n: u64) -> u64 {
    if n.is_power_of_two() {
        let k = n.trailing_zeros();
        if (8..=64).contains(&k) {
            return PRIMES_BELOW_2K[(k - 8) as usize];
        }
    }
    mr_prev_prime_u64(n)
}

/// Smallest prime strictly greater than `n`. Returns `0` if `n` is at or
/// above the largest `u64` prime (`u64::MAX - 58`).
///
/// Routes power-of-two-aligned inputs (`n = 2^k`, `k ∈ [8, 63]`) to the
/// table; everything else falls through to a Miller-Rabin descent.
#[inline]
pub fn next_prime_u64(n: u64) -> u64 {
    if n.is_power_of_two() {
        let k = n.trailing_zeros();
        if (8..=63).contains(&k) {
            return PRIMES_ABOVE_2K[(k - 8) as usize];
        }
    }
    mr_next_prime_u64(n)
}

/// Derives a deterministic ephemeral prime from `seed`, suitable for the
/// pi-brain witness chain (ADR-151 §4.4).
///
/// Maps the seed into the odd lower-2⁶¹ window then walks up to the next
/// prime. The 2⁶¹ ceiling keeps results well inside `u64` even after the
/// MR walk and lets downstream consumers store the value in a single
/// 64-bit field with room to spare.
#[inline]
pub fn ephemeral_prime(seed: u64) -> u64 {
    let mask = (1u64 << 61) - 1;
    let s = (seed | 1) & mask;
    if mr_is_prime_u64(s) {
        s
    } else {
        // Bounded: the prime gap below 2^61 is far smaller than the
        // remaining headroom to u64::MAX, so this never returns 0.
        mr_next_prime_u64(s)
    }
}

// ── Probabilistic u128 mode (opt-in) ─────────────────────────────────────

/// Probabilistic Miller-Rabin for `u128`. Soundness error `< 4^-rounds`;
/// `rounds = 40` gives `< 2⁻⁸⁰`, adequate for hashing but **not** a
/// cryptographic prime generator (see ADR-151 "Security Considerations").
///
/// Gated behind the `unstable-u128` feature: WASM `u128` codegen is ~5×
/// slower than native and we keep it out of default bundles.
#[cfg(feature = "unstable-u128")]
pub fn is_prime_u128(n: u128, rounds: u8) -> bool {
    if n < 2 {
        return false;
    }
    // Cheap divisibility screen — also catches every n that fits in u64
    // and is one of the Sinclair witnesses.
    const SMALL_PRIMES: [u128; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &p in &SMALL_PRIMES {
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }
    // If n fits in u64, defer to the deterministic path.
    if n <= u64::MAX as u128 {
        return mr_is_prime_u64(n as u64);
    }

    // n > u64::MAX, n odd, coprime to first 12 primes. Decompose n - 1.
    let nm1 = n - 1;
    let s = nm1.trailing_zeros();
    let d = nm1 >> s;

    // Tiny inline LCG seeded from n so the test is reproducible across runs.
    // Numerical-Recipes-style multiplier; we only need uniformity, not crypto.
    let mut state: u128 = n ^ 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835u128;
    for _ in 0..rounds {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Witness in [2, n-2].
        let a = 2u128 + (state % (n - 3));
        if mr_is_composite_u128(n, d, s, a) {
            return false;
        }
    }
    true
}

#[cfg(feature = "unstable-u128")]
#[inline]
fn mr_is_composite_u128(n: u128, d: u128, s: u32, a: u128) -> bool {
    let mut x = powmod_u128(a, d, n);
    if x == 1 || x == n - 1 {
        return false;
    }
    for _ in 0..s.saturating_sub(1) {
        x = mulmod_u128(x, x, n);
        if x == n - 1 {
            return false;
        }
    }
    true
}

#[cfg(feature = "unstable-u128")]
#[inline]
fn powmod_u128(mut base: u128, mut exp: u128, m: u128) -> u128 {
    if m == 1 {
        return 0;
    }
    let mut acc: u128 = 1 % m;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mulmod_u128(acc, base, m);
        }
        exp >>= 1;
        if exp > 0 {
            base = mulmod_u128(base, base, m);
        }
    }
    acc
}

// Russian-peasant mulmod for u128 — works for any m < 2^128 without a u256.
#[cfg(feature = "unstable-u128")]
#[inline]
fn mulmod_u128(mut a: u128, mut b: u128, m: u128) -> u128 {
    let mut acc: u128 = 0;
    a %= m;
    while b > 0 {
        if b & 1 == 1 {
            acc = mod_add_u128(acc, a, m);
        }
        a = mod_add_u128(a, a, m);
        b >>= 1;
    }
    acc
}

#[cfg(feature = "unstable-u128")]
#[inline]
fn mod_add_u128(a: u128, b: u128, m: u128) -> u128 {
    // Pre: a < m, b < m, m may be > 2^127. Computed (a + b) mod m without
    // a u256 by detecting wrapping overflow.
    let sum = a.wrapping_add(b);
    if sum < a || sum >= m {
        sum.wrapping_sub(m)
    } else {
        sum
    }
}

// ── Internal sanity tests (run with the rest of the crate's unit tests) ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_primes_under_100() {
        let known: [u64; 25] = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
            83, 89, 97,
        ];
        for n in 0u64..100 {
            assert_eq!(is_prime_u64(n), known.contains(&n), "is_prime_u64({n})");
        }
    }

    #[test]
    fn edges() {
        assert!(!is_prime_u64(0));
        assert!(!is_prime_u64(1));
        assert!(!is_prime_u64(u64::MAX));
        assert!(is_prime_u64(u64::MAX - 58), "largest u64 prime");
    }

    #[test]
    fn table_index_round_trip() {
        // The most heavily-used shard-router entry.
        assert_eq!(prev_prime_below_pow2(32), 4_294_967_291);
        // Smallest table entry.
        assert_eq!(prev_prime_below_pow2(8), 251);
        // Largest table entry.
        assert_eq!(prev_prime_below_pow2(64), u64::MAX - 58);
    }

    #[cfg(feature = "unstable-u128")]
    #[test]
    fn u128_probabilistic_smoke() {
        use super::is_prime_u128;
        // Defers to deterministic u64 path for n <= u64::MAX.
        assert!(is_prime_u128(7, 40));
        assert!(!is_prime_u128(9, 40));
        assert!(is_prime_u128(u64::MAX as u128 - 58, 40));
        // True u128 path: 2^89 - 1 is a Mersenne prime.
        let m89: u128 = (1u128 << 89) - 1;
        assert!(is_prime_u128(m89, 40), "M_89 = 2^89 - 1 is prime");
        // Composite just above 2^64.
        let composite: u128 = (1u128 << 65) + 1; // = 3 * 11 * 67 * ... (divisible by 3)
        assert!(!is_prime_u128(composite, 40));
    }

    #[test]
    fn ephemeral_prime_is_prime_for_assorted_seeds() {
        for seed in [0u64, 1, 42, 0xDEAD_BEEF, u64::MAX, 1_000_003] {
            let p = ephemeral_prime(seed);
            assert!(is_prime_u64(p), "ephemeral_prime({seed}) = {p} not prime");
            // Loose upper bound: largest known prime gap below 2^64 is well under 2^31,
            // so anything below 2^62 means the walk stayed near its 2^61 starting window.
            assert!(p < (1u64 << 62), "ephemeral_prime overshot expected window");
        }
    }
}
