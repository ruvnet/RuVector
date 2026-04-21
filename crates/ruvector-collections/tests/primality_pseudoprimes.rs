//! Pinned pseudoprime regressions for the deterministic Miller-Rabin path.
//!
//! These exist so any future "optimization" that shrinks the Sinclair-12
//! witness set fails CI immediately. Numbers come from OEIS A014233
//! (smallest strong pseudoprimes to the first n primes).

use ruvector_collections::primality::{is_prime_u32, is_prime_u64};

/// OEIS A014233(4): smallest spsp to bases {2, 3, 5, 7}. Detected by base 11.
const SPP_2357: u64 = 3_215_031_751;

/// OEIS A014233(5): smallest spsp to bases {2, 3, 5, 7, 11}. Detected by base 13.
const SPP_235711: u64 = 2_152_302_898_747;

/// OEIS A014233(11): smallest spsp to first 11 primes (through 31).
/// Detected ONLY by the 12th Sinclair witness, base 37 — the canary that
/// catches anyone shrinking the witness set.
const SPP_FIRST_11: u64 = 3_825_123_056_546_413_051;

#[test]
fn detects_strong_pseudoprime_2357() {
    assert!(!is_prime_u64(SPP_2357), "{SPP_2357} is composite (detected by base 11)");
}

#[test]
fn detects_strong_pseudoprime_235711() {
    assert!(!is_prime_u64(SPP_235711), "{SPP_235711} is composite (detected by base 13)");
}

#[test]
fn detects_strong_pseudoprime_first_11_primes() {
    assert!(
        !is_prime_u64(SPP_FIRST_11),
        "{SPP_FIRST_11} is composite — detection requires base 37 (Sinclair's last witness)"
    );
}

#[test]
fn small_prime_sanity_under_100() {
    let primes_under_100: [u64; 25] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
        89, 97,
    ];
    for n in 0u64..=100 {
        let expected = primes_under_100.contains(&n);
        assert_eq!(is_prime_u64(n), expected, "is_prime_u64({n})");
    }
}

#[test]
fn edge_cases() {
    assert!(!is_prime_u64(0));
    assert!(!is_prime_u64(1));
    assert!(!is_prime_u64(u64::MAX), "u64::MAX (= 2^64 - 1) factors");
    assert!(is_prime_u64(u64::MAX - 58), "largest u64 prime: u64::MAX - 58");
    // Largest u32 prime is 2^32 - 5 = 4_294_967_291.
    assert!(is_prime_u32(4_294_967_291), "largest u32 prime");
    assert!(!is_prime_u32(u32::MAX));
}

#[test]
fn assorted_known_primes() {
    // Mersenne and other well-known primes inside u64.
    for &p in &[
        7u64,
        127,
        8191,
        131_071,
        524_287,
        2_147_483_647, // 2^31 - 1
        2_305_843_009_213_693_951u64, // 2^61 - 1
    ] {
        assert!(is_prime_u64(p), "{p} is a known prime");
    }
}

#[test]
fn assorted_known_composites() {
    // Carmichael numbers (Fermat-pseudoprimes) — not strong-pseudoprimes,
    // but worth pinning since textbook Fermat tests fail on them.
    for &n in &[561u64, 1105, 1729, 2465, 2821, 6601, 8911] {
        assert!(!is_prime_u64(n), "{n} is a Carmichael number, composite");
    }
}
