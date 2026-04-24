//! Acceptance criterion #2 of ADR-151: every entry of `PRIMES_BELOW_2K` and
//! `PRIMES_ABOVE_2K` must agree with the runtime Miller-Rabin descent.
//!
//! For each `k ∈ [8, 64]` (BELOW) / `[8, 63]` (ABOVE) we re-run MR on the
//! tabled prime, then sweep every odd integer in the gap to `2^k` and
//! assert no other prime hides there. This is what makes MR — not the
//! table — the source of truth.

use ruvector_collections::primality::{is_prime_u64, PRIMES_ABOVE_2K, PRIMES_BELOW_2K};

/// Iterate odd candidates strictly between `lo` (exclusive) and `hi`
/// (exclusive), without overflowing `u64`. Used to confirm the prime gap
/// reported by the table contains nothing else prime.
fn sweep_odds_strictly_between<F: FnMut(u64)>(lo: u64, hi: u64, mut f: F) {
    let mut n = match lo.checked_add(1) {
        Some(n) => n,
        None => return,
    };
    if n & 1 == 0 {
        n = match n.checked_add(1) {
            Some(n) => n,
            None => return,
        };
    }
    while n < hi {
        f(n);
        n = match n.checked_add(2) {
            Some(n) => n,
            None => return,
        };
    }
}

#[test]
fn primality_below_table_cross_check() {
    for k in 8u32..=64 {
        let p = PRIMES_BELOW_2K[(k - 8) as usize];
        assert!(
            is_prime_u64(p),
            "PRIMES_BELOW_2K[k={k}] = {p} not prime per Miller-Rabin"
        );

        // hi = 2^k, but 2^64 doesn't fit in u64. Cap at u64::MAX + 1 by
        // using checked semantics and treating "no upper bound" as scan
        // up through u64::MAX inclusive.
        let hi = if k == 64 {
            // Sweep p+1..=u64::MAX (inclusive). Using u64::MAX as an
            // exclusive bound and then checking u64::MAX separately.
            sweep_odds_strictly_between(p, u64::MAX, |m| {
                assert!(
                    !is_prime_u64(m),
                    "found prime {m} > PRIMES_BELOW_2K[64] = {p} (within u64)"
                );
            });
            // u64::MAX itself: factor into 3 × ... so trivially composite,
            // but assert anyway.
            assert!(!is_prime_u64(u64::MAX), "u64::MAX is composite");
            continue;
        } else {
            1u64 << k
        };

        sweep_odds_strictly_between(p, hi, |m| {
            assert!(
                !is_prime_u64(m),
                "found prime {m} in (PRIMES_BELOW_2K[k={k}] = {p}, 2^{k} = {hi})"
            );
        });
    }
}

#[test]
fn primality_above_table_cross_check() {
    // k = 64 entry is a sentinel (no u64 prime > 2^64) — skip it.
    for k in 8u32..=63 {
        let p = PRIMES_ABOVE_2K[(k - 8) as usize];
        assert!(
            is_prime_u64(p),
            "PRIMES_ABOVE_2K[k={k}] = {p} not prime per Miller-Rabin"
        );
        let lo = 1u64 << k;
        sweep_odds_strictly_between(lo, p, |m| {
            assert!(
                !is_prime_u64(m),
                "found prime {m} in (2^{k} = {lo}, PRIMES_ABOVE_2K[k={k}] = {p})"
            );
        });
    }

    // Sentinel check: the k=64 slot must remain 0 (any non-zero value
    // would imply a u64 prime > 2^64, which is impossible).
    assert_eq!(
        PRIMES_ABOVE_2K[(64 - 8) as usize],
        0,
        "PRIMES_ABOVE_2K[64] must be the sentinel 0 — there is no u64 prime > 2^64"
    );
}
