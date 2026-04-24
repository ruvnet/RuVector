//! Phase-0 benches for ADR-151 / PIAL.
//!
//! Targets (M-series):
//!
//! | bench                                    | target |
//! |------------------------------------------|--------|
//! | `is_prime_u64` (worst case)              | ≤ 50 ns |
//! | `prev_prime_below_pow2` (table fast path)| ≤ 1 ns  |
//! | `next_prime_u64` (arbitrary)             | ≤ 2 µs  |
//! | `next_prime_u64` (2^61)                  | ≤ 12 µs |

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_collections::primality::{is_prime_u64, next_prime_u64, prev_prime_below_pow2};

fn bench_is_prime_u64_worst_case(c: &mut Criterion) {
    // The Sinclair witness loop runs to completion only on actual primes,
    // so use the largest u64 prime as worst-case input.
    let n = u64::MAX - 58;
    c.bench_function("is_prime_u64/worst_case_largest_u64_prime", |b| {
        b.iter(|| is_prime_u64(black_box(n)))
    });
}

fn bench_prev_prime_below_pow2_table(c: &mut Criterion) {
    c.bench_function("prev_prime_below_pow2/k=32_shard_router", |b| {
        b.iter(|| prev_prime_below_pow2(black_box(32)))
    });
}

fn bench_next_prime_u64_arbitrary(c: &mut Criterion) {
    // Pick a value off the power-of-two grid so the fast path is missed
    // and the general MR descent is exercised.
    let n: u64 = 1_000_003_777;
    c.bench_function("next_prime_u64/arbitrary_~1e9", |b| {
        b.iter(|| next_prime_u64(black_box(n)))
    });
}

fn bench_next_prime_u64_2_pow_61(c: &mut Criterion) {
    // 2^61 hits the table fast path via the power-of-two check; subtract 1
    // to force the general MR descent against a worst-case-shaped input.
    let n: u64 = (1u64 << 61) - 1;
    c.bench_function("next_prime_u64/2^61_minus_1_general_path", |b| {
        b.iter(|| next_prime_u64(black_box(n)))
    });
}

criterion_group!(
    primality_benches,
    bench_is_prime_u64_worst_case,
    bench_prev_prime_below_pow2_table,
    bench_next_prime_u64_arbitrary,
    bench_next_prime_u64_2_pow_61
);
criterion_main!(primality_benches);
