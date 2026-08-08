//! Kernel microbenchmarks (ADR-296 phase 3 / refinements §3).
//!
//! Compares, per dimension: the scalar oracle, the shipped AVX2 kernel
//! (abs/sign + maddubs), and the previous widen-to-i16 AVX2 sequence kept
//! here as a baseline so kernel changes stay *measured*, not assumed.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvector_turboquant::simd;
use ruvector_turboquant::tables::LEVELS_I8;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n).map(|_| (splitmix(&mut s) & 0xFF) as u8).collect()
}

/// Previous AVX2 kernel (4×cvtepi8_epi16 + 2×madd) — baseline for the
/// shipped abs/sign+maddubs variant.
#[cfg(target_arch = "x86_64")]
mod widen_baseline {
    use super::LEVELS_I8;
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn level_table() -> __m256i {
        let t = _mm_loadu_si128(LEVELS_I8.as_ptr() as *const __m128i);
        _mm256_broadcastsi128_si256(t)
    }

    #[inline]
    unsafe fn madd_i8_widen(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
        let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
        let b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b));
        let b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));
        let p = _mm256_add_epi32(_mm256_madd_epi16(a_lo, b_lo), _mm256_madd_epi16(a_hi, b_hi));
        _mm256_add_epi32(acc, p)
    }

    #[inline]
    unsafe fn hsum_i32(v: __m256i) -> i32 {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        _mm_cvtsi128_si32(s)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i8_nibble_widen(nibbles: &[u8], q: &[u8], dim: usize) -> i32 {
        let half = dim / 2;
        let table = level_table();
        let mask = _mm256_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_si256();
        let chunks = half / 32;
        for c in 0..chunks {
            let i = c * 32;
            let packed = _mm256_loadu_si256(nibbles.as_ptr().add(i) as *const __m256i);
            let lo_lev = _mm256_shuffle_epi8(table, _mm256_and_si256(packed, mask));
            let q_lo = _mm256_loadu_si256(q.as_ptr().add(i) as *const __m256i);
            acc = madd_i8_widen(acc, lo_lev, q_lo);
            let hi_lev =
                _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask));
            let q_hi = _mm256_loadu_si256(q.as_ptr().add(half + i) as *const __m256i);
            acc = madd_i8_widen(acc, hi_lev, q_hi);
        }
        let mut total = hsum_i32(acc);
        for i in chunks * 32..half {
            let lo = LEVELS_I8[(nibbles[i] & 0x0F) as usize] as i32;
            let hi = LEVELS_I8[(nibbles[i] >> 4) as usize] as i32;
            total += (q[i] as i8 as i32) * lo + (q[i + half] as i8 as i32) * hi;
        }
        total
    }
}

fn bench_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_i8_nibble");
    for &dim in &[128usize, 768, 1536] {
        let nibbles = random_bytes(dim / 2, 1);
        let q = random_bytes(dim, 2);
        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", dim), &dim, |b, &dim| {
            b.iter(|| simd::dot_i8_nibble(black_box(&nibbles), black_box(&q), dim))
        });

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && dim >= 64 {
            group.bench_with_input(
                BenchmarkId::new("avx2_widen_baseline", dim),
                &dim,
                |b, &dim| {
                    b.iter(|| unsafe {
                        widen_baseline::dot_i8_nibble_widen(black_box(&nibbles), black_box(&q), dim)
                    })
                },
            );
        }
    }
    group.finish();

    let mut group = c.benchmark_group("dot_nibble_nibble");
    for &dim in &[128usize, 768, 1536] {
        let a = random_bytes(dim / 2, 3);
        let b_code = random_bytes(dim / 2, 4);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("dispatch", dim), &dim, |b, &dim| {
            b.iter(|| simd::dot_nibble_nibble(black_box(&a), black_box(&b_code), dim))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
