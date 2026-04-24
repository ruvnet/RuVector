use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand_distr::Normal;
use ruvector_acorn::{AcornConfig, AcornIndex, SearchVariant};

const DIM: usize = 128;
const M: usize = 16;
const EF: usize = 64;
const K: usize = 10;

fn build_index(n: usize, gamma: usize) -> (AcornIndex, Vec<u32>) {
    let mut rng = StdRng::seed_from_u64(99);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    let cfg = AcornConfig {
        dim: DIM,
        m: M,
        gamma,
        ef_construction: 80,
    };
    let mut idx = AcornIndex::new(cfg);
    let tags: Vec<u32> = (0..n as u32).collect();
    for i in 0..n as u32 {
        let v: Vec<f32> = (0..DIM).map(|_| normal.sample(&mut rng)).collect();
        idx.insert(i, v).unwrap();
    }
    idx.build_compression();
    (idx, tags)
}

fn bench_search(c: &mut Criterion) {
    let n = 5_000;
    let mut rng = StdRng::seed_from_u64(7);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    let query: Vec<f32> = (0..DIM).map(|_| normal.sample(&mut rng)).collect();

    let (idx1, tags1) = build_index(n, 1);
    let (idx2, tags2) = build_index(n, 2);

    let mut group = c.benchmark_group("filtered_anns_select10pct");

    let threshold = (n / 10) as u32; // 10 % selectivity

    group.bench_function(BenchmarkId::new("PostFilter", n), |b| {
        b.iter(|| {
            idx1.search(
                black_box(&query),
                K,
                EF * 4,
                |id| tags1[id as usize] < threshold,
                SearchVariant::PostFilter,
            )
            .unwrap()
        })
    });

    group.bench_function(BenchmarkId::new("ACORN-1", n), |b| {
        b.iter(|| {
            idx1.search(
                black_box(&query),
                K,
                EF,
                |id| tags1[id as usize] < threshold,
                SearchVariant::Acorn1,
            )
            .unwrap()
        })
    });

    group.bench_function(BenchmarkId::new("ACORN-gamma2", n), |b| {
        b.iter(|| {
            idx2.search(
                black_box(&query),
                K,
                EF,
                |id| tags2[id as usize] < threshold,
                SearchVariant::AcornGamma,
            )
            .unwrap()
        })
    });

    group.finish();

    // Tight filter (1 %)
    let threshold_1pct = (n / 100) as u32;
    let mut group2 = c.benchmark_group("filtered_anns_select1pct");

    group2.bench_function(BenchmarkId::new("PostFilter", n), |b| {
        b.iter(|| {
            idx1.search(
                black_box(&query),
                K,
                EF * 4,
                |id| tags1[id as usize] < threshold_1pct,
                SearchVariant::PostFilter,
            )
            .unwrap_or_default()
        })
    });

    group2.bench_function(BenchmarkId::new("ACORN-gamma2", n), |b| {
        b.iter(|| {
            idx2.search(
                black_box(&query),
                K,
                EF,
                |id| tags2[id as usize] < threshold_1pct,
                SearchVariant::AcornGamma,
            )
            .unwrap_or_default()
        })
    });

    group2.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
