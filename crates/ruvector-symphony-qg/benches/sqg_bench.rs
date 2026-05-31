use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_symphony_qg::{SqgConfig, SqgIndex};

fn random_vecs(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0f32, 1.0).unwrap();
    (0..n * d).map(|_| dist.sample(&mut rng)).collect()
}

fn bench_search(c: &mut Criterion) {
    let n = 2_000;
    let d = 128;
    let data = random_vecs(n, d, 42);
    let query = random_vecs(1, d, 99);
    let cfg = SqgConfig { pq_subspaces: 8, pq_iters: 20, m_neighbors: 16 };
    let idx = SqgIndex::build(&data, d, cfg).unwrap();

    let mut group = c.benchmark_group("search_n2000_d128");

    group.bench_function(BenchmarkId::new("flat_exact", "k10"), |b| {
        b.iter(|| black_box(idx.flat_exact(black_box(&query), 10)));
    });

    group.bench_function(BenchmarkId::new("sqg_fastscan_ef60", "k10"), |b| {
        b.iter(|| black_box(idx.sqg_fastscan(black_box(&query), 10, 60)));
    });

    group.bench_function(BenchmarkId::new("sqg_rerank_ef60", "k10"), |b| {
        b.iter(|| black_box(idx.sqg_rerank(black_box(&query), 10, 60, 40)));
    });

    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
