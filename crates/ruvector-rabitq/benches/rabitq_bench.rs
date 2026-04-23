use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_rabitq::{
    index::{AnnIndex, FlatF32Index, RabitqIndex, RabitqPlusIndex},
    quantize::BinaryCode,
    rotation::RandomRotation,
};

fn make_vecs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f64, 1.0).unwrap();
    (0..n)
        .map(|_| (0..d).map(|_| normal.sample(&mut rng) as f32).collect())
        .collect()
}

fn bench_distance_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_kernel");
    for d in [64usize, 128, 256, 512] {
        let rot = RandomRotation::random(d, 42);
        let v1: Vec<f32> = (0..d).map(|i| (i as f32).sin()).collect();
        let v2: Vec<f32> = (0..d).map(|i| (i as f32).cos()).collect();

        // f32 dot product (baseline).
        group.bench_with_input(BenchmarkId::new("f32_dot", d), &d, |b, _| {
            b.iter(|| {
                let s: f32 = v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum();
                black_box(s)
            })
        });

        // RaBitQ XNOR-popcount.
        let code1 = BinaryCode::encode(&rot.apply(&v1), 1.0);
        let code2 = BinaryCode::encode(&rot.apply(&v2), 1.0);
        group.bench_with_input(BenchmarkId::new("xnor_popcount", d), &d, |b, _| {
            b.iter(|| black_box(code1.xnor_popcount(&code2)))
        });

        // Full estimated distance.
        group.bench_with_input(BenchmarkId::new("estimated_sq_dist", d), &d, |b, _| {
            b.iter(|| black_box(code1.estimated_sq_distance(&code2)))
        });
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_k10");
    for n in [1_000usize, 10_000] {
        let d = 128;
        let data = make_vecs(n, d, 1);
        let query = make_vecs(1, d, 9)[0].clone();

        let mut f32_idx = FlatF32Index::new(d);
        let mut rq_idx = RabitqIndex::new(d, 42);
        let mut rq_plus = RabitqPlusIndex::new(d, 42, 3);

        for (id, v) in data.iter().enumerate() {
            f32_idx.add(id, v.clone()).unwrap();
            rq_idx.add(id, v.clone()).unwrap();
            rq_plus.add(id, v.clone()).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("FlatF32", n), &n, |b, _| {
            b.iter(|| black_box(f32_idx.search(&query, 10).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("RaBitQ", n), &n, |b, _| {
            b.iter(|| black_box(rq_idx.search(&query, 10).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("RaBitQ+x3", n), &n, |b, _| {
            b.iter(|| black_box(rq_plus.search(&query, 10).unwrap()))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_distance_kernels, bench_search);
criterion_main!(benches);
