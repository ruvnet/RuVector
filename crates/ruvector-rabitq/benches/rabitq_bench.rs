//! Criterion benchmarks for the RaBitQ distance kernels + search throughput.
//!
//! Two benchmark groups:
//!
//!   `distance_kernel` — per-pair cost of f32 dot, symmetric popcount,
//!                       symmetric estimator, and asymmetric estimator at
//!                       D ∈ {64, 128, 256, 512, 1024}.
//!
//!   `search_k10`      — end-to-end `search(..., 10)` at n ∈ {1 k, 10 k, 50 k}
//!                       for all four AnnIndex variants on the SAME clustered
//!                       dataset so the numbers are directly comparable.
//!
//! Run:  cargo bench -p ruvector-rabitq --bench rabitq_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};
use ruvector_rabitq::{
    index::{AnnIndex, FlatF32Index, RabitqAsymIndex, RabitqIndex, RabitqPlusIndex},
    quantize::BinaryCode,
    rotation::RandomRotation,
};

fn clustered(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::Rng as _;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroid = Uniform::new(-2.0f32, 2.0);
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..d).map(|_| centroid.sample(&mut rng)).collect())
        .collect();
    let noise = Normal::new(0.0f64, 0.6).unwrap();
    (0..n)
        .map(|_| {
            let c = &centroids[rng.gen_range(0..n_clusters)];
            c.iter()
                .map(|&x| x + noise.sample(&mut rng) as f32)
                .collect()
        })
        .collect()
}

fn bench_distance_kernels(c: &mut Criterion) {
    let mut g = c.benchmark_group("distance_kernel");
    for d in [64usize, 128, 256, 512, 1024] {
        let rot = RandomRotation::random(d, 42);
        let v1: Vec<f32> = (0..d).map(|i| (i as f32 * 0.03).sin()).collect();
        let v2: Vec<f32> = (0..d).map(|i| (i as f32 * 0.03).cos()).collect();

        // f32 dot product (baseline).
        g.bench_with_input(BenchmarkId::new("f32_dot", d), &d, |b, _| {
            b.iter(|| {
                let s: f32 = v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum();
                black_box(s)
            })
        });

        let n1: f32 = v1.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let n2: f32 = v2.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let u1: Vec<f32> = v1.iter().map(|&x| x / n1).collect();
        let u2: Vec<f32> = v2.iter().map(|&x| x / n2).collect();
        let r1 = rot.apply(&u1);
        let r2 = rot.apply(&u2);
        let code1 = BinaryCode::encode(&r1, n1);
        let code2 = BinaryCode::encode(&r2, n2);

        g.bench_with_input(BenchmarkId::new("masked_xnor_popcount", d), &d, |b, _| {
            b.iter(|| black_box(code1.masked_xnor_popcount(&code2)))
        });
        g.bench_with_input(BenchmarkId::new("sym_estimated_sq", d), &d, |b, _| {
            b.iter(|| black_box(code1.estimated_sq_distance(&code2)))
        });
        g.bench_with_input(BenchmarkId::new("asym_estimated_sq", d), &d, |b, _| {
            b.iter(|| black_box(code1.estimated_sq_distance_asymmetric(&r2, n2)))
        });
    }
    g.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut g = c.benchmark_group("search_k10");
    let d = 128;
    let data = clustered(50_000, d, 100, 1);
    let query = clustered(1, d, 100, 9)[0].clone();

    for n in [1_000usize, 10_000, 50_000] {
        let mut flat = FlatF32Index::new(d);
        let mut rq = RabitqIndex::new(d, 42);
        let mut rq_plus = RabitqPlusIndex::new(d, 42, 5);
        let mut rq_asym1 = RabitqAsymIndex::new(d, 42, 1);
        let mut rq_asym5 = RabitqAsymIndex::new(d, 42, 5);

        for (id, v) in data.iter().take(n).enumerate() {
            flat.add(id, v.clone()).unwrap();
            rq.add(id, v.clone()).unwrap();
            rq_plus.add(id, v.clone()).unwrap();
            rq_asym1.add(id, v.clone()).unwrap();
            rq_asym5.add(id, v.clone()).unwrap();
        }

        g.bench_with_input(BenchmarkId::new("FlatF32", n), &n, |b, _| {
            b.iter(|| black_box(flat.search(&query, 10).unwrap()))
        });
        g.bench_with_input(BenchmarkId::new("RaBitQ_sym", n), &n, |b, _| {
            b.iter(|| black_box(rq.search(&query, 10).unwrap()))
        });
        g.bench_with_input(BenchmarkId::new("RaBitQ_plus_x5", n), &n, |b, _| {
            b.iter(|| black_box(rq_plus.search(&query, 10).unwrap()))
        });
        g.bench_with_input(BenchmarkId::new("RaBitQ_asym_no_rerank", n), &n, |b, _| {
            b.iter(|| black_box(rq_asym1.search(&query, 10).unwrap()))
        });
        g.bench_with_input(BenchmarkId::new("RaBitQ_asym_x5", n), &n, |b, _| {
            b.iter(|| black_box(rq_asym5.search(&query, 10).unwrap()))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_distance_kernels, bench_search);
criterion_main!(benches);
