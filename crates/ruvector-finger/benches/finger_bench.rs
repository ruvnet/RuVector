use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_finger::{FingerIndex, FlatGraph};

const DIM: usize = 128;
const M: usize = 16;
const K: usize = 10;
const EF: usize = 100;

fn gen_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| normal.sample_iter(&mut rng).take(dim).collect()).collect()
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("finger_search");
    group.sample_size(50);

    for n in [1_000usize, 5_000] {
        let corpus = gen_data(n, DIM, 42);
        let query: Vec<f32> = gen_data(1, DIM, 1)[0].clone();

        let graph = FlatGraph::build(&corpus, M).unwrap();
        let exact_idx = FingerIndex::exact(&graph);
        let finger_k4 = FingerIndex::finger_k4(&graph).unwrap();
        let finger_k8 = FingerIndex::finger_k8(&graph).unwrap();

        group.bench_with_input(
            BenchmarkId::new("exact", n),
            &n,
            |b, _| b.iter(|| exact_idx.search(&query, K, EF).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("finger-k4", n),
            &n,
            |b, _| b.iter(|| finger_k4.search(&query, K, EF).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("finger-k8", n),
            &n,
            |b, _| b.iter(|| finger_k8.search(&query, K, EF).unwrap()),
        );
    }

    group.finish();
}

fn bench_basis_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("finger_basis_build");
    group.sample_size(20);

    for n in [500usize, 2_000] {
        let corpus = gen_data(n, DIM, 7);
        let graph = FlatGraph::build(&corpus, M).unwrap();

        group.bench_with_input(
            BenchmarkId::new("build-k4", n),
            &n,
            |b, _| b.iter(|| FingerIndex::finger_k4(&graph).unwrap()),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_search, bench_basis_build);
criterion_main!(benches);
