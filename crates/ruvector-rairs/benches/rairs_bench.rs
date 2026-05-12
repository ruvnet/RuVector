//! Criterion micro-benchmarks for RAIRS IVF kernels.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_rairs::{AnnIndex, IvfFlat, RairsSeil, RairsStrict};

const DIM: usize = 128;
const N: usize = 2_000;
const NCLUSTERS: usize = 32;
const SEED: u64 = 99;

fn corpus(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..DIM).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let vecs = corpus(N, SEED);
    let query: Vec<f32> = vecs[0].clone();

    let mut ivf = IvfFlat::new(DIM, NCLUSTERS, 20, SEED);
    ivf.train(&vecs).unwrap();
    ivf.add(&vecs).unwrap();

    let mut strict = RairsStrict::new(DIM, NCLUSTERS, 20, SEED, 1.0);
    strict.train(&vecs).unwrap();
    strict.add(&vecs).unwrap();

    let mut seil = RairsSeil::new(DIM, NCLUSTERS, 20, SEED, 1.0);
    seil.train(&vecs).unwrap();
    seil.add(&vecs).unwrap();

    let mut g = c.benchmark_group("search_nprobe16");
    g.throughput(Throughput::Elements(1));

    g.bench_function("ivf_flat", |b| {
        b.iter(|| ivf.search(&query, 10, 16).unwrap())
    });
    g.bench_function("rairs_strict", |b| {
        b.iter(|| strict.search(&query, 10, 16).unwrap())
    });
    g.bench_function("rairs_seil", |b| {
        b.iter(|| seil.search(&query, 10, 16).unwrap())
    });
    g.finish();

    let mut g2 = c.benchmark_group("search_nprobe_sweep");
    g2.throughput(Throughput::Elements(1));
    for &np in &[1usize, 4, 16, 32] {
        g2.bench_with_input(BenchmarkId::new("ivf_flat", np), &np, |b, &np| {
            b.iter(|| ivf.search(&query, 10, np).unwrap())
        });
        g2.bench_with_input(BenchmarkId::new("rairs_seil", np), &np, |b, &np| {
            b.iter(|| seil.search(&query, 10, np).unwrap())
        });
    }
    g2.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
