use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_ivfpq::{IvfPqConfig, IvfPqIndex};

fn gen_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> =
        (0..20).map(|_| (0..dim).map(|_| rng.gen_range(-10.0_f32..10.0)).collect()).collect();
    (0..n)
        .map(|_| {
            let c = rng.gen_range(0..20usize);
            (0..dim).map(|d| centers[c][d] + (rng.gen::<f32>() - 0.5)).collect()
        })
        .collect()
}

fn make_index(nlist: usize, m: usize) -> IvfPqIndex {
    let corpus = gen_data(10_000, 128, 0);
    let cfg = IvfPqConfig {
        nlist,
        m,
        ksub: 256,
        nprobe: 4,
        rerank_k: 50,
        max_iter: 20,
    };
    let mut idx = IvfPqIndex::new(cfg);
    idx.train(&corpus);
    idx.add(&corpus);
    idx
}

fn bench_search_nprobe(c: &mut Criterion) {
    let idx = make_index(64, 8);
    let queries = gen_data(100, 128, 99);

    let mut group = c.benchmark_group("search_nprobe");
    for &nprobe in &[1usize, 4, 16] {
        let rerank_k = nprobe * 10 + 10;
        group.bench_with_input(BenchmarkId::new("nprobe", nprobe), &nprobe, |b, &np| {
            b.iter(|| {
                for q in &queries {
                    black_box(idx.search(black_box(q), 10, np, rerank_k));
                }
            });
        });
    }
    group.finish();
}

fn bench_search_m(c: &mut Criterion) {
    let queries = gen_data(100, 128, 99);
    let mut group = c.benchmark_group("search_subspaces_m");
    for &m in &[4usize, 8, 16] {
        let idx = make_index(64, m);
        group.bench_with_input(BenchmarkId::new("m", m), &m, |b, _| {
            b.iter(|| {
                for q in &queries {
                    black_box(idx.search(black_box(q), 10, 4, 50));
                }
            });
        });
    }
    group.finish();
}

fn bench_train(c: &mut Criterion) {
    // Small corpus so criterion completes within its default time budget.
    // Production training cost scales as O(N × nlist × max_iter) for IVF
    // plus O(N × m × ksub × max_iter) for PQ — both dominated by k-means.
    let corpus = gen_data(1_000, 128, 42);
    c.bench_function("train_1k_nlist16_m4_ksub32", |b| {
        b.iter(|| {
            let cfg = IvfPqConfig {
                nlist: 16,
                m: 4,
                ksub: 32,
                nprobe: 4,
                rerank_k: 50,
                max_iter: 10,
            };
            let mut idx = IvfPqIndex::new(cfg);
            idx.train(black_box(&corpus));
        });
    });
}

criterion_group!(benches, bench_search_nprobe, bench_search_m, bench_train);
criterion_main!(benches);
