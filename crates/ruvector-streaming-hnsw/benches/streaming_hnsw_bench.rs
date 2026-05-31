use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_streaming_hnsw::{FlatL2Index, StaticHnsw, StreamingHnsw, AnnIndex};

fn make_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0_f32, 1.0).unwrap();
    (0..n).map(|_| (0..dim).map(|_| dist.sample(&mut rng)).collect()).collect()
}

fn bench_search(c: &mut Criterion) {
    let dim = 128;
    let n = 2_000;
    let k = 10;
    let data = make_vecs(n, dim, 42);
    let queries = make_vecs(20, dim, 99);

    let mut flat = FlatL2Index::new(dim);
    for v in &data { flat.insert(v.clone()).unwrap(); }
    let static_idx = StaticHnsw::build(data.clone(), 16, 40).unwrap();
    let stream_idx = StreamingHnsw::from_vecs(data.clone(), 16, 40).unwrap();

    let mut g = c.benchmark_group("search_qps");
    for (name, idx) in &[
        ("FlatL2", &flat as &dyn AnnIndex),
        ("StaticHnsw", &static_idx as &dyn AnnIndex),
        ("StreamingHnsw", &stream_idx as &dyn AnnIndex),
    ] {
        g.bench_with_input(BenchmarkId::new("search", name), name, |b, _| {
            b.iter(|| {
                for q in &queries {
                    let _ = idx.search(q, k).unwrap_or_default();
                }
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
