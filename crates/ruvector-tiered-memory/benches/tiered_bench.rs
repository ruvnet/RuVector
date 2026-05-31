use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ruvector_tiered_memory::{
    coherence_tiered::CoherenceTieredMemory, flat::FlatMemory, lru_tiered::LruTieredMemory,
    TieredMemoryStore,
};

fn make_corpus(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed.wrapping_add(1);
    (0..n)
        .map(|_| {
            (0..dims)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state >> 40) as f32 / (1u64 << 24) as f32
                })
                .collect()
        })
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let dims = 128;
    let n = 2_000;
    let k = 10;
    let corpus = make_corpus(n, dims, 42);
    let query = make_corpus(1, dims, 99)[0].clone();

    let mut group = c.benchmark_group("tiered_search");

    group.bench_function(BenchmarkId::new("flat", n), |b| {
        let mut store = FlatMemory::new(dims);
        for (i, v) in corpus.iter().enumerate() {
            store.insert(i as u64, v.clone());
        }
        b.iter(|| store.search(&query, k))
    });

    group.bench_function(BenchmarkId::new("lru_tiered", n), |b| {
        let mut store = LruTieredMemory::new(dims, n / 10, n / 3);
        for (i, v) in corpus.iter().enumerate() {
            store.insert(i as u64, v.clone());
        }
        b.iter(|| store.search(&query, k))
    });

    group.bench_function(BenchmarkId::new("coherence_tiered", n), |b| {
        let mut store = CoherenceTieredMemory::new(dims, 0.65, 0.25, 200);
        for (i, v) in corpus.iter().enumerate() {
            store.insert(i as u64, v.clone());
        }
        b.iter(|| store.search(&query, k))
    });

    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
