use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_memory_compaction::{
    CompactionConfig, DecayScoreCompactor, GreedyAgeCompactor, MemoryCompactor, MemoryEntry,
    MemoryStore, MinCutCompactor,
};

fn make_store(n: usize, dim: usize, seed: u64) -> MemoryStore {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0f32, 1.0).unwrap();
    let now_ms: u64 = 1_748_736_000_000;
    let mut store = MemoryStore::new();
    for i in 0..n {
        let vec: Vec<f32> = (0..dim).map(|_| dist.sample(&mut rng)).collect();
        let age = ((i as f64 / n as f64).powi(2) * now_ms as f64) as u64;
        store.insert(MemoryEntry::new(i as u64, vec, now_ms - age));
    }
    store
}

fn bench_compactors(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_compaction");
    group.sample_size(20);

    for n in [500usize, 2_000] {
        let dim = 128;
        let store = make_store(n, dim, 42);
        let config = CompactionConfig {
            retain_fraction: 0.5,
            similarity_threshold: 0.0,
            top_k_neighbors: 8,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("GreedyAge", n), &n, |b, _| {
            b.iter(|| GreedyAgeCompactor.compact(black_box(&store), black_box(&config)))
        });

        group.bench_with_input(BenchmarkId::new("DecayScore", n), &n, |b, _| {
            b.iter(|| DecayScoreCompactor.compact(black_box(&store), black_box(&config)))
        });

        group.bench_with_input(BenchmarkId::new("MinCutGraph", n), &n, |b, _| {
            b.iter(|| MinCutCompactor.compact(black_box(&store), black_box(&config)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_compactors);
criterion_main!(benches);
