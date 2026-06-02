use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_mincut_memory::{AgeEvict, CoherenceEvict, Compactor, MemoryStore, MinCutEvict};

fn gen_store(n: usize, dims: usize, clusters: usize, threshold: f32, seed: u64) -> MemoryStore {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 0.25).unwrap();
    let centroids: Vec<Vec<f32>> = (0..clusters)
        .map(|_| {
            let raw: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            raw.iter().map(|x| x / norm).collect()
        })
        .collect();

    let mut store = MemoryStore::new(dims, threshold);
    for ts in 0..n {
        let c = rng.gen_range(0..clusters);
        let mut v: Vec<f32> = centroids[c]
            .iter()
            .map(|x| x + normal.sample(&mut rng))
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.iter_mut().for_each(|x| *x /= norm);
        store.insert(v, ts as u64);
    }
    store
}

fn bench_compaction(c: &mut Criterion) {
    let dims = 32usize;
    let threshold = 0.4f32;
    let clusters = 6usize;

    let strategies: Vec<(&str, Box<dyn Compactor>)> = vec![
        ("AgeEvict", Box::new(AgeEvict)),
        ("CoherenceEvict", Box::new(CoherenceEvict)),
        ("MinCutEvict", Box::new(MinCutEvict)),
    ];

    let mut group = c.benchmark_group("compaction");

    for n in [100usize, 300usize] {
        let target = n / 2;
        for (name, strat) in &strategies {
            group.bench_with_input(BenchmarkId::new(*name, n), &n, |b, &n| {
                b.iter_batched(
                    || gen_store(n, dims, clusters, threshold, 42),
                    |mut store| strat.compact(&mut store, target),
                    criterion::BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_compaction);
criterion_main!(benches);
