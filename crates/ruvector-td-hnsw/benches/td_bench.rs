use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{Rng, SeedableRng};
use ruvector_td_hnsw::{DecayConfig, IndexVariant, TdIndex};

fn make_data(n: usize, dims: usize, now: u64) -> Vec<(u64, Vec<f32>, u64)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(777);
    (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>()).collect();
            let ts = if i % 5 == 0 { now - 500 } else { now - 86400 };
            (i as u64, v, ts)
        })
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let now = 100_000u64;
    let dims = 128;
    let data = make_data(5_000, dims, now);
    let mut rng = rand::rngs::StdRng::seed_from_u64(888);
    let q: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>()).collect();

    let mut group = c.benchmark_group("td_search");

    let variants: Vec<(IndexVariant, DecayConfig, &str)> = vec![
        (IndexVariant::Baseline, DecayConfig::no_decay(), "baseline"),
        (
            IndexVariant::TemporalDecay,
            DecayConfig::standard(3.0, 3600.0),
            "temporal-decay",
        ),
        (
            IndexVariant::CoherenceGated,
            DecayConfig::with_gate(3.0, 3600.0, 43200.0, 1.5),
            "coherence-gated",
        ),
    ];

    for (var, cfg, name) in variants {
        let mut idx = TdIndex::new(var, cfg);
        for (id, v, ts) in &data {
            idx.insert(*id, v.clone(), *ts);
        }
        group.bench_with_input(BenchmarkId::new("search", name), &idx, |b, index| {
            b.iter(|| index.search(&q, 10, now));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
