use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use ruvector_acorn::{AcornIndex1, AcornIndexGamma, FilteredIndex, FlatFilteredIndex};

fn make_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    // `StdRng` is always available; `SmallRng` is feature-gated and not
    // enabled in the workspace, which broke this bench when the gate flipped.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();
    (0..n)
        .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

fn bench_search(c: &mut Criterion) {
    const N: usize = 2_000;
    const DIM: usize = 64;
    const K: usize = 10;

    let data = make_data(N, DIM, 42);
    let queries = make_data(100, DIM, 99);

    let flat = FlatFilteredIndex::build(data.clone()).unwrap();
    let acorn1 = AcornIndex1::build(data.clone()).unwrap();
    let acorng = AcornIndexGamma::build(data.clone()).unwrap();

    let mut g = c.benchmark_group("filtered_search_sel10pct");

    for (name, idx) in [
        ("flat-baseline", &flat as &dyn FilteredIndex),
        ("acorn1", &acorn1),
        ("acorn-gamma2", &acorng),
    ] {
        g.bench_with_input(BenchmarkId::new(name, N), &(), |b, _| {
            b.iter(|| {
                for q in &queries {
                    black_box(
                        idx.search(q, K, &|id: u32| id % 10 == 0)
                            .unwrap_or_default(),
                    );
                }
            });
        });
    }

    g.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
