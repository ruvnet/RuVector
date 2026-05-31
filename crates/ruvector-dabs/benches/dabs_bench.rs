use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_dabs::{DabsIndex, SearchMode};

fn gen_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();
    (0..n).map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect()).collect()
}

fn bench_search(c: &mut Criterion) {
    let data = gen_data(5_000, 128, 42);
    let queries = gen_data(50, 128, 99);
    let index = DabsIndex::build(data, 16).unwrap();

    let mut group = c.benchmark_group("search_5k_d128");

    group.bench_function("flat", |b| {
        b.iter(|| {
            for q in &queries {
                let _ = black_box(index.search(q, 10, SearchMode::Flat).unwrap());
            }
        })
    });

    for ef in [32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("fixed_ef", ef), &ef, |b, &ef| {
            b.iter(|| {
                for q in &queries {
                    let _ = black_box(index.search(q, 10, SearchMode::FixedEf { ef }).unwrap());
                }
            })
        });
    }

    for gamma in [0.1_f32, 0.5, 1.0] {
        group.bench_with_input(
            BenchmarkId::new("dabs_gamma", format!("{gamma:.1}")),
            &gamma,
            |b, &gamma| {
                b.iter(|| {
                    for q in &queries {
                        let _ =
                            black_box(index.search(q, 10, SearchMode::Dabs { gamma }).unwrap());
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
