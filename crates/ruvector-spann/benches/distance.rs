use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_spann::distance::{cosine_distance, l2_squared};

const DIMENSIONS: [usize; 4] = [128, 384, 768, 1536];
const SEED: u64 = 0x5eed_5eed_5eed_5eed;

fn random_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn bench_l2_squared(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_squared");
    for &dim in DIMENSIONS.iter() {
        let mut rng = StdRng::seed_from_u64(SEED);
        let a = random_vector(&mut rng, dim);
        let b = random_vector(&mut rng, dim);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| black_box(l2_squared(black_box(&a), black_box(&b))));
        });
    }
    group.finish();
}

fn bench_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");
    for &dim in DIMENSIONS.iter() {
        let mut rng = StdRng::seed_from_u64(SEED);
        let a = random_vector(&mut rng, dim);
        let b = random_vector(&mut rng, dim);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| black_box(cosine_distance(black_box(&a), black_box(&b))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_l2_squared, bench_cosine_distance);
criterion_main!(benches);
