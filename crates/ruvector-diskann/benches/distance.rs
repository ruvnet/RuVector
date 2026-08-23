use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_diskann::distance::{
    inner_product, l2_squared, pq_asymmetric_distance, scalar_l2_squared,
};

const DIMS: [usize; 4] = [128, 384, 768, 1536];
const SEED: u64 = 0x5EED_1234_ABCD_EF01;
const PQ_CODEBOOK_SIZE: usize = 256;
const PQ_DSUB: usize = 8;
// (dim, m) pairs at dsub = 8, matching the crate's exercised PQ shape (D=32, M=4).
const PQ_DIM_M: [(usize, usize); 4] = [(128, 16), (384, 48), (768, 96), (1536, 192)];

fn random_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn bench_l2_squared(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_squared");
    let mut rng = StdRng::seed_from_u64(SEED);

    for &dim in DIMS.iter() {
        let a = random_vector(&mut rng, dim);
        let b = random_vector(&mut rng, dim);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| black_box(l2_squared(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

fn bench_scalar_l2_squared(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_l2_squared");
    let mut rng = StdRng::seed_from_u64(SEED);

    for &dim in DIMS.iter() {
        let a = random_vector(&mut rng, dim);
        let b = random_vector(&mut rng, dim);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| black_box(scalar_l2_squared(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_product");
    let mut rng = StdRng::seed_from_u64(SEED);

    for &dim in DIMS.iter() {
        let a = random_vector(&mut rng, dim);
        let b = random_vector(&mut rng, dim);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| black_box(inner_product(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

fn bench_pq_asymmetric_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_asymmetric_distance");
    let mut rng = StdRng::seed_from_u64(SEED);

    for &(dim, m) in PQ_DIM_M.iter() {
        debug_assert_eq!(dim, m * PQ_DSUB);
        let codes: Vec<u8> = (0..m)
            .map(|_| rng.gen_range(0..PQ_CODEBOOK_SIZE) as u8)
            .collect();
        let table: Vec<f32> = (0..m * PQ_CODEBOOK_SIZE)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect();

        let label = format!("d{dim}_m{m}");
        group.bench_with_input(BenchmarkId::from_parameter(&label), &m, |bencher, _| {
            bencher.iter(|| {
                black_box(pq_asymmetric_distance(
                    black_box(&codes),
                    black_box(&table),
                    black_box(PQ_CODEBOOK_SIZE),
                ))
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_l2_squared,
    bench_scalar_l2_squared,
    bench_inner_product,
    bench_pq_asymmetric_distance
);
criterion_main!(benches);
