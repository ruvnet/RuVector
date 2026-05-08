use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ruvector_lvq::{FlatF32, FlatLvqIndex};

fn random_dataset(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen_range(-1.0_f32..1.0)).collect()
}

fn bench_search(c: &mut Criterion) {
    let dim = 128;
    let n = 20_000;
    let data = random_dataset(n, dim, 7);
    let queries = random_dataset(64, dim, 9);

    let mut gt = FlatF32::new(dim);
    for v in data.chunks_exact(dim) {
        gt.push(v).unwrap();
    }

    let mut lvq8 = FlatLvqIndex::new_lvq8(dim);
    lvq8.extend_from_flat(&data).unwrap();

    let mut lvq8x8 = FlatLvqIndex::new_lvq8x8(dim);
    lvq8x8.extend_from_flat(&data).unwrap();

    let q0: Vec<f32> = queries[..dim].to_vec();

    c.bench_function("flat_f32_l2_n20k_d128_k10", |b| {
        b.iter_batched(
            || q0.clone(),
            |q| {
                let h = gt.search_l2(black_box(&q), 10).unwrap();
                black_box(h);
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("lvq8_l2_n20k_d128_k10", |b| {
        b.iter_batched(
            || q0.clone(),
            |q| {
                let h = lvq8.search_l2(black_box(&q), 10).unwrap();
                black_box(h);
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("lvq8x8_full_l2_n20k_d128_k10", |b| {
        b.iter_batched(
            || q0.clone(),
            |q| {
                let h = lvq8x8.search_l2(black_box(&q), 10).unwrap();
                black_box(h);
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("lvq8x8_rerank10x_l2_n20k_d128_k10", |b| {
        b.iter_batched(
            || q0.clone(),
            |q| {
                let h = lvq8x8.search_l2_reranked(black_box(&q), 10, 100).unwrap();
                black_box(h);
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
