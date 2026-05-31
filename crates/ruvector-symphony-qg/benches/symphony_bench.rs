use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use ruvector_symphony_qg::{
    codes::{batch_asym_l2, encode, packed_bytes, QueryProjection},
    graph::l2_sq,
    rotation::random_orthogonal,
};

fn gaussian_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = Normal::new(0.0f32, 1.0).unwrap();
    (0..dim).map(|_| n.sample(&mut rng)).collect()
}

fn bench_distance_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_kernels");

    for dim in [64usize, 128, 256] {
        let r = 32; // R neighbors per hop

        let q = gaussian_vec(dim, 1);
        let xs: Vec<Vec<f32>> = (0..r).map(|i| gaussian_vec(dim, i as u64 + 100)).collect();

        // Build batch code block
        let rot = random_orthogonal(dim, 42);
        let q_rot: Vec<f32> = (0..dim)
            .map(|i| rot[i * dim..i * dim + dim].iter().zip(q.iter()).map(|(a, b)| a * b).sum())
            .collect();

        let nbytes = packed_bytes(dim);
        let mut codes_block = vec![0u8; r * nbytes];
        let mut norms = vec![0.0f32; r];
        for (j, x) in xs.iter().enumerate() {
            let x_rot: Vec<f32> = (0..dim)
                .map(|i| rot[i * dim..i * dim + dim].iter().zip(x.iter()).map(|(a, b)| a * b).sum())
                .collect();
            let (code, norm) = encode(&x_rot);
            codes_block[j * nbytes..(j + 1) * nbytes].copy_from_slice(&code);
            norms[j] = norm;
        }
        let qp = QueryProjection::new(q_rot);
        let norm_q_sq: f32 = q.iter().map(|v| v * v).sum();

        // 1. Exact L2: R individual dot products
        group.bench_with_input(
            BenchmarkId::new("exact_l2_r32", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for x in &xs {
                        sum += l2_sq(black_box(&q), black_box(x));
                    }
                    black_box(sum)
                })
            },
        );

        // 2. Batch asymmetric (SymphonyQG FastScan)
        group.bench_with_input(
            BenchmarkId::new("batch_asym_r32", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    black_box(batch_asym_l2(
                        black_box(&qp),
                        black_box(&codes_block),
                        black_box(&norms),
                        norm_q_sq,
                    ))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_distance_kernels);
criterion_main!(benches);
