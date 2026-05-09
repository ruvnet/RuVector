use criterion::{Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand_distr::Normal;
use ruvector_avq::{AnisotropicPq, Encoder, ProductQuantizer, Scorer};

const N: usize = 4_000;
const DIM: usize = 64;
const M: usize = 16;
const K: usize = 256;

fn make(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    let mut v = vec![0.0f32; n * dim];
    for x in v.iter_mut() {
        *x = normal.sample(&mut rng);
    }
    // l2-normalize
    for i in 0..n {
        let mut s = 0.0f32;
        for j in 0..dim {
            s += v[i * dim + j] * v[i * dim + j];
        }
        let inv = 1.0 / s.sqrt().max(1e-12);
        for j in 0..dim {
            v[i * dim + j] *= inv;
        }
    }
    v
}

fn bench_score(c: &mut Criterion) {
    let db = make(N, DIM, 5);
    let q = make(1, DIM, 9);
    let pq = ProductQuantizer::fit(&db, DIM, M, K, 7).unwrap();
    let avq = AnisotropicPq::fit(&db, DIM, M, K, 4.0, 7).unwrap();
    let mut pq_codes = vec![0u8; N * pq.code_size()];
    let mut avq_codes = vec![0u8; N * avq.code_size()];
    pq.encode(&db, &mut pq_codes);
    avq.encode(&db, &mut avq_codes);
    let mut out = vec![0.0f32; N];

    c.bench_function("pq_score_4k", |b| {
        b.iter(|| pq.score_ip(&q, &pq_codes, &mut out));
    });
    c.bench_function("avq_score_4k", |b| {
        b.iter(|| avq.score_ip(&q, &avq_codes, &mut out));
    });
}

criterion_group!(benches, bench_score);
criterion_main!(benches);
