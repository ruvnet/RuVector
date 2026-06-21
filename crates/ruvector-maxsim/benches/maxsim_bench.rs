//! Criterion benchmarks for MaxSim index variants.
//!
//! Run: cargo bench -p ruvector-maxsim

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{Rng, SeedableRng};
use ruvector_maxsim::{
    types::{DocId, MultiVecDoc, MultiVecQuery},
    BucketMaxSim, FlatMaxSim, HnswMaxSim, MultiVecIndex,
};

const DIMS: usize = 64;
const TOKENS_PER_DOC: usize = 6;
const TOKENS_PER_QUERY: usize = 3;
const K: usize = 10;

fn random_unit_vec(rng: &mut impl Rng, dims: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let len = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if len > 1e-9 {
        v.iter_mut().for_each(|x| *x /= len);
    }
    v
}

fn gen_docs(n: usize, dims: usize, tpd: usize) -> Vec<MultiVecDoc> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    (0..n)
        .map(|i| MultiVecDoc {
            id: DocId(i as u64),
            vecs: (0..tpd).map(|_| random_unit_vec(&mut rng, dims)).collect(),
        })
        .collect()
}

fn gen_queries(n: usize, dims: usize, tpq: usize) -> Vec<MultiVecQuery> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(99);
    (0..n)
        .map(|_| MultiVecQuery {
            vecs: (0..tpq).map(|_| random_unit_vec(&mut rng, dims)).collect(),
        })
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxsim_search");
    group.sample_size(20);

    for n_docs in [500_usize, 2_000] {
        let docs = gen_docs(n_docs, DIMS, TOKENS_PER_DOC);
        let queries = gen_queries(50, DIMS, TOKENS_PER_QUERY);

        // FlatMaxSim
        let mut flat = FlatMaxSim::new(DIMS);
        for d in &docs {
            flat.add(d.clone()).unwrap();
        }
        group.bench_with_input(BenchmarkId::new("FlatMaxSim", n_docs), &n_docs, |b, _| {
            b.iter(|| {
                for q in &queries {
                    black_box(flat.search(black_box(q), K).unwrap());
                }
            });
        });

        // BucketMaxSim (oversampling=40)
        let mut bucket = BucketMaxSim::new(DIMS, 40);
        for d in &docs {
            bucket.add(d.clone()).unwrap();
        }
        group.bench_with_input(BenchmarkId::new("BucketMaxSim", n_docs), &n_docs, |b, _| {
            b.iter(|| {
                for q in &queries {
                    black_box(bucket.search(black_box(q), K).unwrap());
                }
            });
        });

        // HnswMaxSim
        let mut hnsw = HnswMaxSim::new(DIMS, 32);
        for d in &docs {
            hnsw.add(d.clone()).unwrap();
        }
        group.bench_with_input(BenchmarkId::new("HnswMaxSim", n_docs), &n_docs, |b, _| {
            b.iter(|| {
                for q in &queries {
                    black_box(hnsw.search(black_box(q), K).unwrap());
                }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
