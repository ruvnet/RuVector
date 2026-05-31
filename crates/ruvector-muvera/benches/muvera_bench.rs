use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_muvera::{BruteForceMaxSim, FdeEncoder, FlatFdeIndex, HnswFdeIndex, MultiVecIndex};
use std::sync::Arc;

fn make_docs(n: usize, tokens: usize, dim: usize) -> Vec<Vec<Vec<f32>>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();
    (0..n)
        .map(|_| {
            (0..tokens)
                .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
                .collect()
        })
        .collect()
}

fn bench_query_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("muvera_query");
    let dim = 64usize;
    let num_reps = 16usize;
    let tokens_doc = 20usize;
    let tokens_q = 8usize;

    for &n_docs in &[500usize, 2_000, 5_000] {
        let docs = make_docs(n_docs, tokens_doc, dim);
        let queries = make_docs(50, tokens_q, dim);
        let enc = Arc::new(FdeEncoder::new(num_reps, dim, 7).unwrap());

        let bf = BruteForceMaxSim::build(docs.clone(), enc.clone()).unwrap();
        let flat = FlatFdeIndex::build(docs.clone(), enc.clone()).unwrap();
        let hnsw = HnswFdeIndex::build(docs.clone(), enc.clone()).unwrap();

        group.bench_with_input(BenchmarkId::new("BruteForce", n_docs), &n_docs, |b, _| {
            b.iter(|| {
                for q in &queries {
                    bf.search(q, 10).unwrap();
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("FlatFDE", n_docs), &n_docs, |b, _| {
            b.iter(|| {
                for q in &queries {
                    flat.search(q, 10).unwrap();
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("HnswFDE", n_docs), &n_docs, |b, _| {
            b.iter(|| {
                for q in &queries {
                    hnsw.search(q, 10).unwrap();
                }
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_query_throughput);
criterion_main!(benches);
