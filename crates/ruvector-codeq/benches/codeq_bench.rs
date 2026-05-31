use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_codeq::{CoDEQIndex, recall_at_k};
use ruvector_codeq::pq_baseline::{FlatL2IndexCoDEQ, StaticPqIndex};

fn make_vecs(n: usize, dim: usize, seed: u64) -> Vec<(u64, Vec<f32>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0_f32, 1.0).unwrap();
    (0..n as u64)
        .map(|id| (id, (0..dim).map(|_| dist.sample(&mut rng)).collect()))
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let dim = 128;
    let n = 2_000;
    let k = 10;
    let data = make_vecs(n, dim, 42);
    let queries = make_vecs(20, dim, 99);

    let flat = FlatL2IndexCoDEQ::from_vecs(data.clone()).unwrap();
    let pq   = StaticPqIndex::build(&data, 8, 64, 10).unwrap();
    let codeq = CoDEQIndex::from_vecs(&data, 8, 42).unwrap();

    let mut g = c.benchmark_group("codeq_search");
    g.bench_function("FlatL2", |b| b.iter(|| {
        for (_qid, qv) in &queries { let _ = flat.search(qv, k); }
    }));
    g.bench_function("StaticPQ", |b| b.iter(|| {
        for (_qid, qv) in &queries { let _ = pq.search_adc(qv, k); }
    }));
    g.bench_function("CoDEQ", |b| b.iter(|| {
        for (_qid, qv) in &queries { let _ = codeq.search_adc(qv, k); }
    }));
    g.finish();
}

fn bench_insert(c: &mut Criterion) {
    let dim = 128;
    let data = make_vecs(1_000, dim, 1);
    let new_vec = make_vecs(1, dim, 2)[0].1.clone();

    c.bench_function("CoDEQ_streaming_insert", |b| {
        b.iter(|| {
            let mut idx = CoDEQIndex::from_vecs(&data, 8, 42).unwrap();
            idx.insert(99999, new_vec.clone()).unwrap();
        });
    });
}

criterion_group!(benches, bench_search, bench_insert);
criterion_main!(benches);
