use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

use ruvector_symphonyqg::{build_all, Config, Metric};

fn generate(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let unif = Uniform::new(-1.0f64, 1.0);
    let noise = Normal::new(0.0f64, 0.3).unwrap();
    let n_c = 20.min(n / 2).max(1);
    let centroids: Vec<Vec<f64>> = (0..n_c)
        .map(|_| (0..dim).map(|_| unif.sample(&mut rng)).collect())
        .collect();
    use rand::Rng as _;
    (0..n)
        .map(|_| {
            let c = &centroids[rng.gen_range(0..n_c)];
            c.iter()
                .map(|&x| (x + noise.sample(&mut rng)) as f32)
                .collect()
        })
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let dim = 128;
    let n = 5_000;
    let ef = 100;
    let k = 10;
    let data = generate(n, dim, 0xBEEF);
    let query = generate(1, dim, 0xDEAD)[0].clone();

    let cfg = Config {
        dim,
        m_base: 16,
        ef_construction: 200,
        metric: Metric::Euclidean,
        seed: 1,
    };
    let (flat, graph_exact, symphony) = build_all(&data, &cfg);

    let mut grp = c.benchmark_group("search_n5k_dim128");

    grp.bench_function("FlatExact", |b| {
        b.iter(|| flat.search(black_box(&query), black_box(k)))
    });

    for &ef_v in &[50usize, 100, 200] {
        grp.bench_with_input(BenchmarkId::new("GraphExact", ef_v), &ef_v, |b, &ef| {
            b.iter(|| graph_exact.search(black_box(&query), black_box(k), black_box(ef)))
        });
        grp.bench_with_input(BenchmarkId::new("SymphonyQG", ef_v), &ef_v, |b, &ef| {
            b.iter(|| symphony.search(black_box(&query), black_box(k), black_box(ef)))
        });
    }
    grp.finish();
}

fn bench_encode(c: &mut Criterion) {
    let dim = 128;
    let n = 100;
    let data = generate(n, dim, 42);
    let cfg = Config {
        dim,
        m_base: 8,
        ef_construction: 50,
        ..Config::default()
    };
    let q = generate(1, dim, 99)[0].clone();

    c.bench_function("encode_query_dim128", |b| {
        b.iter(|| {
            let graph = ruvector_symphonyqg::build::build(black_box(&data), &cfg);
            graph.encode_query(black_box(&q))
        })
    });
}

criterion_group!(benches, bench_search, bench_encode);
criterion_main!(benches);
