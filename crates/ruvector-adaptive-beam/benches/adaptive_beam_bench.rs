use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_adaptive_beam::graph::build_knn_graph;
use ruvector_adaptive_beam::{AdaptiveBeamIndex, BeamStopPolicy};

fn make_vecs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    (0..n)
        .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

fn bench_policies(c: &mut Criterion) {
    let n = 2_000;
    let d = 64;
    let k = 10;
    let vecs = make_vecs(n, d, 42);
    let (nb, ep) = build_knn_graph(&vecs, 12);
    let idx = AdaptiveBeamIndex::new(vecs, nb, ep);
    let queries = make_vecs(100, d, 999);

    let mut group = c.benchmark_group("beam_search");
    group.throughput(Throughput::Elements(queries.len() as u64));

    let cases: &[(BeamStopPolicy, &str)] = &[
        (BeamStopPolicy::FixedWidth { beam_width: 32 }, "FixedWidth_bw32"),
        (BeamStopPolicy::DistanceAdaptive { gamma: 1.0 }, "DistAdaptive_g1"),
        (BeamStopPolicy::AdaptiveWithFloor { gamma: 0.5, min_expansions: 8 }, "AdaptFloor_g05_m8"),
    ];

    for (policy, name) in cases {
        let policy = *policy;
        group.bench_with_input(BenchmarkId::new("search", name), name, |b, _| {
            b.iter(|| {
                for q in &queries {
                    let _ = idx.search(q, k, policy);
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_policies);
criterion_main!(benches);
