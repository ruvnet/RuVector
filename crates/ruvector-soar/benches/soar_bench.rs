//! Criterion bench — measures build time and per-query latency for the
//! three assignment strategies on a synthetic clustered dataset.

use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_soar::{Assignment, IvfIndex};

fn synth(n: usize, dim: usize, n_clusters: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let anchors: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-5.0..5.0_f32)).collect())
        .collect();
    let db: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let a = &anchors[i % n_clusters];
            (0..dim)
                .map(|d| a[d] + rng.gen_range(-0.6..0.6_f32))
                .collect()
        })
        .collect();
    let q: Vec<Vec<f32>> = (0..50)
        .map(|i| {
            let a = &anchors[i % n_clusters];
            (0..dim)
                .map(|d| a[d] + rng.gen_range(-0.8..0.8_f32))
                .collect()
        })
        .collect();
    (db, q)
}

fn bench(c: &mut Criterion) {
    let (db, queries) = synth(8_000, 64, 80, 0xCAFE);

    let mut g = c.benchmark_group("soar_build_8k_d64_c64");
    g.sample_size(10);
    for (name, asg) in [
        ("single", Assignment::Single),
        ("spillover", Assignment::Spillover),
        ("soar_l1.5", Assignment::Soar { lambda: 1.5 }),
    ] {
        g.bench_function(name, |b| {
            b.iter(|| {
                let _ = IvfIndex::build(db.clone(), 64, asg, 1).unwrap();
            })
        });
    }
    g.finish();

    let mut g = c.benchmark_group("soar_query_8k_d64_c64_p4");
    for (name, asg) in [
        ("single", Assignment::Single),
        ("spillover", Assignment::Spillover),
        ("soar_l1.5", Assignment::Soar { lambda: 1.5 }),
    ] {
        let idx = IvfIndex::build(db.clone(), 64, asg, 1).unwrap();
        g.bench_function(name, |b| {
            b.iter(|| {
                for q in &queries {
                    let _ = idx.search(q, 10, 4);
                }
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
