//! Benchmarks for the spectral sparsifier.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_sparsifier::{AdaptiveGeoSpar, SparseGraph, Sparsifier, SparsifierConfig};

fn build_cycle(n: usize) -> SparseGraph {
    let mut edges = Vec::with_capacity(n);
    for i in 0..n {
        edges.push((i, (i + 1) % n, 1.0));
    }
    SparseGraph::from_edges(&edges)
}

fn bench_build(c: &mut Criterion) {
    let g = build_cycle(100);
    c.bench_function("build_sparsifier_100", |b| {
        b.iter(|| {
            let spar = AdaptiveGeoSpar::build(black_box(&g), SparsifierConfig::default());
            black_box(spar)
        })
    });
}

fn bench_insert(c: &mut Criterion) {
    let g = build_cycle(50);
    c.bench_function("insert_edge", |b| {
        b.iter_batched(
            || AdaptiveGeoSpar::build(&g, SparsifierConfig::default()).unwrap(),
            |mut spar| {
                let _ = spar.insert_edge(0, 25, 1.5);
                black_box(spar)
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_audit(c: &mut Criterion) {
    let g = build_cycle(100);
    let spar = AdaptiveGeoSpar::build(&g, SparsifierConfig::default()).unwrap();
    c.bench_function("audit_100", |b| {
        b.iter(|| {
            let result = spar.audit();
            black_box(result)
        })
    });
}

fn bench_laplacian_qf(c: &mut Criterion) {
    let g = build_cycle(500);
    let x: Vec<f64> = (0..500).map(|i| (i as f64) * 0.01).collect();
    c.bench_function("laplacian_qf_500", |b| {
        b.iter(|| {
            let val = g.laplacian_quadratic_form(black_box(&x));
            black_box(val)
        })
    });
}

criterion_group!(
    benches,
    bench_build,
    bench_insert,
    bench_audit,
    bench_laplacian_qf
);
criterion_main!(benches);
