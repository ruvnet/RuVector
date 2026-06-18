//! Criterion benchmarks for the emergent-time numerical core and clocks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use emergent_time::complex_matrix::schrodinger_propagator;
use emergent_time::real_matrix::RealMatrix;
use emergent_time::structural_clock::{
    self, compression_error, early_warning_lead, Clock, Scenario, StructuralMetric,
    StructuralProperTime,
};
use emergent_time::PageWootters;

fn sym_h(n: usize) -> RealMatrix {
    RealMatrix::from_fn(n, |r, c| {
        if r == c {
            (r as f64) - (n as f64) / 2.0
        } else if (r as i64 - c as i64).abs() == 1 {
            0.3
        } else {
            0.0
        }
    })
}

fn bench_eigensolver(c: &mut Criterion) {
    let mut g = c.benchmark_group("symmetric_eigen");
    for &n in &[4usize, 8, 16, 32] {
        let h = sym_h(n);
        g.bench_function(format!("n{n}"), |b| {
            b.iter(|| black_box(h.symmetric_eigen()))
        });
    }
    g.finish();
}

fn bench_propagator(c: &mut Criterion) {
    let h = sym_h(16);
    c.bench_function("schrodinger_propagator_n16", |b| {
        b.iter(|| black_box(schrodinger_propagator(&h, 1.0)))
    });
}

fn bench_page_wootters(c: &mut Criterion) {
    let pw = PageWootters::new(sym_h(8));
    c.bench_function("page_wootters_conditional_n8", |b| {
        b.iter(|| black_box(pw.conditional_state(black_box(1.3))))
    });

    // P1: cached-eigenbasis Schrödinger evolution vs the from-scratch path that
    // re-diagonalizes H_R and forms the full propagator every call. Same H size
    // (n16) as `schrodinger_propagator_n16` for a like-for-like comparison.
    let pw16 = PageWootters::new(sym_h(16));
    c.bench_function("page_wootters_schrodinger_cached_n16", |b| {
        b.iter(|| black_box(pw16.schrodinger_state(black_box(1.0))))
    });
    c.bench_function("page_wootters_schrodinger_from_scratch_n16", |b| {
        b.iter(|| black_box(pw16.schrodinger_state_from_scratch(black_box(1.0))))
    });
}

fn bench_structural_clock(c: &mut Criterion) {
    let traj = structural_clock::generate_scenario(&Scenario::default());
    let spt = StructuralProperTime::new(StructuralMetric::default());
    let mut g = c.benchmark_group("structural_clock");
    g.bench_function("cumulative", |b| {
        b.iter(|| black_box(spt.cumulative(&traj)))
    });
    g.bench_function("early_warning_lead", |b| {
        b.iter(|| black_box(early_warning_lead(&spt, &traj, 80, 30, 4.0)))
    });
    g.bench_function("compression_error", |b| {
        b.iter(|| black_box(compression_error(&spt, &traj, 10)))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_eigensolver,
    bench_propagator,
    bench_page_wootters,
    bench_structural_clock
);
criterion_main!(benches);
