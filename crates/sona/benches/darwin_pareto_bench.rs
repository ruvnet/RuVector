//! Scaling benchmark for the three Pareto-frontier selection strategies in
//! `ruvector_sona::darwin_pareto` (ADR-335 WP31 / ADR-353).
//!
//! Compares, at increasing candidate counts N with M=2 objectives (the
//! common Darwin promotion-gate shape — see
//! `schemas/optimization-manifest-v1.json`'s `promotion.pareto.objectives`):
//!   - `brute_force_frontier` (baseline, O(M*N^2) worst case, early-exit)
//!   - `nsga2_frontier` (candidate_A, O(M*N^2), no early exit — this run
//!     measured it slower than the baseline at every N, see the module's
//!     doc comment and the companion nightly research doc)
//!   - `kung_frontier_2d` (candidate_B, O(N log N) divide-and-conquer)
//!
//! Run: `cargo bench -p ruvector-sona --bench darwin_pareto_bench`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ruvector_sona::darwin_pareto::{
    brute_force_frontier, kung_frontier_2d, nsga2_frontier, Direction,
};

/// Deterministic xorshift64*, matching the module's own test generator so
/// the benchmark's input distribution is reproducible without pulling in
/// an external RNG dependency for a dev-only benchmark.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn gen_points(seed: u64, n: usize) -> Vec<[f64; 2]> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            let a = (xorshift(&mut s) % 100_000) as f64 / 1000.0;
            let b = (xorshift(&mut s) % 100_000) as f64 / 1000.0;
            [a, b]
        })
        .collect()
}

fn pareto_frontier_benchmark(c: &mut Criterion) {
    const OBJ: [Direction; 2] = [Direction::Maximize, Direction::Minimize];
    let mut group = c.benchmark_group("darwin_pareto_frontier_2d");

    // baseline and candidate_A are both O(M*N^2); kept to N<=5_000 so a full
    // bench run stays tractable (5_000^2 = 25M comparisons/iteration, not
    // 50_000^2 = 2.5B). candidate_B is O(N log N) and is additionally
    // measured up to 100_000 to show the scaling gap the hypothesis claims.
    for &n in &[100usize, 500, 2_000, 5_000] {
        let data = gen_points(0xD17A_u64.wrapping_add(n as u64), n);
        let points: Vec<&[f64]> = data.iter().map(|r| r.as_slice()).collect();

        group.bench_with_input(BenchmarkId::new("baseline_brute_force", n), &n, |b, _| {
            b.iter(|| black_box(brute_force_frontier(&OBJ, black_box(&points))));
        });

        group.bench_with_input(BenchmarkId::new("candidate_a_nsga2", n), &n, |b, _| {
            b.iter(|| black_box(nsga2_frontier(&OBJ, black_box(&points))));
        });

        group.bench_with_input(BenchmarkId::new("candidate_b_kung_2d", n), &n, |b, _| {
            b.iter(|| black_box(kung_frontier_2d(&OBJ, black_box(&points))));
        });
    }

    for &n in &[20_000usize, 100_000] {
        let data = gen_points(0xD17A_u64.wrapping_add(n as u64), n);
        let points: Vec<&[f64]> = data.iter().map(|r| r.as_slice()).collect();
        group.bench_with_input(BenchmarkId::new("candidate_b_kung_2d", n), &n, |b, _| {
            b.iter(|| black_box(kung_frontier_2d(&OBJ, black_box(&points))));
        });
    }

    group.finish();
}

criterion_group!(benches, pareto_frontier_benchmark);
criterion_main!(benches);
