//! Connectivity backend benchmarks (2026-10-07 nightly:
//! `docs/research/nightly/2026-10-07-mincut-polylog-connectivity-backend/`).
//!
//! Two groups, following `jtree_bench.rs`'s conventions:
//!
//! 1. `connectivity_backend_raw` — micro-benchmark of
//!    `DynamicConnectivity` (`ConnectivityBackend::EulerTour`) vs.
//!    `PolylogConnectivity` (`ConnectivityBackend::Polylog`) in isolation:
//!    build + a single `is_connected()`/`connected()` call, at graph sizes
//!    disjoint from `jtree_bench.rs`'s existing (undirected,
//!    baseline-less) `bench_polylog_connectivity` group.
//! 2. `connectivity_backend_partition` — end-to-end
//!    `RuVectorGraphAnalyzer::partition()` latency with each backend
//!    selected, at graph sizes matching
//!    `ruvector-agent-memory/examples/mincut_gated_forgetting_bench.rs`'s
//!    corpus (`n=84`) plus two larger synthetic sizes (`n=100`, `n=200`).
//!    Sample counts are reduced for the larger, slower sizes to keep CI
//!    runtime bounded — see
//!    `examples/connectivity_backend_partition_probe.rs` for the
//!    higher-repetition, multi-seed measurement that is this nightly run's
//!    headline evidence.
//!
//! Run:
//!   cargo bench -p ruvector-mincut --bench connectivity_backend_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_mincut::connectivity::{ConnectivityBackend, ConnectivityStructure};
use ruvector_mincut::graph::DynamicGraph;
use ruvector_mincut::RuVectorGraphAnalyzer;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

/// Random connected graph on `n` vertices (spanning path + extra random
/// edges for `k`-ish average degree). Mirrors
/// `examples/connectivity_backend_partition_probe.rs`'s generator so the
/// criterion numbers and the headline probe numbers are directly
/// comparable.
fn random_connected_graph(n: usize, k: usize, seed: u64) -> DynamicGraph {
    let mut rng = StdRng::seed_from_u64(seed);
    let graph = DynamicGraph::new();

    let mut order: Vec<u64> = (0..n as u64).collect();
    for i in (1..order.len()).rev() {
        let j = rng.gen_range(0..=i);
        order.swap(i, j);
    }
    for w in order.windows(2) {
        let _ = graph.insert_edge(w[0], w[1], 1.0);
    }

    let target_extra_edges = (k * n) / 2;
    let mut added = 0usize;
    let mut attempts = 0usize;
    while added < target_extra_edges && attempts < target_extra_edges * 20 {
        attempts += 1;
        let u = rng.gen_range(0..n as u64);
        let v = rng.gen_range(0..n as u64);
        if u != v && graph.insert_edge(u, v, 1.0).is_ok() {
            added += 1;
        }
    }
    graph
}

fn raw_edges(n: usize, seed: u64) -> Vec<(u64, u64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let m = n * 4;
    let mut edges = Vec::with_capacity(m);
    let mut seen = HashSet::new();
    while edges.len() < m {
        let u = rng.gen_range(0..n as u64);
        let v = rng.gen_range(0..n as u64);
        if u != v {
            let key = if u < v { (u, v) } else { (v, u) };
            if seen.insert(key) {
                edges.push(key);
            }
        }
    }
    edges
}

// ============================================================================
// Group 1: raw connectivity backend micro-benchmark
// ============================================================================

fn bench_connectivity_backend_raw(c: &mut Criterion) {
    let mut group = c.benchmark_group("connectivity_backend_raw");
    group.sample_size(50);

    for size in [100usize, 1_000, 5_000] {
        let edges = raw_edges(size, 7);

        for backend in [ConnectivityBackend::EulerTour, ConnectivityBackend::Polylog] {
            let label = match backend {
                ConnectivityBackend::EulerTour => "euler_tour_build_and_query",
                ConnectivityBackend::Polylog => "polylog_build_and_query",
            };
            group.bench_with_input(BenchmarkId::new(label, size), &size, |b, _| {
                b.iter(|| {
                    let mut conn = ConnectivityStructure::new(backend);
                    for &(u, v) in &edges {
                        conn.insert_edge(u, v);
                    }
                    black_box(conn.is_connected());
                    black_box(conn.connected(0, size as u64 - 1));
                });
            });
        }
    }

    group.finish();
}

// ============================================================================
// Group 2: end-to-end RuVectorGraphAnalyzer::partition() latency
// ============================================================================

fn bench_connectivity_backend_partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("connectivity_backend_partition");

    // (size, sample_size, measurement_time_secs) — reduced sampling for the
    // larger, slower sizes to keep total CI runtime bounded. See the module
    // doc for why the headline, higher-repetition numbers live in
    // `examples/connectivity_backend_partition_probe.rs` instead.
    let cases: &[(usize, usize, u64)] = &[(84, 20, 10), (100, 15, 15), (200, 10, 25)];

    for &(n, sample_size, measurement_secs) in cases {
        group.sample_size(sample_size);
        group.measurement_time(Duration::from_secs(measurement_secs));

        let graph = Arc::new(random_connected_graph(n, 8, 42));

        for backend in [ConnectivityBackend::EulerTour, ConnectivityBackend::Polylog] {
            let label = match backend {
                ConnectivityBackend::EulerTour => "euler_tour_partition",
                ConnectivityBackend::Polylog => "polylog_partition",
            };
            group.bench_with_input(BenchmarkId::new(label, n), &n, |b, _| {
                b.iter(|| {
                    let mut analyzer =
                        RuVectorGraphAnalyzer::new_with_backend(Arc::clone(&graph), backend);
                    black_box(analyzer.partition());
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_connectivity_backend_raw,
    bench_connectivity_backend_partition
);
criterion_main!(benches);
