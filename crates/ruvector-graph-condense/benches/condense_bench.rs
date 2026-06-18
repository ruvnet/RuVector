//! Condensation throughput benchmarks.
//!
//! Run with: `cargo bench -p ruvector-graph-condense --bench condense_bench`
//!
//! Two groups, because the methods differ by orders of magnitude:
//!
//! * **scalable** — `WeakBoundary` (default) and `ConnectedComponents` are
//!   single-pass + union-find, ~microseconds even at thousands of nodes.
//! * **engine** — `MinCutCommunity` and `Partition` delegate to the recursive
//!   dynamic-min-cut engine, which copies the graph per split; they are
//!   super-linear and benchmarked only at small sizes to document the cost.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvector_graph_condense::condense::{CondenseConfig, CondenseMethod, GraphCondenser};
use ruvector_graph_condense::diffcut::{DiffCutCondenser, DiffCutConfig};
use ruvector_graph_condense::metrics::evaluate_full;
use ruvector_graph_condense::synthetic::PlantedPartition;

fn planted(communities: usize, size: usize, seed: u64) -> PlantedPartition {
    PlantedPartition {
        num_communities: communities,
        community_size: size,
        dim: 16,
        p_intra: 0.4,
        p_inter: 0.002,
        seed,
        ..Default::default()
    }
}

/// Fast methods, swept to larger graphs.
fn bench_scalable(c: &mut Criterion) {
    let mut group = c.benchmark_group("condense_scalable");
    for &(communities, size) in &[(8usize, 32usize), (16, 64), (32, 64)] {
        let pp = planted(communities, size, 1);
        let (graph, features) = pp.generate();
        let n = pp.total_vertices();
        group.throughput(Throughput::Elements(n as u64));

        for (name, method) in [
            (
                "weak_boundary",
                CondenseMethod::WeakBoundary {
                    relative_threshold: 0.5,
                },
            ),
            ("connected_components", CondenseMethod::ConnectedComponents),
        ] {
            let condenser = GraphCondenser::new(CondenseConfig {
                method,
                normalize_centroids: false,
            });
            group.bench_with_input(
                BenchmarkId::new(name, n),
                &(condenser, &graph, &features),
                |b, (condenser, graph, features)| {
                    b.iter(|| {
                        let c = condenser.condense(graph, features).unwrap();
                        criterion::black_box(c.node_count())
                    });
                },
            );
        }
    }
    group.finish();
}

/// Engine-backed methods, small sizes only (super-linear cost).
fn bench_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("condense_engine");
    group.sample_size(10);
    // Capped small: recursive global min-cut is super-linear (e.g. ~24s at 96
    // nodes), so larger sizes would make the suite intractable. The point is to
    // document the cost gap vs. the scalable group, not to sweep.
    for &(communities, size) in &[(3usize, 10usize), (4, 12)] {
        let pp = planted(communities, size, 2);
        let (graph, features) = pp.generate();
        let n = pp.total_vertices();
        group.throughput(Throughput::Elements(n as u64));

        for (name, method) in [
            (
                "mincut_community",
                CondenseMethod::MinCutCommunity { min_region_size: 2 },
            ),
            (
                "partition",
                CondenseMethod::Partition {
                    num_regions: communities,
                },
            ),
        ] {
            let condenser = GraphCondenser::new(CondenseConfig {
                method,
                normalize_centroids: false,
            });
            group.bench_with_input(
                BenchmarkId::new(name, n),
                &(condenser, &graph, &features),
                |b, (condenser, graph, features)| {
                    b.iter(|| {
                        let c = condenser.condense(graph, features).unwrap();
                        criterion::black_box(c.node_count())
                    });
                },
            );
        }
    }
    group.finish();
}

/// Cost of the full metric bundle (includes two exact min-cut solves).
fn bench_metrics(c: &mut Criterion) {
    let pp = planted(8, 24, 3);
    let (graph, features) = pp.generate();
    let condenser = GraphCondenser::new(CondenseConfig::default());
    let condensed = condenser.condense(&graph, &features).unwrap();

    c.bench_function("evaluate_full_with_cut", |b| {
        b.iter(|| {
            let m = evaluate_full(&graph, &condensed);
            criterion::black_box(m.node_reduction_ratio)
        });
    });
}

/// Differentiable min-cut training cost (gradient descent over the assignment).
fn bench_diffcut(c: &mut Criterion) {
    let mut group = c.benchmark_group("condense_diffcut");
    group.sample_size(10);
    for &(communities, size) in &[(4usize, 16usize), (8, 24)] {
        let pp = planted(communities, size, 4);
        let (graph, _features) = pp.generate();
        let n = pp.total_vertices();
        group.throughput(Throughput::Elements(n as u64));
        let condenser = DiffCutCondenser::new(DiffCutConfig {
            num_clusters: communities,
            iterations: 100,
            seed: 1,
            ..Default::default()
        });
        group.bench_with_input(
            BenchmarkId::new("train", n),
            &(condenser, &graph),
            |b, (condenser, graph)| {
                b.iter(|| {
                    let r = condenser.train(graph).unwrap();
                    criterion::black_box(r.loss().total)
                });
            },
        );
    }
    group.finish();
}

/// DiffMinCut optimisation levers on a larger graph: full-sequential vs
/// full-parallel vs edge-minibatch (fixed 100 iterations, early-stop off).
fn bench_diffcut_levers(c: &mut Criterion) {
    let mut group = c.benchmark_group("condense_diffcut_levers");
    group.sample_size(10);
    let pp = planted(16, 64, 5); // 1024 nodes
    let (graph, _f) = pp.generate();
    let n = pp.total_vertices();
    group.throughput(Throughput::Elements(n as u64));
    let base = DiffCutConfig {
        num_clusters: 16,
        iterations: 100,
        tolerance: 0.0,
        seed: 1,
        ..Default::default()
    };
    let variants = [
        ("full_sequential", DiffCutConfig { ..base.clone() }),
        (
            "full_parallel",
            DiffCutConfig {
                parallel: true,
                ..base.clone()
            },
        ),
        (
            "minibatch_2048",
            DiffCutConfig {
                minibatch_edges: Some(2048),
                ..base.clone()
            },
        ),
    ];
    for (name, cfg) in variants {
        let condenser = DiffCutCondenser::new(cfg);
        group.bench_with_input(
            BenchmarkId::new(name, n),
            &(condenser, &graph),
            |b, (c, g)| {
                b.iter(|| criterion::black_box(c.train(g).unwrap().loss().total));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_scalable,
    bench_engine,
    bench_diffcut,
    bench_diffcut_levers,
    bench_metrics
);
criterion_main!(benches);
