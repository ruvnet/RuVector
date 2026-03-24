//! Benchmarks for source-anchored canonical minimum cut (ADR-117).
//!
//! Includes Tier 1 (Stoer-Wagner), Tier 2 (Gomory-Hu tree packing),
//! and Tier 3 (dynamic incremental) benchmarks.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use ruvector_mincut::graph::DynamicGraph;
use ruvector_mincut::{
    canonical_mincut, canonical_mincut_fast, GomoryHuTree, SourceAnchoredConfig,
    DynamicCanonicalMinCut, DynamicCanonicalConfig, EdgeMutation,
};
use std::collections::HashSet;

/// Generate a random connected graph with n vertices and ~m edges.
fn random_connected_graph(n: usize, m: usize, seed: u64) -> DynamicGraph {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let g = DynamicGraph::new();

    // First create a spanning tree for connectivity
    for i in 1..n {
        let parent = rng.gen_range(0..i);
        let w = rng.gen_range(1..=10) as f64;
        g.insert_edge(parent as u64, i as u64, w).unwrap();
    }

    // Add remaining random edges
    let mut edge_set: HashSet<(u64, u64)> = HashSet::new();
    for i in 0..n {
        for &(nbr, _) in &g.neighbors(i as u64) {
            let key = if (i as u64) < nbr { (i as u64, nbr) } else { (nbr, i as u64) };
            edge_set.insert(key);
        }
    }

    let target = m.max(n - 1);
    while edge_set.len() < target {
        let u = rng.gen_range(0..n as u64);
        let v = rng.gen_range(0..n as u64);
        if u != v {
            let key = if u < v { (u, v) } else { (v, u) };
            if edge_set.insert(key) {
                let w = rng.gen_range(1..=10) as f64;
                let _ = g.insert_edge(u, v, w);
            }
        }
    }

    g
}

/// Build a random edge list (not a graph) for dynamic benchmarks.
fn random_edge_list(n: usize, m: usize, seed: u64) -> Vec<(u64, u64, f64)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut edges = Vec::new();
    let mut edge_set: HashSet<(u64, u64)> = HashSet::new();

    // Spanning tree
    for i in 1..n {
        let parent = rng.gen_range(0..i);
        let w = rng.gen_range(1..=10) as f64;
        let key = if (parent as u64) < (i as u64) {
            (parent as u64, i as u64)
        } else {
            (i as u64, parent as u64)
        };
        edge_set.insert(key);
        edges.push((parent as u64, i as u64, w));
    }

    while edge_set.len() < m.max(n - 1) {
        let u = rng.gen_range(0..n as u64);
        let v = rng.gen_range(0..n as u64);
        if u != v {
            let key = if u < v { (u, v) } else { (v, u) };
            if edge_set.insert(key) {
                let w = rng.gen_range(1..=10) as f64;
                edges.push((u, v, w));
            }
        }
    }

    edges
}

/// Build a cycle graph on n vertices.
fn cycle_graph(n: usize) -> DynamicGraph {
    let g = DynamicGraph::new();
    for i in 0..n {
        g.insert_edge(i as u64, ((i + 1) % n) as u64, 1.0).unwrap();
    }
    g
}

/// Build a complete graph Kn.
fn complete_graph(n: usize) -> DynamicGraph {
    let g = DynamicGraph::new();
    for i in 0..n {
        for j in (i + 1)..n {
            g.insert_edge(i as u64, j as u64, 1.0).unwrap();
        }
    }
    g
}

fn graph_snapshot_vertices(g: &DynamicGraph) -> Vec<u64> {
    let mut v = g.vertices();
    v.sort_unstable();
    v
}

// ---------------------------------------------------------------------------
// Tier 1: Source-anchored (Stoer-Wagner)
// ---------------------------------------------------------------------------

fn bench_canonical_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("canonical_random");

    for &n in &[10, 20, 50] {
        let m = n * 2;
        let g = random_connected_graph(n, m, 42);
        let config = SourceAnchoredConfig::default();

        group.bench_with_input(
            BenchmarkId::new("source_anchored", n),
            &g,
            |b, graph| {
                b.iter(|| {
                    black_box(canonical_mincut(graph, &config));
                });
            },
        );
    }

    group.finish();
}

fn bench_canonical_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("canonical_cycle");

    for &n in &[6, 10, 20, 50] {
        let g = cycle_graph(n);
        let config = SourceAnchoredConfig::default();

        group.bench_with_input(
            BenchmarkId::new("cycle", n),
            &g,
            |b, graph| {
                b.iter(|| {
                    black_box(canonical_mincut(graph, &config));
                });
            },
        );
    }

    group.finish();
}

fn bench_canonical_complete(c: &mut Criterion) {
    let mut group = c.benchmark_group("canonical_complete");

    for &n in &[4, 6, 8, 10] {
        let g = complete_graph(n);
        let config = SourceAnchoredConfig::default();

        group.bench_with_input(
            BenchmarkId::new("complete", n),
            &g,
            |b, graph| {
                b.iter(|| {
                    black_box(canonical_mincut(graph, &config));
                });
            },
        );
    }

    group.finish();
}

fn bench_hash_stability(c: &mut Criterion) {
    let g = random_connected_graph(20, 40, 42);
    let config = SourceAnchoredConfig::default();

    c.bench_function("hash_stability_100", |b| {
        b.iter(|| {
            for _ in 0..100 {
                black_box(canonical_mincut(&g, &config));
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Tier 2: Tree packing (Gomory-Hu)
// ---------------------------------------------------------------------------

fn bench_tree_packing_vs_stoer_wagner(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_packing_vs_stoer_wagner");

    for &n in &[10, 20, 50] {
        let m = n * 2;
        let g = random_connected_graph(n, m, 42);
        let config = SourceAnchoredConfig::default();

        group.bench_with_input(
            BenchmarkId::new("stoer_wagner", n),
            &g,
            |b, graph| {
                b.iter(|| {
                    black_box(canonical_mincut(graph, &config));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tree_packing", n),
            &g,
            |b, graph| {
                b.iter(|| {
                    black_box(GomoryHuTree::build(graph));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fast_path", n),
            &g,
            |b, graph| {
                b.iter(|| {
                    black_box(canonical_mincut_fast(graph, &config));
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Tier 3: Dynamic incremental
// ---------------------------------------------------------------------------

fn bench_dynamic_add_edge(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic");

    let base_edges = random_edge_list(50, 100, 42);

    group.bench_function("add_edge_single", |b| {
        b.iter_batched(
            || {
                let mut dmc = DynamicCanonicalMinCut::with_edges(
                    base_edges.clone(),
                    DynamicCanonicalConfig::default(),
                ).unwrap();
                dmc.canonical_cut(); // prime the cache
                dmc
            },
            |mut dmc| {
                // Add a single edge that doesn't cross the cut (best case)
                let _ = black_box(dmc.add_edge(100, 101, 1.0));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_dynamic_batch_100(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic");

    let base_edges = random_edge_list(50, 100, 42);

    group.bench_function("batch_100", |b| {
        b.iter_batched(
            || {
                let mut dmc = DynamicCanonicalMinCut::with_edges(
                    base_edges.clone(),
                    DynamicCanonicalConfig::default(),
                ).unwrap();
                dmc.canonical_cut();

                let mut mutations = Vec::new();
                for i in 100..200 {
                    mutations.push(EdgeMutation::Add(i, i + 1, 1.0));
                }
                (dmc, mutations)
            },
            |(mut dmc, mutations)| {
                let _ = black_box(dmc.apply_batch(&mutations));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_dynamic_vs_full_recompute(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_vs_full");

    for &n in &[20, 50] {
        let m = n * 2;
        let base_edges = random_edge_list(n, m, 42);
        let config = SourceAnchoredConfig::default();

        group.bench_with_input(
            BenchmarkId::new("full_recompute", n),
            &base_edges,
            |b, edges| {
                b.iter_batched(
                    || {
                        let g = DynamicGraph::new();
                        for &(u, v, w) in edges {
                            let _ = g.insert_edge(u, v, w);
                        }
                        // Add an extra edge
                        let _ = g.insert_edge(1000, 1001, 1.0);
                        g
                    },
                    |g| {
                        black_box(canonical_mincut(&g, &config));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("incremental_add", n),
            &base_edges,
            |b, edges| {
                b.iter_batched(
                    || {
                        let mut dmc = DynamicCanonicalMinCut::with_edges(
                            edges.clone(),
                            DynamicCanonicalConfig::default(),
                        ).unwrap();
                        dmc.canonical_cut(); // prime
                        dmc
                    },
                    |mut dmc| {
                        let _ = dmc.add_edge(1000, 1001, 1.0);
                        black_box(dmc.canonical_cut());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_canonical_random,
    bench_canonical_cycle,
    bench_canonical_complete,
    bench_hash_stability,
    bench_tree_packing_vs_stoer_wagner,
    bench_dynamic_add_edge,
    bench_dynamic_batch_100,
    bench_dynamic_vs_full_recompute,
);
criterion_main!(benches);
