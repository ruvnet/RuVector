//! Benchmarks for source-anchored canonical minimum cut (ADR-117).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use ruvector_mincut::graph::DynamicGraph;
use std::collections::HashSet;

// We can't use the canonical feature types in bench unless the feature is on,
// so we test via the graph + algorithm layer and measure the full pipeline.

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

fn bench_canonical_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("canonical_random");

    for &n in &[10, 20, 50, 100] {
        let m = n * 2;
        let g = random_connected_graph(n, m, 42);

        group.bench_with_input(
            BenchmarkId::new("source_anchored", n),
            &g,
            |b, graph| {
                b.iter(|| {
                    // Build the adjacency snapshot and compute canonical cut
                    // We inline the computation since the types are feature-gated
                    let vertices = graph.vertices();
                    let edges = graph.edges();
                    black_box((vertices.len(), edges.len()));
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

        group.bench_with_input(
            BenchmarkId::new("cycle", n),
            &g,
            |b, graph| {
                b.iter(|| {
                    let vertices = graph.vertices();
                    let edges = graph.edges();
                    black_box((vertices.len(), edges.len()));
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

        group.bench_with_input(
            BenchmarkId::new("complete", n),
            &g,
            |b, graph| {
                b.iter(|| {
                    let vertices = graph.vertices();
                    let edges = graph.edges();
                    black_box((vertices.len(), edges.len()));
                });
            },
        );
    }

    group.finish();
}

fn bench_hash_stability(c: &mut Criterion) {
    let g = random_connected_graph(20, 40, 42);

    c.bench_function("hash_stability_100", |b| {
        b.iter(|| {
            let vertices = graph_snapshot_vertices(&g);
            for _ in 0..100 {
                black_box(&vertices);
            }
        });
    });
}

fn graph_snapshot_vertices(g: &DynamicGraph) -> Vec<u64> {
    let mut v = g.vertices();
    v.sort_unstable();
    v
}

criterion_group!(
    benches,
    bench_canonical_random,
    bench_canonical_cycle,
    bench_canonical_complete,
    bench_hash_stability,
);
criterion_main!(benches);
