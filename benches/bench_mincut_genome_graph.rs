//! Min-Cut Benchmarks for Genome-Scale Graphs
//!
//! Benchmarks min-cut operations on graph structures resembling genomic
//! overlap / assembly graphs (sparse, with moderately-weighted edges).
//!
//! Run: cargo bench -p ruvector-dna-bench --bench bench_mincut_genome_graph

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};

use ruvector_mincut::{DynamicGraph, DynamicMinCut, MinCutBuilder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random number from seed (LCG).
fn lcg(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Build a genome-like graph: a backbone chain with random cross-links.
///
/// Mimics assembly / overlap graphs where contigs form a chain and
/// repeats create additional edges.
fn build_genome_graph(num_nodes: usize, extra_edges_ratio: f64) -> Vec<(u64, u64, f64)> {
    let mut edges = Vec::new();
    let mut seed = 42u64;

    // Backbone chain
    for i in 0..(num_nodes - 1) {
        seed = lcg(seed);
        let weight = 1.0 + (seed % 100) as f64 / 100.0; // [1.0, 2.0)
        edges.push((i as u64, (i + 1) as u64, weight));
    }

    // Cross-links (simulating repeat regions)
    let num_extra = (num_nodes as f64 * extra_edges_ratio) as usize;
    for _ in 0..num_extra {
        seed = lcg(seed);
        let u = (seed % num_nodes as u64) as u64;
        seed = lcg(seed);
        let v = (seed % num_nodes as u64) as u64;
        if u != v {
            seed = lcg(seed);
            let weight = 0.5 + (seed % 200) as f64 / 100.0;
            edges.push((u, v, weight));
        }
    }

    edges
}

// ---------------------------------------------------------------------------
// Benchmark: Graph Construction
// ---------------------------------------------------------------------------

fn bench_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("mincut_genome/graph_construction");
    group.sample_size(10);

    for &num_nodes in &[1_000usize, 10_000, 50_000] {
        let edges = build_genome_graph(num_nodes, 0.5);

        group.throughput(Throughput::Elements(num_nodes as u64));
        group.bench_with_input(
            BenchmarkId::new("nodes", num_nodes),
            &edges,
            |b, edges| {
                b.iter(|| {
                    MinCutBuilder::new()
                        .exact()
                        .with_edges(edges.clone())
                        .build()
                        .expect("build graph")
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Min-Cut Value Query
// ---------------------------------------------------------------------------

fn bench_mincut_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("mincut_genome/min_cut_query");

    for &num_nodes in &[1_000usize, 5_000, 10_000] {
        let edges = build_genome_graph(num_nodes, 0.5);
        let mincut = MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build()
            .expect("build");

        group.bench_with_input(
            BenchmarkId::new("nodes", num_nodes),
            &mincut,
            |b, mc| {
                b.iter(|| mc.min_cut_value());
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Full Min-Cut Result (with partition)
// ---------------------------------------------------------------------------

fn bench_mincut_full_result(c: &mut Criterion) {
    let mut group = c.benchmark_group("mincut_genome/full_result");

    for &num_nodes in &[1_000usize, 5_000, 10_000] {
        let edges = build_genome_graph(num_nodes, 0.5);
        let mincut = MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build()
            .expect("build");

        group.bench_with_input(
            BenchmarkId::new("nodes", num_nodes),
            &mincut,
            |b, mc| {
                b.iter(|| mc.min_cut());
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Incremental Edge Insertion
// ---------------------------------------------------------------------------

fn bench_incremental_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("mincut_genome/edge_insert");

    for &num_nodes in &[1_000usize, 5_000] {
        let edges = build_genome_graph(num_nodes, 0.3);

        group.bench_with_input(
            BenchmarkId::new("nodes", num_nodes),
            &edges,
            |b, edges| {
                b.iter_batched(
                    || {
                        // Setup: build initial graph
                        let mc = MinCutBuilder::new()
                            .exact()
                            .with_edges(edges.clone())
                            .build()
                            .expect("build");
                        mc
                    },
                    |mut mc| {
                        // Insert 100 new edges
                        let base = num_nodes as u64;
                        for i in 0..100u64 {
                            let u = base + i;
                            let v = i % base;
                            let _ = mc.insert_edge(u, v, 1.0);
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Incremental Edge Deletion
// ---------------------------------------------------------------------------

fn bench_incremental_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("mincut_genome/edge_delete");

    for &num_nodes in &[1_000usize, 5_000] {
        let edges = build_genome_graph(num_nodes, 0.5);

        group.bench_with_input(
            BenchmarkId::new("nodes", num_nodes),
            &edges,
            |b, edges| {
                b.iter_batched(
                    || {
                        MinCutBuilder::new()
                            .exact()
                            .with_edges(edges.clone())
                            .build()
                            .expect("build")
                    },
                    |mut mc| {
                        // Delete backbone edges (which are guaranteed to exist)
                        for i in 0..50u64 {
                            let _ = mc.delete_edge(i, i + 1);
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Graph Statistics (lightweight query)
// ---------------------------------------------------------------------------

fn bench_graph_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("mincut_genome/graph_stats");

    for &num_nodes in &[1_000usize, 10_000, 50_000] {
        let edges = build_genome_graph(num_nodes, 0.5);
        let mincut = MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build()
            .expect("build");

        group.bench_with_input(
            BenchmarkId::new("nodes", num_nodes),
            &mincut,
            |b, mc| {
                b.iter(|| {
                    let graph = mc.graph();
                    let reader = graph.read();
                    reader.stats()
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Connectivity Check
// ---------------------------------------------------------------------------

fn bench_connectivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("mincut_genome/connectivity");

    for &num_nodes in &[1_000usize, 10_000] {
        let edges = build_genome_graph(num_nodes, 0.5);
        let mincut = MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build()
            .expect("build");

        group.bench_with_input(
            BenchmarkId::new("nodes", num_nodes),
            &mincut,
            |b, mc| {
                b.iter(|| mc.is_connected());
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Approximate Min-Cut
// ---------------------------------------------------------------------------

fn bench_approximate_mincut(c: &mut Criterion) {
    let mut group = c.benchmark_group("mincut_genome/approximate");
    group.sample_size(10);

    for &num_nodes in &[1_000usize, 10_000] {
        let edges = build_genome_graph(num_nodes, 0.5);

        for &epsilon in &[0.1, 0.2, 0.5] {
            let mc = MinCutBuilder::new()
                .approximate(epsilon)
                .with_edges(edges.clone())
                .build()
                .expect("build");

            group.bench_with_input(
                BenchmarkId::new(format!("nodes_{}_eps", num_nodes), format!("{:.1}", epsilon)),
                &mc,
                |b, mc| {
                    b.iter(|| mc.min_cut());
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_graph_construction,
    bench_mincut_query,
    bench_mincut_full_result,
    bench_incremental_insert,
    bench_incremental_delete,
    bench_graph_stats,
    bench_connectivity,
    bench_approximate_mincut,
);
criterion_main!(benches);
