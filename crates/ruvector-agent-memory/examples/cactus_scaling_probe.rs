//! Scaling probe for `ruvector_mincut::CactusGraph::{build_from_graph,
//! canonical_cut}` latency, directly comparable to ADR-345's
//! `mincut_scaling_probe` (same ring k-NN topology, same sizes). Not itself
//! part of the acceptance gate — informational, per ADR-345's own precedent.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example cactus_scaling_probe --features mincut-forget-cactus

use ruvector_mincut::{CactusGraph, DynamicGraph};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let sizes = [19usize, 50, 100, 200, 400, 800];
    let k = 8usize;
    for &n in &sizes {
        let t0 = Instant::now();
        let graph = Arc::new(DynamicGraph::new());
        for i in 0..n {
            for d in 1..=k {
                let j = (i + d) % n;
                // Unconditional on i < j: matches ADR-345's `from_knn`,
                // which inserts every (vertex, neighbor) pair regardless of
                // direction; `insert_edge` no-ops on an already-present
                // undirected pair.
                let weight = 0.1 + (d as f64) * 0.01;
                let _ = graph.insert_edge(i as u64, j as u64, weight);
            }
        }
        let build_elapsed = t0.elapsed();

        let t1 = Instant::now();
        let cactus = CactusGraph::build_from_graph(&graph);
        let cactus_build_elapsed = t1.elapsed();

        let t2 = Instant::now();
        let _ = cactus.canonical_cut();
        let cut_elapsed = t2.elapsed();

        println!(
            "n={n:<5} graph_build={:>10.3}ms  cactus_build={:>10.3}ms  canonical_cut={:>10.3}ms  total={:>10.3}ms",
            build_elapsed.as_secs_f64() * 1000.0,
            cactus_build_elapsed.as_secs_f64() * 1000.0,
            cut_elapsed.as_secs_f64() * 1000.0,
            (build_elapsed + cactus_build_elapsed + cut_elapsed).as_secs_f64() * 1000.0,
        );
    }
}
