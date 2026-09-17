//! Throwaway scaling probe (2026-09-17 nightly, ADR-346) for
//! `MinCutBuilder::build()` + `DynamicMinCut::partition()` latency, mirroring
//! `mincut_scaling_probe.rs`'s methodology exactly (same ring topology, same
//! sizes) so the two probes' numbers are directly comparable. Not itself
//! part of the shipped research artifact.
//!
//! Builds a fixed-degree ring k-NN graph (vertex i connects to the next k
//! vertices mod n) at increasing n and times one graph-build + one
//! `MinCutBuilder::build()` (which performs the one exact Stoer-Wagner-style
//! solve) at each size. Extra sizes beyond the original probe's [19, 50,
//! 100, 200, 400] are included to locate `DynamicMinCut`'s own practical
//! ceiling now that the wrapper-instance-replay cost is gone; those extra
//! rows are not part of the reused 2026-09-05 hypothesis or its acceptance
//! thresholds.

use ruvector_mincut::{DynamicGraph, MinCutBuilder};
use std::time::Instant;

fn main() {
    let sizes = [19usize, 50, 100, 200, 400, 800, 1600, 3200];
    let k = 8usize;
    for &n in &sizes {
        let graph = DynamicGraph::new();
        for i in 0..n {
            for d in 1..=k {
                let j = (i + d) % n;
                let weight = 0.1 + (d as f64) * 0.01;
                let _ = graph.insert_edge(i as u64, j as u64, weight);
            }
        }
        let edges: Vec<(u64, u64, f64)> = graph
            .edges()
            .into_iter()
            .map(|e| (e.source, e.target, e.weight))
            .collect();

        let t0 = Instant::now();
        let mincut = MinCutBuilder::new()
            .with_edges(edges)
            .build()
            .expect("valid ring graph builds");
        let build_elapsed = t0.elapsed();

        let t1 = Instant::now();
        let _ = mincut.partition();
        let partition_elapsed = t1.elapsed();

        println!(
            "n={n:<5} build+solve={:>10.3}ms  partition={:>10.3}ms  min_cut={}",
            build_elapsed.as_secs_f64() * 1000.0,
            partition_elapsed.as_secs_f64() * 1000.0,
            mincut.min_cut_value()
        );
    }
}
