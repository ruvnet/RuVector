//! Nightly research benchmark (2026-11-13,
//! `docs/research/nightly/2026-11-13_bounded_instance_scan_dedup`).
//!
//! Re-runs the exact topology/size sweep from
//! `ruvector-agent-memory/examples/mincut_scaling_probe.rs` (ADR-345's
//! latency scaling table), directly against `MinCutWrapper::query()`
//! (`RuVectorGraphAnalyzer::partition()` is `wrapper.query()` plus a cheap
//! O(V) side-materialization — not the bottleneck), comparing
//! `ScanOptLevel::Baseline` against the two candidates this run adds.
//!
//! Also reports `num_instances()` after each cold-start query: the count of
//! `BoundedInstance`s actually materialized, which is the direct mechanism
//! behind any latency difference (fewer materialized instances means fewer
//! redundant brute-force / search_for_cuts calls and fewer redundant
//! `graph.edges()` collect+sorts).
//!
//! Run:
//!   cargo run --release -p ruvector-mincut --example wrapper_scan_opt_bench

use ruvector_mincut::{DynamicGraph, MinCutWrapper, ScanOptLevel};
use std::sync::Arc;
use std::time::Instant;

fn ring_knn_graph(n: usize, k: usize) -> Arc<DynamicGraph> {
    let graph = Arc::new(DynamicGraph::new());
    for i in 0..n {
        for d in 1..=k {
            let j = (i + d) % n;
            let weight = 1.0 / (0.1 + (d as f64) * 0.01);
            let _ = graph.insert_edge(i as u64, j as u64, weight);
        }
    }
    graph
}

fn main() {
    let sizes = [19usize, 50, 100, 200, 400];
    let k = 8usize;
    let levels = [
        ("baseline", ScanOptLevel::Baseline),
        ("dedup_ranges", ScanOptLevel::DedupRanges),
        ("dedup+hoist_edges", ScanOptLevel::DedupRangesAndHoistEdges),
    ];

    println!(
        "{:<6} {:<20} {:>14} {:>12} {:>10}",
        "n", "variant", "query_ms", "instances", "cut_value"
    );
    for &n in &sizes {
        let graph = ring_knn_graph(n, k);
        let mut baseline_value: Option<u64> = None;
        for &(name, level) in &levels {
            let mut wrapper = MinCutWrapper::new(Arc::clone(&graph)).with_scan_opt_level(level);
            // Mirror `RuVectorGraphAnalyzer::new`'s sync step: `MinCutWrapper`
            // tracks its own pending-update buffer independently of the
            // `DynamicGraph` it was built from, so pre-existing edges must be
            // replayed in explicitly.
            for edge in graph.edges() {
                wrapper.insert_edge(edge.id, edge.source, edge.target);
            }
            let t0 = Instant::now();
            let result = wrapper.query();
            let elapsed = t0.elapsed();
            let value = result.value();
            let instances = wrapper.num_instances();

            if name == "baseline" {
                baseline_value = Some(value);
            } else if let Some(bv) = baseline_value {
                assert_eq!(
                    value, bv,
                    "n={n} variant={name} diverged from baseline cut value"
                );
            }

            println!(
                "{:<6} {:<20} {:>14.3} {:>12} {:>10}",
                n,
                name,
                elapsed.as_secs_f64() * 1000.0,
                instances,
                value
            );
        }
    }
}
