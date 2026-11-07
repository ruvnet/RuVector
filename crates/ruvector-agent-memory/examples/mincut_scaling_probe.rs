//! Throwaway scaling probe for min-cut backend latency, used to size
//! `mincut_gated_forgetting_bench`'s corpus and document the "Failure modes"
//! scaling table in
//! docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md and
//! its follow-up in
//! docs/research/nightly/2026-09-28-mincut-backend-latency/README.md. Not
//! itself part of the shipped research artifact.
//!
//! Builds a fixed-degree ring k-NN graph (vertex i connects to the next k
//! vertices mod n) at increasing n and times, at each size:
//!
//! - `RuVectorGraphAnalyzer::from_knn` (build) + `.partition()` (query) — the
//!   original ADR-345 backend, unmodified from the 2026-09-05 nightly.
//! - `graph_forget`'s `DynamicMinCut`-backed boundary computation (build a
//!   `DynamicGraph` with the same edges, then `DynamicMinCut::from_graph`,
//!   which computes the cut immediately) — the ADR-350 candidate backend.
//!   `.partition()` on the resulting `DynamicMinCut` is O(n) (a cached-result
//!   getter), so it is not timed separately; the cost is entirely in
//!   `from_graph`.
//!
//! The ring shape is a stand-in for "a regular, symmetric k-NN graph" — the
//! same shape a k-NN graph over a tightly clustered, roughly evenly spaced
//! embedding tends toward — not a carefully chosen worst case. Both backends
//! see byte-identical edge lists at each `n`.

use ruvector_mincut::{DynamicGraph, DynamicMinCut, MinCutConfig};
use std::time::Instant;

fn ring_neighbors(n: usize, k: usize) -> Vec<(usize, Vec<(usize, f64)>)> {
    (0..n)
        .map(|i| {
            let nbrs: Vec<(usize, f64)> = (1..=k)
                .map(|d| ((i + d) % n, 0.1 + (d as f64) * 0.01))
                .collect();
            (i, nbrs)
        })
        .collect()
}

fn main() {
    // 800/2000 extend past ADR-345's original 400-vertex ceiling to
    // characterize the new backend's own scaling limit (see Next Research):
    // production agent-memory corpora are commonly thousands of entries, and
    // the 84-entry acceptance benchmark corpus was itself capped by the old
    // backend's cost, not a target size.
    let sizes = [19usize, 50, 100, 200, 400, 800, 2000];
    let k = 8usize;

    println!(
        "{:<6} {:>14} {:>14} {:>16} {:>16} {:>10}",
        "n",
        "GA build(ms)",
        "GA partition(ms)",
        "DMC build+cut(ms)",
        "DMC .partition(ms)",
        "speedup"
    );
    for &n in &sizes {
        let neighbors = ring_neighbors(n, k);

        let t0 = Instant::now();
        let mut analyzer = ruvector_mincut::RuVectorGraphAnalyzer::from_knn(&neighbors);
        let ga_build_elapsed = t0.elapsed();

        let t1 = Instant::now();
        let _ = analyzer.partition();
        let ga_partition_elapsed = t1.elapsed();
        let ga_total = ga_build_elapsed + ga_partition_elapsed;

        let t2 = Instant::now();
        let graph = DynamicGraph::new();
        for (i, nbrs) in &neighbors {
            for &(j, dist) in nbrs {
                let weight = if dist > 0.0 { 1.0 / dist } else { 1.0 };
                let _ = graph.insert_edge(*i as u64, j as u64, weight);
            }
        }
        let mincut = DynamicMinCut::from_graph(graph, MinCutConfig::default())
            .expect("valid graph builds a DynamicMinCut");
        let dmc_build_elapsed = t2.elapsed();

        let t3 = Instant::now();
        let _ = mincut.partition();
        let dmc_partition_elapsed = t3.elapsed();

        let speedup = ga_total.as_secs_f64() / dmc_build_elapsed.as_secs_f64().max(1e-9);

        println!(
            "n={n:<4} {:>13.3} {:>17.3} {:>19.3} {:>19.6} {:>9.1}x",
            ga_build_elapsed.as_secs_f64() * 1000.0,
            ga_partition_elapsed.as_secs_f64() * 1000.0,
            dmc_build_elapsed.as_secs_f64() * 1000.0,
            dmc_partition_elapsed.as_secs_f64() * 1000.0,
            speedup
        );
    }
}
