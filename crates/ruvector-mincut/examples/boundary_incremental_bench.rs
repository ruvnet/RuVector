//! Nightly research benchmark (2026-10-03, ADR-352): incremental boundary
//! maintenance in `DeterministicLocalKCut::deterministic_bfs`.
//!
//! Background: the 2026-09-05 nightly (ADR-345) measured
//! `RuVectorGraphAnalyzer::partition()` scaling from ~77ms (n=50) to ~11.4s
//! (n=400) on a ring k-NN graph, and rejected `MincutGatedForgetting` partly
//! on that basis. The 2026-09-11 nightly (ADR-346) fixed a separate
//! determinism bug in the same code path but explicitly left latency
//! "out of scope ... a distinct algorithmic-complexity problem", naming
//! `search_for_cuts`'s seed/budget fan-out as the likely culprit and flagging
//! it as unresolved future work.
//!
//! Root cause (this run): `deterministic_bfs`'s original implementation
//! called `calculate_boundary(graph, &visited)` — an O(edges incident to
//! `visited`) full rescan — once per BFS depth, for every depth in
//! `0..=radius` (`radius` defaults to 20). That cost is paid again for every
//! `(seed, budget)` pair `BoundedInstance::search_for_cuts` tries and again
//! for every one of the O(log n) `MinCutWrapper` range instances, which
//! compounds the redundant rescanning multiplicatively with graph size.
//!
//! Fix: maintain the boundary-edge set incrementally as each new BFS layer
//! is added (`update_boundary_incremental`), touching only edges incident to
//! the vertices newly added at each layer instead of the whole visited set.
//! This computes the exact same boundary values (see the
//! `test_incremental_boundary_matches_from_scratch_across_random_graphs`
//! regression test in `localkcut::paper_impl`) in O(edges touched by the
//! whole BFS) total instead of O(radius * edges incident to visited).
//!
//! This benchmark reproduces that specific cost difference directly: `OLD`
//! is a faithful, unmodified copy of the pre-fix `deterministic_bfs` (full
//! `calculate_boundary` rescan per depth), built only against
//! `ruvector_mincut`'s public `DynamicGraph`/`VertexId`/`EdgeId` API so it
//! exercises the exact same graph representation; `NEW` calls the actual
//! (now-fixed) `DeterministicLocalKCut::search` through the public
//! `LocalKCutOracle` trait. Both run the identical queries against the
//! identical graphs and are asserted to return identical cut values.
//!
//! Run:
//!   cargo run --release -p ruvector-mincut --example boundary_incremental_bench

use ruvector_mincut::localkcut::paper_impl::{
    DeterministicLocalKCut, LocalKCutOracle, LocalKCutQuery, LocalKCutResult,
};
use ruvector_mincut::{DynamicGraph, EdgeId, VertexId};
use std::collections::HashSet;
use std::time::Instant;

/// Faithful copy of the pre-fix `deterministic_bfs`: recomputes the boundary
/// of the whole `visited` set from scratch at every BFS depth.
fn old_calculate_boundary(graph: &DynamicGraph, vertex_set: &HashSet<VertexId>) -> u64 {
    let mut boundary_edges: HashSet<EdgeId> = HashSet::new();
    for &v in vertex_set {
        for (neighbor, edge_id) in graph.neighbors(v) {
            if !vertex_set.contains(&neighbor) {
                boundary_edges.insert(edge_id);
            }
        }
    }
    boundary_edges.len() as u64
}

fn old_deterministic_bfs(
    graph: &DynamicGraph,
    seeds: &[VertexId],
    budget: u64,
    radius: usize,
) -> Option<(HashSet<VertexId>, u64)> {
    if seeds.is_empty() {
        return None;
    }
    let mut visited = HashSet::new();
    let mut best_cut: Option<(HashSet<VertexId>, u64)> = None;

    for &seed in seeds {
        if graph.has_vertex(seed) {
            visited.insert(seed);
        }
    }
    if visited.is_empty() {
        return None;
    }

    let mut current_layer = visited.clone();

    for depth in 0..=radius {
        let boundary_size = old_calculate_boundary(graph, &visited);

        if boundary_size <= budget && !visited.is_empty() && visited.len() < graph.num_vertices() {
            let should_update = match &best_cut {
                None => true,
                Some((_, prev)) => boundary_size < *prev,
            };
            if should_update {
                best_cut = Some((visited.clone(), boundary_size));
            }
        }

        if let Some((_, boundary)) = &best_cut {
            if *boundary == 0 {
                break;
            }
        }
        if depth >= radius {
            break;
        }

        let mut next_layer = HashSet::new();
        let mut layer_vertices: Vec<_> = current_layer.iter().copied().collect();
        layer_vertices.sort_unstable();
        for v in layer_vertices {
            let mut neighbors: Vec<_> = graph
                .neighbors(v)
                .into_iter()
                .map(|(n, _)| n)
                .filter(|n| !visited.contains(n))
                .collect();
            neighbors.sort_unstable();
            for neighbor in neighbors {
                if visited.insert(neighbor) {
                    next_layer.insert(neighbor);
                }
            }
        }
        current_layer = next_layer;
        if current_layer.is_empty() {
            break;
        }
    }

    best_cut
}

/// Ring k-NN graph: vertex i connects to the next k vertices mod n. Same
/// shape as `mincut_scaling_probe.rs` (ADR-345's reproduction graph).
fn ring_knn_graph(n: usize, k: usize) -> DynamicGraph {
    let graph = DynamicGraph::new();
    for i in 0..n {
        for d in 1..=k {
            let j = (i + d) % n;
            if (i as u64) < (j as u64) {
                let _ = graph.insert_edge(i as u64, j as u64, 1.0);
            } else {
                let _ = graph.insert_edge(j as u64, i as u64, 1.0);
            }
        }
    }
    graph
}

fn main() {
    let sizes = [50usize, 100, 200, 400];
    let k = 8usize;
    let budget = 10u64;
    let radius = 20usize;

    println!("# boundary_incremental_bench: OLD (full rescan) vs NEW (incremental)");
    println!("# n, seed_count_tried, old_ms, new_ms, speedup, old_cut, new_cut, agree");

    for &n in &sizes {
        let graph = ring_knn_graph(n, k);
        let oracle = DeterministicLocalKCut::new(radius);

        // Exercise the same seed fan-out `search_for_cuts` performs: every
        // vertex tried as a singleton seed, matching its real call pattern.
        let seeds: Vec<VertexId> = (0..n as u64).collect();

        let t_old = Instant::now();
        let mut old_results = Vec::with_capacity(seeds.len());
        for &seed in &seeds {
            let r = old_deterministic_bfs(&graph, &[seed], budget, radius);
            old_results.push(r.map(|(_, b)| b));
        }
        let old_elapsed = t_old.elapsed();

        let t_new = Instant::now();
        let mut new_results = Vec::with_capacity(seeds.len());
        for &seed in &seeds {
            let query = LocalKCutQuery {
                seed_vertices: vec![seed],
                budget_k: budget,
                radius,
            };
            let r = match oracle.search(&graph, query) {
                LocalKCutResult::Found { cut_value, .. } => Some(cut_value),
                LocalKCutResult::NoneInLocality => None,
            };
            new_results.push(r);
        }
        let new_elapsed = t_new.elapsed();

        let agree = old_results == new_results;
        let old_cut = old_results.iter().flatten().min().copied();
        let new_cut = new_results.iter().flatten().min().copied();

        let old_ms = old_elapsed.as_secs_f64() * 1000.0;
        let new_ms = new_elapsed.as_secs_f64() * 1000.0;
        println!(
            "n={n:<5} seeds={:<5} old_ms={old_ms:>12.3} new_ms={new_ms:>10.3} speedup={:>8.1}x old_cut={old_cut:?} new_cut={new_cut:?} agree={agree}",
            seeds.len(),
            old_ms / new_ms.max(1e-9),
        );

        assert!(
            agree,
            "n={n}: OLD and NEW returned different per-seed cut values — \
             incremental boundary maintenance changed behavior"
        );
    }
}
