//! Deterministic static global minimum cut (Stoer-Wagner), O(V^3).
//!
//! ## Why this module exists
//!
//! `docs/research/nightly/2026-09-05-mincut-gated-forgetting` (ADR-345) measured
//! [`crate::integration::RuVectorGraphAnalyzer::partition()`] at 76ms-11.4s per
//! call on graphs of 50-400 vertices, and non-deterministic across repeated
//! calls on a *byte-identical* 19-vertex graph (empty/degenerate result in 15
//! of 30 trials). That ADR filed the root cause as unresolved: "consistent
//! with internal tie-breaking that depends on hash-map iteration order rather
//! than any property of the graph".
//!
//! Tracing it further: [`crate::wrapper::MinCutWrapper`] implements the
//! bounded-range *dynamic* instance ladder from arxiv:2512.13105 — up to 100
//! geometrically-scaled sub-instances, each replaying the *entire* edge set
//! on first use (`process_instances`'s `is_new_instance` branch). That
//! machinery amortizes well across many incremental edge insert/delete
//! events on one long-lived graph. It is a poor fit for
//! `RuVectorGraphAnalyzer::from_knn(&neighbors).partition()` call sites (this
//! crate's own doc comments list `ruvector-agent-memory` compaction,
//! `CommunityDetector`, and `GraphPartitioner` as such call sites) which
//! discard the graph and rebuild it from scratch on every call — every call
//! pays the full multi-instance replay cost with nothing amortized.
//! Separately, [`crate::graph::DynamicGraph`] stores edges in `dashmap`
//! `DashMap`s, each constructed with a fresh randomized `RandomState` seed
//! (`DynamicGraph::new()` -> `DashMap::new()`); `graph.edges()`'s iteration
//! order therefore differs across otherwise-identical `DynamicGraph`
//! instances, and the wrapper's instance-construction code iterates
//! `graph.edges()` directly with no intervening sort — explaining the
//! observed non-determinism without requiring an intentional randomized
//! algorithm anywhere in the call chain.
//!
//! This module is a **separate, non-incremental code path**: the classical
//! Stoer-Wagner global min-cut algorithm (Stoer & Wagner, *A Simple
//! Min-Cut Algorithm*, J. ACM 1997), O(V^3), over a dense weight matrix built
//! once from `graph.edges()` **sorted by canonical endpoints and edge id**
//! before any use — so its result depends only on the graph's structure, not
//! on any hash map's iteration order. It has no update API and no cache: it
//! is meant exactly for the "rebuild the graph, ask once" pattern the
//! dynamic ladder is mismatched to, not as a replacement for
//! [`crate::wrapper::MinCutWrapper`]'s genuinely incremental use cases.

use crate::graph::{DynamicGraph, VertexId};
use std::collections::{HashMap, HashSet};

/// Result of a static min-cut computation: the cut value and the two sides
/// of the graph's global minimum cut (all vertices partitioned into exactly
/// one side).
#[derive(Debug, Clone, PartialEq)]
pub struct StaticCutResult {
    /// Total weight of edges crossing the cut.
    pub cut_value: f64,
    pub side_a: Vec<VertexId>,
    pub side_b: Vec<VertexId>,
}

/// Compute the global minimum cut of `graph` via Stoer-Wagner.
///
/// Returns `None` for graphs with fewer than 2 vertices (no cut is defined).
/// For a disconnected graph, returns `cut_value == 0.0` with `side_a`/`side_b`
/// a valid separation between two components.
///
/// O(V^3) time, O(V^2) space. Deterministic: repeated calls on an unchanged
/// graph (even a freshly-rebuilt `DynamicGraph` with different internal hash
/// seeds) return byte-identical results, because vertex and edge order is
/// fixed by sorting before the algorithm ever runs.
pub fn stoer_wagner_min_cut(graph: &DynamicGraph) -> Option<StaticCutResult> {
    let mut vertex_ids: Vec<VertexId> = graph.vertices();
    vertex_ids.sort_unstable();
    let n = vertex_ids.len();
    if n < 2 {
        return None;
    }

    let index_of: HashMap<VertexId, usize> = vertex_ids
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    let mut w = vec![vec![0.0f64; n]; n];
    let mut edges = graph.edges();
    // Sort so the weight matrix is built in a fixed order regardless of the
    // backing DashMap's iteration order. Parallel edges (shouldn't occur via
    // `DynamicGraph::insert_edge`'s dedup, but tolerated here) sum weights.
    edges.sort_unstable_by_key(|e| (e.canonical_endpoints(), e.id));
    for e in &edges {
        if e.source == e.target {
            continue; // no self-loops in a cut
        }
        if let (Some(&i), Some(&j)) = (index_of.get(&e.source), index_of.get(&e.target)) {
            w[i][j] += e.weight;
            w[j][i] += e.weight;
        }
    }

    let (cut_value, side_b_local) = stoer_wagner_dense(&mut w, n);
    let side_b_set: HashSet<usize> = side_b_local.into_iter().collect();

    let mut side_a = Vec::with_capacity(n - side_b_set.len());
    let mut side_b = Vec::with_capacity(side_b_set.len());
    for (i, &vid) in vertex_ids.iter().enumerate() {
        if side_b_set.contains(&i) {
            side_b.push(vid);
        } else {
            side_a.push(vid);
        }
    }

    Some(StaticCutResult {
        cut_value,
        side_a,
        side_b,
    })
}

/// Classical Stoer-Wagner min-cut over a dense symmetric weight matrix `w`
/// (`n x n`, diagonal unused, mutated in place by vertex-merge steps).
///
/// Returns `(cut_value, side_local_indices)`: `side_local_indices` are the
/// local indices (`0..n`) on one side of the best cut found; the complement
/// is the other side.
///
/// Deterministic by construction: `active` starts as `0..n` (ascending) and
/// is only ever shrunk via `Vec::retain`, so it stays ascending throughout;
/// every "pick the best candidate" step below scans it in that fixed order
/// and keeps the *first* maximum, so ties always resolve to the
/// lowest-indexed vertex rather than to map/set iteration order.
fn stoer_wagner_dense(w: &mut [Vec<f64>], n: usize) -> (f64, Vec<usize>) {
    let mut merged_of: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active: Vec<usize> = (0..n).collect();

    let mut best_cut = f64::INFINITY;
    let mut best_side: Vec<usize> = Vec::new();

    while active.len() > 1 {
        let (s, t, cut_of_phase) = min_cut_phase(w, &active);
        if cut_of_phase < best_cut {
            best_cut = cut_of_phase;
            best_side = merged_of[t].clone();
        }

        // Merge t into s: fold t's edges into s, then drop t from `active`.
        for &v in &active {
            if v != s && v != t {
                w[s][v] += w[t][v];
                w[v][s] += w[v][t];
            }
        }
        let t_members = std::mem::take(&mut merged_of[t]);
        merged_of[s].extend(t_members);
        active.retain(|&v| v != t);
    }

    if best_side.is_empty() && n >= 2 {
        // Degenerate all-zero-weight graph (e.g. an isolated-vertex pair):
        // every phase ties at cut_of_phase == 0.0, which is not `< INFINITY`...
        // actually 0.0 < INFINITY is true, so this branch is unreachable in
        // practice; kept only as a defensive fallback so callers never see
        // an empty side for n >= 2.
        best_side = vec![*active.last().unwrap_or(&(n - 1))];
    }

    (best_cut, best_side)
}

/// One "minimum cut phase": grow a set `A` from an arbitrary start vertex by
/// repeatedly adding the remaining vertex most tightly connected to `A`,
/// until all of `active` has been absorbed. Returns `(s, t, cut_of_phase)`
/// where `t` is the last vertex added, `s` the second-to-last, and
/// `cut_of_phase` is the total edge weight between `t` and `A \ {t}` — the
/// weight of the cut that isolates `t` from the rest of `active` in this
/// phase.
fn min_cut_phase(w: &[Vec<f64>], active: &[usize]) -> (usize, usize, f64) {
    let n = w.len();
    let mut in_a = vec![false; n];
    let mut weights_to_a = vec![0.0f64; n];

    let first = active[0];
    in_a[first] = true;
    for &v in active {
        if v != first {
            weights_to_a[v] = w[first][v];
        }
    }

    let mut order = vec![first];
    let mut last_cut_weight = 0.0f64;

    for _ in 1..active.len() {
        let mut best_v = None;
        let mut best_w = f64::NEG_INFINITY;
        for &v in active {
            if !in_a[v] && weights_to_a[v] > best_w {
                best_w = weights_to_a[v];
                best_v = Some(v);
            }
        }
        let z = best_v.expect("active set has an un-added vertex while len() > order.len()");
        in_a[z] = true;
        order.push(z);
        last_cut_weight = best_w;
        for &v in active {
            if !in_a[v] {
                weights_to_a[v] += w[z][v];
            }
        }
    }

    let t = order[order.len() - 1];
    let s = order[order.len() - 2];
    (s, t, last_cut_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn triangle_min_cut_is_two() {
        let graph = Arc::new(DynamicGraph::new());
        graph.insert_edge(0, 1, 1.0).unwrap();
        graph.insert_edge(1, 2, 1.0).unwrap();
        graph.insert_edge(2, 0, 1.0).unwrap();

        let result = stoer_wagner_min_cut(&graph).unwrap();
        assert_eq!(result.cut_value, 2.0);
        assert_eq!(result.side_a.len() + result.side_b.len(), 3);
        assert!(!result.side_a.is_empty() && !result.side_b.is_empty());
    }

    #[test]
    fn weighted_triangle_matches_min_edge_pair() {
        // Min cut of a weighted triangle is the sum of the two smallest
        // edge weights (isolate the vertex between them): 2+3=5.
        let graph = Arc::new(DynamicGraph::new());
        graph.insert_edge(1, 2, 5.0).unwrap();
        graph.insert_edge(2, 3, 3.0).unwrap();
        graph.insert_edge(3, 1, 2.0).unwrap();

        let result = stoer_wagner_min_cut(&graph).unwrap();
        assert_eq!(result.cut_value, 5.0);
    }

    #[test]
    fn disconnected_graph_has_zero_cut() {
        let graph = Arc::new(DynamicGraph::new());
        graph.insert_edge(1, 2, 1.0).unwrap();
        graph.insert_edge(3, 4, 1.0).unwrap();

        let result = stoer_wagner_min_cut(&graph).unwrap();
        assert_eq!(result.cut_value, 0.0);
        assert_eq!(result.side_a.len() + result.side_b.len(), 4);
    }

    #[test]
    fn fewer_than_two_vertices_returns_none() {
        let empty = Arc::new(DynamicGraph::new());
        assert!(stoer_wagner_min_cut(&empty).is_none());

        let one = Arc::new(DynamicGraph::new());
        one.insert_edge(0, 0, 1.0).ok(); // self-loop is rejected by the graph
        assert!(stoer_wagner_min_cut(&one).is_none());
    }

    /// Two 9-vertex cliques joined only through a degree-2 relay ("bridge")
    /// vertex — the same shape as `ruvector-agent-memory::graph_forget`'s
    /// `bridge_dataset()` fixture (a `bridge` vertex linked to one "gateway"
    /// member per clique, both gateways otherwise ordinary clique members).
    ///
    /// Because the relay's two edges are each individually a graph-theoretic
    /// cut edge (either one's removal disconnects a whole clique from the
    /// rest — the relay is not itself a 2-edge-connected hub, just the sole
    /// path between two otherwise-disjoint cliques), the graph's *global*
    /// minimum cut is 1.0 (sever either single edge, isolating one 9-vertex
    /// clique), not 2.0 (isolating the relay alone) — matching this same
    /// tie in `ruvector-agent-memory`'s `bridge_dataset()`. Either way the
    /// relay always has an edge crossing the found partition, which is the
    /// property `graph_forget::MincutGatedForgetting` actually depends on.
    fn two_cliques_with_bridge() -> (Arc<DynamicGraph>, VertexId) {
        let graph = Arc::new(DynamicGraph::new());
        let clique_a: Vec<VertexId> = (0..9).collect();
        let clique_b: Vec<VertexId> = (9..18).collect();
        let bridge: VertexId = 18;

        for i in 0..clique_a.len() {
            for j in (i + 1)..clique_a.len() {
                graph.insert_edge(clique_a[i], clique_a[j], 1.0).unwrap();
            }
        }
        for i in 0..clique_b.len() {
            for j in (i + 1)..clique_b.len() {
                graph.insert_edge(clique_b[i], clique_b[j], 1.0).unwrap();
            }
        }
        graph.insert_edge(clique_a[0], bridge, 1.0).unwrap();
        graph.insert_edge(clique_b[0], bridge, 1.0).unwrap();

        (graph, bridge)
    }

    #[test]
    fn bridge_vertex_always_sits_on_the_minimum_cut_boundary() {
        let (graph, bridge) = two_cliques_with_bridge();
        let edges = graph.edges();
        let result = stoer_wagner_min_cut(&graph).unwrap();

        // Severing either of the relay's two individual edges disconnects a
        // whole 9-vertex clique for cost 1.0; that's cheaper than isolating
        // the relay itself (cost 2.0), so 1.0 is the true global minimum.
        assert_eq!(result.cut_value, 1.0);

        let side_a: HashSet<VertexId> = result.side_a.iter().copied().collect();
        let bridge_in_a = side_a.contains(&bridge);
        let has_crossing_neighbor = edges.iter().any(|e| {
            let (u, v) = (e.source, e.target);
            let touches_bridge = u == bridge || v == bridge;
            let other = if u == bridge { v } else { u };
            touches_bridge && side_a.contains(&other) != bridge_in_a
        });
        assert!(
            has_crossing_neighbor,
            "the relay vertex must have at least one neighbor edge crossing the minimum cut"
        );
    }

    #[test]
    fn deterministic_across_repeated_fresh_graphs() {
        // Rebuild the graph from scratch each time (fresh DashMaps, fresh
        // random hash seeds) exactly like `RuVectorGraphAnalyzer::from_knn`
        // call sites do on every compaction/community-detection call. The
        // ADR-345 finding this module fixes: the dynamic wrapper's result
        // varied across calls on graphs built this way.
        let mut results = Vec::new();
        for _ in 0..25 {
            let (graph, _bridge) = two_cliques_with_bridge();
            results.push(stoer_wagner_min_cut(&graph).unwrap());
        }
        for r in &results[1..] {
            assert_eq!(r, &results[0], "static cut must be identical across independently-constructed, structurally-identical graphs");
        }
    }

    /// Builds a fixed-degree ring k-NN graph: vertex `i` connects to the
    /// next `k` vertices mod `n`, weight increasing with ring distance. Pure
    /// function of `(n, k)` — no RNG, so two calls with the same arguments
    /// always produce a structurally identical (if freshly-allocated, with
    /// independent DashMap hash seeds) `DynamicGraph`. Same shape as
    /// `ruvector-agent-memory`'s `examples/mincut_scaling_probe.rs`, reused
    /// here as a frozen corpus definition rather than a throwaway probe.
    fn frozen_ring_graph(n: usize, k: usize) -> Arc<DynamicGraph> {
        let graph = Arc::new(DynamicGraph::new());
        for i in 0..n {
            for d in 1..=k {
                let j = ((i + d) % n) as VertexId;
                let _ = graph.insert_edge(i as VertexId, j, 0.1 + d as f64 * 0.01);
            }
        }
        graph
    }

    /// Promotion-gate regression test (added in response to the 2026-09-09
    /// review on PR #972, which found the single-fixture determinism check
    /// insufficient evidence for "representative scaling" and asked for
    /// "deterministic cut/value parity on a frozen multi-size corpus"
    /// before promotion): a fixed, checked-in, non-random corpus definition
    /// ([`frozen_ring_graph`]) at five sizes spanning two orders of
    /// magnitude, each rebuilt from scratch (fresh `DynamicGraph`, fresh
    /// `DashMap` hash seeds — exactly the non-determinism ADR-345 measured
    /// in the dynamic engine) five independent times, asserting the full
    /// `StaticCutResult` (cut value *and* both partition sides, in order)
    /// is byte-identical across every rebuild at every size.
    #[test]
    fn frozen_multi_size_corpus_cut_value_and_partition_are_stable() {
        const SIZES: [usize; 5] = [10, 19, 50, 100, 250];
        const K: usize = 4;
        const REBUILDS: usize = 5;

        for &n in &SIZES {
            let k = K.min(n - 1);
            let first = stoer_wagner_min_cut(&frozen_ring_graph(n, k))
                .unwrap_or_else(|| panic!("n={n} >= 2 always yields a cut"));
            for trial in 1..REBUILDS {
                let again = stoer_wagner_min_cut(&frozen_ring_graph(n, k))
                    .unwrap_or_else(|| panic!("n={n} >= 2 always yields a cut"));
                assert_eq!(
                    again, first,
                    "n={n}, rebuild #{trial}: cut value and partition must be identical across independently-constructed, structurally-identical graphs"
                );
            }
        }
    }
}
