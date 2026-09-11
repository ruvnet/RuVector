//! Regression tests for run-to-run determinism of `RuVectorGraphAnalyzer::partition()`.
//!
//! Background: the 2026-09-05 nightly research run
//! (`docs/research/nightly/2026-09-05-mincut-gated-forgetting`) measured
//! `RuVectorGraphAnalyzer::partition()` returning an empty/degenerate result
//! in 15/30 repeated calls against a byte-identical 19-vertex graph, and
//! attributed it to "hash-map iteration-order-dependent tie-breaking" without
//! pinpointing the exact source. Root cause: `BoundedInstance` (the
//! `ProperCutInstance` used for small graphs, `<20` vertices) enumerates its
//! candidate cut vertices from a `HashSet<VertexId>`, whose iteration order
//! is not stable across process runs (Rust's default hasher is randomly
//! seeded per `HashSet`/`DashMap` instance, independent of insertion order).
//! That unstable order fed both a strict `<` tie-break in
//! `brute_force_min_cut` and the top-level `DynamicConnectivity` check via
//! `DynamicGraph::vertices()`/`edges()`, which is why the wrapper sometimes
//! reported the graph as disconnected (`MinCutResult::Disconnected`) and
//! `partition()` returned `Some((vec![], vec![]))`.
//!
//! These tests reproduce the exact topology from that probe
//! (`crates/ruvector-agent-memory/examples/mincut_determinism_probe.rs`) and
//! assert the fixed, deterministic behavior.

use ruvector_mincut::RuVectorGraphAnalyzer;

fn normalize3(v: [f64; 3]) -> Vec<f64> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    vec![v[0] / n, v[1] / n, v[2] / n]
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

/// Two 9-point clusters (one axis-aligned "plain" point repeated 8x, plus one
/// "gateway" point tilted toward the other cluster) plus a single equidistant
/// bridge point. This is the same topology the 2026-09-05 nightly used to
/// demonstrate non-determinism: it is symmetric across the two clusters, so
/// the minimum cut has ties that a stable algorithm must break the same way
/// every time.
fn two_cluster_bridge_neighbors() -> (usize, Vec<(usize, Vec<(usize, f64)>)>) {
    let mut entries: Vec<Vec<f64>> = Vec::new();
    for axis in 0..2 {
        let plain = if axis == 0 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let gateway = if axis == 0 {
            normalize3([1.0, 0.0, 0.5])
        } else {
            normalize3([0.0, 1.0, 0.5])
        };
        for _ in 0..8 {
            entries.push(plain.to_vec());
        }
        entries.push(gateway);
    }
    entries.push(vec![0.0, 0.0, 1.0]);
    let n = entries.len();
    let bridge_idx = n - 1;

    let k = 8usize;
    let min_sim = 0.05f64;
    let neighbors: Vec<(usize, Vec<(usize, f64)>)> = (0..n)
        .map(|i| {
            let mut sims: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cosine_sim(&entries[i], &entries[j])))
                .filter(|&(_, s)| s >= min_sim)
                .collect();
            sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sims.truncate(k);
            let dists = sims
                .into_iter()
                .map(|(j, s)| (j, (1.0 - s).max(1e-4)))
                .collect();
            (i, dists)
        })
        .collect();

    (bridge_idx, neighbors)
}

/// `partition()` must never return a degenerate (one side empty) result on a
/// connected graph: that previously happened in 15/30 trials.
#[test]
fn partition_is_never_degenerate_on_connected_graph() {
    let (_bridge_idx, neighbors) = two_cluster_bridge_neighbors();

    for trial in 0..30 {
        let mut analyzer = RuVectorGraphAnalyzer::from_knn(&neighbors);
        let (side_a, side_b) = analyzer
            .partition()
            .unwrap_or_else(|| panic!("trial {trial}: partition() returned None"));
        assert!(
            !side_a.is_empty() && !side_b.is_empty(),
            "trial {trial}: degenerate partition (one side empty) on a connected graph \
             (side_a={}, side_b={})",
            side_a.len(),
            side_b.len(),
        );
    }
}

/// Repeated `partition()` calls against a byte-identical graph must return
/// the *same* partition every time. This is the direct determinism
/// regression test for the fix in `DynamicGraph::vertices()`/`edges()` and
/// `BoundedInstance::{brute_force_min_cut, search_for_cuts}`.
#[test]
fn partition_is_stable_across_repeated_calls() {
    let (_bridge_idx, neighbors) = two_cluster_bridge_neighbors();

    let mut first: Option<(Vec<u64>, Vec<u64>)> = None;
    for trial in 0..30 {
        let mut analyzer = RuVectorGraphAnalyzer::from_knn(&neighbors);
        let (mut side_a, mut side_b) = analyzer.partition().expect("partition should succeed");
        side_a.sort_unstable();
        side_b.sort_unstable();
        // Canonicalize which side is "a" vs "b" by the side containing vertex 0.
        let canon = if side_a.contains(&0) {
            (side_a, side_b)
        } else {
            (side_b, side_a)
        };
        match &first {
            None => first = Some(canon),
            Some(expected) => assert_eq!(
                &canon, expected,
                "trial {trial}: partition differs from trial 0's result"
            ),
        }
    }
}

/// `DynamicGraph::vertices()`/`edges()` must return a canonical (sorted)
/// order regardless of `DashMap`'s internal iteration order.
#[test]
fn graph_vertices_and_edges_are_sorted() {
    use ruvector_mincut::graph::DynamicGraph;
    use std::sync::Arc;

    let graph = Arc::new(DynamicGraph::new());
    // Insert out of order on purpose.
    for &(u, v) in &[(5u64, 1u64), (0, 9), (3, 2), (8, 4), (7, 6)] {
        graph.insert_edge(u, v, 1.0).unwrap();
    }

    let vertices = graph.vertices();
    let mut sorted = vertices.clone();
    sorted.sort_unstable();
    assert_eq!(vertices, sorted, "vertices() must be returned in sorted order");

    let edges = graph.edges();
    let ids: Vec<u64> = edges.iter().map(|e| e.id).collect();
    let mut sorted_ids = ids.clone();
    sorted_ids.sort_unstable();
    assert_eq!(ids, sorted_ids, "edges() must be returned in EdgeId order");
}
