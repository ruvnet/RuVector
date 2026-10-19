//! Regression tests for `RuVectorGraphAnalyzer::{min_cut_exact, partition_exact}`.
//!
//! Background: `docs/research/nightly/2026-09-05-mincut-gated-forgetting` and
//! `docs/research/nightly/2026-09-11-mincut-partition-determinism` found that
//! `RuVectorGraphAnalyzer::min_cut()`/`partition()` (backed by
//! `MinCutWrapper`'s `BoundedInstance` LocalKCut-oracle search) scale
//! worse-than-quadratically with vertex count (~77ms at n=50, ~11.4s at
//! n=400) and are not guaranteed to find the *global* minimum cut — only a
//! cut within a configured `[lambda_min, lambda_max]` budget range found by a
//! seeded local search. `min_cut_exact`/`partition_exact` instead reuse
//! `DynamicMinCut`'s existing sparse Stoer-Wagner solver
//! (`crate::algorithm::exact::minimum_cut`), which is exact and polynomial.
//!
//! These tests establish the correctness contract the scaling/benchmark
//! comparisons in `examples/` and in `ruvector-agent-memory`'s
//! `mincut_gated_forgetting_bench` rely on: the exact path must (a) find the
//! textbook-correct global minimum cut weight, verified against an
//! independent brute-force enumeration, and (b) be deterministic across
//! repeated calls on a byte-identical graph.

use ruvector_mincut::graph::DynamicGraph;
use ruvector_mincut::RuVectorGraphAnalyzer;
use std::sync::Arc;

/// Brute-force global minimum cut weight by enumerating every non-trivial
/// vertex bipartition, O(2^n). Independent of anything under test: used only
/// as ground truth for small graphs.
fn brute_force_min_cut_weight(n: usize, edges: &[(usize, usize, f64)]) -> f64 {
    assert!(
        n >= 2 && n <= 20,
        "brute force is only for small test graphs"
    );
    let mut best = f64::INFINITY;
    for mask in 1u32..(1u32 << n) - 1 {
        let mut cut = 0.0;
        for &(u, v, w) in edges {
            let side_u = (mask >> u) & 1;
            let side_v = (mask >> v) & 1;
            if side_u != side_v {
                cut += w;
            }
        }
        if cut < best {
            best = cut;
        }
    }
    best
}

fn analyzer_from_edges(n: usize, edges: &[(usize, usize, f64)]) -> RuVectorGraphAnalyzer {
    let graph = Arc::new(DynamicGraph::new());
    for v in 0..n as u64 {
        graph.add_vertex(v);
    }
    for &(u, v, w) in edges {
        graph.insert_edge(u as u64, v as u64, w).unwrap();
    }
    RuVectorGraphAnalyzer::new(graph)
}

fn induced_cut_weight(edges: &[(usize, usize, f64)], side_a: &[u64]) -> f64 {
    let side_a: std::collections::HashSet<u64> = side_a.iter().copied().collect();
    edges
        .iter()
        .filter(|&&(u, v, _)| side_a.contains(&(u as u64)) != side_a.contains(&(v as u64)))
        .map(|&(_, _, w)| w)
        .sum()
}

/// Two 9-cliques (within-cluster weight 10.0, deliberately heavy so a wrong
/// cut is obviously wrong) joined by a single bridge edge of weight 1.0 — a
/// graph whose true global minimum cut is unambiguous: exactly the bridge.
fn bridge_graph(cluster_size: usize) -> (usize, Vec<(usize, usize, f64)>) {
    let n = cluster_size * 2;
    let mut edges = Vec::new();
    for cluster in 0..2 {
        let base = cluster * cluster_size;
        for i in 0..cluster_size {
            for j in (i + 1)..cluster_size {
                edges.push((base + i, base + j, 10.0));
            }
        }
    }
    edges.push((0, cluster_size, 1.0));
    (n, edges)
}

#[test]
fn partition_exact_finds_textbook_minimum_cut_on_random_graphs() {
    // Deterministic pseudo-random weighted graphs, n small enough for a
    // 2^n brute-force cross-check (n<=12 keeps this well under a second).
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    for n in [4usize, 6, 8, 10, 12] {
        let mut weight: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::new();
        for u in 0..n {
            for v in (u + 1)..n {
                // ~60% edge density, weights in [1, 10].
                if next() % 5 < 3 {
                    let w = 1.0 + (next() % 10) as f64;
                    *weight.entry((u, v)).or_insert(0.0) += w;
                }
            }
        }
        // Guarantee connectivity with a cheap spanning path so brute force
        // and the exact solver are both well-defined.
        for u in 0..n - 1 {
            *weight.entry((u, u + 1)).or_insert(0.0) += 1.0;
        }
        let edges: Vec<(usize, usize, f64)> =
            weight.into_iter().map(|((u, v), w)| (u, v, w)).collect();

        let expected = brute_force_min_cut_weight(n, &edges);
        let analyzer = analyzer_from_edges(n, &edges);
        let (side_a, _side_b) = analyzer
            .partition_exact()
            .unwrap_or_else(|| panic!("n={n}: partition_exact returned None"));
        let measured = induced_cut_weight(&edges, &side_a);

        assert!(
            (measured - expected).abs() < 1e-9,
            "n={n}: partition_exact cut weight {measured} != brute-force ground truth {expected}"
        );

        let value = analyzer.min_cut_exact();
        assert!(
            (value - expected).abs() < 1e-9,
            "n={n}: min_cut_exact() {value} != brute-force ground truth {expected}"
        );
    }
}

#[test]
fn partition_exact_isolates_the_unique_bridge() {
    let (n, edges) = bridge_graph(9);
    let analyzer = analyzer_from_edges(n, &edges);

    let expected = brute_force_min_cut_weight(n, &edges);
    assert_eq!(expected, 1.0, "test construction sanity check");

    let value = analyzer.min_cut_exact();
    assert!(
        (value - 1.0).abs() < 1e-9,
        "expected the unique bridge (weight 1.0) as the global min cut, got {value}"
    );

    let (side_a, side_b) = analyzer.partition_exact().unwrap();
    assert_eq!(side_a.len() + side_b.len(), n);
    // The bridge is the only min cut, so the two sides must exactly match
    // the two 9-cliques (in some order).
    let mut sides = [side_a, side_b];
    sides.sort_by_key(|s| s.len());
    let mut expected_cluster_a: Vec<u64> = (0..9).collect();
    let mut expected_cluster_b: Vec<u64> = (9..18).collect();
    let mut got_a = sides[0].clone();
    let mut got_b = sides[1].clone();
    got_a.sort_unstable();
    got_b.sort_unstable();
    expected_cluster_a.sort_unstable();
    expected_cluster_b.sort_unstable();
    assert!(
        (got_a == expected_cluster_a && got_b == expected_cluster_b)
            || (got_a == expected_cluster_b && got_b == expected_cluster_a),
        "exact partition did not isolate the two 9-cliques along the bridge: {got_a:?} / {got_b:?}"
    );
}

/// Same determinism contract `determinism_tests.rs` establishes for
/// `partition()`: repeated calls on a byte-identical graph return the exact
/// same sides, not just the same cut *value*.
#[test]
fn partition_exact_is_stable_across_repeated_calls() {
    let (n, edges) = bridge_graph(6);

    let mut first: Option<(Vec<u64>, Vec<u64>)> = None;
    for trial in 0..20 {
        let analyzer = analyzer_from_edges(n, &edges);
        let (mut side_a, mut side_b) = analyzer
            .partition_exact()
            .expect("partition_exact should succeed on a connected graph");
        side_a.sort_unstable();
        side_b.sort_unstable();
        let canon = if side_a.contains(&0) {
            (side_a, side_b)
        } else {
            (side_b, side_a)
        };
        match &first {
            None => first = Some(canon),
            Some(expected) => assert_eq!(
                &canon, expected,
                "trial {trial}: partition_exact differs from trial 0's result"
            ),
        }
    }
}

#[test]
fn partition_exact_returns_none_below_two_vertices() {
    let graph = Arc::new(DynamicGraph::new());
    graph.add_vertex(0);
    let analyzer = RuVectorGraphAnalyzer::new(graph);
    assert_eq!(analyzer.partition_exact(), None);
}
