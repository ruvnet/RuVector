//! Correctness regression tests for `BoundedInstance::with_max_radius` and
//! `RuVectorGraphAnalyzer::with_radius` / `from_knn_with_radius`, added by
//! the 2026-10-30 nightly research run
//! (`docs/research/nightly/2026_10_30_mincut-local-kcut-radius-scaling`).
//!
//! These are regression tests against the crate's own production default
//! (`RuVectorGraphAnalyzer::new`, radius 20), not an independently verified
//! global minimum: the question under test is "does lowering the
//! `DeterministicLocalKCut` search radius change the reported min-cut
//! value on graphs small enough to check by hand", not "is the production
//! default globally optimal".

use ruvector_mincut::graph::DynamicGraph;
use ruvector_mincut::RuVectorGraphAnalyzer;
use std::sync::Arc;

fn path_graph(n: u64) -> Arc<DynamicGraph> {
    let graph = Arc::new(DynamicGraph::new());
    for i in 1..n {
        graph.insert_edge(i, i + 1, 1.0).unwrap();
    }
    graph
}

fn cycle_graph(n: u64) -> Arc<DynamicGraph> {
    let graph = Arc::new(DynamicGraph::new());
    for i in 1..n {
        graph.insert_edge(i, i + 1, 1.0).unwrap();
    }
    graph.insert_edge(n, 1, 1.0).unwrap();
    graph
}

/// Two 5-cliques joined by a single bridge edge: the unique minimum cut is
/// the bridge itself (value 1), independent of clique size, so this is a
/// sharp check that a radius-limited local search still finds it.
fn two_cliques_bridge() -> Arc<DynamicGraph> {
    let graph = Arc::new(DynamicGraph::new());
    // Clique A: vertices 1..=5
    for i in 1..=5u64 {
        for j in (i + 1)..=5u64 {
            graph.insert_edge(i, j, 1.0).unwrap();
        }
    }
    // Clique B: vertices 6..=10
    for i in 6..=10u64 {
        for j in (i + 1)..=10u64 {
            graph.insert_edge(i, j, 1.0).unwrap();
        }
    }
    // Bridge
    graph.insert_edge(5, 6, 1.0).unwrap();
    graph
}

fn assert_min_cut_matches_baseline(graph: Arc<DynamicGraph>, label: &str, radii: &[usize]) {
    let mut baseline = RuVectorGraphAnalyzer::new(Arc::clone(&graph));
    let baseline_value = baseline.min_cut();

    for &radius in radii {
        let mut variant = RuVectorGraphAnalyzer::with_radius(Arc::clone(&graph), radius);
        let variant_value = variant.min_cut();
        assert_eq!(
            variant_value, baseline_value,
            "{label}: radius={radius} min_cut={variant_value} != baseline(radius=20) min_cut={baseline_value}"
        );
    }
}

#[test]
fn path_graph_min_cut_stable_under_radius_reduction() {
    // Min cut of a path is 1 (any edge is a bridge).
    assert_min_cut_matches_baseline(path_graph(12), "path(12)", &[8, 4, 2]);
}

#[test]
fn cycle_graph_min_cut_stable_under_radius_reduction() {
    // Min cut of a cycle is 2.
    assert_min_cut_matches_baseline(cycle_graph(12), "cycle(12)", &[8, 4, 2]);
}

#[test]
fn bridged_cliques_min_cut_stable_under_radius_reduction() {
    // Min cut is exactly 1 (the bridge edge), independent of clique size;
    // a radius-limited local search must still find a BFS path to it from
    // a seed inside either clique.
    assert_min_cut_matches_baseline(two_cliques_bridge(), "two_cliques_bridge", &[8, 4, 3]);
}

#[test]
fn with_radius_and_from_knn_with_radius_agree() {
    // from_knn_with_radius is a convenience wrapper over with_radius via an
    // explicit edge list; both paths must produce the same analyzer state
    // for the same logical graph.
    let neighbors: Vec<(usize, Vec<(usize, f64)>)> = vec![
        (0, vec![(1, 1.0), (2, 1.0)]),
        (1, vec![(0, 1.0), (2, 1.0)]),
        (2, vec![(0, 1.0), (1, 1.0)]),
    ];

    let mut via_knn = RuVectorGraphAnalyzer::from_knn_with_radius(&neighbors, 4);
    let knn_cut = via_knn.min_cut();

    let graph = Arc::new(DynamicGraph::new());
    graph.insert_edge(0, 1, 1.0).unwrap();
    graph.insert_edge(0, 2, 1.0).unwrap();
    graph.insert_edge(1, 2, 1.0).unwrap();
    let mut via_direct = RuVectorGraphAnalyzer::with_radius(graph, 4);
    let direct_cut = via_direct.min_cut();

    assert_eq!(knn_cut, direct_cut);
}

#[test]
fn num_instances_touched_is_available_and_nonzero_after_query() {
    let mut analyzer = RuVectorGraphAnalyzer::with_radius(cycle_graph(10), 6);
    assert_eq!(analyzer.num_instances_touched(), 0, "no query yet");
    let _ = analyzer.min_cut();
    assert!(
        analyzer.num_instances_touched() >= 1,
        "min_cut() must instantiate at least one BoundedInstance"
    );
}

#[test]
fn default_radius_constant_matches_historical_hardcoded_value() {
    // BoundedInstance::new() must keep producing byte-identical behavior to
    // the pre-existing hardcoded `DeterministicLocalKCut::new(20)` — this
    // pins that invariant so a future refactor of the constant is caught.
    assert_eq!(
        ruvector_mincut::instance::BoundedInstance::DEFAULT_MAX_RADIUS,
        20
    );
}
