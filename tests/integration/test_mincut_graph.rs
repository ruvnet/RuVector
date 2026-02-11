//! Integration tests for ruvector-mincut: graph construction, min-cut computation,
//! dynamic updates, approximate mode, and edge cases.
//!
//! All tests use real types and real computations -- no mocks.

use ruvector_mincut::prelude::*;
use ruvector_mincut::{DynamicGraph, MinCutBuilder, MinCutResult};

// ===========================================================================
// 1. Triangle graph: min-cut value
// ===========================================================================
#[test]
fn triangle_graph_min_cut_is_two() {
    let mincut = MinCutBuilder::new()
        .exact()
        .with_edges(vec![(1, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0)])
        .build()
        .expect("build triangle graph");

    assert_eq!(
        mincut.min_cut_value(),
        2.0,
        "min-cut of a triangle with unit weights should be 2.0"
    );
}

// ===========================================================================
// 2. Dynamic edge insertion updates min-cut correctly
// ===========================================================================
#[test]
fn dynamic_insert_updates_min_cut() {
    let mut mincut = MinCutBuilder::new().build().expect("build empty graph");

    assert_eq!(
        mincut.min_cut_value(),
        f64::INFINITY,
        "empty graph should have infinite min-cut"
    );

    mincut.insert_edge(1, 2, 1.0).expect("insert 1-2");
    assert_eq!(
        mincut.min_cut_value(),
        1.0,
        "single edge graph min-cut should be 1.0"
    );

    mincut.insert_edge(2, 3, 1.0).expect("insert 2-3");
    assert_eq!(
        mincut.min_cut_value(),
        1.0,
        "path graph min-cut should be 1.0"
    );

    mincut.insert_edge(3, 1, 1.0).expect("insert 3-1");
    assert_eq!(
        mincut.min_cut_value(),
        2.0,
        "triangle min-cut should be 2.0 after closing the cycle"
    );
}

// ===========================================================================
// 3. Edge deletion reduces min-cut
// ===========================================================================
#[test]
fn edge_deletion_reduces_min_cut() {
    let mut mincut = MinCutBuilder::new()
        .with_edges(vec![(1, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0)])
        .build()
        .expect("build triangle");

    assert_eq!(mincut.min_cut_value(), 2.0);

    mincut.delete_edge(3, 1).expect("delete edge 3-1");
    assert_eq!(
        mincut.min_cut_value(),
        1.0,
        "min-cut should decrease to 1.0 after removing an edge from the triangle"
    );
}

// ===========================================================================
// 4. Disconnected graph has zero min-cut
// ===========================================================================
#[test]
fn disconnected_graph_has_zero_min_cut() {
    let mincut = MinCutBuilder::new()
        .with_edges(vec![(1, 2, 1.0), (3, 4, 1.0)])
        .build()
        .expect("build disconnected graph");

    assert!(!mincut.is_connected(), "graph should be disconnected");
    assert_eq!(
        mincut.min_cut_value(),
        0.0,
        "disconnected graph min-cut should be 0.0"
    );
}

// ===========================================================================
// 5. Weighted graph: min-cut respects weights
// ===========================================================================
#[test]
fn weighted_graph_min_cut() {
    // Triangle with weights 5, 3, 2
    // Cutting node 3 removes edges with weights 3+2=5
    // Cutting node 1 removes edges 5+2=7
    // Cutting node 2 removes edges 5+3=8
    // min cut = 5.0
    let mincut = MinCutBuilder::new()
        .with_edges(vec![(1, 2, 5.0), (2, 3, 3.0), (3, 1, 2.0)])
        .build()
        .expect("build weighted triangle");

    assert_eq!(
        mincut.min_cut_value(),
        5.0,
        "weighted triangle min-cut should be 5.0"
    );
}

// ===========================================================================
// 6. Approximate mode returns non-exact result
// ===========================================================================
#[test]
fn approximate_mode_non_exact() {
    let mincut = MinCutBuilder::new()
        .approximate(0.1)
        .with_edges(vec![(1, 2, 1.0), (2, 3, 1.0)])
        .build()
        .expect("build approximate graph");

    let result = mincut.min_cut();
    assert!(
        !result.is_exact,
        "approximate mode should not claim exact results"
    );
    assert!(
        (result.approximation_ratio - 1.1).abs() < 1e-6,
        "approximation ratio should be 1+epsilon = 1.1"
    );
}

// ===========================================================================
// 7. Min-cut result includes partition
// ===========================================================================
#[test]
fn min_cut_result_has_partition() {
    let mincut = MinCutBuilder::new()
        .with_edges(vec![(1, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0)])
        .build()
        .expect("build triangle");

    let result = mincut.min_cut();
    assert!(result.partition.is_some(), "result should include a partition");

    if let Some((s, t)) = result.partition {
        assert_eq!(
            s.len() + t.len(),
            3,
            "partition should cover all 3 vertices"
        );
        assert!(!s.is_empty(), "partition S should not be empty");
        assert!(!t.is_empty(), "partition T should not be empty");
    }
}

// ===========================================================================
// 8. Large path graph: linear chain has min-cut 1
// ===========================================================================
#[test]
fn large_path_graph_min_cut_is_one() {
    let n = 200;
    let edges: Vec<(u64, u64, f64)> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();

    let mincut = MinCutBuilder::new()
        .with_edges(edges)
        .build()
        .expect("build path graph");

    assert_eq!(mincut.num_vertices(), n as usize);
    assert_eq!(mincut.num_edges(), (n - 1) as usize);
    assert_eq!(
        mincut.min_cut_value(),
        1.0,
        "path graph of length {n} should have min-cut = 1.0"
    );
}

// ===========================================================================
// 9. Error handling: duplicate edge insertion
// ===========================================================================
#[test]
fn duplicate_edge_insertion_fails() {
    let mut mincut = MinCutBuilder::new()
        .with_edges(vec![(1, 2, 1.0)])
        .build()
        .expect("build graph");

    let result = mincut.insert_edge(1, 2, 2.0);
    assert!(
        result.is_err(),
        "inserting a duplicate edge should return an error"
    );
}

// ===========================================================================
// 10. Error handling: deleting non-existent edge
// ===========================================================================
#[test]
fn delete_nonexistent_edge_fails() {
    let mut mincut = MinCutBuilder::new()
        .with_edges(vec![(1, 2, 1.0)])
        .build()
        .expect("build graph");

    let result = mincut.delete_edge(5, 6);
    assert!(
        result.is_err(),
        "deleting a non-existent edge should return an error"
    );
}

// ===========================================================================
// 11. DynamicGraph: direct API usage
// ===========================================================================
#[test]
fn dynamic_graph_direct_api() {
    let graph = DynamicGraph::new();

    graph.insert_edge(0, 1, 2.5).expect("insert 0-1");
    graph.insert_edge(1, 2, 3.0).expect("insert 1-2");
    graph.insert_edge(2, 0, 1.0).expect("insert 2-0");

    let stats = graph.stats();
    assert_eq!(stats.num_vertices, 3, "should have 3 vertices");
    assert_eq!(stats.num_edges, 3, "should have 3 edges");
    assert!(
        (stats.total_weight - 6.5).abs() < 1e-6,
        "total weight should be 6.5, got {}",
        stats.total_weight
    );
}

// ===========================================================================
// 12. Builder pattern: configuration options
// ===========================================================================
#[test]
fn builder_pattern_configuration() {
    let mincut = MinCutBuilder::new()
        .exact()
        .max_cut_size(1000)
        .parallel(true)
        .with_edges(vec![(1, 2, 1.0)])
        .build()
        .expect("build with custom config");

    let config = mincut.config();
    assert!(!config.approximate, "should be in exact mode");
    assert_eq!(config.max_exact_cut_size, 1000);
    assert!(config.parallel, "parallel should be enabled");
}

// ===========================================================================
// 13. Algorithm stats tracking
// ===========================================================================
#[test]
fn algorithm_stats_are_tracked() {
    let mut mincut = MinCutBuilder::new()
        .with_edges(vec![(1, 2, 1.0)])
        .build()
        .expect("build graph");

    mincut.insert_edge(2, 3, 1.0).expect("insert");
    mincut.delete_edge(1, 2).expect("delete");
    let _ = mincut.min_cut_value();

    let stats = mincut.stats();
    assert_eq!(stats.insertions, 1, "should track 1 insertion");
    assert_eq!(stats.deletions, 1, "should track 1 deletion");
    assert_eq!(stats.queries, 1, "should track 1 query");
}
