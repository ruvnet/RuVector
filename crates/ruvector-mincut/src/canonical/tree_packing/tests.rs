//! Tests for Gomory-Hu tree packing fast path (Tier 2).

use super::*;
use crate::canonical::source_anchored::{canonical_mincut, SourceAnchoredConfig};
use crate::canonical::FixedWeight;
use crate::graph::DynamicGraph;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_graph(edges: &[(u64, u64, f64)]) -> DynamicGraph {
    let g = DynamicGraph::new();
    for &(u, v, w) in edges {
        g.insert_edge(u, v, w).unwrap();
    }
    g
}

fn default_config() -> SourceAnchoredConfig {
    SourceAnchoredConfig::default()
}

// ---------------------------------------------------------------------------
// Gomory-Hu tree construction
// ---------------------------------------------------------------------------

#[test]
fn test_gomory_hu_single_edge() {
    let g = make_graph(&[(0, 1, 3.0)]);
    let tree = GomoryHuTree::build(&g).unwrap();

    assert_eq!(tree.n, 2);
    assert_eq!(tree.edges.len(), 1);
    assert_eq!(tree.edges[0].flow, FixedWeight::from_f64(3.0));
}

#[test]
fn test_gomory_hu_triangle() {
    let g = make_graph(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);
    let tree = GomoryHuTree::build(&g).unwrap();

    assert_eq!(tree.n, 3);
    assert_eq!(tree.edges.len(), 2);

    // In a unit-weight triangle, every vertex pair has max-flow 2.
    // The Gomory-Hu tree should have all edges with flow = 2.
    let min_flow = tree.global_mincut_value().unwrap();
    assert_eq!(min_flow, FixedWeight::from_f64(2.0));
}

#[test]
fn test_gomory_hu_path() {
    // Path: 0 -1- 1 -1- 2 -1- 3
    let g = make_graph(&[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
    let tree = GomoryHuTree::build(&g).unwrap();

    assert_eq!(tree.n, 4);
    assert_eq!(tree.edges.len(), 3);

    let min_flow = tree.global_mincut_value().unwrap();
    assert_eq!(min_flow, FixedWeight::from_f64(1.0));
}

#[test]
fn test_gomory_hu_two_clusters() {
    // Two triangles connected by a single edge:
    // {0,1,2} fully connected with weight 5
    // {3,4,5} fully connected with weight 5
    // Bridge: 2-3 with weight 1
    let g = make_graph(&[
        (0, 1, 5.0),
        (1, 2, 5.0),
        (2, 0, 5.0),
        (3, 4, 5.0),
        (4, 5, 5.0),
        (5, 3, 5.0),
        (2, 3, 1.0),
    ]);
    let tree = GomoryHuTree::build(&g).unwrap();

    let min_flow = tree.global_mincut_value().unwrap();
    assert_eq!(min_flow, FixedWeight::from_f64(1.0));
}

#[test]
fn test_gomory_hu_empty_graph() {
    let g = DynamicGraph::new();
    assert!(GomoryHuTree::build(&g).is_none());
}

#[test]
fn test_gomory_hu_single_vertex() {
    let g = DynamicGraph::new();
    g.add_vertex(0);
    assert!(GomoryHuTree::build(&g).is_none());
}

// ---------------------------------------------------------------------------
// Global min-cut from tree
// ---------------------------------------------------------------------------

#[test]
fn test_tree_global_mincut_matches_stoer_wagner() {
    // Build various graphs and verify the Gomory-Hu tree gives the same
    // min-cut value as the Tier 1 source-anchored algorithm.
    let test_cases: Vec<Vec<(u64, u64, f64)>> = vec![
        // Simple edge
        vec![(0, 1, 2.0)],
        // Triangle
        vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
        // Path of 4
        vec![(0, 1, 3.0), (1, 2, 1.0), (2, 3, 5.0)],
        // K4
        vec![
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
        ],
        // Weighted barbell
        vec![
            (0, 1, 10.0),
            (0, 2, 10.0),
            (1, 2, 10.0),
            (3, 4, 10.0),
            (3, 5, 10.0),
            (4, 5, 10.0),
            (2, 3, 2.0),
        ],
    ];

    for edges in &test_cases {
        let g = make_graph(edges);
        let tree = GomoryHuTree::build(&g).unwrap();
        let tree_lambda = tree.global_mincut_value().unwrap();

        let cut = canonical_mincut(&g, &default_config()).unwrap();
        assert_eq!(
            tree_lambda, cut.lambda,
            "Gomory-Hu tree min-cut should match Stoer-Wagner for edges: {:?}",
            edges
        );
    }
}

// ---------------------------------------------------------------------------
// Tree partition
// ---------------------------------------------------------------------------

#[test]
fn test_tree_partition_covers_all_vertices() {
    let g = make_graph(&[
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 0, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
    ]);
    let tree = GomoryHuTree::build(&g).unwrap();
    let result = tree.global_mincut_partition().unwrap();

    let mut all: Vec<VertexId> = result
        .side_a
        .iter()
        .chain(result.side_b.iter())
        .copied()
        .collect();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), 5);
}

#[test]
fn test_tree_partition_single_edge() {
    let g = make_graph(&[(0, 1, 1.0)]);
    let tree = GomoryHuTree::build(&g).unwrap();
    let result = tree.global_mincut_partition().unwrap();

    assert_eq!(result.lambda, FixedWeight::from_f64(1.0));
    assert_eq!(result.side_a.len() + result.side_b.len(), 2);
}

// ---------------------------------------------------------------------------
// Fast canonical cut
// ---------------------------------------------------------------------------

#[test]
fn test_fast_canonical_matches_tier1() {
    let edges = vec![
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 0, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
        (4, 2, 1.0),
    ];
    let g = make_graph(&edges);
    let cfg = default_config();

    let tier1 = canonical_mincut(&g, &cfg);
    let fast = canonical_mincut_fast(&g, &cfg);

    match (tier1, fast) {
        (Some(t1), Some(f)) => {
            assert_eq!(t1.lambda, f.lambda);
            assert_eq!(t1.first_separable_vertex, f.first_separable_vertex);
            assert_eq!(t1.side_vertices, f.side_vertices);
            assert_eq!(t1.cut_hash, f.cut_hash);
        }
        (None, None) => {} // both None is fine
        _ => panic!("Tier 1 and fast path disagree on Some/None"),
    }
}

#[test]
fn test_fast_canonical_empty_graph() {
    let g = DynamicGraph::new();
    assert!(canonical_mincut_fast(&g, &default_config()).is_none());
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_gomory_hu_deterministic_100_runs() {
    let g = make_graph(&[
        (0, 1, 3.0),
        (1, 2, 2.0),
        (2, 3, 4.0),
        (3, 0, 1.0),
        (0, 2, 5.0),
        (1, 3, 2.0),
    ]);

    let first_tree = GomoryHuTree::build(&g).unwrap();
    let first_lambda = first_tree.global_mincut_value().unwrap();

    for _ in 0..100 {
        let tree = GomoryHuTree::build(&g).unwrap();
        let lambda = tree.global_mincut_value().unwrap();
        assert_eq!(lambda, first_lambda, "Gomory-Hu tree must be deterministic");
    }
}

// ---------------------------------------------------------------------------
// Disconnected components
// ---------------------------------------------------------------------------

#[test]
fn test_gomory_hu_disconnected_returns_zero() {
    let g = make_graph(&[(0, 1, 1.0), (2, 3, 1.0)]);
    let tree = GomoryHuTree::build(&g).unwrap();
    let lambda = tree.global_mincut_value().unwrap();
    // Disconnected graphs have min-cut = 0
    assert_eq!(lambda, FixedWeight::from_f64(0.0));
}

// ---------------------------------------------------------------------------
// Weighted graphs
// ---------------------------------------------------------------------------

#[test]
fn test_gomory_hu_weighted_star() {
    // Star: center vertex 0 connected to 1,2,3,4 with various weights
    let g = make_graph(&[(0, 1, 2.0), (0, 2, 3.0), (0, 3, 1.0), (0, 4, 5.0)]);
    let tree = GomoryHuTree::build(&g).unwrap();
    let lambda = tree.global_mincut_value().unwrap();
    // Min-cut of a star is the minimum edge weight
    assert_eq!(lambda, FixedWeight::from_f64(1.0));
}
