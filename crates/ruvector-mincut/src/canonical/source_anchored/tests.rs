//! Tests for source-anchored pseudo-deterministic canonical minimum cut.

use super::*;
use crate::graph::DynamicGraph;

// ---------------------------------------------------------------------------
// Helper: build a graph from edge list
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
// SHA-256 correctness
// ---------------------------------------------------------------------------

#[test]
fn test_sha256_empty() {
    let hash = sha256(b"");
    let expected: [u8; 32] = [
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9,
        0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52,
        0xb8, 0x55,
    ];
    assert_eq!(
        hash, expected,
        "SHA-256 of empty string must match NIST vector"
    );
}

#[test]
fn test_sha256_abc() {
    let hash = sha256(b"abc");
    let expected: [u8; 32] = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22,
        0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00,
        0x15, 0xad,
    ];
    assert_eq!(hash, expected, "SHA-256 of 'abc' must match NIST vector");
}

#[test]
fn test_sha256_deterministic() {
    let data = b"canonical min-cut test";
    let h1 = sha256(data);
    let h2 = sha256(data);
    assert_eq!(h1, h2, "SHA-256 must be deterministic");
}

// ---------------------------------------------------------------------------
// Trivial / edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_empty_graph() {
    let g = DynamicGraph::new();
    assert!(canonical_mincut(&g, &default_config()).is_none());
}

#[test]
fn test_single_vertex() {
    let g = DynamicGraph::new();
    g.add_vertex(0);
    assert!(canonical_mincut(&g, &default_config()).is_none());
}

#[test]
fn test_single_edge() {
    let g = make_graph(&[(0, 1, 1.0)]);
    let cut = canonical_mincut(&g, &default_config()).unwrap();

    assert_eq!(cut.lambda, FixedWeight::from_f64(1.0));
    assert_eq!(cut.source_vertex, 0);
    assert_eq!(cut.first_separable_vertex, 1);
    assert!(cut.side_vertices.contains(&0));
    assert!(!cut.side_vertices.contains(&1));
    assert_eq!(cut.cut_edges.len(), 1);
}

#[test]
fn test_two_edges_path() {
    let g = make_graph(&[(0, 1, 3.0), (1, 2, 1.0)]);
    let cut = canonical_mincut(&g, &default_config()).unwrap();
    assert_eq!(cut.lambda, FixedWeight::from_f64(1.0));
}

// ---------------------------------------------------------------------------
// Triangle (cycle) — all cuts equal → canonical tie-breaking
// ---------------------------------------------------------------------------

#[test]
fn test_triangle_canonical() {
    let g = make_graph(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);
    let cut = canonical_mincut(&g, &default_config()).unwrap();

    assert_eq!(cut.lambda, FixedWeight::from_f64(2.0));
    assert_eq!(cut.first_separable_vertex, 1);
}

#[test]
fn test_triangle_invariance() {
    let g = make_graph(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);
    let cut1 = canonical_mincut(&g, &default_config()).unwrap();

    let g2 = make_graph(&[(2, 0, 1.0), (0, 1, 1.0), (1, 2, 1.0)]);
    let cut2 = canonical_mincut(&g2, &default_config()).unwrap();

    assert_eq!(cut1.cut_hash, cut2.cut_hash);
    assert_eq!(cut1.lambda, cut2.lambda);
    assert_eq!(cut1.first_separable_vertex, cut2.first_separable_vertex);
    assert_eq!(cut1.side_vertices, cut2.side_vertices);
}

// ---------------------------------------------------------------------------
// Complete graph K4 with uniform weights
// ---------------------------------------------------------------------------

#[test]
fn test_complete_k4_uniform() {
    let g = make_graph(&[
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (1, 2, 1.0),
        (1, 3, 1.0),
        (2, 3, 1.0),
    ]);
    let cut = canonical_mincut(&g, &default_config()).unwrap();

    assert_eq!(cut.lambda, FixedWeight::from_f64(3.0));
    assert_eq!(cut.first_separable_vertex, 1);
}

// ---------------------------------------------------------------------------
// Weighted graph with clear unique min-cut
// ---------------------------------------------------------------------------

#[test]
fn test_weighted_barbell() {
    let g = make_graph(&[
        (0, 1, 10.0),
        (1, 2, 10.0),
        (0, 2, 10.0),
        (2, 3, 1.0),
        (3, 4, 10.0),
        (4, 5, 10.0),
        (3, 5, 10.0),
    ]);
    let cut = canonical_mincut(&g, &default_config()).unwrap();

    assert_eq!(cut.lambda, FixedWeight::from_f64(1.0));
    assert_eq!(cut.cut_edges.len(), 1);
    assert!(cut.cut_edges.contains(&(2, 3)));
}

// ---------------------------------------------------------------------------
// Ladder graph: many equal minimum cuts
// ---------------------------------------------------------------------------

#[test]
fn test_ladder_graph() {
    let g = make_graph(&[
        (0, 1, 1.0),
        (2, 3, 1.0),
        (4, 5, 1.0),
        (6, 7, 1.0),
        (0, 2, 1.0),
        (2, 4, 1.0),
        (4, 6, 1.0),
        (1, 3, 1.0),
        (3, 5, 1.0),
        (5, 7, 1.0),
    ]);
    let cut = canonical_mincut(&g, &default_config()).unwrap();

    assert!(cut.lambda.raw() > 0);
    let cut2 = canonical_mincut(&g, &default_config()).unwrap();
    assert_eq!(cut.cut_hash, cut2.cut_hash);
}

// ---------------------------------------------------------------------------
// Determinism: 100 iterations
// ---------------------------------------------------------------------------

#[test]
fn test_determinism_100_runs() {
    let g = make_graph(&[
        (0, 1, 2.0),
        (1, 2, 3.0),
        (2, 3, 1.0),
        (3, 0, 4.0),
        (0, 2, 1.0),
        (1, 3, 2.0),
    ]);

    let reference = canonical_mincut(&g, &default_config()).unwrap();

    for _ in 0..100 {
        let cut = canonical_mincut(&g, &default_config()).unwrap();
        assert_eq!(cut.cut_hash, reference.cut_hash);
        assert_eq!(cut.lambda, reference.lambda);
        assert_eq!(cut.first_separable_vertex, reference.first_separable_vertex);
        assert_eq!(cut.side_vertices, reference.side_vertices);
        assert_eq!(cut.priority_sum, reference.priority_sum);
    }
}

// ---------------------------------------------------------------------------
// Custom source vertex
// ---------------------------------------------------------------------------

#[test]
fn test_custom_source() {
    let g = make_graph(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);

    let config = SourceAnchoredConfig {
        source: Some(2),
        ..Default::default()
    };
    let cut = canonical_mincut(&g, &config).unwrap();

    assert_eq!(cut.source_vertex, 2);
    assert!(cut.side_vertices.contains(&2));
}

// ---------------------------------------------------------------------------
// Custom vertex priorities
// ---------------------------------------------------------------------------

#[test]
fn test_vertex_priorities() {
    let g = make_graph(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);

    let config = SourceAnchoredConfig {
        vertex_priorities: Some(vec![(0, 10), (1, 1), (2, 5)]),
        ..Default::default()
    };
    let cut = canonical_mincut(&g, &config).unwrap();

    assert!(cut.priority_sum > 0);
}

// ---------------------------------------------------------------------------
// Disconnected graph
// ---------------------------------------------------------------------------

#[test]
fn test_disconnected_graph() {
    let g = DynamicGraph::new();
    g.insert_edge(0, 1, 1.0).unwrap();
    g.insert_edge(2, 3, 1.0).unwrap();

    let cut = canonical_mincut(&g, &default_config());
    assert!(cut.is_none());
}

// ---------------------------------------------------------------------------
// SourceAnchoredMinCut stateful wrapper
// ---------------------------------------------------------------------------

#[test]
fn test_stateful_wrapper_basic() {
    let mut mc = SourceAnchoredMinCut::with_edges(
        vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
        SourceAnchoredConfig::default(),
    )
    .unwrap();

    let cut = mc.canonical_cut().unwrap();
    assert_eq!(cut.lambda, FixedWeight::from_f64(2.0));
    assert_eq!(mc.epoch(), 0);
}

#[test]
fn test_stateful_wrapper_mutation() {
    let mut mc = SourceAnchoredMinCut::with_edges(
        vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
        SourceAnchoredConfig::default(),
    )
    .unwrap();

    let _cut_before = mc.canonical_cut().unwrap();

    mc.insert_edge(0, 3, 0.5).unwrap();
    assert_eq!(mc.epoch(), 1);

    let _cut_after = mc.canonical_cut();
    assert_eq!(mc.epoch(), 1);
}

#[test]
fn test_stateful_wrapper_receipt() {
    let mut mc = SourceAnchoredMinCut::with_edges(
        vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
        SourceAnchoredConfig::default(),
    )
    .unwrap();

    let receipt = mc.receipt().unwrap();
    assert_eq!(receipt.epoch, 0);

    let receipt2 = mc.receipt().unwrap();
    assert!(receipts_agree(&receipt, &receipt2));
}

// ---------------------------------------------------------------------------
// Receipt agreement
// ---------------------------------------------------------------------------

#[test]
fn test_receipt_agreement() {
    let g = make_graph(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);
    let cut = canonical_mincut(&g, &default_config()).unwrap();

    let r1 = make_receipt(&cut, 0);
    let r2 = make_receipt(&cut, 0);
    assert!(receipts_agree(&r1, &r2));

    let r3 = make_receipt(&cut, 42);
    assert_eq!(r1.cut_hash, r3.cut_hash);
}

// ---------------------------------------------------------------------------
// FFI result conversion
// ---------------------------------------------------------------------------

#[test]
fn test_ffi_result_conversion() {
    let g = make_graph(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);
    let cut = canonical_mincut(&g, &default_config()).unwrap();

    let ffi: CanonicalMinCutResult = (&cut).into();
    assert_eq!(ffi.lambda_raw, cut.lambda.raw());
    assert_eq!(ffi.source_vertex, cut.source_vertex);
    assert_eq!(ffi.first_separable_vertex, cut.first_separable_vertex);
    assert_eq!(ffi.side_size, cut.side_size as u32);
    assert_eq!(ffi.priority_sum, cut.priority_sum);
    assert_eq!(ffi.cut_hash, cut.cut_hash);
}

// ---------------------------------------------------------------------------
// Dinic's max-flow correctness
// ---------------------------------------------------------------------------

#[test]
fn test_dinic_simple_flow() {
    let n = 4;
    let mut cap = vec![FixedWeight::zero(); n * n];
    cap[0 * n + 1] = FixedWeight::from_f64(3.0);
    cap[0 * n + 2] = FixedWeight::from_f64(2.0);
    cap[1 * n + 3] = FixedWeight::from_f64(2.0);
    cap[2 * n + 3] = FixedWeight::from_f64(3.0);

    let flow = dinic_maxflow(&mut cap, 0, 3, n);
    assert_eq!(flow, FixedWeight::from_f64(4.0));
}

// ---------------------------------------------------------------------------
// Weight edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_large_weight_flow() {
    let g = make_graph(&[(0, 1, 1000.0), (1, 2, 1000.0), (2, 0, 1000.0)]);
    let cut = canonical_mincut(&g, &default_config()).unwrap();
    assert_eq!(cut.lambda, FixedWeight::from_f64(2000.0));
}

#[test]
fn test_fractional_weights() {
    let g = make_graph(&[(0, 1, 0.5), (1, 2, 0.5), (2, 0, 0.5)]);
    let cut = canonical_mincut(&g, &default_config()).unwrap();
    let lambda_f64 = cut.lambda.to_f64();
    assert!((lambda_f64 - 1.0).abs() < 1e-4);
}

// ---------------------------------------------------------------------------
// AdjSnapshot internal tests
// ---------------------------------------------------------------------------

#[test]
fn test_adj_snapshot_construction() {
    let g = make_graph(&[(10, 20, 3.0), (20, 30, 4.0)]);
    let snap = AdjSnapshot::from_graph(&g);

    assert_eq!(snap.n, 3);
    assert_eq!(snap.vertices, vec![10, 20, 30]);
    assert!(snap.id_to_idx.contains_key(&10));
    assert!(snap.id_to_idx.contains_key(&20));
    assert!(snap.id_to_idx.contains_key(&30));
}

#[test]
fn test_global_mincut_value_single_edge() {
    let g = make_graph(&[(0, 1, 5.0)]);
    let snap = AdjSnapshot::from_graph(&g);
    let lambda = snap.global_mincut_value().unwrap();
    assert_eq!(lambda, FixedWeight::from_f64(5.0));
}

#[test]
fn test_global_mincut_value_triangle() {
    let g = make_graph(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);
    let snap = AdjSnapshot::from_graph(&g);
    let lambda = snap.global_mincut_value().unwrap();
    assert_eq!(lambda, FixedWeight::from_f64(2.0));
}

// ---------------------------------------------------------------------------
// Stress: structured graphs
// ---------------------------------------------------------------------------

#[test]
fn test_star_graph_n10() {
    let edges: Vec<(u64, u64, f64)> = (1..10).map(|i| (0, i, 1.0)).collect();
    let g = make_graph(&edges);
    let cut = canonical_mincut(&g, &default_config()).unwrap();

    assert_eq!(cut.lambda, FixedWeight::from_f64(1.0));
    assert_eq!(cut.first_separable_vertex, 1);
}

#[test]
fn test_cycle_n6() {
    let g = make_graph(&[
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
        (4, 5, 1.0),
        (5, 0, 1.0),
    ]);
    let cut = canonical_mincut(&g, &default_config()).unwrap();

    assert_eq!(cut.lambda, FixedWeight::from_f64(2.0));
    let cut2 = canonical_mincut(&g, &default_config()).unwrap();
    assert_eq!(cut.cut_hash, cut2.cut_hash);
}

// ---------------------------------------------------------------------------
// Security: hash stability across 1000 recomputations
// ---------------------------------------------------------------------------

#[test]
fn test_hash_stability_1000_iterations() {
    let g = make_graph(&[
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 0, 1.0),
        (0, 2, 1.0),
        (1, 3, 1.0),
    ]);

    let reference = canonical_mincut(&g, &default_config()).unwrap();

    for i in 0..1000 {
        let cut = canonical_mincut(&g, &default_config()).unwrap();
        assert_eq!(
            cut.cut_hash, reference.cut_hash,
            "Hash diverged at iteration {}",
            i
        );
    }
}

// ---------------------------------------------------------------------------
// Security: source side always contains designated source
// ---------------------------------------------------------------------------

#[test]
fn test_source_always_on_source_side() {
    // Test with various sources
    for source in 0..4u64 {
        let g = make_graph(&[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]);
        let config = SourceAnchoredConfig {
            source: Some(source),
            ..Default::default()
        };
        if let Some(cut) = canonical_mincut(&g, &config) {
            assert!(
                cut.side_vertices.contains(&source),
                "Source {} not on source side",
                source
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Security: cut_hash uniquely identifies different cuts
// ---------------------------------------------------------------------------

#[test]
fn test_different_graphs_different_hashes() {
    let g1 = make_graph(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);
    let g2 = make_graph(&[(0, 1, 2.0), (1, 2, 2.0), (2, 0, 2.0)]);

    let cut1 = canonical_mincut(&g1, &default_config()).unwrap();
    let cut2 = canonical_mincut(&g2, &default_config()).unwrap();

    // Different lambda → different hash
    assert_ne!(cut1.cut_hash, cut2.cut_hash);
}

// ---------------------------------------------------------------------------
// Edge case: all vertices have same degree (complete graph symmetry)
// ---------------------------------------------------------------------------

#[test]
fn test_k5_symmetry() {
    let g = make_graph(&[
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (0, 4, 1.0),
        (1, 2, 1.0),
        (1, 3, 1.0),
        (1, 4, 1.0),
        (2, 3, 1.0),
        (2, 4, 1.0),
        (3, 4, 1.0),
    ]);
    let cut = canonical_mincut(&g, &default_config()).unwrap();

    assert_eq!(cut.lambda, FixedWeight::from_f64(4.0));
    // Determinism in highly symmetric graph
    let cut2 = canonical_mincut(&g, &default_config()).unwrap();
    assert_eq!(cut.cut_hash, cut2.cut_hash);
    assert_eq!(cut.first_separable_vertex, cut2.first_separable_vertex);
}
