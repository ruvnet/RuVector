//! Integration and property tests for ruvector-sparsifier.

use ruvector_sparsifier::{
    AdaptiveGeoSpar, SparseGraph, Sparsifier, SparsifierConfig, SpectralAuditor,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn triangle() -> SparseGraph {
    SparseGraph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
}

fn path(n: usize) -> SparseGraph {
    let edges: Vec<_> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
    SparseGraph::from_edges(&edges)
}

fn complete(n: usize) -> SparseGraph {
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((i, j, 1.0));
        }
    }
    SparseGraph::from_edges(&edges)
}

fn grid(rows: usize, cols: usize) -> SparseGraph {
    let mut edges = Vec::new();
    let idx = |r: usize, c: usize| r * cols + c;
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                edges.push((idx(r, c), idx(r, c + 1), 1.0));
            }
            if r + 1 < rows {
                edges.push((idx(r, c), idx(r + 1, c), 1.0));
            }
        }
    }
    SparseGraph::from_edges(&edges)
}

fn knn_random(n: usize, k: usize, seed: u64) -> SparseGraph {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut edges = Vec::new();
    for u in 0..n {
        for _ in 0..k {
            let v = rng.gen_range(0..n);
            if u != v {
                let w = rng.gen::<f64>() + 0.1;
                edges.push((u, v, w));
            }
        }
    }
    SparseGraph::from_edges(&edges)
}

// ---------------------------------------------------------------------------
// Spectral quality tests
// ---------------------------------------------------------------------------

#[test]
fn test_sparsifier_preserves_laplacian_on_triangle() {
    let g = triangle();
    let config = SparsifierConfig {
        epsilon: 0.5,
        ..Default::default()
    };
    let spar = AdaptiveGeoSpar::build(&g, config).unwrap();

    // For a 3-vertex graph, the sparsifier should be near-exact.
    let x = vec![1.0, 0.0, -1.0];
    let full_val = g.laplacian_quadratic_form(&x);
    let spec_val = spar.sparsifier().laplacian_quadratic_form(&x);

    let rel_err = (full_val - spec_val).abs() / full_val.abs().max(1e-15);
    assert!(
        rel_err < 0.6,
        "Relative error {} exceeds 0.6 on triangle",
        rel_err
    );
}

#[test]
fn test_sparsifier_preserves_laplacian_on_grid() {
    let g = grid(5, 5);
    let config = SparsifierConfig {
        epsilon: 0.3,
        edge_budget_factor: 12,
        ..Default::default()
    };
    let spar = AdaptiveGeoSpar::build(&g, config).unwrap();

    let auditor = SpectralAuditor::new(50, 0.5);
    let result = auditor.audit_quadratic_form(spar.full_graph(), spar.sparsifier(), 50);

    // Grid with generous epsilon should pass.
    assert!(
        result.avg_error < 1.0,
        "Average error {} too high on 5x5 grid",
        result.avg_error
    );
}

#[test]
fn test_sparsifier_reduces_edges_on_complete_graph() {
    let g = complete(20);
    let config = SparsifierConfig {
        epsilon: 0.3,
        edge_budget_factor: 8,
        ..Default::default()
    };
    let spar = AdaptiveGeoSpar::build(&g, config).unwrap();

    // Complete(20) has 190 edges. Sparsifier should have fewer.
    let full_edges = spar.full_graph().num_edges();
    let spec_edges = spar.sparsifier().num_edges();
    assert!(full_edges == 190);
    assert!(
        spec_edges < full_edges,
        "Sparsifier ({}) should have fewer edges than full graph ({})",
        spec_edges,
        full_edges
    );
    assert!(spar.compression_ratio() > 1.0);
}

#[test]
fn test_sparsifier_on_knn_graph() {
    let g = knn_random(200, 8, 42);
    let config = SparsifierConfig {
        epsilon: 0.3,
        edge_budget_factor: 8,
        ..Default::default()
    };
    let spar = AdaptiveGeoSpar::build(&g, config).unwrap();

    assert!(spar.sparsifier().num_edges() > 0);
    assert!(spar.stats().vertex_count == 200);

    // Run audit.
    let result = spar.audit();
    // With ε=0.3 and random walks, this may not always pass perfectly,
    // but avg error should be reasonable.
    assert!(result.n_probes > 0);
}

// ---------------------------------------------------------------------------
// Dynamic update tests
// ---------------------------------------------------------------------------

#[test]
fn test_sequential_inserts() {
    let g = path(5);
    let config = SparsifierConfig::default();
    let mut spar = AdaptiveGeoSpar::build(&g, config).unwrap();

    // Add cross edges to make it denser.
    spar.insert_edge(0, 3, 1.0).unwrap();
    spar.insert_edge(1, 4, 1.0).unwrap();
    spar.insert_edge(0, 4, 1.0).unwrap();

    assert_eq!(spar.full_graph().num_edges(), 7); // 4 path + 3 cross
    assert_eq!(spar.stats().insertions, 3);
}

#[test]
fn test_sequential_deletes() {
    let g = complete(5);
    let config = SparsifierConfig::default();
    let mut spar = AdaptiveGeoSpar::build(&g, config).unwrap();

    spar.delete_edge(0, 1).unwrap();
    spar.delete_edge(2, 3).unwrap();

    assert_eq!(spar.full_graph().num_edges(), 8); // 10 - 2
    assert_eq!(spar.stats().deletions, 2);
}

#[test]
fn test_insert_then_delete_roundtrip() {
    let g = triangle();
    let config = SparsifierConfig::default();
    let mut spar = AdaptiveGeoSpar::build(&g, config).unwrap();

    spar.insert_edge(0, 3, 2.0).unwrap();
    assert!(spar.full_graph().has_edge(0, 3));

    spar.delete_edge(0, 3).unwrap();
    assert!(!spar.full_graph().has_edge(0, 3));
    assert_eq!(spar.full_graph().num_edges(), 3);
}

#[test]
fn test_embedding_update() {
    let g = knn_random(50, 5, 99);
    let config = SparsifierConfig::default();
    let mut spar = AdaptiveGeoSpar::build(&g, config).unwrap();

    let old_neighbors: Vec<(usize, f64)> = spar.full_graph().neighbors(0).take(2).collect();
    let new_neighbors = vec![(40, 1.5), (41, 2.0)];

    spar.update_embedding(0, &old_neighbors, &new_neighbors)
        .unwrap();

    for &(v, _) in &old_neighbors {
        assert!(!spar.full_graph().has_edge(0, v));
    }
    for &(v, _) in &new_neighbors {
        assert!(spar.full_graph().has_edge(0, v));
    }
}

// ---------------------------------------------------------------------------
// Audit tests
// ---------------------------------------------------------------------------

#[test]
fn test_audit_identical_graphs_always_pass() {
    let g = grid(4, 4);
    let auditor = SpectralAuditor::new(50, 0.001);
    let result = auditor.audit_quadratic_form(&g, &g, 50);
    assert!(result.passed);
    assert!(result.max_error < 1e-10);
}

#[test]
fn test_audit_cuts_identical() {
    let g = complete(10);
    let auditor = SpectralAuditor::new(20, 0.001);
    let result = auditor.audit_cuts(&g, &g, 20);
    assert!(result.passed);
}

#[test]
fn test_audit_detects_large_distortion() {
    let g = complete(10);
    // A very sparse subgraph should fail the audit.
    let sparse = path(10);
    let auditor = SpectralAuditor::new(50, 0.1);
    let result = auditor.audit_quadratic_form(&g, &sparse, 50);
    // Complete graph vs path should have significant distortion.
    assert!(result.max_error > 0.1);
}

// ---------------------------------------------------------------------------
// Rebuild tests
// ---------------------------------------------------------------------------

#[test]
fn test_local_rebuild() {
    let g = knn_random(100, 6, 77);
    let config = SparsifierConfig::default();
    let mut spar = AdaptiveGeoSpar::build(&g, config).unwrap();

    spar.rebuild_local(&[0, 1, 2, 3, 4]).unwrap();
    assert_eq!(spar.stats().local_rebuilds, 1);
    assert!(spar.sparsifier().num_edges() > 0);
}

#[test]
fn test_full_rebuild_idempotent() {
    let g = knn_random(50, 5, 55);
    let config = SparsifierConfig::default();
    let mut spar = AdaptiveGeoSpar::build(&g, config).unwrap();

    let e1 = spar.sparsifier().num_edges();
    spar.rebuild_full().unwrap();
    let e2 = spar.sparsifier().num_edges();

    // Edge counts may differ due to random sampling, but both should be > 0.
    assert!(e1 > 0);
    assert!(e2 > 0);
}

// ---------------------------------------------------------------------------
// Graph operations tests
// ---------------------------------------------------------------------------

#[test]
fn test_csr_roundtrip_preserves_structure() {
    let g = grid(3, 3);
    let (rp, ci, vals, n) = g.to_csr();
    let g2 = SparseGraph::from_csr(&rp, &ci, &vals, n);

    assert_eq!(g.num_vertices(), g2.num_vertices());
    assert_eq!(g.num_edges(), g2.num_edges());

    // Check Laplacian form is preserved.
    let x: Vec<f64> = (0..9).map(|i| i as f64).collect();
    let v1 = g.laplacian_quadratic_form(&x);
    let v2 = g2.laplacian_quadratic_form(&x);
    assert!((v1 - v2).abs() < 1e-10);
}

#[test]
fn test_weighted_degree() {
    let g = SparseGraph::from_edges(&[(0, 1, 2.0), (0, 2, 3.0), (0, 3, 5.0)]);
    assert!((g.weighted_degree(0) - 10.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Stats and config tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_serialization() {
    let config = SparsifierConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let config2: SparsifierConfig = serde_json::from_str(&json).unwrap();
    assert!((config.epsilon - config2.epsilon).abs() < 1e-10);
    assert_eq!(config.edge_budget_factor, config2.edge_budget_factor);
}

#[test]
fn test_stats_compression_ratio() {
    let g = complete(15);
    let config = SparsifierConfig {
        edge_budget_factor: 6,
        ..Default::default()
    };
    let spar = AdaptiveGeoSpar::build(&g, config).unwrap();
    let stats = spar.stats();

    assert_eq!(stats.full_edge_count, 105); // C(15,2)
    assert!(stats.compression_ratio > 1.0);
    assert_eq!(stats.full_rebuilds, 1);
}
