//! Exact small-fixture tests against the dense reference oracle and
//! closed-form values (issue #669, Phase 0 + correctness acceptance gate).

mod common;

use common::identity_system;
use ruvector_cohomology::block_sparse::CoboundaryBuilder;
use ruvector_cohomology::harmonic::spectral_summary;
use ruvector_cohomology::operator::RestrictionMap;
use ruvector_cohomology::solvers::SolverConfig;
use ruvector_cohomology::syndrome::{compute_syndrome, compute_syndrome_dense, SyndromeConfig};
use ruvector_cohomology::{AffineSheafSystem, EdgeField, EdgeWeights, GaugePolicy};

const ORACLE_TOL: f64 = 1e-10;

#[test]
fn coherent_triangle_has_zero_syndrome() {
    // b = δx for x = (0, 1, 3) on scalar stalks.
    let x = [0.0, 1.0, 3.0];
    let edges = [(0u64, 1u64), (1, 2), (0, 2)];
    let system = identity_system(
        3,
        1,
        &edges,
        |i| match i {
            0 => vec![x[1] - x[0]],
            1 => vec![x[2] - x[0]], // canonical order sorts (0,2) before (1,2)
            _ => vec![x[2] - x[1]],
        },
        None,
    );
    let cfg = SyndromeConfig::default();
    let result = compute_syndrome(&system, &cfg).unwrap();
    let oracle = compute_syndrome_dense(&system, &cfg).unwrap();
    assert!(result.energy < ORACLE_TOL, "energy = {}", result.energy);
    assert!(oracle.energy < ORACLE_TOL);
    assert!((result.energy - oracle.energy).abs() < ORACLE_TOL);
}

#[test]
fn inconsistent_triangle_matches_closed_form() {
    // Scalar triangle, unit weights, b = (1, 0, 0) on canonical edges
    // (0,1), (0,2), (1,2). The harmonic generator is (1, -1, 1)/√3
    // (b ∈ im δ iff b01 − b02 + b12 = 0), so energy = (b01 − b02 + b12)²/3.
    let system = identity_system(3, 1, &[(0, 1), (1, 2), (0, 2)], |i| match i {
        0 => vec![1.0],
        _ => vec![0.0],
    }, None);
    let cfg = SyndromeConfig::default();
    let result = compute_syndrome(&system, &cfg).unwrap();
    let oracle = compute_syndrome_dense(&system, &cfg).unwrap();
    let expected = 1.0 / 3.0;
    assert!((result.energy - expected).abs() < ORACLE_TOL, "energy = {}", result.energy);
    assert!((oracle.energy - expected).abs() < ORACLE_TOL);
    // Production path agrees with the oracle elementwise.
    for (a, b) in result.residual.data.iter().zip(oracle.residual.data.iter()) {
        assert!((a - b).abs() < ORACLE_TOL);
    }
    // All three cycle edges carry equal syndrome energy.
    for c in &result.ranked_support {
        assert!((c.energy - expected / 3.0).abs() < ORACLE_TOL);
    }
}

#[test]
fn tree_is_always_coherent() {
    // A path graph has no cycles: every observation is explainable.
    let system = identity_system(5, 3, &[(0, 1), (1, 2), (2, 3), (3, 4)], |i| {
        vec![i as f64 + 1.0, -(i as f64), 0.5]
    }, None);
    let cfg = SyndromeConfig::default();
    let result = compute_syndrome(&system, &cfg).unwrap();
    assert!(result.energy < ORACLE_TOL, "energy = {}", result.energy);
}

#[test]
fn weighted_syndrome_matches_oracle() {
    let weights = vec![0.5, 2.0, 3.5];
    let system = identity_system(3, 2, &[(0, 1), (1, 2), (0, 2)], |i| {
        vec![i as f64, 1.0 - i as f64]
    }, Some(weights));
    let cfg = SyndromeConfig::default();
    let result = compute_syndrome(&system, &cfg).unwrap();
    let oracle = compute_syndrome_dense(&system, &cfg).unwrap();
    assert!((result.energy - oracle.energy).abs() < ORACLE_TOL);
    for (a, b) in result.residual.data.iter().zip(oracle.residual.data.iter()) {
        assert!((a - b).abs() < ORACLE_TOL);
    }
    // The syndrome is W-orthogonal to im δ: δᵀ W s = 0.
    let op = &system.operator;
    let mut weighted = result.residual.data.clone();
    for (i, e) in op.edge_blocks().iter().enumerate() {
        for v in &mut weighted[e.offset..e.offset + e.dim] {
            *v *= system.weights.values[i];
        }
    }
    let mut back = vec![0.0; op.dim_c0()];
    op.apply_transpose(&weighted, &mut back);
    for v in back {
        assert!(v.abs() < ORACLE_TOL, "δᵀWs component = {v}");
    }
}

#[test]
fn scale_maps_match_oracle() {
    // Non-identity restrictions: x-values must be scaled before comparison.
    let mut builder = CoboundaryBuilder::new();
    for v in 0..3u64 {
        builder.add_vertex(v, 2).unwrap();
    }
    builder
        .add_edge(0, 1, RestrictionMap::Scale { dim: 2, factor: 2.0 },
                  RestrictionMap::Identity { dim: 2 })
        .unwrap();
    builder
        .add_edge(1, 2, RestrictionMap::Identity { dim: 2 },
                  RestrictionMap::Scale { dim: 2, factor: 0.5 })
        .unwrap();
    builder
        .add_edge(0, 2, RestrictionMap::Identity { dim: 2 },
                  RestrictionMap::Identity { dim: 2 })
        .unwrap();
    let op = builder.build();
    let n1 = op.dim_c1();
    let system = AffineSheafSystem::new(
        op,
        EdgeField { data: (0..n1).map(|i| (i as f64) * 0.3 - 0.7).collect() },
        EdgeWeights { values: vec![1.0, 1.3, 0.8] },
        GaugePolicy::MinimumNorm,
    )
    .unwrap();
    let cfg = SyndromeConfig::default();
    let result = compute_syndrome(&system, &cfg).unwrap();
    let oracle = compute_syndrome_dense(&system, &cfg).unwrap();
    assert!((result.energy - oracle.energy).abs() < ORACLE_TOL);
}

#[test]
fn h0_nullity_of_connected_identity_sheaf() {
    // For the identity sheaf on a connected graph, H⁰ = constants: the
    // Laplacian nullity equals the stalk dimension. Computed with LOBPCG,
    // never dominant power iteration.
    let dim = 3;
    let system = identity_system(4, dim, &[(0, 1), (1, 2), (2, 3), (0, 3)], |_| vec![0.0; dim], None);
    let summary = spectral_summary(
        &system.operator,
        &system.weights.values,
        dim + 2,
        0xC0FFEE,
        1e-8,
        &SolverConfig { tol: 1e-10, max_iter: 400 },
    );
    assert_eq!(summary.nullity_estimate, dim, "eigenvalues: {:?}", summary.smallest_eigenvalues);
    let gap = summary.spectral_gap.expect("nonzero eigenvalue expected");
    assert!(gap > 0.1, "spectral gap = {gap}");
}

#[test]
fn oracle_and_production_witness_hashes_agree_on_small_graphs() {
    // Below the dense-harmonic threshold both paths quantize to identical
    // witnesses except for the solver id — compare canonical content.
    let system = identity_system(3, 1, &[(0, 1), (1, 2), (0, 2)], |i| vec![i as f64], None);
    let cfg = SyndromeConfig::default();
    let a = compute_syndrome(&system, &cfg).unwrap().witness;
    let b = compute_syndrome_dense(&system, &cfg).unwrap().witness;
    assert_eq!(a.energy_q, b.energy_q);
    assert_eq!(a.support, b.support);
    assert_eq!(a.harmonic.coordinates_q, b.harmonic.coordinates_q);
    assert_eq!(a.operator_hash, b.operator_hash);
}
