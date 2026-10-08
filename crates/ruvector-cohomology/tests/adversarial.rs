//! Adversarial and boundary-validation tests (issue #669 security
//! requirements), plus Byzantine containment through the mincut bridge.

mod common;

use common::ring_engine;
use ruvector_cohomology::block_sparse::CoboundaryBuilder;
use ruvector_cohomology::dynamic::{CohomologyEdit, DynamicCohomology, DynamicSheafEngine, EngineConfig};
use ruvector_cohomology::mincut_bridge::{
    quarantine_boundary, run_containment, ContainmentPolicy, TensionGraph, TensionWeights,
};
use ruvector_cohomology::operator::{DenseMatrix, RestrictionMap};
use ruvector_cohomology::repair::{apply_repair, propose_repair, RepairPolicy};
use ruvector_cohomology::syndrome::{compute_syndrome, SyndromeConfig};
use ruvector_cohomology::{CohomologyError, EdgeKey, ValidationLimits};

#[test]
fn non_finite_and_invalid_inputs_are_rejected() {
    let mut engine = ring_engine(4, 2);
    // Non-finite observation.
    let r = engine.apply_edit(CohomologyEdit::UpdateObservation {
        a: 0,
        b: 1,
        value: vec![f64::NAN, 0.0],
    });
    assert!(!r.accepted);
    // Zero/negative/NaN weights.
    for w in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        let r = engine.apply_edit(CohomologyEdit::UpdateWeight { a: 0, b: 1, weight: w });
        assert!(!r.accepted, "weight {w} accepted");
    }
    // Dimension mismatch.
    let r = engine.apply_edit(CohomologyEdit::UpdateObservation { a: 0, b: 1, value: vec![1.0] });
    assert!(!r.accepted);
    // Self-loop.
    let r = engine.apply_edit(CohomologyEdit::InsertEdge {
        a: 2,
        b: 2,
        rho_a: RestrictionMap::Identity { dim: 2 },
        rho_b: RestrictionMap::Identity { dim: 2 },
        weight: 1.0,
        observation: vec![0.0, 0.0],
    });
    assert!(!r.accepted);
    // Duplicates.
    assert!(!engine.apply_edit(CohomologyEdit::InsertVertex { id: 0, dim: 2 }).accepted);
    let r = engine.apply_edit(CohomologyEdit::InsertEdge {
        a: 0,
        b: 1,
        rho_a: RestrictionMap::Identity { dim: 2 },
        rho_b: RestrictionMap::Identity { dim: 2 },
        weight: 1.0,
        observation: vec![0.0, 0.0],
    });
    assert!(!r.accepted);
    // The engine still flushes cleanly after every rejection.
    assert!(engine.flush().unwrap().energy < 1e-16);
}

#[test]
fn malicious_restriction_maps_fail_validation() {
    let limits = ValidationLimits::default();
    // Fake "orthogonal" map.
    let mut skewed = DenseMatrix::identity(3);
    skewed.set(0, 1, 0.5);
    assert!(RestrictionMap::Orthogonal { matrix: skewed }.validate(&limits).is_err());
    // Non-finite dense map.
    let mut inf = DenseMatrix::zeros(2, 2);
    inf.set(0, 0, f64::INFINITY);
    assert!(RestrictionMap::Dense { matrix: inf }.validate(&limits).is_err());
    // Norm bomb (numerical denial of service).
    let mut bomb = DenseMatrix::zeros(2, 2);
    bomb.set(0, 0, 1e12);
    assert!(RestrictionMap::Dense { matrix: bomb }.validate(&limits).is_err());
    // Dimension bomb (pre-allocation bound).
    assert!(RestrictionMap::Identity { dim: usize::MAX / 2 }.validate(&limits).is_err());
    // Out-of-order projection coords.
    assert!(RestrictionMap::Projection { input_dim: 4, coords: vec![2, 1] }
        .validate(&limits)
        .is_err());
    // A near-singular but bounded dense map is admitted (validation bounds
    // resources, not conditioning) and the solver still certifies.
    let mut near_singular = DenseMatrix::identity(2);
    near_singular.set(1, 1, 1e-12);
    assert!(RestrictionMap::Dense { matrix: near_singular.clone() }.validate(&limits).is_ok());
    let mut builder = CoboundaryBuilder::new();
    builder.add_vertex(0, 2).unwrap();
    builder.add_vertex(1, 2).unwrap();
    builder
        .add_edge(0, 1, RestrictionMap::Dense { matrix: near_singular },
                  RestrictionMap::Identity { dim: 2 })
        .unwrap();
    let op = builder.build();
    let system = ruvector_cohomology::AffineSheafSystem::new(
        op,
        ruvector_cohomology::EdgeField { data: vec![1.0, 1.0] },
        ruvector_cohomology::EdgeWeights { values: vec![1.0] },
        ruvector_cohomology::GaugePolicy::MinimumNorm,
    )
    .unwrap();
    let result = compute_syndrome(&system, &SyndromeConfig::default()).unwrap();
    assert!(result.energy.is_finite());
    assert!(result.certificate.residual_norm.is_finite());
}

#[test]
fn protected_edges_are_immutable_without_authorization() {
    let mut engine = ring_engine(6, 1);
    let planted = EdgeKey::new(2, 3).unwrap();
    assert!(engine
        .apply_edit(CohomologyEdit::UpdateObservation { a: 2, b: 3, value: vec![4.0] })
        .accepted);
    engine.set_protected(&planted, true).unwrap();

    // Without authorization the repair must leave the protected edge alone.
    let plan = engine
        .propose_repair(RepairPolicy { lambda: 0.05, ..RepairPolicy::default() })
        .unwrap();
    assert!(
        plan.corrections.iter().all(|(k, _)| *k != planted),
        "protected edge was repaired"
    );

    // With explicit authorization the same edge becomes repairable.
    let authorized = engine
        .propose_repair(RepairPolicy {
            lambda: 0.05,
            authorize_protected: true,
            ..RepairPolicy::default()
        })
        .unwrap();
    assert!(authorized.corrections.iter().any(|(k, _)| *k == planted));

    // Applying an authorized plan without authorization is refused.
    let mut system = engine.build_system().unwrap();
    let protected = engine.protected_mask();
    let err = apply_repair(
        &mut system,
        &authorized,
        Some(&protected),
        &RepairPolicy::default(),
    )
    .unwrap_err();
    assert!(matches!(err, CohomologyError::ProtectedEdge(2, 3)));
}

#[test]
fn byzantine_cluster_is_quarantined() {
    // Two communities joined by two bridge edges; the small community holds
    // mutually contradictory observations (a Byzantine cluster). The
    // tension mincut must cut the cheap bridge, not the healthy community.
    let dim = 1;
    let mut engine = DynamicSheafEngine::new(EngineConfig::default());
    for i in 0..10u64 {
        assert!(engine.apply_edit(CohomologyEdit::InsertVertex { id: i, dim }).accepted);
    }
    let add = |engine: &mut DynamicSheafEngine, a: u64, b: u64, obs: f64| {
        let r = engine.apply_edit(CohomologyEdit::InsertEdge {
            a,
            b,
            rho_a: RestrictionMap::Identity { dim },
            rho_b: RestrictionMap::Identity { dim },
            weight: 1.0,
            observation: vec![obs],
        });
        assert!(r.accepted, "{:?}", r.error);
    };
    // Healthy community: vertices 0..6, coherent ring (all zeros).
    for i in 0..6u64 {
        add(&mut engine, i, (i + 1) % 6, 0.0);
    }
    // Byzantine community: vertices 6..10, contradictory cycle.
    add(&mut engine, 6, 7, 5.0);
    add(&mut engine, 7, 8, 5.0);
    add(&mut engine, 8, 9, 5.0);
    add(&mut engine, 6, 9, 0.0); // closes an inconsistent cycle
    // Bridges.
    add(&mut engine, 0, 6, 0.0);
    add(&mut engine, 3, 8, 0.0);

    let syndrome = engine.flush().unwrap();
    assert!(syndrome.energy > 1.0);

    let graph = TensionGraph::from_syndrome(&syndrome, &TensionWeights::default(), None, None, None);
    // Byzantine vertices carry ~7–9 incident tension, healthy ones ≤ ~1.1
    // (a little syndrome legitimately spreads along the bridge cycle).
    let seeds = graph.contaminated_seeds(2.0);
    assert!(!seeds.is_empty());
    assert!(seeds.iter().all(|v| *v >= 6), "healthy vertex flagged: {seeds:?}");

    let healthy: Vec<u64> = (0..6).collect();
    let boundary = quarantine_boundary(&graph, &seeds, &healthy).unwrap();
    // The cut must separate along the two bridges.
    assert_eq!(
        boundary.cut_edges,
        vec![EdgeKey::new(0, 6).unwrap(), EdgeKey::new(3, 8).unwrap()],
        "unexpected boundary: {:?}",
        boundary.cut_edges
    );
    assert!(boundary.isolated_vertices.iter().all(|v| *v >= 6));
}

#[test]
fn containment_loop_verifies_post_repair_energy() {
    let mut engine = ring_engine(12, 2);
    assert!(engine
        .apply_edit(CohomologyEdit::UpdateObservation { a: 5, b: 6, value: vec![3.0, 3.0] })
        .accepted);
    for policy in [ContainmentPolicy::IsolateFirst, ContainmentPolicy::RepairFirst] {
        let mut e = ring_engine(12, 2);
        assert!(e
            .apply_edit(CohomologyEdit::UpdateObservation { a: 5, b: 6, value: vec![3.0, 3.0] })
            .accepted);
        let receipt = run_containment(
            &mut e,
            &TensionWeights::default(),
            &RepairPolicy { lambda: 0.01, ..RepairPolicy::default() },
            policy,
            0.5,
            1e-6,
        )
        .unwrap();
        assert!(receipt.pre_energy > 1.0);
        assert!(receipt.verified, "{policy:?}: post energy {}", receipt.post_energy);
        assert!(receipt.post_energy < 1e-6);
        assert_ne!(receipt.receipt_hash, [0u8; 32]);
    }
}

#[test]
fn repair_prefers_cheap_edges_under_cost_policy() {
    // Same contradiction, but the planted edge is expensive to modify: the
    // decoder should route the correction through cheaper cycle edges.
    let mut engine = ring_engine(4, 1);
    let expensive = EdgeKey::new(0, 1).unwrap();
    assert!(engine
        .apply_edit(CohomologyEdit::UpdateObservation { a: 0, b: 1, value: vec![2.0] })
        .accepted);
    engine
        .set_edge_cost(&expensive, ruvector_cohomology::repair::EdgeCost {
            business: 100.0,
            ..Default::default()
        })
        .unwrap();

    let syndrome = engine.flush().unwrap();
    let system = engine.build_system().unwrap();
    let plan = propose_repair(
        &system,
        &syndrome,
        Some(&engine.edge_costs()),
        Some(&engine.protected_mask()),
        &RepairPolicy { lambda: 0.05, ..RepairPolicy::default() },
        &SyndromeConfig::default(),
    )
    .unwrap();
    // Correction magnitude on the expensive edge should be (near) zero.
    let expensive_mag: f64 = plan
        .corrections
        .iter()
        .filter(|(k, _)| *k == expensive)
        .map(|(_, q)| q.iter().map(|v| v * v).sum::<f64>().sqrt())
        .sum();
    let cheap_mag: f64 = plan
        .corrections
        .iter()
        .filter(|(k, _)| *k != expensive)
        .map(|(_, q)| q.iter().map(|v| v * v).sum::<f64>().sqrt())
        .sum();
    assert!(
        cheap_mag > expensive_mag * 10.0,
        "cheap {cheap_mag} vs expensive {expensive_mag}"
    );
}
