//! Latent-drift controls (issue #669, §6): orthogonal frame changes are
//! coordinate drift, not semantic contradiction — transported systems must
//! not manufacture syndrome energy.

mod common;

use common::{random_orthogonal, ring_engine, Rng};
use ruvector_cohomology::dynamic::{CohomologyEdit, DynamicCohomology};
use ruvector_cohomology::operator::{DenseMatrix, RestrictionMap};
use ruvector_cohomology::syndrome::SyndromeConfig;
use ruvector_cohomology::transport::{cycle_consistency_defect, frame_change, guard_map_update, transport_state};
use ruvector_cohomology::ValidationLimits;

#[test]
fn frame_updates_leave_syndrome_invariant() {
    // A coherent-but-nonzero system stays coherent under arbitrary
    // orthogonal frame drift on any vertex (false contradiction rate: zero).
    let dim = 3;
    let mut engine = ring_engine(8, dim);
    let mut rng = Rng(2026);
    // Nonzero coherent observations: b = δx for a random x.
    let x: Vec<Vec<f64>> = (0..8).map(|_| (0..dim).map(|_| rng.signed_unit()).collect()).collect();
    for i in 0..8u64 {
        let j = (i + 1) % 8;
        let (u, v) = if i < j { (i, j) } else { (j, i) };
        let value: Vec<f64> = (0..dim).map(|d| x[v as usize][d] - x[u as usize][d]).collect();
        assert!(engine.apply_edit(CohomologyEdit::UpdateObservation { a: i, b: j, value }).accepted);
    }
    let before = engine.flush().unwrap();
    assert!(before.energy < 1e-16);

    // Drift several vertices through random orthogonal frames.
    for vertex in [1u64, 4, 6] {
        let q_new = random_orthogonal(dim, &mut rng);
        let receipt = engine.apply_edit(CohomologyEdit::UpdateFrame {
            vertex,
            q_old: DenseMatrix::identity(dim),
            q_new,
        });
        assert!(receipt.accepted, "{:?}", receipt.error);
    }
    let after = engine.flush().unwrap();
    assert!(
        after.energy < 1e-16,
        "coordinate drift manufactured contradiction: {}",
        after.energy
    );
}

#[test]
fn contradiction_energy_is_invariant_under_drift() {
    // A genuine contradiction keeps exactly the same energy after frame
    // transport: drift neither hides nor inflates it.
    let dim = 2;
    let mut engine = ring_engine(6, dim);
    assert!(engine
        .apply_edit(CohomologyEdit::UpdateObservation { a: 2, b: 3, value: vec![4.0, -1.0] })
        .accepted);
    let before = engine.flush().unwrap();
    assert!(before.energy > 1.0);

    let mut rng = Rng(555);
    for vertex in 0..6u64 {
        let q_new = random_orthogonal(dim, &mut rng);
        assert!(engine
            .apply_edit(CohomologyEdit::UpdateFrame {
                vertex,
                q_old: DenseMatrix::identity(dim),
                q_new,
            })
            .accepted);
    }
    let after = engine.flush().unwrap();
    assert!(
        (after.energy - before.energy).abs() < 1e-9,
        "energy drifted: {} -> {}",
        before.energy,
        after.energy
    );
}

#[test]
fn transport_state_round_trips() {
    let mut rng = Rng(31);
    let dim = 4;
    let q_old = random_orthogonal(dim, &mut rng);
    let q_new = random_orthogonal(dim, &mut rng);
    let limits = ValidationLimits::default();
    let r = frame_change(&q_old, &q_new, &limits).unwrap();
    let r_back = frame_change(&q_new, &q_old, &limits).unwrap();
    let x: Vec<f64> = (0..dim).map(|_| rng.signed_unit()).collect();
    let forward = transport_state(&r, &x).unwrap();
    let back = transport_state(&r_back, &forward).unwrap();
    for (a, b) in x.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn cycle_consistency_flags_bad_transport() {
    let dim = 3;
    let mut rng = Rng(8);
    let q = random_orthogonal(dim, &mut rng);
    let good_cycle = [
        RestrictionMap::Orthogonal { matrix: q.clone() },
        RestrictionMap::Orthogonal { matrix: q.transpose() },
    ];
    let defect = cycle_consistency_defect(&good_cycle.iter().collect::<Vec<_>>()).unwrap();
    assert!(defect < 1e-12, "consistent cycle flagged: {defect}");

    let bad_cycle = [
        RestrictionMap::Orthogonal { matrix: q.clone() },
        RestrictionMap::Scale { dim, factor: 0.5 },
    ];
    let defect = cycle_consistency_defect(&bad_cycle.iter().collect::<Vec<_>>()).unwrap();
    assert!(defect > 0.5, "inconsistent cycle passed: {defect}");
}

#[test]
fn map_update_guard_rejects_contradiction_manufacturing_maps() {
    // Negative control on a coherent system with two independent cycles
    // (ring + chord — on a single cycle *any* modified holonomy keeps the
    // coboundary full-rank, so nothing can look contradictory there).
    // A non-transport scaling map on a shared edge manufactures syndrome
    // energy and must be rejected; a legitimate orthogonal transport passes.
    let dim = 2;
    let mut rng = Rng(99);
    let x: Vec<Vec<f64>> = (0..5).map(|_| (0..dim).map(|_| rng.signed_unit() * 3.0).collect()).collect();
    let build_control = |x: &[Vec<f64>]| {
        let mut engine = ring_engine(5, dim);
        // Chord (0, 2) creates the second cycle; coherent observation.
        let receipt = engine.apply_edit(CohomologyEdit::InsertEdge {
            a: 0,
            b: 2,
            rho_a: RestrictionMap::Identity { dim },
            rho_b: RestrictionMap::Identity { dim },
            weight: 1.0,
            observation: (0..dim).map(|d| x[2][d] - x[0][d]).collect(),
        });
        assert!(receipt.accepted, "{:?}", receipt.error);
        for i in 0..5u64 {
            let j = (i + 1) % 5;
            let (u, v) = if i < j { (i, j) } else { (j, i) };
            let value: Vec<f64> = (0..dim).map(|d| x[v as usize][d] - x[u as usize][d]).collect();
            assert!(engine.apply_edit(CohomologyEdit::UpdateObservation { a: i, b: j, value }).accepted);
        }
        engine
    };
    let control_before = build_control(&x).build_system().unwrap();

    // Bad update: two *inconsistently* re-scaled restriction maps (a single
    // scaled edge is always absorbed by re-gauging its vertex to zero, so it
    // cannot manufacture contradiction — two incompatible ones can).
    let mut bad_engine = build_control(&x);
    for (a, b, factor) in [(0u64, 1u64, 0.3), (2, 3, 0.7)] {
        assert!(bad_engine
            .apply_edit(CohomologyEdit::UpdateRestriction {
                a,
                b,
                endpoint: ruvector_cohomology::Endpoint::Tail,
                map: RestrictionMap::Scale { dim, factor },
            })
            .accepted);
    }
    let control_after = bad_engine.build_system().unwrap();
    let report = guard_map_update(&control_before, &control_after, 1e-9, &SyndromeConfig::default()).unwrap();
    assert!(!report.accepted, "bad map passed the guard: {report:?}");

    // Good update: orthogonal frame transport.
    let mut good_engine = build_control(&x);
    assert!(good_engine
        .apply_edit(CohomologyEdit::UpdateFrame {
            vertex: 0,
            q_old: DenseMatrix::identity(dim),
            q_new: random_orthogonal(dim, &mut rng),
        })
        .accepted);
    let control_after = good_engine.build_system().unwrap();
    let report = guard_map_update(&control_before, &control_after, 1e-9, &SyndromeConfig::default()).unwrap();
    assert!(report.accepted, "legitimate transport rejected: {report:?}");
}
