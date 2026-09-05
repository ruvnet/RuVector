//! Property tests on deterministic pseudo-random systems: syndrome
//! invariants, zero false contradictions on coherent constructions, dynamic
//! flush vs. batch equivalence, and repair localization.

mod common;

use common::{identity_system, ring_engine, Rng};
use ruvector_cohomology::dynamic::{CohomologyEdit, DynamicCohomology};
use ruvector_cohomology::repair::RepairPolicy;
use ruvector_cohomology::syndrome::{compute_syndrome, SyndromeConfig};
use ruvector_cohomology::EdgeKey;

/// Deterministic random connected graph: a ring plus chords.
fn random_edges(n: u64, chords: u64, rng: &mut Rng) -> Vec<(u64, u64)> {
    let mut edges: Vec<(u64, u64)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    let mut added = 0;
    while added < chords {
        let a = rng.next_u64() % n;
        let b = rng.next_u64() % n;
        let key = if a < b { (a, b) } else { (b, a) };
        if a != b && !edges.contains(&key) && !edges.contains(&(key.1, key.0)) {
            edges.push(key);
            added += 1;
        }
    }
    edges
}

#[test]
fn coherent_systems_have_zero_certificates() {
    // Acceptance gate: zero false contradiction certificates on generated
    // coherent systems (b = δx by construction).
    for seed in 0..10u64 {
        let mut rng = Rng(seed.wrapping_mul(0x1234_5678).wrapping_add(1));
        let n = 8 + seed % 8;
        let dim = 1 + (seed % 3) as usize;
        let edges = random_edges(n, n / 2, &mut rng);
        let x: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.signed_unit() * 5.0).collect())
            .collect();
        // Compute canonical edge order first, then b_e = x_v − x_u.
        let mut sorted = edges.clone();
        sorted.sort_by_key(|&(a, b)| if a < b { (a, b) } else { (b, a) });
        let system = identity_system(n, dim, &edges, |i| {
            let (a, b) = sorted[i];
            let (u, v) = if a < b { (a, b) } else { (b, a) };
            (0..dim).map(|d| x[v as usize][d] - x[u as usize][d]).collect()
        }, None);
        let result = compute_syndrome(&system, &SyndromeConfig::default()).unwrap();
        assert!(
            result.energy < 1e-16,
            "false contradiction: seed {seed}, energy {}",
            result.energy
        );
    }
}

#[test]
fn syndrome_is_w_orthogonal_to_image() {
    for seed in 0..5u64 {
        let mut rng = Rng(seed + 77);
        let n = 10;
        let dim = 2;
        let edges = random_edges(n, 6, &mut rng);
        let weights: Vec<f64> = (0..edges.len()).map(|_| 0.1 + rng.signed_unit().abs() * 3.0).collect();
        let mut obs_rng = Rng(seed + 1000);
        let system = identity_system(n, dim, &edges, |_| {
            let mut r = Rng(obs_rng.next_u64());
            (0..dim).map(|_| r.signed_unit() * 2.0).collect()
        }, Some(weights));
        let result = compute_syndrome(&system, &SyndromeConfig::default()).unwrap();

        let op = &system.operator;
        let mut weighted = result.residual.data.clone();
        for (i, e) in op.edge_blocks().iter().enumerate() {
            for v in &mut weighted[e.offset..e.offset + e.dim] {
                *v *= system.weights.values[i];
            }
        }
        let mut back = vec![0.0; op.dim_c0()];
        op.apply_transpose(&weighted, &mut back);
        let defect = back.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(defect < 1e-8, "seed {seed}: ‖δᵀWs‖ = {defect}");
    }
}

#[test]
fn dynamic_flush_matches_batch() {
    // Acceptance gate: dynamic flush() energy agrees with fresh batch
    // assembly within 1e-8, and witnesses are identical.
    let mut engine = ring_engine(20, 3);
    let mut rng = Rng(4242);
    for _ in 0..30 {
        let a = rng.next_u64() % 20;
        let b = (a + 1) % 20;
        let value: Vec<f64> = (0..3).map(|_| rng.signed_unit()).collect();
        assert!(engine
            .apply_edit(CohomologyEdit::UpdateObservation { a, b, value })
            .accepted);
    }
    let flushed = engine.flush().unwrap();
    let batch_system = engine.build_system().unwrap();
    let batch = compute_syndrome(&batch_system, &SyndromeConfig::default()).unwrap();
    assert!(
        (flushed.energy - batch.energy).abs() < 1e-8,
        "flush {} vs batch {}",
        flushed.energy,
        batch.energy
    );
    assert_eq!(flushed.witness.hash, batch.witness.hash);
}

#[test]
fn estimate_is_exact_after_flush_and_marks_dirty_after_edit() {
    let mut engine = ring_engine(12, 2);
    let flushed = engine.flush().unwrap();
    let est = engine.estimate();
    assert!(est.is_exact);
    assert!((est.clean_energy - flushed.energy).abs() < 1e-12);

    assert!(engine
        .apply_edit(CohomologyEdit::UpdateObservation { a: 0, b: 1, value: vec![1.0, 0.0] })
        .accepted);
    let est = engine.estimate();
    assert!(!est.is_exact);
    assert!(est.dirty_cell_count >= 1);
}

#[test]
fn repair_localizes_planted_contradiction() {
    // Acceptance gate: planted contradiction localization + repair drives
    // syndrome energy down; the planted edge is in the repair support.
    //
    // On a bare ring the syndrome is a harmonic 1-form and distributes
    // uniformly over the cycle — no observation can localize which edge is
    // wrong. Localization requires trust asymmetry: the planted edge is the
    // least-trusted one (lowest weight), so the weighted projection
    // concentrates the residual there.
    let mut engine = ring_engine(16, 2);
    let planted = EdgeKey::new(3, 4).unwrap();
    assert!(engine
        .apply_edit(CohomologyEdit::UpdateObservation { a: 3, b: 4, value: vec![5.0, -5.0] })
        .accepted);
    assert!(engine
        .apply_edit(CohomologyEdit::UpdateWeight { a: 3, b: 4, weight: 0.1 })
        .accepted);

    let syndrome = engine.flush().unwrap();
    assert!(syndrome.energy > 1.0);
    // Localization: planted edge ranks first.
    assert_eq!(syndrome.ranked_support[0].edge, planted);

    let plan = engine
        .propose_repair(RepairPolicy { lambda: 0.05, ..RepairPolicy::default() })
        .unwrap();
    assert!(plan.corrections.iter().any(|(k, _)| *k == planted), "support: {:?}",
        plan.corrections.iter().map(|(k, _)| *k).collect::<Vec<_>>());
    assert!(plan.post_energy < syndrome.energy * 0.01,
        "post {} vs pre {}", plan.post_energy, syndrome.energy);
}

#[test]
fn repair_objective_near_exhaustive_optimum_on_triangle() {
    // Tractable fixture: scalar triangle with one inconsistent edge. The
    // group-lasso optimum corrects the planted edge; compare against a
    // brute-force scan over single-edge corrections.
    let mut engine = ring_engine(3, 1);
    assert!(engine
        .apply_edit(CohomologyEdit::UpdateObservation { a: 0, b: 1, value: vec![3.0] })
        .accepted);
    let lambda = 0.05;
    let plan = engine
        .propose_repair(RepairPolicy { lambda, ..RepairPolicy::default() })
        .unwrap();

    // Exhaustive: correcting edge e by magnitude m leaves defect (3 − m),
    // residual energy (3−m)²/3 · ... — instead of the closed form, verify
    // numerically that no single-edge correction of the planted edge with a
    // scanned magnitude beats ADMM by more than 10%.
    let mut best = f64::INFINITY;
    for step in 0..=300 {
        let m = step as f64 * 0.01;
        let defect: f64 = 3.0 - m;
        let residual_energy = defect * defect / 3.0;
        let obj = 0.5 * residual_energy + lambda * m;
        best = best.min(obj);
    }
    assert!(
        plan.objective <= best * 1.10 + 1e-9,
        "ADMM objective {} vs exhaustive {}",
        plan.objective,
        best
    );
}
