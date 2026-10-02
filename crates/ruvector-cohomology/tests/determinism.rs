//! Determinism gate (issue #669): repeated runs produce identical witness
//! hashes; construction/edit order does not change canonical results.

mod common;

use common::{identity_system, ring_engine};
use ruvector_cohomology::dynamic::{CohomologyEdit, DynamicCohomology};
use ruvector_cohomology::operator::{LinearRestriction, RestrictionMap};
use ruvector_cohomology::syndrome::{compute_syndrome, SyndromeConfig};

#[test]
fn hundred_repeated_runs_same_witness_hash() {
    let build = || {
        identity_system(6, 2, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)], |i| {
            vec![i as f64 * 0.7 - 1.0, (i as f64).sin()]
        }, Some(vec![1.0, 0.5, 2.0, 1.5, 0.25, 3.0, 1.0]))
    };
    let reference = compute_syndrome(&build(), &SyndromeConfig::default())
        .unwrap()
        .witness
        .hash;
    for run in 0..100 {
        let hash = compute_syndrome(&build(), &SyndromeConfig::default())
            .unwrap()
            .witness
            .hash;
        assert_eq!(hash, reference, "run {run} diverged");
    }
}

#[test]
fn edge_insertion_order_does_not_change_witness() {
    // The canonical layout is sorted, so permuted edit order must produce
    // the same flushed witness hash (this is what makes parallel drivers
    // that merge by canonical key order safe).
    let dim = 2;
    let edges: Vec<(u64, u64)> = vec![(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)];
    let build = |order: &[usize]| {
        let mut engine = ring_engine(4, dim); // ring edges (0,1),(1,2),(2,3),(0,3)
        // Remove ring edges, reinsert everything in the permuted order.
        for &(a, b) in &edges[..4] {
            assert!(engine.apply_edit(CohomologyEdit::RemoveEdge { a, b }).accepted);
        }
        for &i in order {
            let (a, b) = edges[i];
            let receipt = engine.apply_edit(CohomologyEdit::InsertEdge {
                a,
                b,
                rho_a: RestrictionMap::Identity { dim },
                rho_b: RestrictionMap::Identity { dim },
                weight: 1.0 + i as f64,
                observation: vec![i as f64, -(i as f64)],
            });
            assert!(receipt.accepted, "{:?}", receipt.error);
        }
        engine.flush().unwrap().witness.hash
    };
    let forward = build(&[0, 1, 2, 3, 4]);
    let backward = build(&[4, 3, 2, 1, 0]);
    let shuffled = build(&[2, 0, 4, 1, 3]);
    assert_eq!(forward, backward);
    assert_eq!(forward, shuffled);
}

#[test]
fn restriction_hashes_are_content_addressed() {
    let a = RestrictionMap::Identity { dim: 4 };
    let b = RestrictionMap::Identity { dim: 4 };
    let c = RestrictionMap::Identity { dim: 5 };
    assert_eq!(a.canonical_hash(), b.canonical_hash());
    assert_ne!(a.canonical_hash(), c.canonical_hash());

    let s1 = RestrictionMap::Scale { dim: 4, factor: 1.0 };
    // Same action as identity but different canonical family: distinct hash.
    assert_ne!(a.canonical_hash(), s1.canonical_hash());
}

#[test]
fn edit_receipts_hash_edit_content() {
    let mut engine = ring_engine(4, 1);
    let r1 = engine.apply_edit(CohomologyEdit::UpdateObservation { a: 0, b: 1, value: vec![1.0] });
    let mut engine2 = ring_engine(4, 1);
    let r2 = engine2.apply_edit(CohomologyEdit::UpdateObservation { a: 0, b: 1, value: vec![1.0] });
    let r3 = engine2.apply_edit(CohomologyEdit::UpdateObservation { a: 0, b: 1, value: vec![2.0] });
    assert_eq!(r1.edit_hash, r2.edit_hash);
    assert_ne!(r1.edit_hash, r3.edit_hash);
}

#[test]
fn witness_support_ordering_is_canonical_under_ties() {
    // Symmetric contradiction: equal energies on all cycle edges — support
    // must be ordered by edge key on ties, deterministically.
    let system = identity_system(3, 1, &[(0, 1), (1, 2), (0, 2)], |i| match i {
        0 => vec![1.0],
        _ => vec![0.0],
    }, None);
    let result = compute_syndrome(&system, &SyndromeConfig::default()).unwrap();
    let keys: Vec<(u64, u64)> = result.witness.support.iter().map(|s| (s.edge.u, s.edge.v)).collect();
    assert_eq!(keys, vec![(0, 1), (0, 2), (1, 2)]);
}
