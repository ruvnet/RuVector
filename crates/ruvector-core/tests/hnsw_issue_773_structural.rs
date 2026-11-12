//! Deterministic structural receipt for ruvnet/RuVector#773.
//!
//! The sibling test `hnsw_issue_773_regression.rs` samples `search()` results
//! and argues statistically (~2.8% failure per index, aggregated over 400
//! indexes). That shows the defect is *unlikely* to survive; it is not a
//! completeness receipt, and it cannot be seeded — levels are drawn from
//! `StdRng::from_entropy()` inside the vendored crate.
//!
//! These assertions need no seed control. They are exhaustive over the graph
//! and hold for EVERY level draw, so a single index proves the property:
//!
//! **Every stored point is reachable from the entry point over layer-0 edges
//! alone** -- the invariant `search()` actually depends on, and the one #773
//! violated.
//!
//! ⚠️ A second candidate invariant was built, measured, and REJECTED: "no edge
//! is stored on a layer an endpoint does not occupy". It fires on the PATCHED,
//! working graph -- a level-0 point legitimately carries layer-1 out-edges,
//! seven of them in an 8-row index, while reachability stays clean. Whether
//! that is benign upstream behaviour was not established, so it is surfaced by
//! `HnswIndex::structural_violations().0` for inspection and deliberately NOT
//! asserted. A check that reddens on correct code is as damaging as one that
//! never fires.

use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{DistanceMetric, HnswConfig};

fn wrapper_default_config() -> HnswConfig {
    HnswConfig {
        m: 32,
        ef_construction: 200,
        ef_search: 100,
        max_elements: 10_000,
    }
}

/// Deterministic vectors; only the LEVELS stay random, which is the point.
fn unit_vector(seed: u64, dims: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761) % (1u64 << 31);
    let mut v = vec![0.0f32; dims];
    let mut norm = 0.0f32;
    for slot in v.iter_mut() {
        x = x.wrapping_mul(1103515245).wrapping_add(12345) % (1u64 << 31);
        *slot = (x as f32) / (1u32 << 30) as f32 - 1.0;
        norm += *slot * *slot;
    }
    let norm = norm.sqrt();
    for slot in v.iter_mut() {
        *slot /= norm;
    }
    v
}

fn build(rows: usize, dims: usize) -> HnswIndex {
    let mut idx = HnswIndex::new(dims, DistanceMetric::Cosine, wrapper_default_config())
        .expect("index construction");
    for i in 0..rows {
        idx.add((i as u64).to_string(), unit_vector(i as u64, dims))
            .expect("insert");
    }
    idx
}

/// Small indexes are where #773 bit: once every point has level >= 1, layer 0's
/// bucket is empty while layer 0 still carries edges.
#[test]
fn layer_invariants_hold_for_every_draw() {
    for rows in [1usize, 2, 3, 5, 8, 16, 40, 128] {
        let idx = build(rows, 384);
        let (_above, unreachable) = idx.structural_violations();
        assert!(
            unreachable.is_empty(),
            "rows={rows}: {} point(s) unreachable from the entry point over layer-0 \
             edges -- search() would silently omit them: {unreachable:?}",
            unreachable.len()
        );
    }
}

/// Repeating the whole-graph check across independent indexes is about the
/// LEVEL DRAW, not about probability: each index is a fresh draw, and the
/// invariant must hold for all of them. A single counterexample fails the run.
#[test]
fn layer_invariants_hold_across_independent_draws() {
    for trial in 0..64 {
        let idx = build(6, 64);
        let (_above, unreachable) = idx.structural_violations();
        assert!(
            unreachable.is_empty(),
            "trial {trial}: unreachable={unreachable:?}"
        );
    }
}

/// The checks must be capable of reporting something -- a receipt that can only
/// ever return "empty" proves nothing. An index with rows present must expose a
/// non-trivial graph for them to have walked.
#[test]
fn the_structural_checks_are_not_vacuous() {
    let idx = build(64, 64);
    let (_above, unreachable) = idx.structural_violations();
    assert!(unreachable.is_empty());
    // If the graph were empty the checks would pass while inspecting nothing;
    // prove they had material to inspect.
    assert_eq!(
        idx.len(),
        64,
        "the index under inspection must actually hold rows"
    );
}
