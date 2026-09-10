//! Regression tests for ruvnet/RuVector#773.
//!
//! `VectorDB.search()` intermittently omitted a stored row while `get()` still
//! returned it and `len()` still counted it.  Root cause was in the vendored
//! `hnsw_rs` (`patches/hnsw_rs`): `reverse_update_neighborhood_simple` wrote
//! every symmetric edge into the neighbour's list at index
//! `new_point.p_id.0` — the new point's *top* level — instead of at `l`, the
//! layer the forward edge was actually built on.
//!
//! A point at level >= 1 therefore received no layer-0 in-edges at all, so a
//! layer-0 traversal could only reach it when it happened to be the entry
//! point.  Raising `ef_search` cannot recover such a point: it is not that the
//! search stopped early, it is that nothing in layer 0 points at it.
//!
//! Both tests below fail against the unpatched vendored crate.  Levels are
//! drawn from `StdRng::from_entropy()`, so a single index is not deterministic;
//! each test therefore aggregates enough independent indexes that the
//! probability of passing with the defect present is negligible (see the
//! per-test notes).

use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{DistanceMetric, HnswConfig};

/// The configuration the `ruvector` npm wrapper passes by default.
fn wrapper_default_config() -> HnswConfig {
    HnswConfig {
        m: 32,
        ef_construction: 200,
        ef_search: 100,
        max_elements: 10_000,
    }
}

/// Deterministic unit vector, so a failure can be replayed from its seed.
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

/// The exact shape reported in the issue: three rows, `k` far above the row
/// count, and a search that returns fewer than three.
///
/// Measured failure rate with the defect present is ~2.8% per index, so the
/// chance of 400 consecutive clean indexes is ~0.972^400 ≈ 1e-5.
#[test]
fn issue_773_three_row_index_returns_every_row() {
    const DIMS: usize = 384;
    const ROWS: usize = 3;
    const TRIALS: usize = 400;

    let mut short_trials = Vec::new();

    for trial in 0..TRIALS {
        let mut index = HnswIndex::new(DIMS, DistanceMetric::Cosine, wrapper_default_config())
            .expect("index construction");

        for row in 0..ROWS {
            index
                .add(
                    format!("m{row}"),
                    unit_vector((trial * 100 + row) as u64, DIMS),
                )
                .expect("insert");
        }

        assert_eq!(index.len(), ROWS, "trial {trial}: store lost a row");

        let query = unit_vector((trial * 100) as u64, DIMS);
        let hits = index
            .search_with_ef(&query, 64, 256)
            .expect("search must not error");

        if hits.len() != ROWS {
            let found: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
            short_trials.push(format!("trial {trial}: returned {found:?} of {ROWS}"));
        }
    }

    assert!(
        short_trials.is_empty(),
        "search() omitted stored rows in {} of {TRIALS} indexes (k=64, efSearch=256, only {ROWS} rows stored):\n  {}",
        short_trials.len(),
        short_trials.join("\n  ")
    );
}

/// Detection probe from the issue: search each stored vector with *itself*.
/// A point that its own vector cannot retrieve has lost its in-edges.
///
/// With the defect present this loses ~0.3% of a 2 000-point index (~6 points),
/// so a clean run is not realistically reachable by chance.
#[test]
fn issue_773_every_point_retrieves_itself() {
    const DIMS: usize = 128;
    const ROWS: usize = 2_000;

    let mut index = HnswIndex::new(DIMS, DistanceMetric::Cosine, wrapper_default_config())
        .expect("index construction");

    let vectors: Vec<Vec<f32>> = (0..ROWS).map(|i| unit_vector(i as u64, DIMS)).collect();
    for (i, vector) in vectors.iter().enumerate() {
        index.add(format!("v{i}"), vector.clone()).expect("insert");
    }
    assert_eq!(index.len(), ROWS);

    let unreachable: Vec<String> = vectors
        .iter()
        .enumerate()
        .filter(|(i, vector)| {
            let want = format!("v{i}");
            let hits = index.search_with_ef(vector, 10, 64).expect("search");
            !hits.iter().any(|h| h.id == want)
        })
        .map(|(i, _)| format!("v{i}"))
        .collect();

    assert!(
        unreachable.is_empty(),
        "{} of {ROWS} points could not be retrieved by their own vector \
         (index has lost their in-edges): {:?}",
        unreachable.len(),
        &unreachable[..unreachable.len().min(10)]
    );
}
