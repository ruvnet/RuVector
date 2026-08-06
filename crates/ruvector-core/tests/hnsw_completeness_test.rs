//! Regression tests for issue #773: `search()` silently omitted stored rows.
//!
//! A point inserted at HNSW level >= 1 never received reciprocal edges on the
//! lower layers, so once it stopped being the graph entry point nothing on
//! layer 0 pointed at it and no amount of `efSearch` could reach it again.
//! These tests pin the invariant that every stored id is retrievable.

#![cfg(feature = "hnsw")]

use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{DistanceMetric, HnswConfig};

/// Deterministic unit-vector generator (xorshift64*, seeded per vector).
fn unit_vector(seed: u64, dims: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut v = Vec::with_capacity(dims);
    for _ in 0..dims {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        let bits = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        v.push((bits >> 40) as f32 / 8_388_608.0 - 1.0);
    }
    let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
    if norm > 0.0 {
        for a in &mut v {
            *a /= norm;
        }
    }
    v
}

fn build_index(trial: u64, rows: usize, dims: usize) -> HnswIndex {
    let config = HnswConfig {
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        max_elements: 1_000,
    };
    let mut index = HnswIndex::new(dims, DistanceMetric::Cosine, config).expect("index");
    for i in 0..rows {
        index
            .add(format!("m{i}"), unit_vector(trial * 1_000 + i as u64, dims))
            .expect("add");
    }
    index
}

/// Every stored id must come back when `k` is at least the row count.
///
/// This is the exact shape of the reported reproduction: 3 rows of 384-d
/// cosine vectors, queried with `k` and `efSearch` far above the row count.
#[test]
fn search_returns_every_stored_row() {
    const DIMS: usize = 384;
    const ROWS: usize = 3;
    const TRIALS: u64 = 200;

    let mut short_trials = Vec::new();

    for trial in 0..TRIALS {
        let index = build_index(trial, ROWS, DIMS);
        // Query with the first row's own vector, as the reproduction does.
        let query = unit_vector(trial * 1_000, DIMS);
        let hits = index.search_with_ef(&query, 64, 256).expect("search");

        let mut seen: Vec<_> = hits.iter().map(|h| h.id.as_str()).collect();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != ROWS {
            short_trials.push((trial, seen.len()));
        }
    }

    assert!(
        short_trials.is_empty(),
        "search() omitted stored rows in {}/{} trials (trial, returned): {:?}",
        short_trials.len(),
        TRIALS,
        &short_trials[..short_trials.len().min(10)]
    );
}

/// Probe each row with its own vector — the detection recipe from issue #773.
/// A nearest-neighbour search for a stored vector must return the row holding
/// that exact vector.
#[test]
fn every_row_finds_itself() {
    const DIMS: usize = 128;
    const ROWS: usize = 24;

    for trial in 0..40u64 {
        let index = build_index(trial, ROWS, DIMS);
        for i in 0..ROWS {
            let id = format!("m{i}");
            let query = unit_vector(trial * 1_000 + i as u64, DIMS);
            let hits = index
                .search_with_ef(&query, ROWS, 4 * ROWS)
                .expect("search");
            assert!(
                hits.iter().any(|h| h.id == id),
                "trial {trial}: probing {id} with its own vector did not return it \
                 (returned {} of {ROWS} rows)",
                hits.len()
            );
        }
    }
}

/// End-to-end shape of the report in issue #773: a `VectorDB` holding three
/// 384-d cosine rows, queried with `k` and `efSearch` far above the row count,
/// must return every row.
#[test]
#[cfg(feature = "storage")]
fn vector_db_search_returns_every_stored_row() {
    use ruvector_core::types::{DbOptions, SearchQuery, VectorEntry};
    use ruvector_core::VectorDB;

    const DIMS: usize = 384;
    const ROWS: usize = 3;
    // Opening a store per trial is the expensive part, so this test only has
    // to prove the end-to-end path; the statistical net is the much cheaper
    // index-level sweep above. Seeds 24 and 25 both failed before the fix.
    const TRIALS: u64 = 30;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut short_trials = Vec::new();

    for trial in 0..TRIALS {
        let options = DbOptions {
            dimensions: DIMS,
            distance_metric: DistanceMetric::Cosine,
            storage_path: dir
                .path()
                .join(format!("t{trial}.db"))
                .to_string_lossy()
                .into_owned(),
            hnsw_config: Some(HnswConfig {
                m: 16,
                ef_construction: 100,
                ef_search: 100,
                max_elements: 1_000,
            }),
            quantization: None,
        };
        let db = VectorDB::new(options).expect("db");

        for i in 0..ROWS {
            db.insert(VectorEntry {
                id: Some(format!("m{i}")),
                vector: unit_vector(trial * 1_000 + i as u64, DIMS),
                metadata: None,
            })
            .expect("insert");
        }

        let hits = db
            .search(SearchQuery {
                vector: unit_vector(trial * 1_000, DIMS),
                k: 64,
                filter: None,
                ef_search: Some(256),
            })
            .expect("search");

        if hits.len() != db.len().expect("len") {
            short_trials.push((trial, hits.len()));
        }
    }

    assert!(
        short_trials.is_empty(),
        "VectorDB::search omitted stored rows in {}/{TRIALS} trials (trial, returned): {:?}",
        short_trials.len(),
        short_trials
    );
}

/// `add_batch` switches to `parallel_insert_slice` at 10 000 entries, a code
/// path with different interleaving of the reciprocal-edge update. Probe every
/// id with its own vector to confirm none was left orphaned. Ignored by default
/// because building a 12 000-point graph is slow in a debug build.
#[test]
#[ignore = "builds a 12k-point graph to exercise the parallel insert path"]
fn parallel_batch_insert_leaves_no_orphans() {
    const DIMS: usize = 64;
    const ROWS: usize = 12_000;

    let config = HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 200,
        max_elements: 20_000,
    };
    let mut index = HnswIndex::new(DIMS, DistanceMetric::Cosine, config).expect("index");
    let entries: Vec<_> = (0..ROWS)
        .map(|i| (format!("m{i}"), unit_vector(i as u64, DIMS)))
        .collect();
    index.add_batch(entries.clone()).expect("add_batch");

    let missing: Vec<_> = entries
        .iter()
        .filter(|(id, vector)| {
            let hits = index.search_with_ef(vector, 10, 200).expect("search");
            !hits.iter().any(|h| &h.id == id)
        })
        .map(|(id, _)| id.clone())
        .collect();

    assert!(
        missing.is_empty(),
        "{} of {ROWS} rows could not find themselves after a parallel batch insert: {:?}",
        missing.len(),
        &missing[..missing.len().min(10)]
    );
}

/// Wide scan used to quantify the omission rate. Ignored by default because it
/// builds tens of thousands of graphs; run with
/// `cargo test -p ruvector-core --test hnsw_completeness_test -- --ignored --nocapture`.
#[test]
#[ignore = "long-running rate measurement, not a pass/fail regression guard"]
fn omission_rate_scan() {
    const DIMS: usize = 384;
    const TRIALS: u64 = 5_000;

    let mut total = 0usize;
    let mut short = 0usize;
    for &rows in &[2usize, 3, 4, 5, 8] {
        let mut short_for_rows = 0usize;
        for trial in 0..TRIALS {
            let index = build_index(trial, rows, DIMS);
            let query = unit_vector(trial * 1_000, DIMS);
            let hits = index.search_with_ef(&query, 64, 256).expect("search");
            let mut ids: Vec<_> = hits.iter().map(|h| h.id.as_str()).collect();
            ids.sort_unstable();
            ids.dedup();
            if ids.len() != rows {
                short_for_rows += 1;
            }
            total += 1;
        }
        short += short_for_rows;
        println!(
            "rows={rows}: {short_for_rows}/{TRIALS} trials returned an incomplete result set \
             ({:.2}%)",
            short_for_rows as f64 * 100.0 / TRIALS as f64
        );
    }
    println!(
        "overall: {short}/{total} ({:.2}%)",
        short as f64 * 100.0 / total as f64
    );
}

/// Randomized sweep over sizes and seeds so a future regression surfaces even
/// if it only shows up at a different scale.
#[test]
fn search_is_complete_across_sizes_and_seeds() {
    for &dims in &[16usize, 64, 384] {
        for &rows in &[2usize, 3, 5, 17, 40] {
            for trial in 0..12u64 {
                let seed = trial * 7 + dims as u64 + rows as u64 * 31;
                let index = build_index(seed, rows, dims);

                for probe in 0..rows {
                    let query = unit_vector(seed * 1_000 + probe as u64, dims);
                    let hits = index
                        .search_with_ef(&query, rows, (4 * rows).max(64))
                        .expect("search");
                    let mut ids: Vec<_> = hits.iter().map(|h| h.id.as_str()).collect();
                    ids.sort_unstable();
                    ids.dedup();
                    assert_eq!(
                        ids.len(),
                        rows,
                        "dims={dims} rows={rows} seed={seed} probe={probe}: \
                         search returned {} distinct ids, expected {rows} ({ids:?})",
                        ids.len()
                    );
                }
            }
        }
    }
}
