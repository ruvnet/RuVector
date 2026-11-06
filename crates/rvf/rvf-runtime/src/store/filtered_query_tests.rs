//! Store-level tests for the filtered HNSW path (`filtered_query.rs`).

use super::*;
use crate::filter::FilterValue;
use filtered_query::FILTERED_HNSW_MIN_GRAPH_RECALL;
use rvf_types::quality::QualityPreference;
use tempfile::TempDir;

const DIM: usize = 16;
const N: u64 = 6000;
const K: usize = 10;

fn lcg_vector(seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIM)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

/// Field 0 = `id % 100` (a "bucket"), field 1 = "rare" for three ids.
fn metadata_for(id: u64) -> Vec<MetadataEntry> {
    let mut m = vec![MetadataEntry {
        field_id: 0,
        value: MetadataValue::U64(id % 100),
    }];
    if [17, 1234, 5999].contains(&id) {
        m.push(MetadataEntry {
            field_id: 1,
            value: MetadataValue::String("rare".into()),
        });
    }
    m
}

fn build_store(path: &Path, ids: std::ops::Range<u64>) -> RvfStore {
    let mut store = RvfStore::create(
        path,
        RvfOptions {
            dimension: DIM as u16,
            metric: DistanceMetric::L2,
            ..Default::default()
        },
    )
    .unwrap();
    ingest(&mut store, ids);
    store
}

fn ingest(store: &mut RvfStore, ids: std::ops::Range<u64>) {
    let ids: Vec<u64> = ids.collect();
    let vecs: Vec<Vec<f32>> = ids.iter().map(|&i| lcg_vector(i)).collect();
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let records: Vec<VectorMetadata> = ids
        .iter()
        .map(|&id| VectorMetadata {
            vector_id: id,
            fields: metadata_for(id),
            delete_record: false,
        })
        .collect();
    store
        .ingest_batch_with_metadata(&refs, &ids, &records)
        .unwrap();
}

fn opts(filter: FilterExpr, force_exact: bool) -> QueryOptions {
    QueryOptions {
        filter: Some(filter),
        force_exact,
        ..Default::default()
    }
}

/// Returns (recall@K of the routed query vs exact, whether HNSW served).
fn compare(store: &RvfStore, filter: &FilterExpr, queries: u64) -> (f64, bool) {
    let mut recall = 0.0;
    let mut all_served = true;
    for q in 0..queries {
        let query = lcg_vector(1_000_000 + q);
        let exact = store.query(&query, K, &opts(filter.clone(), true)).unwrap();
        let (got, served) = store
            .query_routed(&query, K, &opts(filter.clone(), false))
            .unwrap();
        all_served &= served;
        assert_eq!(got.len(), exact.len());
        for r in &got {
            assert!(filter::evaluate(filter, r.id, &store.metadata));
            assert!(!store.deletion_bitmap.is_deleted(r.id));
        }
        let hits = got
            .iter()
            .filter(|r| exact.iter().any(|e| e.id == r.id))
            .count();
        recall += hits as f64 / exact.len().max(1) as f64;
    }
    (recall / queries as f64, all_served)
}

#[test]
fn filtered_queries_use_hnsw_when_selective_enough_and_match_exact() {
    let dir = TempDir::new().unwrap();
    let store = build_store(&dir.path().join("f.rvf"), 0..N);
    for (pct, expect_hnsw) in [(50u64, true), (25, true), (5, false), (1, false)] {
        let filter = FilterExpr::Lt(0, FilterValue::U64(pct));
        let (recall, served) = compare(&store, &filter, 20);
        assert_eq!(served, expect_hnsw, "routing at {pct}%");
        assert!(recall >= 0.95, "recall@10 {recall:.3} at {pct}%");
    }
}

#[test]
fn fewer_matches_than_k_falls_back_to_exact() {
    let dir = TempDir::new().unwrap();
    let store = build_store(&dir.path().join("f.rvf"), 0..N);
    let filter = FilterExpr::Eq(1, FilterValue::String("rare".into()));
    let got = store
        .query(&lcg_vector(5), K, &opts(filter.clone(), false))
        .unwrap();
    let mut ids: Vec<u64> = got.iter().map(|r| r.id).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![17, 1234, 5999]);
}

#[test]
fn deleted_ids_are_excluded_from_filtered_hnsw() {
    let dir = TempDir::new().unwrap();
    let mut store = build_store(&dir.path().join("f.rvf"), 0..N);
    // Delete a fifth of the bucket < 50 ids (well under the 25% index cap).
    let doomed: Vec<u64> = (0..N).filter(|id| id % 100 < 50 && id % 5 == 0).collect();
    store.delete(&doomed).unwrap();
    let filter = FilterExpr::Lt(0, FilterValue::U64(50));
    let (recall, served) = compare(&store, &filter, 20);
    assert!(served);
    assert!(recall >= 0.95, "recall@10 {recall:.3}");
}

#[test]
fn filter_budget_still_aborts_on_the_index_path() {
    let dir = TempDir::new().unwrap();
    let store = build_store(&dir.path().join("f.rvf"), 0..N);
    let mut o = opts(FilterExpr::Lt(0, FilterValue::U64(50)), false);
    o.metadata_filter_max_records = 100;
    let e = store.query(&lcg_vector(1), K, &o).unwrap_err();
    assert_eq!(e, err(ErrorCode::FilterBudgetExceeded));
    // The budget covers the index path and any exact fallback together:
    // one and a half scans of the store is enough for a query that falls
    // back, because the fallback is charged to the same counters.
    o.metadata_filter_max_records = N + N / 2;
    assert!(store.query(&lcg_vector(1), K, &o).is_ok());
}

#[test]
fn cow_child_filtered_query_uses_both_graphs_and_matches_exact() {
    let dir = TempDir::new().unwrap();
    let base_path = dir.path().join("base.rvf");
    let base = build_store(&base_path, 0..N);
    base.close().unwrap();
    let base = RvfStore::open(&base_path).unwrap();
    let mut child = base.branch(&dir.path().join("child.rvf")).unwrap();
    base.close().unwrap();

    // Child edits: 1500 new vectors (child graph >= INDEX_MIN_VECTORS),
    // tombstones of inherited ids.
    ingest(&mut child, 10_000..11_500);
    let tombstones: Vec<u64> = (0..200).map(|i| i * 7).collect();
    child.delete(&tombstones).unwrap();

    let filter = FilterExpr::Lt(0, FilterValue::U64(30));
    let (recall, served) = compare(&child, &filter, 20);
    assert!(served, "COW filtered query should be served by the graphs");
    assert!(recall >= 0.9, "COW filtered recall@10 {recall:.3}");
}

#[test]
fn graph_recall_is_measured_once_and_cached() {
    let dir = TempDir::new().unwrap();
    let store = build_store(&dir.path().join("f.rvf"), 0..N);
    let ef = INDEX_MIN_EF_SEARCH;
    let recall = store.with_index(|idx| store.graph_recall(idx, ef)).unwrap();
    assert!(
        recall >= FILTERED_HNSW_MIN_GRAPH_RECALL,
        "graph recall {recall:.3}"
    );
    let cached = store.with_index(|idx| idx.cached_graph_recall(ef)).unwrap();
    assert_eq!(cached, Some(recall));
    // A different ef is a different measurement.
    assert_eq!(
        store
            .with_index(|idx| idx.cached_graph_recall(ef + 1))
            .unwrap(),
        None
    );
}

#[test]
fn low_graph_recall_routes_filtered_queries_to_the_exact_scan() {
    let dir = TempDir::new().unwrap();
    let store = build_store(&dir.path().join("f.rvf"), 0..N);
    let filter = FilterExpr::Lt(0, FilterValue::U64(50));
    // Pretend the graph measured badly: the planner must not trust it.
    store
        .with_index(|idx| idx.store_graph_recall(INDEX_MIN_EF_SEARCH, 0.5))
        .unwrap();
    let (recall, served) = compare(&store, &filter, 10);
    assert!(!served, "a weak graph must not serve filtered queries");
    assert_eq!(recall, 1.0);

    // PreferLatency opts back in to the walk regardless of the measurement.
    let mut o = opts(filter, false);
    o.quality_preference = QualityPreference::PreferLatency;
    let (_, served) = store.query_routed(&lcg_vector(7), K, &o).unwrap();
    assert!(served, "PreferLatency keeps the filtered HNSW walk");
}
