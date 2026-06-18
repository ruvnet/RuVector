//! Integration tests for the HNSW index in the runtime query path.
//!
//! Covers:
//! 1. Recall@10 >= 0.95 vs the brute-force baseline (10k x 128, fixed seed)
//! 2. Index persistence round-trip (ingest -> close -> reopen -> query
//!    served by the loaded index, no rebuild)
//! 3. Honest `QualityEnvelope.evidence` (index layers claimed only when
//!    the index actually served the query)
//! 4. Exact-scan fallbacks: tiny stores, filtered queries, high deleted
//!    fraction, and `force_exact`
//! 5. Incremental maintenance across ingest/reopen and compaction

use rvf_runtime::{FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore};
use rvf_types::quality::QualityPreference;
use tempfile::TempDir;

// Deterministic LCG vectors (fixed seed).
fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    for _ in 0..dim {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((x >> 33) as f32) / (u32::MAX as f32) - 0.5);
    }
    v
}

fn make_store(dir: &TempDir, name: &str, dim: u16, count: usize) -> RvfStore {
    let path = dir.path().join(name);
    let options = RvfOptions {
        dimension: dim,
        security_policy: rvf_types::security::SecurityPolicy::Permissive,
        ..Default::default()
    };
    let mut store = RvfStore::create(&path, options).unwrap();
    if count > 0 {
        let vecs: Vec<Vec<f32>> = (0..count)
            .map(|i| random_vector(dim as usize, i as u64))
            .collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (0..count as u64).collect();
        store.ingest_batch(&refs, &ids, None).unwrap();
    }
    store
}

fn exact_opts() -> QueryOptions {
    QueryOptions {
        force_exact: true,
        ..Default::default()
    }
}

fn envelope_opts() -> QueryOptions {
    QueryOptions {
        quality_preference: QualityPreference::AcceptDegraded,
        ..Default::default()
    }
}

// ── 1. Recall vs brute-force baseline ───────────────────────────────

#[test]
fn hnsw_recall_at_10_meets_095_on_10k_128d() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir, "recall.rvf", 128, 10_000);

    let k = 10;
    let num_queries = 25;
    let mut total_recall = 0.0f64;

    for qi in 0..num_queries {
        let query = random_vector(128, 1_000_000 + qi);

        // Ground truth: exact brute-force scan.
        let truth = store.query(&query, k, &exact_opts()).unwrap();
        assert_eq!(truth.len(), k);

        // Index path (default options).
        let approx = store.query(&query, k, &QueryOptions::default()).unwrap();
        assert_eq!(approx.len(), k);

        let truth_ids: std::collections::BTreeSet<u64> = truth.iter().map(|r| r.id).collect();
        let hit = approx.iter().filter(|r| truth_ids.contains(&r.id)).count();
        total_recall += hit as f64 / k as f64;
    }

    let avg_recall = total_recall / num_queries as f64;
    assert!(
        avg_recall >= 0.95,
        "recall@10 = {avg_recall:.3}, expected >= 0.95"
    );

    // The index must have actually served the default-path queries.
    let envelope = store
        .query_with_envelope(&random_vector(128, 999), k, &envelope_opts())
        .unwrap();
    assert!(envelope.evidence.layers_used.layer_a);
    assert!(envelope.evidence.layers_used.layer_c);

    store.close().unwrap();
}

// ── 2. Persistence round-trip ───────────────────────────────────────

#[test]
fn index_persists_on_close_and_loads_on_reopen() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("persist.rvf");
    let dim = 16usize;
    let count = 1500usize;
    let query = random_vector(dim, 42);

    let before = {
        let store = make_store(&dir, "persist.rvf", dim as u16, count);
        // First query builds the index; close() persists it as INDEX_SEG.
        let results = store.query(&query, 10, &QueryOptions::default()).unwrap();
        assert!(store.index_ready());
        store.close().unwrap();
        results
    };

    {
        let store = RvfStore::open(&path).unwrap();
        // Loaded from the INDEX_SEG at open time, before any query.
        assert!(
            store.index_ready(),
            "index must load from INDEX_SEG on open"
        );

        // The loaded graph must reproduce the pre-close results exactly.
        let after = store.query(&query, 10, &QueryOptions::default()).unwrap();
        assert_eq!(before, after);

        // And the envelope must attribute the query to the index.
        let envelope = store
            .query_with_envelope(&query, 10, &envelope_opts())
            .unwrap();
        assert!(envelope.evidence.layers_used.layer_a);
        store.close().unwrap();
    }
}

#[test]
fn index_tracks_vectors_ingested_after_reopen() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("incremental.rvf");
    let dim = 16usize;

    {
        let store = make_store(&dir, "incremental.rvf", dim as u16, 1200);
        // Build + persist.
        store
            .query(&random_vector(dim, 7), 5, &QueryOptions::default())
            .unwrap();
        store.close().unwrap();
    }

    {
        let mut store = RvfStore::open(&path).unwrap();
        assert!(store.index_ready());

        // Ingest 300 more vectors; the loaded index must pick them up.
        let vecs: Vec<Vec<f32>> = (1200..1500).map(|i| random_vector(dim, i as u64)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (1200..1500).collect();
        store.ingest_batch(&refs, &ids, None).unwrap();

        let query = random_vector(dim, 1357);
        let envelope = store
            .query_with_envelope(&query, 1, &envelope_opts())
            .unwrap();
        assert!(envelope.evidence.layers_used.layer_a);
        assert_eq!(envelope.results[0].id, 1357);
        assert!(envelope.results[0].distance < f32::EPSILON);
        store.close().unwrap();
    }

    // The updated index (1500 nodes) was persisted by the second close.
    {
        let store = RvfStore::open(&path).unwrap();
        assert!(store.index_ready());
        let results = store
            .query(&random_vector(dim, 1499), 1, &QueryOptions::default())
            .unwrap();
        assert_eq!(results[0].id, 1499);
        store.close().unwrap();
    }
}

// ── 3. Honest evidence ──────────────────────────────────────────────

#[test]
fn evidence_layer_flags_are_honest() {
    let dir = TempDir::new().unwrap();

    // Small store: below the index threshold -> exact scan, no index claim.
    let small = make_store(&dir, "small.rvf", 8, 100);
    let envelope = small
        .query_with_envelope(&random_vector(8, 1), 5, &envelope_opts())
        .unwrap();
    assert!(!envelope.evidence.layers_used.layer_a);
    assert!(!envelope.evidence.layers_used.layer_c);
    assert!(
        !small.index_ready(),
        "no index should be built below threshold"
    );
    small.close().unwrap();

    // Large store: index serves the query -> layers claimed.
    let large = make_store(&dir, "large.rvf", 8, 1500);
    let envelope = large
        .query_with_envelope(&random_vector(8, 2), 5, &envelope_opts())
        .unwrap();
    assert!(envelope.evidence.layers_used.layer_a);
    assert!(envelope.evidence.layers_used.layer_c);

    // Same store, forced exact scan -> no index claim.
    let opts = QueryOptions {
        force_exact: true,
        quality_preference: QualityPreference::AcceptDegraded,
        ..Default::default()
    };
    let envelope = large
        .query_with_envelope(&random_vector(8, 2), 5, &opts)
        .unwrap();
    assert!(!envelope.evidence.layers_used.layer_a);
    assert!(!envelope.evidence.layers_used.layer_c);
    large.close().unwrap();
}

// ── 4. Exact-scan fallbacks ─────────────────────────────────────────

#[test]
fn tiny_store_results_match_exact_scan() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir, "tiny.rvf", 8, 50);
    let query = random_vector(8, 33);
    let default_results = store.query(&query, 10, &QueryOptions::default()).unwrap();
    let exact_results = store.query(&query, 10, &exact_opts()).unwrap();
    assert_eq!(default_results, exact_results);
    store.close().unwrap();
}

#[test]
fn filtered_queries_use_exact_path() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("filtered.rvf");
    let dim = 8usize;
    let count = 1500usize;
    let options = RvfOptions {
        dimension: dim as u16,
        ..Default::default()
    };
    let mut store = RvfStore::create(&path, options).unwrap();
    let vecs: Vec<Vec<f32>> = (0..count).map(|i| random_vector(dim, i as u64)).collect();
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let ids: Vec<u64> = (0..count as u64).collect();
    let metadata: Vec<MetadataEntry> = (0..count)
        .map(|i| MetadataEntry {
            field_id: 0,
            value: MetadataValue::U64((i % 2) as u64),
        })
        .collect();
    store.ingest_batch(&refs, &ids, Some(&metadata)).unwrap();

    let opts = QueryOptions {
        filter: Some(FilterExpr::Eq(0, rvf_runtime::filter::FilterValue::U64(1))),
        quality_preference: QualityPreference::AcceptDegraded,
        ..Default::default()
    };
    let envelope = store
        .query_with_envelope(&random_vector(dim, 5), 10, &opts)
        .unwrap();
    // Filtered queries bypass the index: evidence must not claim it.
    assert!(!envelope.evidence.layers_used.layer_a);
    assert!(!envelope.evidence.layers_used.layer_c);

    // The base query path applies the filter exactly. (The envelope's
    // results may additionally include safety-net candidates, which are
    // a pre-existing, separately-reported merge.)
    let results = store.query(&random_vector(dim, 5), 10, &opts).unwrap();
    assert_eq!(results.len(), 10);
    assert!(results.iter().all(|r| r.id % 2 == 1));
    store.close().unwrap();
}

#[test]
fn high_deletion_fraction_falls_back_to_exact_scan() {
    let dir = TempDir::new().unwrap();
    let mut store = make_store(&dir, "deleted.rvf", 8, 2000);

    // Build the index first.
    store
        .query(&random_vector(8, 1), 5, &QueryOptions::default())
        .unwrap();
    assert!(store.index_ready());

    // Delete 50% of the vectors -- far above the 25% index tolerance.
    let del_ids: Vec<u64> = (0..1000).collect();
    store.delete(&del_ids).unwrap();

    let query = random_vector(8, 1500);
    let envelope = store
        .query_with_envelope(&query, 10, &envelope_opts())
        .unwrap();
    assert!(!envelope.evidence.layers_used.layer_a);
    assert!(envelope.results.iter().all(|r| r.id >= 1000));
    assert_eq!(envelope.results[0].id, 1500);

    store.close().unwrap();
}

#[test]
fn index_path_excludes_soft_deleted_vectors() {
    let dir = TempDir::new().unwrap();
    let mut store = make_store(&dir, "softdel.rvf", 8, 1500);

    // Delete a handful of vectors (well under the 25% tolerance).
    store.delete(&[10, 11, 12, 13, 14]).unwrap();

    // Query exactly at a deleted vector: it must not be returned, and the
    // index must still serve the query.
    let query = random_vector(8, 12);
    let envelope = store
        .query_with_envelope(&query, 10, &envelope_opts())
        .unwrap();
    assert!(envelope.evidence.layers_used.layer_a);
    assert!(envelope.results.iter().all(|r| !(10..=14).contains(&r.id)));
    assert_eq!(envelope.results.len(), 10);

    store.close().unwrap();
}

// ── 5. Compaction invalidates and rebuilds ──────────────────────────

#[test]
fn compaction_invalidates_index_and_queries_stay_correct() {
    let dir = TempDir::new().unwrap();
    let mut store = make_store(&dir, "compact.rvf", 8, 1500);

    // Build + use the index.
    store
        .query(&random_vector(8, 3), 5, &QueryOptions::default())
        .unwrap();
    assert!(store.index_ready());

    store.delete(&(0..100).collect::<Vec<u64>>()).unwrap();
    store.compact().unwrap();

    // The index was invalidated by compaction.
    assert!(!store.index_ready());

    // Queries remain correct (index is rebuilt lazily over live vectors).
    let query = random_vector(8, 700);
    let results = store.query(&query, 5, &QueryOptions::default()).unwrap();
    assert_eq!(results[0].id, 700);
    assert!(results.iter().all(|r| r.id >= 100));
    assert!(store.index_ready());

    store.close().unwrap();
}

// ── 6. Stale persisted index is discarded on reopen ─────────────────

#[test]
fn overwriting_an_id_drops_the_stale_index() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("overwrite.rvf");

    {
        let store = make_store(&dir, "overwrite.rvf", 8, 1200);
        store
            .query(&random_vector(8, 1), 5, &QueryOptions::default())
            .unwrap();
        store.close().unwrap(); // persists the index
    }

    {
        let mut store = RvfStore::open(&path).unwrap();
        assert!(store.index_ready());

        // Re-ingest an existing ID with different data: graph edges for
        // that node are stale, so the index must be dropped.
        let v = random_vector(8, 987_654);
        store.ingest_batch(&[v.as_slice()], &[5], None).unwrap();
        assert!(!store.index_ready());

        // Queries still reflect the overwritten vector.
        let results = store.query(&v, 1, &QueryOptions::default()).unwrap();
        assert_eq!(results[0].id, 5);
        assert!(results[0].distance < f32::EPSILON);
        store.close().unwrap();
    }
}
