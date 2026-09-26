//! Integration test: compares `CdcLocalStorage` (feature `cdc`) against the
//! real, production `LocalStorage::save` path on identical, evolving
//! `SnapshotData` — the "existing implementation" comparison named in
//! ADR-350, replacing the crate-local synthetic-format benchmark it
//! started from.

#![cfg(feature = "cdc")]

use ruvector_snapshot::{
    CdcLocalStorage, CollectionConfig, DistanceMetric, LocalStorage, Snapshot, SnapshotData,
    SnapshotStorage, VectorRecord,
};

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn random_vector(state: &mut u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|_| (splitmix64(state) >> 40) as f32 / (1u64 << 24) as f32)
        .collect()
}

/// Builds a real `SnapshotData` snapshot of the current in-memory rows,
/// exactly as production code constructs one before calling `save`.
fn build_snapshot_data(collection: &str, dim: usize, rows: &[(String, Vec<f32>)]) -> SnapshotData {
    let config = CollectionConfig {
        dimension: dim,
        metric: DistanceMetric::Cosine,
        hnsw_config: None,
    };
    let vectors = rows
        .iter()
        .map(|(id, v)| VectorRecord::new(id.clone(), v.clone(), None))
        .collect();
    SnapshotData::new(collection.to_string(), config, vectors)
}

/// One round of realistic agent-memory-style churn: a few new rows, a few
/// removed, a few replaced in place — same shape used by the original
/// synthetic-format benchmark, applied here to real `VectorRecord` rows.
fn churn(
    rows: &mut Vec<(String, Vec<f32>)>,
    state: &mut u64,
    dim: usize,
    next_id: &mut u64,
    n_insert: usize,
    n_delete: usize,
    n_update: usize,
) {
    for _ in 0..n_insert {
        let id = format!("v{next_id}");
        *next_id += 1;
        rows.push((id, random_vector(state, dim)));
    }
    for _ in 0..n_delete {
        if rows.is_empty() {
            break;
        }
        let idx = (splitmix64(state) as usize) % rows.len();
        rows.remove(idx);
    }
    for _ in 0..n_update {
        if rows.is_empty() {
            break;
        }
        let idx = (splitmix64(state) as usize) % rows.len();
        rows[idx].1 = random_vector(state, dim);
    }
}

#[tokio::test]
async fn cdc_backend_round_trips_exactly_against_real_local_storage_format() {
    let dim = 64;
    let mut state = 0xC0FFEEu64;
    let mut next_id = 0u64;
    let mut rows: Vec<(String, Vec<f32>)> = (0..500)
        .map(|_| {
            let id = format!("v{next_id}");
            next_id += 1;
            (id, random_vector(&mut state, dim))
        })
        .collect();

    let local_dir = std::env::temp_dir().join(format!(
        "ruvector-snapshot-cdc-cmp-local-{}",
        std::process::id()
    ));
    let cdc_dir = std::env::temp_dir().join(format!(
        "ruvector-snapshot-cdc-cmp-cdc-{}",
        std::process::id()
    ));
    let local = LocalStorage::new(local_dir.clone());
    let cdc = CdcLocalStorage::new(cdc_dir.clone());

    let mut local_total_bytes: u64 = 0;
    let mut cdc_last_size_bytes: u64 = 0;

    for round in 0..10 {
        if round > 0 {
            churn(&mut rows, &mut state, dim, &mut next_id, 5, 3, 8);
        }
        let data = build_snapshot_data("agent-memory", dim, &rows);

        let local_snapshot: Snapshot = local.save(&data).await.expect("LocalStorage::save");
        let cdc_snapshot: Snapshot = cdc.save(&data).await.expect("CdcLocalStorage::save");

        // Both backends must report the same checksum semantics (SHA-256
        // of the uncompressed bincode-encoded SnapshotData) for the same
        // logical content — this is a compatibility property, not a
        // coincidence: both compute it the same way.
        assert_eq!(local_snapshot.checksum, cdc_snapshot.checksum);
        assert_eq!(local_snapshot.vectors_count, cdc_snapshot.vectors_count);

        local_total_bytes += local_snapshot.size_bytes;
        cdc_last_size_bytes = cdc_snapshot.size_bytes;

        // Round-trip correctness against the REAL production decode path.
        let restored = local
            .load(&local_snapshot.id)
            .await
            .expect("LocalStorage::load");
        assert_eq!(restored.vectors_count(), data.vectors_count());
        let restored_cdc = cdc
            .load(&cdc_snapshot.id)
            .await
            .expect("CdcLocalStorage::load");
        assert_eq!(restored_cdc.vectors_count(), data.vectors_count());
        for (a, b) in restored.vectors.iter().zip(data.vectors.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.vector, b.vector);
        }
        for (a, b) in restored_cdc.vectors.iter().zip(data.vectors.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.vector, b.vector);
        }
    }

    // Real, measured comparison against the production backend: total
    // bytes LocalStorage actually wrote across 10 rounds of real gzip'd
    // full snapshots, vs. the CDC backend's on-disk chunk-directory size
    // after the same 10 rounds (post-dedup, real gzip'd chunks).
    let cdc_dir_bytes = dir_size(&cdc_dir);
    println!(
        "LocalStorage cumulative bytes written (10 rounds): {local_total_bytes}\n\
         CdcLocalStorage on-disk size after 10 rounds:       {cdc_dir_bytes}\n\
         CdcLocalStorage last reported size_bytes (this-snapshot-restorable): {cdc_last_size_bytes}"
    );
    assert!(
        cdc_dir_bytes < local_total_bytes,
        "expected CDC on-disk footprint ({cdc_dir_bytes}) to be smaller than LocalStorage's \
         cumulative bytes written ({local_total_bytes}) after {} rounds of small churn",
        10
    );

    let _ = std::fs::remove_dir_all(&local_dir);
    let _ = std::fs::remove_dir_all(&cdc_dir);
}

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}
