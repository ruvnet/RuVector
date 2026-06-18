//! Integration tests for the opt-in RaBitQ two-stage query path.
//!
//! Covers:
//! 1. Recall@10 >= 0.95 vs exact brute force (10k x 128, fixed seed)
//!    through the public query API with `QueryOptions::rabitq`
//! 2. Compression: code + corrections ~32x smaller than f32 vectors
//! 3. Opt-in semantics: default options never use the RaBitQ path
//! 4. Soft-deleted vectors are excluded from two-stage results
//! 5. Vectors ingested after the code book is built are still found

use rvf_quant::rabitq::{RabitqQuantizer, CORRECTION_BYTES};
use rvf_runtime::{QueryOptions, RvfOptions, RvfStore};
use tempfile::TempDir;

// Deterministic LCG vectors (fixed seed), matching index_integration.rs.
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

fn rabitq_opts() -> QueryOptions {
    QueryOptions {
        rabitq: true,
        ..Default::default()
    }
}

fn exact_opts() -> QueryOptions {
    QueryOptions {
        force_exact: true,
        ..Default::default()
    }
}

// ── 1. Recall gate: two-stage RaBitQ vs exact brute force ───────────

#[test]
fn rabitq_recall_at_10_meets_095_on_10k_128d() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir, "rabitq_recall.rvf", 128, 10_000);

    let k = 10;
    let num_queries = 25;
    let mut total_recall = 0.0f64;

    for qi in 0..num_queries {
        let query = random_vector(128, 1_000_000 + qi);

        // Ground truth: exact brute-force scan.
        let truth = store.query(&query, k, &exact_opts()).unwrap();
        assert_eq!(truth.len(), k);

        // Two-stage RaBitQ path (default options: 4x oversampling with
        // the internal candidate-pool floor).
        let approx = store.query(&query, k, &rabitq_opts()).unwrap();
        assert_eq!(approx.len(), k);

        // Rescored distances must be the exact f32 distances.
        for r in &approx {
            let gt = truth.iter().find(|t| t.id == r.id);
            if let Some(gt) = gt {
                assert!((gt.distance - r.distance).abs() < 1e-5);
            }
        }

        let truth_ids: std::collections::BTreeSet<u64> = truth.iter().map(|r| r.id).collect();
        let hit = approx.iter().filter(|r| truth_ids.contains(&r.id)).count();
        total_recall += hit as f64 / k as f64;
    }

    let avg_recall = total_recall / num_queries as f64;
    eprintln!("rabitq two-stage recall@10 (10k x 128, default options) = {avg_recall:.4}");
    assert!(
        avg_recall >= 0.95,
        "rabitq recall@10 = {avg_recall:.3}, expected >= 0.95"
    );

    store.close().unwrap();
}

// ── 2. Compression ratio ────────────────────────────────────────────

#[test]
fn rabitq_codes_are_32x_smaller_than_f32() {
    let dim = 128usize;
    let data: Vec<Vec<f32>> = (0..256).map(|i| random_vector(dim, i as u64)).collect();
    let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
    let rq = RabitqQuantizer::train(&refs, 42);

    let f32_bytes = dim * 4;
    let code_bytes = rq.stored_bytes_per_vector() - CORRECTION_BYTES;
    // The 1-bit code itself is exactly 32x smaller than f32.
    assert_eq!(f32_bytes / code_bytes, 32);
    // Including the 8 correction bytes, the total stays ~32x (>= 20x at
    // 128 dims, asymptotically 32x as dims grow — see rvf-quant tests).
    assert!(
        rq.compression_ratio() >= 20.0,
        "compression ratio {} too low",
        rq.compression_ratio()
    );

    // And the serialized per-vector payload matches the accounting.
    let encoded = rq.code_to_bytes(&rq.encode_code(&data[0]));
    assert_eq!(encoded.len(), rq.stored_bytes_per_vector());
}

// ── 3. Opt-in semantics ─────────────────────────────────────────────

#[test]
fn rabitq_is_opt_in_and_matches_exact_on_small_stores() {
    // Defaults must not enable the two-stage path.
    assert!(!QueryOptions::default().rabitq);

    let dir = TempDir::new().unwrap();
    let store = make_store(&dir, "optin.rvf", 16, 200);
    let query = random_vector(16, 777);

    // Default path on a small store is the exact scan; the rabitq path
    // must return identical (id, distance) results here because the
    // candidate-pool floor covers all 200 vectors and the rescore is
    // exact.
    let exact = store.query(&query, 5, &exact_opts()).unwrap();
    let default_path = store.query(&query, 5, &QueryOptions::default()).unwrap();
    let two_stage = store.query(&query, 5, &rabitq_opts()).unwrap();
    assert_eq!(exact, default_path);
    assert_eq!(exact.len(), two_stage.len());
    for (e, t) in exact.iter().zip(two_stage.iter()) {
        assert_eq!(e.id, t.id);
        assert!((e.distance - t.distance).abs() < 1e-6);
    }

    store.close().unwrap();
}

// ── 4. Soft deletions ───────────────────────────────────────────────

#[test]
fn rabitq_path_excludes_soft_deleted_vectors() {
    let dir = TempDir::new().unwrap();
    let mut store = make_store(&dir, "rabitq_del.rvf", 16, 500);

    // Prime the code book, then delete a handful of vectors.
    store
        .query(&random_vector(16, 1), 5, &rabitq_opts())
        .unwrap();
    store.delete(&[10, 11, 12, 13, 14]).unwrap();

    // Query at a deleted vector: it must not be returned.
    let query = random_vector(16, 12);
    let results = store.query(&query, 10, &rabitq_opts()).unwrap();
    assert_eq!(results.len(), 10);
    assert!(results.iter().all(|r| !(10..=14).contains(&r.id)));

    store.close().unwrap();
}

// ── 5. Incremental ingest ───────────────────────────────────────────

#[test]
fn rabitq_code_book_tracks_new_and_overwritten_vectors() {
    let dir = TempDir::new().unwrap();
    let mut store = make_store(&dir, "rabitq_sync.rvf", 16, 300);

    // Build the code book.
    store
        .query(&random_vector(16, 2), 5, &rabitq_opts())
        .unwrap();

    // Ingest new vectors: they must be found via the two-stage path.
    let v = random_vector(16, 123_456);
    store.ingest_batch(&[v.as_slice()], &[900], None).unwrap();
    let results = store.query(&v, 1, &rabitq_opts()).unwrap();
    assert_eq!(results[0].id, 900);
    assert!(results[0].distance < f32::EPSILON);

    // Overwrite an existing ID with different data: the stale code must
    // not shadow the new contents.
    let w = random_vector(16, 654_321);
    store.ingest_batch(&[w.as_slice()], &[42], None).unwrap();
    let results = store.query(&w, 1, &rabitq_opts()).unwrap();
    assert_eq!(results[0].id, 42);
    assert!(results[0].distance < f32::EPSILON);

    store.close().unwrap();
}
