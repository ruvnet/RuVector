//! Integration tests for `PersistentTrajectoryStore` (P1 sidecar).
//!
//! Whole module gated on the `persistence` feature so default builds skip it.

#![cfg(feature = "persistence")]

use ruvllm::sona::persist::{PersistError, PersistentTrajectoryStore};
use ruvllm::sona::types::QueryTrajectory;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Wait until `total_seen` count reflects all submissions and the writer has
/// drained the channel. We don't have a direct "writer queue len" hook, so we
/// rely on `shutdown()` to flush + join, which is the contractual flush point.
fn fresh_path(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.db"));
    (dir, path)
}

#[test]
fn test_record_n_zero_drops() {
    let (_dir, path) = fresh_path("record_n");
    // Channel capacity 20_000 — well above the 10_000 records we submit so the
    // bounded queue should never reject.
    let store = PersistentTrajectoryStore::new(path, 20_000).expect("open");

    let n = 10_000;
    for i in 0..n {
        let t = QueryTrajectory::new(i as u64, vec![i as f32, (i + 1) as f32]);
        // Tight loop: producer outpaces writer, but channel is large enough.
        assert!(store.record(t), "record returned false at i={i}");
    }

    // Flush + join writer.
    let dropped = store.dropped_count();
    let total = store.total_seen();
    store.shutdown().expect("shutdown");

    assert_eq!(dropped, 0, "expected zero drops, got {dropped}");
    assert_eq!(total, n as u64, "total_seen mismatch");
}

#[test]
fn test_restart_replay() {
    let (_dir, path) = fresh_path("restart_replay");

    let store = PersistentTrajectoryStore::new(path.clone(), 256).expect("open");
    let mut originals: Vec<QueryTrajectory> = Vec::with_capacity(50);
    for i in 0..50u64 {
        let t = QueryTrajectory::new(i, vec![i as f32, i as f32 * 0.5, i as f32 * 0.25]);
        originals.push(t.clone());
        assert!(store.record(t));
    }
    store.shutdown().expect("shutdown");

    // Reopen + replay.
    let store2 = PersistentTrajectoryStore::new(path, 256).expect("reopen");
    let recent = store2.load_recent(50).expect("load_recent");
    assert_eq!(recent.len(), 50);

    // load_recent returns newest-first by created_at. Compare query_embedding
    // sets ignoring order — created_at is monotonic but rapid inserts can
    // share timestamps, so sort-by-id is the stable invariant.
    let mut got = recent.clone();
    got.sort_by_key(|t| t.query_embedding[0] as u64);
    let mut want = originals.clone();
    want.sort_by_key(|t| t.query_embedding[0] as u64);

    for (a, b) in got.iter().zip(want.iter()) {
        assert_eq!(a.query_embedding, b.query_embedding);
        assert_eq!(a.steps.len(), b.steps.len());
        assert!((a.final_quality - b.final_quality).abs() < 1e-6);
    }

    store2.shutdown().expect("shutdown 2");
}

#[test]
fn test_p95_latency_under_contention() {
    let (_dir, path) = fresh_path("p95_latency");
    // Generous channel so we measure pure record() overhead (mpsc try_send +
    // counters), not back-pressure.
    let store = Arc::new(
        PersistentTrajectoryStore::new(path, 64_000).expect("open"),
    );

    const THREADS: usize = 4;
    const PER_THREAD: usize = 1_000;

    let mut handles = Vec::with_capacity(THREADS);
    for tid in 0..THREADS {
        let s = Arc::clone(&store);
        handles.push(thread::spawn(move || -> Vec<u128> {
            let mut samples = Vec::with_capacity(PER_THREAD);
            for i in 0..PER_THREAD {
                let id = (tid * PER_THREAD + i) as u64;
                let t = QueryTrajectory::new(id, vec![tid as f32, i as f32]);
                let start = Instant::now();
                s.record(t);
                samples.push(start.elapsed().as_nanos());
            }
            samples
        }));
    }

    let mut all: Vec<u128> = handles
        .into_iter()
        .flat_map(|h| h.join().expect("join"))
        .collect();
    all.sort_unstable();
    let p95_idx = (all.len() as f64 * 0.95) as usize;
    let p95_ns = all[p95_idx.min(all.len() - 1)];
    let p95_us = p95_ns as f64 / 1_000.0;

    // Report only — handoff says "report the number, no strict gate".
    eprintln!("P95 record() latency: {:.3} us ({} ns)", p95_us, p95_ns);

    // Force-flush before tempdir drops (avoids writer racing the dir cleanup).
    drop(store);
}

#[test]
fn test_schema_version_mismatch() {
    let (_dir, path) = fresh_path("schema_mismatch");

    // Phase 1: open + close to materialize schema.
    let store = PersistentTrajectoryStore::new(path.clone(), 16).expect("open");
    store.shutdown().expect("shutdown");

    // Phase 2: tamper with schema_meta to a version we don't support.
    {
        let conn = rusqlite::Connection::open(&path).expect("raw open");
        conn.execute("UPDATE schema_meta SET version = 999", [])
            .expect("tamper");
        // Wait briefly for any WAL flush.
        thread::sleep(Duration::from_millis(20));
    }

    // Phase 3: reopen via PersistentTrajectoryStore — must error.
    let res = PersistentTrajectoryStore::new(path, 16);
    match res {
        Err(PersistError::SchemaMismatch { db, expected }) => {
            assert_eq!(db, 999);
            assert_eq!(expected, 1);
        }
        Ok(_) => panic!("expected SchemaMismatch, got Ok"),
        Err(other) => panic!("expected SchemaMismatch, got {other:?}"),
    }
}
