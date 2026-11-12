//! ADR-351 §6.1 M2 limits and alarm maintenance: the 14 MB resident cap
//! (index bytes included) is `413 budget_exceeded` before any allocation;
//! query `ef`/`rerank` and the HNSW sync batch are validated; l2/dot
//! requantize once past their first-batch sample; tombstone-heavy shards
//! compact (flat) or repair + purge (HNSW) and renumber iids densely.

mod common;

use common::m2::*;
use common::T0;
use ruvector_edge_store::shard::{
    Due, IndexSource, HNSW_SYNC_UPSERT, MAX_EF, MAX_RERANK, SHARD_RESIDENT_CAP_BYTES,
};
use ruvector_edge_store::{ErrorCode, MemSqlStore, Metric, UpsertRow, VectorShard};
use serde_json::json;

#[test]
fn resident_cap_counts_index_bytes_and_refuses_with_413() {
    // Flat at 1536-d: codes dominate (≈ 1.5 KB per row).
    let st = common::sqlite::SqliteStore::default();
    let mut s = VectorShard::open(&st).unwrap();
    let (dm, cfg) = (dm(40), cfg(1536, Metric::Cosine, flat()));
    let err = ingest(&mut s, &st, &dm, &cfg, &data(12_000, 1536, 4)).unwrap_err();
    assert_eq!(err, ErrorCode::BudgetExceeded);
    assert!(s.resident_bytes() <= SHARD_RESIDENT_CAP_BYTES);
    assert!(s.len() > 5_000 && s.len() < 9_000, "{} rows", s.len());
    assert!(s.index_bytes() >= s.len() as u64 * 1536);
    // Refused whole: the shard still serves, and nothing was half-applied.
    let n = s.len();
    let q = data(1, 1536, 5).remove(0).1;
    assert_eq!(
        s.query(&st, &dm, &cfg, &req(&q, 5)).unwrap().matches.len(),
        5
    );
    assert_eq!(reopen(&st).len(), n);

    // HNSW: links count too (and a 4 KB filterable value per row).
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    let (dm, cfg) = (common::m2::dm(41), common::m2::cfg(8, Metric::L2, hnsw()));
    let blob = "x".repeat(4000);
    let mut err = None;
    for b in 0..100 {
        let rows = (0..HNSW_SYNC_UPSERT)
            .map(|i| UpsertRow {
                id: format!("h{b:03}-{i:02}"),
                values: vec![(b * 64 + i) as f32; 8],
                metadata: Some(json!({ "g": blob })),
            })
            .collect();
        match s.plan_upsert(&dm, &cfg, rows) {
            Ok(p) => {
                s.apply_upsert(&st, p, ACTOR, T0).unwrap();
            }
            Err(e) => {
                err = Some(e.code);
                break;
            }
        }
    }
    assert_eq!(err, Some(ErrorCode::BudgetExceeded));
    assert!(s.resident_bytes() <= SHARD_RESIDENT_CAP_BYTES);
}

#[test]
fn query_parameters_and_hnsw_batches_are_validated() {
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    let (dm, cfg) = (dm(42), cfg(16, Metric::L2, hnsw()));
    ingest(&mut s, &st, &dm, &cfg, &data(200, 16, 6)).unwrap();
    let q = data(1, 16, 7).remove(0).1;
    let bad = |f: &dyn Fn(&mut ruvector_edge_store::QueryRequest)| {
        let mut r = req(&q, 10);
        f(&mut r);
        r
    };
    for r in [
        bad(&|r| r.ef = Some(0)),
        bad(&|r| r.ef = Some(MAX_EF + 1)),
        bad(&|r| r.rerank = Some(9)),
        bad(&|r| r.rerank = Some(MAX_RERANK + 1)),
    ] {
        let e = s.query(&st, &dm, &cfg, &r).unwrap_err();
        assert_eq!(e.code, ErrorCode::InvalidRequest, "{r:?}");
    }
    let ok = bad(&|r| {
        r.ef = Some(MAX_EF);
        r.rerank = Some(MAX_RERANK);
    });
    assert_eq!(s.query(&st, &dm, &cfg, &ok).unwrap().matches.len(), 10);
    let big = rows(&data(HNSW_SYNC_UPSERT + 1, 16, 8));
    let e = s.plan_upsert(&dm, &cfg, big).unwrap_err();
    assert_eq!(e.code, ErrorCode::PayloadTooLarge);
}

#[test]
fn l2_requantizes_once_past_its_first_batch_sample() {
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    let (dm, cfg) = (dm(43), cfg(16, Metric::L2, flat()));
    let d = data(1500, 16, 9);
    let plan = s.plan_upsert(&dm, &cfg, rows(&d[..100])).unwrap();
    s.apply_upsert(&st, plan, ACTOR, T0).unwrap();
    assert_eq!(
        s.quant().map(|q| (q.rows, q.params.epoch())),
        Some((100, 1))
    );
    ingest(&mut s, &st, &dm, &cfg, &d[100..]).unwrap();
    // Retrained from a strided sample of `vectors` (a rebase flush).
    let q = s.quant().unwrap();
    assert!(
        q.params.epoch() >= 2 && q.rows >= 1000,
        "{:?}",
        (q.rows, q.params.epoch())
    );
    assert_rebase(&s);
    assert_ne!(s.maintenance_due(), Some(Due::Now));
    let mut cold = reopen(&st);
    assert_eq!(cold.quant(), s.quant());
    for (_, v) in data(5, 16, 10) {
        let want = brute(Metric::L2, &d, &v, 10);
        let got = ranked(&mut cold, &st, &dm, &cfg, &req(&v, 10));
        assert_eq!(got.iter().map(|g| g.0.clone()).collect::<Vec<_>>(), want);
    }
}

#[test]
fn tombstones_compact_and_renumber_densely() {
    // Real SQLite enforces `vectors.iid UNIQUE` per statement: the
    // ascending `new ≤ old` renumbering must never collide.
    tombstones::<common::sqlite::SqliteStore>();
    tombstones::<MemSqlStore>();
}

fn tombstones<S: ruvector_edge_store::SqlStore + Default>() {
    for index in [flat(), hnsw()] {
        let st = S::default();
        let mut s = VectorShard::open(&st).unwrap();
        let (dm, cfg) = (dm(44), cfg(16, Metric::Cosine, index));
        let d = data(2000, 16, 11);
        ingest(&mut s, &st, &dm, &cfg, &d).unwrap();
        let gone: Vec<String> = (0..2000).step_by(3).map(|i| format!("v{i:05}")).collect();
        for c in gone.chunks(100) {
            s.delete(&st, &dm, c, ACTOR, false, T0).unwrap();
        }
        let kept: Vec<_> = d
            .iter()
            .filter(|(id, _)| !gone.contains(id))
            .cloned()
            .collect();
        let exact = |s: &mut VectorShard, q: &[f32]| {
            let mut r = req(q, 10);
            r.rerank = Some(1000);
            r.ef = Some(1000);
            ranked(s, &st, &dm, &cfg, &r)
        };
        let qs = data(6, 16, 12);
        let before: Vec<_> = qs.iter().map(|(_, q)| exact(&mut s, q)).collect();
        assert_eq!(s.maintenance_due(), Some(Due::Now), "{index:?}");
        let rep = s.maintain(&st).unwrap();
        assert!(rep.compacted && rep.flushed, "{rep:?}");
        assert_rebase(&s);
        let after: Vec<_> = qs.iter().map(|(_, q)| exact(&mut s, q)).collect();
        assert_eq!(before, after, "{index:?}");
        for ((_, q), got) in qs.iter().zip(&after) {
            let ids: Vec<String> = got.iter().map(|g| g.0.clone()).collect();
            assert_eq!(ids, brute(Metric::Cosine, &kept, q, 10));
        }
        // Densely renumbered and persisted: a cold load decodes it.
        let mut cold = reopen(&st);
        assert_eq!(cold.index_source(), Some(IndexSource::Current));
        assert_eq!(cold.state_digest(), s.state_digest());
        let again: Vec<_> = qs.iter().map(|(_, q)| exact(&mut cold, q)).collect();
        assert_eq!(again, after);
        // New writes allocate after the dense range and stay queryable.
        ingest(&mut s, &st, &dm, &cfg, &data(10, 16, 13)[..]).unwrap();
        assert_eq!(s.len(), kept.len() + 4); // v00000, 03, 06, 09 were deleted
    }
}

/// A rebase is persisted as two epochs with the same payload and `seq`
/// (the older one is the corrupted-chunk fallback).
fn assert_rebase(s: &VectorShard) {
    let st = s.index_state();
    let (cur, prev) = (st.cur.unwrap(), st.prev.unwrap());
    assert_eq!((cur.seq, cur.sha256), (prev.seq, prev.sha256));
    assert!(prev.epoch < cur.epoch);
}
