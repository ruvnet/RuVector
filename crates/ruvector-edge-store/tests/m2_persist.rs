//! ADR-351 §6.1 / §15 M2: index persistence in `index_chunks` and lazy
//! load — round trip, bounded replay, corrupted chunks falling back with
//! identical results, `iid_base = 1`, and the same behaviour on the mock
//! and on real SQLite.

mod common;

use common::m2::*;
use common::sqlite::SqliteStore;
use ruvector_edge_store::shard::{
    ann::hnsw_params, Due, IndexConfig, IndexSource, FLUSH_OPS, IID_BASE,
};
use ruvector_edge_store::{MemSqlStore, Metric, SqlStore, VectorShard};

const DIM: usize = 32;

/// Live shard with `n` rows plus a tail of `extra` more ops (< 200), so the
/// store holds a current and a previous epoch and an unpersisted tail.
fn build<S: SqlStore + Default>(
    index: IndexConfig,
    metric: Metric,
    n: usize,
    extra: usize,
) -> (
    S,
    VectorShard,
    ruvector_edge_tenancy::DoMeta,
    ruvector_edge_store::ShardConfig,
) {
    let st = S::default();
    let mut s = VectorShard::open(&st).unwrap();
    let (dm, cfg) = (dm(7), cfg(DIM as u32, metric, index));
    ingest(&mut s, &st, &dm, &cfg, &data(n, DIM, 1)).unwrap();
    // Re-upserts (in-place updates) and deletes in the unpersisted tail.
    let mut tail = data(extra, DIM, 2);
    for (i, t) in tail.iter_mut().enumerate() {
        t.0 = format!("v{:05}", i * 3);
    }
    ingest(&mut s, &st, &dm, &cfg, &tail).unwrap();
    let gone: Vec<String> = (0..extra / 4)
        .map(|i| format!("v{:05}", i * 7 + 1))
        .collect();
    s.delete(&st, &dm, &gone, ACTOR, false, T0).unwrap();
    assert!(s.pending_ops() > 0 && s.pending_ops() < FLUSH_OPS);
    (st, s, dm, cfg)
}

use common::T0;

fn same_answers(
    a: &mut VectorShard,
    b: &mut VectorShard,
    st: &dyn SqlStore,
    dm: &ruvector_edge_tenancy::DoMeta,
    cfg: &ruvector_edge_store::ShardConfig,
) {
    for (_, q) in data(12, DIM, 99) {
        let mut r = req(&q, 10);
        assert_eq!(ranked(a, st, dm, cfg, &r), ranked(b, st, dm, cfg, &r));
        r.filter = Some(serde_json::json!({"g": 1}));
        assert_eq!(ranked(a, st, dm, cfg, &r), ranked(b, st, dm, cfg, &r));
    }
}

#[test]
fn round_trip_decodes_the_current_epoch_and_replays_the_tail() {
    for index in [flat(), hnsw()] {
        let (st, mut live, dm, cfg) = build::<SqliteStore>(index, Metric::L2, 1500, 120);
        let state = live.index_state();
        assert!(state.cur.is_some() && state.prev.is_some(), "{state:?}");
        let mut cold = reopen(&st);
        assert_eq!(cold.index_source(), Some(IndexSource::Current));
        assert_eq!(cold.pending_ops(), live.pending_ops());
        assert!(cold.pending_ops() < FLUSH_OPS, "replay ≤ 199 ops");
        assert_eq!(cold.state_digest(), live.state_digest());
        same_answers(&mut cold, &mut live, &st, &dm, &cfg);
        // The timer flush (alarm) persists the tail; nothing left to replay.
        live.maintain(&st).unwrap();
        assert_eq!(live.pending_ops(), 0);
        let mut cold = reopen(&st);
        assert_eq!(cold.index_source(), Some(IndexSource::Current));
        same_answers(&mut cold, &mut live, &st, &dm, &cfg);
    }
}

#[test]
fn corrupted_current_epoch_falls_back_to_previous_plus_replay() {
    for index in [flat(), hnsw()] {
        let (st, mut live, dm, cfg) = build::<SqliteStore>(index, Metric::Cosine, 1500, 150);
        let cur = live.index_state().cur.unwrap();
        corrupt_chunk(&st, cur.epoch, 0);
        let mut cold = reopen(&st);
        assert_eq!(
            cold.index_source(),
            Some(IndexSource::Previous),
            "{index:?}"
        );
        // Identical results: epoch N-1 + the op tail rebuilds the live index
        // bit for bit (same codes, same per-op HNSW levels).
        same_answers(&mut cold, &mut live, &st, &dm, &cfg);
        // The fallback re-persisted the state: the next load is clean.
        let mut again = reopen(&st);
        assert_eq!(again.index_source(), Some(IndexSource::Current));
        same_answers(&mut again, &mut live, &st, &dm, &cfg);
    }
}

#[test]
fn every_epoch_corrupt_rebuilds_flat_identically_from_vectors() {
    for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
        let (st, mut live, dm, cfg) = build::<MemSqlStore>(flat(), metric, 1200, 90);
        let s = live.index_state();
        for rec in [s.cur.unwrap(), s.prev.unwrap()] {
            corrupt_chunk(&st, rec.epoch, rec.parts - 1);
        }
        let mut cold = reopen(&st);
        assert_eq!(cold.index_source(), Some(IndexSource::Rebuilt));
        // Quantizer params are persisted in meta, so the rebuilt codes are
        // the live codes: identical results.
        same_answers(&mut cold, &mut live, &st, &dm, &cfg);
    }
}

#[test]
fn hnsw_every_epoch_corrupt_rebuilds_in_the_alarm_not_the_request() {
    // The rebuild renumbers `vectors.iid` (UNIQUE on real SQLite).
    hnsw_rebuild::<SqliteStore>();
    hnsw_rebuild::<MemSqlStore>();
}

fn hnsw_rebuild<S: SqlStore + Default>() {
    let (st, mut live, dm, cfg) = build::<S>(hnsw(), Metric::L2, 800, 60);
    // A graph rebuilt from `vectors` is a different (valid) graph, so compare
    // exact answers (rerank ≥ N). They are taken before the cold load: its
    // rebuild renumbers `vectors.iid` (deletes left gaps), which a second
    // resident copy of the same DO — impossible in a Worker — would miss.
    let exact: Vec<_> = data(5, DIM, 5)
        .into_iter()
        .map(|(_, q)| {
            let mut r = req(&q, 10);
            r.rerank = Some(1000);
            r.ef = Some(1000);
            let want = ranked(&mut live, &st, &dm, &cfg, &r);
            (r, want)
        })
        .collect();
    let s = live.index_state();
    for rec in [s.cur.unwrap(), s.prev.unwrap()] {
        corrupt_chunk(&st, rec.epoch, 0);
    }
    // A request never builds a graph (tens of CPU seconds at the cap): it
    // answers 503 without poisoning, so the alarm is scheduled and runs.
    let mut cold = VectorShard::open(&st).unwrap();
    let err = cold.load_index(&st).unwrap_err();
    assert_eq!(err.code, ruvector_edge_store::ErrorCode::ShardUnavailable);
    assert!(!cold.is_poisoned() && cold.rebuild_pending());
    let err = cold.query(&st, &dm, &cfg, &exact[0].0).unwrap_err();
    assert_eq!(err.code, ruvector_edge_store::ErrorCode::ShardUnavailable);
    assert_eq!(cold.maintenance_due(), Some(Due::Now));
    let rep = cold.maintain(&st).unwrap();
    assert!(rep.rebuilt && rep.flushed, "{rep:?}");
    assert_eq!(cold.index_source(), Some(IndexSource::Rebuilt));
    assert!(!cold.rebuild_pending());
    assert_eq!(cold.len(), live.len());
    for (r, want) in &exact {
        assert_eq!(&ranked(&mut cold, &st, &dm, &cfg, r), want);
    }
    // It persisted densely, as two copies: the next load decodes it, and
    // with one copy corrupted it falls back with identical answers.
    let mut again = reopen(&st);
    assert_eq!(again.index_source(), Some(IndexSource::Current));
    same_answers(&mut again, &mut cold, &st, &dm, &cfg);
    corrupt_chunk(&st, again.index_state().cur.unwrap().epoch, 0);
    let mut fell_back = reopen(&st);
    assert_eq!(fell_back.index_source(), Some(IndexSource::Previous));
    same_answers(&mut fell_back, &mut cold, &st, &dm, &cfg);
}

#[test]
fn a_rebase_keeps_a_fallback_copy_with_identical_answers() {
    // ADR §6.1: "corrupted chunk → fall back to replay with identical
    // results" must hold right after a rebase too (compaction here), with
    // no write in between to create a second epoch.
    for index in [flat(), hnsw()] {
        let st = SqliteStore::default();
        let mut live = VectorShard::open(&st).unwrap();
        let (dm, cfg) = (dm(9), cfg(DIM as u32, Metric::Cosine, index));
        ingest(&mut live, &st, &dm, &cfg, &data(1600, DIM, 21)).unwrap();
        let gone: Vec<String> = (0..1600).step_by(2).map(|i| format!("v{i:05}")).collect();
        for c in gone.chunks(100) {
            live.delete(&st, &dm, c, ACTOR, false, T0).unwrap();
        }
        while live.maintenance_due() == Some(Due::Now) {
            live.maintain(&st).unwrap();
        }
        let s = live.index_state();
        let (cur, prev) = (s.cur.unwrap(), s.prev.unwrap());
        assert_eq!((cur.seq, cur.sha256), (prev.seq, prev.sha256), "{index:?}");
        corrupt_chunk(&st, cur.epoch, cur.parts - 1);
        let mut cold = reopen(&st);
        assert_eq!(cold.index_source(), Some(IndexSource::Previous));
        same_answers(&mut cold, &mut live, &st, &dm, &cfg);
    }
}

#[test]
fn hnsw_iids_start_at_the_index_base() {
    let cfg = cfg(DIM as u32, Metric::L2, hnsw());
    assert_eq!(IID_BASE, 1);
    assert_eq!(hnsw_params(&cfg).iid_base, 1);
    // The first flush of a dense 1.. iid space succeeds (a base of 0 would
    // leave slot 0 absent and the encoder refuses gapped iids).
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    ingest(&mut s, &st, &dm(8), &cfg, &data(64, DIM, 3)).unwrap();
    assert!(s.index_state().cur.is_none());
    s.maintain(&st).unwrap();
    assert!(s.index_state().cur.is_some());
    assert_eq!(s.maintenance_due(), None);
    assert_eq!(reopen(&st).index_source(), Some(IndexSource::Current));
}

#[test]
fn mock_and_sqlite_answer_identically() {
    for index in [flat(), hnsw()] {
        let (m, mut ms, dm, cfg) = build::<MemSqlStore>(index, Metric::Dot, 600, 70);
        let (q, mut qs, _, _) = build::<SqliteStore>(index, Metric::Dot, 600, 70);
        assert_eq!(ms.state_digest(), qs.state_digest());
        assert_eq!(ms.index_state(), qs.index_state());
        for (_, v) in data(8, DIM, 42) {
            let r = req(&v, 10);
            assert_eq!(
                ranked(&mut ms, &m, &dm, &cfg, &r),
                ranked(&mut qs, &q, &dm, &cfg, &r)
            );
        }
        // Fetch by id (values + metadata from SQLite), deleted ids omitted.
        let ids: Vec<String> = ["v00001", "v00002", "v00003", "nope"]
            .map(String::from)
            .to_vec();
        let got = qs.fetch(&q, &dm, &ids, true).unwrap();
        assert_eq!(got, ms.fetch(&m, &dm, &ids, true).unwrap());
        let got: Vec<&str> = got.iter().map(|g| g.id.as_str()).collect();
        assert_eq!(got, ["v00002", "v00003"], "v00001 was deleted");
        let one = qs.fetch(&q, &dm, &ids[1..2], true).unwrap().remove(0);
        assert_eq!(one.values.map(|v| v.len()), Some(DIM));
        assert!(one.metadata.is_some_and(|m| m.get("g").is_some()));
    }
}
