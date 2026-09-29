//! rv-quant on Workers Free (ADR-351 §10, M4 review): a turn that loads a
//! shard never also flushes it (either fits 10 ms, both may not — and a
//! CPU-killed turn loses its writes, so the shard would re-open and be
//! killed on every alarm), the row cap is a `413` checked before any load,
//! a rebuilding shard counts against the resident budget, and the three M4
//! / M2 resident budgets split the isolate cap.

use crate::quant_shard::{self, QuantHost, QUANT_RESIDENT_CAP_BYTES};
use crate::quant_store::QMeta;
use crate::quant_tests::{clustered, query, rows, upsert};
use crate::sqlite_mem::SqliteStore;
use ruvector_edge_store::resident::ISOLATE_RESIDENT_CAP_BYTES;
use ruvector_edge_store::{ErrorCode, ResidentRegistry};

#[test]
fn load_turn_never_flushes_then_next_alarm_does() {
    const DIM: u32 = 64;
    let st = SqliteStore::default();
    let mut h = QuantHost::default();
    let data = clustered(600, DIM as usize, 11);
    upsert(&mut h, &st, DIM, rows(&data[..300], 0)).unwrap();
    quant_shard::alarm(&mut h, "q0", &st);
    let snap = QMeta::read(&st).unwrap().snap_seq;
    assert!(snap > 0);
    // Rows after the snapshot, then the isolate dies before flushing them.
    upsert(&mut h, &st, DIM, rows(&data[300..], 300)).unwrap();
    let mut cold = QuantHost::default();
    // Alarm 1 cold-opens and replays: that is the whole turn — no flush.
    assert!(quant_shard::alarm(&mut cold, "q0", &st).is_some());
    assert!(cold.is_ready("q0"));
    assert_eq!(
        QMeta::read(&st).unwrap().snap_seq,
        snap,
        "flushed in the load turn"
    );
    // Alarm 2 (shard warm) writes the snapshot.
    assert_eq!(quant_shard::alarm(&mut cold, "q0", &st), None);
    let after = QMeta::read(&st).unwrap();
    assert!(after.snap_seq > snap);
    assert_eq!(after.snap_seq, after.write_seq);
    let mut again = QuantHost::default();
    let got = loop {
        match query(&mut again, &st, DIM, &data[450]) {
            Ok(ids) => break ids,
            Err(e) => assert_eq!(e.code, ErrorCode::ShardUnavailable),
        }
    };
    assert_eq!(got[0], "v450");
    assert_eq!(again.last_load.unwrap().snapshot_rows, 600);
}

#[test]
fn row_cap_is_413_before_any_load() {
    const DIM: u32 = 16;
    let st = SqliteStore::default();
    let mut h = QuantHost::default();
    h.budget.max_vectors = 5;
    let data = clustered(6, DIM as usize, 5);
    upsert(&mut h, &st, DIM, rows(&data[..5], 0)).unwrap();
    // A cold isolate: the over-cap write is refused from the counters,
    // without opening (loading) the shard.
    let mut cold = QuantHost::default();
    cold.budget.max_vectors = 5;
    let e = upsert(&mut cold, &st, DIM, rows(&data[5..], 5)).unwrap_err();
    assert_eq!((e.code, e.code.status()), (ErrorCode::BudgetExceeded, 413));
    assert!(cold.last_load.is_none(), "loaded before refusing");
    // Overwriting an existing id adds no row: still accepted.
    upsert(&mut cold, &st, DIM, rows(&data[..1], 0)).unwrap();
    assert_eq!(QMeta::read(&st).unwrap().count, 5);
    // The Free profile keeps the 50k design point loadable in one turn.
    let b = quant_shard::edge_budget();
    assert_eq!(b.max_vectors, 50_000);
    assert!(b.max_load_units < ruvector_edge_quant::Budget::default().max_load_units);
}

#[test]
fn rebuilding_shards_count_against_the_resident_budget() {
    const DIM: u32 = 384;
    let st = SqliteStore::default();
    let mut h = QuantHost::default();
    let data = clustered(2_000, DIM as usize, 21);
    for (b, c) in data.chunks(500).enumerate() {
        upsert(&mut h, &st, DIM, rows(c, b * 500)).unwrap();
    }
    // No snapshot: a fresh isolate rebuilds over several turns; after the
    // first one the partly built shard is already registered, so another
    // shard's load can evict it (it is never invisible to the budget).
    let mut cold = QuantHost::default();
    assert!(query(&mut cold, &st, DIM, &data[0]).is_err());
    assert!(!cold.is_ready("q0"));
    assert!(cold.registered_total() > 0);
}

#[test]
fn isolate_resident_budget_is_split_not_tripled() {
    let total = crate::shard_core::VECTOR_RESIDENT_CAP_BYTES
        + QUANT_RESIDENT_CAP_BYTES
        + crate::graph_store::GRAPH_RESIDENT_CAP_BYTES;
    assert!(total <= ISOLATE_RESIDENT_CAP_BYTES, "{total}");
    // Two full 14 MB vector shards still fit (M2 acceptance).
    let mut r = ResidentRegistry::new(crate::shard_core::VECTOR_RESIDENT_CAP_BYTES);
    assert!(r.touch("a", 14_000_000).is_empty());
    assert!(r.touch("b", 14_000_000).is_empty());
}

/// The replay page of a cold open reads only rows written after the
/// snapshot: its key scan is on the `wseq` index (a rowid scan read every
/// row, ≈ 3–4 ms at 50k × 384 with nothing to replay).
#[test]
fn replay_keys_use_the_wseq_index() {
    use ruvector_edge_store::{SqlStore, Value};
    let st = SqliteStore::default();
    crate::quant_store::ensure_schema(&st).unwrap();
    let sql = format!("EXPLAIN QUERY PLAN {}", crate::quant_store::REPLAY_KEYS);
    let plan = format!("{:?}", st.query(&sql, &[Value::Int(1)]).unwrap());
    assert!(plan.contains("qrows_wseq"), "{plan}");
}
