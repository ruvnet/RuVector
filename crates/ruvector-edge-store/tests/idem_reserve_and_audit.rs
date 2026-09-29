//! `op_id` reservations (concurrent requests with one key never both
//! execute) and the `ops.act_sub` audit column.

mod common;

use common::*;
use ruvector_edge_store::ledger::{
    IdemKey, IdemLookup, IDEM_PENDING_TTL_SECS, MAX_IDEM_RESPONSE_BYTES,
};
use ruvector_edge_store::shard::Actor;
use ruvector_edge_store::{
    ledger_meta_for, schema, shard_meta_for, MemSqlStore, Metric, ShardConfig, SqlStore,
    TenantLedger, UpsertRow, Value, VectorShard,
};
use ruvector_edge_tenancy::{CollectionUid, ShardIndex};

fn key<'a>(k: &'a str, h: &'a [u8; 32]) -> IdemKey<'a> {
    IdemKey {
        sub: "es1_a",
        key: k,
        body_sha256: h,
    }
}

#[test]
fn reservation_serialises_one_key() {
    let st = MemSqlStore::new();
    let lm = ledger_meta_for(&tenant("org-r")).unwrap();
    let mut l = TenantLedger::open(&st, limits()).unwrap();
    let (h, other) = ([1u8; 32], [2u8; 32]);
    let k = op_id(1);
    // First request reserves; a concurrent twin sees it in flight, a
    // different body under the same key is a conflict.
    assert_eq!(
        l.idem_reserve(&st, &lm, key(&k, &h), T0).unwrap(),
        IdemLookup::Miss
    );
    assert_eq!(st.row_count("idempotency"), 1);
    assert_eq!(
        l.idem_reserve(&st, &lm, key(&k, &h), T0).unwrap(),
        IdemLookup::InFlight
    );
    assert_eq!(
        l.idem_lookup(&st, &lm, key(&k, &h), T0).unwrap(),
        IdemLookup::InFlight
    );
    assert_eq!(
        l.idem_reserve(&st, &lm, key(&k, &other), T0).unwrap(),
        IdemLookup::Conflict
    );
    // A pending row is uncharged; completing it replaces it and charges.
    assert_eq!(l.usage(&lm, T0).unwrap().bytes, 0);
    assert!(l
        .idem_store(&st, &lm, key(&k, &h), "{\"ok\":1}", T0)
        .unwrap());
    assert_eq!(
        l.idem_reserve(&st, &lm, key(&k, &h), T0).unwrap(),
        IdemLookup::Replay("{\"ok\":1}".into())
    );
    assert_eq!(
        l.usage(&lm, T0).unwrap().bytes,
        key(&k, &h).row_bytes("{\"ok\":1}")
    );
}

#[test]
fn release_and_expiry_free_the_key() {
    let st = MemSqlStore::new();
    let lm = ledger_meta_for(&tenant("org-r")).unwrap();
    let mut l = TenantLedger::open(&st, limits()).unwrap();
    let (h, other) = ([1u8; 32], [2u8; 32]);
    let k = op_id(2);
    assert_eq!(
        l.idem_reserve(&st, &lm, key(&k, &h), T0).unwrap(),
        IdemLookup::Miss
    );
    // Another body's release never clears this reservation.
    l.idem_release(&st, &lm, key(&k, &other)).unwrap();
    assert_eq!(st.row_count("idempotency"), 1);
    l.idem_release(&st, &lm, key(&k, &h)).unwrap();
    assert_eq!(st.row_count("idempotency"), 0);
    // A crashed request's reservation expires on its own.
    assert_eq!(
        l.idem_reserve(&st, &lm, key(&k, &h), T0).unwrap(),
        IdemLookup::Miss
    );
    let later = T0 + IDEM_PENDING_TTL_SECS;
    assert_eq!(
        l.idem_reserve(&st, &lm, key(&k, &other), later).unwrap(),
        IdemLookup::Miss
    );
    // Release never touches a completed row.
    assert!(l
        .idem_store(&st, &lm, key(&k, &other), "{}", later)
        .unwrap());
    l.idem_release(&st, &lm, key(&k, &other)).unwrap();
    assert_eq!(
        l.idem_lookup(&st, &lm, key(&k, &other), later).unwrap(),
        IdemLookup::Replay("{}".into())
    );
}

#[test]
fn unremembered_store_clears_the_reservation_and_stale_bytes_are_released() {
    let st = MemSqlStore::new();
    let lm = ledger_meta_for(&tenant("org-r")).unwrap();
    let mut l = TenantLedger::open(&st, limits()).unwrap();
    let h = [3u8; 32];
    let k = op_id(3);
    l.idem_reserve(&st, &lm, key(&k, &h), T0).unwrap();
    let huge = "x".repeat(MAX_IDEM_RESPONSE_BYTES + 1);
    assert!(!l.idem_store(&st, &lm, key(&k, &h), &huge, T0).unwrap());
    assert_eq!(st.row_count("idempotency"), 0);
    // An expired, charged row that a reservation replaces is released.
    assert!(l.idem_store(&st, &lm, key(&k, &h), "{}", T0).unwrap());
    assert!(l.usage(&lm, T0).unwrap().bytes > 0);
    let later = T0 + 86_401;
    assert_eq!(
        l.idem_reserve(&st, &lm, key(&k, &h), later).unwrap(),
        IdemLookup::Miss
    );
    assert_eq!(l.usage(&lm, later).unwrap().bytes, 0);
}

#[test]
fn ops_log_records_the_adapter_that_acted() {
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    let dm = shard_meta_for(
        &tenant("org-a"),
        CollectionUid::from_bytes([4; 16]),
        ShardIndex::ZERO,
    )
    .unwrap();
    let cfg = ShardConfig {
        dim: 2,
        metric: Metric::L2,
        filterable_keys: vec![],
        float_cap: 1000,
    };
    let row = |id: &str| UpsertRow {
        id: id.into(),
        values: vec![0.0, 1.0],
        metadata: None,
    };
    let user = Actor {
        sub: "es1_u",
        jti: "j1",
        family_id: "f",
        act_sub: None,
    };
    let adapter = Actor {
        act_sub: Some("team-ruv-io"),
        ..user
    };
    let plan = s.plan_upsert(&dm, &cfg, vec![row("a"), row("b")]).unwrap();
    s.apply_upsert(&st, plan, user, T0).unwrap();
    s.delete(&st, &dm, &["a".into()], adapter, false, T0)
        .unwrap();
    let rows = st
        .query(schema::OPS_ACTORS, &[0i64.into(), 100i64.into()])
        .unwrap();
    let got: Vec<(String, String, Option<String>)> = rows
        .iter()
        .map(|r| {
            let text = |i: usize| r[i].as_text().map(str::to_string);
            (text(1).unwrap(), text(2).unwrap(), text(3))
        })
        .collect();
    assert_eq!(
        got,
        [
            ("upsert".into(), "es1_u".into(), None),
            ("upsert".into(), "es1_u".into(), None),
            ("delete".into(), "es1_u".into(), Some("team-ruv-io".into())),
        ]
    );
    assert!(matches!(rows[0][3], Value::Null));
}
