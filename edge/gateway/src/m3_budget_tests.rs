//! Post-merge review fixes (ADR-351 M3 on the Workers Free budget): body
//! caps per route, which requests ship audit messages, the synchronous
//! snapshot / export / restore size budget, the drop-vs-restore journal,
//! and queued-import pacing.

use crate::m3_api::{audited, parse};
use crate::m3_http::refusal_audited;
use crate::m3_mem::*;
use crate::m3_ports::QueueName;
use crate::uploads::{MAX_INLINE_BYTES, MAX_PART_BODY};
use serde_json::{json, Value as Json};
use worker::Method;

#[test]
fn only_upload_parts_get_8_mib_and_inline_imports_get_512_kib() {
    let cap = |m: Method, p: &str| parse(&m, p).and_then(|r| r.body_cap());
    assert_eq!(cap(Method::Post, "/v1/uploads/u1/parts/1"), Some(8 << 20));
    assert_eq!(MAX_PART_BODY, 8 << 20);
    assert_eq!(
        cap(Method::Post, "/v1/collections/c:import"),
        Some(512 << 10)
    );
    assert_eq!(MAX_INLINE_BYTES, 512 << 10);
    for p in [
        "/v1/collections/c/snapshots",
        "/v1/collections/c:export",
        "/v1/uploads",
        "/v1/collections/c/vectors:upsert",
    ] {
        assert_eq!(cap(Method::Post, p), None, "{p}");
    }
}

#[test]
fn an_inline_import_over_512_kib_is_refused_before_hashing_or_storing() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    let body = json!({ "name": "docs", "dim": 8, "metric": "cosine" });
    assert_eq!(w.req(&o, Method::Post, "/v1/collections", body).0, 201);
    let path = "/v1/collections/docs:import";
    let big = vec![0u8; MAX_INLINE_BYTES + 1];
    let (s, v) = w.raw(&o, Method::Post, path, big, true, None);
    assert_eq!(s, 413, "{v}");
    assert!(w.blob.keys("staging/").is_empty());
    assert!(w.q.take(QueueName::Ingest).is_empty());
}

#[test]
fn polled_reads_and_throttled_refusals_ship_no_audit_message() {
    let r = |m: Method, p: &str| parse(&m, p).unwrap();
    let job = r(Method::Get, "/v1/jobs/j1");
    let list = r(Method::Get, "/v1/collections/c/snapshots");
    assert!(!audited(&job, b""));
    assert!(!audited(&list, b""));
    let snap = r(Method::Post, "/v1/collections/c/snapshots");
    assert!(audited(&snap, b""));
    // A throttled write is not audited; a denied (401) write still is.
    assert!(!refusal_audited(&snap, 429));
    assert!(refusal_audited(&snap, 401));
    assert!(!refusal_audited(&job, 401));
    assert!(!refusal_audited(&list, 403));
    // End to end: polling a job status ships nothing.
    let w = World::default();
    let o = w.owner("org-a", "alice");
    w.q.take(QueueName::Audit);
    for _ in 0..3 {
        w.req(
            &o,
            Method::Get,
            "/v1/jobs/000000000000000000000000",
            Json::Null,
        );
    }
    assert!(w.q.take(QueueName::Audit).is_empty());
}

/// Seed `n` rows into a new 1-shard, 8-dim collection `big`; returns its uid.
fn seeded(w: &World, o: &crate::rest::Caller, n: usize) -> String {
    let body = json!({ "name": "big", "dim": 8, "metric": "cosine", "shards": 1 });
    assert_eq!(w.req(o, Method::Post, "/v1/collections", body).0, 201);
    let rows: Vec<Json> = (0..n)
        .map(|i| json!({ "id": format!("v{i:04}"), "values": [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, i as f32] }))
        .collect();
    let up = "/v1/collections/big/vectors:upsert";
    assert_eq!(
        w.req(o, Method::Post, up, json!({ "vectors": rows })).0,
        200
    );
    crate::testkit::block_on(w.m3(o).collection("big"))
        .unwrap()
        .uid
}

/// A restore killed after admitting its growth (CPU limit): the journal
/// holds `admitted`, the ledger holds the growth, no settle ever ran.
fn interrupted_restore(w: &World, o: &crate::rest::Caller, uid: &str, grow: i64) {
    use crate::m3_wire::{ledger3, M3LedgerCall, Ns};
    let j = crate::restore::Journal {
        rid: "r1x1".into(),
        epoch: 1,
        started_ms: w.now_ms.get(),
        admitted: Some((grow, grow * 8)),
    };
    let call = M3LedgerCall::KvSwap {
        ns: Ns::Restore,
        key: uid.into(),
        expect: None,
        value: Some(serde_json::to_string(&j).unwrap()),
        admit: Some(ruvector_edge_tenancy::QuotaDelta {
            vectors: grow,
            float_budget: grow * 8,
            ..Default::default()
        }),
        adjust: None,
        now: w.now_ms.get() / 1000,
    };
    crate::testkit::block_on(ledger3(&w.b, o.ctx.tenant_key(), call)).unwrap();
}

fn usage(w: &World, o: &crate::rest::Caller) -> (u64, u64) {
    let u = w.req(o, Method::Get, "/v1/usage", Json::Null).1["usage"].clone();
    (
        u["vectors"].as_u64().unwrap(),
        u["float_budget"].as_u64().unwrap(),
    )
}

#[test]
fn dropping_a_collection_releases_an_interrupted_restores_admitted_growth() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    let uid = seeded(&w, &o, 10);
    interrupted_restore(&w, &o, &uid, 5);
    assert_eq!(usage(&w, &o), (15, 120));
    // Past the lease the journal is stale: the drop settles it first.
    w.now_ms
        .set(w.now_ms.get() + crate::restore::RESTORE_LEASE_MS + 1);
    let (s, v) = w.req(&o, Method::Delete, "/v1/collections/big", Json::Null);
    assert_eq!(s, 200, "{v}");
    assert_eq!(usage(&w, &o), (0, 0), "no phantom usage after the purge");
}

#[test]
fn a_live_restore_journal_defers_the_drop_until_the_lease_ends() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    let uid = seeded(&w, &o, 10);
    interrupted_restore(&w, &o, &uid, 3);
    let (s, v) = w.req(&o, Method::Delete, "/v1/collections/big", Json::Null);
    assert_eq!(s, 409, "{v}");
    // Tombstoned: invisible, and the retry after the lease finishes it.
    assert_eq!(
        w.req(&o, Method::Get, "/v1/collections/big", Json::Null).0,
        404
    );
    w.now_ms
        .set(w.now_ms.get() + crate::restore::RESTORE_LEASE_MS + 1);
    let (s, v) = w.req(&o, Method::Delete, "/v1/collections/big", Json::Null);
    assert_eq!(s, 200, "{v}");
    assert_eq!(usage(&w, &o), (0, 0));
}
