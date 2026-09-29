//! Regressions for the M3 review: import-job audit events and the 24 h
//! window, the multipart sha256 pass, compare-and-swap job saves, enqueue
//! failures, the per-tenant audit chain, what is (not) shipped, and the
//! embedding pre-check.

use crate::audit::{self, ship, ymd, AuditEvent};
use crate::audit_http::{body_event, rest_event};
use crate::ingest::Delivery;
use crate::jobs;
use crate::m3_import_tests::{
    collection, deliver, hash_pass, hex, job, rows, rvf, sink, submit, upload, usage_vectors,
};
use crate::m3_mem::*;
use crate::m3_ports::QueueName;
use crate::m3_wire::{kv_range, Ns};
use crate::rest::ApiRoute;
use crate::testkit::block_on;
use ruvector_edge_auth::Capability;
use ruvector_edge_store::ErrorCode;
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};
use worker::Method;

const HOUR: u64 = 3600 * 1000;

#[test]
fn import_jobs_ship_terminal_audit_events() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    collection(&w, &o, "docs", 8);
    let file = rvf(&rows(600, 8, 1), 8);
    let msg = submit(&w, &o, "docs", &upload(&w, &o, &file, None));
    hash_pass(&w, &msg);
    let r = deliver(&w, &sink(&w), &msg, 1000);
    assert_eq!(r.outcome, Delivery::Done);
    let ev = r.audit.expect("job.done event");
    assert_eq!(
        (ev.route.as_str(), ev.outcome, ev.rows),
        ("job.done", 200, 600)
    );
    assert_eq!(ev.tenant_key, o.ctx.tenant_key().as_str());
    assert_eq!(
        (ev.sub.as_str(), ev.jti.as_str()),
        (o.ctx.sub(), o.ctx.jti())
    );
    assert_eq!(ev.bytes, file.len() as u64);
    assert_eq!(ev.scope.as_deref(), Some("ruvector:write"));
    // A redelivery of a finished job ships nothing more.
    assert_eq!(deliver(&w, &sink(&w), &msg, 1000).audit, None);

    let lie = hex(&[9u8; 32]);
    let bad = submit(&w, &o, "docs", &upload(&w, &o, &file, Some(lie)));
    let ev = deliver(&w, &sink(&w), &bad, 1000).audit.unwrap();
    assert_eq!((ev.route.as_str(), ev.outcome), ("job.failed", 422));

    w.now_ms.set(w.now_ms.get() + 1000); // ids are minted per ms
    let stale = submit(&w, &o, "docs", &upload(&w, &o, &file, None));
    w.now_ms.set(w.now_ms.get() + 25 * HOUR);
    let ev = deliver(&w, &sink(&w), &stale, 1000).audit.unwrap();
    assert_eq!(
        (ev.route.as_str(), ev.outcome, ev.rows),
        ("job.cancelled", 410, 0)
    );
}

#[test]
fn the_op_id_window_counts_from_submission_not_the_last_delivery() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    collection(&w, &o, "docs", 8);
    let file = rvf(&rows(300, 8, 2), 8);
    let msg = submit(&w, &o, "docs", &upload(&w, &o, &file, None));
    // Deliveries that commit nothing (here: the hash pass) do not extend it.
    w.now_ms.set(w.now_ms.get() + 23 * HOUR);
    hash_pass(&w, &msg);
    w.now_ms.set(w.now_ms.get() + HOUR + 1000);
    let r = deliver(&w, &sink(&w), &msg, 1000);
    assert_eq!((r.outcome, r.batches), (Delivery::Done, 0));
    let j = job(&w, &o, &msg.job_id);
    assert_eq!(
        (j["state"].as_str(), j["rows_done"].as_u64()),
        (Some("failed"), Some(0))
    );
    assert_eq!(j["error"], crate::ingest::EXPIRED);
    assert_eq!(usage_vectors(&w, &o), 0);
    // A job that never got a delivery is reported expired, not `queued`.
    let idle = submit(&w, &o, "docs", &upload(&w, &o, &file, None));
    w.now_ms.set(w.now_ms.get() + 25 * HOUR);
    let j = job(&w, &o, &idle.job_id);
    assert_eq!(
        (j["state"].as_str(), j["failure"].as_str()),
        (Some("failed"), Some("cancelled"))
    );
}

#[test]
fn a_multipart_upload_with_a_wrong_sha256_fails_before_any_row() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    collection(&w, &o, "docs", 8);
    // 60k rows: far more than one delivery's batches, so the old
    // end-of-stream check never ran for such files.
    let file = rvf(&rows(60_000, 8, 3), 8);
    let lie = hex(&Sha256::digest(b"not the file"));
    let msg = submit(&w, &o, "docs", &upload(&w, &o, &file, Some(lie)));
    // The pass spans several bounded deliveries, none of which applies a row.
    let (mut r, mut deliveries) = (deliver(&w, &sink(&w), &msg, 50), 1);
    while r.outcome == Delivery::Continue {
        assert_eq!(r.batches, 0, "hash pass applies nothing");
        r = deliver(&w, &sink(&w), &msg, 50);
        deliveries += 1;
    }
    let slices = (file.len() as u64).div_ceil(crate::ingest_hash::HASH_BYTES_PER_DELIVERY);
    assert_eq!(deliveries, slices);
    assert_eq!((r.outcome, r.batches), (Delivery::Done, 0));
    let j = job(&w, &o, &msg.job_id);
    assert_eq!(
        (
            j["state"].as_str(),
            j["failure"].as_str(),
            j["rows_done"].as_u64()
        ),
        (Some("failed"), Some("integrity"), Some(0))
    );
    assert_eq!(usage_vectors(&w, &o), 0);
}

#[test]
fn a_stale_job_record_can_never_overwrite_a_newer_one() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    collection(&w, &o, "docs", 8);
    let file = rvf(&rows(200, 8, 4), 8);
    let msg = submit(&w, &o, "docs", &upload(&w, &o, &file, None));
    let t = o.ctx.tenant_key();
    let mut stale = block_on(jobs::load(&w.b, t, &msg.job_id)).unwrap().unwrap();
    hash_pass(&w, &msg);
    assert_eq!(deliver(&w, &sink(&w), &msg, 1000).outcome, Delivery::Done);
    // The slower chain's save is refused; the finished job stays finished.
    stale.meta.error = Some("rolled back".into());
    let e = block_on(jobs::save(&w.b, t, &mut stale)).unwrap_err();
    assert!(jobs::is_job_changed(&e));
    let j = job(&w, &o, &msg.job_id);
    assert_eq!(
        (j["state"].as_str(), j["rows_done"].as_u64()),
        (Some("done"), Some(200))
    );
}

#[test]
fn a_failed_enqueue_never_leaves_a_job_queued() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    collection(&w, &o, "docs", 8);
    let up = upload(&w, &o, &rvf(&rows(50, 8, 5), 8), None);
    w.q.fail_ingest.set(true);
    let path = "/v1/collections/docs:import";
    let (s, _) = w.req(&o, Method::Post, path, json!({ "upload_id": up }));
    assert_eq!(s, 503);
    let t = o.ctx.tenant_key();
    let recs = block_on(kv_range(&w.b, t, Ns::Job, "0".into(), "g".into(), 10)).unwrap();
    assert_eq!(recs.len(), 1);
    let j = block_on(jobs::load(&w.b, t, &recs[0].0)).unwrap().unwrap();
    assert!(matches!(
        j.job.state,
        ruvector_edge_snapshot::JobState::Failed(_)
    ));
    assert_eq!(j.meta.error.as_deref(), Some("enqueue failed"));
    // The upload is still usable.
    w.q.fail_ingest.set(false);
    w.now_ms.set(w.now_ms.get() + 1000);
    assert_eq!(
        w.req(&o, Method::Post, path, json!({ "upload_id": up })).0,
        202
    );
}

fn ev(tenant: &str, ts_ms: u64, route: &str) -> AuditEvent {
    let mut e = AuditEvent::new(&caller("org-x", "sub", &[]).ctx, route, 200, ts_ms);
    e.tenant_key = tenant.into();
    e.rows = 1;
    e
}

/// Every line of `bodies` (in seq order) chains from the previous one.
fn verify(bodies: &[String]) -> (usize, [u8; 32]) {
    let (mut prev, mut n) = ([0u8; 32], 0);
    for (i, body) in bodies.iter().enumerate() {
        for line in body.lines() {
            let mut v: Json = serde_json::from_str(line).unwrap();
            let o = v.as_object_mut().unwrap();
            assert_eq!(o.remove("seq").unwrap(), json!(i + 1));
            let p = o.remove("prev").unwrap();
            let hash = o.remove("hash").unwrap();
            assert_eq!(p, crate::m3_wire::hex32(&prev).as_str(), "link");
            let plain = serde_json::to_string(&serde_json::from_value::<AuditEvent>(v).unwrap());
            let mut h = Sha256::new();
            h.update(prev);
            h.update(plain.unwrap().as_bytes());
            prev = h.finalize().into();
            assert_eq!(hash, crate::m3_wire::hex32(&prev).as_str());
            n += 1;
        }
    }
    (n, prev)
}

#[test]
fn shipped_audit_is_one_chain_per_tenant_and_redelivery_safe() {
    assert_eq!(ymd(0), (1970, 1, 1));
    assert_eq!(ymd(951_782_400_000), (2000, 2, 29));
    let w = World::default();
    let t1 = crate::testkit::tenant("org-a").as_str().to_string();
    let t2 = crate::testkit::tenant("org-b").as_str().to_string();
    let t0 = 1_790_035_200_000u64; // 2026-09-22T00:00:00Z
    let batch = |r: std::ops::Range<u64>| -> Vec<(String, AuditEvent)> {
        let mut v: Vec<_> = r
            .map(|i| {
                let t = if i % 2 == 0 { &t1 } else { &t2 };
                (format!("m{i}"), ev(t, t0 + i * 1000, "snapshot.create"))
            })
            .collect();
        v.push(("bad".into(), ev("../../etc", t0, "x")));
        v
    };
    let now = t0 + HOUR;
    assert_eq!(block_on(ship(&w.b, &w.blob, batch(0..10), now)).unwrap(), 2);
    // Redelivered with other messages: only the new ones are written.
    assert_eq!(block_on(ship(&w.b, &w.blob, batch(5..15), now)).unwrap(), 2);
    // An R2 outage: nothing confirmed; the retry writes the pending object.
    w.blob.fail_puts.set(true);
    assert!(block_on(ship(&w.b, &w.blob, batch(15..17), now)).is_err());
    w.blob.fail_puts.set(false);
    assert_eq!(
        block_on(ship(&w.b, &w.blob, batch(15..17), now)).unwrap(),
        2
    );
    assert_eq!(block_on(ship(&w.b, &w.blob, batch(0..17), now)).unwrap(), 0);
    let keys = w.blob.keys(&format!("audit/{t1}/"));
    let want: Vec<String> = (1..=3)
        .map(|s| format!("audit/{t1}/2026/09/22/{s:012}.ndjson"))
        .collect();
    assert_eq!(keys, want);
    let bodies: Vec<String> = keys
        .iter()
        .map(|k| String::from_utf8(w.blob.bytes(k).unwrap()).unwrap())
        .collect();
    let (lines, last) = verify(&bodies);
    assert_eq!(lines, 9, "m0,2,…,16 once each");
    let t = ruvector_edge_tenancy::TenantKey::parse(&t1).unwrap();
    assert_eq!(block_on(audit::head(&w.b, &t)).unwrap(), (4, last));
    assert_eq!(w.blob.keys(&format!("audit/{t2}/")).len(), 3);
    assert_eq!(w.blob.keys("audit/").len(), 6, "malformed tenant dropped");
}

#[test]
fn plain_queries_ship_nothing_writes_and_m3_work_ship_their_counts() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    let body = json!({ "name": "docs", "dim": 4, "metric": "cosine" });
    assert_eq!(w.req(&o, Method::Post, "/v1/collections", body).0, 201);
    let evs = w.q.take(QueueName::Audit);
    assert_eq!(evs.len(), 1);
    assert_eq!(evs[0]["route"], "collection.create");
    let q = json!({ "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1 });
    let path = "/v1/collections/docs/query";
    assert_eq!(w.req(&o, Method::Post, path, q).0, 200);
    assert!(w.q.take(QueueName::Audit).is_empty(), "M1 read hot path");
    let up = json!({ "vectors": [{ "id": "a", "values": [1.0, 0.0, 0.0, 0.0] }] });
    let path = "/v1/collections/docs/vectors";
    assert_eq!(w.req(&o, Method::Post, path, up).0, 200);
    let evs = w.q.take(QueueName::Audit);
    assert_eq!(evs.len(), 1);
    let e = &evs[0];
    assert_eq!(
        (e["route"].as_str(), e["rows"].as_u64()),
        (Some("vector.upsert"), Some(1))
    );
    assert!(!e.to_string().contains("values"), "never the payload");
    let (s, _) = w.req(
        &o,
        Method::Post,
        "/v1/collections/docs/snapshots",
        Json::Null,
    );
    assert_eq!(s, 202);
    let evs = w.q.take(QueueName::Audit);
    assert_eq!(evs.len(), 1);
    let e = &evs[0];
    assert_eq!(
        (e["route"].as_str(), e["rows"].as_u64()),
        (Some("snapshot.create"), Some(1))
    );
    assert_eq!(
        (e["scope"].as_str(), e["role"].as_str()),
        (Some("ruvector:write"), Some("owner"))
    );
    assert!(e["bytes"].as_u64().unwrap() > 0 && e["work_units"].as_u64().unwrap() > 0);
    // Export downloads are shipped by `serve`, after the stream's status.
    let (s, v) = w.req(&o, Method::Post, "/v1/collections/docs:export", Json::Null);
    assert_eq!(s, 201, "{v}");
    w.q.take(QueueName::Audit);
    let (s, _) = w.req(&o, Method::Get, v["download"].as_str().unwrap(), Json::Null);
    assert_eq!(s, 200);
    assert!(w.q.take(QueueName::Audit).is_empty());
}

#[test]
fn m1_writes_mutating_ops_and_mcp_tools_are_audited_reads_are_not() {
    let c = |r| rest_event(&r).map(|(n, _)| n);
    assert_eq!(c(ApiRoute::Delete("d".into())), Some("vector.delete"));
    assert_eq!(c(ApiRoute::Claim), Some("tenant.claim"));
    assert_eq!(c(ApiRoute::Query("d".into())), None);
    assert_eq!(c(ApiRoute::Fetch("d".into())), None);
    let call = |name: &str| {
        json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                    "params": { "name": name, "arguments": {} } })
    };
    let mcp = |v: Json| body_event(true, v.to_string().as_bytes());
    assert_eq!(
        mcp(call("vector_delete")),
        Some(("mcp.vector_delete".into(), Capability::Write))
    );
    assert_eq!(mcp(call("vector_query")), None);
    assert_eq!(
        mcp(json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" })),
        None
    );
    let op = |name: &str| body_event(false, json!({ "v": 1, "op": name }).to_string().as_bytes());
    assert_eq!(
        op("vector_upsert").map(|e| e.0),
        Some("ops.vector_upsert".into())
    );
    assert_eq!(op("vector_query"), None);
}

#[test]
fn a_text_upsert_m1_would_refuse_never_reaches_the_model() {
    let mut limits = crate::ledger_core::FREE_PLAN;
    limits.max_vectors = 1;
    let w = World::with_limits(limits);
    let o = w.owner("org-a", "alice");
    let body = json!({ "name": "notes", "metric": "cosine", "embedder": "bge-small-en-v1.5" });
    assert_eq!(w.req(&o, Method::Post, "/v1/collections", body).0, 201);
    let path = "/v1/collections/notes/vectors:upsert";
    let bad_id = json!({ "vectors": [{ "id": "x".repeat(1000), "text": "hello" }] });
    let (s, v) = w.req(&o, Method::Post, path, bad_id);
    assert_eq!(s, 400, "{v}");
    let two = json!({ "vectors": [{ "id": "a", "text": "one" }, { "id": "b", "text": "two" }] });
    let (s, v) = w.req(&o, Method::Post, path, two);
    assert!(is(&v, ErrorCode::QuotaExceeded), "{s} {v}");
    let meta = json!({ "vectors": [{ "id": "a", "text": "one", "metadata": [1, 2] }] });
    assert_eq!(w.req(&o, Method::Post, path, meta).0, 400);
    assert_eq!(w.ai.calls.get(), 0, "no model call for a refused body");
    let one = json!({ "vectors": [{ "id": "a", "text": "one" }] });
    assert_eq!(w.req(&o, Method::Post, path, one).0, 200);
    assert_eq!(w.ai.calls.get(), 1);
}

/// M2 §5.8 × M3: a queued import acts as its submitter, so denying the
/// submitter cancels the job at its next delivery, before any row lands.
#[test]
fn a_denied_submitter_s_queued_import_is_cancelled_before_any_row() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    collection(&w, &o, "docs", 8);
    let bob = caller("org-a", "bob", ALL);
    let t = o.ctx.tenant_key().clone();
    let editor = ruvector_edge_tenancy::Role::Editor;
    w.b.invite(
        &t,
        o.ctx.sub(),
        bob.ctx.sub(),
        editor,
        w.now_ms.get() / 1000,
    )
    .unwrap();
    let file = rvf(&rows(300, 8, 2), 8);
    let msg = submit(&w, &bob, "docs", &upload(&w, &bob, &file, None));
    hash_pass(&w, &msg);
    let deny = json!({ "kind": "sub", "value": bob.ctx.sub(), "ttl_s": 3600 });
    let (s, v) = w.req(&o, Method::Post, "/v1/tenant/deny", deny);
    assert_eq!(s, 201, "{v}");
    let r = deliver(&w, &sink(&w), &msg, 1000);
    assert_eq!((r.outcome, r.batches), (Delivery::Done, 0));
    let j = job(&w, &o, &msg.job_id);
    assert_eq!(
        (
            j["state"].as_str(),
            j["failure"].as_str(),
            j["error"].as_str()
        ),
        (Some("failed"), Some("cancelled"), Some("submitter denied"))
    );
    assert_eq!(usage_vectors(&w, &o), 0);
}
