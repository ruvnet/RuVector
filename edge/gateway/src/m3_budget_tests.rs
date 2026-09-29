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

mod ingest {
    use super::*;
    use crate::backend::mem::MemBackend;
    use crate::ingest::{BatchSink, Delivery, MAX_STALLED_DELIVERIES, STALLED};
    use crate::jobs::Job;
    use crate::m3_import_tests::{
        collection, deliver, hash_pass, job, rows, rvf, sink, submit, upload,
    };
    use ruvector_edge_snapshot::{Metric, Row, RvfExporter};
    use ruvector_edge_store::{ErrorCode, OpError};

    /// A sink whose every batch fails transiently (a killed or flaky
    /// delivery that never commits).
    struct Down;
    impl BatchSink for Down {
        async fn apply(&self, _: &Job, _: &str, _: Vec<Row>) -> Result<(), OpError> {
            Err(OpError::new(ErrorCode::ShardUnavailable, "down"))
        }
    }

    #[test]
    fn a_segment_over_1_mib_fails_the_queued_import_as_incompatible() {
        let w = World::default();
        let o = w.owner("org-a", "alice");
        collection(&w, &o, "big", 384);
        // 1024 × 384 f32 = 1.5 MiB VEC payload.
        let mut file = Vec::new();
        let mut exp = RvfExporter::new(384, Metric::Cosine, 1024).unwrap();
        for r in rows(1024, 384, 3) {
            exp.push(&r, &mut |b: &[u8]| file.extend_from_slice(b))
                .unwrap();
        }
        exp.finish(&mut |b: &[u8]| file.extend_from_slice(b));
        let msg = submit(&w, &o, "big", &upload(&w, &o, &file, None));
        hash_pass(&w, &msg);
        let r = deliver(&w, &sink(&w), &msg, 1000);
        assert_eq!((r.outcome, r.batches), (Delivery::Done, 0));
        let j = job(&w, &o, &msg.job_id);
        assert_eq!(
            (j["state"].as_str(), j["failure"].as_str()),
            (Some("failed"), Some("incompatible"))
        );
        // The gateway's own segment size is importable at any dim.
        for dim in [8u16, 384, 768, 1536, 4096] {
            let seg = crate::export::rows_per_segment(dim) as usize;
            assert!(seg >= 1 && seg * usize::from(dim) * 4 <= 1 << 20, "{dim}");
            assert!(seg * crate::export::SIDECAR_ROW_MAX <= 1 << 20, "{dim}");
        }
    }

    #[test]
    fn queued_batches_after_the_first_pay_a_write_token() {
        let w = World {
            b: MemBackend::new().with_fanout_limiter(),
            ..World::default()
        };
        let o = w.owner("org-a", "alice");
        collection(&w, &o, "docs", 8);
        // 6000 rows at 8 dims: 500-row batches, 3 per 1024-row record.
        let msg = submit(
            &w,
            &o,
            "docs",
            &upload(&w, &o, &rvf(&rows(6000, 8, 4), 8), None),
        );
        hash_pass(&w, &msg);
        let limiter = w.b.fanout_limiter.as_ref().unwrap();
        limiter.reset();
        // Batch 0 rides on the `:import` token, batches 1-10 pay the user
        // write budget (10 / 10 s), batch 11 is refused: retry later.
        let r = deliver(&w, &sink(&w), &msg, 1000);
        assert_eq!((r.outcome, r.batches), (Delivery::Retry, 11));
        let writes: u32 = limiter
            .seen
            .borrow()
            .iter()
            .filter(|((b, _), _)| b == "RL_WRITE_USER")
            .map(|(_, n)| *n)
            .sum();
        assert_eq!(writes, 11, "10 admitted + 1 refused");
        limiter.reset();
        let r = deliver(&w, &sink(&w), &msg, 1000);
        assert_eq!(r.outcome, Delivery::Done);
        assert_eq!(job(&w, &o, &msg.job_id)["rows_done"], 6000);
    }

    #[test]
    fn a_job_whose_deliveries_never_progress_fails_and_frees_its_upload() {
        let w = World::default();
        let o = w.owner("org-a", "alice");
        collection(&w, &o, "docs", 8);
        let up = upload(&w, &o, &rvf(&rows(300, 8, 5), 8), None);
        let msg = submit(&w, &o, "docs", &up);
        hash_pass(&w, &msg);
        for _ in 0..MAX_STALLED_DELIVERIES {
            let r = deliver(&w, &Down, &msg, 1000);
            assert_eq!((r.outcome, r.batches), (Delivery::Retry, 0));
        }
        // Before the queue would dead-letter the message: failed, not running.
        let r = deliver(&w, &Down, &msg, 1000);
        assert_eq!(r.outcome, Delivery::Done);
        let ev = r.audit.unwrap();
        assert_eq!((ev.route.as_str(), ev.outcome), ("job.cancelled", 410));
        let j = job(&w, &o, &msg.job_id);
        assert_eq!(
            (j["state"].as_str(), j["error"].as_str()),
            (Some("failed"), Some(STALLED))
        );
        // The upload is `complete` again: the client resubmits it.
        w.now_ms.set(w.now_ms.get() + 1000); // job ids are minted per ms
        let again = submit(&w, &o, "docs", &up);
        hash_pass(&w, &again);
        assert_eq!(deliver(&w, &sink(&w), &again, 1000).outcome, Delivery::Done);
        assert_eq!(job(&w, &o, &again.job_id)["rows_done"], 300);
    }

    #[test]
    fn a_dropped_collection_cancels_the_job_before_its_hash_pass() {
        let w = World::default();
        let o = w.owner("org-a", "alice");
        collection(&w, &o, "docs", 8);
        let msg = submit(
            &w,
            &o,
            "docs",
            &upload(&w, &o, &rvf(&rows(300, 8, 6), 8), None),
        );
        assert!(
            w.req(&o, Method::Delete, "/v1/collections/docs", Json::Null)
                .0
                < 300
        );
        let r = deliver(&w, &sink(&w), &msg, 1000);
        assert_eq!((r.outcome, r.batches), (Delivery::Done, 0));
        let j = job(&w, &o, &msg.job_id);
        assert_eq!(j["failure"], "incompatible", "{j}");
        let t = o.ctx.tenant_key();
        let stored = crate::testkit::block_on(crate::jobs::load(&w.b, t, &msg.job_id))
            .unwrap()
            .unwrap();
        assert!(stored.meta.hash_pass.is_none() && !stored.meta.sha_verified);
    }
}

#[test]
fn only_a_top_level_or_row_text_field_selects_the_embedding_path() {
    use crate::m3_api::has_text_field;
    let yes = |v: Json| has_text_field(v.to_string().as_bytes());
    assert!(yes(json!({ "text": "q", "top_k": 3 })));
    assert!(yes(json!({ "text": null })));
    assert!(yes(
        json!({ "vectors": [{ "id": "a", "values": [1.0] }, { "id": "b", "text": "t" }] })
    ));
    // RAG shape: `text` only inside metadata.
    assert!(!yes(
        json!({ "vectors": [{ "id": "a", "values": [1.0], "metadata": { "text": "t" } }] })
    ));
    assert!(!yes(
        json!({ "vector": [1.0], "filter": { "text": { "$eq": "x" } } })
    ));
    assert!(!has_text_field(b"not json \"text\""));
    // The plain M1 upsert with metadata text still stores via the M1 path.
    let w = World::default();
    let o = w.owner("org-a", "alice");
    let body = json!({ "name": "docs", "dim": 2, "metric": "cosine" });
    assert_eq!(w.req(&o, Method::Post, "/v1/collections", body).0, 201);
    let rows =
        json!({ "vectors": [{ "id": "a", "values": [1.0, 0.0], "metadata": { "text": "t" } }] });
    let (s, v) = w.req(
        &o,
        Method::Post,
        "/v1/collections/docs/vectors:upsert",
        rows,
    );
    assert_eq!(s, 200, "{v}");
}
