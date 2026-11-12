//! Post-merge review fixes (ADR-351 M3; caps now at their Workers Paid
//! values): body caps per route, which requests ship audit messages, the synchronous
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
fn upload_parts_and_inline_imports_get_8_mib_and_nothing_else_does() {
    let cap = |m: Method, p: &str| parse(&m, p).and_then(|r| r.body_cap());
    assert_eq!(cap(Method::Post, "/v1/uploads/u1/parts/1"), Some(8 << 20));
    assert_eq!(MAX_PART_BODY, 8 << 20);
    assert_eq!(cap(Method::Post, "/v1/collections/c:import"), Some(8 << 20));
    // Workers Paid: the inline cap is the part cap (Free: 512 KiB).
    assert_eq!(MAX_INLINE_BYTES, 8 << 20);
    assert_eq!(MAX_INLINE_BYTES, MAX_PART_BODY);
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
fn an_inline_import_over_8_mib_is_refused_before_hashing_or_storing() {
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

/// Workers Paid boundary: a body of exactly [`MAX_INLINE_BYTES`] is hashed
/// and staged in the request (content is checked by the queued job).
#[test]
fn an_inline_import_at_8_mib_is_staged_and_queued() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    let body = json!({ "name": "docs", "dim": 8, "metric": "cosine" });
    assert_eq!(w.req(&o, Method::Post, "/v1/collections", body).0, 201);
    let path = "/v1/collections/docs:import";
    let (s, v) = w.raw(
        &o,
        Method::Post,
        path,
        vec![7u8; MAX_INLINE_BYTES],
        true,
        None,
    );
    assert_eq!(s, 202, "{v}");
    assert_eq!(w.blob.keys("staging/").len(), 1);
    assert_eq!(w.q.take(QueueName::Ingest).len(), 1);
}

/// Workers Paid: export's own budget is 2^24 stored floats (43,690 rows at
/// 384 dims; Free: 2^18), snapshot / restore stay at 2^18 (memory-bound
/// restore commit). The meter admits exactly the cap and refuses one more.
#[test]
fn export_budget_is_2_pow_24_floats_and_refuses_one_row_over() {
    use crate::sync_budget::{export_max_rows, max_rows, Meter, EXPORT_MAX_FLOATS};
    assert_eq!(EXPORT_MAX_FLOATS, 1 << 24);
    assert_eq!((export_max_rows(384), max_rows(384)), (43_690, 682));
    assert_eq!(export_max_rows(1536), 10_922);
    let cap = export_max_rows(384);
    let mut m = Meter::within(cap);
    for _ in 0..cap / 512 {
        m.add(512).unwrap();
    }
    m.add((cap % 512) as usize).unwrap();
    let e = m.add(1).unwrap_err();
    assert_eq!(e.code.status(), 413);
}

/// Review finding (Workers Paid): the export float cap alone let a low-dim
/// collection push ≈ 4 KiB of metadata per row through the pass. The
/// export meter also counts stored bytes (values + id + metadata) against
/// 128 MiB: at 16 dims with 4 KiB metadata it refuses near 32k rows, where
/// the float cap would admit 1,048,576.
#[test]
fn export_byte_budget_refuses_metadata_heavy_low_dim_exports() {
    use crate::export::stored_bytes;
    use crate::m3_wire::RowWire;
    use crate::sync_budget::{export_max_rows, Meter, EXPORT_MAX_BYTES};
    assert_eq!(EXPORT_MAX_BYTES, 128 << 20);
    let dim = 16u16;
    assert_eq!(export_max_rows(16), 1 << 20);
    let meta = format!("{{\"k\":\"{}\"}}", "x".repeat(4088));
    assert_eq!(meta.len(), 4096);
    let page: Vec<RowWire> = (0..512)
        .map(|i| RowWire::from_parts(format!("v{i:07}"), &[0u8; 64], Some(meta.clone())))
        .collect();
    let per_page = stored_bytes(&page, dim);
    assert_eq!(per_page, 512 * (64 + 8 + 4096));
    let mut m = Meter::within(export_max_rows(16)).with_bytes(EXPORT_MAX_BYTES);
    let mut rows = 0u64;
    let e = loop {
        let r = m.add(page.len()).and_then(|()| m.add_bytes(per_page));
        match r {
            Ok(()) => rows += page.len() as u64,
            Err(e) => break e,
        }
    };
    assert_eq!(e.code.status(), 413);
    assert_eq!(rows, EXPORT_MAX_BYTES / per_page * 512);
    assert!((31_000..33_000).contains(&rows), "{rows}");
    // A full 384-d export (43,690 rows) with 256-byte ids stays in budget.
    const _: () = assert!(43_690 * (384 * 4 + 256) <= EXPORT_MAX_BYTES);
    // Without metadata the float cap binds first, as before.
    let bare: Vec<RowWire> = (0..512)
        .map(|i| RowWire::from_parts(format!("v{i:07}"), &[0u8; 64], None))
        .collect();
    let mut m = Meter::within(export_max_rows(16)).with_bytes(EXPORT_MAX_BYTES);
    for _ in 0..(1 << 20) / 512 {
        m.add(512).unwrap();
        m.add_bytes(stored_bytes(&bare, dim)).unwrap();
    }
    assert_eq!(m.add(1).unwrap_err().code.status(), 413);
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

    /// Workers Paid: the queued import takes segments up to the format's
    /// 8 MiB (Free: 1 MiB), so the `rvf` CLI default (`--batch-size 1000`,
    /// 1.5 MiB at 384 dims) and a segment at the cap both import. A larger
    /// segment is `SegmentTooLarge` → `incompatible` (the crate's
    /// `rvf_roundtrip` covers that refusal; no writer can produce one).
    #[test]
    fn segments_up_to_8_mib_import_including_the_rvf_cli_default() {
        use ruvector_edge_snapshot::rvf_format::MAX_SEGMENT_PAYLOAD;
        assert_eq!(
            crate::ingest::QUEUED_MAX_SEGMENT_PAYLOAD,
            MAX_SEGMENT_PAYLOAD as u64
        );
        // 1024 rows (1.5 MiB), then 5,433 rows = 8 MiB − 50 B of VEC
        // payload (6 + 5,433 × 1,544): the exporter's largest segment.
        for (name, n) in [("cli", 1024usize), ("cap", 5_433)] {
            let w = World::default();
            let o = w.owner("org-a", "alice");
            collection(&w, &o, name, 384);
            let mut file = Vec::new();
            let mut exp = RvfExporter::new(384, Metric::Cosine, 65_536).unwrap();
            for r in rows(n, 384, 3) {
                exp.push(&r, &mut |b: &[u8]| file.extend_from_slice(b))
                    .unwrap();
            }
            let sum = exp.finish(&mut |b: &[u8]| file.extend_from_slice(b));
            // One VEC + sidecar pair and the manifest: a single record.
            assert_eq!(sum.segments, 3, "{name}: one record");
            let msg = submit(&w, &o, name, &upload(&w, &o, &file, None));
            hash_pass(&w, &msg);
            let r = deliver(&w, &sink(&w), &msg, 1000);
            assert_eq!(r.outcome, Delivery::Done, "{name}");
            let j = job(&w, &o, &msg.job_id);
            assert_eq!(
                (j["state"].as_str(), j["rows_done"].as_u64()),
                (Some("done"), Some(n as u64)),
                "{name}: {j}"
            );
        }
        // The gateway's own exports keep 1 MiB segments (importable at any dim).
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
        // 3000 rows at 8 dims: 218-row batches (the 1 MiB encoded-body
        // bound), 5 per 1024-row record, 15 in all.
        let msg = submit(
            &w,
            &o,
            "docs",
            &upload(&w, &o, &rvf(&rows(3000, 8, 4), 8), None),
        );
        hash_pass(&w, &msg);
        let limiter = w.b.fanout_limiter.as_ref().unwrap();
        limiter.reset();
        // Batch 0 rides on the `:import` token, batches 1-10 pay the user
        // write budget (10 / 10 s), batch 11 is refused: retry later.
        assert_eq!(
            crate::ingest::batch_rows(8, ruvector_edge_store::IndexConfig::Flat),
            218
        );
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
        assert_eq!(job(&w, &o, &msg.job_id)["rows_done"], 3000);
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
