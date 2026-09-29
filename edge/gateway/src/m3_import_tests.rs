//! M3 acceptance (ADR-351 §15) for rv-ingest: upload session → staging →
//! queue → consumer into real shards; foreign `upload_id` refused; a
//! 100k × 384 import streams within quota and memory; resume after a crash
//! without double-applying; the 24 h `op_id` window; integrity.

use crate::backend::mem::MemBackend;
use crate::ingest::{batch_rows, process, BatchSink, Delivery, Report, UpsertSink, TAIL_BYTES};
use crate::jobs::{IngestMsg, Job};
use crate::m3_mem::*;
use crate::m3_ports::QueueName;
use crate::rest::Caller;
use crate::testkit::{block_on, Rng};
use crate::uploads::PART_BYTES;
use ruvector_edge_snapshot::{Metric, Row, RvfExporter};
use ruvector_edge_store::{ErrorCode, OpError};
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use worker::Method;

pub(crate) fn rows(n: usize, dim: usize, seed: u64) -> Vec<Row> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|i| Row {
            id: format!("r{i:06}"),
            values: rng.vec(dim),
            metadata: (i % 3 == 0).then(|| format!("{{\"i\":{i}}}")),
        })
        .collect()
}

pub(crate) fn rvf(rows: &[Row], dim: u16) -> Vec<u8> {
    let mut out = Vec::new();
    let mut exp = RvfExporter::new(dim, Metric::Cosine, 1024).unwrap();
    for r in rows {
        exp.push(r, &mut |b: &[u8]| out.extend_from_slice(b))
            .unwrap();
    }
    exp.finish(&mut |b: &[u8]| out.extend_from_slice(b));
    out
}

pub(crate) fn collection(w: &World, c: &Caller, name: &str, dim: usize) {
    let body = json!({ "name": name, "dim": dim, "metric": "cosine", "shards": 3 });
    assert_eq!(w.req(c, Method::Post, "/v1/collections", body).0, 201);
}

pub(crate) fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Upload session: create, parts, complete.
pub(crate) fn upload(w: &World, c: &Caller, file: &[u8], declared_sha: Option<String>) -> String {
    let sha = declared_sha.unwrap_or_else(|| hex(&Sha256::digest(file)));
    let (s, v) = w.req(
        c,
        Method::Post,
        "/v1/uploads",
        json!({ "size": file.len(), "sha256": sha }),
    );
    assert_eq!(s, 201, "{v}");
    let id = v["upload_id"].as_str().unwrap().to_string();
    let parts = v["parts"].as_u64().unwrap();
    assert_eq!(v["part_size"], PART_BYTES);
    for n in 1..=parts {
        let lo = ((n - 1) * PART_BYTES) as usize;
        let hi = (n * PART_BYTES).min(file.len() as u64) as usize;
        let path = format!("/v1/uploads/{id}/parts/{n}");
        let (s, v) = w.raw(c, Method::Post, &path, file[lo..hi].to_vec(), true, None);
        assert_eq!(s, 200, "{v}");
    }
    let (s, v) = w.req(
        c,
        Method::Post,
        &format!("/v1/uploads/{id}:complete"),
        Json::Null,
    );
    assert_eq!(s, 200, "{v}");
    id
}

pub(crate) fn submit(w: &World, c: &Caller, coll: &str, upload_id: &str) -> IngestMsg {
    let path = format!("/v1/collections/{coll}:import");
    let (s, v) = w.req(c, Method::Post, &path, json!({ "upload_id": upload_id }));
    assert_eq!(s, 202, "{v}");
    let mut msgs = w.q.take(QueueName::Ingest);
    assert_eq!(msgs.len(), 1);
    let msg: IngestMsg = serde_json::from_value(msgs.remove(0)).unwrap();
    assert_eq!(msg.job_id, v["job_id"].as_str().unwrap());
    msg
}

pub(crate) fn deliver<S: BatchSink>(w: &World, sink: &S, msg: &IngestMsg, budget: u32) -> Report {
    block_on(process(&w.b, &w.blob, sink, msg, w.now_ms.get(), budget)).unwrap()
}

/// The first deliveries of a multipart upload are the sha256 pass alone,
/// one `HASH_BYTES_PER_DELIVERY` slice each, its state persisted between.
pub(crate) fn hash_pass(w: &World, msg: &IngestMsg) {
    let t = ruvector_edge_tenancy::TenantKey::parse(&msg.tenant_key).unwrap();
    loop {
        let r = deliver(w, &sink(w), msg, 1000);
        assert_eq!((r.outcome, r.batches), (Delivery::Continue, 0), "hash pass");
        let j = block_on(crate::jobs::load(&w.b, &t, &msg.job_id))
            .unwrap()
            .unwrap();
        if j.meta.sha_verified {
            assert_eq!(j.meta.hash_pass, None);
            return;
        }
        assert!(j.meta.hash_pass.is_some(), "pass state persisted");
    }
}

pub(crate) fn sink(w: &World) -> UpsertSink<'_, MemBackend> {
    UpsertSink {
        b: &w.b,
        now: w.now_ms.get() / 1000,
    }
}

pub(crate) fn job(w: &World, c: &Caller, id: &str) -> Json {
    let (s, v) = w.req(c, Method::Get, &format!("/v1/jobs/{id}"), Json::Null);
    assert_eq!(s, 200, "{v}");
    v
}

pub(crate) fn usage_vectors(w: &World, c: &Caller) -> u64 {
    w.req(c, Method::Get, "/v1/usage", Json::Null).1["usage"]["vectors"]
        .as_u64()
        .unwrap()
}

/// Every source row is stored bit-exactly with its metadata.
fn assert_stored(w: &World, c: &Caller, coll: &str, src: &[Row]) {
    for part in src.chunks(100) {
        let ids: Vec<&str> = part.iter().map(|r| r.id.as_str()).collect();
        let body = json!({ "ids": ids, "include_values": true });
        let path = format!("/v1/collections/{coll}/vectors:fetch");
        let (s, v) = w.req(c, Method::Post, &path, body);
        assert_eq!(s, 200);
        let got = v["vectors"].as_array().unwrap();
        assert_eq!(got.len(), part.len());
        for (g, r) in got.iter().zip(part) {
            let vals: Vec<f32> = serde_json::from_value(g["values"].clone()).unwrap();
            assert!(vals
                .iter()
                .zip(&r.values)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            let meta: Option<Json> = r
                .metadata
                .as_deref()
                .map(|m| serde_json::from_str(m).unwrap());
            assert_eq!(g["metadata"], meta.unwrap_or(Json::Null));
        }
    }
}

#[test]
fn upload_session_import_lands_in_shards() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    collection(&w, &o, "docs", 8);
    let src = rows(3000, 8, 1);
    let file = rvf(&src, 8);
    let up = upload(&w, &o, &file, None);
    let msg = submit(&w, &o, "docs", &up);
    assert_eq!(job(&w, &o, &msg.job_id)["state"], "queued");
    hash_pass(&w, &msg);
    let r = deliver(&w, &sink(&w), &msg, 1000);
    assert_eq!(r.outcome, Delivery::Done);
    let j = job(&w, &o, &msg.job_id);
    assert_eq!(
        (j["state"].as_str(), j["rows_done"].as_u64()),
        (Some("done"), Some(3000))
    );
    assert_stored(&w, &o, "docs", &src);
    assert_eq!(usage_vectors(&w, &o), 3000);
    // The upload feeds one job only; a finished job's redelivery is a no-op.
    let path = "/v1/collections/docs:import";
    assert_eq!(
        w.req(&o, Method::Post, path, json!({ "upload_id": up })).0,
        409
    );
    assert_eq!(deliver(&w, &sink(&w), &msg, 1000).batches, 0);
    // Inline (≤ 512 KiB, octet-stream) import.
    collection(&w, &o, "inline", 8);
    let small = rows(700, 8, 2);
    let (s, v) = w.raw(
        &o,
        Method::Post,
        "/v1/collections/inline:import",
        rvf(&small, 8),
        true,
        None,
    );
    assert_eq!(s, 202, "{v}");
    let msg: IngestMsg = serde_json::from_value(w.q.take(QueueName::Ingest).remove(0)).unwrap();
    assert_eq!(deliver(&w, &sink(&w), &msg, 1000).outcome, Delivery::Done);
    assert_stored(&w, &o, "inline", &small);
}

#[test]
fn foreign_upload_and_job_ids_are_refused() {
    let w = World::default();
    let a = w.owner("org-a", "alice");
    let b = w.owner("org-b", "bob");
    collection(&w, &a, "docs", 8);
    collection(&w, &b, "docs", 8);
    let file = rvf(&rows(50, 8, 3), 8);
    let up = upload(&w, &a, &file, None);
    let (s, _) = w.req(
        &b,
        Method::Post,
        "/v1/collections/docs:import",
        json!({ "upload_id": up }),
    );
    assert_eq!(s, 404, "B cannot import A's upload");
    let (s, _) = w.raw(
        &b,
        Method::Post,
        &format!("/v1/uploads/{up}/parts/1"),
        file.clone(),
        true,
        None,
    );
    assert_eq!(s, 404);
    let msg = submit(&w, &a, "docs", &up);
    assert_eq!(
        w.req(
            &b,
            Method::Get,
            &format!("/v1/jobs/{}", msg.job_id),
            Json::Null
        )
        .0,
        404
    );
    // A message naming B's tenant with A's job id finds nothing to run.
    let forged = IngestMsg {
        tenant_key: b.ctx.tenant_key().as_str().to_string(),
        job_id: msg.job_id.clone(),
    };
    assert_eq!(deliver(&w, &sink(&w), &forged, 1000).batches, 0);
    assert_eq!(usage_vectors(&w, &b), 0);
    // A viewer cannot start imports.
    let viewer = caller("org-a", "alice", &[ruvector_edge_auth::Capability::Read]);
    let (s, _) = w.req(
        &viewer,
        Method::Post,
        "/v1/uploads",
        json!({ "size": 10, "sha256": hex(&[0; 32]) }),
    );
    assert_eq!(s, 403);
}

/// Counts rows, checks `op_id` uniqueness and batch size.
#[derive(Default)]
struct Meter {
    rows: Cell<u64>,
    ops: RefCell<BTreeSet<String>>,
    max_batch: Cell<usize>,
}

impl BatchSink for Meter {
    async fn apply(&self, _j: &Job, op_id: &str, rows: Vec<Row>) -> Result<(), OpError> {
        assert!(
            self.ops.borrow_mut().insert(op_id.to_string()),
            "op_id reused"
        );
        assert_eq!(op_id.len(), 26);
        assert!(rows.iter().all(|r| r.values.len() == 384));
        self.max_batch.set(self.max_batch.get().max(rows.len()));
        self.rows.set(self.rows.get() + rows.len() as u64);
        Ok(())
    }
}

#[test]
fn import_100k_x_384_streams_within_quota_and_memory() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    collection(&w, &o, "big", 384);
    let mut file = Vec::new();
    let mut exp = RvfExporter::new(384, Metric::Cosine, 1024).unwrap();
    let mut rng = Rng(5);
    for i in 0..100_000 {
        let r = Row {
            id: format!("r{i:06}"),
            values: rng.vec(384),
            metadata: None,
        };
        exp.push(&r, &mut |b: &[u8]| file.extend_from_slice(b))
            .unwrap();
    }
    exp.finish(&mut |b: &[u8]| file.extend_from_slice(b));
    assert!(file.len() > 140 << 20, "~147 MiB of vectors");
    let up = upload(&w, &o, &file, None);
    drop(file);
    let msg = submit(&w, &o, "big", &up);
    hash_pass(&w, &msg);
    let meter = Meter::default();
    let (mut peak, mut deliveries) = (0usize, 0);
    loop {
        let r = deliver(&w, &meter, &msg, 50);
        peak = peak.max(r.peak_buffered);
        deliveries += 1;
        match r.outcome {
            Delivery::Continue => continue,
            Delivery::Done => break,
            Delivery::Retry => panic!("unexpected retry"),
        }
    }
    assert_eq!(meter.rows.get(), 100_000);
    // Batches are `batch_rows(384)` = 64 rows (the per-delivery CPU bound)
    // and never span 1024-row records: 97 × 16 + (10 × 64 + 32).
    let per = batch_rows(384);
    assert_eq!(per, 64);
    let ops = 97 * 1024_usize.div_ceil(per) + 672_usize.div_ceil(per);
    assert_eq!(meter.ops.borrow().len(), ops);
    assert_eq!(meter.max_batch.get(), per);
    assert_eq!(
        deliveries,
        ops.div_ceil(50),
        "re-enqueued every 50 batches (after the hash pass)"
    );
    // One record (≈ 1.6 MB at 1024 × 384) plus one 1 MiB piece, never the file.
    assert!(peak < 6 << 20, "peak importer buffer {peak}");
    assert!(w.blob.max_range.get() <= TAIL_BYTES);
    let j = job(&w, &o, &msg.job_id);
    assert_eq!(
        (j["state"].as_str(), j["rows_done"].as_u64()),
        (Some("done"), Some(100_000))
    );
    // The job pinned the tenant's remaining quota (FREE_PLAN: 250k vectors).
    let stored = crate::jobs::load(&w.b, o.ctx.tenant_key(), &msg.job_id);
    let binding = block_on(stored).unwrap().unwrap().job.binding.unwrap();
    assert_eq!(binding.max_rows, 250_000);
    assert_eq!(binding.max_batch_rows, 64);
}

#[test]
fn import_over_the_remaining_quota_fails_before_any_batch() {
    let mut limits = crate::ledger_core::FREE_PLAN;
    limits.max_vectors = 1000;
    let w = World::with_limits(limits);
    let o = w.owner("org-a", "alice");
    collection(&w, &o, "docs", 8);
    let up = upload(&w, &o, &rvf(&rows(1500, 8, 6), 8), None);
    let msg = submit(&w, &o, "docs", &up);
    hash_pass(&w, &msg);
    let r = deliver(&w, &sink(&w), &msg, 1000);
    assert_eq!((r.outcome, r.batches), (Delivery::Done, 0));
    let j = job(&w, &o, &msg.job_id);
    assert_eq!(
        (j["state"].as_str(), j["failure"].as_str()),
        (Some("failed"), Some("quota_exceeded"))
    );
    assert_eq!(usage_vectors(&w, &o), 0);
}

/// Applies batch `at` for real, then reports a crash (transport failure)
/// before the job can record it.
struct Crash<'a> {
    inner: UpsertSink<'a, MemBackend>,
    at: Cell<Option<u64>>,
    seen: Cell<u64>,
}

impl BatchSink for Crash<'_> {
    async fn apply(&self, j: &Job, op_id: &str, rows: Vec<Row>) -> Result<(), OpError> {
        self.inner.apply(j, op_id, rows).await?;
        self.seen.set(self.seen.get() + 1);
        if self.at.get() == Some(self.seen.get()) {
            self.at.set(None);
            return Err(OpError::new(ErrorCode::ShardUnavailable, "crash"));
        }
        Ok(())
    }
}

#[test]
fn import_resumes_after_a_crash_without_double_applying() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    collection(&w, &o, "docs", 8);
    let src = rows(3000, 8, 7);
    let msg = submit(&w, &o, "docs", &upload(&w, &o, &rvf(&src, 8), None));
    hash_pass(&w, &msg);
    let crash = Crash {
        inner: sink(&w),
        at: Cell::new(Some(4)),
        seen: Cell::new(0),
    };
    let r = deliver(&w, &crash, &msg, 1000);
    assert_eq!(
        (r.outcome, r.batches),
        (Delivery::Retry, 3),
        "batch 4 applied, not recorded"
    );
    assert_eq!(job(&w, &o, &msg.job_id)["batches"], 3);
    w.b.restart();
    // Redelivery: batch 4 is deduplicated by its op_id, then continues in
    // budget-bounded deliveries.
    let mut outcomes = Vec::new();
    loop {
        let r = deliver(&w, &crash, &msg, 1);
        outcomes.push(r.outcome);
        if r.outcome == Delivery::Done {
            break;
        }
    }
    // 3000 rows = records of 1024/1024/952 → batches 500/500/24 ×2 + 500/452.
    assert_eq!(
        outcomes
            .iter()
            .filter(|d| **d == Delivery::Continue)
            .count(),
        5
    );
    let j = job(&w, &o, &msg.job_id);
    assert_eq!(
        (j["state"].as_str(), j["rows_done"].as_u64()),
        (Some("done"), Some(3000))
    );
    assert_stored(&w, &o, "docs", &src);
    assert_eq!(usage_vectors(&w, &o), 3000, "no batch counted twice");
}

#[test]
fn stale_or_tampered_jobs_fail_with_stable_codes() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    collection(&w, &o, "docs", 8);
    let file = rvf(&rows(400, 8, 8), 8);
    let stale = submit(&w, &o, "docs", &upload(&w, &o, &file, None));
    w.now_ms.set(w.now_ms.get() + 25 * 3600 * 1000);
    let r = deliver(&w, &sink(&w), &stale, 1000);
    assert_eq!((r.outcome, r.batches), (Delivery::Done, 0));
    let j = job(&w, &o, &stale.job_id);
    assert_eq!(
        (j["state"].as_str(), j["failure"].as_str()),
        (Some("failed"), Some("cancelled"))
    );
    assert!(j["error"].as_str().unwrap().contains("24 h"));
    // A declared sha256 that the streamed bytes do not match.
    let lie = hex(&[7u8; 32]);
    let bad = submit(&w, &o, "docs", &upload(&w, &o, &file, Some(lie)));
    assert_eq!(deliver(&w, &sink(&w), &bad, 1000).outcome, Delivery::Done);
    let j = job(&w, &o, &bad.job_id);
    assert_eq!(
        (j["state"].as_str(), j["failure"].as_str()),
        (Some("failed"), Some("integrity"))
    );
    // A dimension mismatch is refused before any batch.
    collection(&w, &o, "wide", 16);
    let m = submit(&w, &o, "wide", &upload(&w, &o, &file, None));
    hash_pass(&w, &m);
    assert_eq!(deliver(&w, &sink(&w), &m, 1000).batches, 0);
    assert_eq!(job(&w, &o, &m.job_id)["failure"], "incompatible");
}
