//! `ruvector-edge-ingest` consumer core (ADR-351 §3 rv-ingest): one
//! delivery of one import job.
//!
//! A delivery loads the job from its tenant's ledger and fails it once it
//! is older than the 24 h `op_id` window **since submission** (every batch
//! it applied was recorded under an `op_id` whose idempotency slot lives
//! 24 h from that batch, so inside the window a replayed batch is always
//! deduplicated; past it, it could not be), or once
//! [`MAX_STALLED_DELIVERIES`] deliveries in a row committed nothing (its
//! upload is released for a resubmit). It re-checks the submitter against
//! the tenant deny list (a revoked credential's queued job is cancelled)
//! and resolves the collection by name **and** uid — both before the hash
//! pass, so neither spends it. It checks the staging object (`head`); an
//! upload R2 holds no sha256 for (multipart) gets a hash pass against the
//! declared sha256 before any batch is applied, split into deliveries of
//! its own of at most `HASH_BYTES_PER_DELIVERY` each with the hasher state
//! persisted in the job (`ingest_hash`). It then pins the tenant's
//! remaining row quota, range-reads the tail (`inspect_tail`; after
//! the first delivery only from the recorded manifest offset), `deliver`s
//! the job and streams the object from the cursor in 1 MiB ranges through
//! `RvfImporter`. Each batch goes to a [`BatchSink`] under
//! `job.op_id(seq)` and the cursor is persisted after every batch, so a
//! crash at any point resumes without loss and a batch applied but not
//! recorded is replayed by the sink's `op_id` idempotency, not re-applied.
//!
//! **Per delivery** (Workers Paid, `cpu_ms = 30000`): batches are sized to
//! at most [`FLOATS_PER_BATCH`] floats and at most 1 MiB of worst-case
//! encoded JSON ([`batch_rows`], ≤ 500 rows) so the `VectorShard` path
//! never parses a larger body than a client upsert, and after
//! [`BATCHES_PER_DELIVERY`] batches the delivery stops with
//! [`Delivery::Continue`] (the consumer re-enqueues and acks). Batches
//! never span a record, so a delivery re-reads and re-verifies the record
//! it resumes in: segments are capped at [`QUEUED_MAX_SEGMENT_PAYLOAD`], a
//! memory bound (the segment is buffered whole). Every batch after the
//! first pays one §10 write token, as the synchronous RVF import does
//! (over budget: [`Delivery::Retry`]) — that limiter, not CPU, paces it.
//!
//! Job saves are compare-and-swap (`jobs::save`): a delivery that loses to
//! a concurrent one (duplicate or stale message) stops and acks. A delivery
//! that finishes or fails the job returns its audit event
//! (`job.done` / `job.failed` / `job.cancelled`, as the submitter).

use crate::audit::AuditEvent;
use crate::backend::Backend;
use crate::ingest_hash::HashStep;
use crate::ingest_sink::{sink_fail, submitter};
pub use crate::ingest_sink::{BatchSink, UpsertSink};
use crate::jobs::{self, IngestMsg, Job};
use crate::m3_ctx::snap_metric;
use crate::m3_ports::Blob;
use crate::m3_wire::{kv_get, kv_put, M3Backend, Ns};
use crate::rest::admin::deny_check;
use crate::service::{self, Call};
use crate::wire::{LedgerCall, LedgerOut};
use ruvector_edge_auth::Capability;
use ruvector_edge_snapshot::{
    inspect_tail, FailCode, ImportError, ImportLimits, ImportSpec, JobState, RvfImporter,
};
use ruvector_edge_store::shard::HNSW_SYNC_UPSERT;
use ruvector_edge_store::{ErrorCode, IndexConfig, OpError};
use ruvector_edge_tenancy::quota::limits::MAX_UPSERT_BATCH;
use ruvector_edge_tenancy::TenantKey;

/// `op_id` idempotency window (ADR-351 §16.3), from submission.
pub const JOB_WINDOW_MS: u64 = 24 * 3600 * 1000;
/// Failure reason of a job past the window.
pub const EXPIRED: &str = "expired: older than the 24 h op_id window";
/// Failure reason of a job whose deliveries stopped making progress.
pub const STALLED: &str = "stalled: deliveries keep failing without progress";
/// Consecutive import deliveries without a committed batch before the job
/// fails: below the ingest queue's `max_retries` (20), so a job whose
/// message would be dead-lettered ends `failed`, not `running`.
pub const MAX_STALLED_DELIVERIES: u32 = 15;
/// Largest segment payload a queued import accepts: the snapshot format's
/// `MAX_SEGMENT_PAYLOAD` and the [`TAIL_BYTES`] buffer already accepted,
/// so it adds no new peak. Batches never span a record, so every delivery
/// re-reads and re-verifies (≈ 84 ms of wasm SHA-256 at 8 MiB) the record
/// it resumes in; the record is buffered whole, which makes this a memory
/// bound. The `rvf` CLI default (`--batch-size 1000`: 1.54 MB at 384-d,
/// ≈ 6.1 MB at 1536-d) imports unchanged. (Free: 1 MiB.)
pub const QUEUED_MAX_SEGMENT_PAYLOAD: u64 = 8 << 20;
/// Range read size.
pub const PIECE_BYTES: u64 = 1 << 20;
/// Tail read: the largest manifest segment plus its header and padding.
pub const TAIL_BYTES: u64 = (8 << 20) + 128;
/// Batches per delivery before re-enqueueing: ≈ 20–40 ms of JSON encoding
/// at [`FLOATS_PER_BATCH`]. More only turns into [`Delivery::Retry`]: each
/// batch after the first pays a write token and `RL_WRITE_USER` is 10 per
/// 10 s. (Free: 2.)
pub const BATCHES_PER_DELIVERY: u32 = 8;
/// Float values per upsert batch, an upper bound only: the encoded-bytes
/// bound of [`batch_rows`] binds first at every dim (93 rows at 384 dims,
/// 34 at 1536). Pinned per job ([`batch_rows`]), so jobs queued before a
/// change keep their batching. (Free: 24,576.)
pub const FLOATS_PER_BATCH: u64 = 65_536;

/// Rows per batch for a `dim`-dimensional collection: [`FLOATS_PER_BATCH`]
/// worth and at most `ingest_sink::rows_within_body` (the batch's upsert
/// body is ≤ 1 MiB whatever its metadata), between 1 and the M1 upsert
/// cap (500, or `HNSW_SYNC_UPSERT` = 64 for an `hnsw` collection, whose
/// shards refuse larger batches). Pinned in the job on its first delivery
/// (it feeds the batch `op_id`s).
pub fn batch_rows(dim: u32, index: IndexConfig) -> usize {
    let cap = match index {
        IndexConfig::Hnsw { .. } => HNSW_SYNC_UPSERT as u64,
        _ => u64::from(MAX_UPSERT_BATCH),
    };
    let rows =
        (FLOATS_PER_BATCH / u64::from(dim.max(1))).min(crate::ingest_sink::rows_within_body(dim));
    rows.clamp(1, cap) as usize
}

/// What the consumer does with the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delivery {
    /// Ack (done, failed, or nothing to do).
    Done,
    /// Ack and send a continuation message.
    Continue,
    /// Leave for redelivery (transient failure; cursor is persisted).
    Retry,
}

/// Outcome of one delivery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    /// Consumer action.
    pub outcome: Delivery,
    /// Batches committed in this delivery.
    pub batches: u64,
    /// Peak importer buffer (bytes).
    pub peak_buffered: usize,
    /// Audit event of a job this delivery finished or failed.
    pub audit: Option<AuditEvent>,
}

impl Report {
    fn of(outcome: Delivery) -> Self {
        Report {
            outcome,
            batches: 0,
            peak_buffered: 0,
            audit: None,
        }
    }
}

/// The audit event of a job reaching a terminal state (as its submitter).
pub fn job_event(j: &Job, route: &str, outcome: u16, now_ms: u64) -> AuditEvent {
    let m = &j.meta;
    let c = &j.job.cursor;
    AuditEvent {
        v: 1,
        ts_ms: now_ms,
        tenant_key: j.job.tenant_key.clone(),
        sub: m.sub.clone(),
        client_id: m.client_id.clone(),
        act_sub: m.act_sub.clone(),
        jti: m.jti.clone(),
        family_id: m.family_id.clone(),
        route: route.to_string(),
        scope: Some(Capability::Write.satisfying_scope().to_string()),
        role: None,
        outcome,
        rows: c.rows_done,
        // Bytes read: the whole upload once done, else up to the cursor.
        bytes: match j.job.state {
            JobState::Done => j.job.upload_size,
            _ => c.byte_offset,
        },
        // The M1 upsert executor charges 1 + rows per batch.
        work_units: c.batch_seq.saturating_add(c.rows_done),
    }
}

/// Outcome code of a failed job (HTTP-like).
pub fn fail_status(code: FailCode) -> u16 {
    match code {
        FailCode::Malformed => 400,
        FailCode::Incompatible => 409,
        FailCode::Integrity => 422,
        FailCode::QuotaExceeded => 429,
        FailCode::Cancelled => 410,
    }
}

fn import_fail(e: &ImportError) -> FailCode {
    match e {
        // A segment over the queued cap: the file is valid, just not
        // importable here (re-export with smaller segments).
        ImportError::DimensionMismatch { .. }
        | ImportError::MetricMismatch
        | ImportError::SegmentTooLarge(_) => FailCode::Incompatible,
        ImportError::QuotaExceeded { .. } | ImportError::FileTooLarge => FailCode::QuotaExceeded,
        ImportError::Checksum(_) | ImportError::ManifestMismatch(_) | ImportError::Truncated => {
            FailCode::Integrity
        }
        _ => FailCode::Malformed,
    }
}

async fn fail<B: M3Backend>(
    b: &B,
    t: &TenantKey,
    j: &mut Job,
    code: FailCode,
    why: &str,
    now_ms: u64,
) -> Result<Report, OpError> {
    // `fail` only refuses from a terminal state, which was checked first.
    let _ = j.job.fail(code);
    j.meta.error = Some(why.to_string());
    j.meta.updated_at_ms = now_ms;
    jobs::save(b, t, j).await?;
    let route = match code {
        FailCode::Cancelled => "job.cancelled",
        _ => "job.failed",
    };
    Ok(Report {
        audit: Some(job_event(j, route, fail_status(code), now_ms)),
        ..Report::of(Delivery::Done)
    })
}

/// The tenant's remaining row quota (vectors and float budget).
pub(crate) async fn remaining<B: Backend>(
    b: &B,
    t: &TenantKey,
    dim: u32,
    now: u64,
) -> Result<u64, OpError> {
    let out = crate::backend::ledger(b, t, LedgerCall::Usage { now }).await?;
    let LedgerOut::Usage { report } = out else {
        return Err(service::unexpected());
    };
    let u = |k: &str| report["usage"][k].as_u64().unwrap_or(u64::MAX);
    let l = |k: &str| report["limits"][k].as_u64().unwrap_or(0);
    let vectors = l("max_vectors").saturating_sub(u("vectors"));
    let floats = l("max_float_budget").saturating_sub(u("float_budget")) / u64::from(dim.max(1));
    Ok(vectors.min(floats))
}

/// Run one delivery of `msg`.
pub async fn process<B: M3Backend, R: Blob, S: BatchSink>(
    b: &B,
    blob: &R,
    sink: &S,
    msg: &IngestMsg,
    now_ms: u64,
    budget: u32,
) -> Result<Report, OpError> {
    match run(b, blob, sink, msg, now_ms, budget).await {
        // A concurrent delivery owns the job now: this message is spent.
        Err(e) if jobs::is_job_changed(&e) => Ok(Report::of(Delivery::Done)),
        r => r,
    }
}

async fn run<B: M3Backend, R: Blob, S: BatchSink>(
    b: &B,
    blob: &R,
    sink: &S,
    msg: &IngestMsg,
    now_ms: u64,
    budget: u32,
) -> Result<Report, OpError> {
    let Ok(t) = TenantKey::parse(&msg.tenant_key) else {
        return Ok(Report::of(Delivery::Done));
    };
    let Some(mut j) = jobs::load(b, &t, &msg.job_id).await? else {
        return Ok(Report::of(Delivery::Done));
    };
    if matches!(j.job.state, JobState::Done | JobState::Failed(_)) {
        return Ok(Report::of(Delivery::Done));
    }
    if now_ms.saturating_sub(j.meta.created_at_ms) > JOB_WINDOW_MS {
        return fail(b, &t, &mut j, FailCode::Cancelled, EXPIRED, now_ms).await;
    }
    if j.meta.stalled >= MAX_STALLED_DELIVERIES {
        release_upload(b, &t, &j).await;
        return fail(b, &t, &mut j, FailCode::Cancelled, STALLED, now_ms).await;
    }
    // Cheap ledger reads first: a denied submitter or a dropped collection
    // stops the job before any hash-pass delivery is spent on it.
    let ctx = submitter(&j)?;
    // ADR-351 §5.8: a denied submitter (jti / family / sub / client) stops
    // its queued import, as it would stop its next request.
    match deny_check(b, &ctx, now_ms / 1000).await {
        Ok(()) => {}
        Err(e) if e.code == ErrorCode::InvalidToken => {
            let why = "submitter denied";
            return fail(b, &t, &mut j, FailCode::Cancelled, why, now_ms).await;
        }
        Err(e) => return Err(e),
    }
    let call = Call {
        b,
        ctx: &ctx,
        dry_run: false,
        now: now_ms / 1000,
    };
    let e = match service::lookup(&call, &j.meta.collection).await {
        Ok(e) if e.uid == j.meta.uid => e,
        Ok(_)
        | Err(OpError {
            code: ErrorCode::NotFound,
            ..
        }) => {
            let why = "collection deleted or recreated";
            return fail(b, &t, &mut j, FailCode::Incompatible, why, now_ms).await;
        }
        Err(err) => return Err(err),
    };
    let key = j.job.staging_key();
    let Some(facts) = blob.head(&key).await? else {
        return fail(b, &t, &mut j, FailCode::Integrity, "upload missing", now_ms).await;
    };
    if facts.sha256.is_none() && !j.meta.sha_verified {
        // Multipart objects carry no sha256: hash the whole upload before
        // the first batch is applied, one bounded slice per delivery.
        if facts.size != j.job.upload_size {
            let why = "upload size mismatch";
            return fail(b, &t, &mut j, FailCode::Integrity, why, now_ms).await;
        }
        let pass = j.meta.hash_pass.unwrap_or_default();
        let (off, len) = pass.next_range(facts.size);
        let piece = blob.get_range(&key, off, len).await?.unwrap_or_default();
        match pass.step(&piece, facts.size) {
            Some(HashStep::More(next)) => j.meta.hash_pass = Some(next),
            Some(HashStep::Done(d)) if d == j.job.upload_sha256 => {
                j.meta.hash_pass = None;
                j.meta.sha_verified = true;
            }
            Some(HashStep::Done(_)) => {
                let why = "upload sha256 mismatch";
                return fail(b, &t, &mut j, FailCode::Integrity, why, now_ms).await;
            }
            None => return fail(b, &t, &mut j, FailCode::Integrity, "short read", now_ms).await,
        }
        j.meta.updated_at_ms = now_ms;
        jobs::save(b, &t, &mut j).await?;
        return Ok(Report::of(Delivery::Continue));
    }
    let dim = u16::try_from(e.cfg.dim).map_err(|_| OpError::invalid("dimension"))?;
    let limits = ImportLimits {
        max_rows: remaining(b, &t, e.cfg.dim, now_ms / 1000).await?,
        max_batch_rows: batch_rows(e.cfg.dim, e.cfg.index),
        max_segment_payload: QUEUED_MAX_SEGMENT_PAYLOAD,
        ..ImportLimits::default()
    };
    let tail_from = match j.meta.tail_from {
        Some(at) if at < facts.size => at,
        _ => facts.size - facts.size.min(TAIL_BYTES),
    };
    let tail_len = facts.size - tail_from;
    let tail = blob
        .get_range(&key, tail_from, tail_len)
        .await?
        .unwrap_or_default();
    let summary = match inspect_tail(&tail, facts.size, &limits) {
        Ok(s) => s,
        Err(ie) => return fail(b, &t, &mut j, import_fail(&ie), "rvf tail", now_ms).await,
    };
    drop(tail);
    let Ok((cursor, pinned)) = j.job.deliver(facts, &summary, limits) else {
        let why = "upload mismatch";
        return fail(b, &t, &mut j, FailCode::Integrity, why, now_ms).await;
    };
    j.meta.tail_from = Some(summary.manifest_offset);
    // Counted before any batch; the first committed batch resets it.
    j.meta.stalled = j.meta.stalled.saturating_add(1);
    j.meta.updated_at_ms = now_ms;
    jobs::save(b, &t, &mut j).await?;
    let spec = ImportSpec {
        dim,
        metric: snap_metric(e.cfg.metric),
        limits: pinned,
    };
    // The upload is already verified (R2 checksum at `deliver`, or the
    // resumable pass): no second whole-file hash in the import deliveries.
    let mut imp = match RvfImporter::new(spec, summary, cursor) {
        Ok(i) => i.without_file_hash(),
        Err(ie) => return fail(b, &t, &mut j, import_fail(&ie), "rvf refused", now_ms).await,
    };
    let (mut off, mut n) = (cursor.byte_offset, 0u64);
    while off < facts.size {
        let len = PIECE_BYTES.min(facts.size - off);
        let piece = blob.get_range(&key, off, len).await?.unwrap_or_default();
        if piece.len() as u64 != len {
            return fail(b, &t, &mut j, FailCode::Integrity, "short read", now_ms).await;
        }
        if let Err(ie) = imp.feed(&piece) {
            return fail(b, &t, &mut j, import_fail(&ie), "rvf stream", now_ms).await;
        }
        drop(piece);
        off += len;
        loop {
            let batch = match imp.next_batch() {
                Ok(Some(batch)) => batch,
                Ok(None) => break,
                Err(ie) => return fail(b, &t, &mut j, import_fail(&ie), "rvf batch", now_ms).await,
            };
            let op_id = j.job.op_id(batch.seq).map_err(|_| service::unexpected())?;
            // ADR-351 §10 layer 2, as the synchronous RVF import: the `:import`
            // request's token paid batch 0, every later batch pays one more.
            if batch.seq > 0 && b.charge_writes(&ctx, 1).await.is_err() {
                return Ok(Report {
                    batches: n,
                    peak_buffered: imp.peak_buffered_bytes(),
                    ..Report::of(Delivery::Retry)
                });
            }
            if let Err(err) = sink.apply(&j, &op_id, batch.rows.clone()).await {
                return match sink_fail(&err) {
                    Some(code) => fail(b, &t, &mut j, code, err.code.as_str(), now_ms).await,
                    None => Ok(Report {
                        batches: n,
                        peak_buffered: imp.peak_buffered_bytes(),
                        ..Report::of(Delivery::Retry)
                    }),
                };
            }
            j.job
                .commit_batch(&batch)
                .map_err(|_| service::unexpected())?;
            j.meta.stalled = 0;
            j.meta.updated_at_ms = now_ms;
            jobs::save(b, &t, &mut j).await?;
            n += 1;
            if n >= u64::from(budget) {
                return Ok(Report {
                    batches: n,
                    peak_buffered: imp.peak_buffered_bytes(),
                    ..Report::of(Delivery::Continue)
                });
            }
        }
    }
    let peak = imp.peak_buffered_bytes();
    let totals = match imp.finish() {
        Ok(t) => t,
        Err(ie) => return fail(b, &t, &mut j, import_fail(&ie), "rvf end", now_ms).await,
    };
    if j.job.complete(&totals).is_err() {
        return fail(b, &t, &mut j, FailCode::Integrity, "upload sha256", now_ms).await;
    }
    j.meta.updated_at_ms = now_ms;
    jobs::save(b, &t, &mut j).await?;
    Ok(Report {
        outcome: Delivery::Done,
        batches: n,
        peak_buffered: peak,
        audit: Some(job_event(&j, "job.done", 200, now_ms)),
    })
}

/// Give a stalled job's upload back (`consumed` → `complete`) so the
/// client can submit it again. Best effort: the job fails either way.
async fn release_upload<B: M3Backend>(b: &B, t: &TenantKey, j: &Job) {
    let id = &j.job.upload_id;
    let Ok(Some(raw)) = kv_get(b, t, Ns::Upload, id).await else {
        return;
    };
    let Ok(mut u) = serde_json::from_str::<crate::uploads::Upload>(&raw) else {
        return;
    };
    if u.state == "consumed" {
        u.state = "complete".into();
        if let Ok(v) = serde_json::to_string(&u) {
            let _saved = kv_put(b, t, Ns::Upload, id, v).await;
        }
    }
}
