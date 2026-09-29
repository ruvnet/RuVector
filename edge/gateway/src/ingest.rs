//! `ruvector-edge-ingest` consumer core (ADR-351 §3 rv-ingest): one
//! delivery of one import job.
//!
//! A delivery loads the job from its tenant's ledger and fails it once it
//! is older than the 24 h `op_id` window **since submission** (every batch
//! it applied was recorded under an `op_id` whose idempotency slot lives
//! 24 h from that batch, so inside the window a replayed batch is always
//! deduplicated; past it, it could not be). It checks the staging object
//! (`head`); an upload R2 holds no sha256 for (multipart) gets one streamed
//! hash pass against the declared sha256, in a delivery of its own, before
//! any batch is applied. It then resolves the collection by name **and**
//! uid, pins the tenant's remaining row quota, range-reads the tail
//! (`inspect_tail`), `deliver`s the job and streams the object from the
//! cursor in 1 MiB ranges through `RvfImporter`. Each batch goes to a
//! [`BatchSink`] under `job.op_id(seq)` and the cursor is persisted after
//! every batch, so a crash at any point resumes without loss and a batch
//! applied but not recorded is replayed by the sink's `op_id` idempotency,
//! not re-applied. After [`BATCHES_PER_DELIVERY`] batches the delivery stops
//! with [`Delivery::Continue`] (the consumer re-enqueues and acks), keeping
//! each invocation's CPU bounded.
//!
//! Job saves are compare-and-swap (`jobs::save`): a delivery that loses to
//! a concurrent one (duplicate or stale message) stops and acks. A delivery
//! that finishes or fails the job returns its audit event
//! (`job.done` / `job.failed` / `job.cancelled`, as the submitter).

use crate::audit::AuditEvent;
use crate::backend::Backend;
use crate::idem::{self, Seen, Slot};
use crate::jobs::{self, IngestMsg, Job};
use crate::m3_ctx::snap_metric;
use crate::m3_ports::Blob;
use crate::m3_wire::M3Backend;
use crate::service::{self, Call};
use crate::wire::{LedgerCall, LedgerOut};
use ruvector_edge_auth::{Capability, CapabilitySet};
use ruvector_edge_snapshot::{
    inspect_tail, FailCode, ImportError, ImportLimits, ImportSpec, JobState, Row, RvfImporter,
};
use ruvector_edge_store::{CallerContext, ErrorCode, Op, OpError};
use ruvector_edge_tenancy::TenantKey;
use serde_json::json;

/// `op_id` idempotency window (ADR-351 §16.3), from submission.
pub const JOB_WINDOW_MS: u64 = 24 * 3600 * 1000;
/// Failure reason of a job past the window.
pub const EXPIRED: &str = "expired: older than the 24 h op_id window";
/// Range read size.
pub const PIECE_BYTES: u64 = 1 << 20;
/// Tail read: the largest manifest segment plus its header and padding.
pub const TAIL_BYTES: u64 = (8 << 20) + 128;
/// Batches per delivery before re-enqueueing.
pub const BATCHES_PER_DELIVERY: u32 = 50;

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

/// Where batches go.
#[allow(async_fn_in_trait)]
pub trait BatchSink {
    /// Apply `rows` under `op_id` (idempotent per `op_id`).
    async fn apply(&self, job: &Job, op_id: &str, rows: Vec<Row>) -> Result<(), OpError>;
}

/// The production sink: the M1 upsert executor, re-authorized as the
/// submitter (scope `write` ∩ current role) and deduplicated through the
/// ledger's `op_id` table under `import:{op_id}`.
pub struct UpsertSink<'a, B> {
    /// DO transport.
    pub b: &'a B,
    /// Unix seconds.
    pub now: u64,
}

fn submitter(j: &Job) -> Result<CallerContext, OpError> {
    let t = TenantKey::parse(&j.job.tenant_key).map_err(|_| OpError::invalid("tenant"))?;
    let mut caps = CapabilitySet::EMPTY;
    caps.insert(Capability::Write);
    let m = &j.meta;
    Ok(CallerContext::new(
        t,
        m.sub.clone(),
        m.client_id.clone(),
        m.jti.clone(),
        m.family_id.clone(),
        m.act_sub.clone(),
        caps,
    ))
}

impl<B: Backend> BatchSink for UpsertSink<'_, B> {
    async fn apply(&self, job: &Job, op_id: &str, rows: Vec<Row>) -> Result<(), OpError> {
        use sha2::{Digest, Sha256};
        let ctx = submitter(job)?;
        let slot = Slot {
            sub: ctx.sub().to_string(),
            key: format!("import:{op_id}"),
            sha256: Sha256::digest(format!("{}|{op_id}", job.job.job_id)).into(),
        };
        let reused = "import op_id reused";
        if let Seen::Replay(_) = idem::check(self.b, &ctx, &slot, true, reused, self.now).await? {
            return Ok(());
        }
        let mut vectors = Vec::with_capacity(rows.len());
        for r in rows {
            let metadata = match r.metadata {
                None => None,
                Some(t) => Some(
                    serde_json::from_str::<serde_json::Value>(&t)
                        .map_err(|_| OpError::invalid("row metadata"))?,
                ),
            };
            vectors.push(json!({ "id": r.id, "values": r.values, "metadata": metadata }));
        }
        let args = json!({ "collection": job.meta.collection, "vectors": vectors }).to_string();
        let call = Call {
            b: self.b,
            ctx: &ctx,
            dry_run: false,
            now: self.now,
        };
        let res = match service::authorize_op(self.b, &ctx, Op::VectorUpsert).await {
            Ok(a) => service::run(&call, a, Op::VectorUpsert, &args).await,
            Err(e) => Err(e),
        };
        match res {
            Ok(_) => {
                idem::finish(self.b, &ctx, slot, Some("ok".into()), self.now).await;
                Ok(())
            }
            Err(e) => {
                idem::finish(self.b, &ctx, slot, None, self.now).await;
                Err(e)
            }
        }
    }
}

fn import_fail(e: &ImportError) -> FailCode {
    match e {
        ImportError::DimensionMismatch { .. } | ImportError::MetricMismatch => {
            FailCode::Incompatible
        }
        ImportError::QuotaExceeded { .. }
        | ImportError::FileTooLarge
        | ImportError::SegmentTooLarge(_) => FailCode::QuotaExceeded,
        ImportError::Checksum(_) | ImportError::ManifestMismatch(_) | ImportError::Truncated => {
            FailCode::Integrity
        }
        _ => FailCode::Malformed,
    }
}

/// `None` = transient (redeliver); otherwise the job fails with this code.
fn sink_fail(e: &OpError) -> Option<FailCode> {
    use ErrorCode as C;
    match e.code {
        C::QuotaExceeded | C::BudgetExceeded | C::PayloadTooLarge => Some(FailCode::QuotaExceeded),
        C::DimensionMismatch => Some(FailCode::Incompatible),
        C::InvalidRequest | C::NonFiniteValue => Some(FailCode::Malformed),
        C::OpReplayed => Some(FailCode::Integrity),
        C::InsufficientScope | C::RoleRequired | C::NotClaimed | C::NotFound => {
            Some(FailCode::Cancelled)
        }
        _ => None,
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
    let key = j.job.staging_key();
    let Some(facts) = blob.head(&key).await? else {
        return fail(b, &t, &mut j, FailCode::Integrity, "upload missing", now_ms).await;
    };
    if facts.sha256.is_none() && !j.meta.sha_verified {
        // Multipart objects carry no sha256: hash the whole upload once
        // (streamed, one subrequest) before the first batch is applied.
        let hashed = blob.sha256_stream(&key).await?;
        if hashed != Some((j.job.upload_sha256, j.job.upload_size)) {
            let why = "upload sha256 mismatch";
            return fail(b, &t, &mut j, FailCode::Integrity, why, now_ms).await;
        }
        j.meta.sha_verified = true;
        j.meta.updated_at_ms = now_ms;
        jobs::save(b, &t, &mut j).await?;
        return Ok(Report::of(Delivery::Continue));
    }
    let ctx = submitter(&j)?;
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
    let dim = u16::try_from(e.cfg.dim).map_err(|_| OpError::invalid("dimension"))?;
    let limits = ImportLimits {
        max_rows: remaining(b, &t, e.cfg.dim, now_ms / 1000).await?,
        ..ImportLimits::default()
    };
    let tail_len = facts.size.min(TAIL_BYTES);
    let tail = blob
        .get_range(&key, facts.size - tail_len, tail_len)
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
    j.meta.updated_at_ms = now_ms;
    jobs::save(b, &t, &mut j).await?;
    let spec = ImportSpec {
        dim,
        metric: snap_metric(e.cfg.metric),
        limits: pinned,
    };
    let mut imp = match RvfImporter::new(spec, summary, cursor) {
        Ok(i) => i,
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
