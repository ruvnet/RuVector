//! Bulk-import jobs (ADR-351 §3 rv-ingest, §7.2): `POST
//! /v1/collections/{c}:import` (write + editor) and `GET /v1/jobs/{id}`
//! (read + viewer). The consumer side is `ingest`.
//!
//! `:import` takes `{upload_id}` — resolved in the caller's own ledger, so a
//! foreign or unknown id is `404` and the job can only ever address
//! `staging/{caller tenant}/{upload_id}` — or an inline `.rvf` body
//! (`application/octet-stream`, ≤ 512 KiB). The job (`ImportJob`, RVJ2) plus
//! the submitter's identity (for re-authorization at every delivery) is
//! stored in the ledger and `{tenant_key, job_id}` goes to
//! `ruvector-edge-ingest`.

use crate::m3_ctx::{mint, object, M3};
use crate::m3_ports::{Blob, QueueName, Queues};
use crate::m3_wire::{kv_get, kv_swap, unhex32, M3Backend, Ns};
use crate::uploads;
use base64ct::{Base64, Encoding};
use ruvector_edge_auth::Capability;
use ruvector_edge_snapshot::{FailCode, ImportJob, JobState};
use ruvector_edge_store::{ErrorCode, OpError};
use ruvector_edge_tenancy::{Role, TenantKey};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

/// Who submitted a job and where it goes (stored next to the RVJ2 bytes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobMeta {
    /// Target collection name.
    pub collection: String,
    /// Its uid at submit time (a recreated name is refused).
    pub uid: String,
    /// Submitter `sub`.
    pub sub: String,
    /// Submitter `client_id`.
    pub client_id: String,
    /// Submitter `jti` (recorded in the shard op log).
    pub jti: String,
    /// Submitter `family_id`.
    pub family_id: String,
    /// `act.sub` of an exchanged token.
    #[serde(default)]
    pub act_sub: Option<String>,
    /// Unix ms.
    pub created_at_ms: u64,
    /// Unix ms of the last state change.
    pub updated_at_ms: u64,
    /// The upload's declared sha256 was checked by a full hash pass.
    #[serde(default)]
    pub sha_verified: bool,
    /// State of an unfinished sha256 pass (`ingest_hash`), persisted after
    /// every delivery of it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash_pass: Option<crate::ingest_hash::HashPass>,
    /// Offset of the upload's final manifest segment, learned on the first
    /// import delivery: later deliveries read only `[tail_from, size)`
    /// instead of the 8 MiB tail window (the pinned manifest hash still
    /// binds it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail_from: Option<u64>,
    /// Stable failure reason.
    #[serde(default)]
    pub error: Option<String>,
    /// Consecutive import deliveries that started (reached the early save)
    /// without committing a batch: a delivery killed mid-record, or one
    /// that keeps failing transiently. Reset by every committed batch; at
    /// `ingest::MAX_STALLED_DELIVERIES` the job fails (before the queue
    /// dead-letters its message) and its upload is released.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub stalled: u32,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// A job with its metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    /// State machine (RVJ2).
    pub job: ImportJob,
    /// Metadata.
    pub meta: JobMeta,
    /// The record as last loaded or saved (`None`: not stored yet); saves
    /// are compare-and-swap against it.
    pub stored: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct Record {
    job: String,
    meta: JobMeta,
}

/// Queue message on `ruvector-edge-ingest`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestMsg {
    /// Tenant (the job is loaded from this tenant's ledger).
    pub tenant_key: String,
    /// Job id.
    pub job_id: String,
}

fn id_ok(id: &str) -> bool {
    id.len() == 24 && id.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Load a job from `tenant`'s ledger.
pub async fn load<B: M3Backend>(
    b: &B,
    tenant: &TenantKey,
    id: &str,
) -> Result<Option<Job>, OpError> {
    if !id_ok(id) {
        return Ok(None);
    }
    let Some(v) = kv_get(b, tenant, Ns::Job, id).await? else {
        return Ok(None);
    };
    let bad = || OpError::new(ErrorCode::ServerError, "job record");
    let r: Record = serde_json::from_str(&v).map_err(|_| bad())?;
    let bytes = Base64::decode_vec(&r.job).map_err(|_| bad())?;
    let job = ImportJob::decode(&bytes).map_err(|_| bad())?;
    if job.tenant_key != tenant.as_str() {
        return Ok(None);
    }
    Ok(Some(Job {
        job,
        meta: r.meta,
        stored: Some(v),
    }))
}

/// Persist a job in its tenant's ledger, compare-and-swap against the
/// record it was loaded as: two deliveries of one job (at-least-once, or a
/// lost ack after a continuation) cannot overwrite each other, so a stale
/// chain can never roll a cursor back or resurrect a finished job; the
/// loser gets [`job_changed`].
pub async fn save<B: M3Backend>(b: &B, tenant: &TenantKey, j: &mut Job) -> Result<(), OpError> {
    let r = Record {
        job: Base64::encode_string(&j.job.encode()),
        meta: j.meta.clone(),
    };
    let v = serde_json::to_string(&r).map_err(|_| crate::service::unexpected())?;
    let expect = j.stored.clone();
    match kv_swap(b, tenant, Ns::Job, &j.job.job_id, expect, Some(v.clone())).await {
        Ok(()) => {
            j.stored = Some(v);
            Ok(())
        }
        Err(e) if e.code == ErrorCode::Conflict => Err(job_changed()),
        Err(e) => Err(e),
    }
}

/// Another delivery changed the job first.
pub fn job_changed() -> OpError {
    OpError::new(ErrorCode::Conflict, "job changed by another delivery")
}

/// `true` for [`job_changed`].
pub fn is_job_changed(e: &OpError) -> bool {
    e.code == ErrorCode::Conflict && e.detail == job_changed().detail
}

/// `POST /v1/collections/{c}:import` → 202 `{job_id, state}`.
pub async fn submit<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    collection: &str,
    body: Vec<u8>,
    octet_stream: bool,
) -> Result<(u16, Json), OpError> {
    m.require(Capability::Write, Role::Editor).await?;
    let e = m.collection(collection).await?;
    let mut upload = if octet_stream {
        uploads::inline(m, body).await?
    } else {
        let mut o = object(&body)?;
        let id = o
            .remove("upload_id")
            .and_then(|v| v.as_str().map(str::to_string));
        let (Some(id), true) = (id, o.is_empty()) else {
            return Err(OpError::invalid("expected {upload_id}"));
        };
        uploads::load(m, &id).await?
    };
    if upload.state != "complete" {
        return Err(OpError::new(ErrorCode::Conflict, "upload not complete"));
    }
    m.charge(1).await?;
    let tenant = m.ctx.tenant_key();
    let now = m.now_ms.to_le_bytes();
    let job_id = mint(
        &[
            tenant.as_str().as_bytes(),
            upload.upload_id.as_bytes(),
            e.uid.as_bytes(),
            &now,
        ],
        12,
    );
    let sha = unhex32(&upload.sha256)?;
    let job = ImportJob::new(
        &job_id,
        tenant.as_str(),
        &e.uid,
        &upload.upload_id,
        upload.size,
        sha,
    )
    .map_err(|_| OpError::invalid("import job"))?;
    let meta = JobMeta {
        collection: collection.to_string(),
        uid: e.uid.clone(),
        sub: m.ctx.sub().to_string(),
        client_id: m.ctx.client_id().to_string(),
        jti: m.ctx.jti().to_string(),
        family_id: m.ctx.family_id().to_string(),
        act_sub: m.ctx.act_sub().map(str::to_string),
        created_at_ms: m.now_ms,
        updated_at_ms: m.now_ms,
        sha_verified: false,
        hash_pass: None,
        tail_from: None,
        error: None,
        stalled: 0,
    };
    let mut j = Job {
        job,
        meta,
        stored: None,
    };
    save(m.b, tenant, &mut j).await?;
    let msg = IngestMsg {
        tenant_key: tenant.as_str().to_string(),
        job_id: job_id.clone(),
    };
    let body = serde_json::to_value(&msg).map_err(|_| crate::service::unexpected())?;
    if let Err(err) = m.queues.send(QueueName::Ingest, body).await {
        // Never left `queued` with no message to run it: the upload stays
        // `complete`, so the client can simply submit again.
        let _ = j.job.fail(FailCode::Cancelled);
        j.meta.error = Some("enqueue failed".into());
        let _saved = save(m.b, tenant, &mut j).await;
        return Err(err);
    }
    // One upload feeds one job, marked only once the job is enqueued.
    upload.state = "consumed".into();
    uploads::save(m, &upload).await?;
    Ok((202, json!({ "job_id": job_id, "state": "queued" })))
}

/// Public view of a job.
pub fn view(j: &Job) -> Json {
    let (state, fail) = match j.job.state {
        JobState::Queued => ("queued", None),
        JobState::Running => ("running", None),
        JobState::Done => ("done", None),
        JobState::Failed(f) => ("failed", Some(fail_name(f))),
    };
    let c = &j.job.cursor;
    json!({
        "job_id": j.job.job_id,
        "state": state,
        "failure": fail,
        "error": j.meta.error,
        "collection": j.meta.collection,
        "rows_done": c.rows_done,
        "rows_skipped": c.rows_skipped,
        "batches": c.batch_seq,
        "bytes_done": c.byte_offset,
        "size": j.job.upload_size,
        "attempts": j.job.attempts,
        "created_at": j.meta.created_at_ms / 1000,
        "updated_at": j.meta.updated_at_ms / 1000,
    })
}

/// Stable failure names.
pub fn fail_name(f: FailCode) -> &'static str {
    match f {
        FailCode::Malformed => "malformed",
        FailCode::Incompatible => "incompatible",
        FailCode::QuotaExceeded => "quota_exceeded",
        FailCode::Integrity => "integrity",
        FailCode::Cancelled => "cancelled",
    }
}

/// `GET /v1/jobs/{id}` (read + viewer; foreign ids are `404`).
pub async fn status<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    id: &str,
) -> Result<(u16, Json), OpError> {
    m.require(Capability::Read, Role::Viewer).await?;
    let j = load(m.b, m.ctx.tenant_key(), id)
        .await?
        .ok_or(OpError::not_found())?;
    m.charge(1).await?;
    let mut v = view(&j);
    // A job past the window fails at its next delivery; one with no
    // delivery left (e.g. dead-lettered) is reported as such already.
    let open = matches!(j.job.state, JobState::Queued | JobState::Running);
    if open && m.now_ms.saturating_sub(j.meta.created_at_ms) > crate::ingest::JOB_WINDOW_MS {
        v["state"] = json!("failed");
        v["failure"] = json!(fail_name(FailCode::Cancelled));
        v["error"] = json!(crate::ingest::EXPIRED);
    }
    Ok((200, v))
}
