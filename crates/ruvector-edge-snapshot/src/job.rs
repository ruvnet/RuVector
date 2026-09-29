//! Bulk-import job state machine (`/v1/jobs/{id}`, ADR-351 §3 rv-ingest).
//!
//! `Queued → Running → Done | Failed`. A running job persists its
//! [`ImportCursor`] after every committed batch. After a crash the Queue
//! redelivers, the job resumes from the cursor, and the importer reproduces
//! the same batches with the same sequence numbers; each batch is applied
//! under [`ImportJob::op_id`], so a batch that was applied but not yet
//! recorded is deduplicated by the shard's `op_id` idempotency and the final
//! state is identical to an uninterrupted run.
//!
//! **Binding.** The job records the upload's declared `{size, sha256}`
//! (`POST /v1/uploads`) and addresses the object only through
//! [`ImportJob::staging_key`], derived from its own validated `tenant_key`
//! and `upload_id` — a foreign upload id cannot be reached. The first
//! delivery pins the file's tail-manifest hash, the batch size and the
//! tenant's quota at job start ([`JobBinding`]); every later delivery must
//! present the same object (size, stored sha256 when R2 has one, manifest
//! hash) and gets the pinned limits back, so a redelivery can neither mix
//! two files nor regenerate a batch with different contents under the same
//! `op_id` (the op id also hashes the binding). A delivery that streamed the
//! whole file from byte 0 has its sha256 checked at [`ImportJob::complete`];
//! a resumed delivery relies on the pinned manifest hash + size, the
//! per-segment content hashes, and the object's stored sha256.
//!
//! `op_id` idempotency lasts 24 h (ADR-351 §16.3): the Worker must fail a job
//! whose `updated_at` is older than that instead of redelivering it.

use crate::bytes::{put_str16, Reader};
use crate::error::JobError;
use crate::rvf_import::{ImportBatch, ImportCursor, ImportTotals};
use crate::rvf_summary::{ImportLimits, RvfSummary};
use crate::types::{sha256, validate_collection_uid, validate_opaque_id, validate_tenant_key};

/// Job lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    /// Accepted, not started.
    Queued,
    /// Importing.
    Running,
    /// Finished; totals are final.
    Done,
    /// Refused or aborted with a stable code.
    Failed(FailCode),
}

/// Stable failure codes surfaced on `GET /v1/jobs/{id}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailCode {
    /// The upload is not a valid RVF file.
    Malformed,
    /// Dimension or metric mismatch.
    Incompatible,
    /// Quota or size limit.
    QuotaExceeded,
    /// Integrity check failed.
    Integrity,
    /// Operator / user cancel.
    Cancelled,
}

impl FailCode {
    fn code(self) -> u8 {
        self as u8
    }
    fn from_code(c: u8) -> Option<Self> {
        [
            Self::Malformed,
            Self::Incompatible,
            Self::QuotaExceeded,
            Self::Integrity,
            Self::Cancelled,
        ]
        .into_iter()
        .find(|f| f.code() == c)
    }
}

impl JobState {
    fn name(self) -> &'static str {
        match self {
            JobState::Queued => "queued",
            JobState::Running => "running",
            JobState::Done => "done",
            JobState::Failed(_) => "failed",
        }
    }
}

/// Parameters pinned at the first delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobBinding {
    /// sha256 of the file's final manifest payload ([`RvfSummary::manifest_hash`]).
    pub manifest_hash: [u8; 32],
    /// Rows per batch.
    pub max_batch_rows: u32,
    /// The tenant's remaining row quota when the job started.
    pub max_rows: u64,
}

/// What the Worker observed about the staging object at delivery time
/// (R2 `head()`): its size and, when the upload stored one, its sha256.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectFacts {
    /// Object size in bytes.
    pub size: u64,
    /// `R2Object.checksums.sha256`, if present.
    pub sha256: Option<[u8; 32]>,
}

/// A persisted bulk-import job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportJob {
    /// Server-minted job id.
    pub job_id: String,
    /// Owning tenant.
    pub tenant_key: String,
    /// Target collection.
    pub collection_uid: String,
    /// Server-minted upload id (see [`ImportJob::staging_key`]).
    pub upload_id: String,
    /// Declared upload size.
    pub upload_size: u64,
    /// Declared upload sha256.
    pub upload_sha256: [u8; 32],
    /// State.
    pub state: JobState,
    /// Resume point (last committed batch).
    pub cursor: ImportCursor,
    /// Deliveries so far (1 on first run).
    pub attempts: u32,
    /// Pinned at the first delivery.
    pub binding: Option<JobBinding>,
}

/// Deterministic 26-char `op_id` for batch `seq` of a job bound to `binding`:
/// `base32lower(sha256("v2|import|" ‖ job_id ‖ "|" ‖ manifest_hash ‖
/// max_batch_rows ‖ seq))[..26]`, matching the `/v1/ops` `OP_ID_LEN` of 26.
pub fn batch_op_id(job_id: &str, binding: &JobBinding, seq: u64) -> String {
    let d = sha256(&[
        b"v2|import|",
        job_id.as_bytes(),
        b"|",
        &binding.manifest_hash,
        &binding.max_batch_rows.to_le_bytes(),
        &seq.to_le_bytes(),
    ]);
    let mut s = data_encoding::BASE32_NOPAD.encode(&d).to_ascii_lowercase();
    s.truncate(26);
    s
}

impl ImportJob {
    /// A queued job; every identifier is validated.
    pub fn new(
        job_id: &str,
        tenant_key: &str,
        collection_uid: &str,
        upload_id: &str,
        upload_size: u64,
        upload_sha256: [u8; 32],
    ) -> Result<Self, JobError> {
        if !validate_opaque_id(job_id) {
            return Err(JobError::InvalidIdentifier("job_id"));
        }
        if !validate_opaque_id(upload_id) {
            return Err(JobError::InvalidIdentifier("upload_id"));
        }
        validate_tenant_key(tenant_key).map_err(|_| JobError::InvalidIdentifier("tenant_key"))?;
        validate_collection_uid(collection_uid)
            .map_err(|_| JobError::InvalidIdentifier("collection_uid"))?;
        Ok(ImportJob {
            job_id: job_id.to_string(),
            tenant_key: tenant_key.to_string(),
            collection_uid: collection_uid.to_string(),
            upload_id: upload_id.to_string(),
            upload_size,
            upload_sha256,
            state: JobState::Queued,
            cursor: ImportCursor::default(),
            attempts: 0,
            binding: None,
        })
    }

    /// The only R2 key this job reads: `staging/{tenant_key}/{upload_id}`,
    /// from validated fields (no `/`, `..` or foreign tenant possible).
    pub fn staging_key(&self) -> String {
        format!("staging/{}/{}", self.tenant_key, self.upload_id)
    }

    /// Start or resume (Queue delivery): `Queued|Running → Running`.
    /// Checks the object against the declaration and the pinned binding
    /// (pinning it on first delivery) and returns the cursor to stream from
    /// plus `limits` with the pinned batch size and quota.
    pub fn deliver(
        &mut self,
        object: ObjectFacts,
        summary: &RvfSummary,
        limits: ImportLimits,
    ) -> Result<(ImportCursor, ImportLimits), JobError> {
        if !matches!(self.state, JobState::Queued | JobState::Running) {
            return Err(JobError::IllegalTransition {
                from: self.state.name(),
            });
        }
        if object.size != self.upload_size || summary.file_len != self.upload_size {
            return Err(JobError::UploadMismatch("size"));
        }
        if object.sha256.is_some_and(|h| h != self.upload_sha256) {
            return Err(JobError::UploadMismatch("sha256"));
        }
        let b = match self.binding {
            Some(b) if b.manifest_hash != summary.manifest_hash => {
                return Err(JobError::UploadMismatch("manifest"));
            }
            Some(b) => b,
            None => JobBinding {
                manifest_hash: summary.manifest_hash,
                max_batch_rows: u32::try_from(limits.max_batch_rows)
                    .map_err(|_| JobError::InvalidIdentifier("max_batch_rows"))?,
                max_rows: limits.max_rows,
            },
        };
        self.binding = Some(b);
        self.state = JobState::Running;
        self.attempts += 1;
        let pinned = ImportLimits {
            max_batch_rows: b.max_batch_rows as usize,
            max_rows: b.max_rows,
            ..limits
        };
        Ok((self.cursor, pinned))
    }

    /// `op_id` of batch `seq` (requires a delivered job).
    pub fn op_id(&self, seq: u64) -> Result<String, JobError> {
        let b = self.binding.as_ref().ok_or(JobError::IllegalTransition {
            from: self.state.name(),
        })?;
        Ok(batch_op_id(&self.job_id, b, seq))
    }

    /// Record a batch after the shard applied it. A batch already recorded
    /// (replayed after a crash) returns `Ok(false)`; a gap is an error.
    pub fn commit_batch(&mut self, batch: &ImportBatch) -> Result<bool, JobError> {
        if self.state != JobState::Running {
            return Err(JobError::IllegalTransition {
                from: self.state.name(),
            });
        }
        let expected = self.cursor.batch_seq;
        if batch.seq < expected {
            return Ok(false);
        }
        if batch.seq > expected {
            return Err(JobError::BatchGap {
                expected,
                got: batch.seq,
            });
        }
        self.cursor = batch.cursor_after;
        Ok(true)
    }

    /// `Running → Done`, checking the stream totals agree with the cursor,
    /// the file length with the declaration and — when this delivery hashed
    /// the whole file — its sha256. On `Err` the caller fails the job
    /// (`FailCode::Integrity` for an upload mismatch).
    pub fn complete(&mut self, totals: &ImportTotals) -> Result<(), JobError> {
        if self.state != JobState::Running {
            return Err(JobError::IllegalTransition {
                from: self.state.name(),
            });
        }
        if totals.batches != self.cursor.batch_seq || totals.rows != self.cursor.rows_done {
            return Err(JobError::BatchGap {
                expected: totals.batches,
                got: self.cursor.batch_seq,
            });
        }
        if totals.bytes != self.upload_size {
            return Err(JobError::UploadMismatch("size"));
        }
        if totals.sha256.is_some_and(|h| h != self.upload_sha256) {
            return Err(JobError::UploadMismatch("sha256"));
        }
        self.cursor.rows_skipped = totals.skipped_deleted;
        self.state = JobState::Done;
        Ok(())
    }

    /// `Queued|Running → Failed(code)`.
    pub fn fail(&mut self, code: FailCode) -> Result<(), JobError> {
        match self.state {
            JobState::Queued | JobState::Running => {
                self.state = JobState::Failed(code);
                Ok(())
            }
            s => Err(JobError::IllegalTransition { from: s.name() }),
        }
    }

    /// Compact persisted form (DO storage row).
    pub fn encode(&self) -> Vec<u8> {
        let mut o = Vec::with_capacity(256);
        o.extend_from_slice(b"RVJ2");
        for s in [
            &self.job_id,
            &self.tenant_key,
            &self.collection_uid,
            &self.upload_id,
        ] {
            put_str16(&mut o, s);
        }
        o.extend_from_slice(&self.upload_size.to_le_bytes());
        o.extend_from_slice(&self.upload_sha256);
        let (tag, code) = match self.state {
            JobState::Queued => (0u8, 0u8),
            JobState::Running => (1, 0),
            JobState::Done => (2, 0),
            JobState::Failed(f) => (3, f.code()),
        };
        o.extend_from_slice(&[tag, code]);
        let c = &self.cursor;
        o.extend_from_slice(&c.byte_offset.to_le_bytes());
        o.extend_from_slice(&c.row_in_record.to_le_bytes());
        o.extend_from_slice(&c.batch_seq.to_le_bytes());
        o.extend_from_slice(&c.rows_done.to_le_bytes());
        o.extend_from_slice(&c.rows_skipped.to_le_bytes());
        o.extend_from_slice(&self.attempts.to_le_bytes());
        match &self.binding {
            None => o.push(0),
            Some(b) => {
                o.push(1);
                o.extend_from_slice(&b.manifest_hash);
                o.extend_from_slice(&b.max_batch_rows.to_le_bytes());
                o.extend_from_slice(&b.max_rows.to_le_bytes());
            }
        }
        o
    }

    /// Inverse of [`ImportJob::encode`] (identifiers re-validated).
    pub fn decode(b: &[u8]) -> Result<Self, JobError> {
        let bad = JobError::Corrupt;
        let mut r = Reader::new(b);
        if r.take(4) != Some(b"RVJ2".as_slice()) {
            return Err(bad("magic"));
        }
        let mut ids = [""; 4];
        for id in &mut ids {
            *id = r.str16(64).ok_or(bad("identifier"))?;
        }
        let size = r.u64().ok_or(bad("upload"))?;
        let digest = r.arr32().ok_or(bad("upload"))?;
        let mut job = ImportJob::new(ids[0], ids[1], ids[2], ids[3], size, digest)?;
        let (tag, code) = (r.u8().ok_or(bad("state"))?, r.u8().ok_or(bad("state"))?);
        job.state = match tag {
            0 => JobState::Queued,
            1 => JobState::Running,
            2 => JobState::Done,
            3 => JobState::Failed(FailCode::from_code(code).ok_or(bad("fail code"))?),
            _ => return Err(bad("state")),
        };
        job.cursor = ImportCursor {
            byte_offset: r.u64().ok_or(bad("cursor"))?,
            row_in_record: r.u32().ok_or(bad("cursor"))?,
            batch_seq: r.u64().ok_or(bad("cursor"))?,
            rows_done: r.u64().ok_or(bad("cursor"))?,
            rows_skipped: r.u64().ok_or(bad("cursor"))?,
        };
        job.attempts = r.u32().ok_or(bad("attempts"))?;
        job.binding = match r.u8().ok_or(bad("binding"))? {
            0 => None,
            1 => Some(JobBinding {
                manifest_hash: r.arr32().ok_or(bad("binding"))?,
                max_batch_rows: r.u32().ok_or(bad("binding"))?,
                max_rows: r.u64().ok_or(bad("binding"))?,
            }),
            _ => return Err(bad("binding")),
        };
        if r.remaining() != 0 {
            return Err(bad("trailing bytes"));
        }
        Ok(job)
    }
}
