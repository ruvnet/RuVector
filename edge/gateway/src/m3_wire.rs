//! M3 side channel to the two Durable Object classes (ADR-351 §6.2, §6.3,
//! §15 M3). The M1 protocol (`wire`) is untouched: M3 calls are posted to
//! the DO path [`M3_PATH`] and served by `m3_ledger` / `m3_shard`, which
//! run their own tables next to the M1 ones and, where a step must be atomic
//! with an M1 write, call the M1 cores synchronously in the same DO turn.
//!
//! As in M1, every request names the tenant from the verified token; DO
//! names are derived from it (never from request input) and each DO asserts
//! the identity it stores.

pub use crate::m3_transport::{ledger3, shard3, M3Backend};
use crate::wire::{ActorWire, CfgWire, CollectionWire, DeltaWire, Reply};
use base64ct::{Base64, Encoding};
use ruvector_edge_snapshot::{Row, WitnessEntry};
use ruvector_edge_store::{ErrorCode, OpError};
use ruvector_edge_tenancy::{QuotaDelta, TenantKey};
use serde::{Deserialize, Serialize};

/// DO path of the M3 side channel (M1 calls use `/rpc`).
pub const M3_PATH: &str = "/m3";

/// One stored row across the DO boundary: `f32` is the little-endian
/// vector bytes, base64 (never JSON floats); `metadata` the stored text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowWire {
    /// Vector id.
    pub id: String,
    /// base64 of `f32` LE bytes.
    pub f32: String,
    /// Canonical metadata JSON text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
}

impl RowWire {
    /// From raw stored parts.
    pub fn from_parts(id: String, f32_le: &[u8], metadata: Option<String>) -> Self {
        RowWire {
            id,
            f32: Base64::encode_string(f32_le),
            metadata,
        }
    }

    /// From a snapshot/import row.
    pub fn from_row(r: &Row) -> Self {
        let bytes: Vec<u8> = r.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::from_parts(r.id.clone(), &bytes, r.metadata.clone())
    }

    /// The raw `f32` bytes.
    pub fn f32_bytes(&self) -> Result<Vec<u8>, OpError> {
        Base64::decode_vec(&self.f32).map_err(|_| OpError::invalid("row encoding"))
    }

    /// Decode into a validated row of `dim` values.
    pub fn into_row(self, dim: u16) -> Result<Row, OpError> {
        let bytes = self.f32_bytes()?;
        if bytes.len() != usize::from(dim) * 4 {
            return Err(OpError::new(ErrorCode::DimensionMismatch, "row dimension"));
        }
        let values = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let row = Row {
            id: self.id,
            values,
            metadata: self.metadata,
        };
        row.check(dim).map_err(|_| OpError::invalid("row"))?;
        Ok(row)
    }
}

/// Lowercase hex of 32 bytes.
pub fn hex32(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Inverse of [`hex32`].
pub fn unhex32(s: &str) -> Result<[u8; 32], OpError> {
    let bad = || OpError::invalid("hex");
    if s.len() != 64 || !s.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(bad());
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| bad())?;
    }
    Ok(out)
}

/// A witness entry across the DO boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryWire {
    /// Position.
    pub seq: u64,
    /// Previous hash (hex).
    pub prev: String,
    /// Manifest root (hex).
    pub root: String,
    /// Collection uid.
    pub uid: String,
    /// Shard.
    pub shard: u16,
    /// Epoch.
    pub epoch: u64,
    /// Audit head recorded in the manifest (hex).
    pub audit_head: String,
    /// Entry hash (hex).
    pub hash: String,
}

impl EntryWire {
    /// From a chain entry.
    pub fn from_entry(e: &WitnessEntry) -> Self {
        EntryWire {
            seq: e.seq,
            prev: hex32(&e.prev),
            root: hex32(&e.manifest_root),
            uid: e.collection_uid.clone(),
            shard: e.shard,
            epoch: e.epoch,
            audit_head: hex32(&e.audit_head),
            hash: hex32(&e.hash),
        }
    }

    /// Back to a chain entry.
    pub fn to_entry(&self) -> Result<WitnessEntry, OpError> {
        Ok(WitnessEntry {
            seq: self.seq,
            prev: unhex32(&self.prev)?,
            manifest_root: unhex32(&self.root)?,
            collection_uid: self.uid.clone(),
            shard: self.shard,
            epoch: self.epoch,
            audit_head: unhex32(&self.audit_head)?,
            hash: unhex32(&self.hash)?,
        })
    }
}

/// Ledger key-value namespaces (fixed set; keys are server-derived).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ns {
    /// `{uid}/{epoch:020}` → snapshot record.
    Snapshot,
    /// `{export_id}` → export record.
    Export,
    /// `{upload_id}` → upload session.
    Upload,
    /// `{upload_id}/{part:05}` → part etag.
    UploadPart,
    /// `{job_id}` → import job.
    Job,
    /// `{uid}` → embedder model.
    Embedder,
    /// `{uid}` → restore journal (`restore::Journal`).
    Restore,
}

impl Ns {
    /// Stored tag.
    pub fn as_str(self) -> &'static str {
        match self {
            Ns::Snapshot => "snapshot",
            Ns::Export => "export",
            Ns::Upload => "upload",
            Ns::UploadPart => "upload_part",
            Ns::Job => "job",
            Ns::Embedder => "embedder",
            Ns::Restore => "restore",
        }
    }
}

/// A `TenantLedger` M3 request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct M3LedgerRequest {
    /// Tenant key from the verified token.
    pub tenant_key: String,
    /// The call.
    pub call: M3LedgerCall,
}

/// `TenantLedger` M3 calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum M3LedgerCall {
    /// Store a record.
    KvPut { ns: Ns, key: String, value: String },
    /// Read a record.
    KvGet { ns: Ns, key: String },
    /// Records with `from <= key < to`, key order.
    KvRange {
        ns: Ns,
        from: String,
        to: String,
        limit: u32,
    },
    /// Create a collection with an embedder in one DO turn (validate,
    /// admit the op, create, record the model).
    CreateWithEmbedder {
        spec: String,
        model: String,
        sub: String,
        now: u64,
    },
    /// Allocate the next snapshot epoch of a collection (starts at 1).
    NextEpoch { uid: String },
    /// Witness a sealed manifest (`manifest.rvf` bytes, base64).
    WitnessAppend { manifest: String },
    /// The whole witness chain plus the trusted head.
    Chain,
    /// Compare-and-swap a record: only if the stored value equals `expect`
    /// (`None` = absent), store `value` (`None` = delete); else `409`. In
    /// the same DO turn, before the write: `admit` growth (limits enforced;
    /// a refusal writes nothing) and apply the limit-free `adjust`.
    KvSwap {
        ns: Ns,
        key: String,
        expect: Option<String>,
        value: Option<String>,
        #[serde(default)]
        admit: Option<QuotaDelta>,
        #[serde(default)]
        adjust: Option<QuotaDelta>,
        #[serde(default)]
        now: u64,
    },
    /// Chain audit lines (deduplicated by queue message id) into the next
    /// object; returns every object not yet confirmed written.
    AuditAppend { lines: Vec<AuditLine>, now_ms: u64 },
    /// Objects `seqs` are in R2: forget their pending bodies.
    AuditCommit { seqs: Vec<u64> },
    /// The audit chain position.
    AuditHead,
}

/// One audit event for [`M3LedgerCall::AuditAppend`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditLine {
    /// Queue message id (stable across redeliveries).
    pub mid: String,
    /// Event time (Unix ms).
    pub ts_ms: u64,
    /// Event JSON object text.
    pub json: String,
}

/// One shipped audit object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditObject {
    /// Object sequence number (per tenant, from 1).
    pub seq: u64,
    /// R2 key.
    pub key: String,
    /// NDJSON body.
    pub body: String,
}

/// `TenantLedger` M3 results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "out", rename_all = "snake_case")]
pub enum M3LedgerOut {
    /// Write done.
    Done,
    /// For `KvGet`.
    Value { value: Option<String> },
    /// For `KvRange`: `(key, value)`.
    Values { items: Vec<(String, String)> },
    /// For `CreateWithEmbedder`.
    Created { entry: CollectionWire },
    /// For `NextEpoch`.
    Epoch { epoch: u64 },
    /// For `WitnessAppend`.
    Witnessed { entry: EntryWire },
    /// For `Chain`.
    Chain {
        entries: Vec<EntryWire>,
        head: String,
    },
    /// For `AuditAppend`: objects to write (oldest first).
    AuditPending { objects: Vec<AuditObject> },
    /// For `AuditHead`: next object seq and the last line hash (hex).
    AuditHead { next: u64, head: String },
}

/// A `VectorShard` M3 request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct M3ShardRequest {
    /// Tenant key from the verified token.
    pub tenant_key: String,
    /// `collection_uid` (32 hex).
    pub uid: String,
    /// Shard index.
    pub shard: u32,
    /// The call.
    pub call: M3ShardCall,
}

/// `VectorShard` M3 calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum M3ShardCall {
    /// Rows with `id > after` in id order, plus the `write_seq` they were
    /// read at (one DO turn: consistent).
    Page { after: Option<String>, limit: u32 },
    /// Stage restore rows under `rid` (not visible to queries).
    Stage { rid: String, rows: Vec<RowWire> },
    /// Drop the rows staged under `rid`.
    Unstage { rid: String },
    /// Replace the shard's rows with exactly the `rows` staged under `rid`,
    /// in one DO turn, then drop the staging rows.
    Commit {
        rid: String,
        cfg: CfgWire,
        rows: u64,
        actor: ActorWire,
        now: u64,
    },
    /// Settle an interrupted restore: drop the rows staged under `rid` and
    /// return the shard's last commit record if it is `rid`'s (kept, so
    /// settling is idempotent until the ledger records the correction).
    Settle { rid: String },
}

/// `VectorShard` M3 results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "out", rename_all = "snake_case")]
pub enum M3ShardOut {
    /// For `Page`.
    Page {
        write_seq: u64,
        count: u64,
        rows: Vec<RowWire>,
    },
    /// For `Stage` / `Unstage`.
    Done,
    /// For `Commit`: usage removed (≤ 0) and added (≥ 0).
    Committed {
        removed: DeltaWire,
        added: DeltaWire,
        write_seq: u64,
        rows: u64,
    },
    /// For `Settle`: `(removed, added)` if `rid` committed here.
    Settled {
        committed: Option<(DeltaWire, DeltaWire)>,
    },
}

/// Encode a DO reply.
pub fn reply<T: Serialize>(r: Result<T, OpError>) -> String {
    let wire: Reply<T> = r.map_err(|e| crate::wire::WireErr::from_op(&e));
    serde_json::to_string(&wire)
        .unwrap_or_else(|_| String::from(r#"{"Err":{"code":"server_error"}}"#))
}

/// A record in a ledger namespace.
pub async fn kv_get<B: M3Backend>(
    b: &B,
    tenant: &TenantKey,
    ns: Ns,
    key: &str,
) -> Result<Option<String>, OpError> {
    let call = M3LedgerCall::KvGet {
        ns,
        key: key.to_string(),
    };
    match ledger3(b, tenant, call).await? {
        M3LedgerOut::Value { value } => Ok(value),
        _ => Err(crate::service::unexpected()),
    }
}

/// Store a record in a ledger namespace.
pub async fn kv_put<B: M3Backend>(
    b: &B,
    tenant: &TenantKey,
    ns: Ns,
    key: &str,
    value: String,
) -> Result<(), OpError> {
    let call = M3LedgerCall::KvPut {
        ns,
        key: key.to_string(),
        value,
    };
    ledger3(b, tenant, call).await.map(|_| ())
}

/// Records with `from <= key < to`.
pub async fn kv_range<B: M3Backend>(
    b: &B,
    tenant: &TenantKey,
    ns: Ns,
    from: String,
    to: String,
    limit: u32,
) -> Result<Vec<(String, String)>, OpError> {
    let call = M3LedgerCall::KvRange {
        ns,
        from,
        to,
        limit,
    };
    match ledger3(b, tenant, call).await? {
        M3LedgerOut::Values { items } => Ok(items),
        _ => Err(crate::service::unexpected()),
    }
}

/// Compare-and-swap a record (see [`M3LedgerCall::KvSwap`]); `409` when
/// the stored value is not `expect`.
pub async fn kv_swap<B: M3Backend>(
    b: &B,
    tenant: &TenantKey,
    ns: Ns,
    key: &str,
    expect: Option<String>,
    value: Option<String>,
) -> Result<(), OpError> {
    let call = M3LedgerCall::KvSwap {
        ns,
        key: key.to_string(),
        expect,
        value,
        admit: None,
        adjust: None,
        now: 0,
    };
    ledger3(b, tenant, call).await.map(|_| ())
}

/// Text column `i` of a result row.
pub fn col_text(row: &[ruvector_edge_store::Value], i: usize) -> Result<String, OpError> {
    row.get(i)
        .and_then(ruvector_edge_store::Value::as_text)
        .map(str::to_string)
        .ok_or(OpError::new(ErrorCode::ServerError, "stored column"))
}

/// Integer column `i` of a result row.
pub fn col_int(row: &[ruvector_edge_store::Value], i: usize) -> Result<i64, OpError> {
    row.get(i)
        .and_then(ruvector_edge_store::Value::as_int)
        .ok_or(OpError::new(ErrorCode::ServerError, "stored column"))
}
