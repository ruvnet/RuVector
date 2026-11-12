//! `TenantLedger` side of shipped audit (ADR-351 §6.3 `audit/`, §6.5): one
//! hash chain per tenant across every shipped object.
//!
//! `AuditAppend` (one DO turn) drops lines whose queue message id was
//! already chained, gives the rest the next object `seq`, chains them from
//! the stored head (`hash = sha256(prev ‖ event json)`, the first line's
//! `prev` being the previous object's last hash) and keeps the NDJSON body
//! **pending** until `AuditCommit` confirms the R2 write. Every append
//! returns all pending objects, so a consumer that died after the append
//! but before the PUT rewrites the same key with the same bytes on the
//! redelivery, whatever the batch composition. Deleting or replacing a
//! whole object breaks the `seq` sequence and the `prev` link of the next.
//!
//! Tables: `m3_audit_seen(mid, ts)` (message ids, kept
//! [`SEEN_RETENTION_MS`], longer than the queue's message retention),
//! `m3_audit_pending(seq, key, body)`; meta `audit_head`, `audit_next`.

use crate::m3_ledger::{db, meta_get, meta_put};
use crate::m3_wire::{hex32, unhex32, AuditLine, AuditObject, M3LedgerOut};
use ruvector_edge_store::{ErrorCode, OpError, SqlStore, Value};

const SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS m3_audit_seen (mid TEXT PRIMARY KEY, ts INTEGER)",
    "CREATE TABLE IF NOT EXISTS m3_audit_pending (seq INTEGER PRIMARY KEY, key TEXT, body TEXT)",
];
const SEEN_GET: &str = "SELECT mid FROM m3_audit_seen WHERE mid = ?";
const SEEN_PUT: &str = "INSERT OR REPLACE INTO m3_audit_seen (mid, ts) VALUES (?, ?)";
const SEEN_PRUNE: &str = "DELETE FROM m3_audit_seen WHERE ts < ?";
const PENDING_PUT: &str = "INSERT INTO m3_audit_pending (seq, key, body) VALUES (?, ?, ?)";
const PENDING_ALL: &str = "SELECT seq, key, body FROM m3_audit_pending ORDER BY seq LIMIT ?";
const PENDING_DROP: &str = "DELETE FROM m3_audit_pending WHERE seq = ?";

/// How long a chained message id is remembered (queue retention is 4 d).
pub const SEEN_RETENTION_MS: u64 = 5 * 86_400_000;
/// Most lines per append (the consumer batch is 100).
pub const MAX_LINES: usize = 500;
/// Largest event JSON.
pub const MAX_LINE_BYTES: usize = 4096;

fn schema(store: &dyn SqlStore) -> Result<(), OpError> {
    for ddl in SCHEMA {
        store.exec(ddl, &[]).map_err(db)?;
    }
    Ok(())
}

/// `(next seq, head)`; a fresh chain is `(1, 32 zero bytes)`.
fn position(store: &dyn SqlStore) -> Result<(u64, [u8; 32]), OpError> {
    let bad = || OpError::new(ErrorCode::ServerError, "audit chain state");
    match (
        meta_get(store, "audit_next")?,
        meta_get(store, "audit_head")?,
    ) {
        (Some(n), Some(h)) => Ok((n.parse().map_err(|_| bad())?, unhex32(&h)?)),
        _ => Ok((1, [0u8; 32])),
    }
}

/// The R2 key of object `seq` (UTC day of its first line).
pub fn object_key(tenant: &str, first_ts_ms: u64, seq: u64) -> String {
    let (y, m, d) = crate::audit::ymd(first_ts_ms);
    format!("audit/{tenant}/{y:04}/{m:02}/{d:02}/{seq:012}.ndjson")
}

/// One chained line: the event object plus `seq`, `prev`, `hash`.
pub fn chain_line(json: &str, seq: u64, prev: &[u8; 32]) -> Result<(String, [u8; 32]), OpError> {
    use sha2::{Digest, Sha256};
    let t = json.trim_end();
    if !t.starts_with('{') || !t.ends_with('}') || t.len() < 2 {
        return Err(OpError::invalid("audit line"));
    }
    let mut h = Sha256::new();
    h.update(prev);
    h.update(t.as_bytes());
    let hash: [u8; 32] = h.finalize().into();
    let sep = if t.len() == 2 { "" } else { "," };
    let line = format!(
        "{}{sep}\"seq\":{seq},\"prev\":\"{}\",\"hash\":\"{}\"}}\n",
        &t[..t.len() - 1],
        hex32(prev),
        hex32(&hash)
    );
    Ok((line, hash))
}

fn pending(store: &dyn SqlStore) -> Result<Vec<AuditObject>, OpError> {
    let mut out = Vec::new();
    for r in store.query(PENDING_ALL, &[Value::Int(64)]).map_err(db)? {
        let seq = crate::m3_wire::col_int(&r, 0)?;
        out.push(AuditObject {
            seq: u64::try_from(seq).map_err(|_| OpError::invalid("seq"))?,
            key: crate::m3_wire::col_text(&r, 1)?,
            body: crate::m3_wire::col_text(&r, 2)?,
        });
    }
    Ok(out)
}

/// `AuditAppend` (see the module docs).
pub fn append(
    store: &dyn SqlStore,
    tenant: &str,
    lines: Vec<AuditLine>,
    now_ms: u64,
) -> Result<M3LedgerOut, OpError> {
    schema(store)?;
    if lines.len() > MAX_LINES
        || lines
            .iter()
            .any(|l| l.json.len() > MAX_LINE_BYTES || l.mid.is_empty() || l.mid.len() > 128)
    {
        return Err(OpError::invalid("audit lines"));
    }
    let cutoff = now_ms.saturating_sub(SEEN_RETENTION_MS);
    let cutoff = i64::try_from(cutoff).unwrap_or(i64::MAX);
    store.exec(SEEN_PRUNE, &[Value::Int(cutoff)]).map_err(db)?;
    let mut fresh = Vec::with_capacity(lines.len());
    for l in lines {
        let seen = store
            .query(SEEN_GET, &[l.mid.as_str().into()])
            .map_err(db)?;
        if seen.is_empty() && !fresh.iter().any(|f: &AuditLine| f.mid == l.mid) {
            fresh.push(l);
        }
    }
    if let Some(first) = fresh.first() {
        let (seq, mut prev) = position(store)?;
        let key = object_key(tenant, first.ts_ms, seq);
        let mut body = String::new();
        for l in &fresh {
            let (line, hash) = chain_line(&l.json, seq, &prev)?;
            body.push_str(&line);
            prev = hash;
        }
        let seq_i = i64::try_from(seq).map_err(|_| OpError::invalid("seq"))?;
        let now_i = i64::try_from(now_ms).unwrap_or(i64::MAX);
        for l in &fresh {
            let p = [l.mid.as_str().into(), Value::Int(now_i)];
            store.exec(SEEN_PUT, &p).map_err(db)?;
        }
        let p = [Value::Int(seq_i), key.into(), body.into()];
        store.exec(PENDING_PUT, &p).map_err(db)?;
        meta_put(store, "audit_head", hex32(&prev))?;
        meta_put(store, "audit_next", (seq + 1).to_string())?;
    }
    Ok(M3LedgerOut::AuditPending {
        objects: pending(store)?,
    })
}

/// `AuditCommit`.
pub fn commit(store: &dyn SqlStore, seqs: Vec<u64>) -> Result<M3LedgerOut, OpError> {
    schema(store)?;
    for s in seqs {
        let s = i64::try_from(s).map_err(|_| OpError::invalid("seq"))?;
        store.exec(PENDING_DROP, &[Value::Int(s)]).map_err(db)?;
    }
    Ok(M3LedgerOut::Done)
}

/// `AuditHead`.
pub fn head(store: &dyn SqlStore) -> Result<M3LedgerOut, OpError> {
    let (next, head) = position(store)?;
    Ok(M3LedgerOut::AuditHead {
        next,
        head: hex32(&head),
    })
}
