//! The `RegistryScope` sweep's durable R2 work queue (ADR-351 §3
//! rv-registry, M5).
//!
//! The sweep changes the index and records every R2 side effect it implies
//! in the **same synchronous step** (one coalesced commit): a pending abort
//! `gw/pa/{upload_id}` for each staging multipart, a pending delete
//! `gw/pd/{object key}` for each staging object and released blob. A row
//! goes only after R2 confirms the operation, so an alarm that dies half
//! way (CPU or subrequest limit, eviction, deploy) or an R2 failure leaves
//! the rest queued for the next alarm, which drains the queue first.
//!
//! A pending delete of a released blob is re-checked against `blob/{key}`
//! right before it is issued (under the sweep gate, so no request
//! interleaves): a finalize that re-pinned the bytes since keeps them.

use crate::registry_core::err;
use crate::registry_kv::SqlKv;
use crate::registry_wire::RvfError;
use ruvector_edge_registry::ports::KvStore;
use ruvector_edge_registry::upload::UploadId;
use ruvector_edge_store::SqlStore;
use serde::{Deserialize, Serialize};

/// Pending deletes: `gw/pd/{object key}`.
pub const PD: &str = "gw/pd/";
/// Pending multipart aborts: `gw/pa/{upload_id}`.
pub const PA: &str = "gw/pa/";
/// Failed attempts after which an abort is given up (R2's lifecycle rule
/// aborts incomplete multipart uploads on its own).
pub const MAX_ABORT_TRIES: u32 = 8;

/// A queued multipart abort.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct AbortRow {
    key: String,
    multipart: String,
    #[serde(default)]
    tries: u32,
}

/// One queued R2 operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PendingOp {
    /// Abort multipart `multipart` of object `key`.
    Abort {
        /// Queue row.
        row: String,
        /// Staging key.
        key: String,
        /// R2 multipart upload id.
        multipart: String,
        /// Failed attempts so far.
        tries: u32,
    },
    /// Delete object `key`.
    Delete {
        /// Queue row.
        row: String,
        /// Object key.
        key: String,
    },
}

fn se(e: ruvector_edge_store::StoreError) -> RvfError {
    err(e.into())
}

/// Queue a delete of `key`.
pub fn enqueue_delete<K: KvStore>(kv: &K, key: &str) -> Result<(), RvfError> {
    kv.put(&format!("{PD}{key}"), b"{}").map_err(se)
}

/// Queue an abort of `multipart` at `key` for session `id`.
pub fn enqueue_abort<K: KvStore>(
    kv: &K,
    id: &UploadId,
    key: &str,
    multipart: &str,
) -> Result<(), RvfError> {
    let row = AbortRow {
        key: key.to_string(),
        multipart: multipart.to_string(),
        tries: 0,
    };
    let b = serde_json::to_vec(&row).map_err(|_| RvfError::unexpected())?;
    kv.put(&format!("{PA}{}", id.as_str()), &b).map_err(se)
}

/// `true` while any R2 operation is queued.
pub fn queued<S: SqlStore>(sql: S) -> Result<bool, RvfError> {
    let kv = SqlKv::open(sql).map_err(se)?;
    Ok(kv.count_keys(PA, 1).map_err(se)? + kv.count_keys(PD, 1).map_err(se)? > 0)
}

/// Up to `n` queued operations, aborts first.
pub fn take<S: SqlStore>(sql: S, n: usize) -> Result<Vec<PendingOp>, RvfError> {
    let kv = SqlKv::open(sql).map_err(se)?;
    let mut out = Vec::new();
    for (row, v) in kv.list(PA, None, n).map_err(se)? {
        match serde_json::from_slice::<AbortRow>(&v) {
            Ok(a) => out.push(PendingOp::Abort {
                row,
                key: a.key,
                multipart: a.multipart,
                tries: a.tries,
            }),
            // Unreadable: nothing to act on.
            Err(_) => kv.delete(&row).map_err(se)?,
        }
    }
    let rest = n.saturating_sub(out.len());
    for (row, _) in kv.list(PD, None, rest).map_err(se)? {
        let key = row[PD.len()..].to_string();
        out.push(PendingOp::Delete { row, key });
    }
    Ok(out)
}

/// `false` (and the row dropped) for a delete whose blob was pinned again
/// since it was queued. Call right before issuing the operation.
pub fn still_due<S: SqlStore>(sql: S, op: &PendingOp) -> Result<bool, RvfError> {
    let PendingOp::Delete { row, key } = op else {
        return Ok(true);
    };
    let kv = SqlKv::open(sql).map_err(se)?;
    if kv.get(&format!("blob/{key}")).map_err(se)?.is_some() {
        kv.delete(row).map_err(se)?;
        return Ok(false);
    }
    Ok(true)
}

/// Record the outcome: a confirmed operation leaves the queue; a failed
/// delete stays for the next alarm; a failed abort stays until it has
/// failed [`MAX_ABORT_TRIES`] times.
pub fn settle<S: SqlStore>(sql: S, op: &PendingOp, ok: bool) -> Result<(), RvfError> {
    let kv = SqlKv::open(sql).map_err(se)?;
    match op {
        PendingOp::Delete { row, .. } if ok => kv.delete(row).map_err(se),
        PendingOp::Delete { .. } => Ok(()),
        PendingOp::Abort { row, .. } if ok => kv.delete(row).map_err(se),
        PendingOp::Abort {
            row,
            key,
            multipart,
            tries,
        } => {
            if tries + 1 >= MAX_ABORT_TRIES {
                return kv.delete(row).map_err(se);
            }
            let a = AbortRow {
                key: key.clone(),
                multipart: multipart.clone(),
                tries: tries + 1,
            };
            let b = serde_json::to_vec(&a).map_err(|_| RvfError::unexpected())?;
            kv.put(row, &b).map_err(se)
        }
    }
}
