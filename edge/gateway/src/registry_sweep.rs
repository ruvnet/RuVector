//! The `RegistryScope` maintenance sweep (ADR-351 §3 rv-registry, M5):
//! the index side ([`sweep_scope`]) and the gate ([`SweepGate`]) that keeps
//! requests out of the DO while an alarm awaits R2.

use crate::registry_core::err;
use crate::registry_kv::{EntropyRef, SqlKv};
use crate::registry_pending::{enqueue_abort, enqueue_delete};
use crate::registry_upload_core::{k_mp, load_mp};
use crate::registry_wire::RvfError;
use core::cell::Cell;
use ruvector_edge_auth::Clock;
use ruvector_edge_registry::ports::KvStore;
use ruvector_edge_registry::registry::RegistryConfig;
use ruvector_edge_registry::upload::UploadId;
use ruvector_edge_registry::Registry;
use ruvector_edge_store::SqlStore;
use ruvector_edge_tenancy::EntropySource;

/// Where the next sweep resumes its walk over `upload/`.
pub const K_CURSOR: &str = "gw/sweep_cursor";
/// Most `Registry::sweep` pages one alarm walks.
pub const MAX_PAGES: usize = 10;

/// What one alarm did in R2.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SweepWork {
    /// Objects deleted (released blobs, staging objects of ended sessions).
    pub delete: Vec<String>,
    /// Multipart uploads aborted: (staging key, R2 multipart upload id).
    pub abort: Vec<(String, String)>,
    /// Sessions that expired unfinished (now failed).
    pub expired: Vec<String>,
    /// Queued R2 work is left for another alarm (re-arm soon).
    pub remaining: bool,
}

fn staging_id(key: &str) -> Option<UploadId> {
    key.strip_prefix("staging/")
        .and_then(|rest| rest.rsplit('/').next())
        .and_then(|id| UploadId::parse(id).ok())
}

/// One sweep pass over a `RegistryScope` index (inside the DO's alarm, one
/// synchronous step): runs `Registry::sweep` page by page from the stored
/// cursor until `max_ops` R2 operations are queued, at most [`MAX_PAGES`]
/// pages of `page` sessions, or the end (the cursor then wraps). Every key
/// the sweep hands out is queued as a pending delete, and every staging
/// multipart still on record as a pending abort, in the same commit that
/// drops its `gw/mp` record. Returns the sessions that expired.
pub fn sweep_scope<S: SqlStore, C: Clock>(
    sql: S,
    clock: &C,
    entropy: &dyn EntropySource,
    cfg: RegistryConfig,
    page: usize,
    max_ops: usize,
) -> Result<Vec<String>, RvfError> {
    let kv = SqlKv::open(sql).map_err(|e| err(e.into()))?;
    let reg = Registry::new(&kv, clock, EntropyRef(entropy), cfg);
    let mut after = kv
        .get(K_CURSOR)
        .map_err(|e| err(e.into()))?
        .and_then(|b| String::from_utf8(b).ok());
    let (mut queued, mut expired) = (0usize, Vec::new());
    for _ in 0..MAX_PAGES {
        let r = reg.sweep(after.as_deref(), page.max(1)).map_err(err)?;
        expired.extend(r.expired.iter().map(|u| u.as_str().to_string()));
        for key in r.delete {
            let key = key.to_string();
            if let Some(id) = staging_id(&key) {
                if let Some(mp) = load_mp(&kv, &id)? {
                    if !mp.multipart.is_empty() {
                        enqueue_abort(&kv, &id, &key, &mp.multipart)?;
                        queued += 1;
                    }
                    kv.delete(&k_mp(&id)).map_err(|e| err(e.into()))?;
                }
            }
            enqueue_delete(&kv, &key)?;
            queued += 1;
        }
        after = r.next;
        if after.is_none() || queued >= max_ops {
            break;
        }
    }
    match &after {
        Some(a) => kv.put(K_CURSOR, a.as_bytes()),
        None => kv.delete(K_CURSOR),
    }
    .map_err(|e| err(e.into()))?;
    Ok(expired)
}

/// Longest a sweep may hold the gate: well above one alarm's operation
/// budget, well below the 15-minute alarm limit. A sweep whose future was
/// abandoned (never resumed, never dropped) stops refusing requests then.
pub const SWEEP_DEADLINE_MS: u64 = 120_000;

/// Refuses requests while an alarm sweep awaits R2.
#[derive(Debug, Default)]
pub struct SweepGate {
    started: Cell<Option<u64>>,
}

impl SweepGate {
    /// `true` while a sweep that started less than
    /// [`SWEEP_DEADLINE_MS`] ago holds the gate.
    pub fn busy(&self, now_ms: u64) -> bool {
        self.started
            .get()
            .is_some_and(|t| now_ms.saturating_sub(t) < SWEEP_DEADLINE_MS)
    }

    /// Take the gate for a sweep; dropping the guard releases it.
    pub fn enter(&self, now_ms: u64) -> GateGuard<'_> {
        self.started.set(Some(now_ms));
        GateGuard {
            gate: self,
            at: now_ms,
        }
    }
}

/// A held [`SweepGate`].
#[derive(Debug)]
pub struct GateGuard<'a> {
    gate: &'a SweepGate,
    at: u64,
}

impl GateGuard<'_> {
    /// `true` while this sweep still holds the gate within its deadline
    /// (a sweep past it must stop issuing R2 operations).
    pub fn live(&self, now_ms: u64) -> bool {
        self.gate.started.get() == Some(self.at) && self.gate.busy(now_ms)
    }
}

impl Drop for GateGuard<'_> {
    fn drop(&mut self) {
        // A newer sweep's hold is not ours to release.
        if self.gate.started.get() == Some(self.at) {
            self.gate.started.set(None);
        }
    }
}
