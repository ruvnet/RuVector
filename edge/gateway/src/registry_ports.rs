//! Ports of the registry Worker flows: the R2 bucket ([`BlobStore`]) and
//! the transport to the registry Durable Objects ([`RegistryRpc`]). The
//! Worker implementations live in `registry_do` / `blob_r2`; native tests
//! use in-memory doubles (`registry_mem`).

use crate::registry_core::{root_reply, scope_reply};
use crate::registry_pending::{self as pending, PendingOp};
use crate::registry_sweep::SweepWork;
use crate::registry_wire::{Reply, RootCall, RootOut, RvfError, ScopeCall, ScopeOut};
use futures_util::future::join_all;
use ruvector_edge_registry::keys::registry_do_name;
use ruvector_edge_registry::upload::ObjectEvidence;
use ruvector_edge_registry::Scope;

/// R2 operations the registry needs. Errors are storage failures
/// (`RvfError::storage`) unless stated otherwise.
#[allow(async_fn_in_trait)] // Workers futures are !Send.
pub trait BlobStore {
    /// Start a multipart upload at `key`; returns its upload id.
    async fn mp_create(&self, key: &str) -> Result<String, RvfError>;
    /// Upload part `n` of `upload` at `key`; returns the part's etag.
    async fn mp_part(
        &self,
        key: &str,
        upload: &str,
        n: u16,
        bytes: Vec<u8>,
    ) -> Result<String, RvfError>;
    /// Complete `upload` with exactly these (part number, etag) pairs.
    async fn mp_complete(
        &self,
        key: &str,
        upload: &str,
        parts: &[(u16, String)],
    ) -> Result<(), RvfError>;
    /// Abort `upload` (an absent or finished upload is not an error; a
    /// transient failure is).
    async fn mp_abort(&self, key: &str, upload: &str) -> Result<(), RvfError>;
    /// Object size, `None` if absent.
    async fn size(&self, key: &str) -> Result<Option<u64>, RvfError>;
    /// `len` bytes at `offset`, `None` if the object is absent.
    async fn get_range(
        &self,
        key: &str,
        offset: u64,
        len: u64,
    ) -> Result<Option<Vec<u8>>, RvfError>;
    /// Put `bytes` at `key`; R2 refuses unless they hash to `sha256`.
    /// Returns the stored object as R2 reports it (R2 verified the hash,
    /// so no re-read is needed).
    async fn put_checked(
        &self,
        key: &str,
        bytes: Vec<u8>,
        sha256: [u8; 32],
    ) -> Result<ObjectEvidence, RvfError>;
    /// Stream `from` into `to` **create-only** (never overwriting), with R2
    /// checking `sha256`. `Ok(false)` if `to` already existed.
    async fn copy_create_only(
        &self,
        from: &str,
        to: &str,
        sha256: [u8; 32],
    ) -> Result<bool, RvfError>;
    /// Stream `from` into `to` (overwriting: blob keys are content
    /// addressed), with R2 checking `sha256`; returns the stored object as
    /// R2 reports it (R2 verified the hash, so no re-read is needed).
    async fn copy_checked(
        &self,
        from: &str,
        to: &str,
        sha256: [u8; 32],
    ) -> Result<ObjectEvidence, RvfError>;
    /// The object's SHA-256 and size: the SHA-256 R2 verified when the
    /// object was written with one, else measured by streaming it.
    async fn measure(&self, key: &str) -> Result<Option<ObjectEvidence>, RvfError>;
    /// Delete (absent is not an error).
    async fn delete(&self, key: &str) -> Result<(), RvfError>;
}

/// Encoded-body transport to the registry DOs. `Err` means the call's fate
/// is unknown.
#[allow(async_fn_in_trait)]
pub trait RegistryRpc {
    /// POST `body` to the `RegistryRoot` (scope directory).
    async fn call_root(&self, body: String) -> Result<String, RvfError>;
    /// POST `body` to the `RegistryScope` named `do_name`.
    async fn call_scope(&self, do_name: &str, body: String) -> Result<String, RvfError>;
    /// POST one stepped-finalize `body` (`rvf_finalize_step`) to the
    /// `RegistryScope` named `do_name`.
    async fn call_scope_step(&self, do_name: &str, body: String) -> Result<String, RvfError>;
}

/// `idFromName` of the single `RegistryRoot`.
pub const ROOT_DO_NAME: &str = "v1|registry|root";

/// One `RegistryRoot` call.
pub async fn root<R: RegistryRpc>(r: &R, call: RootCall) -> Reply<RootOut> {
    let body = serde_json::to_string(&call).map_err(|_| RvfError::unexpected())?;
    root_reply(&r.call_root(body).await?)
}

/// One `RegistryScope` call to `scope`'s index.
pub async fn scope<R: RegistryRpc>(r: &R, scope: &Scope, call: ScopeCall) -> Reply<ScopeOut> {
    let body = serde_json::to_string(&call).map_err(|_| RvfError::unexpected())?;
    scope_reply(&r.call_scope(&registry_do_name(scope), body).await?)
}

/// R2 operations one alarm issues at most (aborts plus deletes).
pub const OPS_PER_ALARM: usize = 300;
/// R2 operations in flight at once.
pub const SWEEP_CONCURRENCY: usize = 6;
/// `upload/` sessions per `Registry::sweep` page.
pub const SWEEP_PAGE: usize = 200;

/// The body of a `RegistryScope` alarm. The caller must not serve another
/// request while this runs (the DO's `SweepGate`), and `live` turns false
/// once the gate's deadline passed: no further R2 operation is issued then.
///
/// 1. Drain the durable R2 queue left by earlier alarms.
/// 2. Sweep the index, queueing new work in the same commit.
/// 3. Drain again. At most [`OPS_PER_ALARM`] operations in all, run
///    [`SWEEP_CONCURRENCY`] at a time; `remaining` says to re-arm soon.
pub async fn apply_sweep<Q, C, S>(
    sql: Q,
    clock: &C,
    entropy: &dyn ruvector_edge_tenancy::EntropySource,
    cfg: ruvector_edge_registry::registry::RegistryConfig,
    r2: &S,
    live: &dyn Fn() -> bool,
) -> Result<SweepWork, RvfError>
where
    Q: ruvector_edge_store::SqlStore + Copy,
    C: ruvector_edge_auth::Clock,
    S: BlobStore,
{
    let mut work = SweepWork::default();
    let mut budget = OPS_PER_ALARM;
    drain(sql, r2, &mut budget, &mut work, live).await?;
    if budget > 0 && live() {
        work.expired =
            crate::registry_sweep::sweep_scope(sql, clock, entropy, cfg, SWEEP_PAGE, budget)?;
        drain(sql, r2, &mut budget, &mut work, live).await?;
    }
    work.remaining = pending::queued(sql)?;
    Ok(work)
}

async fn run_op<S: BlobStore>(r2: &S, op: &PendingOp) -> Result<(), RvfError> {
    match op {
        PendingOp::Abort { key, multipart, .. } => r2.mp_abort(key, multipart).await,
        PendingOp::Delete { key, .. } => r2.delete(key).await,
    }
}

async fn drain<Q, S>(
    sql: Q,
    r2: &S,
    budget: &mut usize,
    work: &mut SweepWork,
    live: &dyn Fn() -> bool,
) -> Result<(), RvfError>
where
    Q: ruvector_edge_store::SqlStore + Copy,
    S: BlobStore,
{
    let ops = pending::take(sql, *budget)?;
    for chunk in ops.chunks(SWEEP_CONCURRENCY) {
        if !live() {
            break;
        }
        let mut due = Vec::with_capacity(chunk.len());
        for op in chunk {
            if pending::still_due(sql, op)? {
                due.push(op);
            }
        }
        let results = join_all(due.iter().map(|op| run_op(r2, op))).await;
        for (op, res) in due.into_iter().zip(results) {
            pending::settle(sql, op, res.is_ok())?;
            match (op, res.is_ok()) {
                (PendingOp::Abort { key, multipart, .. }, true) => {
                    work.abort.push((key.clone(), multipart.clone()))
                }
                (PendingOp::Delete { key, .. }, true) => work.delete.push(key.clone()),
                _ => {}
            }
        }
        *budget = budget.saturating_sub(chunk.len());
    }
    Ok(())
}
