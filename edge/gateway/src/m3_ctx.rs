//! Request context shared by the M3 executors, the M3 authorization rule
//! (scope first, then role — the same order as `ops::authorize`) and the
//! mapping of the snapshot crate's typed refusals to stable problem codes.

use crate::audit::Note;
use crate::m3_ports::{Blob, Queues};
use crate::m3_wire::M3Backend;
use crate::service::{self, Access, Call};
use crate::wire::CollectionWire;
use ruvector_edge_auth::Capability;
use ruvector_edge_snapshot::{Metric as SnapMetric, SnapshotError};
use ruvector_edge_store::{CallerContext, ErrorCode, Metric, OpError};
use ruvector_edge_tenancy::Role;
use serde_json::Value as Json;

/// One M3 request: transports, the verified caller and the clock.
pub struct M3<'a, B, R, Q> {
    /// DO transport (M1 + M3 side channel).
    pub b: &'a B,
    /// R2.
    pub blob: &'a R,
    /// Queues producers.
    pub queues: &'a Q,
    /// Verified caller.
    pub ctx: &'a CallerContext,
    /// Unix milliseconds.
    pub now_ms: u64,
    /// What the request did (scope, role, rows, bytes, work units) for its
    /// audit event.
    pub note: Note,
}

impl<'a, B: M3Backend, R: Blob, Q: Queues> M3<'a, B, R, Q> {
    /// The M1 executor context.
    pub fn call(&self) -> Call<'a, B> {
        Call {
            b: self.b,
            ctx: self.ctx,
            dry_run: false,
            now: self.now_ms / 1000,
        }
    }

    /// Scope first (`insufficient_scope` with the step-up scope), then the
    /// ledger role (`not_claimed` / `role_required`).
    pub async fn require(&self, cap: Capability, min: Role) -> Result<Access, OpError> {
        self.note.scope.set(Some(cap.satisfying_scope()));
        if !self.ctx.scope_caps().contains(cap) {
            return Err(OpError {
                code: ErrorCode::InsufficientScope,
                detail: "insufficient scope",
                scope: Some(cap.satisfying_scope()),
            });
        }
        let a = service::access(self.b, self.ctx).await?;
        match a.role {
            None if !a.claimed => Err(OpError::new(ErrorCode::NotClaimed, "tenant not claimed")),
            None => Err(OpError::new(ErrorCode::RoleRequired, "membership required")),
            Some(r) if r < min => Err(OpError::new(ErrorCode::RoleRequired, "role required")),
            Some(r) => {
                self.note.role.set(Some(r));
                Ok(a)
            }
        }
    }

    /// A live collection of the caller's tenant (`404` otherwise).
    pub async fn collection(&self, name: &str) -> Result<CollectionWire, OpError> {
        service::lookup(&self.call(), name).await
    }

    /// Admit `work_units` (and one op) before doing the work.
    pub async fn charge(&self, work_units: u64) -> Result<(), OpError> {
        service::charge(&self.call(), service::one_op(), work_units).await?;
        self.note.charged(work_units);
        Ok(())
    }
}

/// The snapshot crate's metric for a collection metric.
pub fn snap_metric(m: Metric) -> SnapMetric {
    match m {
        Metric::Cosine => SnapMetric::Cosine,
        Metric::L2 => SnapMetric::L2,
        Metric::Dot => SnapMetric::Dot,
    }
}

/// Collection dimension as the snapshot crate's `u16`.
pub fn dim16(e: &CollectionWire) -> Result<u16, OpError> {
    u16::try_from(e.cfg.dim).map_err(|_| OpError::invalid("dimension"))
}

/// Snapshot / restore refusal → problem code. A manifest naming another
/// tenant is `403 tenant_mismatch`; every integrity failure (root, chain,
/// chunk size / sha256 / segment, order, counts, signature) is `409
/// conflict`; limits are `413`.
pub fn snap_err(e: SnapshotError) -> OpError {
    use SnapshotError as S;
    match e {
        S::TenantMismatch => OpError::new(ErrorCode::TenantMismatch, "snapshot tenant mismatch"),
        S::CollectionMismatch | S::ShardMismatch => {
            OpError::new(ErrorCode::Conflict, "snapshot target mismatch")
        }
        S::DimensionMismatch => OpError::new(ErrorCode::DimensionMismatch, "snapshot dimension"),
        S::QuotaExceeded { .. } => OpError::new(ErrorCode::QuotaExceeded, "snapshot quota"),
        S::InvalidIdentifier(_) | S::InvalidRow { .. } => {
            OpError::new(ErrorCode::Conflict, "snapshot row refused")
        }
        S::NotInChain | S::ChainBreak { .. } | S::ChainHeadMismatch => {
            OpError::new(ErrorCode::Conflict, "snapshot not witnessed")
        }
        S::SignatureMissing | S::SignatureInvalid | S::SignerFailed(_) => {
            OpError::new(ErrorCode::Conflict, "snapshot signature")
        }
        _ => OpError::new(ErrorCode::Conflict, "snapshot integrity"),
    }
}

/// Parse a JSON object body (empty = `{}`).
pub fn object(body: &[u8]) -> Result<serde_json::Map<String, Json>, OpError> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(serde_json::Map::new());
    }
    serde_json::from_slice(body).map_err(|_| OpError::invalid("body must be a JSON object"))
}

/// Lowercase hex id of `n` bytes of `sha256(parts)` (minted ids).
pub fn mint(parts: &[&[u8]], n: usize) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for p in parts {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p);
    }
    let d: [u8; 32] = h.finalize().into();
    d[..n.min(32)].iter().map(|b| format!("{b:02x}")).collect()
}
