//! Idempotency keys over the `TenantLedger` (ADR-351 §7 conventions,
//! §16.3): `/v1/ops` `op_id`s and REST `Idempotency-Key`s share one table,
//! scoped `(tenant, sub, key)`, bound to a request hash, kept 24 h.
//!
//! A mutating request **reserves** its key in the same DO turn as the
//! lookup, so two concurrent requests with one key never both execute: the
//! second sees `in_flight` (`409 conflict`, retry shortly). Success replaces
//! the reservation with the stored response; failure releases it (a crash
//! leaves it to expire, `IDEM_PENDING_TTL_SECS`).

use crate::backend::{ledger, Backend};
use crate::service::unexpected;
use crate::wire::{LedgerCall, LedgerOut};
use ruvector_edge_store::{CallerContext, ErrorCode, OpError};

/// Largest `Idempotency-Key` (§7: ≤ 255 bytes).
pub const MAX_KEY_BYTES: usize = 255;
/// `retry_after_s` of an in-flight answer.
pub const IN_FLIGHT_RETRY_S: u32 = 1;

/// A syntactically valid key: 1..=255 visible ASCII bytes.
pub fn key_ok(k: &str) -> bool {
    !k.is_empty() && k.len() <= MAX_KEY_BYTES && k.bytes().all(|b| b.is_ascii_graphic())
}

/// Another request holds the key and has not finished.
pub fn in_flight() -> OpError {
    OpError::new(ErrorCode::Conflict, "request with this key in flight")
}

/// One key of one caller, bound to the request hash.
#[derive(Debug, Clone)]
pub struct Slot {
    /// Caller subject.
    pub sub: String,
    /// Stored key (`op_id`, or the namespaced REST key).
    pub key: String,
    /// sha256 of what the key is bound to.
    pub sha256: [u8; 32],
}

/// What a lookup found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Seen {
    /// Execute (and, if reserved, the key is now held by this request).
    Miss,
    /// Return this stored response.
    Replay(String),
}

/// Look the key up; `reserve` also holds it on a miss. A different body is
/// `reused` (409 `op_replayed`), an unfinished twin [`in_flight`].
pub async fn check<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    s: &Slot,
    reserve: bool,
    reused: &'static str,
    now: u64,
) -> Result<Seen, OpError> {
    let (sub, key, sha256) = (s.sub.clone(), s.key.clone(), s.sha256);
    let call = if reserve {
        LedgerCall::IdemReserve {
            sub,
            key,
            sha256,
            now,
        }
    } else {
        LedgerCall::IdemLookup {
            sub,
            key,
            sha256,
            now,
        }
    };
    match ledger(b, ctx.tenant_key(), call).await? {
        LedgerOut::Idem { conflict: true, .. } => Err(OpError::new(ErrorCode::OpReplayed, reused)),
        LedgerOut::Idem {
            in_flight: true, ..
        } => Err(in_flight()),
        LedgerOut::Idem {
            replay: Some(body), ..
        } => Ok(Seen::Replay(body)),
        LedgerOut::Idem { .. } => Ok(Seen::Miss),
        _ => Err(unexpected()),
    }
}

/// Finish a reserved key: remember `response` (success) or release the
/// reservation (`None`). Best effort: a failure to remember does not undo
/// the executed op, and a failed release expires on its own.
pub async fn finish<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    s: Slot,
    response: Option<String>,
    now: u64,
) {
    let call = match response {
        Some(response) => LedgerCall::IdemStore {
            sub: s.sub,
            key: s.key,
            sha256: s.sha256,
            response,
            now,
        },
        None => LedgerCall::IdemRelease {
            sub: s.sub,
            key: s.key,
            sha256: s.sha256,
        },
    };
    let _done = ledger(b, ctx.tenant_key(), call).await;
}
