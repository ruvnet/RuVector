//! `POST /v1/ops` (ADR-351 §16.3) over the Durable Object backend.
//!
//! Check order, as the store's in-process dispatcher: body size → envelope
//! shape → `Idempotency-Key` = `op_id` → `target` → `tenant_key` echo →
//! known op → scope → role → `op_id` reserve (replay / conflict /
//! in flight) → execute → remember or release.
//! Authorization precedes the replay lookup, so a replay is re-authorized.
//! The `op_id` binding hashes the raw body bytes; only successful,
//! non-dry-run responses of the mutating ops are remembered. The route
//! only admits tokens whose `aud` is exactly the `…/v1` resource (checked
//! by `auth::authenticate` before this runs).

use crate::backend::Backend;
use crate::idem::{self, Seen, Slot};
use crate::service::{authorize_op, run, Call};
use ruvector_edge_store::ops::{Op, OpRequest, OpResponse};
use ruvector_edge_store::{CallerContext, ErrorCode, OpError};
use ruvector_edge_tenancy::quota::limits::MAX_UPSERT_BYTES;
use sha2::{Digest, Sha256};

/// Largest `/v1/ops` body (the §10 1 MiB limit).
pub const MAX_OPS_BODY_BYTES: usize = MAX_UPSERT_BYTES as usize;

/// HTTP-ready reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpsReply {
    /// HTTP status.
    pub status: u16,
    /// `OpResponse` JSON (verbatim stored bytes on a replay).
    pub body: String,
    /// Served from the idempotency store.
    pub replayed: bool,
    /// For `insufficient_scope`: the scope a client must request.
    pub step_up: Option<&'static str>,
}

fn reply(r: &OpResponse, step_up: Option<&'static str>) -> OpsReply {
    let body = serde_json::to_string(r).unwrap_or_else(|_| {
        String::from(
            r#"{"v":1,"op_id":"","ok":false,"error":{"code":"server_error","status":500}}"#,
        )
    });
    OpsReply {
        status: r.status(),
        body,
        replayed: false,
        step_up,
    }
}

fn fail(op_id: &str, e: &OpError) -> OpsReply {
    let step_up = (e.code == ErrorCode::InsufficientScope)
        .then_some(e.scope)
        .flatten();
    reply(&OpResponse::err(op_id, e), step_up)
}

fn remembered(op: Op) -> bool {
    matches!(
        op,
        Op::CollectionCreate | Op::VectorUpsert | Op::VectorDelete
    )
}

/// Handle one `/v1/ops` body for a verified caller. `target` is this
/// route's canonical URL; `idempotency_key` the header, if sent.
pub async fn dispatch<B: Backend>(
    b: &B,
    target: &str,
    ctx: &CallerContext,
    body: &[u8],
    idempotency_key: Option<&str>,
    now: u64,
) -> OpsReply {
    if body.len() > MAX_OPS_BODY_BYTES {
        return fail(
            "",
            &OpError::new(ErrorCode::PayloadTooLarge, "body too large"),
        );
    }
    let req: OpRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(_) => return fail("", &OpError::invalid("malformed envelope")),
    };
    if let Err(e) = req.validate_shape() {
        return fail("", &e);
    }
    let op_id = req.op_id.as_str();
    if idempotency_key.is_some_and(|k| k != op_id) {
        return fail(op_id, &OpError::invalid("Idempotency-Key must equal op_id"));
    }
    if req.target != target {
        return fail(
            op_id,
            &OpError::new(ErrorCode::TargetMismatch, "target mismatch"),
        );
    }
    if req.tenant_key != ctx.tenant_key().as_str() {
        return fail(
            op_id,
            &OpError::new(ErrorCode::TenantMismatch, "tenant mismatch"),
        );
    }
    let Some(op) = Op::parse(&req.op) else {
        return fail(op_id, &OpError::new(ErrorCode::UnknownOp, "unknown op"));
    };
    let sha256: [u8; 32] = Sha256::digest(body).into();
    match authorized(b, ctx, &req, op, sha256, now).await {
        Ok(r) => r,
        Err(e) => fail(op_id, &e),
    }
}

async fn authorized<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    req: &OpRequest,
    op: Op,
    sha256: [u8; 32],
    now: u64,
) -> Result<OpsReply, OpError> {
    let access = authorize_op(b, ctx, op).await?;
    let slot = Slot {
        sub: ctx.sub().to_string(),
        key: req.op_id.clone(),
        sha256,
    };
    // Mutating, non-dry-run ops reserve the op_id in the lookup's DO turn,
    // so a concurrent twin gets 409 instead of executing a second time.
    let reserve = !req.dry_run && remembered(op);
    match idem::check(b, ctx, &slot, reserve, "op_id reused", now).await {
        Ok(Seen::Replay(body)) => {
            let status = serde_json::from_str::<OpResponse>(&body).map_or(500, |r| r.status());
            return Ok(OpsReply {
                status,
                body,
                replayed: true,
                step_up: None,
            });
        }
        Ok(Seen::Miss) => {}
        Err(e) if e.code == ErrorCode::Conflict => {
            let mut r = OpResponse::err(&req.op_id, &e);
            if let Some(w) = r.error.as_mut() {
                w.retry_after_s = Some(idem::IN_FLIGHT_RETRY_S);
            }
            return Ok(reply(&r, None));
        }
        Err(e) => return Err(e),
    }
    let call = Call {
        b,
        ctx,
        dry_run: req.dry_run,
        now,
    };
    let out = match run(&call, access, op, req.args_text()).await {
        Ok((result, usage)) => reply(&OpResponse::ok(&req.op_id, result, usage), None),
        Err(e) => {
            let step_up = (e.code == ErrorCode::InsufficientScope)
                .then_some(e.scope)
                .flatten();
            reply(&OpResponse::err(&req.op_id, &e), step_up)
        }
    };
    if reserve {
        // Only successes are remembered; a failure releases the op_id.
        let ok = out.status < 300;
        idem::finish(b, ctx, slot, ok.then(|| out.body.clone()), now).await;
    }
    Ok(out)
}
