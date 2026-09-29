//! `POST /v1/ops` dispatcher (ADR-351 §16.3).
//!
//! Check order: body size → envelope shape → `Idempotency-Key` = `op_id`
//! → `target` → `tenant_key` echo → known op → scope → role → `op_id`
//! replay → execute → remember the response. Authorization precedes the
//! replay lookup, so a replay is re-authorized (roles can be revoked).
//!
//! The body is parsed once, straight from bytes; `args` stays raw JSON text
//! until the executor parses it into its typed struct, so no `Value` tree
//! of the vectors is ever built. The `op_id` binding hashes the raw body
//! bytes (§16.3 "same body"): a retry must resend the same bytes.
//!
//! **What is remembered** (bounded growth, §6.1): only successful,
//! non-dry-run responses of the mutating ops (`collection_create`,
//! `vector_upsert`, `vector_delete`). Reads are idempotent by nature;
//! errors (including `413` quota/budget, which clear when capacity is
//! freed) and dry runs re-execute on retry, so a preview can be followed by
//! the real call under the same `op_id`. Stored rows are charged to the
//! tenant's `bytes` quota; over quota the response is not remembered and a
//! retry re-executes (safe for upsert/delete by id; a retried create then
//! answers `409 conflict`).

use super::authz::authorize;
use super::cluster::LocalCluster;
use super::exec::{self, Call};
use super::types::{malformed, Op, OpRequest, OpResponse};
use super::vector;
use crate::context::{ledger_meta_for, CallerContext};
use crate::error::{ErrorCode, OpError};
use crate::ledger::{IdemKey, IdemLookup};
use crate::ports::{Clock, EntropySource, SqlStore};
use ruvector_edge_tenancy::quota::limits::MAX_UPSERT_BYTES;
use serde_json::json;
use sha2::{Digest, Sha256};

/// Largest `/v1/ops` body accepted (the §10 1 MiB upsert limit), enforced
/// here as well as by the gateway's pre-auth `Content-Length` check.
pub const MAX_OPS_BODY_BYTES: usize = MAX_UPSERT_BYTES as usize;

/// HTTP-ready reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpReply {
    /// HTTP status.
    pub status: u16,
    /// `OpResponse` JSON (verbatim stored bytes on a replay).
    pub body: String,
    /// `true` when served from the idempotency store.
    pub replayed: bool,
}

impl OpReply {
    fn from_response(r: &OpResponse) -> Self {
        let body = serde_json::to_string(r).unwrap_or_else(|_| {
            String::from(
                r#"{"v":1,"op_id":"","ok":false,"error":{"code":"server_error","status":500}}"#,
            )
        });
        OpReply {
            status: r.status(),
            body,
            replayed: false,
        }
    }
}

/// `true` for ops whose successful responses are remembered under `op_id`.
fn remembered(op: Op) -> bool {
    matches!(
        op,
        Op::CollectionCreate | Op::VectorUpsert | Op::VectorDelete
    )
}

/// Dispatcher bound to this route's canonical URL.
#[derive(Debug, Clone)]
pub struct Dispatcher {
    target: String,
}

impl Dispatcher {
    /// `target` is the canonical `…/v1/ops` URL every request must name.
    pub fn new(target: impl Into<String>) -> Self {
        Dispatcher {
            target: target.into(),
        }
    }

    /// Handle one request body for an already-verified caller.
    /// `idempotency_key` is the `Idempotency-Key` header, if sent.
    pub fn dispatch<S: SqlStore + Default>(
        &self,
        cluster: &mut LocalCluster<S>,
        ctx: &CallerContext,
        body: &[u8],
        idempotency_key: Option<&str>,
        clock: &dyn Clock,
        entropy: &dyn EntropySource,
    ) -> OpReply {
        if body.len() > MAX_OPS_BODY_BYTES {
            let e = OpError::new(ErrorCode::PayloadTooLarge, "body too large");
            return OpReply::from_response(&OpResponse::err("", &e));
        }
        let req: OpRequest = match serde_json::from_slice(body) {
            Ok(r) => r,
            Err(_) => return OpReply::from_response(&OpResponse::err("", &malformed())),
        };
        if let Err(e) = req.validate_shape() {
            return OpReply::from_response(&OpResponse::err("", &e));
        }
        let op_id = req.op_id.as_str();
        let fail = |e: OpError| OpReply::from_response(&OpResponse::err(op_id, &e));
        if idempotency_key.is_some_and(|k| k != op_id) {
            return fail(OpError::invalid("Idempotency-Key must equal op_id"));
        }
        if req.target != self.target {
            return fail(OpError::new(ErrorCode::TargetMismatch, "target mismatch"));
        }
        if req.tenant_key != ctx.tenant_key().as_str() {
            return fail(OpError::new(ErrorCode::TenantMismatch, "tenant mismatch"));
        }
        let Some(op) = Op::parse(&req.op) else {
            return fail(OpError::new(ErrorCode::UnknownOp, "unknown op"));
        };
        let now = clock.now_unix();
        let hash: [u8; 32] = Sha256::digest(body).into();
        match self.authorized(cluster, ctx, &req, op, &hash, now, entropy) {
            Ok(reply) => reply,
            Err(e) => fail(e),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn authorized<S: SqlStore + Default>(
        &self,
        cluster: &mut LocalCluster<S>,
        ctx: &CallerContext,
        req: &OpRequest,
        op: Op,
        hash: &[u8; 32],
        now: u64,
        entropy: &dyn EntropySource,
    ) -> Result<OpReply, OpError> {
        let lm = ledger_meta_for(ctx.tenant_key())?;
        let (role, claimed) = {
            let (_, ledger) = cluster.ledger(ctx.tenant_key())?;
            (ledger.role_of(&lm, ctx.sub())?, ledger.is_claimed(&lm)?)
        };
        authorize(op, ctx.scope_caps(), role, claimed)?;
        let key = IdemKey {
            sub: ctx.sub(),
            key: &req.op_id,
            body_sha256: hash,
        };
        {
            let (store, ledger) = cluster.ledger(ctx.tenant_key())?;
            match ledger.idem_lookup(store, &lm, key, now)? {
                IdemLookup::Miss => {}
                IdemLookup::Conflict => {
                    return Err(OpError::new(ErrorCode::OpReplayed, "op_id reused"))
                }
                IdemLookup::Replay(body) => {
                    let status =
                        serde_json::from_str::<OpResponse>(&body).map_or(500, |r| r.status());
                    return Ok(OpReply {
                        status,
                        body,
                        replayed: true,
                    });
                }
            }
        }
        let call = Call {
            ctx,
            lm: &lm,
            dry_run: req.dry_run,
            now,
        };
        let a = req.args_text();
        let outcome = match op {
            Op::TenantMe => Ok((
                json!({
                    "tenant_key": ctx.tenant_key().as_str(),
                    "sub": ctx.sub(),
                    "client_id": ctx.client_id(),
                    "act_sub": ctx.act_sub(),
                    "role": role.map(|r| r.as_str()),
                    "claimed": claimed,
                }),
                Default::default(),
            )),
            Op::CollectionList => exec::collection_list(cluster, &call, a),
            Op::CollectionCreate => exec::collection_create(cluster, &call, a, entropy),
            Op::VectorUpsert => vector::vector_upsert(cluster, &call, a),
            Op::VectorQuery => vector::vector_query(cluster, &call, a),
            Op::VectorFetch => vector::vector_fetch(cluster, &call, a),
            Op::VectorDelete => vector::vector_delete(cluster, &call, a),
            Op::UsageGet => exec::usage_get(cluster, &call, a),
        };
        let resp = match outcome {
            Ok((result, usage)) => OpResponse::ok(&req.op_id, result, usage),
            Err(e) => OpResponse::err(&req.op_id, &e),
        };
        let reply = OpReply::from_response(&resp);
        if resp.ok && !req.dry_run && remembered(op) {
            // A failure to remember does not undo the executed op; it
            // poisons the ledger, which is reopened from storage on next
            // use. The client's retry re-executes, which is safe for
            // upsert/delete by id.
            if let Ok((store, ledger)) = cluster.ledger(ctx.tenant_key()) {
                let _stored = ledger.idem_store(store, &lm, key, &reply.body, now);
            }
        }
        Ok(reply)
    }
}
