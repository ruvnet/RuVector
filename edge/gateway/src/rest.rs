//! REST surface (ADR-351 §7.2) mapped onto the same ops as `/v1/ops` and
//! `/v1/mcp`. Pure over [`Backend`]: routing in, [`ApiReply`] out.
//!
//! Task spellings and the ADR/`ROUTE_TABLE` spellings are both served:
//! `POST /v1/claim` = `POST /v1/tenant:claim`; `POST …/vectors` =
//! `…/vectors:upsert`; `DELETE …/vectors` = `POST …/vectors:delete`;
//! `POST …/fetch` = `…/vectors:fetch`.

use crate::backend::Backend;
use crate::idem::{self, Seen, Slot};
use crate::service::{self, access, capability_names, Call, Exec};
use ruvector_edge_auth::prm;
use ruvector_edge_auth::Capability;
use ruvector_edge_store::{CallerContext, ErrorCode, Op, OpError};
use ruvector_edge_tenancy::problem::PROBLEM_CONTENT_TYPE;
use ruvector_edge_tenancy::Problem;
use serde_json::{json, Map, Value as Json};
use sha2::{Digest, Sha256};
use worker::Method;

/// A pure HTTP reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiReply {
    /// Status.
    pub status: u16,
    /// Body text (empty for 202/204).
    pub body: String,
    /// `Content-Type`.
    pub content_type: &'static str,
    /// `WWW-Authenticate` (403 step-up).
    pub www_authenticate: Option<String>,
}

impl ApiReply {
    /// JSON reply.
    pub fn json(status: u16, v: &Json) -> Self {
        ApiReply {
            status,
            body: v.to_string(),
            content_type: "application/json",
            www_authenticate: None,
        }
    }

    /// RFC 9457 problem for `e`; `insufficient_scope` carries the §5.3
    /// step-up challenge naming `metadata_url`.
    pub fn problem(e: &OpError, metadata_url: &str) -> Self {
        let status = e.code.status();
        let p = Problem {
            type_: "about:blank".into(),
            title: e.code.as_str().into(),
            status,
            code: e.code.as_str().into(),
            detail: Some(e.detail.to_string()),
            request_id: None,
        };
        ApiReply {
            status,
            body: p.to_json(),
            content_type: PROBLEM_CONTENT_TYPE,
            www_authenticate: step_up(e, metadata_url),
        }
    }
}

/// `WWW-Authenticate` for an `insufficient_scope` error.
pub fn step_up(e: &OpError, metadata_url: &str) -> Option<String> {
    if e.code != ErrorCode::InsufficientScope {
        return None;
    }
    let cap = match e.scope {
        Some("ruvector:read") => Capability::Read,
        Some("ruvector:admin") => Capability::Admin,
        Some("ruvector:publish") => Capability::PublishPublic,
        _ => Capability::Write,
    };
    Some(prm::www_authenticate_insufficient_scope(
        metadata_url,
        &prm::step_up_scope(cap),
    ))
}

/// A verified caller plus the token fields `/v1/me` reports.
#[derive(Debug, Clone)]
pub struct Caller {
    /// Store-facing context (tenant, sub, scope-derived capabilities).
    pub ctx: CallerContext,
    /// Upstream org.
    pub org_id: String,
    /// Upstream workspace.
    pub workspace_id: String,
    /// Granted scopes.
    pub scopes: Vec<String>,
}

/// REST routes behind the `/v1` resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiRoute {
    /// `GET /v1/me`.
    Me,
    /// `POST /v1/claim` | `/v1/tenant:claim`.
    Claim,
    /// `GET /v1/usage`.
    Usage,
    /// `POST /v1/collections`.
    Create,
    /// `GET /v1/collections`.
    List,
    /// `GET /v1/collections/{c}`.
    Get(String),
    /// Upsert into `{c}`.
    Upsert(String),
    /// Delete from `{c}`.
    Delete(String),
    /// `POST /v1/collections/{c}/query`.
    Query(String),
    /// Fetch from `{c}`.
    Fetch(String),
    /// `POST /v1/ops`.
    Ops,
}

fn name_ok(c: &str) -> bool {
    !c.is_empty() && !c.contains(':') && c != "." && c != ".."
}

/// Exact route match (no normalisation; unknown → `None` → 404).
pub fn parse(method: &Method, path: &str) -> Option<ApiRoute> {
    let rest = path.strip_prefix("/v1/")?;
    let seg: Vec<&str> = rest.split('/').collect();
    let (get, post, del) = (
        *method == Method::Get,
        *method == Method::Post,
        *method == Method::Delete,
    );
    let r = match seg.as_slice() {
        ["me"] if get => ApiRoute::Me,
        ["claim" | "tenant:claim"] if post => ApiRoute::Claim,
        ["usage"] if get => ApiRoute::Usage,
        ["ops"] if post => ApiRoute::Ops,
        ["collections"] if post => ApiRoute::Create,
        ["collections"] if get => ApiRoute::List,
        ["collections", c] if get && name_ok(c) => ApiRoute::Get(c.to_string()),
        ["collections", c, tail] if name_ok(c) => {
            let c = c.to_string();
            match (*tail, post, del) {
                ("vectors" | "vectors:upsert", true, _) => ApiRoute::Upsert(c),
                ("vectors", _, true) | ("vectors:delete", true, _) => ApiRoute::Delete(c),
                ("query", true, _) => ApiRoute::Query(c),
                ("fetch" | "vectors:fetch", true, _) => ApiRoute::Fetch(c),
                _ => return None,
            }
        }
        _ => return None,
    };
    Some(r)
}

/// Body object plus the path's collection as op `args`; `dry_run` (only
/// where `allow_dry_run`) is lifted out of the body.
pub fn op_args(
    body: &[u8],
    collection: Option<&str>,
    allow_dry_run: bool,
) -> Result<(String, bool), OpError> {
    let mut m: Map<String, Json> = if body.iter().all(u8::is_ascii_whitespace) {
        Map::new()
    } else {
        serde_json::from_slice(body).map_err(|_| OpError::invalid("body must be a JSON object"))?
    };
    let dry_run = match m.remove("dry_run") {
        None => false,
        Some(Json::Bool(d)) if allow_dry_run => d,
        Some(_) => return Err(OpError::invalid("dry_run not accepted")),
    };
    if let Some(c) = collection {
        if m.insert("collection".into(), json!(c)).is_some() {
            return Err(OpError::invalid("collection comes from the path"));
        }
    }
    Ok((Json::Object(m).to_string(), dry_run))
}

/// `/v1/me`: token identity plus the ledger role and effective
/// capabilities (scope ∩ role).
pub async fn me<B: Backend>(b: &B, caller: &Caller) -> Result<Json, OpError> {
    let a = access(b, &caller.ctx).await?;
    let ctx = &caller.ctx;
    Ok(json!({
        "tenant_key": ctx.tenant_key().as_str(),
        "org_id": caller.org_id,
        "workspace_id": caller.workspace_id,
        "sub": ctx.sub(),
        "client_id": ctx.client_id(),
        "act_sub": ctx.act_sub(),
        "scopes": caller.scopes,
        "role": a.role.map(|r| r.as_str()),
        "claimed": a.claimed,
        "capabilities": capability_names(a.effective(ctx.scope_caps())),
    }))
}

async fn exec_route<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    route: &ApiRoute,
    body: &[u8],
    idempotency_key: Option<&str>,
    now: u64,
) -> Result<(u16, Json), OpError> {
    let call = |dry_run| Call {
        b,
        ctx,
        dry_run,
        now,
    };
    let op = |op: Op, coll: Option<&str>, dry: bool| -> Result<(Op, String, bool), OpError> {
        let (args, dry_run) = op_args(body, coll, dry)?;
        Ok((op, args, dry_run))
    };
    let (op, args, dry_run) = match route {
        ApiRoute::Claim => return service::claim(&call(false)).await.map(|(r, _)| (201, r)),
        ApiRoute::Get(c) => {
            let (args, _) = op_args(b"{}", Some(c), false)?;
            let out: Exec = service::collection_get(&call(false), &args).await;
            return out.map(|(r, _)| (200, r));
        }
        ApiRoute::Usage => op(Op::UsageGet, None, false)?,
        ApiRoute::Create => op(Op::CollectionCreate, None, false)?,
        ApiRoute::List => op(Op::CollectionList, None, false)?,
        ApiRoute::Upsert(c) => op(Op::VectorUpsert, Some(c), true)?,
        ApiRoute::Delete(c) => op(Op::VectorDelete, Some(c), true)?,
        ApiRoute::Query(c) => op(Op::VectorQuery, Some(c), false)?,
        ApiRoute::Fetch(c) => op(Op::VectorFetch, Some(c), false)?,
        ApiRoute::Me | ApiRoute::Ops => return Err(OpError::not_found()),
    };
    let status = if op == Op::CollectionCreate && !dry_run {
        201
    } else {
        200
    };
    let mutating = matches!(
        op,
        Op::CollectionCreate | Op::VectorUpsert | Op::VectorDelete
    );
    let Some(key) = idempotency_key.filter(|_| mutating && !dry_run) else {
        let (result, _usage) = service::execute(&call(dry_run), op, &args).await?;
        return Ok((status, result));
    };
    if !idem::key_ok(key) {
        return Err(OpError::invalid(
            "Idempotency-Key must be 1..=255 visible ASCII bytes",
        ));
    }
    // Scope and role first, so a replay is re-authorized.
    let access = service::authorize_op(b, ctx, op).await?;
    let slot = rest_slot(ctx, route, key, body);
    let reused = "Idempotency-Key reused with a different request";
    if let Seen::Replay(stored) = idem::check(b, ctx, &slot, true, reused, now).await? {
        return replayed(&stored);
    }
    match service::run(&call(false), access, op, &args).await {
        Ok((result, _usage)) => {
            let stored = format!("{status} {result}");
            idem::finish(b, ctx, slot, Some(stored), now).await;
            Ok((status, result))
        }
        Err(e) => {
            idem::finish(b, ctx, slot, None, now).await;
            Err(e)
        }
    }
}

/// The REST key slot: namespaced away from `/v1/ops` `op_id`s (same table)
/// and bound to the route (so one body on two collections never replays
/// the other's response) plus the raw body.
fn rest_slot(ctx: &CallerContext, route: &ApiRoute, key: &str, body: &[u8]) -> Slot {
    let tag = match route {
        ApiRoute::Create => "create".to_string(),
        ApiRoute::Upsert(c) => format!("upsert/{c}"),
        ApiRoute::Delete(c) => format!("delete/{c}"),
        other => format!("{other:?}"),
    };
    let mut h = Sha256::new();
    h.update(tag.as_bytes());
    h.update(b"\n");
    h.update(body);
    Slot {
        sub: ctx.sub().to_string(),
        key: format!("rest:{key}"),
        sha256: h.finalize().into(),
    }
}

/// A stored `"<status> <json>"` REST response.
fn replayed(stored: &str) -> Result<(u16, Json), OpError> {
    stored
        .split_once(' ')
        .and_then(|(s, j)| Some((s.parse().ok()?, serde_json::from_str(j).ok()?)))
        .ok_or(OpError::new(ErrorCode::ServerError, "stored response"))
}

/// Handle a data route (everything but `/v1/ops`) for a verified caller.
/// `metadata_url` is the `/v1` RFC 9728 document (step-up challenges);
/// `idempotency_key` the `Idempotency-Key` header (§7: honoured on the
/// mutating routes — create, upsert, delete — except dry runs).
pub async fn handle<B: Backend>(
    b: &B,
    caller: &Caller,
    route: &ApiRoute,
    body: &[u8],
    idempotency_key: Option<&str>,
    now: u64,
    metadata_url: &str,
) -> ApiReply {
    let res = match route {
        ApiRoute::Me => me(b, caller).await.map(|v| (200, v)),
        _ => exec_route(b, &caller.ctx, route, body, idempotency_key, now).await,
    };
    match res {
        Ok((status, v)) => ApiReply::json(status, &v),
        Err(e) => ApiReply::problem(&e, metadata_url),
    }
}
