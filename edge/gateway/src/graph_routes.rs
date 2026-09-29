//! REST surface of rv-graph and rv-mincut (ADR-351 §3, §7.2, M4):
//!
//! - `POST /v1/graphs` `{"name"}` → create (write + editor), `201` / `409`
//! - `GET /v1/graphs` → list (read + viewer)
//! - `GET /v1/graphs/{g}` → counters (read)
//! - `DELETE /v1/graphs/{g}` → erase the graph and its catalog row
//!   (write and editor, like a mutating Cypher `DELETE` of every node);
//!   `200`, or `404` when neither existed. The graph's rows go first, then the
//!   catalog row, so a retry after a partial failure finishes the job and
//!   a failure never leaves unlisted graph storage behind.
//! - `POST /v1/graphs/{g}/cypher` `{"query"}` → `MATCH`/`RETURN`/`WITH`
//!   only: read + viewer; anything else: write + editor (a viewer's
//!   mutating query is `403 role_required`)
//! - `POST /v1/graphs/{g}/edges` `{"edges": [[a, b, w?], …], "type"?,
//!   "label"?}` → bulk insert (write + editor)
//! - `POST /v1/mincut`, `GET /v1/mincut/jobs/{id}` (`mincut_routes`)
//!
//! `Idempotency-Key` (§7) is honoured on the bulk edge insert and on a
//! mutating Cypher query, in the REST key table (`rest::slot`): a retry
//! with the same key and body replays the stored answer without running
//! or charging again; the same key on another request is `409`.
//!
//! Every graph and job DO is named from the verified tenant key, so
//! another tenant's graph or job is simply absent (`404`).

use crate::api::ratelimit::Class;
use crate::backend::Backend;
use crate::graph_cypher as gc;
use crate::graph_cypher_dx::Refusal;
use crate::graph_wire::{
    graph_do_name, job_do_name, name_ok, GraphCall, GraphOut, GraphRequest, JobCall, JobRequest,
    JobView, CATALOG,
};
use crate::idem::{self, Seen};
use crate::rest::{ApiReply, Caller};
use crate::service::{self, Access, Call};
use crate::wire::{unavailable, Reply};
use ruvector_edge_auth::Capability;
use ruvector_edge_store::{CallerContext, ErrorCode, OpError};
use ruvector_edge_tenancy::{EntropySource, Role};
use serde_json::{json, Map, Value as Json};
use worker::Method;

/// Graph / min-cut routes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphRoute {
    /// `POST /v1/graphs`.
    Create,
    /// `GET /v1/graphs`.
    List,
    /// `GET /v1/graphs/{g}`.
    Get(String),
    /// `DELETE /v1/graphs/{g}`.
    Delete(String),
    /// `POST /v1/graphs/{g}/cypher`.
    Cypher(String),
    /// `POST /v1/graphs/{g}/edges`.
    Edges(String),
    /// `POST /v1/mincut`.
    Mincut,
    /// `GET /v1/mincut/jobs/{id}`.
    Job(String),
}

/// `[A-Za-z0-9_-]{1,64}` (the analytics job id rule).
pub fn job_id_ok(id: &str) -> bool {
    (1..=64).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// Exact route match (unknown → `None`).
pub fn parse(method: &Method, path: &str) -> Option<GraphRoute> {
    let seg: Vec<&str> = path.strip_prefix("/v1/")?.split('/').collect();
    let (get, post) = (*method == Method::Get, *method == Method::Post);
    let delete = *method == Method::Delete;
    Some(match seg.as_slice() {
        ["graphs"] if post => GraphRoute::Create,
        ["graphs"] if get => GraphRoute::List,
        ["graphs", g] if get && name_ok(g) => GraphRoute::Get(g.to_string()),
        ["graphs", g] if delete && name_ok(g) => GraphRoute::Delete(g.to_string()),
        ["graphs", g, "cypher"] if post && name_ok(g) => GraphRoute::Cypher(g.to_string()),
        ["graphs", g, "edges"] if post && name_ok(g) => GraphRoute::Edges(g.to_string()),
        ["mincut"] if post => GraphRoute::Mincut,
        ["mincut", "jobs", id] if get && job_id_ok(id) => GraphRoute::Job(id.to_string()),
        _ => return None,
    })
}

/// The §10 rate class charged before the body is read.
pub fn class(r: &GraphRoute) -> Class {
    match r {
        GraphRoute::Create | GraphRoute::Edges(_) | GraphRoute::Delete(_) => Class::Write,
        _ => Class::Read,
    }
}

/// The extra class a body adds (a mutating Cypher query is also a write).
pub fn extra_class(r: &GraphRoute, body: &[u8]) -> Option<Class> {
    let GraphRoute::Cypher(_) = r else {
        return None;
    };
    let q = serde_json::from_slice::<Json>(body).ok()?;
    let parsed = gc::parse(q.get("query")?.as_str()?).ok()?;
    (!parsed.read_only).then_some(Class::Write)
}

/// What an operation needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Need {
    /// `ruvector:read` + viewer.
    Read,
    /// `ruvector:write` + editor.
    Write,
}

/// Scope first (`403 insufficient_scope` + step-up), then role.
pub async fn authorize<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    need: Need,
) -> Result<Access, OpError> {
    let (cap, min) = match need {
        Need::Read => (Capability::Read, Role::Viewer),
        Need::Write => (Capability::Write, Role::Editor),
    };
    if !ctx.scope_caps().contains(cap) {
        return Err(OpError {
            code: ErrorCode::InsufficientScope,
            detail: "insufficient scope",
            scope: Some(cap.satisfying_scope()),
        });
    }
    let a = service::access(b, ctx).await?;
    match a.role {
        None if !a.claimed => Err(OpError::new(ErrorCode::NotClaimed, "tenant not claimed")),
        None => Err(OpError::new(ErrorCode::RoleRequired, "membership required")),
        Some(r) if r < min => Err(OpError::new(ErrorCode::RoleRequired, "role required")),
        Some(_) => Ok(a),
    }
}

/// Authorize and charge one op to the ledger (§10 layer 4).
pub async fn admit<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    need: Need,
    now: u64,
) -> Result<(), OpError> {
    authorize(b, ctx, need).await?;
    let c = Call {
        b,
        ctx,
        dry_run: false,
        now,
    };
    service::charge(&c, service::one_op(), 1).await
}

fn decode<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, Refusal> {
    match serde_json::from_str::<Reply<T>>(text) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(e.into_refusal()),
        Err(_) => Err(unavailable().into()),
    }
}

/// One `GraphStore` call for the caller's tenant.
pub async fn graph_call<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    graph: &str,
    call: GraphCall,
) -> Result<GraphOut, Refusal> {
    let tenant = ctx.tenant_key();
    let req = GraphRequest {
        tenant_key: tenant.as_str().to_string(),
        graph: graph.to_string(),
        call,
    };
    let body = serde_json::to_string(&req).map_err(|_| unavailable())?;
    decode(&b.call_graph(&graph_do_name(tenant, graph), body).await?)
}

/// One `AnalyticsJob` call for the caller's tenant.
pub async fn job_call<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    job_id: &str,
    call: JobCall,
) -> Result<Json, OpError> {
    let tenant = ctx.tenant_key();
    let req = JobRequest {
        tenant_key: tenant.as_str().to_string(),
        job_id: job_id.to_string(),
        call,
    };
    let body = serde_json::to_string(&req).map_err(|_| unavailable())?;
    let v: JobView =
        decode(&b.call_job(&job_do_name(tenant, job_id), body).await?).map_err(|r| r.op)?;
    Ok(v.view)
}

/// Body object with only `allowed` keys.
pub fn body_obj(body: &[u8], allowed: &[&str]) -> Result<Map<String, Json>, OpError> {
    let m: Map<String, Json> = if body.iter().all(u8::is_ascii_whitespace) {
        Map::new()
    } else {
        serde_json::from_slice(body).map_err(|_| OpError::invalid("body must be a JSON object"))?
    };
    if m.keys().any(|k| !allowed.contains(&k.as_str())) {
        return Err(OpError::invalid("unknown field"));
    }
    Ok(m)
}

fn text<'a>(m: &'a Map<String, Json>, k: &str) -> Result<Option<&'a str>, OpError> {
    match m.get(k) {
        None => Ok(None),
        Some(Json::String(s)) => Ok(Some(s)),
        Some(_) => Err(OpError::invalid("field must be a string")),
    }
}

/// Create a graph: catalog entry, then the graph DO (idempotent init).
pub async fn create<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    name: &str,
    now: u64,
) -> Result<(u16, Json), OpError> {
    if !name_ok(name) {
        return Err(OpError::invalid(
            "graph name must match ^[a-z0-9][a-z0-9_-]{0,62}$",
        ));
    }
    admit(b, ctx, Need::Write, now).await?;
    let add = GraphCall::CatalogAdd {
        name: name.to_string(),
        sub: ctx.sub().to_string(),
        now,
    };
    let GraphOut::Added { created } = graph_call(b, ctx, CATALOG, add).await? else {
        return Err(service::unexpected());
    };
    let GraphOut::Info { view } = graph_call(b, ctx, name, GraphCall::Init { now }).await? else {
        return Err(service::unexpected());
    };
    if !created {
        return Err(OpError::new(ErrorCode::Conflict, "graph exists"));
    }
    Ok((201, view))
}

/// Run a Cypher query: authorization follows the parsed statement kinds.
pub async fn cypher<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    graph: &str,
    query: &str,
    now: u64,
) -> Result<Json, Refusal> {
    let parsed = gc::parse(query)?;
    let need = if parsed.read_only {
        Need::Read
    } else {
        Need::Write
    };
    admit(b, ctx, need, now).await?;
    let call = GraphCall::Cypher {
        query: query.to_string(),
        write: !parsed.read_only,
    };
    match graph_call(b, ctx, graph, call).await? {
        GraphOut::Rows { mut result } => {
            result["read_only"] = json!(parsed.read_only);
            Ok(result)
        }
        _ => Err(service::unexpected().into()),
    }
}

/// A graph mutation under an optional `Idempotency-Key`: authorize (a
/// replay is re-authorized), reserve the key, replay a stored answer or
/// run `op` (which charges) and store its answer.
async fn keyed<
    B: Backend,
    E: From<OpError>,
    F: std::future::Future<Output = Result<(u16, Json), E>>,
>(
    b: &B,
    ctx: &CallerContext,
    key: Option<&str>,
    tag: String,
    body: &[u8],
    now: u64,
    op: F,
) -> Result<(u16, Json), E> {
    let Some(key) = key else {
        return op.await;
    };
    if !idem::key_ok(key) {
        return Err(OpError::invalid("Idempotency-Key must be 1..=255 visible ASCII bytes").into());
    }
    authorize(b, ctx, Need::Write).await?;
    let slot = crate::rest::slot(ctx, &tag, key, body);
    let reused = "Idempotency-Key reused with a different request";
    if let Seen::Replay(stored) = idem::check(b, ctx, &slot, true, reused, now).await? {
        return crate::rest::replayed(&stored).map_err(E::from);
    }
    let out = op.await;
    let stored = out.as_ref().ok().map(|(s, v)| format!("{s} {v}"));
    idem::finish(b, ctx, slot, stored, now).await;
    out
}

/// Delete a graph: wipe its rows, then its catalog row.
pub async fn delete<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    name: &str,
    now: u64,
) -> Result<(u16, Json), OpError> {
    admit(b, ctx, Need::Write, now).await?;
    let GraphOut::Removed { existed: stored } = graph_call(b, ctx, name, GraphCall::Wipe).await?
    else {
        return Err(service::unexpected());
    };
    let forget = GraphCall::CatalogRemove {
        name: name.to_string(),
    };
    let GraphOut::Removed { existed: listed } = graph_call(b, ctx, CATALOG, forget).await? else {
        return Err(service::unexpected());
    };
    if !stored && !listed {
        return Err(OpError::not_found());
    }
    Ok((200, json!({ "name": name, "deleted": true })))
}

/// List the tenant's graphs.
pub async fn list<B: Backend>(b: &B, ctx: &CallerContext, now: u64) -> Result<Json, OpError> {
    admit(b, ctx, Need::Read, now).await?;
    match graph_call(b, ctx, CATALOG, GraphCall::CatalogList).await? {
        GraphOut::Graphs { graphs } => Ok(json!({ "graphs": graphs })),
        _ => Err(service::unexpected()),
    }
}

/// `POST /v1/graphs/{g}/cypher`; a refusal keeps its explanation.
async fn cypher_route<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    g: &str,
    body: &[u8],
    key: Option<&str>,
    now: u64,
) -> Result<(u16, Json), Refusal> {
    let m = body_obj(body, &["query"])?;
    let q = text(&m, "query")?.ok_or(OpError::invalid("query required"))?;
    // Only a mutating query is keyed (a read replays nothing).
    let key = key.filter(|_| gc::parse(q).is_ok_and(|p| !p.read_only));
    let run = async { cypher(b, ctx, g, q, now).await.map(|v| (200, v)) };
    keyed(b, ctx, key, format!("graph-cypher/{g}"), body, now, run).await
}

/// One route; `key` is the `Idempotency-Key` header.
async fn route<B: Backend>(
    b: &B,
    caller: &Caller,
    r: &GraphRoute,
    body: &[u8],
    key: Option<&str>,
    now: u64,
    entropy: &dyn EntropySource,
) -> Result<(u16, Json), OpError> {
    let ctx = &caller.ctx;
    match r {
        GraphRoute::Create => {
            let m = body_obj(body, &["name"])?;
            let name = text(&m, "name")?.ok_or(OpError::invalid("name required"))?;
            create(b, ctx, name, now).await
        }
        GraphRoute::List => list(b, ctx, now).await.map(|v| (200, v)),
        GraphRoute::Get(g) => {
            admit(b, ctx, Need::Read, now).await?;
            match graph_call(b, ctx, g, GraphCall::Stats).await? {
                GraphOut::Info { view } => Ok((200, view)),
                _ => Err(service::unexpected()),
            }
        }
        GraphRoute::Delete(g) => delete(b, ctx, g, now).await,
        GraphRoute::Cypher(g) => Ok(cypher_route(b, ctx, g, body, key, now).await?),
        GraphRoute::Edges(g) => {
            let m = body_obj(body, &["edges", "type", "label"])?;
            let edges = serde_json::from_value(m.get("edges").cloned().unwrap_or(Json::Null))
                .map_err(|_| OpError::invalid("edges must be [[from, to, weight?], …]"))?;
            let call = GraphCall::AddEdges {
                edges,
                edge_type: text(&m, "type")?.unwrap_or("LINK").to_string(),
                label: text(&m, "label")?.unwrap_or("Node").to_string(),
            };
            let run = async {
                admit(b, ctx, Need::Write, now).await?;
                match graph_call(b, ctx, g, call).await? {
                    GraphOut::Info { view } => Ok((200, view)),
                    _ => Err(service::unexpected()),
                }
            };
            keyed(b, ctx, key, format!("graph-edges/{g}"), body, now, run).await
        }
        GraphRoute::Mincut => crate::mincut_routes::post(b, ctx, body, now, Some(entropy)).await,
        GraphRoute::Job(id) => crate::mincut_routes::get(b, ctx, id, now)
            .await
            .map(|v| (200, v)),
    }
}

/// Handle one graph / min-cut route for a verified caller; `key` is the
/// `Idempotency-Key` header.
#[allow(clippy::too_many_arguments)]
pub async fn handle<B: Backend>(
    b: &B,
    caller: &Caller,
    r: &GraphRoute,
    body: &[u8],
    key: Option<&str>,
    now: u64,
    metadata_url: &str,
    entropy: &dyn EntropySource,
) -> ApiReply {
    let out = match r {
        GraphRoute::Cypher(g) => cypher_route(b, &caller.ctx, g, body, key, now).await,
        _ => route(b, caller, r, body, key, now, entropy)
            .await
            .map_err(Refusal::from),
    };
    match out {
        Ok((status, v)) => ApiReply::json(status, &v),
        Err(e) => ApiReply::problem_with(&e.op, &e.shown(), metadata_url),
    }
}
