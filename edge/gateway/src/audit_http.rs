//! Audit of the M1 surface and Workers glue for sending audit events off
//! the request path (ADR-351 §6.5; what is shipped: `audit` module docs).
//!
//! `serve` wraps `api::serve` for `/v1/…` (REST table incl. the M2 drop /
//! members / deny routes, the M5 registry routes, `/v1/ops`) and `/v1/mcp`:
//! the mutating REST and registry routes are named from the path; `/v1/ops`
//! and `POST /v1/mcp` name their op / tool in the body, which is read from
//! a clone (`Request::clone` tees the stream; both reads stay capped at
//! 1 MiB). The event carries the HTTP status (an MCP tool error is a
//! JSON-RPC result, so `200`), the scope the op needs and the request body
//! size; M1 executors do not report rows or work units back through the
//! HTTP layer, so those stay `0` here, and the role is not re-read.
//!
//! [`Deferred`] collects audit sends during a request; [`flush`] hands them
//! to `ctx.wait_until`, so the response never waits on the Queue.

use crate::api::{self, ratelimit::MUTATING_OPS};
use crate::audit::AuditEvent;
use crate::auth::Authenticated;
use crate::config::GatewayConfig;
use crate::m3_ports::{QueueName, Queues, WorkerQueues};
use crate::platform::WorkerClock;
use crate::registry_routes::{self, RvfRoute};
use crate::rest::{self, ApiRoute};
use crate::routes::Route;
use ruvector_edge_auth::{Capability, Clock};
use ruvector_edge_store::{CallerContext, OpError};
use serde_json::Value as Json;
use std::cell::RefCell;
use worker::{Context, Env, Method, Request, Response, Result};

/// Queues producer that holds audit messages for [`flush`] and sends
/// everything else (import jobs) immediately.
pub struct Deferred<'a> {
    /// The real producer.
    pub inner: WorkerQueues<'a>,
    /// Held audit bodies.
    pub audit: RefCell<Vec<Json>>,
}

impl<'a> Deferred<'a> {
    /// Over the Worker's queue bindings.
    pub fn new(env: &'a Env) -> Self {
        Deferred {
            inner: WorkerQueues(env),
            audit: RefCell::new(Vec::new()),
        }
    }
}

impl Queues for Deferred<'_> {
    async fn send(&self, q: QueueName, body: Json) -> std::result::Result<(), OpError> {
        match q {
            QueueName::Audit => {
                self.audit.borrow_mut().push(body);
                Ok(())
            }
            QueueName::Ingest => self.inner.send(q, body).await,
        }
    }
}

/// Send the held audit messages after the response (best effort).
pub fn flush(d: &Deferred<'_>, env: &Env, ctx: &Context) {
    let held = d.audit.take();
    if held.is_empty() {
        return;
    }
    let env = env.clone();
    ctx.wait_until(async move {
        let q = WorkerQueues(&env);
        for body in held {
            let _sent = q.send(QueueName::Audit, body).await;
        }
    });
}

/// Audit name and required capability of a mutating REST route (path only):
/// the M1 writes plus the M2 drop, members and deny routes.
pub fn rest_event(route: &ApiRoute) -> Option<(&'static str, Capability)> {
    match route {
        ApiRoute::Claim => Some(("tenant.claim", Capability::Write)),
        ApiRoute::Create => Some(("collection.create", Capability::CreateCollection)),
        ApiRoute::Upsert(_) => Some(("vector.upsert", Capability::Write)),
        ApiRoute::Delete(_) => Some(("vector.delete", Capability::Write)),
        ApiRoute::Drop(_) => Some(("collection.drop", Capability::CreateCollection)),
        ApiRoute::Invite => Some(("member.invite", Capability::Admin)),
        ApiRoute::RemoveMember(_) => Some(("member.remove", Capability::Admin)),
        ApiRoute::Deny => Some(("tenant.deny", Capability::Admin)),
        _ => None,
    }
}

/// Audit name and required capability of a mutating M5 registry route.
pub fn rvf_event(route: &RvfRoute) -> Option<(&'static str, Capability)> {
    match route {
        RvfRoute::ClaimScope(_) => Some(("rvf.scope.claim", Capability::Admin)),
        RvfRoute::Begin(..) => Some(("rvf.upload.begin", Capability::Write)),
        RvfRoute::Part(..) => Some(("rvf.upload.part", Capability::Write)),
        RvfRoute::Finalize(..) => Some(("rvf.upload.finalize", Capability::Write)),
        RvfRoute::Publish(..) => Some(("rvf.publish", Capability::Write)),
        RvfRoute::Yank(..) => Some(("rvf.yank", Capability::Write)),
        RvfRoute::Unyank(..) => Some(("rvf.unyank", Capability::Write)),
        RvfRoute::Import(_) => Some(("collection.import_rvf", Capability::Write)),
        RvfRoute::ListScopes | RvfRoute::Get(..) | RvfRoute::Blob(..) | RvfRoute::Versions(_) => {
            None
        }
    }
}

/// A mutating `/v1/ops` op or MCP tool ([`MUTATING_OPS`], which also
/// drives the extra write budget) and the capability it needs.
fn mutating(name: &str) -> Option<(&'static str, Capability)> {
    let n = *MUTATING_OPS.iter().find(|m| **m == name)?;
    let cap = match n {
        "collection_create" => Capability::CreateCollection,
        _ => Capability::Write,
    };
    Some((n, cap))
}

/// Audit name and capability of a mutating `/v1/ops` op or MCP
/// `tools/call` (`mcp` selects the JSON-RPC shape).
pub fn body_event(mcp: bool, body: &[u8]) -> Option<(String, Capability)> {
    let v: Json = serde_json::from_slice(body).ok()?;
    if mcp {
        if v.get("method")?.as_str()? != "tools/call" {
            return None;
        }
        let (n, cap) = mutating(v.get("params")?.get("name")?.as_str()?)?;
        Some((format!("mcp.{n}"), cap))
    } else {
        let (n, cap) = mutating(v.get("op")?.as_str()?)?;
        Some((format!("ops.{n}"), cap))
    }
}

async fn peek(req: &Request, mcp: bool) -> Result<Option<(String, Capability)>> {
    let mut copy = req.clone()?;
    if copy.inner().body().is_none() {
        return Ok(None);
    }
    let body = api::read_capped(copy.stream()?, api::MAX_BODY_BYTES).await?;
    Ok(body.and_then(|b| body_event(mcp, &b)))
}

/// `api::serve` plus one audit event for a mutating request.
pub async fn serve(
    req: Request,
    env: &Env,
    cfg: &GatewayConfig,
    route: Route,
    auth: Authenticated,
    ctx: &Context,
) -> Result<Response> {
    let (method, path) = (req.method(), req.path());
    let event = match route {
        Route::Mcp if method == Method::Post && path == "/v1/mcp" => peek(&req, true).await?,
        // Registry routes first, as `api::serve` dispatches them.
        Route::OtherV1 => match registry_routes::parse(&method, &path) {
            Some(r) => rvf_event(&r).map(|(n, c)| (n.to_string(), c)),
            None => match rest::parse(&method, &path) {
                Some(ApiRoute::Ops) => peek(&req, false).await?,
                Some(r) => rest_event(&r).map(|(n, c)| (n.to_string(), c)),
                None => None,
            },
        },
        _ => None,
    };
    let Some((name, cap)) = event else {
        return api::serve(req, env, cfg, route, auth).await;
    };
    let bytes = req
        .headers()
        .get("Content-Length")
        .ok()
        .flatten()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let caller: CallerContext = api::caller(auth.clone()).ctx;
    let resp = api::serve(req, env, cfg, route, auth).await?;
    let now_ms = WorkerClock.now_unix() * 1000;
    let mut ev = AuditEvent::new(&caller, &name, resp.status_code(), now_ms);
    ev.scope = Some(cap.satisfying_scope().to_string());
    ev.bytes = bytes;
    let d = Deferred::new(env);
    crate::audit::emit(&d, &ev).await;
    flush(&d, env, ctx);
    Ok(resp)
}
