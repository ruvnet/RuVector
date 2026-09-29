//! Workers glue for the authenticated data surface: `/v1/me`, the REST
//! table, `/v1/ops` and `/v1/mcp`. Reads the body (bounded), builds the
//! store-facing caller from the verified token, runs the pure handler over
//! the Durable Object backend and renders the reply.

use crate::auth::Authenticated;
use crate::backend::Backend;
use crate::config::GatewayConfig;
use crate::durable::DoBackend;
use crate::platform::WorkerClock;
use crate::rest::{self, ApiReply, ApiRoute, Caller};
use crate::routes::Route;
use crate::{mcp, ops, respond};
use ruvector_edge_auth::{prm, Clock};
use ruvector_edge_store::{CallerContext, ErrorCode, OpError};
use ruvector_edge_tenancy::ProblemCode;
use worker::{Env, Method, Request, Response, Result};

/// Largest request body on any data route (ADR-351 §10: 1 MiB).
pub const MAX_BODY_BYTES: usize = 1 << 20;

/// `true` when a declared `Content-Length` exceeds [`MAX_BODY_BYTES`]
/// (checked before authentication).
pub fn declared_too_large(req: &Request) -> bool {
    req.headers()
        .get("Content-Length")
        .ok()
        .flatten()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .is_some_and(|n| n > MAX_BODY_BYTES)
}

fn render(r: ApiReply) -> Result<Response> {
    respond::raw(
        r.status,
        r.body,
        r.content_type,
        r.www_authenticate.as_deref(),
    )
}

/// The store-facing caller of a verified request, carrying the exchanged
/// token's `act.sub` (ADR-351 §16.3: records which adapter acted).
pub fn caller(a: Authenticated) -> Caller {
    Caller {
        ctx: CallerContext::from_tenant(&a.ctx, a.scope_caps, a.act_sub),
        org_id: a.ctx.org_id().to_string(),
        workspace_id: a.ctx.workspace_id().to_string(),
        scopes: a.scopes,
    }
}

/// A verified `/v1` data request (REST table or `/v1/ops`) whose body was
/// read: the pure half of [`serve`], shared with the native tests.
/// `idempotency_key` is the `Idempotency-Key` header (`/v1/ops` `op_id`, or
/// the REST key of a mutating route).
pub async fn data<B: Backend>(
    b: &B,
    cfg: &GatewayConfig,
    api: &ApiRoute,
    caller: &Caller,
    body: &[u8],
    idempotency_key: Option<&str>,
    now: u64,
) -> ApiReply {
    let md = prm::metadata_url(&cfg.rest_resource);
    if *api != ApiRoute::Ops {
        return rest::handle(b, caller, api, body, idempotency_key, now, &md).await;
    }
    let target = format!("{}/ops", cfg.rest_resource.as_str());
    let r = ops::dispatch(b, &target, &caller.ctx, body, idempotency_key, now).await;
    let www = r.step_up.and_then(|scope| {
        let e = OpError {
            code: ErrorCode::InsufficientScope,
            detail: "insufficient scope",
            scope: Some(scope),
        };
        rest::step_up(&e, &md)
    });
    ApiReply {
        status: r.status,
        body: r.body,
        content_type: "application/json",
        www_authenticate: www,
    }
}

/// Serve an authenticated `Me` / `Mcp` / `OtherV1` request.
pub async fn serve(
    mut req: Request,
    env: &Env,
    cfg: &GatewayConfig,
    route: Route,
    auth: Authenticated,
) -> Result<Response> {
    let (method, path) = (req.method(), req.path());
    let caller = caller(auth);
    let backend = DoBackend { env };
    let now = WorkerClock.now_unix();
    if route == Route::Mcp {
        if path != "/v1/mcp" {
            return respond::problem(ProblemCode::NotFound, None);
        }
        if method != Method::Post {
            // No server-initiated SSE stream (Streamable HTTP allows 405).
            let mut resp = respond::raw(405, String::new(), "application/json", None)?;
            resp.headers_mut().set("Allow", "POST, OPTIONS")?;
            return Ok(resp);
        }
        let version = req.headers().get("MCP-Protocol-Version").ok().flatten();
        if !mcp::protocol_header_ok(version.as_deref()) {
            return render(mcp::bad_protocol_version());
        }
        let md = prm::metadata_url(&cfg.mcp_resource);
        let Some(body) = body(&mut req).await? else {
            return too_large(&md);
        };
        return render(mcp::handle(&backend, &caller.ctx, &body, now, &md).await);
    }
    let md = prm::metadata_url(&cfg.rest_resource);
    let Some(api) = rest::parse(&method, &path) else {
        return respond::problem(ProblemCode::NotFound, None);
    };
    let Some(body) = body(&mut req).await? else {
        return too_large(&md);
    };
    let key = req.headers().get("Idempotency-Key").ok().flatten();
    render(data(&backend, cfg, &api, &caller, &body, key.as_deref(), now).await)
}

/// Read the body, `None` once it exceeds [`MAX_BODY_BYTES`]. Streamed and
/// counted while reading, so an undeclared or chunked body (no
/// `Content-Length`, which the pre-auth check needs) is cut off at the cap
/// instead of being buffered whole into the isolate.
async fn body(req: &mut Request) -> Result<Option<Vec<u8>>> {
    if matches!(req.method(), Method::Get | Method::Head) || req.inner().body().is_none() {
        return Ok(Some(Vec::new()));
    }
    read_capped(req.stream()?, MAX_BODY_BYTES).await
}

/// Collect `chunks` into one buffer, `None` as soon as more than `max`
/// bytes have arrived (the rest is never read).
pub async fn read_capped<S, E>(chunks: S, max: usize) -> std::result::Result<Option<Vec<u8>>, E>
where
    S: futures_util::Stream<Item = std::result::Result<Vec<u8>, E>>,
{
    use futures_util::StreamExt;
    let mut chunks = std::pin::pin!(chunks);
    let mut out = Vec::new();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk?;
        if chunk.len() > max.saturating_sub(out.len()) {
            return Ok(None);
        }
        out.extend_from_slice(&chunk);
    }
    Ok(Some(out))
}

fn too_large(md: &str) -> Result<Response> {
    let e = OpError::new(ErrorCode::PayloadTooLarge, "body too large");
    render(ApiReply::problem(&e, md))
}
