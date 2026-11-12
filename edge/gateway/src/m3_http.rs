//! Workers glue of the M3 routes (split out of `m3_api`): the pre-auth
//! body-size check and [`serve`] (guard, read the body, [`dispatch`],
//! render or stream an export; audit sends go out through
//! `ctx.wait_until` after the reply).

use crate::api::{self, ratelimit::EnvLimiter};
use crate::audit::{self, AuditEvent};
use crate::audit_http::{flush, Deferred};
use crate::auth::Authenticated;
use crate::config::GatewayConfig;
use crate::durable::DoBackend;
use crate::m3_api::{dispatch, parse, Input, M3Route, Out};
use crate::m3_ctx::M3;
use crate::m3_ports::{WorkersAi, DATA_BINDING, R2};
use crate::rest::ApiReply;
use crate::{export, respond};
use ruvector_edge_auth::{prm, Clock};
use ruvector_edge_store::{ErrorCode, OpError};
use worker::{Context, Env, Method, Request, Response, Result};

/// Pre-auth size check: the M3 route's own cap ([`M3Route::body_cap`]:
/// 8 MiB upload parts, 512 KiB inline imports), else the M5 registry check
/// (part cap, else the M1 1 MiB).
pub fn declared_too_large(req: &Request) -> bool {
    let cap = parse(&req.method(), &req.path()).and_then(|r| r.body_cap());
    let Some(cap) = cap else {
        return crate::registry_http::declared_too_large(req);
    };
    req.headers()
        .get("Content-Length")
        .ok()
        .flatten()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .is_some_and(|n| n > cap)
}

/// Whether a request refused by the rate / deny guard is audited: not the
/// read routes the dispatcher never audits, and not a `429` (a throttled
/// flood must not turn into one queue message per request; a deny-list
/// `401` is bounded by the rate budget checked first).
pub fn refusal_audited(route: &M3Route, status: u16) -> bool {
    status != 429
        && !matches!(
            route,
            M3Route::Query(_) | M3Route::ExportGet(_) | M3Route::Job(_) | M3Route::SnapshotList(_)
        )
}

/// Workers glue: read the body, run [`dispatch`], render (or stream an
/// export); audit sends go out through `ctx.wait_until` after the reply.
pub async fn serve(
    mut req: Request,
    env: &Env,
    cfg: &GatewayConfig,
    route: M3Route,
    auth: Authenticated,
    wctx: &Context,
) -> Result<Response> {
    let caller = api::caller(auth);
    let md = prm::metadata_url(&cfg.rest_resource);
    let b = DoBackend { env };
    let now_ms = crate::platform::WorkerClock.now_unix() * 1000;
    // ADR-351 §10 layer 2 then §5.8, as on every `/v1` route: rate budget
    // and deny list before the body (up to 8 MiB) is read. A refusal is
    // audited per [`refusal_audited`].
    let limiter = EnvLimiter(env);
    let guarded = api::guard(&b, &limiter, route.class(), &caller.ctx, now_ms / 1000, &md).await;
    if let Err(r) = guarded {
        if refusal_audited(&route, r.reply.status) {
            let queues = Deferred::new(env);
            let ev = AuditEvent::new(&caller.ctx, route.name(), r.reply.status, now_ms);
            audit::emit(&queues, &ev).await;
            flush(&queues, env, wctx);
        }
        return api::refused(r);
    }
    let max = route.body_cap().unwrap_or(api::MAX_BODY_BYTES);
    let body = if matches!(req.method(), Method::Get) || req.inner().body().is_none() {
        Some(Vec::new())
    } else {
        api::read_capped(req.stream()?, max).await?
    };
    let Some(body) = body else {
        let e = OpError::new(ErrorCode::PayloadTooLarge, "body too large");
        let r = ApiReply::problem(&e, &md);
        return respond::raw(r.status, r.body, r.content_type, None);
    };
    let ct = req.headers().get("Content-Type").ok().flatten();
    let key = req.headers().get("Idempotency-Key").ok().flatten();
    let url = req.url()?;
    let blob = R2(env.bucket(DATA_BINDING)?);
    let queues = Deferred::new(env);
    let ai = WorkersAi(env);
    let m = M3 {
        b: &b,
        blob: &blob,
        queues: &queues,
        ctx: &caller.ctx,
        now_ms,
        note: audit::Note::default(),
    };
    let inp = Input {
        route: &route,
        body,
        octet_stream: ct.is_some_and(|c| c.starts_with("application/octet-stream")),
        query: url.query(),
        key: key.as_deref(),
    };
    let out = dispatch(&m, &ai, &caller, &md, inp).await;
    let resp = match out {
        // `api::render`: a 429 (e.g. query fan-out over budget) carries
        // `Retry-After`, as on the M1 / M2 routes.
        Out::Reply(r) => api::render(r),
        Out::Download(d) => {
            let size = d.size;
            let resp = stream(&blob, d).await;
            let status = resp.as_ref().map_or(500, Response::status_code);
            if status == 200 {
                m.note.bytes.set(size);
            }
            let ev = AuditEvent::new(m.ctx, route.name(), status, m.now_ms).with_note(&m.note);
            audit::emit(&queues, &ev).await;
            resp
        }
    };
    flush(&queues, env, wctx);
    resp
}

async fn stream(blob: &R2, d: export::Download) -> Result<Response> {
    let Some(obj) = blob.0.get(&d.key).execute().await? else {
        return respond::problem(ruvector_edge_tenancy::ProblemCode::NotFound, None);
    };
    let Some(body) = obj.body() else {
        return respond::problem(ruvector_edge_tenancy::ProblemCode::NotFound, None);
    };
    let mut resp = Response::from_body(body.response_body()?)?;
    let h = resp.headers_mut();
    h.set("Content-Type", "application/octet-stream")?;
    h.set("Content-Length", &d.size.to_string())?;
    h.set(
        "Content-Disposition",
        &format!("attachment; filename=\"{}\"", d.filename),
    )?;
    h.set("Cache-Control", "no-store")?;
    for (k, v) in respond::RESPONSE_CORS {
        h.set(k, v)?;
    }
    Ok(resp)
}
