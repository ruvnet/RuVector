//! Workers glue for the registry routes: reads the body under the route's
//! cap (a part upload up to `max_part_size`, every other route 1 MiB),
//! runs `registry_routes::handle` over the DO and R2 ports and renders the
//! reply; a pull streams the R2 object straight into the response.

use crate::api::{read_capped, MAX_BODY_BYTES};
use crate::blob_r2::{R2, R2_BINDING};
use crate::config::GatewayConfig;
use crate::durable::DoBackend;
use crate::registry_core::gateway_config;
use crate::registry_do::DoRegistry;
use crate::registry_routes::{self, Query, RvfReply, RvfRoute};
use crate::registry_wire::RvfError;
use crate::respond;
use crate::rest::Caller;
use crate::rvf_finalize_step::INLINE_FINALIZE_BYTES;
use crate::rvf_upload::Deps;
use ruvector_edge_auth::prm;
use ruvector_edge_tenancy::ProblemCode;
use worker::{Env, Headers, Method, Request, Response, ResponseBody, Result};

/// Largest request body a registry route accepts (the pre-auth
/// `Content-Length` check uses it too).
pub fn body_cap(route: &RvfRoute) -> usize {
    match route {
        RvfRoute::Part(..) => {
            usize::try_from(gateway_config().upload.max_part_size).unwrap_or(usize::MAX)
        }
        _ => MAX_BODY_BYTES,
    }
}

fn render(r: crate::rest::ApiReply) -> Result<Response> {
    respond::raw(
        r.status,
        r.body,
        r.content_type,
        r.www_authenticate.as_deref(),
    )
}

fn query(req: &Request) -> Result<Query> {
    let url = req.url()?;
    let mut q = Query::default();
    for (k, v) in url.query_pairs() {
        match &*k {
            "cursor" => q.cursor = Some(v.into_owned()),
            "limit" => q.limit = v.parse().ok(),
            _ => {}
        }
    }
    Ok(q)
}

/// Serve a verified registry request.
pub async fn serve(
    mut req: Request,
    env: &Env,
    cfg: &GatewayConfig,
    route: RvfRoute,
    caller: Caller,
    now: u64,
) -> Result<Response> {
    let md = prm::metadata_url(&cfg.rest_resource);
    let body = if matches!(req.method(), Method::Get | Method::Head) || req.inner().body().is_none()
    {
        Some(Vec::new())
    } else {
        read_capped(req.stream()?, body_cap(&route)).await?
    };
    let Some(body) = body else {
        let e = RvfError::new(ProblemCode::PayloadTooLarge, "body too large");
        return render(e.reply(&md));
    };
    let q = query(&req)?;
    let Ok(bucket) = env.bucket(R2_BINDING) else {
        return render(RvfError::storage().reply(&md));
    };
    let r2 = R2(bucket);
    let deps = Deps {
        b: &DoBackend { env },
        r: &DoRegistry { env },
        s: &r2,
        cfg: gateway_config(),
        inline_finalize: INLINE_FINALIZE_BYTES,
        finalize_steps: crate::rvf_finalize::FINALIZE_STEPS,
    };
    match registry_routes::handle(&deps, &caller.ctx, &route, body, &q, now, &md).await {
        RvfReply::Api(r) => render(r),
        RvfReply::Blob {
            key,
            size,
            sha256,
            yanked,
        } => stream(&r2, &key, size, &sha256, yanked, &md).await,
    }
}

async fn stream(
    r2: &R2,
    key: &str,
    size: u64,
    sha256: &str,
    yanked: bool,
    md: &str,
) -> Result<Response> {
    let obj = match r2.0.get(key).execute().await {
        Ok(Some(o)) if o.size() == size => o,
        _ => return render(RvfError::storage().reply(md)),
    };
    let Some(body) = obj.body() else {
        return render(RvfError::storage().reply(md));
    };
    let headers = Headers::new();
    for (k, v) in respond::RESPONSE_CORS {
        headers.set(k, v)?;
    }
    headers.set("Content-Type", "application/octet-stream")?;
    headers.set("Content-Length", &size.to_string())?;
    headers.set("ETag", &format!("\"sha256:{sha256}\""))?;
    headers.set("Cache-Control", "no-store")?;
    if yanked {
        headers.set("X-RVF-Yanked", "true")?;
    }
    let stream: ResponseBody = body.response_body()?;
    Ok(Response::from_body(stream)?.with_headers(headers))
}

/// Pre-auth `Content-Length` check (ADR-351 §10 layer 1) with the
/// registry's per-route cap: a part upload may declare up to
/// `max_part_size`, every other route 1 MiB.
pub fn declared_too_large(req: &Request) -> bool {
    if !registry_routes::is_part_upload(&req.method(), &req.path()) {
        return crate::api::declared_too_large(req);
    }
    let cap = usize::try_from(gateway_config().upload.max_part_size).unwrap_or(usize::MAX);
    req.headers()
        .get("Content-Length")
        .ok()
        .flatten()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .is_some_and(|n| n > cap)
}

/// `POST /v1/mcp` with the registry tools (`rvf_mcp`) next to the M1 ones.
pub async fn mcp(
    env: &Env,
    ctx: &ruvector_edge_store::CallerContext,
    body: &[u8],
    now: u64,
    metadata_url: &str,
) -> crate::rest::ApiReply {
    let backend = DoBackend { env };
    let Ok(bucket) = env.bucket(R2_BINDING) else {
        return crate::mcp::handle(&backend, ctx, body, now, metadata_url).await;
    };
    let r2 = R2(bucket);
    let deps = Deps {
        b: &backend,
        r: &DoRegistry { env },
        s: &r2,
        cfg: gateway_config(),
        inline_finalize: INLINE_FINALIZE_BYTES,
        finalize_steps: crate::rvf_finalize::FINALIZE_STEPS,
    };
    let tools = crate::rvf_mcp::RvfTools(&deps);
    crate::mcp::handle_with(&backend, &tools, ctx, body, now, metadata_url).await
}
