//! M1 router. Route classification, the per-route authentication target
//! and the verification-error mapping are pure functions (tested natively);
//! `handle` is the Workers glue.

use crate::config::GatewayConfig;
use crate::platform::{JwksFetch, WorkerClock, EDGE_AUTH_BINDING};
use crate::{auth, keys, respond};
use ruvector_edge_auth::prm::{ProtectedResourceMetadata, MCP_SCOPES, PRM_WELL_KNOWN, REST_SCOPES};
use ruvector_edge_auth::{ResourceUrl, RouteSurface};
use ruvector_edge_tenancy::{ProblemCode, TenantContext};
use serde::Serialize;
use worker::{Env, Method, Request, Response, Result};

/// What a request is, before any authentication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// `GET /v1/health` (anonymous).
    Health,
    /// RFC 9728 metadata for the REST resource (`…/oauth-protected-resource/v1`).
    PrmRest,
    /// RFC 9728 metadata for the MCP resource (`…/oauth-protected-resource/v1/mcp`).
    PrmMcp,
    /// `GET /v1/me`.
    Me,
    /// The MCP resource `/v1/mcp[/…]` (any method but `OPTIONS`):
    /// authenticated against the MCP resource and surface, then 404 until
    /// the MCP handler lands.
    Mcp,
    /// CORS preflight (public documents and every `/v1` path; never
    /// authenticated).
    Preflight,
    /// Any other `/v1` path: authenticate against the REST resource, then 404.
    OtherV1,
    /// Outside the API surface (including the bare
    /// `/.well-known/oauth-protected-resource`, ADR-351 §5.7: its RFC 9728
    /// `resource` would have to be the bare origin, which is not a resource).
    NotFound,
}

fn is_under(path: &str, prefix: &str) -> bool {
    path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|r| r.starts_with('/'))
}

/// Classify by method and path (exact matches, no normalisation).
pub fn classify(method: &Method, path: &str) -> Route {
    let prm = path.strip_prefix(PRM_WELL_KNOWN);
    let prm_rest = prm == Some("/v1");
    let prm_mcp = prm == Some("/v1/mcp");
    let v1 = is_under(path, "/v1");
    match method {
        Method::Options if prm_rest || prm_mcp || v1 => Route::Preflight,
        Method::Get if path == "/v1/health" => Route::Health,
        Method::Get if prm_rest => Route::PrmRest,
        Method::Get if prm_mcp => Route::PrmMcp,
        Method::Get if path == "/v1/me" => Route::Me,
        _ if is_under(path, "/v1/mcp") => Route::Mcp,
        _ if v1 => Route::OtherV1,
        _ => Route::NotFound,
    }
}

/// The resource (exact `aud`) and surface a route authenticates against,
/// or `None` for anonymous routes. `/v1/mcp` is bound to the MCP resource,
/// so a `/v1` token is 403 there, its 401 names the `/v1/mcp` metadata, and
/// upstream first-party tokens are refused (REST-only, ADR-351 §5.5).
pub fn auth_target(route: Route, cfg: &GatewayConfig) -> Option<(&ResourceUrl, RouteSurface)> {
    match route {
        Route::Me | Route::OtherV1 => Some((&cfg.rest_resource, RouteSurface::Rest)),
        Route::Mcp => Some((&cfg.mcp_resource, RouteSurface::Mcp)),
        Route::Health | Route::PrmRest | Route::PrmMcp | Route::Preflight | Route::NotFound => None,
    }
}

/// Dispatch one request.
pub async fn handle(req: Request, env: &Env, cfg: &GatewayConfig) -> Result<Response> {
    let route = classify(&req.method(), &req.path());
    // Authenticate first so unknown routes do not leak existence to
    // anonymous callers.
    let ctx = match auth_target(route, cfg) {
        Some((resource, surface)) => match authenticate(&req, env, cfg, resource, surface).await {
            Ok(ctx) => Some(ctx),
            Err(resp) => return Ok(resp),
        },
        None => None,
    };
    match (route, ctx) {
        (Route::Preflight, _) => respond::preflight(),
        (Route::Health, _) => respond::json(&Health { ok: true }, false),
        (Route::PrmRest, _) => respond::public_json(&ProtectedResourceMetadata::new(
            &cfg.rest_resource,
            &cfg.edge_issuer,
            &REST_SCOPES,
        )),
        (Route::PrmMcp, _) => respond::public_json(&ProtectedResourceMetadata::new(
            &cfg.mcp_resource,
            &cfg.edge_issuer,
            &MCP_SCOPES,
        )),
        (Route::Me, Some(ctx)) => me(&ctx),
        _ => respond::problem(ProblemCode::NotFound, None),
    }
}

#[derive(Serialize)]
struct Health {
    ok: bool,
}

#[derive(Serialize)]
struct Me<'a> {
    tenant_key: &'a str,
    org_id: &'a str,
    workspace_id: &'a str,
    sub: &'a str,
    client_id: &'a str,
    capabilities: Vec<String>,
}

fn me(ctx: &TenantContext) -> Result<Response> {
    let body = Me {
        tenant_key: ctx.tenant_key().as_str(),
        org_id: ctx.org_id(),
        workspace_id: ctx.workspace_id(),
        sub: ctx.sub(),
        client_id: ctx.client_id(),
        capabilities: ctx
            .capabilities()
            .iter()
            .map(|c| format!("{c:?}").to_lowercase())
            .collect(),
    };
    respond::json(&body, true)
}

/// Verify the bearer token for `resource` and build the tenant context, or
/// produce the exact error response (401 + `resource_metadata`, 403, 503).
async fn authenticate(
    req: &Request,
    env: &Env,
    cfg: &GatewayConfig,
    resource: &ResourceUrl,
    surface: RouteSurface,
) -> std::result::Result<TenantContext, Response> {
    let header = req.headers().get("Authorization").ok().flatten();
    // Service Binding only (Cloudflare 1042): no public fallback.
    let edge_keys = keys::edge(&cfg.edge_jwks_url, || {
        edge_fetch(env.service(EDGE_AUTH_BINDING).ok())
    });
    let upstream_kids = cfg
        .upstream
        .as_ref()
        .map(|u| u.accepted_kids.as_slice())
        .unwrap_or_default();
    auth::authenticate(
        header.as_deref(),
        cfg,
        resource,
        surface,
        edge_keys,
        || keys::upstream(&cfg.upstream_jwks_url, upstream_kids),
        WorkerClock,
    )
    .await
    .map_err(|d| challenge(d.code, d.www_authenticate))
}

/// The edge JWKS fetcher: the `EDGE_AUTH` binding when present, otherwise
/// [`JwksFetch::Unbound`] (503 `jwks_unavailable`), never the public
/// internet.
pub fn edge_fetch(binding: Option<worker::Fetcher>) -> JwksFetch {
    binding.map_or(JwksFetch::Unbound, JwksFetch::Service)
}

fn challenge(code: ProblemCode, www: Option<String>) -> Response {
    respond::problem(code, www).unwrap_or_else(|_| {
        Response::error("server_error", 500)
            .unwrap_or_else(|_| Response::empty().expect("empty response"))
    })
}

#[cfg(test)]
#[path = "routes_tests.rs"]
mod tests;
