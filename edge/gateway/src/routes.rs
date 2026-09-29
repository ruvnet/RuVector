//! M1 router.

use crate::config::GatewayConfig;
use crate::platform::{WorkerClock, WorkerFetch};
use crate::respond;
use ruvector_edge_auth::prm::{self, ProtectedResourceMetadata, PRM_WELL_KNOWN};
use ruvector_edge_auth::{
    bearer_token, AuthError, ClaimsPolicy, JwksCache, JwksCachePolicy, ResourceUrl, RouteSurface,
    Verifier,
};
use ruvector_edge_tenancy::{ProblemCode, TenantContext};
use serde::Serialize;
use worker::{Method, Request, Response, Result};

/// Dispatch one request.
pub async fn handle(req: Request, cfg: &GatewayConfig) -> Result<Response> {
    let path = req.path();
    let method = req.method();
    match (method, path.as_str()) {
        (Method::Get, "/v1/health") => respond::json(&Health { ok: true }, false),
        (Method::Get, p) if p == PRM_WELL_KNOWN || p == format!("{PRM_WELL_KNOWN}/v1") => {
            respond::json(
                &ProtectedResourceMetadata::new(&cfg.rest_resource, &cfg.edge_issuer),
                false,
            )
        }
        (Method::Get, p) if p == format!("{PRM_WELL_KNOWN}/v1/mcp") => respond::json(
            &ProtectedResourceMetadata::new(&cfg.mcp_resource, &cfg.edge_issuer),
            false,
        ),
        (Method::Get, "/v1/me") => me(&req, cfg).await,
        (_, p) if p.starts_with("/v1/") || p == "/v1" => {
            // No optional auth: authenticate first so unknown routes don't
            // leak existence to anonymous callers, then 404.
            match authenticate(&req, cfg, &cfg.rest_resource, RouteSurface::Rest).await {
                Ok(_) => respond::problem(ProblemCode::NotFound, None),
                Err(resp) => Ok(resp),
            }
        }
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

async fn me(req: &Request, cfg: &GatewayConfig) -> Result<Response> {
    let ctx = match authenticate(req, cfg, &cfg.rest_resource, RouteSurface::Rest).await {
        Ok(ctx) => ctx,
        Err(resp) => return Ok(resp),
    };
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
    cfg: &GatewayConfig,
    resource: &ResourceUrl,
    surface: RouteSurface,
) -> std::result::Result<TenantContext, Response> {
    let metadata_url = prm::metadata_url(resource);
    let header = req.headers().get("Authorization").ok().flatten();
    if let Err(AuthError::MissingToken) = bearer_token(header.as_deref()) {
        return Err(challenge(
            ProblemCode::InvalidToken,
            Some(prm::www_authenticate_missing(&metadata_url)),
        ));
    }
    // TODO(M1 auth stage): hoist the caches into an isolate-global so the
    // 10-minute in-isolate TTL (ADR §5.1.5) actually spans requests.
    let edge_keys = JwksCache::new(
        WorkerFetch,
        WorkerClock,
        JwksCachePolicy::with_defaults(cfg.edge_jwks_url.clone()),
    );
    let mut verifier = Verifier::edge_only(
        edge_keys,
        WorkerClock,
        cfg.audience_for(resource),
        ClaimsPolicy::with_defaults(cfg.edge_issuer.clone()),
    );
    if let Some(up) = &cfg.upstream {
        let keys = JwksCache::new(
            WorkerFetch,
            WorkerClock,
            JwksCachePolicy::with_defaults(cfg.upstream_jwks_url.clone()),
        );
        let mut claims = ClaimsPolicy::with_defaults(up.issuer.clone());
        claims.typ_claim = Some("access".into());
        verifier = verifier.with_upstream(keys, claims);
    }
    let claims = match verifier.verify(header.as_deref()).await {
        Ok(c) => c,
        Err(e) => {
            let code = match e.http_status() {
                403 => ProblemCode::AudienceNotAllowed,
                503 => ProblemCode::JwksUnavailable,
                500 => ProblemCode::ServerError,
                _ => ProblemCode::InvalidToken,
            };
            let www = (code == ProblemCode::InvalidToken)
                .then(|| prm::www_authenticate_invalid_token(&metadata_url));
            return Err(challenge(code, www));
        }
    };
    TenantContext::from_verified(claims, surface).map_err(|_| {
        challenge(
            ProblemCode::InvalidToken,
            Some(prm::www_authenticate_invalid_token(&metadata_url)),
        )
    })
}

fn challenge(code: ProblemCode, www: Option<String>) -> Response {
    respond::problem(code, www).unwrap_or_else(|_| {
        Response::error("server_error", 500)
            .unwrap_or_else(|_| Response::empty().expect("empty response"))
    })
}
