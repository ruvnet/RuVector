//! rv-gateway Worker (ADR-351 M1 router).
//!
//! Thin workers-rs glue over the pure cores: `ruvector-edge-auth` verifies
//! bearer tokens (edge-AS-issued, `aud` = exact resource URL; upstream
//! first-party tokens only behind `UPSTREAM_FIRST_PARTY=true`) and
//! `ruvector-edge-tenancy` derives the tenant. Routes (M1):
//!
//! - `GET /v1/health` -> `{"ok":true}` (anonymous, no version string)
//! - `GET /.well-known/oauth-protected-resource[/v1|/v1/mcp]` -> RFC 9728
//! - `GET /v1/me` -> tenant identity (any valid token)
//! - everything else under `/v1` -> 401 without a token, otherwise 404
//!
//! Every 401 carries `WWW-Authenticate: Bearer resource_metadata="..."`.

#![forbid(unsafe_code)]

mod config;
mod platform;
mod respond;
mod routes;

use worker::{event, Context, Env, Request, Response, Result};

/// Worker entry point.
#[event(fetch)]
async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    console_error_panic_hook::set_once();
    let cfg = match config::GatewayConfig::from_env(&env) {
        Ok(cfg) => cfg,
        Err(_) => return respond::problem(ruvector_edge_tenancy::ProblemCode::ServerError, None),
    };
    routes::handle(req, &cfg).await
}
