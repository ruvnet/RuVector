//! `ruvector-edge-auth` authorization-server Worker.
//!
//! Glue over `ruvector-edge-authz`. Stateless documents (RFC 8414 metadata,
//! JWKS) are served directly; every stateful endpoint (register, authorize,
//! callback, token, revoke) is forwarded to the single `AuthStore` Durable
//! Object, whose synchronous SQLite backs the authz storage ports.
//!
//! No changes to `auth.cognitum.one` are needed: this Worker is an ordinary
//! OAuth client of it (authorization code + PKCE S256) and mints its own
//! ES256 access tokens with `aud` = the requested resource URL.

#![forbid(unsafe_code)]

mod config;
mod platform;
mod sql_ports;
mod store;

pub use store::AuthStore;

use ruvector_edge_authz::metadata::{paths, AuthorizationServerMetadata};
use ruvector_edge_authz::{OAuthError, OAuthErrorCode};
use worker::{event, Context, Env, Headers, Method, Request, Response, Result};

/// Name of the one global `AuthStore` instance (`idFromName`).
pub const AUTH_STORE_NAME: &str = "auth-store-v1";

/// Worker entry point.
#[event(fetch)]
async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    console_error_panic_hook::set_once();
    let cfg = match config::AuthConfig::from_env(&env) {
        Ok(cfg) => cfg,
        Err(e) => return oauth_error(&e),
    };
    let path = req.path();
    match (req.method(), path.as_str()) {
        (Method::Get, paths::METADATA) => Response::from_json(&AuthorizationServerMetadata::build(
            &cfg.issuer,
            &cfg.scopes_supported,
        )),
        (Method::Get, paths::JWKS) => Response::from_json(&platform::public_jwks(&env)),
        (Method::Post, paths::REGISTER)
        | (Method::Get, paths::AUTHORIZE)
        | (Method::Get, paths::CALLBACK)
        | (Method::Post, paths::TOKEN)
        | (Method::Post, paths::REVOKE) => {
            let stub = env
                .durable_object("AUTH_STORE")?
                .id_from_name(AUTH_STORE_NAME)?
                .get_stub()?;
            stub.fetch_with_request(req).await
        }
        _ => Response::error("not found", 404),
    }
}

/// RFC 6749 JSON error with `Cache-Control: no-store`.
pub(crate) fn oauth_error(e: &OAuthError) -> Result<Response> {
    let headers = Headers::new();
    headers.set("Content-Type", "application/json")?;
    headers.set("Cache-Control", "no-store")?;
    Ok(Response::ok(e.to_json())?
        .with_status(e.error.http_status())
        .with_headers(headers))
}

/// Configuration failure as an OAuth error.
pub(crate) fn config_error(what: &'static str) -> OAuthError {
    OAuthError::new(OAuthErrorCode::ServerError, what)
}
