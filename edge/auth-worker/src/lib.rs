//! `ruvector-edge-auth` authorization-server Worker.
//!
//! Glue over `ruvector-edge-authz`. Stateless documents (RFC 8414 metadata,
//! JWKS) are served directly; every stateful endpoint (register, authorize,
//! callback, token, revoke) is forwarded to the single `AuthStore` Durable
//! Object, whose synchronous SQLite backs the authz storage ports.
//!
//! No changes to `auth.cognitum.one` are needed: this Worker is an ordinary
//! public OAuth client of it (authorization code + PKCE S256) and mints its
//! own ES256 access tokens with `aud` = the requested resource URL. The
//! signing key comes from the `EDGE_AUTH_SIGNING_JWK` secret.

#![forbid(unsafe_code)]

mod abuse;
mod config;
mod endpoints;
mod http;
mod platform;
mod signer;
mod sql;
mod sql_ports;
mod store;
mod upstream;

#[cfg(test)]
mod testutil;

pub use store::AuthStore;

use http::Reply;
use ruvector_edge_authz::metadata::{jwks_document, paths, AuthorizationServerMetadata};
use ruvector_edge_authz::{OAuthError, OAuthErrorCode};
use worker::{event, Context, Env, Method, Request, Response, Result};

/// Name of the one global `AuthStore` instance (`idFromName`).
pub const AUTH_STORE_NAME: &str = "auth-store-v1";

/// Seconds public documents may be cached by clients and resource servers.
const DOC_MAX_AGE: &str = "public, max-age=300";

/// Worker entry point.
#[event(fetch)]
async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    console_error_panic_hook::set_once();
    let cfg = match config::AuthConfig::from_env(&env) {
        Ok(cfg) => cfg,
        Err(e) => return store::to_response(Reply::oauth_error(&e), false),
    };
    let path = req.path();
    let public = matches!(
        path.as_str(),
        paths::METADATA | paths::JWKS | paths::REGISTER | paths::TOKEN | paths::REVOKE
    );
    match (req.method(), path.as_str()) {
        (Method::Options, _) if public => store::to_response(Reply::empty(204), true),
        (Method::Get, paths::METADATA) => {
            let doc = AuthorizationServerMetadata::build(&cfg.issuer, &cfg.scopes_supported);
            public_doc(Reply::json(200, &doc))
        }
        (Method::Get, paths::JWKS) => {
            let secret = env
                .secret(signer::SIGNING_KEY_SECRET)
                .ok()
                .map(|s| s.to_string());
            let keys = signer::EnvSigner::from_secret(secret.as_deref()).public_keys();
            if keys.is_empty() {
                return store::to_response(
                    Reply::oauth_error(&OAuthError::new(
                        OAuthErrorCode::TemporarilyUnavailable,
                        "signing key not configured",
                    )),
                    true,
                );
            }
            public_doc(Reply::json(200, &jwks_document(&keys)))
        }
        (Method::Post, paths::REGISTER)
        | (Method::Get, paths::AUTHORIZE)
        | (Method::Post, endpoints::consent::CONSENT_PATH)
        | (Method::Get, paths::CALLBACK)
        | (Method::Post, paths::TOKEN)
        | (Method::Post, paths::REVOKE) => {
            // Everything below runs before the single global DO is touched.
            if req.method() == Method::Post {
                let length = req.headers().get("Content-Length").ok().flatten();
                if let Err(r) = http::check_body_length(&path, length.as_deref()) {
                    return store::to_response(r, true);
                }
            }
            if let Some(r) = throttle(&req, &env, &path).await {
                let cors = matches!(
                    path.as_str(),
                    paths::REGISTER | paths::TOKEN | paths::REVOKE
                );
                return store::to_response(r, cors);
            }
            let stub = env
                .durable_object("AUTH_STORE")?
                .id_from_name(AUTH_STORE_NAME)?
                .get_stub()?;
            stub.fetch_with_request(req).await
        }
        _ => Response::error("not found", 404),
    }
}

/// Workers Rate Limiting binding that guards `path`, if any (ADR-351 §5.6).
fn limiter_for(path: &str) -> Option<&'static str> {
    match path {
        paths::REGISTER => Some("DCR_RATE_LIMITER"),
        paths::AUTHORIZE | paths::TOKEN | paths::REVOKE | endpoints::consent::CONSENT_PATH => {
            Some("AUTHZ_RATE_LIMITER")
        }
        _ => None,
    }
}

/// Per-IP throttle before the DO hop. Optional: an absent binding, or a
/// limiter error, is skipped (the DO keeps its durable DCR window), so a
/// misconfiguration never locks out login.
async fn throttle(req: &Request, env: &Env, path: &str) -> Option<Reply> {
    let limiter = env.rate_limiter(limiter_for(path)?).ok()?;
    let salt = env
        .secret(abuse::IP_HASH_SALT_SECRET)
        .map(|s| s.to_string())
        .unwrap_or_default();
    let ip = req.headers().get(abuse::CLIENT_IP_HEADER).ok().flatten();
    let key = format!("{path}|{}", abuse::ip_bucket(ip.as_deref(), &salt));
    match limiter.limit(key).await {
        Ok(outcome) if !outcome.success => Some(abuse::rate_limited(60)),
        _ => None,
    }
}

/// A cacheable public document (replaces `no-store`) with CORS.
fn public_doc(mut reply: Reply) -> Result<Response> {
    for h in reply.headers.iter_mut() {
        if h.0 == "Cache-Control" {
            h.1 = DOC_MAX_AGE.to_string();
        }
    }
    store::to_response(reply, true)
}

/// Configuration failure as an OAuth error.
pub(crate) fn config_error(what: &'static str) -> OAuthError {
    OAuthError::new(OAuthErrorCode::ServerError, what)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_stateful_endpoint_but_the_callback_is_throttled() {
        assert_eq!(limiter_for(paths::REGISTER), Some("DCR_RATE_LIMITER"));
        for p in [
            paths::AUTHORIZE,
            paths::TOKEN,
            paths::REVOKE,
            endpoints::consent::CONSENT_PATH,
        ] {
            assert_eq!(limiter_for(p), Some("AUTHZ_RATE_LIMITER"));
        }
        // The callback completes a flow already paid for at /authorize.
        assert_eq!(limiter_for(paths::CALLBACK), None);
        assert_eq!(limiter_for(paths::JWKS), None);
    }
}
