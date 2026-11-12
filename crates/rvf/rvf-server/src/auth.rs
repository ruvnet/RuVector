//! Bearer-token authentication for the HTTP surface.
//!
//! When a token is configured every route requires
//! `Authorization: Bearer <token>`, except `GET /v1/health`, which stays open
//! so load balancers and liveness probes can reach it without the secret.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::Router;
use subtle::ConstantTimeEq;

/// Environment variable the binaries read the token from, so it never has
/// to appear on a command line or in shell history.
pub const TOKEN_ENV: &str = "RVF_SERVER_TOKEN";

/// Route left open when a token is configured.
pub const HEALTH_PATH: &str = "/v1/health";

/// A bearer token. `Debug` is redacted so the secret cannot reach logs.
#[derive(Clone)]
pub struct AuthToken(Arc<str>);

impl AuthToken {
    /// Wrap a token; an empty or all-whitespace value means "no token".
    pub fn new(token: impl AsRef<str>) -> Option<Self> {
        let t = token.as_ref().trim();
        (!t.is_empty()).then(|| Self(Arc::from(t)))
    }

    /// Read the token from [`TOKEN_ENV`], if set and non-empty.
    pub fn from_env() -> Option<Self> {
        std::env::var(TOKEN_ENV).ok().and_then(Self::new)
    }

    /// Constant-time comparison against a presented credential.
    pub fn matches(&self, presented: &[u8]) -> bool {
        self.0.as_bytes().ct_eq(presented).into()
    }
}

impl std::fmt::Debug for AuthToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AuthToken(<redacted>)")
    }
}

/// Wrap `router` so every route except [`HEALTH_PATH`] requires the token.
/// With no token the router is returned unchanged.
pub fn protect(router: Router, token: Option<AuthToken>) -> Router {
    match token {
        Some(t) => router.layer(middleware::from_fn_with_state(t, require_bearer)),
        None => router,
    }
}

async fn require_bearer(State(token): State<AuthToken>, req: Request, next: Next) -> Response {
    if req.uri().path() == HEALTH_PATH {
        return next.run(req).await;
    }
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.as_bytes().strip_prefix(b"Bearer "));
    match presented {
        Some(p) if token.matches(p) => next.run(req).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "unauthorized",
        )
            .into_response(),
    }
}
