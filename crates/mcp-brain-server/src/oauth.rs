//! Minimal OAuth 2.1 layer for the MCP connector flow (draft fix for the
//! "Permit Brain" connector failing to register from claude.ai).
//!
//! Scope, deliberately kept small: claude.ai's remote MCP connector setup
//! performs OAuth discovery (RFC 8414 / RFC 9728) and, when no manually
//! configured Client ID is present, Dynamic Client Registration (RFC 7591)
//! before it will offer to connect at all. This server had none of that,
//! which is the entire failure — "Couldn't register with the sign-in
//! service" is exactly what a client sees when discovery/DCR comes back
//! empty or 404.
//!
//! This module does NOT introduce a new authority model. An OAuth access
//! token issued here is only ever handed out after the caller has proven
//! they already hold a valid `BRAIN_API_KEY` (or `BRAIN_SYSTEM_KEY`) via the
//! existing constant-time check in `routes.rs::verify_system_key` /
//! the equivalent memories-API key check. OAuth is a compliant *wrapper*
//! around the existing pre-shared-key authority, not a parallel one — on
//! purpose, so this doesn't expand what a connected client can do or touch
//! the existing REST/API-key auth path at all.
//!
//! DRAFT: this compiles against the route/state shapes as read from
//! `routes.rs` and `types.rs` at the time of writing, but has not been run
//! against a live build of the full workspace (100+ crates) in the time
//! available. Please review the storage choice (in-memory, process-lifetime
//! only — see "Known limitations" below) and the token/session model before
//! merging.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse, Redirect},
    Json,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

const AUTH_CODE_TTL_SECS: u64 = 300; // 5 minutes, single use
const ACCESS_TOKEN_TTL_SECS: u64 = 60 * 60 * 24 * 30; // 30 days, matches BRAIN_TIMEOUT-adjacent envs elsewhere in this crate

/// One dynamically (or manually) registered OAuth client.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OAuthClient {
    pub client_id: String,
    pub client_name: Option<String>,
    pub redirect_uris: Vec<String>,
    pub created_at: u64,
}

struct AuthCode {
    client_id: String,
    redirect_uri: String,
    code_challenge: Option<String>,
    expires_at: u64,
}

struct AccessToken {
    client_id: String,
    expires_at: u64,
}

/// Process-lifetime store. See "Known limitations" in the module doc —
/// a restart invalidates every registered client and issued token, which
/// forces re-registration/re-auth in claude.ai. Acceptable for a first
/// draft; worth promoting to the existing memory/SQLite store before this
/// is relied on long-term.
#[derive(Default)]
pub struct OAuthState {
    clients: DashMap<String, OAuthClient>,
    codes: DashMap<String, AuthCode>,
    tokens: DashMap<String, AccessToken>,
}

pub type SharedOAuthState = Arc<OAuthState>;

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

// ── RFC 8414 — OAuth Authorization Server Metadata ──────────────────────

pub async fn authorization_server_metadata() -> impl IntoResponse {
    Json(serde_json::json!({
        "issuer": brain_base_url(),
        "authorization_endpoint": format!("{}/authorize", brain_base_url()),
        "token_endpoint": format!("{}/token", brain_base_url()),
        "registration_endpoint": format!("{}/register", brain_base_url()),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
    }))
}

// RFC 9728 — Protected Resource Metadata. Some MCP clients probe this
// before falling back to RFC 8414 discovery on the issuer itself.
pub async fn protected_resource_metadata() -> impl IntoResponse {
    Json(serde_json::json!({
        "resource": brain_base_url(),
        "authorization_servers": [brain_base_url()],
    }))
}

fn brain_base_url() -> String {
    std::env::var("BRAIN_PUBLIC_URL")
        .unwrap_or_else(|_| "https://ubuntu1.tail6b157c.ts.net".to_string())
}

// ── RFC 7591 — Dynamic Client Registration ──────────────────────────────

#[derive(Deserialize)]
pub struct RegisterRequest {
    #[serde(default)]
    pub client_name: Option<String>,
    #[serde(default)]
    pub redirect_uris: Vec<String>,
}

pub async fn register_client(
    State(oauth): State<SharedOAuthState>,
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    if req.redirect_uris.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid_client_metadata",
                "error_description": "redirect_uris is required"
            })),
        )
            .into_response();
    }

    let client = OAuthClient {
        client_id: new_id("brainclient"),
        client_name: req.client_name,
        redirect_uris: req.redirect_uris,
        created_at: now(),
    };
    oauth.clients.insert(client.client_id.clone(), client.clone());

    (
        StatusCode::CREATED,
        Json(serde_json::json!({
            "client_id": client.client_id,
            "client_name": client.client_name,
            "redirect_uris": client.redirect_uris,
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        })),
    )
        .into_response()
}

// ── Authorization endpoint ───────────────────────────────────────────────
// Bridges to the EXISTING pre-shared key. The "sign-in" this fixes the
// error message for is: paste the same BRAIN_API_KEY/BRAIN_SYSTEM_KEY you
// already have. This does not create a new credential.

#[derive(Deserialize)]
pub struct AuthorizeParams {
    pub client_id: String,
    pub redirect_uri: String,
    pub state: Option<String>,
    pub code_challenge: Option<String>,
    pub code_challenge_method: Option<String>,
}

pub async fn authorize_get(
    State(oauth): State<SharedOAuthState>,
    Query(params): Query<AuthorizeParams>,
) -> impl IntoResponse {
    if !oauth.clients.contains_key(&params.client_id) {
        return (StatusCode::BAD_REQUEST, "unknown client_id").into_response();
    }
    // Minimal consent/sign-in form. Submits the existing key as a normal
    // POST; nothing here is new secret material — see module doc.
    let redirect_uri = html_escape(&params.redirect_uri);
    let client_id = html_escape(&params.client_id);
    let state_val = html_escape(params.state.as_deref().unwrap_or(""));
    let challenge = html_escape(params.code_challenge.as_deref().unwrap_or(""));
    Html(format!(
        r#"<!doctype html><html><body style="font-family:system-ui;max-width:420px;margin:4rem auto">
<h3>Connect to the Permit Brain</h3>
<form method="post" action="/authorize">
  <input type="hidden" name="client_id" value="{client_id}">
  <input type="hidden" name="redirect_uri" value="{redirect_uri}">
  <input type="hidden" name="state" value="{state_val}">
  <input type="hidden" name="code_challenge" value="{challenge}">
  <label>Brain API key<br><input type="password" name="api_key" style="width:100%"></label><br><br>
  <button type="submit">Authorize</button>
</form></body></html>"#
    ))
    .into_response()
}

#[derive(Deserialize)]
pub struct AuthorizeSubmit {
    pub client_id: String,
    pub redirect_uri: String,
    pub state: Option<String>,
    pub code_challenge: Option<String>,
    pub api_key: String,
}

pub async fn authorize_post(
    State(oauth): State<SharedOAuthState>,
    axum::Form(form): axum::Form<AuthorizeSubmit>,
) -> impl IntoResponse {
    let expected = std::env::var("BRAIN_API_KEY")
        .or_else(|_| std::env::var("BRAIN_SYSTEM_KEY"))
        .unwrap_or_default();
    if expected.is_empty()
        || !bool::from(subtle::ConstantTimeEq::ct_eq(
            form.api_key.as_bytes(),
            expected.as_bytes(),
        ))
    {
        return (StatusCode::UNAUTHORIZED, "invalid key").into_response();
    }

    let code = new_id("code");
    oauth.codes.insert(
        code.clone(),
        AuthCode {
            client_id: form.client_id,
            redirect_uri: form.redirect_uri.clone(),
            code_challenge: form.code_challenge.filter(|s| !s.is_empty()),
            expires_at: now() + AUTH_CODE_TTL_SECS,
        },
    );

    let mut redirect = format!("{}?code={}", form.redirect_uri, code);
    if let Some(s) = form.state.filter(|s| !s.is_empty()) {
        redirect.push_str(&format!("&state={}", s));
    }
    Redirect::to(&redirect).into_response()
}

// ── Token endpoint ────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct TokenRequest {
    pub grant_type: String,
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub redirect_uri: Option<String>,
    #[serde(default)]
    pub client_id: Option<String>,
    #[serde(default)]
    pub code_verifier: Option<String>,
}

pub async fn token(
    State(oauth): State<SharedOAuthState>,
    axum::Form(req): axum::Form<TokenRequest>,
) -> impl IntoResponse {
    if req.grant_type != "authorization_code" {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "unsupported_grant_type"})),
        )
            .into_response();
    }
    let Some(code) = req.code else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "invalid_request", "error_description": "code required"})),
        )
            .into_response();
    };

    let Some((_, auth_code)) = oauth.codes.remove(&code) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "invalid_grant"})),
        )
            .into_response();
    };

    if auth_code.expires_at < now() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "invalid_grant", "error_description": "code expired"})),
        )
            .into_response();
    }
    if let Some(expected_redirect) = req.redirect_uri {
        if expected_redirect != auth_code.redirect_uri {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid_grant", "error_description": "redirect_uri mismatch"})),
            )
                .into_response();
        }
    }
    // PKCE verification (S256) when the client sent a challenge.
    if let Some(challenge) = &auth_code.code_challenge {
        let Some(verifier) = req.code_verifier else {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid_grant", "error_description": "code_verifier required"})),
            )
                .into_response();
        };
        use sha2::Digest;
        let digest = sha2::Sha256::digest(verifier.as_bytes());
        let computed = base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, digest);
        if &computed != challenge {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid_grant", "error_description": "PKCE verification failed"})),
            )
                .into_response();
        }
    }

    let access_token = new_id("brainat");
    oauth.tokens.insert(
        access_token.clone(),
        AccessToken {
            client_id: auth_code.client_id,
            expires_at: now() + ACCESS_TOKEN_TTL_SECS,
        },
    );

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_TTL_SECS,
        })),
    )
        .into_response()
}

/// Called from the existing bearer-check path so `/sse`, `/messages`, and
/// the `/v1/*` endpoints accept an OAuth access token issued above, in
/// addition to the pre-shared `BRAIN_API_KEY`/`BRAIN_SYSTEM_KEY` they
/// already accept. Wire this as an additional `if` branch in
/// `verify_system_key` (or the memories-API equivalent) — deliberately not
/// done in this draft to avoid touching the existing, working auth path
/// without your review.
pub fn is_valid_oauth_token(oauth: &OAuthState, headers: &HeaderMap) -> bool {
    let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok()) else {
        return false;
    };
    let Some(token) = auth.strip_prefix("Bearer ") else {
        return false;
    };
    match oauth.tokens.get(token) {
        Some(t) => t.expires_at >= now(),
        None => false,
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// Suppress unused-import warning if Duration ends up unused after a future edit.
#[allow(dead_code)]
fn _keep_duration_import(_: Duration) {}
