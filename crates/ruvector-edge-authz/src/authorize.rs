//! Authorization endpoint request validation (RFC 6749 §4.1.1 + RFC 7636 +
//! RFC 8707) and response construction (RFC 9207 `iss`).

use crate::client::ClientRecord;
use crate::error::{OAuthError, OAuthErrorCode};
use crate::params::Params;
use crate::resource::{grant_scopes, ResourceAllowlist};
use ruvector_edge_auth::ResourceUrl;
use serde::{Deserialize, Serialize};
use url::Url;

/// Raw query parameters of `GET /authorize`. Build it with
/// [`AuthorizationRequest::from_pairs`], which rejects repeated parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[allow(missing_docs)]
pub struct AuthorizationRequest {
    pub response_type: Option<String>,
    pub client_id: Option<String>,
    pub redirect_uri: Option<String>,
    pub scope: Option<String>,
    pub state: Option<String>,
    pub code_challenge: Option<String>,
    pub code_challenge_method: Option<String>,
    /// RFC 8707 target resource; required.
    pub resource: Option<String>,
}

impl AuthorizationRequest {
    /// Parse decoded query pairs. Errors here are never redirected (the
    /// redirect target is not yet trusted): [`AuthorizeError::UserAgent`].
    pub fn from_pairs(pairs: &[(String, String)]) -> Result<Self, AuthorizeError> {
        let p = Params::from_pairs(pairs).map_err(AuthorizeError::UserAgent)?;
        Ok(AuthorizationRequest {
            response_type: p.take("response_type"),
            client_id: p.take("client_id"),
            redirect_uri: p.take("redirect_uri"),
            scope: p.take("scope"),
            state: p.take("state"),
            code_challenge: p.take("code_challenge"),
            code_challenge_method: p.take("code_challenge_method"),
            resource: p.take("resource"),
        })
    }
}

/// A request that passed every check; persisted inside the federation flow
/// state while the user logs in upstream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidatedAuthorization {
    /// Registered client.
    pub client_id: String,
    /// Exact redirect URI to return to (as presented; matched a registration).
    pub redirect_uri: String,
    /// Granted scopes (requested ∩ client ceiling ∩ resource scopes).
    pub scopes: Vec<String>,
    /// Client `state` echoed back (opaque, bounded).
    pub state: Option<String>,
    /// PKCE S256 challenge.
    pub code_challenge: String,
    /// Target resource -> future `aud`.
    pub resource: ResourceUrl,
}

/// Maximum accepted `state` length.
pub const MAX_STATE_LEN: usize = 512;

/// How an authorization error must be delivered (RFC 6749 §4.1.2.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorizeError {
    /// Client or redirect URI is not trusted: render an error page, never
    /// redirect.
    UserAgent(OAuthError),
    /// Redirect back to the (verified) client with `error`.
    Redirect {
        /// Verified redirect URI.
        redirect_uri: String,
        /// Client `state` to echo.
        state: Option<String>,
        /// The error.
        error: OAuthError,
    },
}

impl AuthorizeError {
    /// The underlying OAuth error.
    pub fn oauth(&self) -> &OAuthError {
        match self {
            AuthorizeError::UserAgent(e) | AuthorizeError::Redirect { error: e, .. } => e,
        }
    }

    /// Redirect URL for [`AuthorizeError::Redirect`] (`error`,
    /// `error_description`, `state`, `iss`); `None` for user-agent errors.
    pub fn redirect_url(&self, issuer: &str) -> Option<String> {
        match self {
            AuthorizeError::UserAgent(_) => None,
            AuthorizeError::Redirect {
                redirect_uri,
                state,
                error,
            } => {
                let code = serde_json::to_value(error.error).ok()?;
                let code = code.as_str()?.to_string();
                let mut pairs = vec![
                    ("error", code),
                    ("error_description", error.error_description.to_string()),
                ];
                if let Some(s) = state {
                    pairs.push(("state", s.clone()));
                }
                pairs.push(("iss", issuer.to_string()));
                append_query(redirect_uri, &pairs)
            }
        }
    }
}

fn append_query(base: &str, pairs: &[(&str, String)]) -> Option<String> {
    let mut url = Url::parse(base).ok()?;
    {
        let mut q = url.query_pairs_mut();
        for (k, v) in pairs {
            q.append_pair(k, v);
        }
    }
    Some(url.into())
}

fn ua(code: OAuthErrorCode, desc: &'static str) -> AuthorizeError {
    AuthorizeError::UserAgent(OAuthError::new(code, desc))
}

/// Whether a (registered) redirect URI is verified — see
/// [`crate::client::RedirectUri::is_verified`]. Unparseable => unverified.
pub fn redirect_is_verified(redirect_uri: &str) -> bool {
    crate::client::validate_redirect_uri(redirect_uri).is_ok_and(|r| r.is_verified())
}

impl ValidatedAuthorization {
    /// Deliver an error that occurs **before** the user has interacted with
    /// the consent page (e.g. upstream not configured, flow storage failure):
    /// redirected only to a verified redirect URI, otherwise shown to the
    /// user agent. Prevents an open redirector via anonymous DCR
    /// (RFC 9700 §4.11.2).
    pub fn error(&self, error: OAuthError) -> AuthorizeError {
        if redirect_is_verified(&self.redirect_uri) {
            AuthorizeError::Redirect {
                redirect_uri: self.redirect_uri.clone(),
                state: self.state.clone(),
                error,
            }
        } else {
            AuthorizeError::UserAgent(error)
        }
    }

    /// Deliver an error **after** user interaction (the consent page's
    /// Cancel, or the upstream callback): always redirected, since the user
    /// saw the redirect host and its verified/unverified marker.
    pub fn error_after_consent(&self, error: OAuthError) -> AuthorizeError {
        AuthorizeError::Redirect {
            redirect_uri: self.redirect_uri.clone(),
            state: self.state.clone(),
            error,
        }
    }
}

/// Consent form token (ADR-351 §5.6): base64url of
/// `sha256("ruvector-edge/consent/v1|" ‖ cookie ‖ "|" ‖ request digest)`,
/// where the request digest is the JSON of the validated request. A
/// cross-site page cannot compute it without the `__Host-` cookie, so an
/// auto-submitted consent is refused. `cookie` is the browser secret.
pub fn consent_token(cookie: &str, auth: &ValidatedAuthorization) -> String {
    use sha2::{Digest, Sha256};
    let digest = serde_json::to_vec(auth).unwrap_or_default();
    let mut h = Sha256::new();
    h.update(b"ruvector-edge/consent/v1|");
    h.update(cookie.as_bytes());
    h.update(b"|");
    h.update(Sha256::digest(&digest));
    ruvector_edge_auth::jws::b64url_encode(&h.finalize())
}

/// Constant-time check of a presented consent form token. An empty cookie
/// never verifies.
pub fn verify_consent_token(cookie: &str, auth: &ValidatedAuthorization, presented: &str) -> bool {
    use subtle::ConstantTimeEq;
    !cookie.is_empty()
        && bool::from(
            consent_token(cookie, auth)
                .as_bytes()
                .ct_eq(presented.as_bytes()),
        )
}

/// Validate an authorization request against the registered client.
///
/// Error delivery: missing/mismatched `client_id`, missing or unregistered
/// `redirect_uri`, or an oversize/non-printable `state` -> user agent (never
/// redirected). Later errors are redirected **only if the redirect URI is
/// verified** ([`redirect_is_verified`]); for unverified targets every
/// pre-consent error goes to the user agent (no open redirect via DCR).
/// Checks: `response_type == "code"` else `unsupported_response_type`; PKCE
/// via [`crate::pkce::validate_challenge`] (`S256` only); `resource` via
/// [`ResourceAllowlist::resolve_entry`] (`invalid_target`, required); `scope`
/// via [`grant_scopes`] (ADR-351 §5.3: requested ∩ client ceiling ∩ the
/// resource's scopes; out-of-ceiling vocabulary scopes are dropped, unknown
/// ones `invalid_scope`; omitted -> the resource's default grant, e.g.
/// `ruvector:read`).
pub fn validate_authorization(
    req: &AuthorizationRequest,
    client: &ClientRecord,
    resources: &ResourceAllowlist,
) -> Result<ValidatedAuthorization, AuthorizeError> {
    if req.client_id.as_deref() != Some(client.client_id.as_str()) {
        return Err(ua(OAuthErrorCode::InvalidClient, "unknown client_id"));
    }
    let redirect_uri = req
        .redirect_uri
        .as_deref()
        .ok_or(ua(OAuthErrorCode::InvalidRequest, "redirect_uri required"))?;
    if client.match_redirect(redirect_uri).is_none() {
        return Err(ua(
            OAuthErrorCode::InvalidRequest,
            "redirect_uri not registered",
        ));
    }
    if let Some(s) = &req.state {
        if s.len() > MAX_STATE_LEN || !s.bytes().all(|b| b.is_ascii_graphic() || b == b' ') {
            return Err(ua(OAuthErrorCode::InvalidRequest, "invalid state"));
        }
    }
    let verified = redirect_is_verified(redirect_uri);
    let redirect = |error: OAuthError| {
        if verified {
            AuthorizeError::Redirect {
                redirect_uri: redirect_uri.to_string(),
                state: req.state.clone(),
                error,
            }
        } else {
            AuthorizeError::UserAgent(error)
        }
    };
    if req.response_type.as_deref() != Some("code") {
        return Err(redirect(OAuthError::new(
            OAuthErrorCode::UnsupportedResponseType,
            "response_type must be code",
        )));
    }
    let challenge = req.code_challenge.as_deref().ok_or_else(|| {
        redirect(OAuthError::new(
            OAuthErrorCode::InvalidRequest,
            "code_challenge required",
        ))
    })?;
    crate::pkce::validate_challenge(challenge, req.code_challenge_method.as_deref())
        .map_err(redirect)?;
    let entry = resources
        .resolve_entry(req.resource.as_deref())
        .map_err(redirect)?;
    let scopes =
        grant_scopes(req.scope.as_deref(), &client.scope, entry, resources).map_err(redirect)?;
    let resource = entry.url().clone();
    Ok(ValidatedAuthorization {
        client_id: client.client_id.clone(),
        redirect_uri: redirect_uri.to_string(),
        scopes,
        state: req.state.clone(),
        code_challenge: challenge.to_string(),
        resource,
    })
}

/// Build the success redirect `redirect_uri?code=..&state=..&iss=..`
/// (RFC 9207 `iss` included), preserving any registered query component.
/// Returns an empty string only if the stored redirect URI no longer parses
/// (it was validated at registration, so this is unreachable in practice;
/// callers must treat empty as `server_error`).
pub fn success_redirect(auth: &ValidatedAuthorization, code: &str, issuer: &str) -> String {
    let mut pairs = vec![("code", code.to_string())];
    if let Some(s) = &auth.state {
        pairs.push(("state", s.clone()));
    }
    pairs.push(("iss", issuer.to_string()));
    append_query(&auth.redirect_uri, &pairs).unwrap_or_default()
}
