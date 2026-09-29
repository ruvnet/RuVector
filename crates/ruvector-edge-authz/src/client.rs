//! Dynamic Client Registration (RFC 7591) for public clients.

use crate::error::{OAuthError, OAuthErrorCode};
use serde::{Deserialize, Serialize};

/// Maximum redirect URIs per client.
pub const MAX_REDIRECT_URIS: usize = 8;
/// Maximum length of one redirect URI.
pub const MAX_REDIRECT_URI_LEN: usize = 512;
/// Maximum `client_name` length.
pub const MAX_CLIENT_NAME_LEN: usize = 128;

/// Registration request body. Unknown metadata members are ignored (RFC 7591
/// §2); only the members below influence behaviour.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct RegistrationRequest {
    /// Required, 1..=[`MAX_REDIRECT_URIS`].
    #[serde(default)]
    pub redirect_uris: Vec<String>,
    /// Must be `none` (or absent, meaning `none`).
    #[serde(default)]
    pub token_endpoint_auth_method: Option<String>,
    /// Subset of `["authorization_code","refresh_token"]`; default
    /// `["authorization_code"]`.
    #[serde(default)]
    pub grant_types: Option<Vec<String>>,
    /// Must be `["code"]` if present.
    #[serde(default)]
    pub response_types: Option<Vec<String>>,
    /// Display name (printable, <= [`MAX_CLIENT_NAME_LEN`]).
    #[serde(default)]
    pub client_name: Option<String>,
    /// Space-separated requested scope ceiling; must be a subset of the AS's
    /// `scopes_supported`. Absent -> the default public ceiling.
    #[serde(default)]
    pub scope: Option<String>,
}

/// A redirect URI that passed [`validate_redirect_uri`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RedirectUri(String);

impl RedirectUri {
    /// The exact registered string (authorization requests match byte-exact,
    /// except the loopback port rule in [`RedirectUri::matches`]).
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether `presented` matches this registration: byte-exact, or, for
    /// loopback `http` URIs, equal after ignoring the port (RFC 8252 §7.3).
    pub fn matches(&self, presented: &str) -> bool {
        let _ = presented;
        false
    }
}

/// Validate one redirect URI.
///
/// Contract: absolute URL <= [`MAX_REDIRECT_URI_LEN`], parsed with `url`;
/// no fragment, no userinfo; scheme `https` with a non-empty host, **or**
/// scheme `http` with host exactly `127.0.0.1` or `[::1]` (RFC 8252 §7.3;
/// `localhost` is refused). Everything else is `invalid_redirect_uri`.
pub fn validate_redirect_uri(uri: &str) -> Result<RedirectUri, OAuthError> {
    let _ = uri;
    Err(OAuthError::not_implemented("client::validate_redirect_uri"))
}

/// Server-side DCR policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DcrPolicy {
    /// Scopes a registration may request.
    pub scopes_supported: Vec<String>,
    /// Ceiling granted when `scope` is omitted.
    pub default_scope: Vec<String>,
}

/// Persisted client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientRecord {
    /// Server-generated opaque id (`edc-` + random).
    pub client_id: String,
    /// Validated redirect URIs.
    pub redirect_uris: Vec<RedirectUri>,
    /// Allowed grant types.
    pub grant_types: Vec<String>,
    /// Scope ceiling.
    pub scope: Vec<String>,
    /// Display name.
    pub client_name: Option<String>,
    /// Registration time (unix seconds).
    pub client_id_issued_at: u64,
}

/// RFC 7591 §3.2.1 response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RegistrationResponse {
    /// Issued id.
    pub client_id: String,
    /// Issue time.
    pub client_id_issued_at: u64,
    /// Echo of registered redirect URIs.
    pub redirect_uris: Vec<String>,
    /// Always `none`.
    pub token_endpoint_auth_method: &'static str,
    /// Registered grant types.
    pub grant_types: Vec<String>,
    /// Always `["code"]`.
    pub response_types: Vec<&'static str>,
    /// Granted scope ceiling (space-separated).
    pub scope: String,
    /// Display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
}

/// Validate a registration request into a record (id and time supplied by
/// the caller from the RNG/clock ports).
///
/// Contract: `redirect_uris` 1..=8 each via [`validate_redirect_uri`];
/// `token_endpoint_auth_method` absent or `none`; `response_types` absent or
/// `["code"]`; `grant_types` subset of `authorization_code`/`refresh_token`
/// and must include `authorization_code`; `scope` subset of
/// `policy.scopes_supported`; `client_name` printable and bounded.
pub fn validate_registration(
    req: &RegistrationRequest,
    policy: &DcrPolicy,
    client_id: String,
    now: u64,
) -> Result<ClientRecord, OAuthError> {
    let _ = (
        req,
        policy,
        client_id,
        now,
        OAuthErrorCode::InvalidClientMetadata,
    );
    Err(OAuthError::not_implemented("client::validate_registration"))
}

impl ClientRecord {
    /// Build the RFC 7591 response for this record.
    pub fn to_response(&self) -> RegistrationResponse {
        RegistrationResponse {
            client_id: self.client_id.clone(),
            client_id_issued_at: self.client_id_issued_at,
            redirect_uris: self
                .redirect_uris
                .iter()
                .map(|r| r.as_str().to_string())
                .collect(),
            token_endpoint_auth_method: "none",
            grant_types: self.grant_types.clone(),
            response_types: vec!["code"],
            scope: self.scope.join(" "),
            client_name: self.client_name.clone(),
        }
    }
}
