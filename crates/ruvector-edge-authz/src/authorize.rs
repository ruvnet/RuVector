//! Authorization endpoint request validation (RFC 6749 §4.1.1 + RFC 7636 +
//! RFC 8707).

use crate::client::ClientRecord;
use crate::error::OAuthError;
use crate::resource::ResourceAllowlist;
use ruvector_edge_auth::ResourceUrl;
use serde::{Deserialize, Serialize};

/// Raw query parameters of `GET /authorize`. The Worker must reject repeated
/// parameters before building this.
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

/// A request that passed every check; persisted inside the federation flow
/// state while the user logs in upstream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidatedAuthorization {
    /// Registered client.
    pub client_id: String,
    /// Exact redirect URI to return to.
    pub redirect_uri: String,
    /// Granted scopes (subset of the client's ceiling).
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

/// Validate an authorization request against the registered client.
///
/// Contract, and error delivery: `client_id` unknown or `redirect_uri` not
/// matching -> error shown to the user agent (never redirected);
/// otherwise errors are redirected: `response_type == "code"` else
/// `unsupported_response_type`; PKCE via [`crate::pkce::validate_challenge`];
/// `resource` via [`ResourceAllowlist::resolve`] (`invalid_target`); `scope`
/// subset of the client ceiling (`invalid_scope`); `state` <=
/// [`MAX_STATE_LEN`].
pub fn validate_authorization(
    req: &AuthorizationRequest,
    client: &ClientRecord,
    resources: &ResourceAllowlist,
) -> Result<ValidatedAuthorization, OAuthError> {
    let _ = (req, client, resources);
    Err(OAuthError::not_implemented(
        "authorize::validate_authorization",
    ))
}

/// Build the success redirect `redirect_uri?code=..&state=..&iss=..`
/// (RFC 9207 `iss` included), with proper query encoding.
pub fn success_redirect(auth: &ValidatedAuthorization, code: &str, issuer: &str) -> String {
    let _ = (auth, code, issuer);
    String::new()
}
