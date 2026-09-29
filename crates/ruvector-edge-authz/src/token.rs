//! Token endpoint: request parsing, access-token minting (RFC 9068 JWT,
//! ES256), and the response body.

use crate::error::OAuthError;
use crate::federation::UpstreamIdentity;
use crate::ports::{Clock, Rng, Signer};
use ruvector_edge_auth::ResourceUrl;
use serde::Serialize;

/// Access-token lifetime (seconds). Kept <= the RS `max_lifetime_secs`.
pub const ACCESS_TOKEN_TTL_SECS: u64 = 900;

/// Parsed `application/x-www-form-urlencoded` token request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenRequest {
    /// `grant_type=authorization_code`.
    AuthorizationCode {
        /// The code.
        code: String,
        /// Must equal the authorization request's.
        redirect_uri: String,
        /// Public client id.
        client_id: String,
        /// PKCE verifier.
        code_verifier: String,
        /// Optional RFC 8707 resource (must equal the authorized one).
        resource: Option<String>,
    },
    /// `grant_type=refresh_token`.
    RefreshToken {
        /// Presented refresh token.
        refresh_token: String,
        /// Public client id.
        client_id: String,
        /// Optional down-scoping.
        scope: Option<String>,
        /// Optional RFC 8707 resource (must equal the family's).
        resource: Option<String>,
    },
}

impl TokenRequest {
    /// Parse form pairs.
    ///
    /// Contract: repeated parameters => `invalid_request`; unknown
    /// `grant_type` => `unsupported_grant_type`; required members present and
    /// bounded (each value <= 2 KiB); `client_secret` present => `invalid_client`
    /// (public clients only).
    pub fn from_form(pairs: &[(String, String)]) -> Result<Self, OAuthError> {
        let _ = pairs;
        Err(OAuthError::not_implemented(
            "token::TokenRequest::from_form",
        ))
    }
}

/// Claims of an edge-minted access token (RFC 9068).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AccessTokenClaims {
    /// Edge AS issuer.
    pub iss: String,
    /// Exact resource URL.
    pub aud: String,
    /// Upstream user `sub`.
    pub sub: String,
    /// Downstream client.
    pub client_id: String,
    /// Tenant input, copied from the upstream identity.
    pub org_id: String,
    /// Tenant input, copied from the upstream identity.
    pub workspace_id: String,
    /// Space-separated granted scopes.
    pub scope: String,
    /// Unique token id (random).
    pub jti: String,
    /// Issued-at.
    pub iat: u64,
    /// Expiry (`iat + ACCESS_TOKEN_TTL_SECS`).
    pub exp: u64,
}

/// Inputs for [`mint_access_token`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MintRequest<'a> {
    /// Edge AS issuer URL.
    pub issuer: &'a str,
    /// Target resource (already resolved against the allowlist).
    pub resource: &'a ResourceUrl,
    /// Downstream client.
    pub client_id: &'a str,
    /// User identity.
    pub identity: &'a UpstreamIdentity,
    /// Granted scopes.
    pub scopes: &'a [String],
}

/// Mint a compact JWS access token.
///
/// Contract: header `{"alg":"ES256","typ":"at+jwt","kid":signer.kid()}`;
/// claims per [`AccessTokenClaims`] with `aud = resource.as_str()`, `iat =
/// now`, `exp = now + ACCESS_TOKEN_TTL_SECS`, `jti` from 16 RNG bytes;
/// signature via [`Signer::sign_es256`] over `b64(header).b64(claims)`.
pub fn mint_access_token<S: Signer, R: Rng, C: Clock>(
    signer: &S,
    rng: &R,
    clock: &C,
    req: &MintRequest<'_>,
) -> Result<String, OAuthError> {
    let _ = (signer.kid(), rng, clock.now_unix(), req);
    Err(OAuthError::not_implemented("token::mint_access_token"))
}

/// RFC 6749 §5.1 success body (`Cache-Control: no-store` is the Worker's job).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TokenResponse {
    /// The JWT.
    pub access_token: String,
    /// Always `Bearer`.
    pub token_type: &'static str,
    /// Seconds until expiry.
    pub expires_in: u64,
    /// Rotated refresh token, if the client has the grant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Granted scope.
    pub scope: String,
}
