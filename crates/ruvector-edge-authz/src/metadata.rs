//! RFC 8414 Authorization Server Metadata and the AS JWKS document.

use ruvector_edge_auth::{Jwk, JwkSet, VerifyingKey};
use serde::Serialize;

/// Endpoint paths on the edge AS. `JWKS_PATH` is also what resource servers
/// configure as their edge JWKS URL (`<issuer>` + `JWKS_PATH`).
pub mod paths {
    /// RFC 8414 well-known.
    pub const METADATA: &str = "/.well-known/oauth-authorization-server";
    /// JWKS.
    pub const JWKS: &str = "/.well-known/jwks.json";
    /// DCR.
    pub const REGISTER: &str = "/register";
    /// Authorization endpoint.
    pub const AUTHORIZE: &str = "/authorize";
    /// Upstream callback.
    pub const CALLBACK: &str = "/callback";
    /// Token endpoint.
    pub const TOKEN: &str = "/token";
    /// RFC 7009 revocation.
    pub const REVOKE: &str = "/revoke";
}

/// RFC 8414 §2 metadata document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[allow(missing_docs)]
pub struct AuthorizationServerMetadata {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub registration_endpoint: String,
    pub revocation_endpoint: String,
    pub jwks_uri: String,
    pub scopes_supported: Vec<String>,
    pub response_types_supported: Vec<&'static str>,
    pub grant_types_supported: Vec<&'static str>,
    pub token_endpoint_auth_methods_supported: Vec<&'static str>,
    /// RFC 8414: algorithms for `private_key_jwt` client assertions.
    pub token_endpoint_auth_signing_alg_values_supported: Vec<&'static str>,
    pub revocation_endpoint_auth_methods_supported: Vec<&'static str>,
    pub code_challenge_methods_supported: Vec<&'static str>,
    /// RFC 9207.
    pub authorization_response_iss_parameter_supported: bool,
}

impl AuthorizationServerMetadata {
    /// Build the document for `issuer` (no trailing slash) and `scopes`.
    /// Advertises PKCE `S256` only; `none` client auth for public (DCR)
    /// clients and `private_key_jwt` (ES256) for operator-registered
    /// confidential clients; code + refresh_token grants, plus the RFC 8693
    /// exchange grant (confidential clients only).
    pub fn build(issuer: &str, scopes: &[String]) -> Self {
        let u = |p: &str| format!("{issuer}{p}");
        AuthorizationServerMetadata {
            issuer: issuer.to_string(),
            authorization_endpoint: u(paths::AUTHORIZE),
            token_endpoint: u(paths::TOKEN),
            registration_endpoint: u(paths::REGISTER),
            revocation_endpoint: u(paths::REVOKE),
            jwks_uri: u(paths::JWKS),
            scopes_supported: scopes.to_vec(),
            response_types_supported: vec!["code"],
            grant_types_supported: vec![
                "authorization_code",
                "refresh_token",
                crate::exchange::TOKEN_EXCHANGE_GRANT,
            ],
            token_endpoint_auth_methods_supported: vec!["none", "private_key_jwt"],
            token_endpoint_auth_signing_alg_values_supported: vec!["ES256"],
            revocation_endpoint_auth_methods_supported: vec!["none"],
            code_challenge_methods_supported: vec!["S256"],
            authorization_response_iss_parameter_supported: true,
        }
    }
}

/// JWKS document for the AS's public keys (current + previous during
/// rotation), each with `kid` = RFC 7638 thumbprint.
pub fn jwks_document(keys: &[VerifyingKey]) -> JwkSet {
    JwkSet {
        keys: keys.iter().map(Jwk::from_verifying_key).collect(),
    }
}
