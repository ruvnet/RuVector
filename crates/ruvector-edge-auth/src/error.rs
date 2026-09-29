//! Error type for token verification. Every variant fails closed.

use thiserror::Error;

/// Why a bearer token was refused (or could not be checked).
///
/// Map to HTTP with [`AuthError::http_status`]: key-availability failures are
/// `503` (ADR §5.1.5 "no key ever fetched"), audience failures `403`
/// (`audience_not_allowed`), everything else `401 invalid_token`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AuthError {
    /// No `Authorization: Bearer` header (query/cookie tokens are never read).
    #[error("missing bearer token")]
    MissingToken,
    /// Structural problem: size, segment count, strict base64url, JSON.
    #[error("malformed token: {0}")]
    Malformed(&'static str),
    /// `alg` is not exactly `ES256` (rejected before any key lookup).
    #[error("unsupported alg")]
    UnsupportedAlg,
    /// Header carries `jku`, `x5u`, `x5c` or `jwk`.
    #[error("forbidden header parameter: {0}")]
    ForbiddenHeader(&'static str),
    /// `kid` missing from the header.
    #[error("missing kid")]
    MissingKid,
    /// Header `typ` not allowed for the token kind.
    #[error("bad typ")]
    BadTyp,
    /// `kid` not present in the (refreshed) key set.
    #[error("unknown kid")]
    UnknownKid,
    /// Keys could not be obtained and none are cached (maps to 503).
    #[error("signing keys unavailable")]
    KeysUnavailable,
    /// ES256 signature did not verify (also: DER / non-64-byte signatures).
    #[error("bad signature")]
    BadSignature,
    /// A required claim is missing or has the wrong type/format.
    #[error("invalid claim: {0}")]
    InvalidClaim(&'static str),
    /// `exp` is in the past beyond skew.
    #[error("token expired")]
    Expired,
    /// `nbf` is in the future beyond skew.
    #[error("token not yet valid")]
    NotYetValid,
    /// `iat` is in the future beyond skew.
    #[error("token issued in the future")]
    IssuedInFuture,
    /// `exp - iat` exceeds the policy maximum.
    #[error("token lifetime too long")]
    LifetimeTooLong,
    /// `iss` is not an accepted issuer.
    #[error("wrong issuer")]
    WrongIssuer,
    /// `aud` does not exactly match the resource / allowlist.
    #[error("audience not allowed")]
    AudienceNotAllowed,
    /// Token (jti/sub/org/client) is on the deny-list.
    #[error("token denied")]
    Denied,
    /// A configuration value (issuer, resource URL, ...) failed validation.
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
    /// Scaffold placeholder: the code path is not implemented yet. Fails closed.
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
}

impl AuthError {
    /// HTTP status a resource server should answer with.
    pub fn http_status(&self) -> u16 {
        match self {
            AuthError::KeysUnavailable | AuthError::NotImplemented(_) => 503,
            AuthError::AudienceNotAllowed => 403,
            AuthError::InvalidConfig(_) => 500,
            _ => 401,
        }
    }

    /// RFC 6750 / ADR §7 stable error code for problem+json and
    /// `WWW-Authenticate`.
    pub fn code(&self) -> &'static str {
        match self {
            AuthError::KeysUnavailable | AuthError::NotImplemented(_) => "jwks_unavailable",
            AuthError::AudienceNotAllowed => "audience_not_allowed",
            AuthError::InvalidConfig(_) => "server_error",
            _ => "invalid_token",
        }
    }
}
