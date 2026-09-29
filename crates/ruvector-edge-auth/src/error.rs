//! Error type for token verification. Every variant fails closed.

use thiserror::Error;

/// Why a bearer token was refused (or could not be checked).
///
/// Map to HTTP with [`AuthError::http_status`]: key-availability failures are
/// `503` (ADR §5.4.6 "never fetched"), an edge token for a sibling resource
/// `403 audience_not_allowed` (§5.4.7), a malformed `Authorization` header
/// `400 invalid_request` (RFC 6750 §3.1), everything else `401
/// invalid_token`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AuthError {
    /// No `Authorization: Bearer` header (query/cookie tokens are never read).
    #[error("missing bearer token")]
    MissingToken,
    /// `Authorization: Bearer` present but not `Bearer <token>` (empty token,
    /// extra whitespace). RFC 6750 §3.1 `invalid_request` -> 400.
    #[error("malformed authorization header")]
    MalformedAuthorization,
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
    /// `aud` names a sibling edge resource of this gateway (403). Any other
    /// audience mismatch is [`AuthError::InvalidClaim`]`("aud")` (401).
    #[error("audience not allowed")]
    AudienceNotAllowed,
    /// Token (jti/family_id/sub/client_id/kid) is on the deny-list.
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
            AuthError::MalformedAuthorization => 400,
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
            AuthError::MalformedAuthorization => "invalid_request",
            AuthError::InvalidConfig(_) => "server_error",
            _ => "invalid_token",
        }
    }

    /// RFC 6750 §3.1 `error` attribute for the `WWW-Authenticate` challenge.
    ///
    /// `None` when the request carried no credentials ([`AuthError::MissingToken`],
    /// §3.1: "SHOULD NOT include an error code"), for server-side failures
    /// (5xx), which are not bearer-token errors, and for the 403
    /// [`AuthError::AudienceNotAllowed`] (`invalid_token` pairs only with 401,
    /// RFC 6750 §3.1). A malformed `Authorization` header is
    /// `invalid_request` (400); every other token failure is `invalid_token`
    /// (401). The status from [`AuthError::http_status`] always agrees.
    pub fn rfc6750_error(&self) -> Option<&'static str> {
        match self {
            AuthError::MissingToken
            | AuthError::KeysUnavailable
            | AuthError::NotImplemented(_)
            | AuthError::InvalidConfig(_)
            | AuthError::AudienceNotAllowed => None,
            AuthError::MalformedAuthorization => Some("invalid_request"),
            _ => Some("invalid_token"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AuthError;

    #[test]
    fn status_code_and_rfc6750_mapping() {
        let cases: &[(AuthError, u16, &str, Option<&str>)] = &[
            (AuthError::MissingToken, 401, "invalid_token", None),
            (
                AuthError::MalformedAuthorization,
                400,
                "invalid_request",
                Some("invalid_request"),
            ),
            (
                AuthError::Malformed("base64url"),
                401,
                "invalid_token",
                Some("invalid_token"),
            ),
            (
                AuthError::UnsupportedAlg,
                401,
                "invalid_token",
                Some("invalid_token"),
            ),
            (
                AuthError::Expired,
                401,
                "invalid_token",
                Some("invalid_token"),
            ),
            (
                AuthError::InvalidClaim("aud"),
                401,
                "invalid_token",
                Some("invalid_token"),
            ),
            (
                AuthError::AudienceNotAllowed,
                403,
                "audience_not_allowed",
                None,
            ),
            (AuthError::KeysUnavailable, 503, "jwks_unavailable", None),
            (AuthError::InvalidConfig("x"), 500, "server_error", None),
        ];
        for (err, status, code, rfc) in cases {
            assert_eq!(err.http_status(), *status, "{err:?}");
            assert_eq!(err.code(), *code, "{err:?}");
            assert_eq!(err.rfc6750_error(), *rfc, "{err:?}");
        }
    }

    /// Regression: `invalid_token` is never paired with a non-401 status and
    /// `invalid_request` never with a non-400 one (RFC 6750 §3.1).
    #[test]
    fn rfc6750_error_agrees_with_status() {
        let all = [
            AuthError::MissingToken,
            AuthError::MalformedAuthorization,
            AuthError::Malformed("x"),
            AuthError::UnsupportedAlg,
            AuthError::ForbiddenHeader("jku"),
            AuthError::MissingKid,
            AuthError::BadTyp,
            AuthError::UnknownKid,
            AuthError::KeysUnavailable,
            AuthError::BadSignature,
            AuthError::InvalidClaim("aud"),
            AuthError::Expired,
            AuthError::NotYetValid,
            AuthError::IssuedInFuture,
            AuthError::LifetimeTooLong,
            AuthError::WrongIssuer,
            AuthError::AudienceNotAllowed,
            AuthError::Denied,
            AuthError::InvalidConfig("x"),
            AuthError::NotImplemented("x"),
        ];
        for err in all {
            match err.rfc6750_error() {
                Some("invalid_token") => assert_eq!(err.http_status(), 401, "{err:?}"),
                Some("invalid_request") => assert_eq!(err.http_status(), 400, "{err:?}"),
                Some(other) => panic!("unexpected {other} for {err:?}"),
                None => {}
            }
        }
    }
}
