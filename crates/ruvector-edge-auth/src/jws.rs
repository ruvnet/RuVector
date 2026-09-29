//! Compact JWS parsing and ES256 verification (ADR-351 §5.1 steps 1-4).

use crate::error::AuthError;
use p256::ecdsa::VerifyingKey;

/// Maximum accepted compact token size in bytes (ADR §5.1.2).
pub const MAX_TOKEN_BYTES: usize = 8 * 1024;

/// Header parameters that are always rejected (ADR §5.1.3).
pub const FORBIDDEN_HEADER_PARAMS: [&str; 4] = ["jku", "x5u", "x5c", "jwk"];

/// Decoded, policy-checked JOSE header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JwsHeader {
    /// Always `"ES256"` once parsed.
    pub alg: String,
    /// Key id; required.
    pub kid: String,
    /// Media type, if present. Edge-issued tokens use `at+jwt` (RFC 9068);
    /// upstream tokens may carry `JWT` or omit it. Kind-specific checks live
    /// in [`crate::audience`].
    pub typ: Option<String>,
}

/// A parsed compact JWS whose header passed the static checks. The signature
/// has **not** been verified; call [`verify_es256`].
#[derive(Debug, Clone)]
pub struct CompactJws<'a> {
    /// Checked header.
    pub header: JwsHeader,
    /// `base64url(header) "." base64url(payload)`: the ES256 signing input.
    pub signing_input: &'a str,
    /// Raw payload segment (still base64url).
    pub payload_b64: &'a str,
    /// Fixed-width `r || s` signature (64 bytes). DER is rejected at parse.
    pub signature: [u8; 64],
}

impl<'a> CompactJws<'a> {
    /// Strictly base64url-decode the payload segment into JSON bytes.
    ///
    /// Contract: rejects padding, non-alphabet bytes and non-canonical
    /// trailing bits.
    pub fn payload_bytes(&self) -> Result<Vec<u8>, AuthError> {
        b64url_decode(self.payload_b64)
    }
}

/// Extract the token from an `Authorization` header value.
///
/// Contract: accepts exactly `Bearer <token>` (scheme case-insensitive, one
/// space, non-empty token with no whitespace). `None`, other schemes or an
/// empty token yield [`AuthError::MissingToken`] / [`AuthError::Malformed`].
/// Tokens from query strings or cookies are never consulted by callers.
pub fn bearer_token(authorization: Option<&str>) -> Result<&str, AuthError> {
    let value = authorization.ok_or(AuthError::MissingToken)?;
    let (scheme, token) = value.split_once(' ').ok_or(AuthError::MissingToken)?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return Err(AuthError::MissingToken);
    }
    if token.is_empty() || token.bytes().any(|b| b.is_ascii_whitespace()) {
        return Err(AuthError::Malformed("bearer token"));
    }
    Ok(token)
}

/// Parse a compact JWS and apply the header rules.
///
/// Contract (ADR §5.1.2-3), in order: size <= [`MAX_TOKEN_BYTES`]; exactly
/// three non-empty segments; strict base64url; header is a JSON object;
/// `alg == "ES256"` (else [`AuthError::UnsupportedAlg`], before any key
/// lookup); none of [`FORBIDDEN_HEADER_PARAMS`]; `kid` present and a string;
/// `typ`, if present, a string; signature decodes to exactly 64 bytes.
pub fn parse_compact(token: &str) -> Result<CompactJws<'_>, AuthError> {
    let _ = token;
    Err(AuthError::NotImplemented("jws::parse_compact"))
}

/// Verify the ES256 signature of `jws` with `key`.
///
/// Contract: SHA-256 over `jws.signing_input`, ECDSA P-256 verify of the
/// fixed `r || s` form via `p256::ecdsa`. High-S signatures follow the p256
/// crate's verify semantics. Returns [`AuthError::BadSignature`] on failure.
pub fn verify_es256(jws: &CompactJws<'_>, key: &VerifyingKey) -> Result<(), AuthError> {
    let _ = (jws, key);
    Err(AuthError::NotImplemented("jws::verify_es256"))
}

/// Strict unpadded base64url decode used for every JOSE segment.
pub fn b64url_decode(input: &str) -> Result<Vec<u8>, AuthError> {
    use base64ct::{Base64UrlUnpadded, Encoding};
    Base64UrlUnpadded::decode_vec(input).map_err(|_| AuthError::Malformed("base64url"))
}

/// Unpadded base64url encode (JOSE form).
pub fn b64url_encode(input: &[u8]) -> String {
    use base64ct::{Base64UrlUnpadded, Encoding};
    Base64UrlUnpadded::encode_string(input)
}
