//! Compact JWS parsing and ES256 verification (ADR-351 §5.1 steps 1-4).

use crate::error::AuthError;
use p256::ecdsa::VerifyingKey;
use serde::Deserialize;

/// Maximum accepted compact token size in bytes (ADR §5.1.2).
pub const MAX_TOKEN_BYTES: usize = 8 * 1024;

/// Header parameters that are always rejected (ADR §5.1.3). `crit` is also
/// rejected ([`AuthError::ForbiddenHeader`]`("crit")`): no extension is
/// understood (RFC 7515 §4.1.11).
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
/// empty token yield [`AuthError::MissingToken`] /
/// [`AuthError::MalformedAuthorization`] (400 `invalid_request`).
/// Tokens from query strings or cookies are never consulted by callers.
pub fn bearer_token(authorization: Option<&str>) -> Result<&str, AuthError> {
    let value = authorization.ok_or(AuthError::MissingToken)?;
    let (scheme, token) = value.split_once(' ').ok_or(AuthError::MissingToken)?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return Err(AuthError::MissingToken);
    }
    if token.is_empty() || token.bytes().any(|b| b.is_ascii_whitespace()) {
        return Err(AuthError::MalformedAuthorization);
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
    if token.len() > MAX_TOKEN_BYTES {
        return Err(AuthError::Malformed("token too large"));
    }
    let mut parts = token.split('.');
    let (h, p, s) = match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(h), Some(p), Some(s), None) => (h, p, s),
        _ => return Err(AuthError::Malformed("segment count")),
    };
    if h.is_empty() || p.is_empty() || s.is_empty() {
        return Err(AuthError::Malformed("empty segment"));
    }
    // Strict decode of every segment up front, so a malformed payload or
    // signature never reaches key lookup.
    let header_bytes = b64url_decode(h)?;
    b64url_decode(p)?;
    let sig_bytes = b64url_decode(s)?;

    let header = parse_header(&header_bytes)?;
    let signature: [u8; 64] = sig_bytes
        .as_slice()
        .try_into()
        .map_err(|_| AuthError::BadSignature)?;
    Ok(CompactJws {
        header,
        signing_input: &token[..h.len() + 1 + p.len()],
        payload_b64: p,
        signature,
    })
}

/// Wire form of the JOSE header. Known fields only; serde's derive rejects
/// duplicates of these names. Unknown members are ignored.
#[derive(Deserialize)]
struct RawHeader {
    alg: Option<serde_json::Value>,
    kid: Option<serde_json::Value>,
    typ: Option<serde_json::Value>,
    jku: Option<serde_json::Value>,
    x5u: Option<serde_json::Value>,
    x5c: Option<serde_json::Value>,
    jwk: Option<serde_json::Value>,
    crit: Option<serde_json::Value>,
}

/// Maximum accepted `kid` length (RFC 7638 SHA-256 thumbprints are 43).
pub const MAX_KID_LEN: usize = 128;

fn parse_header(bytes: &[u8]) -> Result<JwsHeader, AuthError> {
    if !starts_with_object(bytes) {
        return Err(AuthError::Malformed("header not an object"));
    }
    let raw: RawHeader =
        serde_json::from_slice(bytes).map_err(|_| AuthError::Malformed("header json"))?;
    // alg first: none / HS* / RS* / anything else is refused before any other
    // header rule and before any key lookup.
    match raw.alg {
        Some(serde_json::Value::String(ref a)) if a == "ES256" => {}
        _ => return Err(AuthError::UnsupportedAlg),
    }
    let forbidden = [
        ("jku", raw.jku.is_some()),
        ("x5u", raw.x5u.is_some()),
        ("x5c", raw.x5c.is_some()),
        ("jwk", raw.jwk.is_some()),
        // RFC 7515 §4.1.11: we understand no extensions, so any `crit` fails.
        ("crit", raw.crit.is_some()),
    ];
    if let Some((name, _)) = forbidden.iter().find(|(_, present)| *present) {
        return Err(AuthError::ForbiddenHeader(name));
    }
    let kid = match raw.kid {
        None => return Err(AuthError::MissingKid),
        Some(serde_json::Value::String(k)) => k,
        Some(_) => return Err(AuthError::Malformed("kid type")),
    };
    if kid.is_empty() || kid.len() > MAX_KID_LEN || !kid.bytes().all(is_b64url_byte) {
        return Err(AuthError::Malformed("kid"));
    }
    let typ = match raw.typ {
        None => None,
        Some(serde_json::Value::String(t)) => Some(t),
        Some(_) => return Err(AuthError::BadTyp),
    };
    Ok(JwsHeader {
        alg: "ES256".to_string(),
        kid,
        typ,
    })
}

/// True if the first non-whitespace byte is `{` (serde derives would
/// otherwise accept a JSON array positionally).
pub(crate) fn starts_with_object(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .find(|b| !b.is_ascii_whitespace())
        .is_some_and(|b| *b == b'{')
}

fn is_b64url_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-' || b == b'_'
}

/// Verify the ES256 signature of `jws` with `key`.
///
/// Contract: SHA-256 over `jws.signing_input`, ECDSA P-256 verify of the
/// fixed `r || s` form via `p256::ecdsa`. High-S signatures follow the p256
/// crate's verify semantics. Returns [`AuthError::BadSignature`] on failure.
pub fn verify_es256(jws: &CompactJws<'_>, key: &VerifyingKey) -> Result<(), AuthError> {
    use p256::ecdsa::signature::Verifier as _;
    use p256::ecdsa::Signature;
    let sig = Signature::from_slice(&jws.signature).map_err(|_| AuthError::BadSignature)?;
    key.verify(jws.signing_input.as_bytes(), &sig)
        .map_err(|_| AuthError::BadSignature)
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

#[cfg(test)]
mod tests;
