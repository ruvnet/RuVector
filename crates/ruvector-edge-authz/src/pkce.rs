//! PKCE (RFC 7636), `S256` only. `plain` is never accepted.

use crate::error::{OAuthError, OAuthErrorCode};

/// Minimum verifier / challenge length (RFC 7636 §4.1).
pub const MIN_LEN: usize = 43;
/// Maximum verifier length (RFC 7636 §4.1).
pub const MAX_LEN: usize = 128;

fn unreserved(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~')
}

/// Validate a `code_verifier`: 43-128 chars of `[A-Za-z0-9-._~]`.
pub fn validate_verifier(verifier: &str) -> Result<(), OAuthError> {
    let ok = (MIN_LEN..=MAX_LEN).contains(&verifier.len()) && verifier.bytes().all(unreserved);
    ok.then_some(()).ok_or(OAuthError::new(
        OAuthErrorCode::InvalidGrant,
        "invalid code_verifier",
    ))
}

/// Validate a `code_challenge` for `S256`: exactly 43 base64url chars
/// (32-byte digest), and `method == "S256"`.
pub fn validate_challenge(challenge: &str, method: Option<&str>) -> Result<(), OAuthError> {
    if method != Some("S256") {
        return Err(OAuthError::new(
            OAuthErrorCode::InvalidRequest,
            "code_challenge_method must be S256",
        ));
    }
    let ok = challenge.len() == MIN_LEN
        && challenge
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    ok.then_some(()).ok_or(OAuthError::new(
        OAuthErrorCode::InvalidRequest,
        "invalid code_challenge",
    ))
}

/// `BASE64URL(SHA256(verifier)) == challenge`, compared in constant time.
/// Returns `false` for a malformed verifier.
pub fn verify_s256(verifier: &str, challenge: &str) -> bool {
    use sha2::{Digest, Sha256};
    use subtle::ConstantTimeEq;
    if validate_verifier(verifier).is_err() {
        return false;
    }
    let computed = ruvector_edge_auth::jws::b64url_encode(&Sha256::digest(verifier.as_bytes()));
    computed.as_bytes().ct_eq(challenge.as_bytes()).into()
}
