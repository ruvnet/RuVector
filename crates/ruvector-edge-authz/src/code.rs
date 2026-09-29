//! Authorization codes: issuance and one-time redemption.

use crate::authorize::ValidatedAuthorization;
use crate::error::OAuthError;
use crate::federation::UpstreamIdentity;
use crate::ports::{Clock, CodeStore, Rng};
use serde::{Deserialize, Serialize};

/// Code lifetime (seconds).
pub const CODE_TTL_SECS: u64 = 60;
/// Entropy of an authorization code in bytes.
pub const CODE_BYTES: usize = 32;

/// Stored code record (keyed by `secret_hash(code)`; plaintext never stored).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizationCodeRecord {
    /// SHA-256 of the code.
    pub code_hash: [u8; 32],
    /// The authorization it completes.
    pub authorization: ValidatedAuthorization,
    /// User identity established upstream.
    pub identity: UpstreamIdentity,
    /// Absolute expiry (unix seconds).
    pub expires_at: u64,
}

/// Issue a code for a completed authorization.
///
/// Contract: [`CODE_BYTES`] of RNG entropy, base64url; store the hashed
/// record with `expires_at = now + CODE_TTL_SECS`; return the plaintext code.
pub fn issue_code<S: CodeStore, R: Rng, C: Clock>(
    store: &S,
    rng: &R,
    clock: &C,
    authorization: ValidatedAuthorization,
    identity: UpstreamIdentity,
) -> Result<String, OAuthError> {
    let _ = (store, rng, clock.now_unix(), authorization, identity);
    Err(OAuthError::not_implemented("code::issue_code"))
}

/// Redeem a code at the token endpoint.
///
/// Contract: `take_code` (atomic, one-time) by hash -> none => `invalid_grant`;
/// expired => `invalid_grant`; `client_id` and `redirect_uri` must equal the
/// stored values; `resource`, if sent, must equal the stored resource;
/// [`crate::pkce::verify_s256`] must hold. Any failure after the take still
/// consumes the code (no retry).
pub fn redeem_code<S: CodeStore, C: Clock>(
    store: &S,
    clock: &C,
    code: &str,
    client_id: &str,
    redirect_uri: &str,
    code_verifier: &str,
    resource: Option<&str>,
) -> Result<AuthorizationCodeRecord, OAuthError> {
    let _ = (
        store,
        clock.now_unix(),
        code,
        client_id,
        redirect_uri,
        code_verifier,
        resource,
    );
    Err(OAuthError::not_implemented("code::redeem_code"))
}
