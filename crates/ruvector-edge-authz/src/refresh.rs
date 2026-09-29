//! Refresh tokens: issuance, rotation, and family reuse detection
//! (OAuth 2.1 §4.3.1 / BCP 9700 §4.14).

use crate::error::OAuthError;
use crate::federation::UpstreamIdentity;
use crate::ports::{Clock, RefreshStore, Rng};
use ruvector_edge_auth::ResourceUrl;
use serde::{Deserialize, Serialize};

/// Refresh-token lifetime (seconds, absolute per token).
pub const REFRESH_TTL_SECS: u64 = 30 * 24 * 3600;
/// Entropy of a refresh token in bytes.
pub const REFRESH_BYTES: usize = 32;

/// Stored refresh token (keyed by `secret_hash(token)`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefreshTokenRecord {
    /// SHA-256 of the token.
    pub token_hash: [u8; 32],
    /// Family id shared by every rotation descendant.
    pub family_id: String,
    /// Client the family belongs to.
    pub client_id: String,
    /// User identity.
    pub identity: UpstreamIdentity,
    /// Bound resource (future `aud`).
    pub resource: ResourceUrl,
    /// Scope ceiling for the family.
    pub scopes: Vec<String>,
    /// Absolute expiry.
    pub expires_at: u64,
    /// Set once this token has been rotated.
    pub rotated: bool,
}

/// Start a new family after a code redemption.
///
/// Contract: new random `family_id` and token; store the hashed record;
/// return the plaintext token.
pub fn issue_refresh<S: RefreshStore, R: Rng, C: Clock>(
    store: &S,
    rng: &R,
    clock: &C,
    client_id: &str,
    identity: &UpstreamIdentity,
    resource: &ResourceUrl,
    scopes: &[String],
) -> Result<String, OAuthError> {
    let _ = (
        store,
        rng,
        clock.now_unix(),
        client_id,
        identity,
        resource,
        scopes,
    );
    Err(OAuthError::not_implemented("refresh::issue_refresh"))
}

/// Rotate a presented refresh token.
///
/// Contract: unknown hash => `invalid_grant`; family revoked or token
/// expired => `invalid_grant`; `client_id` mismatch => `invalid_grant`;
/// **already rotated** (or `mark_rotated` CAS returns false) => revoke the
/// whole family, then `invalid_grant` (reuse detection); `scope` must be a
/// subset of the family ceiling; `resource`, if sent, must equal the bound
/// one. On success returns `(new_plaintext_token, record_of_new_token)`.
pub fn rotate_refresh<S: RefreshStore, R: Rng, C: Clock>(
    store: &S,
    rng: &R,
    clock: &C,
    presented: &str,
    client_id: &str,
    scope: Option<&str>,
    resource: Option<&str>,
) -> Result<(String, RefreshTokenRecord), OAuthError> {
    let _ = (
        store,
        rng,
        clock.now_unix(),
        presented,
        client_id,
        scope,
        resource,
    );
    Err(OAuthError::not_implemented("refresh::rotate_refresh"))
}
