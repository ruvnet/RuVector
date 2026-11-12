//! Refresh tokens: issuance, rotation, and family reuse detection
//! (OAuth 2.1 §4.3.1 / BCP 9700 §4.14).

use crate::error::{OAuthError, OAuthErrorCode};
use crate::federation::UpstreamIdentity;
use crate::params::{ensure_subset, split_scope};
use crate::ports::{Clock, RefreshStore, Rng};
use ruvector_edge_auth::ResourceUrl;
use serde::{Deserialize, Serialize};

/// Refresh-token lifetime (seconds). Sliding: each rotation gets a fresh
/// lifetime, capped by the family's absolute lifetime.
pub const REFRESH_TTL_SECS: u64 = 30 * 24 * 3600;
/// Absolute lifetime of a refresh family from its first issuance (seconds);
/// after this the user must log in again.
pub const FAMILY_MAX_LIFETIME_SECS: u64 = 90 * 24 * 3600;
/// Entropy of a refresh token in bytes.
pub const REFRESH_BYTES: usize = 32;
/// Entropy of a family / grant id in bytes.
pub const FAMILY_ID_BYTES: usize = 16;

/// Stored refresh token (keyed by `secret_hash(token)`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefreshTokenRecord {
    /// SHA-256 of the token.
    pub token_hash: [u8; 32],
    /// Family id shared by every rotation descendant (also the access
    /// token's `family_id` claim).
    pub family_id: String,
    /// Client the family belongs to.
    pub client_id: String,
    /// User identity.
    pub identity: UpstreamIdentity,
    /// Bound resource (future `aud`).
    pub resource: ResourceUrl,
    /// Scope ceiling for the family.
    pub scopes: Vec<String>,
    /// Absolute expiry of this token.
    pub expires_at: u64,
    /// Absolute expiry of the whole family.
    pub family_expires_at: u64,
    /// Set once this token has been rotated.
    pub rotated: bool,
}

/// Mint a new random family / grant id.
pub fn new_family_id<R: Rng + ?Sized>(rng: &R) -> Result<String, OAuthError> {
    Ok(crate::random_secret(rng, FAMILY_ID_BYTES)?)
}

/// Start a new family after a code redemption.
///
/// Contract: new random token; the record carries `family_id` (from
/// [`new_family_id`]), `expires_at = now + REFRESH_TTL_SECS` and
/// `family_expires_at = now + FAMILY_MAX_LIFETIME_SECS`; store the hashed
/// record; return the plaintext token and the record.
#[allow(clippy::too_many_arguments)]
pub fn issue_refresh<S, R, C>(
    store: &S,
    rng: &R,
    clock: &C,
    family_id: &str,
    client_id: &str,
    identity: &UpstreamIdentity,
    resource: &ResourceUrl,
    scopes: &[String],
) -> Result<(String, RefreshTokenRecord), OAuthError>
where
    S: RefreshStore + ?Sized,
    R: Rng + ?Sized,
    C: Clock + ?Sized,
{
    let now = clock.now_unix();
    let token = crate::random_secret(rng, REFRESH_BYTES)?;
    let record = RefreshTokenRecord {
        token_hash: crate::secret_hash(&token),
        family_id: family_id.to_string(),
        client_id: client_id.to_string(),
        identity: identity.clone(),
        resource: resource.clone(),
        scopes: scopes.to_vec(),
        expires_at: now.saturating_add(REFRESH_TTL_SECS),
        family_expires_at: now.saturating_add(FAMILY_MAX_LIFETIME_SECS),
        rotated: false,
    };
    store.insert_refresh(&record)?;
    Ok((token, record))
}

/// `offline_access` (ADR-351 §5.3): accepted and echoed in the granted
/// scope, but it does **not** gate refresh tokens, which follow the
/// client's registered `refresh_token` grant.
pub const OFFLINE_ACCESS: &str = "offline_access";

/// Outcome of a successful rotation. `Debug` redacts the token.
#[derive(Clone, PartialEq, Eq)]
pub struct RotatedRefresh {
    /// New plaintext refresh token.
    pub token: String,
    /// Record of the new token (family ceiling unchanged).
    pub record: RefreshTokenRecord,
    /// Scopes for the access token minted now (requested subset, or the
    /// family ceiling when `scope` was omitted).
    pub granted_scopes: Vec<String>,
}

impl std::fmt::Debug for RotatedRefresh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RotatedRefresh")
            .field("token", &crate::REDACTED)
            .field("record", &self.record)
            .field("granted_scopes", &self.granted_scopes)
            .finish()
    }
}

/// A validated rotation that has **not** changed any state yet
/// ([`prepare_rotation`]); [`PendingRotation::commit`] consumes the old token.
/// Lets the token endpoint mint the access token first, so a signing/RNG
/// failure or config change never burns the old token (which would turn a
/// retry into reuse detection and revoke the family).
#[derive(Clone, PartialEq, Eq)]
pub struct PendingRotation {
    old_hash: [u8; 32],
    token: String,
    /// The new token's record (same family, sliding expiry).
    pub record: RefreshTokenRecord,
    /// Scopes for the access token minted now.
    pub granted_scopes: Vec<String>,
}

impl std::fmt::Debug for PendingRotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingRotation")
            .field("token", &crate::REDACTED)
            .field("record", &self.record)
            .field("granted_scopes", &self.granted_scopes)
            .finish()
    }
}

impl PendingRotation {
    /// Consume the old token: `mark_rotated` CAS (`false` = concurrent use
    /// => revoke the family, `invalid_grant`), then store the new token.
    pub fn commit<S: RefreshStore + ?Sized>(self, store: &S) -> Result<RotatedRefresh, OAuthError> {
        if !store.mark_rotated(&self.old_hash)? {
            store.revoke_family(&self.record.family_id)?;
            return Err(grant_err("refresh token reuse detected"));
        }
        store.insert_refresh(&self.record)?;
        Ok(RotatedRefresh {
            token: self.token,
            record: self.record,
            granted_scopes: self.granted_scopes,
        })
    }
}

fn grant_err(desc: &'static str) -> OAuthError {
    OAuthError::new(OAuthErrorCode::InvalidGrant, desc)
}

/// Rotate a presented refresh token: [`prepare_rotation`] then
/// [`PendingRotation::commit`].
pub fn rotate_refresh<S, R, C>(
    store: &S,
    rng: &R,
    clock: &C,
    presented: &str,
    client_id: &str,
    scope: Option<&str>,
    resource: Option<&str>,
) -> Result<RotatedRefresh, OAuthError>
where
    S: RefreshStore + ?Sized,
    R: Rng + ?Sized,
    C: Clock + ?Sized,
{
    prepare_rotation(store, rng, clock, presented, client_id, scope, resource)?.commit(store)
}

/// Validate a presented refresh token and prepare its successor.
///
/// Contract, in order: unknown hash => `invalid_grant`; `client_id` mismatch
/// => `invalid_grant`; family revoked => `invalid_grant`; **already
/// rotated** => revoke the whole family, then `invalid_grant` (reuse
/// detection); token or family expired => `invalid_grant`; `resource`, if
/// sent, must equal the bound one (`invalid_target`); `scope` must be a
/// subset of the family ceiling (`invalid_scope`). The family ceiling itself
/// never changes, so a narrowing `scope` narrows only this access token and
/// the family keeps refreshing (whether `offline_access` was granted or
/// not; the token endpoint checks the client's `refresh_token` grant).
/// Nothing but a reuse-detection revocation is written; the successor is
/// random, stored only on commit, with a sliding expiry capped at
/// `family_expires_at`.
pub fn prepare_rotation<S, R, C>(
    store: &S,
    rng: &R,
    clock: &C,
    presented: &str,
    client_id: &str,
    scope: Option<&str>,
    resource: Option<&str>,
) -> Result<PendingRotation, OAuthError>
where
    S: RefreshStore + ?Sized,
    R: Rng + ?Sized,
    C: Clock + ?Sized,
{
    let now = clock.now_unix();
    let hash = crate::secret_hash(presented);
    let old = store
        .get_refresh(&hash)?
        .ok_or(grant_err("invalid refresh token"))?;
    if old.client_id != client_id {
        return Err(grant_err("refresh token was issued to another client"));
    }
    if store.is_family_revoked(&old.family_id)? {
        return Err(grant_err("refresh token revoked"));
    }
    if old.rotated {
        store.revoke_family(&old.family_id)?;
        return Err(grant_err("refresh token reuse detected"));
    }
    if now >= old.expires_at || now >= old.family_expires_at {
        return Err(grant_err("refresh token expired"));
    }
    if let Some(r) = resource {
        if !ResourceUrl::parse(r).is_ok_and(|r| r == old.resource) {
            return Err(OAuthError::new(
                OAuthErrorCode::InvalidTarget,
                "resource differs from the bound one",
            ));
        }
    }
    let granted_scopes = match scope {
        None => old.scopes.clone(),
        Some(s) => {
            let s = split_scope(s)?;
            ensure_subset(&s, &old.scopes)?;
            s
        }
    };
    let token = crate::random_secret(rng, REFRESH_BYTES)?;
    let record = RefreshTokenRecord {
        token_hash: crate::secret_hash(&token),
        expires_at: now
            .saturating_add(REFRESH_TTL_SECS)
            .min(old.family_expires_at),
        rotated: false,
        ..old
    };
    Ok(PendingRotation {
        old_hash: hash,
        token,
        record,
        granted_scopes,
    })
}
