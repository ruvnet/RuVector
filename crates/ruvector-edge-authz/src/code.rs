//! Authorization codes: issuance and one-time redemption.

use crate::authorize::ValidatedAuthorization;
use crate::error::{OAuthError, OAuthErrorCode};
use crate::federation::UpstreamIdentity;
use crate::ports::{Clock, CodeStore, Rng};
use serde::{Deserialize, Serialize};

/// Code lifetime (seconds).
pub const CODE_TTL_SECS: u64 = 60;
/// How long a redeemed-code tombstone outlives the code's own expiry
/// (seconds): covers clock skew and slow replays.
pub const REDEEMED_RETENTION_SECS: u64 = 600;
/// Entropy of an authorization code in bytes.
pub const CODE_BYTES: usize = 32;

/// Stored code record (keyed by `secret_hash(code)`; plaintext never stored).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizationCodeRecord {
    /// SHA-256 of the code.
    pub code_hash: [u8; 32],
    /// The authorization it completes (binds client_id, redirect_uri,
    /// code_challenge, resource and scopes).
    pub authorization: ValidatedAuthorization,
    /// User identity established upstream (binds `sub`).
    pub identity: UpstreamIdentity,
    /// Absolute expiry (unix seconds).
    pub expires_at: u64,
}

/// Issue a code for a completed authorization.
///
/// Contract: [`CODE_BYTES`] of RNG entropy, base64url; store the hashed
/// record with `expires_at = now + CODE_TTL_SECS`; return the plaintext code.
/// RNG failure aborts before anything is stored.
pub fn issue_code<S, R, C>(
    store: &S,
    rng: &R,
    clock: &C,
    authorization: ValidatedAuthorization,
    identity: UpstreamIdentity,
) -> Result<String, OAuthError>
where
    S: CodeStore + ?Sized,
    R: Rng + ?Sized,
    C: Clock + ?Sized,
{
    let code = crate::random_secret(rng, CODE_BYTES)?;
    let record = AuthorizationCodeRecord {
        code_hash: crate::secret_hash(&code),
        authorization,
        identity,
        expires_at: clock.now_unix().saturating_add(CODE_TTL_SECS),
    };
    store.insert_code(&record)?;
    Ok(code)
}

fn grant_err(desc: &'static str) -> OAuthError {
    OAuthError::new(OAuthErrorCode::InvalidGrant, desc)
}

/// Redeem a code at the token endpoint.
///
/// Contract: `take_code` (atomic, one-time) by hash -> none => `invalid_grant`;
/// expired (`now >= expires_at`) => `invalid_grant`; `client_id` and
/// `redirect_uri` must equal the stored values (`invalid_grant`); `resource`,
/// if sent, must equal the stored resource (`invalid_target`);
/// [`crate::pkce::verify_s256`] must hold (`invalid_grant`). Any failure
/// after the take still consumes the code (no retry).
pub fn redeem_code<S, C>(
    store: &S,
    clock: &C,
    code: &str,
    client_id: &str,
    redirect_uri: &str,
    code_verifier: &str,
    resource: Option<&str>,
) -> Result<AuthorizationCodeRecord, OAuthError>
where
    S: CodeStore + ?Sized,
    C: Clock + ?Sized,
{
    let record = store
        .take_code(&crate::secret_hash(code))?
        .ok_or(grant_err("invalid authorization code"))?;
    if clock.now_unix() >= record.expires_at {
        return Err(grant_err("authorization code expired"));
    }
    let auth = &record.authorization;
    if auth.client_id != client_id {
        return Err(grant_err("code was issued to another client"));
    }
    if auth.redirect_uri != redirect_uri {
        return Err(grant_err("redirect_uri mismatch"));
    }
    if let Some(r) = resource {
        let same = ruvector_edge_auth::ResourceUrl::parse(r).is_ok_and(|r| r == auth.resource);
        if !same {
            return Err(OAuthError::new(
                OAuthErrorCode::InvalidTarget,
                "resource differs from the authorized one",
            ));
        }
    }
    if !crate::pkce::verify_s256(code_verifier, &auth.code_challenge) {
        return Err(grant_err("PKCE verification failed"));
    }
    Ok(record)
}

/// Remember that `record` was redeemed and started grant `family_id`
/// (tombstone until `record.expires_at + REDEEMED_RETENTION_SECS`). Call
/// after a successful [`redeem_code`] and before issuing any token.
pub fn record_redemption<S: CodeStore + ?Sized>(
    store: &S,
    record: &AuthorizationCodeRecord,
    family_id: &str,
) -> Result<(), OAuthError> {
    let until = record.expires_at.saturating_add(REDEEMED_RETENTION_SECS);
    store.record_redeemed(&record.code_hash, family_id, until)?;
    Ok(())
}

/// The grant started by an earlier redemption of `code`, if `code` is a
/// replay of a redeemed code (unexpired tombstone). The token endpoint
/// revokes that family (RFC 6749 §4.1.2, OAuth 2.1 §4.1.3).
pub fn replayed_family<S, C>(store: &S, clock: &C, code: &str) -> Result<Option<String>, OAuthError>
where
    S: CodeStore + ?Sized,
    C: Clock + ?Sized,
{
    Ok(store.redeemed_family(&crate::secret_hash(code), clock.now_unix())?)
}
