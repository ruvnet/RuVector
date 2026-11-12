//! Token revocation (RFC 7009).

use crate::error::{OAuthError, OAuthErrorCode};
use crate::params::Params;
use crate::ports::RefreshStore;

/// Parsed revocation request. `Debug` redacts the token.
#[derive(Clone, PartialEq, Eq)]
pub struct RevocationRequest {
    /// Presented token.
    pub token: String,
    /// Public client id.
    pub client_id: String,
    /// Advisory hint.
    pub token_type_hint: Option<String>,
}

impl std::fmt::Debug for RevocationRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RevocationRequest")
            .field("token", &crate::REDACTED)
            .field("client_id", &self.client_id)
            .field("token_type_hint", &self.token_type_hint)
            .finish()
    }
}

impl RevocationRequest {
    /// Parse form pairs: `token` and `client_id` required (`invalid_request`);
    /// `client_secret` => `invalid_client`; repeats => `invalid_request`.
    pub fn from_form(pairs: &[(String, String)]) -> Result<Self, OAuthError> {
        let p = Params::from_pairs(pairs)?;
        if p.get("client_secret").is_some() || p.get("client_assertion").is_some() {
            return Err(OAuthError::new(
                OAuthErrorCode::InvalidClient,
                "only public clients are supported",
            ));
        }
        Ok(RevocationRequest {
            token: p.require("token", "token required")?,
            client_id: p.require("client_id", "client_id required")?,
            token_type_hint: p.take("token_type_hint"),
        })
    }
}

/// Revoke a presented token.
///
/// Contract: public client identified by `client_id`; if the token is a
/// known refresh token belonging to `client_id`, revoke its whole family.
/// Unknown tokens, other clients' tokens and access tokens (JWTs are
/// short-lived and not tracked here) all succeed silently (RFC 7009 §2.2).
/// `token_type_hint` is advisory only. Only storage failures are errors.
pub fn revoke<S: RefreshStore + ?Sized>(
    store: &S,
    token: &str,
    client_id: &str,
    token_type_hint: Option<&str>,
) -> Result<(), OAuthError> {
    let _ = token_type_hint;
    if let Some(rec) = store.get_refresh(&crate::secret_hash(token))? {
        if rec.client_id == client_id {
            store.revoke_family(&rec.family_id)?;
        }
    }
    Ok(())
}
