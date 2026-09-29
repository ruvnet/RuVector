//! Token revocation (RFC 7009).

use crate::error::OAuthError;
use crate::ports::RefreshStore;

/// Revoke a presented token.
///
/// Contract: public client identified by `client_id`; if the token is a
/// known refresh token belonging to `client_id`, revoke its whole family.
/// Unknown tokens, other clients' tokens and access tokens (JWTs are
/// short-lived and not tracked here) all succeed silently (RFC 7009 §2.2).
/// `token_type_hint` is advisory only. Only storage failures are errors.
pub fn revoke<S: RefreshStore>(
    store: &S,
    token: &str,
    client_id: &str,
    token_type_hint: Option<&str>,
) -> Result<(), OAuthError> {
    let _ = (store, token, client_id, token_type_hint);
    Err(OAuthError::not_implemented("revoke::revoke"))
}
