//! End-to-end bearer verification: header -> JWS -> key -> signature ->
//! audience -> claims.

use crate::audience::AudiencePolicy;
use crate::claims::{validate, ClaimsPolicy, RawClaims, TokenKind, VerifiedClaims};
use crate::clock::Clock;
use crate::error::AuthError;
use crate::jwks::KeySource;
use crate::jws::{bearer_token, parse_compact, starts_with_object, verify_es256};

/// Resource-server verifier.
///
/// `edge_keys` resolves keys from the edge AS JWKS; `upstream_keys` (only
/// consulted when `audience.upstream` is `Some`) from
/// `https://auth.cognitum.one/.well-known/jwks.json`.
pub struct Verifier<E: KeySource, U: KeySource, C: Clock> {
    edge_keys: E,
    upstream_keys: Option<U>,
    clock: C,
    audience: AudiencePolicy,
    edge_claims: ClaimsPolicy,
    upstream_claims: Option<ClaimsPolicy>,
}

impl<E: KeySource, U: KeySource, C: Clock> Verifier<E, U, C> {
    /// Edge-only verifier (upstream path disabled).
    pub fn edge_only(
        edge_keys: E,
        clock: C,
        audience: AudiencePolicy,
        edge_claims: ClaimsPolicy,
    ) -> Self {
        Verifier {
            edge_keys,
            upstream_keys: None,
            clock,
            audience,
            edge_claims,
            upstream_claims: None,
        }
    }

    /// Enable the upstream first-party path. Requires `audience.upstream` to
    /// be `Some`; otherwise the upstream path stays inert. Upstream tokens are
    /// accepted only for a `kid` in `audience.upstream.accepted_kids`
    /// (enforced in [`Verifier::verify`] whatever `keys` trusts), and always
    /// need claim `typ == "access"` and `exp - iat <= 3600`
    /// (enforced by [`crate::claims::validate`] whatever `claims` says).
    pub fn with_upstream(mut self, keys: U, claims: ClaimsPolicy) -> Self {
        self.upstream_keys = Some(keys);
        self.upstream_claims = Some(claims);
        self
    }

    /// The audience policy in force.
    pub fn audience(&self) -> &AudiencePolicy {
        &self.audience
    }

    /// Verify an `Authorization` header value.
    ///
    /// Contract, in order: [`crate::bearer_token`]; [`crate::parse_compact`];
    /// decode payload to [`crate::RawClaims`]; [`AudiencePolicy::classify`];
    /// select key source by kind (upstream disabled -> `WrongIssuer`;
    /// upstream `kid` outside the pinned `accepted_kids` -> `UnknownKid`, and
    /// an empty pin -> `InvalidConfig`, both before any key lookup); fetch
    /// key by `kid`; [`crate::verify_es256`];
    /// [`AudiencePolicy::check_audience`]; [`crate::claims::validate`] with the
    /// kind's [`ClaimsPolicy`] at `clock.now_unix()`.
    ///
    /// The returned claims carry the header `kid`
    /// ([`VerifiedClaims::kid`]) so the Worker can apply the ADR §5.4.7 /
    /// §5.8 deny-list on `jti`, `family_id`, `sub`, `client_id` **and**
    /// `kid` after this returns (a Worker may also deny a compromised `kid`
    /// before the signature step by wrapping its [`KeySource`] to return
    /// [`AuthError::Denied`]).
    pub async fn verify(&self, authorization: Option<&str>) -> Result<VerifiedClaims, AuthError> {
        let token = bearer_token(authorization)?;
        let jws = parse_compact(token)?;
        let payload = jws.payload_bytes()?;
        if !starts_with_object(&payload) {
            return Err(AuthError::Malformed("claims not an object"));
        }
        let raw: RawClaims =
            serde_json::from_slice(&payload).map_err(|_| AuthError::Malformed("claims json"))?;

        let kind = self
            .audience
            .classify(raw.iss.as_deref(), jws.header.typ.as_deref())?;
        let kid = jws.header.kid.as_str();
        let (key, policy) = match kind {
            TokenKind::EdgeIssued => (self.edge_keys.verifying_key(kid).await?, &self.edge_claims),
            TokenKind::UpstreamFirstParty => match (&self.upstream_keys, &self.upstream_claims) {
                (Some(keys), Some(policy)) => {
                    self.audience.check_upstream_kid(kid)?;
                    (keys.verifying_key(kid).await?, policy)
                }
                _ => return Err(AuthError::WrongIssuer),
            },
        };
        verify_es256(&jws, &key)?;
        self.audience.check_audience(kind, raw.aud.as_ref())?;
        Ok(validate(raw, policy, kind, self.clock.now_unix())?.with_kid(kid))
    }
}

#[cfg(test)]
mod tests;
