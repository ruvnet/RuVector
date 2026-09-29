//! End-to-end bearer verification: header -> JWS -> key -> signature ->
//! audience -> claims.

use crate::audience::AudiencePolicy;
use crate::claims::{ClaimsPolicy, VerifiedClaims};
use crate::clock::Clock;
use crate::error::AuthError;
use crate::jwks::KeySource;

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
    /// be `Some`; otherwise the upstream path stays inert.
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
    /// select key source by kind (upstream disabled -> `WrongIssuer`); fetch
    /// key by `kid`; [`crate::verify_es256`];
    /// [`AudiencePolicy::check_audience`]; [`crate::claims::validate`] with the
    /// kind's [`ClaimsPolicy`] at `clock.now_unix()`. Deny-list checks are the
    /// caller's (Worker) responsibility after this returns.
    pub async fn verify(&self, authorization: Option<&str>) -> Result<VerifiedClaims, AuthError> {
        let _ = (
            authorization,
            &self.edge_keys,
            self.upstream_keys.is_some(),
            self.clock.now_unix(),
            &self.edge_claims,
            &self.upstream_claims,
        );
        Err(AuthError::NotImplemented("verifier::verify"))
    }
}
