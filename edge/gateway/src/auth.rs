//! Request authentication, independent of the Workers types so the full
//! verify -> tenant path is tested natively with mocked key sources.

use crate::config::GatewayConfig;
use ruvector_edge_auth::prm;
use ruvector_edge_auth::{
    bearer_token, AuthError, ClaimsPolicy, Clock, KeySource, ResourceUrl, RouteSurface, Verifier,
};
use ruvector_edge_tenancy::{ProblemCode, TenantContext, UPSTREAM_ISSUER};

/// A refused request: problem code plus optional `WWW-Authenticate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Denied {
    /// RFC 9457 problem code (sets the status).
    pub code: ProblemCode,
    /// Challenge for 401s (always carries `resource_metadata`).
    pub www_authenticate: Option<String>,
}

/// Map a verification error to the problem code and whether the 401
/// `invalid_token` challenge applies (503 JWKS and 500 config failures carry
/// no challenge). Every audience failure — including an edge token for the
/// gateway's other resource — is 401 `invalid_token` (ADR-351 §5.4.7);
/// `audience_not_allowed` survives only as [`AuthError::code`], the log
/// reason.
pub fn denial(err: &AuthError) -> (ProblemCode, bool) {
    match err.http_status() {
        503 => (ProblemCode::JwksUnavailable, false),
        500 => (ProblemCode::ServerError, false),
        _ => (ProblemCode::InvalidToken, true),
    }
}

/// Verify `authorization` for `resource`: edge-AS tokens with `aud` exactly
/// `resource`; upstream first-party tokens only when configured.
pub async fn authenticate<E: KeySource, U: KeySource, C: Clock>(
    authorization: Option<&str>,
    cfg: &GatewayConfig,
    resource: &ResourceUrl,
    surface: RouteSurface,
    edge_keys: E,
    upstream_keys: impl FnOnce() -> U,
    clock: C,
) -> Result<TenantContext, Denied> {
    let metadata_url = prm::metadata_url(resource);
    let invalid_described = |description: Option<&str>| Denied {
        code: ProblemCode::InvalidToken,
        www_authenticate: Some(prm::www_authenticate_invalid_token_described(
            &metadata_url,
            prm::CHALLENGE_SCOPE,
            description,
        )),
    };
    let invalid = || invalid_described(None);
    if let Err(AuthError::MissingToken) = bearer_token(authorization) {
        return Err(Denied {
            code: ProblemCode::InvalidToken,
            www_authenticate: Some(prm::www_authenticate_missing(
                &metadata_url,
                prm::CHALLENGE_SCOPE,
            )),
        });
    }
    let mut verifier = Verifier::edge_only(
        edge_keys,
        clock,
        cfg.audience_for(resource),
        // exp - iat <= 900 and `upstream_iss` == the compiled upstream issuer.
        ClaimsPolicy::edge(cfg.edge_issuer.clone(), UPSTREAM_ISSUER),
    );
    // The audience policy carries the upstream path only for the REST
    // resource (`GatewayConfig::audience_for`).
    if let (Some(up), true) = (&cfg.upstream, *resource == cfg.rest_resource) {
        let claims = ClaimsPolicy::upstream_access(up.issuer.clone());
        verifier = verifier.with_upstream(upstream_keys(), claims);
    }
    let claims = verifier
        .verify(authorization)
        .await
        .map_err(|e| match denial(&e) {
            (_, true) if e.is_audience_mismatch() => {
                invalid_described(Some(prm::AUDIENCE_MISMATCH))
            }
            (_, true) => invalid(),
            (code, false) => Denied {
                code,
                www_authenticate: None,
            },
        })?;
    // M1 has no membership source (ADR-351 §1.1: no org-role claim, no
    // membership store yet). `None` grants no capabilities; `/v1/me` needs none.
    TenantContext::from_verified(claims, None, surface).map_err(|_| invalid())
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
