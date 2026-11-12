//! Request authentication, independent of the Workers types so the full
//! verify -> tenant path is tested natively with mocked key sources.

use crate::config::GatewayConfig;
use ruvector_edge_auth::prm;
use ruvector_edge_auth::scopes::capabilities_for;
use ruvector_edge_auth::{
    bearer_token, AuthError, CapabilitySet, ClaimsPolicy, Clock, KeySource, ResourceUrl,
    RouteSurface, Verifier,
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

/// A verified request: the tenant context plus what the token itself
/// grants — scope-derived capabilities before the role intersection (the
/// role lives in the tenant's `TenantLedger`) — and its scopes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Authenticated {
    /// Tenant context (built with no role: its own capabilities are empty).
    pub ctx: TenantContext,
    /// `capabilities_for(scopes, kind, surface)`.
    pub scope_caps: CapabilitySet,
    /// Granted scopes, verbatim.
    pub scopes: Vec<String>,
    /// `act.sub` of an RFC 8693-exchanged token (the adapter acting for the
    /// user, ADR-351 §16.3); `None` otherwise.
    pub act_sub: Option<String>,
}

/// [`authenticate_full`], tenant context only (native tests).
#[cfg(test)]
pub async fn authenticate<E: KeySource, U: KeySource, C: Clock>(
    authorization: Option<&str>,
    cfg: &GatewayConfig,
    resource: &ResourceUrl,
    surface: RouteSurface,
    edge_keys: E,
    upstream_keys: impl FnOnce() -> U,
    clock: C,
) -> Result<TenantContext, Denied> {
    let a = authenticate_full(
        authorization,
        cfg,
        resource,
        surface,
        edge_keys,
        upstream_keys,
        clock,
    )
    .await?;
    Ok(a.ctx)
}

/// Verify `authorization` for `resource`: edge-AS tokens with `aud` exactly
/// `resource`; upstream first-party tokens only when configured.
pub async fn authenticate_full<E: KeySource, U: KeySource, C: Clock>(
    authorization: Option<&str>,
    cfg: &GatewayConfig,
    resource: &ResourceUrl,
    surface: RouteSurface,
    edge_keys: E,
    upstream_keys: impl FnOnce() -> U,
    clock: C,
) -> Result<Authenticated, Denied> {
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
    // Exchanged tokens are minted for `…/v1` only (ADR-351 §5.6), so the
    // audience check already refuses them on `/v1/mcp`; an actor on the MCP
    // surface is refused here as well, whatever the audience policy says.
    let act_sub = claims.act_sub().map(str::to_string);
    if act_sub.is_some() && surface == RouteSurface::Mcp {
        return Err(invalid());
    }
    // The role is read from the tenant's ledger DO afterwards, so the
    // context is built without one; callers intersect `scope_caps` with it
    // (scope before role, ADR-351 §5.3).
    let scope_caps = capabilities_for(claims.scopes(), claims.kind(), surface);
    let scopes = claims.scopes().to_vec();
    let ctx = TenantContext::from_verified(claims, None, surface).map_err(|_| invalid())?;
    Ok(Authenticated {
        ctx,
        scope_caps,
        scopes,
        act_sub,
    })
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
