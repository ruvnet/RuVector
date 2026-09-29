//! Per-request tenant context (ADR-351 §4.1).

use crate::error::TenancyError;
use crate::membership::{effective_capabilities, Role};
use crate::names::{derive_tenant_key, TenantKey};
use crate::validate::{validate_edge_subject, validate_edge_token_id};
use ruvector_edge_auth::subject::edge_subject;
use ruvector_edge_auth::{CapabilitySet, RouteSurface, TokenKind, VerifiedClaims};

/// The compiled-in upstream issuer (ADR-351 §4.1 step 1). Every tenant is
/// keyed on this value, never on the edge AS hostname, so the edge path and
/// the §5.5 upstream-first-party path land on the same tenant and an edge
/// hostname change (custom domain, dev vs prod issuer) never re-keys tenants.
pub const UPSTREAM_ISSUER: &str = "https://auth.cognitum.one";

/// Built once per request from verified claims. Storage APIs take this, never
/// a raw tenant string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantContext {
    tenant_key: TenantKey,
    upstream_iss: String,
    org_id: String,
    workspace_id: String,
    sub: String,
    client_id: String,
    family_id: String,
    jti: String,
    capabilities: CapabilitySet,
    token_kind: TokenKind,
    role: Option<Role>,
}

/// Resolve the upstream issuer that namespaces the tenant.
///
/// - `UpstreamFirstParty`: the token's own `iss` (§5.5), which must equal
///   [`UPSTREAM_ISSUER`].
/// - `EdgeIssued`: the token's `upstream_iss` claim (§5.2), which is
///   **required** and must equal [`UPSTREAM_ISSUER`]. The edge `iss` never
///   feeds the tenant (§4.1 step 2, §16.2).
///
/// The returned value is the claim itself (checked equal to the constant),
/// so the tenant key is a function of the token's own `upstream_iss`.
fn upstream_iss_of(claims: &VerifiedClaims) -> Result<&str, TenancyError> {
    let claimed = match claims.kind() {
        TokenKind::UpstreamFirstParty => Some(claims.iss()),
        TokenKind::EdgeIssued => claims.upstream_iss(),
    };
    match claimed {
        Some(iss) if iss == UPSTREAM_ISSUER => Ok(iss),
        _ => Err(TenancyError::InvalidTenantClaim("upstream_iss")),
    }
}

#[allow(missing_docs)]
impl TenantContext {
    /// The only constructor.
    ///
    /// - `upstream_iss` (edge: the claim; upstream: `iss`) must equal
    ///   [`UPSTREAM_ISSUER`]; `org_id` and
    ///   `workspace_id` must be present and match `^[A-Za-z0-9_-]{1,64}$`;
    ///   `sub` must be an edge subject `^es1_[a-z2-7]{26}$` (edge tokens) or
    ///   is normalised to one with `subject::edge_subject(upstream_iss, sub)`
    ///   (upstream first-party tokens, §5.5);
    ///   `jti` and `family_id` are required (edge tokens: 16-byte base64url).
    ///   Any failure is [`TenancyError::InvalidTenantClaim`] (401).
    /// - Upstream-first-party tokens are accepted **only** on
    ///   [`RouteSurface::Rest`] (§5.5); elsewhere `InvalidTenantClaim
    ///   ("token_kind")`.
    /// - `tenant_key = derive_tenant_key(upstream_iss, org_id, workspace_id)`
    ///   (§16.2 formula `base32lower(sha256("v1|" + upstream_iss + "|" +
    ///   org_id + "|" + workspace_id))[0..26]`).
    ///   `account_id`, `client_id`, `aud`, `family_id` and `jti` never feed it.
    /// - `membership` is the caller's role read from the tenant's
    ///   `TenantLedger` (§4.2). Capabilities = scope ∩ role (§5.3);
    ///   `membership = None` (not a member) grants none.
    pub fn from_verified(
        claims: VerifiedClaims,
        membership: Option<Role>,
        surface: RouteSurface,
    ) -> Result<Self, TenancyError> {
        let kind = claims.kind();
        if kind == TokenKind::UpstreamFirstParty && surface != RouteSurface::Rest {
            return Err(TenancyError::InvalidTenantClaim("token_kind"));
        }
        let upstream_iss = upstream_iss_of(&claims)?;
        let org_id = claims
            .org_id()
            .ok_or(TenancyError::InvalidTenantClaim("org_id"))?;
        let workspace_id = claims
            .workspace_id()
            .ok_or(TenancyError::InvalidTenantClaim("workspace_id"))?;
        let tenant_key = derive_tenant_key(upstream_iss, org_id, workspace_id)?;
        // §5.5: an upstream first-party token carries the raw upstream `sub`
        // (a UUID); it is normalised with the same edge-subject function the
        // edge AS mints with, so both paths yield one actor per user.
        let sub = match kind {
            TokenKind::UpstreamFirstParty => {
                if claims.sub().is_empty() {
                    return Err(TenancyError::InvalidTenantClaim("sub"));
                }
                edge_subject(upstream_iss, claims.sub())
            }
            TokenKind::EdgeIssued => claims.sub().to_string(),
        };
        validate_edge_subject(&sub)?;
        let jti = claims
            .jti()
            .filter(|v| !v.is_empty())
            .ok_or(TenancyError::InvalidTenantClaim("jti"))?;
        let family_id = claims
            .family_id()
            .filter(|v| !v.is_empty())
            .ok_or(TenancyError::InvalidTenantClaim("family_id"))?;
        if kind == TokenKind::EdgeIssued {
            validate_edge_token_id(jti, "jti")?;
            validate_edge_token_id(family_id, "family_id")?;
        }
        let capabilities = effective_capabilities(claims.scopes(), kind, surface, membership);
        Ok(TenantContext {
            tenant_key,
            upstream_iss: upstream_iss.to_string(),
            org_id: org_id.to_string(),
            workspace_id: workspace_id.to_string(),
            sub,
            client_id: claims.client_id().to_string(),
            family_id: family_id.to_string(),
            jti: jti.to_string(),
            capabilities,
            token_kind: kind,
            role: membership,
        })
    }

    pub fn tenant_key(&self) -> &TenantKey {
        &self.tenant_key
    }
    /// The tenant namespace (always [`UPSTREAM_ISSUER`] today).
    pub fn upstream_iss(&self) -> &str {
        &self.upstream_iss
    }
    pub fn org_id(&self) -> &str {
        &self.org_id
    }
    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }
    /// Edge subject: actor for audit rows, memberships and the per-user
    /// rate-limit sub-key.
    pub fn sub(&self) -> &str {
        &self.sub
    }
    pub fn client_id(&self) -> &str {
        &self.client_id
    }
    /// Grant / refresh family (ops, audit, logs, §5.8 deny list).
    pub fn family_id(&self) -> &str {
        &self.family_id
    }
    pub fn jti(&self) -> &str {
        &self.jti
    }
    /// Effective capabilities (scope ∩ role).
    pub fn capabilities(&self) -> CapabilitySet {
        self.capabilities
    }
    pub fn token_kind(&self) -> TokenKind {
        self.token_kind
    }
    /// The caller's membership role, `None` if not a member.
    pub fn role(&self) -> Option<Role> {
        self.role
    }
}
