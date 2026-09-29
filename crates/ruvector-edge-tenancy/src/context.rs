//! Per-request tenant context (ADR-351 §4.1).

use crate::error::TenancyError;
use crate::names::TenantKey;
use ruvector_edge_auth::{CapabilitySet, RouteSurface, TokenKind, VerifiedClaims};

/// Built once per request from verified claims. Storage APIs take this, never
/// a raw tenant string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantContext {
    tenant_key: TenantKey,
    iss: String,
    org_id: String,
    workspace_id: String,
    sub: String,
    client_id: String,
    capabilities: CapabilitySet,
    token_kind: TokenKind,
    jti: Option<String>,
}

#[allow(missing_docs)]
impl TenantContext {
    /// The only constructor.
    ///
    /// Contract: `org_id` and `workspace_id` present and valid per
    /// [`crate::validate_claim_component`] (else
    /// [`TenancyError::InvalidTenantClaim`] -> 401); `tenant_key` from
    /// [`crate::derive_tenant_key`]`(iss, org_id, workspace_id)`;
    /// capabilities from `ruvector_edge_auth::scopes::capabilities_for`
    /// for `surface`. `account_id` and `client_id`/`aud` never feed the key.
    pub fn from_verified(
        claims: VerifiedClaims,
        surface: RouteSurface,
    ) -> Result<Self, TenancyError> {
        let _ = (claims, surface);
        Err(TenancyError::NotImplemented(
            "context::TenantContext::from_verified",
        ))
    }

    pub fn tenant_key(&self) -> &TenantKey {
        &self.tenant_key
    }
    pub fn iss(&self) -> &str {
        &self.iss
    }
    pub fn org_id(&self) -> &str {
        &self.org_id
    }
    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }
    /// Actor for audit rows and the per-user rate-limit sub-key.
    pub fn sub(&self) -> &str {
        &self.sub
    }
    pub fn client_id(&self) -> &str {
        &self.client_id
    }
    pub fn capabilities(&self) -> CapabilitySet {
        self.capabilities
    }
    pub fn token_kind(&self) -> TokenKind {
        self.token_kind
    }
    pub fn jti(&self) -> Option<&str> {
        self.jti.as_deref()
    }
}
