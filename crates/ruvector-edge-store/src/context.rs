//! The already-verified caller context the store trusts.
//!
//! The gateway verifies the bearer token (`ruvector-edge-auth`), derives the
//! tenant (`ruvector-edge-tenancy::TenantContext`) and hands this crate a
//! [`CallerContext`]. It carries the **token-derived** capabilities (scope
//! only, before the role intersection) separately from the role, because
//! the role is looked up here, in the tenant's ledger, and §16.3 must tell
//! `insufficient_scope` apart from `role_required` / `not_claimed`.

use crate::error::{ErrorCode, OpError};
use ruvector_edge_auth::CapabilitySet;
use ruvector_edge_tenancy::meta::{META_COLLECTION_UID, META_SERVICE, META_SHARD, META_TENANT_KEY};
use ruvector_edge_tenancy::{
    CollectionUid, DoMeta, LedgerMeta, Service, ShardIndex, TenantContext, TenantKey,
};

/// Verified caller identity for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallerContext {
    tenant_key: TenantKey,
    sub: String,
    client_id: String,
    jti: String,
    family_id: String,
    act_sub: Option<String>,
    scope_caps: CapabilitySet,
}

impl CallerContext {
    /// From a verified [`TenantContext`] plus the token's scope-derived
    /// capabilities (`scopes::capabilities_for(scopes, kind, surface)`) and
    /// the exchanged token's `act.sub`, if any.
    pub fn from_tenant(
        ctx: &TenantContext,
        scope_caps: CapabilitySet,
        act_sub: Option<String>,
    ) -> Self {
        CallerContext {
            tenant_key: ctx.tenant_key().clone(),
            sub: ctx.sub().to_string(),
            client_id: ctx.client_id().to_string(),
            jti: ctx.jti().to_string(),
            family_id: ctx.family_id().to_string(),
            act_sub,
            scope_caps,
        }
    }

    /// Field-wise constructor for gateways that already hold verified
    /// values (and for tests). `tenant_key` is a typed [`TenantKey`], so a
    /// raw tenant string can never reach storage.
    pub fn new(
        tenant_key: TenantKey,
        sub: impl Into<String>,
        client_id: impl Into<String>,
        jti: impl Into<String>,
        family_id: impl Into<String>,
        act_sub: Option<String>,
        scope_caps: CapabilitySet,
    ) -> Self {
        CallerContext {
            tenant_key,
            sub: sub.into(),
            client_id: client_id.into(),
            jti: jti.into(),
            family_id: family_id.into(),
            act_sub,
            scope_caps,
        }
    }

    /// Tenant key derived from the token.
    pub fn tenant_key(&self) -> &TenantKey {
        &self.tenant_key
    }
    /// Edge subject.
    pub fn sub(&self) -> &str {
        &self.sub
    }
    /// OAuth client.
    pub fn client_id(&self) -> &str {
        &self.client_id
    }
    /// Token id.
    pub fn jti(&self) -> &str {
        &self.jti
    }
    /// Grant family.
    pub fn family_id(&self) -> &str {
        &self.family_id
    }
    /// Acting adapter (`act.sub` on exchanged tokens).
    pub fn act_sub(&self) -> Option<&str> {
        self.act_sub.as_deref()
    }
    /// Scope-derived capabilities (before the role intersection).
    pub fn scope_caps(&self) -> CapabilitySet {
        self.scope_caps
    }
}

/// The `TenantLedger` identity for a verified tenant key.
///
/// Built through `LedgerMeta::from_kv` because tenancy's only other
/// constructor takes a full `TenantContext`.
pub fn ledger_meta_for(tenant: &TenantKey) -> Result<LedgerMeta, OpError> {
    LedgerMeta::from_kv([(META_TENANT_KEY, tenant.as_str())])
        .ok()
        .flatten()
        .ok_or(OpError::new(ErrorCode::ServerError, "ledger identity"))
}

/// The `VectorShard` identity for `(tenant, collection_uid, shard)`.
pub fn shard_meta_for(
    tenant: &TenantKey,
    uid: CollectionUid,
    shard: ShardIndex,
) -> Result<DoMeta, OpError> {
    let (hex, idx) = (uid.to_hex(), shard.get().to_string());
    DoMeta::from_kv([
        (META_TENANT_KEY, tenant.as_str()),
        (META_SERVICE, Service::Vector.as_str()),
        (META_COLLECTION_UID, hex.as_str()),
        (META_SHARD, idx.as_str()),
    ])
    .ok()
    .flatten()
    .ok_or(OpError::new(ErrorCode::ServerError, "shard identity"))
}
