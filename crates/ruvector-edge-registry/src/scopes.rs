//! The global scope directory: who owns which `@scope`.
//!
//! Scopes are the cross-tenant trust anchor for public pulls, so claiming
//! one is an explicit, audited step (`POST /v1/rvf/scopes/{scope}`), never a
//! side effect of a push. The directory lives in **one** registry-root
//! Durable Object (its own [`KvStore`], not D1 and not a per-scope DO),
//! because only a global view can refuse a look-alike of a scope some other
//! tenant already holds:
//!
//! | Key | Value |
//! |---|---|
//! | `skel/{skeleton}` | [`ScopeClaim`] ([`crate::name::scope_skeleton`]) |
//! | `tenant/{tenant_key}/{scope}` | `{}` (per-tenant count) |
//!
//! A claim is refused if its skeleton is taken by a different scope or
//! another tenant (`ScopeTaken`), or if the tenant already holds
//! `max_scopes_per_tenant` scopes. Reserved and brand look-alikes never
//! parse as a [`Scope`] in the first place. After a claim, the gateway
//! hands the [`ScopeClaim`] to the scope's registry DO
//! ([`crate::Registry::adopt_scope`]); pushes require it there.

use crate::authz::Caller;
use crate::error::{RegistryError, Result};
use crate::manifest::tenant_key_serde;
use crate::name::{scope_skeleton, Scope};
use crate::ports::{Clock, KvStore, StoreError};
use ruvector_edge_auth::Capability;
use ruvector_edge_tenancy::TenantKey;
use serde::{Deserialize, Serialize};

/// Default cap on scopes per tenant.
pub const DEFAULT_MAX_SCOPES_PER_TENANT: usize = 10;

/// An accepted scope claim (also the audit record).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeClaim {
    /// The scope.
    pub scope: String,
    /// Owning tenant.
    #[serde(with = "tenant_key_serde")]
    pub owner: TenantKey,
    /// Who claimed it (edge subject; internal, never shown cross-tenant).
    pub claimed_by: String,
    /// When (seconds).
    pub claimed_at: u64,
}

impl ScopeClaim {
    /// The claimed scope, re-parsed.
    pub fn scope(&self) -> Result<Scope> {
        Ok(Scope::parse(&self.scope)?)
    }
}

/// The directory service over the root DO's store.
pub struct ScopeDirectory<S, C> {
    store: S,
    clock: C,
    max_scopes_per_tenant: usize,
}

fn k_skel(scope: &Scope) -> String {
    format!("skel/{}", scope_skeleton(scope.as_str()))
}

fn k_tenant(t: &TenantKey) -> String {
    format!("tenant/{}/", t.as_str())
}

impl<S: KvStore, C: Clock> ScopeDirectory<S, C> {
    /// A directory with a per-tenant cap.
    pub fn new(store: S, clock: C, max_scopes_per_tenant: usize) -> Self {
        ScopeDirectory {
            store,
            clock,
            max_scopes_per_tenant,
        }
    }

    fn load(&self, key: &str) -> Result<Option<ScopeClaim>> {
        match self.store.get(key)? {
            None => Ok(None),
            Some(b) => serde_json::from_slice(&b)
                .map(Some)
                .map_err(|_| StoreError::Corrupt("scope claim").into()),
        }
    }

    /// The claim on `scope` or on a look-alike of it.
    pub fn lookup(&self, scope: &Scope) -> Result<Option<ScopeClaim>> {
        self.load(&k_skel(scope))
    }

    /// Claim `scope` for the caller's tenant. Needs `Admin` in the tenant.
    /// Idempotent for the same tenant and exact scope.
    pub fn claim(&self, caller: &Caller, scope: &Scope) -> Result<ScopeClaim> {
        if !caller.caps.contains(Capability::Admin) {
            return Err(RegistryError::Forbidden(Capability::Admin));
        }
        if let Some(existing) = self.lookup(scope)? {
            if existing.owner == caller.tenant && existing.scope == scope.as_str() {
                return Ok(existing);
            }
            return Err(RegistryError::ScopeTaken);
        }
        let held = self
            .store
            .list(&k_tenant(&caller.tenant), None, self.max_scopes_per_tenant)?;
        if held.len() >= self.max_scopes_per_tenant {
            return Err(RegistryError::TooManyScopes);
        }
        let claim = ScopeClaim {
            scope: scope.as_str().to_string(),
            owner: caller.tenant.clone(),
            claimed_by: caller.sub.clone(),
            claimed_at: self.clock.now_unix(),
        };
        let bytes = serde_json::to_vec(&claim).map_err(|_| StoreError::Corrupt("encode"))?;
        self.store.put(&k_skel(scope), &bytes)?;
        self.store.put(
            &format!("{}{}", k_tenant(&caller.tenant), scope.as_str()),
            b"{}",
        )?;
        Ok(claim)
    }

    /// Scopes held by the caller's tenant.
    pub fn scopes_of(&self, tenant: &TenantKey) -> Result<Vec<String>> {
        let prefix = k_tenant(tenant);
        Ok(self
            .store
            .list(&prefix, None, self.max_scopes_per_tenant.max(1) * 2)?
            .into_iter()
            .map(|(k, _)| k[prefix.len()..].to_string())
            .collect())
    }
}
