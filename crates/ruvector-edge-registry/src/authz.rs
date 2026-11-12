//! The registry authorization matrix (ADR-351 §3: pull → read, push /
//! import → write, public publish → `ruvector:publish` ∩ owner).
//!
//! Capabilities arrive already intersected with the caller's role
//! (`TenantContext::capabilities`), so this module only checks set
//! membership plus the tenant / visibility relation. Denials are ordered:
//!
//! 1. **Not visible → [`Denial::NotFound`]**, whatever the capabilities. A
//!    cross-tenant caller can never tell a private or tenant version from a
//!    missing one (404, not 403).
//! 2. **Write actions on another tenant's visible package →
//!    [`Denial::NotOwner`]**.
//! 3. **Missing capability → [`Denial::Missing`]** (403 `insufficient_scope`
//!    with the step-up scope).
//!
//! | action | capability | tenant | also |
//! |---|---|---|---|
//! | `Pull` (get, pull, list) | `Read` | any (if visible) | — |
//! | `Yank` / unyank | `Write` | owner | uploader, or `Admin`; unyank of someone else's yank needs `Admin` |
//! | `Publish` (public) | `PublishPublic` | owner | — |
//!
//! Pushing a new version is decided on the **scope**, not on an existing
//! version ([`authorize_scope_push`]): `Write`, and the scope must be owned
//! by the caller's tenant (claimed in the scope directory and adopted by the
//! scope's registry DO; see [`crate::scopes`]). Scopes are a public
//! namespace, so a foreign scope is `NotOwner` (403) and reveals nothing
//! about the packages in it. Every upload-session mutation (parts, abort,
//! plan, finalize) re-checks `Write`.

use crate::manifest::Visibility;
use ruvector_edge_auth::{Capability, CapabilitySet};
use ruvector_edge_tenancy::{TenantContext, TenantKey};

/// The authenticated caller, as the registry sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Caller {
    /// Caller's tenant.
    pub tenant: TenantKey,
    /// Edge subject.
    pub sub: String,
    /// Effective capabilities (scope ∩ role).
    pub caps: CapabilitySet,
}

impl Caller {
    /// From a request's tenant context.
    pub fn from_context(ctx: &TenantContext) -> Self {
        Caller {
            tenant: ctx.tenant_key().clone(),
            sub: ctx.sub().to_string(),
            caps: ctx.capabilities(),
        }
    }

    fn has(&self, c: Capability) -> bool {
        self.caps.contains(c)
    }
}

/// What the caller wants to do to one version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    /// Read the manifest or bytes, or see it in a listing.
    Pull,
    /// Yank or unyank.
    Yank,
    /// Make public.
    Publish,
}

impl Action {
    /// Every action, for exhaustive tests.
    pub const ALL: [Action; 3] = [Action::Pull, Action::Yank, Action::Publish];
}

/// The version being acted on.
#[derive(Debug, Clone, Copy)]
pub struct Target<'a> {
    /// Owning tenant.
    pub owner: &'a TenantKey,
    /// Uploader.
    pub created_by: &'a str,
    /// Current visibility.
    pub visibility: Visibility,
}

/// Why an action was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denial {
    /// 404: invisible to this caller.
    NotFound,
    /// 403: another tenant's (or, for yank, another uploader's) package.
    NotOwner,
    /// 403 `insufficient_scope`: this capability is missing.
    Missing(Capability),
}

impl From<Denial> for crate::error::RegistryError {
    fn from(d: Denial) -> Self {
        match d {
            Denial::NotFound => crate::error::RegistryError::NotFound,
            Denial::NotOwner => crate::error::RegistryError::NotOwner,
            Denial::Missing(c) => crate::error::RegistryError::Forbidden(c),
        }
    }
}

/// `true` if `caller` may know the version exists.
pub fn visible(caller: &Caller, t: &Target<'_>) -> bool {
    let same = caller.tenant == *t.owner;
    match t.visibility {
        Visibility::Public => true,
        Visibility::Tenant => same,
        Visibility::Private => {
            same && (caller.sub == t.created_by || caller.has(Capability::Admin))
        }
    }
}

/// Decide `action` on `t`.
pub fn authorize(caller: &Caller, action: Action, t: &Target<'_>) -> Result<(), Denial> {
    if !visible(caller, t) {
        return Err(Denial::NotFound);
    }
    let need = match action {
        Action::Pull => Capability::Read,
        Action::Yank => Capability::Write,
        Action::Publish => Capability::PublishPublic,
    };
    if action != Action::Pull && caller.tenant != *t.owner {
        return Err(Denial::NotOwner);
    }
    if !caller.has(need) {
        return Err(Denial::Missing(need));
    }
    if action == Action::Yank && caller.sub != t.created_by && !caller.has(Capability::Admin) {
        return Err(Denial::NotOwner);
    }
    Ok(())
}

/// Decide a push into a scope whose current owner is `scope_owner`. Scopes
/// are a public namespace, so another tenant's scope is `NotOwner`, not
/// `NotFound`. (The registry refuses an unadopted scope with
/// `ScopeUnclaimed` before calling this.)
pub fn authorize_scope_push(
    caller: &Caller,
    scope_owner: Option<&TenantKey>,
) -> Result<(), Denial> {
    if scope_owner.is_some_and(|o| *o != caller.tenant) {
        return Err(Denial::NotOwner);
    }
    if !caller.has(Capability::Write) {
        return Err(Denial::Missing(Capability::Write));
    }
    Ok(())
}
