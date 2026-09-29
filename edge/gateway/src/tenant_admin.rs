//! Tenant administration over the DO backend (ADR-351 §4.2, §5.8, §7.2):
//! `GET/POST /v1/tenant/members`, `DELETE /v1/tenant/members/{sub}`,
//! `POST /v1/tenant/deny` (`ruvector:admin` + owner) and `DELETE
//! /v1/collections/{c}` (`ruvector:write` + owner, ADR §5.3/§7.2),
//! plus the per-request tenant deny check with its 30 s isolate cache.
//!
//! Scope before role, like every other route: a token without the scope
//! gets `403 insufficient_scope` (step-up), a claimed tenant's non-owner
//! `403 role_required`, an unclaimed tenant `403 not_claimed`. The ledger
//! re-checks ownership on every mutating call.

use crate::backend::Backend;
use crate::durable::wipe::WipeRequest;
use crate::ledger_core::admin::{AdminCall, AdminOut, AdminRequest, DenyKind, DenyPut, DenyWire};
use crate::service::{self, access, Call};
use crate::wire::{unavailable, DeltaWire, Reply};
use ruvector_edge_auth::Capability;
use ruvector_edge_store::{CallerContext, ErrorCode, OpError};
use ruvector_edge_tenancy::validate::{validate_edge_subject, validate_edge_token_id};
use ruvector_edge_tenancy::{ledger_do_name, Role, TenantKey};
use serde::Deserialize;
use serde_json::{json, Value as Json};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// Result of an admin route: status and JSON body.
pub type Out = Result<(u16, Json), OpError>;

/// Deny-list cache lifetime per tenant and isolate (§5.8: exposure after a
/// write elsewhere ≤ 30 s; a write through this isolate invalidates).
pub const DENY_CACHE_TTL_S: u64 = 30;
/// Tenants cached per isolate (the whole cache is dropped when full).
pub const DENY_CACHE_MAX: usize = 1024;
/// Longest `reason`.
pub const MAX_REASON_BYTES: usize = 256;

thread_local! {
    static DENY_CACHE: RefCell<BTreeMap<String, (u64, Vec<DenyWire>)>> =
        const { RefCell::new(BTreeMap::new()) };
}

/// Forget `tenant`'s cached deny list (after a write through this isolate).
pub fn invalidate(tenant: &TenantKey) {
    DENY_CACHE.with(|c| c.borrow_mut().remove(tenant.as_str()));
}

async fn ledger<B: Backend>(
    b: &B,
    tenant: &TenantKey,
    admin: AdminCall,
) -> Result<AdminOut, OpError> {
    let req = AdminRequest {
        tenant_key: tenant.as_str().to_string(),
        admin,
    };
    let body = serde_json::to_string(&req).map_err(|_| unavailable())?;
    let text = b.call_ledger(&ledger_do_name(tenant), body).await?;
    match serde_json::from_str::<Reply<AdminOut>>(&text) {
        Ok(Ok(out)) => Ok(out),
        Ok(Err(e)) => Err(e.into_op()),
        Err(_) => Err(unavailable()),
    }
}

fn unexpected() -> OpError {
    OpError::new(ErrorCode::ServerError, "unexpected durable object reply")
}

fn step_up(cap: Capability) -> OpError {
    OpError {
        code: ErrorCode::InsufficientScope,
        detail: "insufficient scope",
        scope: Some(cap.satisfying_scope()),
    }
}

/// `ruvector:admin` (step-up names it).
fn need_admin(ctx: &CallerContext) -> Result<(), OpError> {
    if ctx.scope_caps().contains(Capability::Admin) {
        Ok(())
    } else {
        Err(step_up(Capability::Admin))
    }
}

/// The caller must own the tenant.
async fn need_owner<B: Backend>(b: &B, ctx: &CallerContext) -> Result<(), OpError> {
    let a = access(b, ctx).await?;
    if !a.claimed {
        return Err(OpError::new(ErrorCode::NotClaimed, "tenant not claimed"));
    }
    if a.role != Some(Role::Owner) {
        return Err(OpError::new(ErrorCode::RoleRequired, "owner required"));
    }
    Ok(())
}

fn body<T: for<'de> Deserialize<'de>>(raw: &[u8]) -> Result<T, OpError> {
    serde_json::from_slice(raw).map_err(|_| OpError::invalid("malformed body"))
}

/// `GET /v1/tenant/members`.
pub async fn members<B: Backend>(b: &B, ctx: &CallerContext) -> Out {
    need_admin(ctx)?;
    need_owner(b, ctx).await?;
    let actor = ctx.sub().to_string();
    match ledger(b, ctx.tenant_key(), AdminCall::Members { actor }).await? {
        AdminOut::Members { members } => Ok((200, json!({ "members": members }))),
        _ => Err(unexpected()),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InviteBody {
    sub: String,
    #[serde(default)]
    role: Option<String>,
}

/// `POST /v1/tenant/members` `{sub, role?: viewer|editor}` (default viewer).
pub async fn invite<B: Backend>(b: &B, ctx: &CallerContext, raw: &[u8], now: u64) -> Out {
    need_admin(ctx)?;
    need_owner(b, ctx).await?;
    let InviteBody { sub, role } = body(raw)?;
    let role = role.unwrap_or_else(|| "viewer".into());
    if !matches!(role.as_str(), "viewer" | "editor") {
        return Err(OpError::invalid("role must be viewer|editor"));
    }
    validate_edge_subject(&sub).map_err(|_| OpError::invalid("invalid member sub"))?;
    let actor = ctx.sub().to_string();
    let call = AdminCall::Invite {
        actor,
        sub,
        role,
        now,
    };
    let out = ledger(b, ctx.tenant_key(), call).await;
    invalidate(ctx.tenant_key());
    match out? {
        AdminOut::Member { sub, role } => Ok((201, json!({ "sub": sub, "role": role }))),
        _ => Err(unexpected()),
    }
}

/// `DELETE /v1/tenant/members/{sub}`: removes a non-owner member (the
/// owner, hence the last owner, is `409`) and denies its `sub` for the
/// remaining access-token life.
pub async fn remove<B: Backend>(b: &B, ctx: &CallerContext, sub: &str, now: u64) -> Out {
    need_admin(ctx)?;
    need_owner(b, ctx).await?;
    validate_edge_subject(sub).map_err(|_| OpError::not_found())?;
    let call = AdminCall::Remove {
        actor: ctx.sub().to_string(),
        sub: sub.to_string(),
        now,
    };
    let out = ledger(b, ctx.tenant_key(), call).await;
    invalidate(ctx.tenant_key());
    match out? {
        AdminOut::Done => Ok((200, json!({ "sub": sub, "removed": true }))),
        _ => Err(unexpected()),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DenyBody {
    kind: String,
    value: String,
    ttl_s: u64,
    #[serde(default)]
    reason: Option<String>,
}

fn client_id_ok(v: &str) -> bool {
    (1..=128).contains(&v.len())
        && v.bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
}

/// Validate a deny body into what the ledger stores.
pub fn deny_entry(ctx: &CallerContext, raw: &[u8]) -> Result<DenyPut, OpError> {
    let d: DenyBody = body(raw)?;
    let kind = match d.kind.as_str() {
        "jti" => DenyKind::Jti,
        "family_id" => DenyKind::FamilyId,
        "sub" => DenyKind::Sub,
        "client_id" => DenyKind::ClientId,
        "org" | "kid" => return Err(OpError::invalid("org and kid entries are operator-only")),
        _ => return Err(OpError::invalid("kind must be jti|family_id|sub|client_id")),
    };
    let shape_ok = match kind {
        DenyKind::Jti | DenyKind::FamilyId => validate_edge_token_id(&d.value, "deny").is_ok(),
        DenyKind::Sub => validate_edge_subject(&d.value).is_ok(),
        DenyKind::ClientId => client_id_ok(&d.value),
    };
    if !shape_ok {
        return Err(OpError::invalid("malformed deny value"));
    }
    let own = match kind {
        DenyKind::Jti => ctx.jti(),
        DenyKind::FamilyId => ctx.family_id(),
        DenyKind::Sub => ctx.sub(),
        DenyKind::ClientId => ctx.client_id(),
    };
    if d.value == own {
        return Err(OpError::invalid("entry would deny the caller"));
    }
    if d.ttl_s == 0 || d.ttl_s > kind.max_ttl_s() {
        return Err(OpError::invalid("ttl_s out of range for kind"));
    }
    if let Some(r) = &d.reason {
        if r.len() > MAX_REASON_BYTES || r.chars().any(char::is_control) {
            return Err(OpError::invalid(
                "reason too long or has control characters",
            ));
        }
    }
    Ok(DenyPut {
        kind,
        value: d.value,
        ttl_s: d.ttl_s,
        reason: d.reason,
    })
}

/// `POST /v1/tenant/deny` `{kind, value, ttl_s, reason?}`.
pub async fn deny<B: Backend>(b: &B, ctx: &CallerContext, raw: &[u8], now: u64) -> Out {
    need_admin(ctx)?;
    need_owner(b, ctx).await?;
    let entry = deny_entry(ctx, raw)?;
    let call = AdminCall::Deny {
        actor: ctx.sub().to_string(),
        entry,
        now,
    };
    let out = ledger(b, ctx.tenant_key(), call).await;
    invalidate(ctx.tenant_key());
    match out? {
        AdminOut::Denied { entry } => Ok((201, json!(entry))),
        _ => Err(unexpected()),
    }
}

/// `DELETE /v1/collections/{c}`, tombstone first: the ledger marks the
/// named collection `Deleting` and names its uid (atomically, so a drop can
/// never land on a newer same-name collection); lookups and new writes are
/// `404` from then on. Every shard of **that uid** is wiped (releasing
/// exactly the usage each held, one at a time, and leaving a marker that
/// refuses any write that raced the drop), then the uid is purged
/// (`Deleted`, never reused, name free). A retry after a partial failure
/// finds the `Deleting` entry and finishes the wipe.
pub async fn drop<B: Backend>(b: &B, ctx: &CallerContext, name: &str, now: u64) -> Out {
    // `ROUTE_TABLE`: DELETE /v1/collections/{c} needs `CreateCollection`
    // (`ruvector:write`), and §5.3 the owner role.
    if !ctx.scope_caps().contains(Capability::CreateCollection) {
        return Err(step_up(Capability::Write));
    }
    need_owner(b, ctx).await?;
    let c = Call {
        b,
        ctx,
        dry_run: false,
        now,
    };
    service::charge(&c, service::one_op(), 1).await?;
    let begin = AdminCall::Drop {
        actor: ctx.sub().to_string(),
        name: name.to_string(),
    };
    let e = match ledger(b, ctx.tenant_key(), begin).await? {
        AdminOut::Dropped { entry } => entry,
        _ => return Err(unexpected()),
    };
    let mut released = DeltaWire::default();
    for i in service::count_of(&e)?.indices() {
        let dm = service::shard_meta(ctx, &e, i)?;
        let req = serde_json::to_string(&WipeRequest::for_shard(&dm)).map_err(|_| unavailable())?;
        let text = crate::quant_route::wipe_for(b, &e, &dm, req).await?;
        let d = match serde_json::from_str::<Reply<DeltaWire>>(&text) {
            Ok(Ok(d)) => d,
            Ok(Err(e)) => return Err(e.into_op()),
            Err(_) => return Err(unavailable()),
        };
        service::correct(&c, d.neg().quota()).await?;
        released = released.plus(d);
    }
    let purge = AdminCall::Purge {
        actor: ctx.sub().to_string(),
        uid: e.uid.clone(),
    };
    match ledger(b, ctx.tenant_key(), purge).await? {
        AdminOut::Dropped { entry } => Ok((
            200,
            json!({
                "name": name,
                "collection_uid": entry.uid,
                "deleted": true,
                "released": released.to_json(),
            }),
        )),
        _ => Err(unexpected()),
    }
}

fn hit(e: &DenyWire, ctx: &CallerContext, now: u64) -> bool {
    e.expires_at > now
        && e.value
            == match e.kind {
                DenyKind::Jti => ctx.jti(),
                DenyKind::FamilyId => ctx.family_id(),
                DenyKind::Sub => ctx.sub(),
                DenyKind::ClientId => ctx.client_id(),
            }
}

/// The tenant deny check (§5.8) for a verified caller: `401 invalid_token`
/// when its `jti`, `family_id`, `sub` or `client_id` is denied.
pub async fn deny_check<B: Backend>(b: &B, ctx: &CallerContext, now: u64) -> Result<(), OpError> {
    let tk = ctx.tenant_key().as_str();
    let cached = DENY_CACHE.with(|c| {
        c.borrow()
            .get(tk)
            .filter(|(at, _)| *at <= now && now < at.saturating_add(DENY_CACHE_TTL_S))
            .map(|(_, v)| v.clone())
    });
    let entries = match cached {
        Some(v) => v,
        None => {
            let v = match ledger(b, ctx.tenant_key(), AdminCall::DenyList { now }).await? {
                AdminOut::DenyList { entries } => entries,
                _ => return Err(unexpected()),
            };
            DENY_CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.len() >= DENY_CACHE_MAX {
                    c.clear();
                }
                c.insert(tk.to_string(), (now, v.clone()));
            });
            v
        }
    };
    if entries.iter().any(|e| hit(e, ctx, now)) {
        return Err(OpError::new(ErrorCode::InvalidToken, "token denied"));
    }
    Ok(())
}
