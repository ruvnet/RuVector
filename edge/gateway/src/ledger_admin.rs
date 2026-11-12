//! `TenantLedger` administration calls (ADR-351 §4.2 members, §5.8 tenant
//! deny entries, §7.2 collection drop): the wire types plus their DO-side
//! handler. A separate request shape (`{"tenant_key", "admin"}`), so the
//! core `LedgerRequest` wire is untouched; `ledger_core::serve` tries it
//! when a body is not a `LedgerRequest`.
//!
//! Every mutating call names the acting `sub` and is re-checked here
//! (owner only), independently of the gateway's own role check.

use crate::wire::{CollectionWire, Reply, WireErr};
use ruvector_edge_store::{
    ledger_meta_for, ErrorCode, OpError, SqlStore, StoreError, TenantLedger, Value,
};
use ruvector_edge_tenancy::{CollectionUid, LedgerMeta, QuotaLimits, Role, TenantKey};
use serde::{Deserialize, Serialize};

/// Tenant deny table (§5.8: `scope = tenant`, `tenant_key` = this ledger).
pub const DENY_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS tenant_deny (kind TEXT, value TEXT, \
     expires_at INTEGER, created_by TEXT, reason TEXT, created_at INTEGER, \
     PRIMARY KEY (kind, value))";
const DENY_PUT: &str = "INSERT OR REPLACE INTO tenant_deny (kind, value, expires_at, created_by, \
     reason, created_at) VALUES (?, ?, ?, ?, ?, ?)";
const DENY_ACTIVE: &str = "SELECT kind, value, expires_at FROM tenant_deny WHERE expires_at > ?";
const DENY_PURGE: &str = "DELETE FROM tenant_deny WHERE expires_at <= ?";
const DENY_DELETE: &str = "DELETE FROM tenant_deny WHERE kind = ? AND value = ?";

/// Live tenant deny entries per tenant (bounded; expired rows are purged on
/// every write).
pub const MAX_DENY_ENTRIES: usize = 1000;
/// A removed member's `sub` stays denied for the longest access-token life
/// (§4.2 removal + §5.8; 15 min), so outstanding tokens die with it.
pub const REMOVAL_DENY_TTL_S: u64 = 900;

/// Tenant deny kinds (§5.8). `org` and `kid` are global, operator-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenyKind {
    /// One access token (TTL ≤ 900 s).
    Jti,
    /// One grant family (≤ 7 d).
    FamilyId,
    /// One member (≤ 30 d, renewable).
    Sub,
    /// One OAuth client for this tenant (≤ 365 d).
    ClientId,
}

impl DenyKind {
    /// Wire / storage string.
    pub fn as_str(self) -> &'static str {
        match self {
            DenyKind::Jti => "jti",
            DenyKind::FamilyId => "family_id",
            DenyKind::Sub => "sub",
            DenyKind::ClientId => "client_id",
        }
    }

    /// Parse the storage string.
    pub fn parse(s: &str) -> Option<DenyKind> {
        [
            DenyKind::Jti,
            DenyKind::FamilyId,
            DenyKind::Sub,
            DenyKind::ClientId,
        ]
        .into_iter()
        .find(|k| k.as_str() == s)
    }

    /// Longest TTL an owner may set (§5.8 caps).
    pub fn max_ttl_s(self) -> u64 {
        const DAY: u64 = 86_400;
        match self {
            DenyKind::Jti => 900,
            DenyKind::FamilyId => 7 * DAY,
            DenyKind::Sub => 30 * DAY,
            DenyKind::ClientId => 365 * DAY,
        }
    }
}

/// One active deny entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenyWire {
    /// Kind.
    pub kind: DenyKind,
    /// Denied value.
    pub value: String,
    /// Unix seconds (exclusive).
    pub expires_at: u64,
}

/// A validated deny entry to write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenyPut {
    /// Kind.
    pub kind: DenyKind,
    /// Value (already shape-checked by the gateway).
    pub value: String,
    /// `1..=kind.max_ttl_s()`.
    pub ttl_s: u64,
    /// Free text, ≤ 256 bytes.
    pub reason: Option<String>,
}

/// A membership row on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberWire {
    /// Edge subject.
    pub sub: String,
    /// `owner` | `editor` | `viewer`.
    pub role: String,
    /// Inviter (`None` for the claimant).
    pub invited_by: Option<String>,
    /// Unix seconds.
    pub created_at: u64,
}

/// An administration request to one tenant's ledger.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdminRequest {
    /// Tenant key from the verified token.
    pub tenant_key: String,
    /// The call.
    pub admin: AdminCall,
}

/// Administration calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum AdminCall {
    /// List members (owner).
    Members { actor: String },
    /// Add or change a member (owner; clears a `sub` deny entry for it).
    Invite {
        actor: String,
        sub: String,
        role: String,
        now: u64,
    },
    /// Remove a member and deny its `sub` for [`REMOVAL_DENY_TTL_S`].
    Remove {
        actor: String,
        sub: String,
        now: u64,
    },
    /// Write (or renew) a tenant deny entry (owner).
    Deny {
        actor: String,
        entry: DenyPut,
        now: u64,
    },
    /// Active entries (enforcement; any caller of this tenant).
    DenyList { now: u64 },
    /// Begin dropping a collection by name (owner): `Live` → `Deleting`
    /// (a `Deleting` one is returned as is). The reply names the uid whose
    /// shards the caller then wipes.
    Drop { actor: String, name: String },
    /// Finish a drop (owner): the `Deleting` uid, shards wiped, becomes
    /// `Deleted`. By uid, so it can never touch a newer same-name
    /// collection.
    Purge { actor: String, uid: String },
}

/// Administration results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "out", rename_all = "snake_case")]
pub enum AdminOut {
    /// For `Members`.
    Members { members: Vec<MemberWire> },
    /// For `Invite`.
    Member { sub: String, role: String },
    /// For `Deny`.
    Denied { entry: DenyWire },
    /// For `DenyList`.
    DenyList { entries: Vec<DenyWire> },
    /// For `Drop` and `Purge`.
    Dropped { entry: CollectionWire },
    /// For `Remove`.
    Done,
}

fn owner(ledger: &TenantLedger, lm: &LedgerMeta, actor: &str) -> Result<(), OpError> {
    match ledger.role_of(lm, actor)? {
        Some(Role::Owner) => Ok(()),
        _ => Err(OpError::new(ErrorCode::RoleRequired, "owner required")),
    }
}

fn int(n: u64) -> Result<Value, OpError> {
    i64::try_from(n)
        .map(Value::Int)
        .map_err(|_| OpError::invalid("timestamp out of range"))
}

fn active(store: &dyn SqlStore, now: u64) -> Result<Vec<DenyWire>, StoreError> {
    store.exec(DENY_SCHEMA, &[])?;
    let rows = store.query(DENY_ACTIVE, &[Value::Int(now.min(i64::MAX as u64) as i64)])?;
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        let text = |i: usize| r.get(i).and_then(|v| v.as_text()).map(str::to_string);
        let kind = text(0)
            .as_deref()
            .and_then(DenyKind::parse)
            .ok_or(StoreError::Corrupt("tenant_deny.kind"))?;
        let value = text(1).ok_or(StoreError::Corrupt("tenant_deny.value"))?;
        let expires_at = match r.get(2) {
            Some(Value::Int(i)) => u64::try_from(*i).ok(),
            _ => None,
        }
        .ok_or(StoreError::Corrupt("tenant_deny.expires_at"))?;
        out.push(DenyWire {
            kind,
            value,
            expires_at,
        });
    }
    Ok(out)
}

/// Write (or renew) a deny entry. `capped`: refuse a new entry past
/// [`MAX_DENY_ENTRIES`] (owner-written entries); the removal deny is not
/// capped — the removal it accompanies is already committed, and the deny
/// is defence in depth that must not turn it into an error.
fn put_deny(
    store: &dyn SqlStore,
    actor: &str,
    e: &DenyPut,
    now: u64,
    capped: bool,
) -> Result<DenyWire, OpError> {
    if e.ttl_s == 0 || e.ttl_s > e.kind.max_ttl_s() {
        return Err(OpError::invalid("ttl_s out of range for kind"));
    }
    let expires_at = now.saturating_add(e.ttl_s);
    store.exec(DENY_SCHEMA, &[])?;
    store.exec(DENY_PURGE, &[int(now)?])?;
    let live = active(store, now)?;
    let renew = live.iter().any(|d| d.kind == e.kind && d.value == e.value);
    if capped && !renew && live.len() >= MAX_DENY_ENTRIES {
        return Err(OpError::new(
            ErrorCode::QuotaExceeded,
            "too many deny entries",
        ));
    }
    let reason = e.reason.clone().map_or(Value::Null, Value::Text);
    store.exec(
        DENY_PUT,
        &[
            e.kind.as_str().into(),
            e.value.as_str().into(),
            int(expires_at)?,
            actor.into(),
            reason,
            int(now)?,
        ],
    )?;
    Ok(DenyWire {
        kind: e.kind,
        value: e.value.clone(),
        expires_at,
    })
}

/// Serve one decoded admin request over the tenant's resident ledger.
pub fn handle(
    slot: &mut Option<TenantLedger>,
    store: &dyn SqlStore,
    limits: QuotaLimits,
    req: AdminRequest,
) -> Result<AdminOut, OpError> {
    let tenant =
        TenantKey::parse(&req.tenant_key).map_err(|_| OpError::invalid("malformed tenant"))?;
    let lm = ledger_meta_for(&tenant)?;
    let ledger = crate::ledger_core::open(slot, store, limits)?;
    Ok(match req.admin {
        AdminCall::Members { actor } => {
            owner(ledger, &lm, &actor)?;
            let members = ledger
                .members(&lm)?
                .iter()
                .map(|(sub, m)| MemberWire {
                    sub: sub.clone(),
                    role: m.role.as_str().to_string(),
                    invited_by: m.invited_by.clone(),
                    created_at: m.created_at,
                })
                .collect();
            AdminOut::Members { members }
        }
        AdminCall::Invite {
            actor,
            sub,
            role,
            now,
        } => {
            let role = Role::parse(&role).ok_or(OpError::invalid("role must be viewer|editor"))?;
            // Only a genuinely new member clears a `sub` entry (the removal
            // deny); a role change keeps an owner's suspension in force.
            let new_member = ledger.role_of(&lm, &sub)?.is_none();
            ledger.put_member(store, &lm, &actor, &sub, role, now)?;
            if new_member {
                store.exec(DENY_SCHEMA, &[])?;
                store.exec(
                    DENY_DELETE,
                    &[DenyKind::Sub.as_str().into(), sub.as_str().into()],
                )?;
            }
            AdminOut::Member {
                sub,
                role: role.as_str().to_string(),
            }
        }
        AdminCall::Remove { actor, sub, now } => {
            ledger.remove_member(store, &lm, &actor, &sub)?;
            let e = DenyPut {
                kind: DenyKind::Sub,
                value: sub,
                ttl_s: REMOVAL_DENY_TTL_S,
                reason: Some("member removed".into()),
            };
            put_deny(store, &actor, &e, now, false)?;
            AdminOut::Done
        }
        AdminCall::Deny { actor, entry, now } => {
            owner(ledger, &lm, &actor)?;
            if entry.kind == DenyKind::Sub {
                // §5.8 "members only": a non-owner member (so never the
                // owner locking itself out).
                match ledger.role_of(&lm, &entry.value)? {
                    Some(Role::Viewer | Role::Editor) => {}
                    _ => return Err(OpError::invalid("sub deny needs a non-owner member")),
                }
            }
            AdminOut::Denied {
                entry: put_deny(store, &actor, &entry, now, true)?,
            }
        }
        AdminCall::DenyList { now } => {
            // Unclaimed: no owner has ever written an entry.
            let entries = if ledger.is_claimed(&lm)? {
                active(store, now)?
            } else {
                Vec::new()
            };
            AdminOut::DenyList { entries }
        }
        AdminCall::Drop { actor, name } => {
            owner(ledger, &lm, &actor)?;
            let e = ledger.drop_collection(store, &lm, &name)?;
            AdminOut::Dropped {
                entry: crate::ledger_core::collection_wire(&e),
            }
        }
        AdminCall::Purge { actor, uid } => {
            owner(ledger, &lm, &actor)?;
            let uid = CollectionUid::parse(&uid).map_err(|_| OpError::invalid("malformed uid"))?;
            let e = ledger.purge_collection(store, &lm, uid)?;
            AdminOut::Dropped {
                entry: crate::ledger_core::collection_wire(&e),
            }
        }
    })
}

/// Encoded reply for an admin request body (`None`: not an admin request).
pub fn serve(
    slot: &mut Option<TenantLedger>,
    store: &dyn SqlStore,
    limits: QuotaLimits,
    body: &[u8],
) -> Option<String> {
    let req = serde_json::from_slice::<AdminRequest>(body).ok()?;
    let reply: Reply<AdminOut> = handle(slot, store, limits, req).map_err(|e| WireErr::from_op(&e));
    Some(
        serde_json::to_string(&reply)
            .unwrap_or_else(|_| String::from(r#"{"Err":{"code":"server_error"}}"#)),
    )
}
