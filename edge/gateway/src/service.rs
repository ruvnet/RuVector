//! Operation executors over the Durable Object backend (ADR-351 §7.2,
//! §16.3). REST, MCP and `/v1/ops` all land here, on the same [`Op`] set
//! and the same authorization table (`ruvector_edge_store::ops::authorize`:
//! scope before role, default-deny).
//!
//! Semantics follow the store's in-process dispatcher: every collection is
//! resolved through the caller's own ledger (a foreign or absent name is
//! `404`), shards are addressed by the caller's tenant key, and writes
//! charge the ledger before touching shards and correct it afterwards.

use crate::backend::{ledger, Backend};
use crate::wire::{CollectionWire, LedgerCall, LedgerOut, ShardCall, ShardOut};
use ruvector_edge_auth::{Capability, CapabilitySet};
use ruvector_edge_store::ops::{authorize, OpUsage};
use ruvector_edge_store::{shard_meta_for, CallerContext, ErrorCode, Op, OpError};
use ruvector_edge_tenancy::membership::intersect;
use ruvector_edge_tenancy::{
    shard_for, CollectionUid, DoMeta, QuotaDelta, Role, ShardCount, ShardIndex, VectorId,
};
use serde::Deserialize;
use serde_json::{json, Value as Json};

/// Outcome of an executor.
pub type Exec = Result<(Json, OpUsage), OpError>;

/// The caller's standing in the tenant, read from its ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Access {
    /// Membership role (`None`: not a member).
    pub role: Option<Role>,
    /// Someone has claimed the tenant.
    pub claimed: bool,
}

impl Access {
    /// Effective capabilities: scope ∩ role ceiling (none without a role).
    pub fn effective(&self, scope_caps: CapabilitySet) -> CapabilitySet {
        self.role
            .map_or(CapabilitySet::EMPTY, |r| intersect(scope_caps, r.ceiling()))
    }
}

/// Request-scoped inputs.
pub struct Call<'a, B> {
    /// DO transport.
    pub b: &'a B,
    /// Verified caller.
    pub ctx: &'a CallerContext,
    /// Validate and report without writing.
    pub dry_run: bool,
    /// Unix seconds.
    pub now: u64,
}

/// Read the caller's role and the claim state.
pub async fn access<B: Backend>(b: &B, ctx: &CallerContext) -> Result<Access, OpError> {
    let call = LedgerCall::Access {
        sub: ctx.sub().to_string(),
    };
    match ledger(b, ctx.tenant_key(), call).await? {
        LedgerOut::Access { role, claimed } => Ok(Access {
            role: role.as_deref().and_then(Role::parse),
            claimed,
        }),
        _ => Err(unexpected()),
    }
}

/// Authorize `op` (scope first, then role).
pub async fn authorize_op<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    op: Op,
) -> Result<Access, OpError> {
    let a = access(b, ctx).await?;
    authorize(op, ctx.scope_caps(), a.role, a.claimed)?;
    Ok(a)
}

pub(crate) fn unexpected() -> OpError {
    OpError::new(ErrorCode::ServerError, "unexpected durable object reply")
}

/// Parse typed args from raw JSON text.
pub(crate) fn args<T: for<'de> Deserialize<'de>>(raw: &str) -> Result<T, OpError> {
    serde_json::from_str(raw).map_err(|_| OpError::invalid("malformed args"))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NameArgs {
    collection: String,
}

/// Run an already-authorized op.
pub async fn run<B: Backend>(c: &Call<'_, B>, a: Access, op: Op, raw: &str) -> Exec {
    match op {
        Op::TenantMe => Ok((tenant_me(c.ctx, a), OpUsage::default())),
        Op::CollectionList => collection_list(c, raw).await,
        Op::CollectionCreate => collection_create(c, raw).await,
        Op::UsageGet => usage_get(c, raw).await,
        Op::VectorUpsert => crate::vectors::upsert(c, raw).await,
        Op::VectorQuery => crate::vectors::query(c, raw).await,
        Op::VectorFetch => crate::vectors::fetch(c, raw).await,
        Op::VectorDelete => crate::vectors::delete(c, raw).await,
    }
}

/// Authorize and run.
pub async fn execute<B: Backend>(c: &Call<'_, B>, op: Op, raw: &str) -> Exec {
    let a = authorize_op(c.b, c.ctx, op).await?;
    run(c, a, op, raw).await
}

/// `tenant_me` result (identical to the store dispatcher's).
pub fn tenant_me(ctx: &CallerContext, a: Access) -> Json {
    json!({
        "tenant_key": ctx.tenant_key().as_str(),
        "sub": ctx.sub(),
        "client_id": ctx.client_id(),
        "act_sub": ctx.act_sub(),
        "role": a.role.map(|r| r.as_str()),
        "claimed": a.claimed,
    })
}

/// Capability names, in declaration order.
pub fn capability_names(caps: CapabilitySet) -> Vec<&'static str> {
    caps.iter()
        .map(|c| match c {
            Capability::Read => "read",
            Capability::Write => "write",
            Capability::CreateCollection => "create",
            Capability::Admin => "admin",
            Capability::PublishPublic => "publish",
        })
        .collect()
}

/// Live collection by name through the caller's own ledger (`404` if
/// absent or foreign).
pub(crate) async fn lookup<B: Backend>(
    c: &Call<'_, B>,
    name: &str,
) -> Result<CollectionWire, OpError> {
    let call = LedgerCall::Collection {
        name: name.to_string(),
    };
    match ledger(c.b, c.ctx.tenant_key(), call).await? {
        LedgerOut::Collection { entry } => entry.ok_or(OpError::not_found()),
        _ => Err(unexpected()),
    }
}

/// Shard identity for `(caller tenant, collection, index)`.
pub(crate) fn shard_meta(
    c: &CallerContext,
    e: &CollectionWire,
    i: ShardIndex,
) -> Result<DoMeta, OpError> {
    let uid = CollectionUid::parse(&e.uid).map_err(|_| unexpected())?;
    shard_meta_for(c.tenant_key(), uid, i)
}

/// The collection's shard count.
pub(crate) fn count_of(e: &CollectionWire) -> Result<ShardCount, OpError> {
    ShardCount::new(e.shard_count).map_err(|_| unexpected())
}

/// Stable routing of a vector id.
pub(crate) fn route(e: &CollectionWire, id: &str) -> Result<ShardIndex, OpError> {
    Ok(shard_for(&VectorId::parse(id)?, count_of(e)?))
}

/// Admit caller-driven growth (limits enforced).
pub(crate) async fn charge<B: Backend>(
    c: &Call<'_, B>,
    delta: QuotaDelta,
    wu: u64,
) -> Result<(), OpError> {
    if delta == QuotaDelta::default() && wu == 0 {
        return Ok(());
    }
    let call = LedgerCall::Admit {
        delta,
        work_units: wu,
        now: c.now,
    };
    ledger(c.b, c.ctx.tenant_key(), call).await.map(|_| ())
}

/// Internal correction (no limits, clamped).
pub(crate) async fn correct<B: Backend>(c: &Call<'_, B>, delta: QuotaDelta) -> Result<(), OpError> {
    if delta == QuotaDelta::default() {
        return Ok(());
    }
    let call = LedgerCall::Adjust { delta, now: c.now };
    ledger(c.b, c.ctx.tenant_key(), call).await.map(|_| ())
}

pub(crate) fn one_op() -> QuotaDelta {
    QuotaDelta {
        ops: 1,
        ..Default::default()
    }
}

async fn collection_list<B: Backend>(c: &Call<'_, B>, raw: &str) -> Exec {
    args::<NoArgs>(raw)?;
    let list: Vec<Json> = match ledger(c.b, c.ctx.tenant_key(), LedgerCall::Collections).await? {
        LedgerOut::Collections { entries } => entries.into_iter().map(|e| e.view).collect(),
        _ => return Err(unexpected()),
    };
    let n = list.len() as u64;
    charge(c, one_op(), 1).await?;
    let usage = OpUsage {
        work_units: 1,
        rows: n,
        bytes: 0,
    };
    Ok((json!({ "collections": list }), usage))
}

async fn collection_create<B: Backend>(c: &Call<'_, B>, raw: &str) -> Exec {
    let tenant = c.ctx.tenant_key();
    let spec = raw.to_string();
    let validated = ledger(
        c.b,
        tenant,
        LedgerCall::ValidateCreate { spec: spec.clone() },
    )
    .await?;
    let LedgerOut::Validated { name, shards } = validated else {
        return Err(unexpected());
    };
    if c.dry_run {
        let r = json!({ "dry_run": true, "name": name, "shards": shards });
        return Ok((r, OpUsage::default()));
    }
    // Charge the op before the write, so a daily-ops 413 never follows a
    // committed create.
    charge(c, one_op(), 1).await?;
    let call = LedgerCall::CreateCollection {
        spec,
        sub: c.ctx.sub().to_string(),
        now: c.now,
    };
    match ledger(c.b, tenant, call).await? {
        LedgerOut::Collections { mut entries } if entries.len() == 1 => {
            let e = entries.remove(0);
            let usage = OpUsage {
                work_units: 1,
                rows: 1,
                bytes: 0,
            };
            Ok((e.view, usage))
        }
        _ => Err(unexpected()),
    }
}

async fn usage_get<B: Backend>(c: &Call<'_, B>, raw: &str) -> Exec {
    args::<NoArgs>(raw)?;
    charge(c, one_op(), 1).await?;
    match ledger(c.b, c.ctx.tenant_key(), LedgerCall::Usage { now: c.now }).await? {
        LedgerOut::Usage { report } => Ok((
            report,
            OpUsage {
                work_units: 1,
                rows: 0,
                bytes: 0,
            },
        )),
        _ => Err(unexpected()),
    }
}

/// `GET /v1/collections/{c}` / MCP `collection_get` (read + viewer, like
/// `collection_list`): catalog view plus live shard counters.
pub async fn collection_get<B: Backend>(c: &Call<'_, B>, raw: &str) -> Exec {
    authorize_op(c.b, c.ctx, Op::CollectionList).await?;
    let NameArgs { collection } = args(raw)?;
    let e = lookup(c, &collection).await?;
    // Admitted before any shard is loaded (§10 layer 4).
    charge(c, one_op(), 1).await?;
    let (mut count, mut resident, mut snapshot_seq) = (0u64, 0u64, 0u64);
    for i in count_of(&e)?.indices() {
        let dm = shard_meta(c.ctx, &e, i)?;
        match crate::backend::shard(c.b, &dm, ShardCall::Stats).await {
            Ok(ShardOut::Stats {
                count: n,
                resident_bytes,
                snapshot_seq: s,
                ..
            }) => {
                count += n;
                resident += resident_bytes;
                snapshot_seq = snapshot_seq.max(s);
            }
            Ok(_) => return Err(unexpected()),
            Err(e) => return Err(e.into_op()),
        }
    }
    let mut view = e.view;
    view["count"] = json!(count);
    view["resident_bytes"] = json!(resident);
    view["snapshot_seq"] = json!(snapshot_seq);
    let usage = OpUsage {
        work_units: 1,
        rows: 1,
        bytes: 0,
    };
    Ok((view, usage))
}

/// `POST /v1/claim` (ADR §4.2): `ruvector:write` or `ruvector:admin`; the
/// first claimant of an unclaimed tenant becomes owner, then `409`.
pub async fn claim<B: Backend>(c: &Call<'_, B>) -> Exec {
    let caps = c.ctx.scope_caps();
    if !caps.contains(Capability::Write) && !caps.contains(Capability::Admin) {
        return Err(OpError {
            code: ErrorCode::InsufficientScope,
            detail: "insufficient scope",
            scope: Some(Capability::Write.satisfying_scope()),
        });
    }
    let call = LedgerCall::Claim {
        sub: c.ctx.sub().to_string(),
        now: c.now,
    };
    match ledger(c.b, c.ctx.tenant_key(), call).await? {
        LedgerOut::Role { role } => Ok((json!({ "role": role }), OpUsage::default())),
        _ => Err(unexpected()),
    }
}
