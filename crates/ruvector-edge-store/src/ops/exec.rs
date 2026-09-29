//! Op executors shared by every op, plus the collection and usage ops (the
//! vector ops are in [`super::vector`]). Each resolves the collection
//! through the caller's own ledger (so a foreign or absent name is `404`),
//! addresses shards by the caller's tenant key, and asserts shard identity
//! on every call.

use super::cluster::LocalCluster;
use super::types::OpUsage;
use crate::context::{shard_meta_for, CallerContext};
use crate::error::OpError;
use crate::ledger::{CatalogEntry, CreateCollection};
use crate::ports::{EntropySource, SqlStore};
use crate::shard::{Actor, UsageDelta};
use ruvector_edge_tenancy::{shard_for, DoMeta, LedgerMeta, QuotaDelta, ShardIndex, VectorId};
use serde::Deserialize;
use serde_json::{json, Value as Json};

/// Outcome of an executor.
pub(crate) type Exec = Result<(Json, OpUsage), OpError>;

/// Request-scoped inputs shared by every executor.
pub(crate) struct Call<'a> {
    pub ctx: &'a CallerContext,
    pub lm: &'a LedgerMeta,
    pub dry_run: bool,
    pub now: u64,
}

impl Call<'_> {
    pub(crate) fn actor(&self) -> Actor<'_> {
        Actor {
            sub: self.ctx.sub(),
            jti: self.ctx.jti(),
            family_id: self.ctx.family_id(),
        }
    }
}

/// Parse typed args straight from the raw `args` JSON text (no
/// intermediate `Value` tree).
pub(crate) fn args<T: for<'de> Deserialize<'de>>(raw: &str) -> Result<T, OpError> {
    serde_json::from_str(raw).map_err(|_| OpError::invalid("malformed args"))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

pub(crate) fn lookup<S: SqlStore + Default>(
    cl: &mut LocalCluster<S>,
    c: &Call<'_>,
    name: &str,
) -> Result<CatalogEntry, OpError> {
    let (_, ledger) = cl.ledger(c.ctx.tenant_key())?;
    ledger
        .collection(c.lm, name)?
        .cloned()
        .ok_or(OpError::not_found())
}

pub(crate) fn shard_meta(c: &Call<'_>, e: &CatalogEntry, i: ShardIndex) -> Result<DoMeta, OpError> {
    shard_meta_for(c.ctx.tenant_key(), e.uid, i)
}

pub(crate) fn route(e: &CatalogEntry, id: &str) -> Result<ShardIndex, OpError> {
    Ok(shard_for(&VectorId::parse(id)?, e.shard_count))
}

/// Admit caller-driven usage growth (limits enforced).
pub(crate) fn charge<S: SqlStore + Default>(
    cl: &mut LocalCluster<S>,
    c: &Call<'_>,
    d: QuotaDelta,
    wu: u64,
) -> Result<(), OpError> {
    let (store, ledger) = cl.ledger(c.ctx.tenant_key())?;
    ledger.admit(store, c.lm, d, wu, c.now).map(|_| ())
}

/// Apply an internal correction (refund / reconcile / post-commit release):
/// no limit checks, releases clamp at zero, so it cannot fail on quota.
pub(crate) fn correct<S: SqlStore + Default>(
    cl: &mut LocalCluster<S>,
    c: &Call<'_>,
    d: UsageDelta,
) -> Result<(), OpError> {
    if d == UsageDelta::default() {
        return Ok(());
    }
    let (store, ledger) = cl.ledger(c.ctx.tenant_key())?;
    ledger.adjust(store, c.lm, release_of(d), c.now).map(|_| ())
}

pub(crate) fn release_of(d: UsageDelta) -> QuotaDelta {
    QuotaDelta {
        collections: 0,
        vectors: d.vectors,
        float_budget: d.floats,
        bytes: d.bytes,
        ops: 0,
    }
}

pub(crate) fn add(a: &mut UsageDelta, b: UsageDelta) {
    a.vectors = a.vectors.saturating_add(b.vectors);
    a.floats = a.floats.saturating_add(b.floats);
    a.bytes = a.bytes.saturating_add(b.bytes);
}

pub(crate) fn neg(d: UsageDelta) -> UsageDelta {
    UsageDelta {
        vectors: d.vectors.saturating_neg(),
        floats: d.floats.saturating_neg(),
        bytes: d.bytes.saturating_neg(),
    }
}

pub(crate) fn collection_list<S: SqlStore + Default>(
    cl: &mut LocalCluster<S>,
    c: &Call<'_>,
    a: &str,
) -> Exec {
    args::<NoArgs>(a)?;
    let (_, ledger) = cl.ledger(c.ctx.tenant_key())?;
    let list: Vec<Json> = ledger
        .collections(c.lm)?
        .into_iter()
        .map(CatalogEntry::to_json)
        .collect();
    let n = list.len() as u64;
    charge(
        cl,
        c,
        QuotaDelta {
            ops: 1,
            ..Default::default()
        },
        1,
    )?;
    Ok((
        json!({ "collections": list }),
        OpUsage {
            work_units: 1,
            rows: n,
            bytes: 0,
        },
    ))
}

pub(crate) fn collection_create<S: SqlStore + Default>(
    cl: &mut LocalCluster<S>,
    c: &Call<'_>,
    a: &str,
    entropy: &dyn EntropySource,
) -> Exec {
    let spec: CreateCollection = args(a)?;
    let (_, ledger) = cl.ledger(c.ctx.tenant_key())?;
    let (name, shards, _) = ledger.validate_create(c.lm, &spec)?;
    if c.dry_run {
        let r = json!({ "dry_run": true, "name": name, "shards": shards.get() });
        return Ok((r, OpUsage::default()));
    }
    // Charge the op before the write, like every other mutating op, so a
    // daily-ops 413 never follows a committed create.
    charge(
        cl,
        c,
        QuotaDelta {
            ops: 1,
            ..Default::default()
        },
        1,
    )?;
    let (store, ledger) = cl.ledger(c.ctx.tenant_key())?;
    let entry = ledger.create_collection(store, c.lm, &spec, c.ctx.sub(), entropy, c.now)?;
    Ok((
        entry.to_json(),
        OpUsage {
            work_units: 1,
            rows: 1,
            bytes: 0,
        },
    ))
}

pub(crate) fn usage_get<S: SqlStore + Default>(
    cl: &mut LocalCluster<S>,
    c: &Call<'_>,
    a: &str,
) -> Exec {
    args::<NoArgs>(a)?;
    charge(
        cl,
        c,
        QuotaDelta {
            ops: 1,
            ..Default::default()
        },
        1,
    )?;
    let (_, ledger) = cl.ledger(c.ctx.tenant_key())?;
    let r = json!({
        "usage": ledger.usage(c.lm, c.now)?,
        "limits": ledger.limits(),
        "work_units": ledger.work_units(),
    });
    Ok((
        r,
        OpUsage {
            work_units: 1,
            rows: 0,
            bytes: 0,
        },
    ))
}
