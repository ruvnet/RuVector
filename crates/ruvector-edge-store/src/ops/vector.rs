//! Vector op executors: upsert, query, fetch, delete.
//!
//! Writes charge the ledger before touching shards and correct it after:
//! a shard that failed before issuing any statement is refunded in full; a
//! shard poisoned mid-write is reopened (write-through replay makes its
//! durable state whole) and the ledger is reconciled to what the shard
//! really holds; if it cannot be reopened, its planned delta stays charged
//! (a conservative over-count, never an under-count).

use super::cluster::LocalCluster;
use super::exec::{add, args, charge, correct, lookup, neg, route, shard_meta, Call, Exec};
use super::types::OpUsage;
use crate::error::{ErrorCode, OpError};
use crate::ports::SqlStore;
use crate::shard::{
    validate_query, Match, QueryRequest, UpsertRow, UsageDelta, MAX_DELETE_IDS, MAX_FETCH_IDS,
    MAX_QUERY_STEPS,
};
use ruvector_edge_tenancy::quota::limits::MAX_UPSERT_BATCH;
use ruvector_edge_tenancy::{DoName, QuotaDelta, ShardIndex};
use serde::Deserialize;
use serde_json::{json, Value as Json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpsertArgs {
    collection: String,
    vectors: Vec<UpsertRow>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryArgs {
    collection: String,
    vector: Vec<f32>,
    top_k: u32,
    #[serde(default)]
    filter: Option<Json>,
    #[serde(default)]
    include: Vec<String>,
    #[serde(default)]
    ef: Option<u32>,
    #[serde(default)]
    rerank: Option<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IdsArgs {
    collection: String,
    ids: Vec<String>,
    #[serde(default)]
    include_values: bool,
}

fn sub(a: UsageDelta, b: UsageDelta) -> UsageDelta {
    let mut out = a;
    add(&mut out, neg(b));
    out
}

/// What a failed shard write really changed: `None` when unknowable (the
/// shard could not be reopened). A shard that was not poisoned issued no
/// statement, so it changed nothing.
fn applied_after_failure<S: SqlStore + Default>(
    cl: &mut LocalCluster<S>,
    name: &DoName,
    before: UsageDelta,
) -> Option<UsageDelta> {
    if !cl.shard_poisoned(name) {
        return Some(UsageDelta::default());
    }
    cl.shard(name)
        .ok()
        .map(|(_, s)| sub(s.usage_totals(), before))
}

fn totals<S: SqlStore + Default>(
    cl: &mut LocalCluster<S>,
    name: &DoName,
) -> Result<UsageDelta, OpError> {
    cl.shard(name).map(|(_, s)| s.usage_totals())
}

pub(crate) fn vector_upsert<S: SqlStore + Default>(
    cl: &mut LocalCluster<S>,
    c: &Call<'_>,
    a: &str,
) -> Exec {
    let UpsertArgs {
        collection,
        vectors,
    } = args(a)?;
    if vectors.len() > MAX_UPSERT_BATCH as usize {
        return Err(OpError::new(
            ErrorCode::PayloadTooLarge,
            "upsert batch too large",
        ));
    }
    let n = vectors.len() as u64;
    let e = lookup(cl, c, &collection)?;
    let cfg = e.shard_config();
    let mut groups: BTreeMap<ShardIndex, Vec<UpsertRow>> = BTreeMap::new();
    for r in vectors {
        groups.entry(route(&e, &r.id)?).or_default().push(r);
    }
    if groups.is_empty() {
        return Err(OpError::invalid("empty upsert"));
    }
    let mut plans = Vec::new();
    let mut total = UsageDelta::default();
    for (i, rows) in groups {
        let dm = shard_meta(c, &e, i)?;
        let (store, shard) = cl.shard(&dm.do_name())?;
        shard.load_index(store)?;
        let plan = shard.plan_upsert(&dm, &cfg, rows)?;
        add(&mut total, plan.delta);
        plans.push((dm, plan));
    }
    let bytes = total.bytes.unsigned_abs();
    if c.dry_run {
        let r = json!({ "upserted": n, "dry_run": true, "delta": total });
        let usage = OpUsage {
            work_units: 0,
            rows: n,
            bytes,
        };
        return Ok((r, usage));
    }
    let wu = 1 + n;
    let grow = QuotaDelta {
        ops: 1,
        ..super::exec::release_of(total)
    };
    charge(cl, c, grow, wu)?;
    let mut write_seq = 0;
    // Charged but not (yet) applied.
    let mut pending = total;
    for (dm, plan) in plans {
        let name = dm.do_name();
        let planned = plan.delta;
        let res = match totals(cl, &name) {
            Ok(before) => cl
                .shard(&name)
                .and_then(|(store, shard)| shard.apply_upsert(store, plan, c.actor(), c.now))
                .map_err(|e| (e, before)),
            Err(e) => Err((e, UsageDelta::default())),
        };
        match res {
            Ok(o) => {
                write_seq = write_seq.max(o.write_seq);
                pending = sub(pending, planned);
                cl.touch(&name);
            }
            Err((err, before)) => {
                // Unknown (shard not reopenable) ⇒ keep this shard's
                // planned delta charged.
                let applied = applied_after_failure(cl, &name, before).unwrap_or(planned);
                pending = sub(pending, applied);
                // Refund what was charged and not applied. A refund that
                // fails poisons the ledger (reopened from storage next
                // time); usage then stays over-counted, never under.
                let _ledger_poisoned = correct(cl, c, neg(pending));
                return Err(err);
            }
        }
    }
    let r = json!({ "upserted": n, "write_seq": write_seq, "dry_run": false });
    let usage = OpUsage {
        work_units: wu,
        rows: n,
        bytes,
    };
    Ok((r, usage))
}

pub(crate) fn vector_query<S: SqlStore + Default>(
    cl: &mut LocalCluster<S>,
    c: &Call<'_>,
    a: &str,
) -> Exec {
    let q: QueryArgs = args(a)?;
    let e = lookup(cl, c, &q.collection)?;
    let cfg = e.shard_config();
    let req = QueryRequest {
        vector: q.vector,
        top_k: q.top_k,
        filter: q.filter,
        include: q.include,
        ef: q.ef,
        rerank: q.rerank,
    };
    // Every request-only check (includes, dimension, filter) runs before
    // any shard is loaded.
    let v = validate_query(&req, &cfg)?;
    let mut all: Vec<Match> = Vec::new();
    let (mut scanned, mut steps) = (0u64, 0u64);
    for i in e.shard_count.indices() {
        let dm = shard_meta(c, &e, i)?;
        let name = dm.do_name();
        let (store, shard) = cl.shard(&name)?;
        steps = steps.saturating_add(shard.query_steps(&v));
        if steps > MAX_QUERY_STEPS {
            return Err(OpError::new(ErrorCode::BudgetExceeded, "query step budget"));
        }
        let out = shard.query(store, &dm, &cfg, &req)?;
        scanned += out.scanned;
        all.extend(out.matches);
        cl.touch(&name);
    }
    all.sort_by(Match::rank_cmp);
    all.truncate(req.top_k as usize);
    let wu = 1 + steps / 1024;
    let op = QuotaDelta {
        ops: 1,
        ..Default::default()
    };
    charge(cl, c, op, wu)?;
    let r = json!({ "matches": all, "shards_queried": e.shard_count.get() });
    let usage = OpUsage {
        work_units: wu,
        rows: scanned,
        bytes: 0,
    };
    Ok((r, usage))
}

pub(crate) fn vector_fetch<S: SqlStore + Default>(
    cl: &mut LocalCluster<S>,
    c: &Call<'_>,
    a: &str,
) -> Exec {
    let IdsArgs {
        collection,
        ids,
        include_values,
    } = args(a)?;
    if ids.is_empty() {
        return Err(OpError::invalid("no ids"));
    }
    if ids.len() > MAX_FETCH_IDS {
        return Err(OpError::new(ErrorCode::PayloadTooLarge, "too many ids"));
    }
    let mut seen = BTreeSet::new();
    let ids: Vec<String> = ids.into_iter().filter(|i| seen.insert(i.clone())).collect();
    let e = lookup(cl, c, &collection)?;
    let mut groups: BTreeMap<ShardIndex, Vec<String>> = BTreeMap::new();
    for id in &ids {
        groups.entry(route(&e, id)?).or_default().push(id.clone());
    }
    let mut found: BTreeMap<String, Match> = BTreeMap::new();
    for (i, part) in groups {
        let dm = shard_meta(c, &e, i)?;
        let name = dm.do_name();
        let (store, shard) = cl.shard(&name)?;
        let got = shard.fetch(store, &dm, &part, include_values)?;
        found.extend(got.into_iter().map(|m| (m.id.clone(), m)));
        cl.touch(&name);
    }
    let vectors: Vec<Json> = ids
        .iter()
        .filter_map(|id| found.get(id))
        .map(|m| json!({ "id": m.id, "metadata": m.metadata, "values": m.values }))
        .collect();
    let n = vectors.len() as u64;
    let op = QuotaDelta {
        ops: 1,
        ..Default::default()
    };
    charge(cl, c, op, 1)?;
    let usage = OpUsage {
        work_units: 1,
        rows: n,
        bytes: 0,
    };
    Ok((json!({ "vectors": vectors }), usage))
}

pub(crate) fn vector_delete<S: SqlStore + Default>(
    cl: &mut LocalCluster<S>,
    c: &Call<'_>,
    a: &str,
) -> Exec {
    let IdsArgs {
        collection,
        ids,
        include_values,
    } = args(a)?;
    if include_values {
        return Err(OpError::invalid("include_values not valid for delete"));
    }
    if ids.is_empty() {
        return Err(OpError::invalid("no ids"));
    }
    if ids.len() > MAX_DELETE_IDS {
        return Err(OpError::new(ErrorCode::PayloadTooLarge, "too many ids"));
    }
    let e = lookup(cl, c, &collection)?;
    let mut groups: BTreeMap<ShardIndex, Vec<String>> = BTreeMap::new();
    for id in &ids {
        groups.entry(route(&e, id)?).or_default().push(id.clone());
    }
    let wu = 1 + ids.len() as u64 / 100;
    if !c.dry_run {
        let op = QuotaDelta {
            ops: 1,
            ..Default::default()
        };
        charge(cl, c, op, wu)?;
    }
    let (mut deleted, mut write_seq, mut total) = (0u64, 0u64, UsageDelta::default());
    for (i, part) in groups {
        let dm = shard_meta(c, &e, i)?;
        let name = dm.do_name();
        let res = match totals(cl, &name) {
            Ok(before) => cl
                .shard(&name)
                .and_then(|(store, shard)| {
                    shard.delete(store, &dm, &part, c.actor(), c.dry_run, c.now)
                })
                .map_err(|e| (e, before)),
            Err(e) => Err((e, UsageDelta::default())),
        };
        let o = match res {
            Ok(o) => o,
            Err((err, before)) => {
                if !c.dry_run {
                    // Release whatever the torn delete really removed
                    // (unknown ⇒ nothing: over-count, never under), plus
                    // the shards already deleted from.
                    let applied = applied_after_failure(cl, &name, before);
                    add(&mut total, applied.unwrap_or_default());
                    let _ledger_poisoned = correct(cl, c, total);
                }
                return Err(err);
            }
        };
        deleted += o.deleted;
        write_seq = write_seq.max(o.write_seq);
        add(&mut total, o.delta);
        cl.touch(&name);
    }
    if !c.dry_run {
        // Post-commit release: limit-free and clamped, so it cannot turn a
        // committed delete into a quota or underflow error.
        correct(cl, c, total)?;
    }
    let r =
        json!({ "deleted": deleted, "write_seq": write_seq, "dry_run": c.dry_run, "delta": total });
    let usage = OpUsage {
        work_units: if c.dry_run { 0 } else { wu },
        rows: deleted,
        bytes: total.bytes.unsigned_abs(),
    };
    Ok((r, usage))
}
