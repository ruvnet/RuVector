//! Vector executors (upsert, query, fetch, delete) fanning out to
//! `VectorShard` DOs, with the gateway merging top-k (ADR-351 §4.4).
//!
//! Every executor admits its op (`ops: 1` + base work units) through the
//! ledger **before** the first shard call, so an over-quota tenant never
//! loads a shard and failed or dry-run attempts still count; extra work
//! (scan steps, rows) is admitted after the fan-out.
//!
//! Upsert is two-phase per shard: `Plan` (validate, report the delta) →
//! ledger `Admit` of the growth → `Apply`, where the shard re-plans
//! atomically and refuses (`409`) if the fresh delta grows beyond what was
//! admitted, so a write never exceeds its charge. What was charged and not
//! applied is refunded; a shard write whose effect is unknown (transport
//! failure) stays charged — a conservative over-count, never an under-count.

use crate::backend::{Backend, ShardErr};
use crate::quant_route::shard_for;
use crate::service::{
    args, charge, correct, count_of, lookup, one_op, route, shard_meta, unexpected, Call, Exec,
};
use crate::wire::{ActorWire, DeltaWire, MatchWire, ShardCall, ShardOut};
use ruvector_edge_store::ops::OpUsage;
use ruvector_edge_store::shard::{validate_query, UpsertRow, MAX_DELETE_IDS, MAX_FETCH_IDS};
use ruvector_edge_store::{ErrorCode, OpError, QueryRequest};
use ruvector_edge_tenancy::quota::limits::MAX_UPSERT_BATCH;
use ruvector_edge_tenancy::{QuotaDelta, ShardIndex};
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
    /// HNSW beam width (validated and capped by the shard).
    #[serde(default)]
    ef: Option<u32>,
    /// Candidates reranked exactly from SQLite.
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

fn actor<B>(c: &Call<'_, B>) -> ActorWire {
    ActorWire {
        sub: c.ctx.sub().to_string(),
        jti: c.ctx.jti().to_string(),
        family_id: c.ctx.family_id().to_string(),
        act_sub: c.ctx.act_sub().map(str::to_string),
    }
}

pub(crate) async fn upsert<B: Backend>(c: &Call<'_, B>, raw: &str) -> Exec {
    let UpsertArgs {
        collection,
        vectors,
    } = args(raw)?;
    if vectors.len() > MAX_UPSERT_BATCH as usize {
        return Err(OpError::new(
            ErrorCode::PayloadTooLarge,
            "upsert batch too large",
        ));
    }
    let n = vectors.len() as u64;
    let e = lookup(c, &collection).await?;
    let mut groups: BTreeMap<ShardIndex, Vec<UpsertRow>> = BTreeMap::new();
    for r in vectors {
        groups.entry(route(&e, &r.id)?).or_default().push(r);
    }
    if groups.is_empty() {
        return Err(OpError::invalid("empty upsert"));
    }
    // Admit the op before any shard is loaded: an over-quota tenant is
    // refused here, and a failed or dry-run plan still counts (§10 layer 4).
    charge(c, one_op(), 1).await?;
    let mut plans = Vec::with_capacity(groups.len());
    let mut total = DeltaWire::default();
    for (i, rows) in groups {
        let dm = shard_meta(c.ctx, &e, i)?;
        let call = ShardCall::Plan {
            cfg: e.cfg.clone(),
            rows: rows.clone(),
        };
        let delta = match shard_for(c.b, &e, &dm, call)
            .await
            .map_err(ShardErr::into_op)?
        {
            ShardOut::Planned { delta } => delta,
            _ => return Err(unexpected()),
        };
        total = total.plus(delta);
        plans.push((dm, rows, delta));
    }
    let bytes = total.bytes.unsigned_abs();
    if c.dry_run {
        let r = json!({ "upserted": n, "dry_run": true, "delta": total.to_json() });
        return Ok((
            r,
            OpUsage {
                work_units: 1,
                rows: n,
                bytes,
            },
        ));
    }
    let wu = 1 + n;
    // The growth itself (the op was admitted above).
    charge(c, total.quota(), n).await?;
    let mut write_seq = 0;
    // Charged but not applied (released at the end or on failure).
    let mut unapplied = DeltaWire::default();
    for (idx, (dm, rows, planned)) in plans.iter().enumerate() {
        let call = ShardCall::Apply {
            cfg: e.cfg.clone(),
            rows: rows.clone(),
            admitted: *planned,
            actor: actor(c),
            now: c.now,
        };
        let applied = match shard_for(c.b, &e, dm, call).await {
            Ok(ShardOut::Written {
                write_seq: s,
                delta,
                ..
            }) => {
                write_seq = write_seq.max(s);
                unapplied = unapplied.plus(planned.minus(delta));
                continue;
            }
            Ok(ShardOut::Failed { err, applied }) => (err.into_op(), applied.unwrap_or(*planned)),
            Ok(_) => (unexpected(), *planned),
            Err(ShardErr::Refused(err)) => (err, DeltaWire::default()),
            Err(ShardErr::Unknown(err)) => (err, *planned),
        };
        let (err, done) = applied;
        unapplied = unapplied.plus(planned.minus(done));
        for (_, _, later) in &plans[idx + 1..] {
            unapplied = unapplied.plus(*later);
        }
        // A refund that fails leaves usage over-counted, never under.
        let _refund = correct(c, unapplied.neg().quota()).await;
        return Err(err);
    }
    // Post-commit release of what the shards did not need: a failure
    // leaves usage over-counted, never fails a committed write.
    let _released = correct(c, unapplied.neg().quota()).await;
    let r = json!({ "upserted": n, "write_seq": write_seq, "dry_run": false });
    Ok((
        r,
        OpUsage {
            work_units: wu,
            rows: n,
            bytes,
        },
    ))
}

pub(crate) async fn query<B: Backend>(c: &Call<'_, B>, raw: &str) -> Exec {
    let q: QueryArgs = args(raw)?;
    let e = lookup(c, &q.collection).await?;
    let req = QueryRequest {
        vector: q.vector,
        top_k: q.top_k,
        filter: q.filter,
        include: q.include,
        ef: q.ef,
        rerank: q.rerank,
    };
    // Every request-only check runs before any shard is loaded.
    validate_query(&req, &e.cfg.to_config())?;
    // §10 layer 2: a query costs one read token per shard (the request
    // itself paid the first).
    let fanout = count_of(&e)?.get();
    c.b.charge_fanout(c.ctx, fanout.saturating_sub(1)).await?;
    // Admitted before the fan-out; the scan's extra work units after it.
    charge(c, one_op(), 1).await?;
    let mut all: Vec<MatchWire> = Vec::new();
    let (mut scanned, mut steps) = (0u64, 0u64);
    for i in count_of(&e)?.indices() {
        let dm = shard_meta(c.ctx, &e, i)?;
        let call = ShardCall::Query {
            cfg: e.cfg.clone(),
            req: req.clone(),
            steps_before: steps,
        };
        let out = match shard_for(c.b, &e, &dm, call)
            .await
            .map_err(ShardErr::into_op)
        {
            Ok(out) => out,
            Err(err) => {
                // Scans already done (e.g. up to `budget_exceeded`) count.
                let _charged = charge(c, QuotaDelta::default(), steps / 1024).await;
                return Err(err);
            }
        };
        match out {
            ShardOut::Matches {
                matches,
                scanned: s,
                steps: total,
            } => {
                scanned += s;
                steps = total;
                all.extend(matches);
            }
            _ => return Err(unexpected()),
        }
    }
    all.sort_by(MatchWire::rank_cmp);
    all.truncate(req.top_k as usize);
    let wu = 1 + steps / 1024;
    charge(c, QuotaDelta::default(), wu - 1).await?;
    let matches: Vec<Json> = all.iter().map(MatchWire::to_public).collect();
    let r = json!({ "matches": matches, "shards_queried": e.shard_count });
    Ok((
        r,
        OpUsage {
            work_units: wu,
            rows: scanned,
            bytes: 0,
        },
    ))
}

pub(crate) async fn fetch<B: Backend>(c: &Call<'_, B>, raw: &str) -> Exec {
    let IdsArgs {
        collection,
        ids,
        include_values,
    } = args(raw)?;
    if ids.is_empty() {
        return Err(OpError::invalid("no ids"));
    }
    if ids.len() > MAX_FETCH_IDS {
        return Err(OpError::new(ErrorCode::PayloadTooLarge, "too many ids"));
    }
    let mut seen = BTreeSet::new();
    let ids: Vec<String> = ids.into_iter().filter(|i| seen.insert(i.clone())).collect();
    let e = lookup(c, &collection).await?;
    let mut groups: BTreeMap<ShardIndex, Vec<String>> = BTreeMap::new();
    for id in &ids {
        groups.entry(route(&e, id)?).or_default().push(id.clone());
    }
    charge(c, one_op(), 1).await?;
    let mut found: BTreeMap<String, MatchWire> = BTreeMap::new();
    for (i, part) in groups {
        let dm = shard_meta(c.ctx, &e, i)?;
        let call = ShardCall::Fetch {
            ids: part,
            include_values,
        };
        match shard_for(c.b, &e, &dm, call)
            .await
            .map_err(ShardErr::into_op)?
        {
            ShardOut::Fetched { matches } => {
                found.extend(matches.into_iter().map(|m| (m.id.clone(), m)));
            }
            _ => return Err(unexpected()),
        }
    }
    let vectors: Vec<Json> = ids
        .iter()
        .filter_map(|id| found.get(id))
        .map(|m| json!({ "id": m.id, "metadata": m.metadata, "values": m.values }))
        .collect();
    let n = vectors.len() as u64;
    Ok((
        json!({ "vectors": vectors }),
        OpUsage {
            work_units: 1,
            rows: n,
            bytes: 0,
        },
    ))
}

pub(crate) async fn delete<B: Backend>(c: &Call<'_, B>, raw: &str) -> Exec {
    let IdsArgs {
        collection,
        ids,
        include_values,
    } = args(raw)?;
    if include_values {
        return Err(OpError::invalid("include_values not valid for delete"));
    }
    if ids.is_empty() {
        return Err(OpError::invalid("no ids"));
    }
    if ids.len() > MAX_DELETE_IDS {
        return Err(OpError::new(ErrorCode::PayloadTooLarge, "too many ids"));
    }
    let e = lookup(c, &collection).await?;
    let mut groups: BTreeMap<ShardIndex, Vec<String>> = BTreeMap::new();
    for id in &ids {
        groups.entry(route(&e, id)?).or_default().push(id.clone());
    }
    let wu = if c.dry_run {
        1
    } else {
        1 + ids.len() as u64 / 100
    };
    // Admitted before any shard is loaded, dry runs included.
    charge(c, one_op(), wu).await?;
    let (mut deleted, mut write_seq, mut total) = (0u64, 0u64, DeltaWire::default());
    for (i, part) in groups {
        let dm = shard_meta(c.ctx, &e, i)?;
        let call = ShardCall::Delete {
            ids: part,
            actor: actor(c),
            dry_run: c.dry_run,
            now: c.now,
        };
        let (err, applied) = match shard_for(c.b, &e, &dm, call).await {
            Ok(ShardOut::Written {
                count,
                write_seq: s,
                delta,
            }) => {
                deleted += count;
                write_seq = write_seq.max(s);
                total = total.plus(delta);
                continue;
            }
            // Unknown effect ⇒ release nothing for this shard (over-count).
            Ok(ShardOut::Failed { err, applied }) => (err.into_op(), applied.unwrap_or_default()),
            Ok(_) => (unexpected(), DeltaWire::default()),
            Err(e) => (e.into_op(), DeltaWire::default()),
        };
        if !c.dry_run {
            let _ledger = correct(c, total.plus(applied).quota()).await;
        }
        return Err(err);
    }
    if !c.dry_run {
        // Post-commit release: limit-free and clamped.
        correct(c, total.quota()).await?;
    }
    let r = json!({
        "deleted": deleted,
        "write_seq": write_seq,
        "dry_run": c.dry_run,
        "delta": total.to_json(),
    });
    let usage = OpUsage {
        work_units: wu,
        rows: deleted,
        bytes: total.bytes.unsigned_abs(),
    };
    Ok((r, usage))
}
