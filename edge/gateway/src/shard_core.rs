//! `VectorShard` Durable Object core: decode a [`ShardRequest`], run it
//! against the resident [`VectorShard`] over the DO's [`SqlStore`], encode
//! the reply. Pure and synchronous (one call = one coalesced commit).
//!
//! Resident shards live in a [`ShardHost`]: in the Worker, one per isolate
//! (every `VectorShard` instance in the isolate shares it), so the
//! isolate-wide [`ResidentRegistry`] can evict cold shards LRU. A host entry
//! is keyed by the **DO's own id**, never by request input, and each entry
//! is only ever opened over that DO's own storage.

use crate::wire::{DeltaWire, MatchWire, Reply, ShardCall, ShardOut, ShardRequest, WireErr};
use ruvector_edge_store::shard::{validate_query, Actor, Match, MAX_FETCH_IDS, MAX_QUERY_STEPS};
use ruvector_edge_store::{
    shard_meta_for, ErrorCode, OpError, ResidentRegistry, SqlStore, VectorShard,
};
use ruvector_edge_tenancy::{CollectionUid, DoMeta, ShardIndex, TenantKey};
use std::collections::BTreeMap;

/// Resident shard states of one isolate plus the eviction registry.
#[derive(Debug, Default)]
pub struct ShardHost {
    registry: ResidentRegistry,
    shards: BTreeMap<String, VectorShard>,
}

impl ShardHost {
    /// A host with an explicit registry (tests use small caps).
    #[cfg(test)]
    pub fn with_registry(registry: ResidentRegistry) -> Self {
        ShardHost {
            registry,
            shards: BTreeMap::new(),
        }
    }

    /// Number of resident shards.
    #[cfg(test)]
    pub fn resident_count(&self) -> usize {
        self.shards.len()
    }

    /// The shard for `key`, cold-loaded from `store` on first use or after
    /// eviction / poisoning, registered (evicting LRU others) on load.
    fn get(&mut self, key: &str, store: &dyn SqlStore) -> Result<&mut VectorShard, OpError> {
        if self.shards.get(key).is_some_and(VectorShard::is_poisoned) {
            self.shards.remove(key);
            self.registry.remove(key);
        }
        if !self.shards.contains_key(key) {
            let shard = VectorShard::open(store)?;
            for v in self.registry.touch(key, shard.resident_bytes()) {
                self.shards.remove(&v);
            }
            self.shards.insert(key.to_string(), shard);
        }
        self.shards
            .get_mut(key)
            .ok_or(OpError::new(ErrorCode::ServerError, "shard vanished"))
    }

    /// Record `key`'s current size and recency (may evict others).
    fn touch(&mut self, key: &str) {
        let Some(bytes) = self.shards.get(key).map(VectorShard::resident_bytes) else {
            return;
        };
        for v in self.registry.touch(key, bytes) {
            self.shards.remove(&v);
        }
    }

    /// After a failed write: what the write really changed. A shard that is
    /// not poisoned issued no statement (zero); a poisoned one is reopened
    /// from storage (write-through replay makes it whole) and diffed;
    /// `None` if it cannot be reopened.
    fn applied_after_failure(
        &mut self,
        key: &str,
        store: &dyn SqlStore,
        before: DeltaWire,
    ) -> Option<DeltaWire> {
        if !self.shards.get(key).is_some_and(VectorShard::is_poisoned) {
            return Some(DeltaWire::default());
        }
        self.get(key, store)
            .ok()
            .map(|s| DeltaWire::from(s.usage_totals()).minus(before))
    }
}

/// Serve one encoded request for the DO whose stable id is `key`.
/// `own_name`, when the runtime exposes it, must equal the name the request
/// maps to (defence in depth; the stored identity check is authoritative).
pub fn serve(
    host: &mut ShardHost,
    key: &str,
    own_name: Option<&str>,
    store: &dyn SqlStore,
    body: &[u8],
) -> String {
    let reply: Reply<ShardOut> = match serde_json::from_slice::<ShardRequest>(body) {
        Ok(req) => handle(host, key, own_name, store, req).map_err(|e| WireErr::from_op(&e)),
        Err(_) => Err(WireErr::from_op(&OpError::invalid("malformed shard call"))),
    };
    serde_json::to_string(&reply)
        .unwrap_or_else(|_| String::from(r#"{"Err":{"code":"server_error"}}"#))
}

/// The identity a request expects.
pub fn request_meta(req: &ShardRequest) -> Result<DoMeta, OpError> {
    let tenant =
        TenantKey::parse(&req.tenant_key).map_err(|_| OpError::invalid("malformed tenant"))?;
    let uid = CollectionUid::parse(&req.uid).map_err(|_| OpError::invalid("malformed uid"))?;
    let shard = ShardIndex::parse(&req.shard.to_string())
        .map_err(|_| OpError::invalid("malformed shard"))?;
    shard_meta_for(&tenant, uid, shard)
}

fn wire_match(m: Match) -> MatchWire {
    MatchWire {
        score_bits: m.rank_score().to_bits(),
        id: m.id,
        distance: m.distance,
        metadata: m.metadata,
        values: m.values,
    }
}

fn actor(a: &crate::wire::ActorWire) -> Actor<'_> {
    Actor {
        sub: &a.sub,
        jti: &a.jti,
        family_id: &a.family_id,
        act_sub: a.act_sub.as_deref(),
    }
}

/// Run one call.
pub fn handle(
    host: &mut ShardHost,
    key: &str,
    own_name: Option<&str>,
    store: &dyn SqlStore,
    req: ShardRequest,
) -> Result<ShardOut, OpError> {
    let dm = request_meta(&req)?;
    if own_name.is_some_and(|n| n != dm.do_name().as_str()) {
        return Err(OpError::not_found());
    }
    match req.call {
        ShardCall::Plan { cfg, rows } => {
            let shard = host.get(key, store)?;
            let plan = shard.plan_upsert(&dm, &cfg.to_config(), rows)?;
            let delta = plan.delta.into();
            host.touch(key);
            Ok(ShardOut::Planned { delta })
        }
        ShardCall::Apply {
            cfg,
            rows,
            admitted,
            actor: who,
            now,
        } => {
            let shard = host.get(key, store)?;
            let plan = shard.plan_upsert(&dm, &cfg.to_config(), rows)?;
            if DeltaWire::from(plan.delta).exceeds(admitted) {
                // The shard changed since the plan the ledger admitted:
                // never write more than was charged.
                return Err(OpError::new(
                    ErrorCode::Conflict,
                    "concurrent change, retry",
                ));
            }
            let before = DeltaWire::from(shard.usage_totals());
            let res = shard.apply_upsert(store, plan, actor(&who), now);
            written_or_failed(
                host,
                key,
                store,
                before,
                res.map(|o| (o.upserted, o.write_seq, o.delta.into())),
            )
        }
        ShardCall::Delete {
            ids,
            actor: who,
            dry_run,
            now,
        } => {
            let shard = host.get(key, store)?;
            let before = DeltaWire::from(shard.usage_totals());
            let res = shard.delete(store, &dm, &ids, actor(&who), dry_run, now);
            written_or_failed(
                host,
                key,
                store,
                before,
                res.map(|o| (o.deleted, o.write_seq, o.delta.into())),
            )
        }
        ShardCall::Query {
            cfg,
            req: q,
            steps_before,
        } => {
            let cfg = cfg.to_config();
            let v = validate_query(&q, &cfg)?;
            let shard = host.get(key, store)?;
            let steps = steps_before.saturating_add(shard.query_steps(&v.filter));
            if steps > MAX_QUERY_STEPS {
                return Err(OpError::new(ErrorCode::BudgetExceeded, "query step budget"));
            }
            let out = shard.query(&dm, &cfg, &q)?;
            host.touch(key);
            Ok(ShardOut::Matches {
                matches: out.matches.into_iter().map(wire_match).collect(),
                scanned: out.scanned,
                steps,
            })
        }
        ShardCall::Fetch {
            ids,
            include_values,
        } => {
            if ids.len() > MAX_FETCH_IDS {
                return Err(OpError::new(ErrorCode::PayloadTooLarge, "too many ids"));
            }
            let shard = host.get(key, store)?;
            let got = shard.fetch(&dm, &ids, include_values)?;
            host.touch(key);
            Ok(ShardOut::Fetched {
                matches: got.into_iter().map(wire_match).collect(),
            })
        }
        ShardCall::Stats => {
            let shard = host.get(key, store)?;
            if shard.identity().is_some_and(|id| *id != dm) {
                return Err(OpError::not_found());
            }
            let out = ShardOut::Stats {
                count: shard.len() as u64,
                resident_bytes: shard.resident_bytes(),
                write_seq: shard.write_seq(),
                snapshot_seq: shard.snapshot_seq(),
            };
            host.touch(key);
            Ok(out)
        }
    }
}

fn written_or_failed(
    host: &mut ShardHost,
    key: &str,
    store: &dyn SqlStore,
    before: DeltaWire,
    res: Result<(u64, u64, DeltaWire), OpError>,
) -> Result<ShardOut, OpError> {
    match res {
        Ok((count, write_seq, delta)) => {
            host.touch(key);
            Ok(ShardOut::Written {
                count,
                write_seq,
                delta,
            })
        }
        Err(e) => Ok(ShardOut::Failed {
            err: WireErr::from_op(&e),
            applied: host.applied_after_failure(key, store, before),
        }),
    }
}
