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
use ruvector_edge_store::shard::{
    validate_query, Actor, Due, Match, MAX_FETCH_IDS, MAX_QUERY_STEPS,
};
use ruvector_edge_store::{
    shard_meta_for, ErrorCode, OpError, ResidentRegistry, SqlStore, VectorShard,
};
use ruvector_edge_tenancy::{CollectionUid, DoMeta, ShardIndex, TenantKey};
use std::collections::BTreeMap;

/// `VectorShard`'s share of the 56 MB isolate resident cap (§6.1): the
/// isolate may also host `QuantShard`s and `GraphStore`s (M4), each with
/// its own explicit share (16 MB, 12 MB). Two full 14 MB shards fit.
pub const VECTOR_RESIDENT_CAP_BYTES: u64 = 28_000_000;

/// Resident shard states of one isolate plus the eviction registry.
#[derive(Debug)]
pub struct ShardHost {
    registry: ResidentRegistry,
    shards: BTreeMap<String, VectorShard>,
}

impl Default for ShardHost {
    fn default() -> Self {
        ShardHost {
            registry: ResidentRegistry::new(VECTOR_RESIDENT_CAP_BYTES),
            shards: BTreeMap::new(),
        }
    }
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

    /// Forget `key`'s resident state (its storage was wiped): a later
    /// request or alarm cold-loads from storage instead of flushing stale
    /// index state back into it.
    pub fn evict(&mut self, key: &str) {
        self.shards.remove(key);
        self.registry.remove(key);
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

/// Delay of an alarm for work that is due now (a floor, so a shard that
/// stays due can never spin the alarm).
pub const URGENT_ALARM_MS: u64 = 1_000;

/// Delay (ms) until the DO `key` needs its maintenance alarm (ADR-351 §6.1
/// alarms: timer flush, compaction, requantize), `None` when idle or not
/// resident.
pub fn next_alarm(host: &ShardHost, key: &str) -> Option<u64> {
    match host.shards.get(key)?.maintenance_due()? {
        Due::Now => Some(URGENT_ALARM_MS),
        Due::After(ms) => Some(ms.max(URGENT_ALARM_MS)),
    }
}

/// The DO alarm: run one maintenance step over the DO's own storage
/// (cold-loading the shard if it was evicted) and return the next delay.
/// A storage error poisons the shard; the next request reopens it.
pub fn alarm(host: &mut ShardHost, key: &str, store: &dyn SqlStore) -> Option<u64> {
    let shard = host.get(key, store).ok()?;
    let _outcome = shard.maintain(store);
    host.touch(key);
    next_alarm(host, key)
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
            shard.load_index(store)?;
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
            shard.load_index(store)?;
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
            let steps = steps_before.saturating_add(shard.query_steps(&v));
            if steps > MAX_QUERY_STEPS {
                return Err(OpError::new(ErrorCode::BudgetExceeded, "query step budget"));
            }
            let out = shard.query(store, &dm, &cfg, &q)?;
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
            let got = shard.fetch(store, &dm, &ids, include_values)?;
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

#[cfg(test)]
mod m2_tests {
    //! M2 wiring: `hnsw` config over the wire, validated `ef`, the alarm's
    //! timer flush, and a cold isolate answering from the flushed epoch.
    use super::*;
    use crate::testkit::{tenant, T0};
    use crate::wire::{ActorWire, CfgWire, ShardCall};
    use ruvector_edge_store::{
        ErrorCode, IndexConfig, MemSqlStore, Metric, QueryRequest, UpsertRow,
    };

    fn call(host: &mut ShardHost, st: &MemSqlStore, call: ShardCall) -> Reply<ShardOut> {
        let req = ShardRequest {
            tenant_key: tenant("org-m2").as_str().to_string(),
            uid: "0123456789abcdef0123456789abcdef".into(),
            shard: 0,
            call,
        };
        let body = serde_json::to_vec(&req).unwrap();
        serde_json::from_str(&serve(host, "do-1", None, st, &body)).unwrap()
    }

    fn query(ef: Option<u32>) -> ShardCall {
        ShardCall::Query {
            cfg: cfg(),
            req: QueryRequest {
                vector: vec![3.0, 3.0, 3.0, 3.0],
                top_k: 3,
                filter: None,
                include: vec![],
                ef,
                rerank: None,
            },
            steps_before: 0,
        }
    }

    fn cfg() -> CfgWire {
        CfgWire {
            dim: 4,
            metric: Metric::L2,
            filterable_keys: vec![],
            index: IndexConfig::HNSW_DEFAULT,
        }
    }

    fn ids(r: Reply<ShardOut>) -> Vec<String> {
        match r {
            Ok(ShardOut::Matches { matches, .. }) => matches.into_iter().map(|m| m.id).collect(),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn hnsw_shard_flushes_on_alarm_and_reloads() {
        // An M1 gateway's config (no `index`) still reads as `flat`.
        let m1: CfgWire =
            serde_json::from_str(r#"{"dim":4,"metric":"l2","filterable_keys":[]}"#).unwrap();
        assert_eq!(m1.index, IndexConfig::Flat);
        let (mut host, st) = (ShardHost::default(), MemSqlStore::new());
        let rows: Vec<UpsertRow> = (0..10)
            .map(|i| UpsertRow {
                id: format!("r{i}"),
                values: vec![i as f32; 4],
                metadata: None,
            })
            .collect();
        let delta = match call(
            &mut host,
            &st,
            ShardCall::Plan {
                cfg: cfg(),
                rows: rows.clone(),
            },
        ) {
            Ok(ShardOut::Planned { delta }) => delta,
            other => panic!("{other:?}"),
        };
        let actor = ActorWire {
            sub: "es1_m2".into(),
            jti: "j".into(),
            family_id: "f".into(),
            act_sub: None,
        };
        let apply = ShardCall::Apply {
            cfg: cfg(),
            rows,
            admitted: delta,
            actor,
            now: T0,
        };
        assert!(matches!(
            call(&mut host, &st, apply),
            Ok(ShardOut::Written { count: 10, .. })
        ));
        // Ten unpersisted ops: a timer flush is scheduled, not an urgent one.
        assert_eq!(
            next_alarm(&host, "do-1"),
            Some(ruvector_edge_store::shard::FLUSH_AFTER_MS)
        );
        assert_eq!(alarm(&mut host, "do-1", &st), None);
        assert_eq!(ids(call(&mut host, &st, query(None))), ["r3", "r2", "r4"]);
        let e = call(
            &mut host,
            &st,
            query(Some(ruvector_edge_store::shard::MAX_EF + 1)),
        );
        assert_eq!(e.unwrap_err().code, ErrorCode::InvalidRequest);
        // A cold isolate decodes the flushed epoch (nothing to replay).
        let mut cold = ShardHost::default();
        assert_eq!(
            ids(call(&mut cold, &st, query(Some(64)))),
            ["r3", "r2", "r4"]
        );
        assert_eq!(next_alarm(&cold, "do-1"), None);
    }
}
