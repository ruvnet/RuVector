//! `QuantShard` Durable Object core (ADR-351 §3 rv-quant, M4): one shard
//! of an `index = rabitq` collection. It speaks the `VectorShard` wire
//! ([`ShardCall`] / [`ShardOut`] and the drop [`WipeRequest`]), so the
//! gateway's upsert / query / fetch / delete executors — and with them the
//! ledger's admit-before-write and refund accounting — run unchanged; only
//! the transport differs (`quant_route`).
//!
//! f32 originals and metadata live in DO SQLite (`quant_store`); the
//! resident state is the RaBitQ code set (`ruvector-edge-quant`), which a
//! query scans before reranking the best candidates exactly from SQLite.
//! Resident shards live in a [`QuantHost`] (one per isolate), keyed by the
//! DO's own id, with LRU eviction through a `ResidentRegistry`.
//!
//! Differences from `VectorShard`, all refused explicitly: metadata
//! `filter` is `400` (no filter index on the quant path), and no per-row
//! audit (`ops`) log is kept.

use crate::durable::wipe::WipeRequest;
use crate::quant_load::{self as ql, Rebuild, Resident, FLUSH_AFTER_MS, FLUSH_ROWS, TURN_UNITS};
use crate::quant_store::{self as qs, QMeta};
use crate::shard_core::URGENT_ALARM_MS;
use crate::wire::{DeltaWire, Reply, ShardCall, ShardOut, ShardRequest, WireErr};
use ruvector_edge_quant::Budget;
use ruvector_edge_store::shard::{MAX_DELETE_IDS, MAX_FETCH_IDS};
use ruvector_edge_store::{ErrorCode, OpError, ResidentRegistry, SqlStore};
use ruvector_edge_tenancy::{DoMeta, Service};
use std::collections::BTreeMap;

pub use crate::quant_load::LoadReport;

enum Slot {
    Ready(Box<Resident>),
    Rebuilding(Box<Rebuild>),
}

/// This isolate's resident quant budget: the quant share of the 56 MB
/// isolate cap (ADR-351 §6.1), split explicitly with `VectorShard` and
/// `GraphStore` (DOs of one script can share an isolate). Three 50k × 384
/// shards (≈ 4.7 MB each) fit.
pub const QUANT_RESIDENT_CAP_BYTES: u64 = 16_000_000;

/// Load work above which a turn that cold-opened / rebuilt a shard ends
/// there (`503`, retry) instead of also serving the request. Workers Paid:
/// the load cap itself, so any cold load the budget admits is served in
/// its own turn — a 50k × 384 cold load (≈ 3.4 ms native) plus a warm
/// query (≈ 2.1 ms) or a 500-row upsert (≈ 5 ms) is ≈ 15 ms of wasm — and
/// only a turn that also re-encoded rows past it answers `503`. No extra
/// memory: the load is paid for in one turn or two. (Free: `TURN_UNITS /
/// 2`, so every 50k cold load answered one `503`.)
pub const LOAD_TURN_UNITS: u64 = MAX_LOAD_UNITS;

/// Cold-load work cap (the crate default): the row cap binds first, the
/// memory-bound ≈ 114.5k rows at 384 dims being ≈ 12.4M units (≈ 12 ms
/// wasm); 400M units is ≤ 0.6 s of wasm at the crate's 1.1–1.5 ns/unit.
/// (Free: 6M, which also refused 50k-row shards at dims above 384.)
pub const MAX_LOAD_UNITS: u64 = 400_000_000;

/// Per-shard limits on Workers Paid (30 s CPU per request / alarm).
///
/// Measured (native, the Worker's `opt-level = "z"` release profile; wasm
/// ≈ 1.1–1.5× that): at 50k × 384 a snapshot read + decode is ≈ 3.4 ms
/// (5.39M load units), a snapshot flush (encode + write) ≈ 4–4.5 ms, a warm
/// query ≈ 2.1 ms, a 500-row upsert ≈ 5 ms, so CPU does not bind.
/// `max_vectors` stays at the 50k design point for **memory**: the 14 MB
/// shard cap refuses growth at ≈ 114.5k rows × 384 (the upsert peak doubles
/// the packed buffer), a 100k shard would take most of
/// [`QUANT_RESIDENT_CAP_BYTES`], and a cold load reads every frame at once
/// (≈ 3 MB at 50k, ≈ 6 MB at 100k plus the JS copy) against the isolate's
/// ≈ 5 MB spare. Raising it needs frames streamed on load first. The crate
/// default (150k rows) is never reachable at 384 dims.
pub fn edge_budget() -> Budget {
    Budget {
        max_vectors: 50_000,
        max_load_units: MAX_LOAD_UNITS,
        ..Budget::default()
    }
}

/// Resident quant shards of one isolate.
pub struct QuantHost {
    slots: BTreeMap<String, Slot>,
    registry: ResidentRegistry,
    /// DOs whose tables this isolate has already ensured.
    schema: std::collections::BTreeSet<String>,
    /// Per-shard limits (policy; tests shrink them).
    pub budget: Budget,
    /// The last cold open's cost.
    pub last_load: Option<LoadReport>,
}

impl Default for QuantHost {
    fn default() -> Self {
        QuantHost {
            slots: BTreeMap::new(),
            registry: ResidentRegistry::new(QUANT_RESIDENT_CAP_BYTES),
            schema: Default::default(),
            budget: edge_budget(),
            last_load: None,
        }
    }
}

impl QuantHost {
    /// `true` when `key` is loaded and ready (no load work pending).
    pub fn is_ready(&self, key: &str) -> bool {
        matches!(self.slots.get(key), Some(Slot::Ready(_)))
    }

    /// Bytes this host's registry accounts for (tests).
    #[cfg(test)]
    pub fn registered_total(&self) -> u64 {
        self.registry.total()
    }

    fn register(&mut self, key: &str, bytes: u64) {
        for v in self.registry.touch(key, bytes) {
            self.slots.remove(&v);
        }
    }

    /// Create the tables once per DO per isolate (not on every call).
    fn ensure_schema(&mut self, key: &str, store: &dyn SqlStore) -> Result<(), OpError> {
        if !self.schema.contains(key) {
            qs::ensure_schema(store)?;
            self.schema.insert(key.to_string());
        }
        Ok(())
    }

    /// Forget `key`'s resident state.
    pub fn evict(&mut self, key: &str) {
        self.slots.remove(key);
        self.registry.remove(key);
    }

    /// Resident bytes of `key` (0 when cold).
    pub fn resident_bytes(&self, key: &str) -> u64 {
        match self.slots.get(key) {
            Some(Slot::Ready(r)) => r.q.resident_bytes(),
            _ => 0,
        }
    }

    fn touch(&mut self, key: &str) {
        let bytes = self.resident_bytes(key);
        self.register(key, bytes);
    }

    /// The ready shard, cold-opening / advancing a rebuild by one turn.
    /// `503` (retryable) while a rebuild is still in progress, and also
    /// when this turn's load work reached [`LOAD_TURN_UNITS`]: the turn
    /// that loads a large shard does not also query, write or flush it.
    pub(crate) fn ready(
        &mut self,
        key: &str,
        store: &dyn SqlStore,
        meta: &QMeta,
    ) -> Result<&mut Resident, OpError> {
        let mut spent = 0;
        if !self.slots.contains_key(key) {
            let (rb, report) = ql::open(store, meta, self.budget)?;
            spent = report.load_units;
            self.last_load = Some(report);
            let bytes = rb.resident_bytes();
            self.slots
                .insert(key.into(), Slot::Rebuilding(Box::new(rb)));
            // Registered while rebuilding: a decoded snapshot is already
            // (almost) the whole shard.
            self.register(key, bytes);
        }
        if let Some(Slot::Rebuilding(rb)) = self.slots.get_mut(key) {
            let before = rb.done_rows;
            let step = ql::step(rb, store, TURN_UNITS.saturating_sub(spent).max(1));
            spent = spent.saturating_add(rb.encode_units(rb.done_rows - before));
            let bytes = rb.resident_bytes();
            match step {
                Err(e) => {
                    self.evict(key);
                    return Err(e);
                }
                Ok(false) => {
                    self.register(key, bytes);
                    return Err(rebuilding());
                }
                Ok(true) => {}
            }
            let Some(Slot::Rebuilding(rb)) = self.slots.remove(key) else {
                return Err(rebuilding());
            };
            self.slots
                .insert(key.into(), Slot::Ready(Box::new(rb.into_resident())));
            self.touch(key);
            if spent >= LOAD_TURN_UNITS {
                return Err(rebuilding());
            }
        }
        let budget = self.budget;
        match self.slots.get_mut(key) {
            Some(Slot::Ready(r)) => {
                // Limits are host policy, not persisted state.
                if r.q.config().budget != budget {
                    r.q.set_budget(budget);
                }
                Ok(r)
            }
            _ => Err(rebuilding()),
        }
    }
}

fn rebuilding() -> OpError {
    OpError::new(ErrorCode::ShardUnavailable, "quant shard loading, retry")
}

/// The quant identity a shard request names (`service = quant`).
pub fn request_meta(req_tenant: &str, uid: &str, shard: u32) -> Result<DoMeta, OpError> {
    let s = shard.to_string();
    DoMeta::from_kv([
        (ruvector_edge_tenancy::meta::META_TENANT_KEY, req_tenant),
        (
            ruvector_edge_tenancy::meta::META_SERVICE,
            Service::Quant.as_str(),
        ),
        (ruvector_edge_tenancy::meta::META_COLLECTION_UID, uid),
        (ruvector_edge_tenancy::meta::META_SHARD, s.as_str()),
    ])
    .ok()
    .flatten()
    .ok_or(OpError::invalid("malformed shard identity"))
}

/// Delay (ms) until `key` needs its alarm: urgent while rebuilding or when
/// many rows are unflushed, else a timer flush; `None` when idle.
pub fn next_alarm(host: &QuantHost, key: &str) -> Option<u64> {
    match host.slots.get(key)? {
        Slot::Rebuilding(_) => Some(URGENT_ALARM_MS),
        Slot::Ready(r) if r.dirty_rows >= FLUSH_ROWS => Some(URGENT_ALARM_MS),
        Slot::Ready(r) if r.dirty_rows > 0 => Some(FLUSH_AFTER_MS),
        Slot::Ready(_) => None,
    }
}

fn maintain(host: &mut QuantHost, key: &str, store: &dyn SqlStore) -> Result<(), OpError> {
    host.ensure_schema(key, store)?;
    let meta = QMeta::read(store)?;
    if meta.ident.is_none() || meta.wiped {
        host.evict(key);
        return Ok(());
    }
    // A turn that loads (cold open, replay, rebuild) never also flushes.
    // On Workers Free that was CPU (both together could overrun 10 ms and a
    // killed turn loses its writes); on Paid both fit easily, but the split
    // is kept for memory: the load's frame buffers (and their JS copies)
    // and the flush's freshly encoded frames are never live in one turn,
    // so the peak stays one frame set (≈ 3 MB at 50k × 384) against the
    // isolate's ≈ 5 MB spare. It costs one extra alarm.
    let loaded = host.is_ready(key);
    let r = host.ready(key, store, &meta)?;
    if !loaded {
        return Ok(());
    }
    if r.dirty_rows > 0 {
        let mut after = meta.clone();
        ql::flush(r, store, &mut after)?;
        after.write(store, &meta)?;
    }
    Ok(())
}

/// The DO alarm: one rebuild turn or one snapshot flush. A failure other
/// than "still rebuilding" drops the resident state (the next request
/// re-opens from storage).
pub fn alarm(host: &mut QuantHost, key: &str, store: &dyn SqlStore) -> Option<u64> {
    if let Err(e) = maintain(host, key, store) {
        if e.detail != rebuilding().detail {
            host.evict(key);
        }
    }
    next_alarm(host, key)
}

/// Serve one encoded request (a [`ShardRequest`] or a [`WipeRequest`]);
/// returns the reply and whether the shard was wiped.
pub fn serve(
    host: &mut QuantHost,
    key: &str,
    own_name: Option<&str>,
    store: &dyn SqlStore,
    body: &[u8],
) -> (String, bool) {
    if let Ok(w) = serde_json::from_slice::<WipeRequest>(body) {
        let r: Reply<DeltaWire> = wipe(own_name, store, &w).map_err(|e| WireErr::from_op(&e));
        let ok = r.is_ok();
        if ok {
            host.evict(key);
        }
        return (encode(&r), ok);
    }
    let reply: Reply<ShardOut> = match serde_json::from_slice::<ShardRequest>(body) {
        Ok(req) => handle(host, key, own_name, store, req).map_err(|e| WireErr::from_op(&e)),
        Err(_) => Err(WireErr::from_op(&OpError::invalid("malformed shard call"))),
    };
    (encode(&reply), false)
}

fn encode<T: serde::Serialize>(r: &Reply<T>) -> String {
    serde_json::to_string(r).unwrap_or_else(|_| String::from(r#"{"Err":{"code":"server_error"}}"#))
}

fn wipe(
    own_name: Option<&str>,
    store: &dyn SqlStore,
    w: &WipeRequest,
) -> Result<DeltaWire, OpError> {
    let dm = request_meta(&w.tenant_key, &w.uid, w.shard)?;
    if !w.wipe || own_name.is_some_and(|n| n != dm.do_name().as_str()) {
        return Err(OpError::not_found());
    }
    qs::ensure_schema(store)?;
    let meta = QMeta::read(store)?;
    let released = match &meta.ident {
        None => DeltaWire::default(),
        Some(i) if *i == dm.do_name().as_str() && !meta.wiped => usage(&meta),
        Some(_) => return Err(OpError::not_found()),
    };
    qs::wipe(store)?;
    Ok(released)
}

fn usage(m: &QMeta) -> DeltaWire {
    DeltaWire {
        vectors: m.count as i64,
        floats: (m.count * u64::from(m.dim)) as i64,
        bytes: m.bytes as i64,
    }
}

/// Run one call.
pub fn handle(
    host: &mut QuantHost,
    key: &str,
    own_name: Option<&str>,
    store: &dyn SqlStore,
    req: ShardRequest,
) -> Result<ShardOut, OpError> {
    let dm = request_meta(&req.tenant_key, &req.uid, req.shard)?;
    let name = dm.do_name();
    if own_name.is_some_and(|n| n != name.as_str()) {
        return Err(OpError::not_found());
    }
    host.ensure_schema(key, store)?;
    let meta = QMeta::read(store)?;
    if meta.wiped || meta.ident.as_deref().is_some_and(|i| i != name.as_str()) {
        return Err(OpError::not_found());
    }
    let out = match req.call {
        ShardCall::Stats => Ok(ShardOut::Stats {
            count: meta.count,
            resident_bytes: host.resident_bytes(key),
            write_seq: meta.write_seq,
            snapshot_seq: meta.snap_seq,
        }),
        ShardCall::Plan { cfg, rows } => {
            crate::quant_write::check_cfg(&cfg, &meta)?;
            let delta = crate::quant_write::plan(store, &cfg, rows)?.delta;
            Ok(ShardOut::Planned { delta })
        }
        ShardCall::Apply {
            cfg,
            rows,
            admitted,
            ..
        } => {
            crate::quant_write::check_cfg(&cfg, &meta)?;
            let p = crate::quant_write::plan(store, &cfg, rows)?;
            if p.delta.exceeds(admitted) {
                return Err(OpError::new(
                    ErrorCode::Conflict,
                    "concurrent change, retry",
                ));
            }
            let mut after = meta.clone();
            if after.ident.is_none() {
                after.ident = Some(name.as_str().to_string());
                after.dim = cfg.dim;
                after.metric = Some(cfg.metric);
                after.seed = crate::quant_write::seed_of(&dm);
            }
            crate::quant_write::apply(host, key, store, &meta, after, p)
        }
        ShardCall::Query {
            cfg,
            req: q,
            steps_before,
        } => {
            crate::quant_write::check_cfg(&cfg, &meta)?;
            crate::quant_query::query(host, key, store, &meta, &q, steps_before)
        }
        ShardCall::Fetch {
            ids,
            include_values,
        } => {
            if ids.len() > MAX_FETCH_IDS {
                return Err(OpError::new(ErrorCode::PayloadTooLarge, "too many ids"));
            }
            let rows = qs::rows_by_ids(store, &ids)?;
            let matches = rows
                .into_iter()
                .map(|r| crate::quant_query::wire(r, 0.0, true, include_values))
                .collect();
            Ok(ShardOut::Fetched { matches })
        }
        ShardCall::Delete { ids, dry_run, .. } => {
            if ids.len() > MAX_DELETE_IDS {
                return Err(OpError::new(ErrorCode::PayloadTooLarge, "too many ids"));
            }
            crate::quant_write::delete(host, key, store, &meta, &ids, dry_run)
        }
    };
    if out.is_ok() {
        host.touch(key);
    }
    out
}
