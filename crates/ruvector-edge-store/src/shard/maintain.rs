//! Background maintenance driven by the Durable Object alarm (ADR-351
//! §6.1 `flush`, `compact_slice`, `requantize`): pure and synchronous, so
//! the DO shell only schedules it ([`VectorShard::maintenance_due`]) and
//! native tests drive it directly.
//!
//! Order per run (one kind of work per call, each a coalesced commit):
//! 0. **rebuild** of an HNSW index with no usable epoch (requests do not
//!    build graphs; they answer `503` until this has run);
//! 1. **requantize** (l2/dot): once the shard has doubled past its training
//!    sample (< [`super::ann::TRAIN_ROWS`] rows, so this rebuild is never
//!    cap-sized), retrain from a strided sample of `vectors` and rebuild —
//!    a rebase;
//! 2. **compact**: flat when ≥ [`FLAT_COMPACT_DEAD`] of its slots are empty,
//!    HNSW when ≥ [`HNSW_REPAIR_DEAD`] of its nodes are tombstones, and
//!    either whenever dead slots are what blocks admission at the resident
//!    cap. HNSW link repair runs in slices of
//!    [`super::ann::repair_slice_nodes`] (one per alarm, re-armed until
//!    done); the final slice purges, renumbers iids densely in `vectors`
//!    and writes a rebase;
//! 3. **flush** of any op not yet in a persisted epoch (the write path
//!    flushes synchronously at [`flush_ops`]; this covers the
//!    [`FLUSH_AFTER_MS`] timer and a failed post-write flush).
//!
//! A deterministic encoder refusal (an HNSW iid gap) heals by a dense
//! rebuild instead of poisoning the shard; other storage errors poison.

use super::ann::{estimate_bytes, hnsw_node_ms, repair_slice_nodes, Ann};
use super::codec::{IndexConfig, ShardConfig};
use super::persist::is_encode_refused;
use super::{VectorShard, SHARD_RESIDENT_CAP_BYTES};
use crate::distance::Metric;
use crate::error::OpError;
use crate::ports::{SqlStore, StoreError};
use ruvector_edge_index::memory;

/// Unpersisted ops that force a synchronous flush in the write path at the
/// default HNSW parameters and for flat, so a cold load replays fewer than
/// 200 ops (ADR §15 M2b). Heavier HNSW configurations flush sooner
/// ([`flush_ops`]).
pub const FLUSH_OPS: u64 = 200;
/// Timer flush for a shard with fewer pending ops.
pub const FLUSH_AFTER_MS: u64 = 60_000;
/// Flat compaction threshold (share of empty slots).
pub const FLAT_COMPACT_DEAD: f64 = 0.25;
/// HNSW tombstone repair/purge threshold (share of deleted nodes).
pub const HNSW_REPAIR_DEAD: f64 = 0.10;
/// Below this many slots compaction is not worth a rebase.
pub const COMPACT_MIN_SLOTS: u64 = 1024;
/// Replay share of the §15 lazy-load budget (ms, wasm) for the fallback
/// path, which replays up to two flush intervals.
pub const REPLAY_BUDGET_MS: f64 = 600.0;
/// Decode share of the lazy-load budget (ms, wasm).
pub const DECODE_BUDGET_MS: f64 = 300.0;
/// CPU budget of a full HNSW rebuild in one alarm (ms, wasm): half of the
/// Worker's `limits.cpu_ms` (wrangler.toml).
pub const REBUILD_BUDGET_MS: f64 = 60_000.0;

/// Rows per write batch assumed when checking whether dead slots block
/// admission.
const ADMIT_PROBE_ROWS: u64 = 64;

/// Unpersisted ops that force a flush for `cfg`: [`FLUSH_OPS`], lowered
/// for HNSW configurations whose per-op replay cost would push the
/// fallback replay (two intervals plus a sync batch each) past
/// [`REPLAY_BUDGET_MS`]; never below 16.
pub fn flush_ops(cfg: &ShardConfig) -> u64 {
    let ms = hnsw_node_ms(cfg);
    if ms <= 0.0 {
        return FLUSH_OPS;
    }
    let batch = super::write::HNSW_SYNC_UPSERT as f64;
    let n = REPLAY_BUDGET_MS / ms / 2.0 - batch;
    (n.max(0.0) as u64).clamp(16, FLUSH_OPS)
}

/// Largest HNSW slot count for `cfg` whose full alarm rebuild fits
/// [`REBUILD_BUDGET_MS`] and whose chunk decode fits [`DECODE_BUDGET_MS`]
/// (the load-derived cap; the resident byte cap applies separately).
/// `u64::MAX` for flat.
pub fn hnsw_node_cap(cfg: &ShardConfig) -> u64 {
    let ms = hnsw_node_ms(cfg);
    if ms <= 0.0 {
        return u64::MAX;
    }
    let p = super::ann::hnsw_params(cfg);
    let decode = memory::max_nodes_for_load_budget(
        DECODE_BUDGET_MS,
        memory::DECODE_MS_PER_MIB_WASM,
        cfg.dim as usize,
        &p,
    ) as u64;
    ((REBUILD_BUDGET_MS / ms) as u64).min(decode)
}

/// When the alarm should next run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Due {
    /// Work is ready now.
    Now,
    /// Only a timer flush is pending.
    After(u64),
}

/// What one maintenance run did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaintainReport {
    /// HNSW index rebuilt from `vectors` (no usable epoch, or healed).
    pub rebuilt: bool,
    /// Quantizer retrained and index rebuilt.
    pub requantized: bool,
    /// One HNSW link-repair slice ran (more remain).
    pub repaired: bool,
    /// Iids renumbered densely (flat compaction or HNSW purge).
    pub compacted: bool,
    /// A new epoch was written.
    pub flushed: bool,
}

impl VectorShard {
    fn needs_requantize(&self) -> bool {
        match (&self.config, &self.quant) {
            (Some(c), Some(q)) if c.metric != Metric::Cosine => {
                q.rows < super::ann::TRAIN_ROWS as u64
                    && self.slab.len() as u64 >= 2 * q.rows.max(1)
            }
            _ => false,
        }
    }

    fn needs_compact(&self) -> bool {
        let Some(ann) = &self.ann else { return false };
        let threshold = match ann {
            Ann::Flat(_) => FLAT_COMPACT_DEAD,
            Ann::Hnsw(_) => HNSW_REPAIR_DEAD,
        };
        ann.slots() >= COMPACT_MIN_SLOTS
            && (ann.dead_ratio() >= threshold || self.dead_slots_block_admission())
    }

    /// Dead slots (never reused before compaction) are what keeps a small
    /// write batch out: it is refused at the resident cap now and would be
    /// admitted after compacting.
    fn dead_slots_block_admission(&self) -> bool {
        let Some(cfg) = &self.config else {
            return false;
        };
        let live = self.slab.len() as u64;
        let slots = u64::try_from(self.next_iid).unwrap_or(u64::MAX);
        if slots <= live + 1 {
            return false;
        }
        // The same projection as `plan_upsert` admission.
        let per_row = self.slab.resident / live.max(1);
        let rows = self.slab.resident + per_row * ADMIT_PROBE_ROWS;
        let allocated = self.ann.as_ref().map_or(0, Ann::memory_bytes);
        let over = |slots_after: u64, index: u64| {
            rows.saturating_add(slots_after.saturating_mul(4))
                .saturating_add(index)
                > SHARD_RESIDENT_CAP_BYTES
        };
        let (now, dense) = (slots + ADMIT_PROBE_ROWS, live + 1 + ADMIT_PROBE_ROWS);
        // Compaction releases the dead capacity (`shrink_to_fit`).
        over(now, allocated.max(estimate_bytes(cfg, now)))
            && !over(dense, estimate_bytes(cfg, dense))
    }

    /// When maintenance should run next (`None`: nothing to do).
    pub fn maintenance_due(&self) -> Option<Due> {
        if self.poisoned || self.wiped || self.config.is_none() {
            return None;
        }
        if self.rebuild_pending
            || self.flush_failed
            || self.repair_cursor.is_some()
            || self.needs_requantize()
            || self.needs_compact()
        {
            return Some(Due::Now);
        }
        let limit = self.config.as_ref().map_or(FLUSH_OPS, flush_ops);
        match self.pending_ops() {
            0 => None,
            n if n >= limit => Some(Due::Now),
            _ => Some(Due::After(FLUSH_AFTER_MS)),
        }
    }

    /// Run the most urgent maintenance step. Storage errors poison the
    /// shard like any write; an encoder refusal heals (see module docs).
    pub fn maintain(&mut self, store: &dyn SqlStore) -> Result<MaintainReport, OpError> {
        self.ensure_live()?;
        let mut rep = MaintainReport::default();
        if self.config.is_none() || self.wiped {
            return Ok(rep);
        }
        let was_pending = self.rebuild_pending;
        let res = self.ensure_index_with(store, true).and_then(|_| {
            if was_pending {
                rep.rebuilt = true;
                rep.flushed = true;
                Ok(())
            } else {
                self.maintain_step(store, &mut rep)
            }
        });
        let res = match res {
            Err(e) if is_encode_refused(&e) => {
                rep.rebuilt = true;
                rep.flushed = true;
                self.heal(store)
            }
            other => other,
        };
        match res {
            Ok(()) => {
                self.flush_failed = false;
                Ok(rep)
            }
            Err(e) => self.poison(e),
        }
    }

    fn maintain_step(
        &mut self,
        store: &dyn SqlStore,
        rep: &mut MaintainReport,
    ) -> Result<(), StoreError> {
        if self.needs_requantize() {
            rep.requantized = true;
            rep.flushed = true;
            let epoch = self.quant.as_ref().map_or(1, |q| q.params.epoch() + 1);
            self.train_from_vectors(store, epoch)?;
            self.rebuild_from_vectors(store)?;
            self.flush_rebase(store)
        } else if self.repair_cursor.is_some() || self.needs_compact() {
            if self.repair_slice() {
                rep.repaired = true;
                return Ok(());
            }
            rep.compacted = true;
            rep.flushed = true;
            self.compact(store)?;
            self.flush_rebase(store)
        } else if self.pending_ops() > 0 || self.flush_failed {
            rep.flushed = true;
            self.flush_keep(store)
        } else {
            Ok(())
        }
    }

    /// Run one link-repair slice; `true` while more slices remain.
    fn repair_slice(&mut self) -> bool {
        let (Some(cfg), Some(ann)) = (self.config.as_ref(), self.ann.as_mut()) else {
            return false;
        };
        if !matches!(cfg.index, IndexConfig::Hnsw { .. }) {
            return false;
        }
        let from = self.repair_cursor.unwrap_or(0);
        self.repair_cursor = ann.repair_links(from, repair_slice_nodes(cfg));
        self.repair_cursor.is_some()
    }

    /// Recover from an encoder refusal: dense renumber, rebuild, rebase.
    fn heal(&mut self, store: &dyn SqlStore) -> Result<(), StoreError> {
        self.ann = None;
        self.repair_cursor = None;
        self.rebuild_from_vectors(store)?;
        self.rebuild_pending = false;
        self.flush_rebase(store)
    }

    /// Dense renumbering of the index, mirrored into `vectors` and the slab.
    fn compact(&mut self, store: &dyn SqlStore) -> Result<(), StoreError> {
        let Some(ann) = self.ann.as_mut() else {
            return Ok(());
        };
        let map = ann.compact()?;
        // Give the dead slots' capacity back (the resident cap counts
        // allocated bytes, so compaction is what makes room again).
        ann.shrink_to_fit();
        let mut pairs: Vec<(i64, i64)> = self
            .slab
            .iids
            .iter()
            .map(|&old| {
                let new = usize::try_from(old)
                    .ok()
                    .and_then(|o| map.get(o).copied())
                    .filter(|&n| n != u32::MAX)
                    .map_or(old, i64::from);
                (old, new)
            })
            .collect();
        pairs.sort_unstable();
        self.renumber(store, &pairs)
    }
}
