//! Shard upsert path (ADR-351 §6.1): plan (validate, quota delta, resident
//! growth) in memory, then issue every statement back to back, then mutate
//! the slab and the index, then flush the index if enough ops are pending.
//!
//! Statement order per row is `ops` → `vectors` → `filter_idx`, with the
//! `meta` counters and the op-log prune last, so a torn write can only
//! leave ops logged ahead of `meta.write_seq` with their row statements
//! partly applied; [`VectorShard::open`] replays those ops write-through,
//! and the lazy index load replays them onto the persisted epoch. After any
//! storage error the shard is **poisoned** and every call fails with `503
//! shard_unavailable` until it is reopened. A failed post-write *flush*
//! does not poison: rows and index are consistent in memory and the next
//! flush (write path or alarm) retries.
//!
//! Admission checks the per-shard stored-float cap and the per-shard
//! resident byte cap ([`SHARD_RESIDENT_CAP_BYTES`]: row bookkeeping plus
//! the index's projected allocation); either is `413 budget_exceeded`.

use super::ann::{estimate_bytes, Ann, QuantState, MAX_SLOTS};
use super::codec::{encode_f32, encode_upsert_body, IndexConfig, ShardConfig};
use super::slab::{row_resident, Slab};
use super::{salt_of, Actor, VectorShard, FLUSH_OPS, SHARD_RESIDENT_CAP_BYTES};
use crate::distance::Metric;
use crate::error::{ErrorCode, OpError};
use crate::filter::{compact, index_rows, validate_metadata, Compact};
use crate::ports::{SqlStore, StoreError, Value};
use crate::schema;
use ruvector_edge_tenancy::quota::limits::MAX_UPSERT_BATCH;
use ruvector_edge_tenancy::{DoMeta, IdentityCheck, VectorId};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use std::collections::BTreeSet;

/// Synchronous upsert batch limit for `hnsw` collections (ADR §7).
pub const HNSW_SYNC_UPSERT: usize = 64;

/// Stored-float cap per shard at M2 (f32 rows live in SQLite; the resident
/// limit is [`SHARD_RESIDENT_CAP_BYTES`]). 16M floats = 64 MB of rows.
pub const M2_SHARD_FLOAT_CAP: u64 = 16_000_000;

/// One vector in an upsert (`values` required at M1; `text` is M3).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpsertRow {
    /// Vector id (1..=256 bytes UTF-8, no control/format characters).
    pub id: String,
    /// Values, length = collection dimension.
    pub values: Vec<f32>,
    /// Optional metadata object (≤ 4 KiB serialized).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Json>,
}

/// Signed change to tenant usage caused by a write.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct UsageDelta {
    /// Net vectors.
    pub vectors: i64,
    /// Net floats (`vectors × dim`).
    pub floats: i64,
    /// Net stored bytes (id + f32 + metadata).
    pub bytes: i64,
}

#[derive(Debug, Clone)]
struct PlannedRow {
    id: String,
    values: Vec<f32>,
    meta_text: Option<String>,
    indexed: Vec<(String, String)>,
    filt: Compact,
}

/// A validated upsert, ready to apply against the same shard state.
#[derive(Debug, Clone)]
pub struct UpsertPlan {
    rows: Vec<PlannedRow>,
    init: Option<(DoMeta, ShardConfig)>,
    new_quant: Option<QuantState>,
    base_seq: u64,
    /// Ids not yet present.
    pub inserted: u64,
    /// Ids replaced.
    pub replaced: u64,
    /// Usage change the ledger must admit before apply.
    pub delta: UsageDelta,
    /// Projected resident bytes after the write.
    pub resident_after: u64,
}

/// Result of an upsert.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UpsertOutcome {
    /// Rows written (or that would be, on `dry_run`).
    pub upserted: u64,
    /// Shard `write_seq` after the write.
    pub write_seq: u64,
    /// Usage change.
    pub delta: UsageDelta,
    /// `true` when nothing was written.
    pub dry_run: bool,
}

pub(crate) fn to_i64(v: u64) -> Result<i64, OpError> {
    i64::try_from(v).map_err(|_| OpError::new(ErrorCode::BudgetExceeded, "counter overflow"))
}

impl VectorShard {
    pub(crate) fn ensure_live(&self) -> Result<(), OpError> {
        if self.poisoned {
            Err(OpError::new(
                ErrorCode::ShardUnavailable,
                "shard must be reopened",
            ))
        } else {
            Ok(())
        }
    }

    pub(crate) fn poison<T>(&mut self, e: StoreError) -> Result<T, OpError> {
        self.poisoned = true;
        Err(e.into())
    }

    /// Validate an upsert and compute its usage delta and resident growth.
    /// `init_config` is used only when the shard is uninitialised (first
    /// write persists identity and config); otherwise the stored config
    /// governs.
    pub fn plan_upsert(
        &self,
        expected: &DoMeta,
        init_config: &ShardConfig,
        rows: Vec<UpsertRow>,
    ) -> Result<UpsertPlan, OpError> {
        self.ensure_live()?;
        let init = match self.check(expected, true)? {
            IdentityCheck::InitializeOnWrite => Some((expected.clone(), init_config.clone())),
            _ => None,
        };
        let cfg = self
            .config
            .as_ref()
            .or(init.as_ref().map(|(_, c)| c))
            .ok_or(OpError::not_found())?
            .clone();
        if rows.is_empty() {
            return Err(OpError::invalid("empty upsert"));
        }
        let hnsw = matches!(cfg.index, IndexConfig::Hnsw { .. });
        if rows.len() > MAX_UPSERT_BATCH as usize || (hnsw && rows.len() > HNSW_SYNC_UPSERT) {
            return Err(OpError::new(
                ErrorCode::PayloadTooLarge,
                "upsert batch too large",
            ));
        }
        let dim = cfg.dim as usize;
        let mut seen = BTreeSet::new();
        let (mut inserted, mut replaced) = (0u64, 0u64);
        let mut bytes: i64 = 0;
        let mut resident = self.slab.resident;
        let mut planned = Vec::with_capacity(rows.len());
        for r in rows {
            let id = VectorId::parse(&r.id)?.as_str().to_string();
            if !seen.insert(id.clone()) {
                return Err(OpError::invalid("duplicate id in batch"));
            }
            if r.values.len() != dim {
                return Err(OpError::new(
                    ErrorCode::DimensionMismatch,
                    "dimension mismatch",
                ));
            }
            if !f32_norm_ok(&r.values) {
                return Err(OpError::new(ErrorCode::NonFiniteValue, "non-finite value"));
            }
            if cfg.metric == Metric::Cosine && ruvector_edge_index::norm(&r.values) <= 0.0 {
                return Err(OpError::invalid("zero vector norm"));
            }
            let meta_text = r.metadata.as_ref().map(validate_metadata).transpose()?;
            let obj = r.metadata.as_ref().and_then(Json::as_object);
            let indexed = obj
                .map(|m| index_rows(m, &cfg.filterable_keys))
                .unwrap_or_default();
            let filt = compact(obj, &cfg.filterable_keys);
            let meta_len = meta_text.as_ref().map_or(0, String::len);
            let new_bytes = to_i64(Slab::stored_of(id.len(), dim, meta_len))?;
            match self.slab.index.get(&id) {
                Some(&slot) => {
                    replaced += 1;
                    bytes += new_bytes - to_i64(self.slab.row_bytes(slot, dim))?;
                    resident = resident.saturating_sub(self.slab.slot_resident(slot));
                }
                None => {
                    inserted += 1;
                    bytes += new_bytes;
                }
            }
            resident = resident.saturating_add(row_resident(id.len(), &filt));
            planned.push(PlannedRow {
                id,
                values: r.values,
                meta_text,
                indexed,
                filt,
            });
        }
        let budget = || OpError::new(ErrorCode::BudgetExceeded, "shard float cap");
        let after_rows = (self.slab.len() as u64)
            .checked_add(inserted)
            .and_then(|n| n.checked_mul(u64::from(cfg.dim)))
            .ok_or_else(budget)?;
        if after_rows > cfg.float_cap {
            return Err(budget());
        }
        // Iids are never reused before a compaction: the index slot space.
        let slots_after = u64::try_from(self.next_iid)
            .unwrap_or(u64::MAX)
            .saturating_add(inserted);
        if slots_after > u64::from(MAX_SLOTS) {
            return Err(OpError::new(
                ErrorCode::BudgetExceeded,
                "iid space exhausted (compaction pending)",
            ));
        }
        // HNSW: the load-derived cap (full alarm rebuild and chunk decode
        // within budget), besides the resident byte cap below.
        if slots_after > super::maintain::hnsw_node_cap(&cfg) {
            return Err(OpError::new(ErrorCode::BudgetExceeded, "shard load budget"));
        }
        let index_after = self
            .ann
            .as_ref()
            .map_or(0, Ann::memory_bytes)
            .max(estimate_bytes(&cfg, slots_after));
        let resident_after = resident
            .saturating_add(slots_after.saturating_mul(4))
            .saturating_add(index_after);
        if resident_after > SHARD_RESIDENT_CAP_BYTES {
            return Err(OpError::new(
                ErrorCode::BudgetExceeded,
                "shard resident memory cap",
            ));
        }
        let new_quant = if self.quant.is_none() && self.slab.len() == 0 {
            let sample: Vec<f32> = planned
                .iter()
                .take(super::ann::TRAIN_ROWS)
                .flat_map(|r| r.values.iter().copied())
                .collect();
            Some(QuantState::train(&cfg, &sample, 1).map_err(OpError::from)?)
        } else {
            None
        };
        Ok(UpsertPlan {
            rows: planned,
            init,
            new_quant,
            base_seq: self.write_seq,
            inserted,
            replaced,
            delta: UsageDelta {
                vectors: to_i64(inserted)?,
                floats: to_i64(inserted * u64::from(cfg.dim))?,
                bytes,
            },
            resident_after,
        })
    }

    /// Outcome a plan would produce, without writing (`dry_run`).
    pub fn preview(&self, plan: &UpsertPlan) -> UpsertOutcome {
        UpsertOutcome {
            upserted: plan.rows.len() as u64,
            write_seq: self.write_seq,
            delta: plan.delta,
            dry_run: true,
        }
    }

    /// Apply a plan made against the current state.
    pub fn apply_upsert(
        &mut self,
        store: &dyn SqlStore,
        plan: UpsertPlan,
        actor: Actor<'_>,
        now: u64,
    ) -> Result<UpsertOutcome, OpError> {
        self.ensure_live()?;
        if plan.base_seq != self.write_seq {
            return Err(OpError::new(ErrorCode::Conflict, "stale upsert plan"));
        }
        self.index_for_request(store)?;
        let ts = to_i64(now)?;
        let snapshot = match self.write_upsert(store, &plan, actor, ts) {
            Ok(s) => s,
            Err(e) => return self.poison(e),
        };
        if let Some((id, cfg)) = plan.init {
            self.salt = salt_of(&id);
            self.identity = Some(id);
            self.config = Some(cfg);
        }
        if let (None, Some(q)) = (&self.quant, plan.new_quant) {
            self.quant = Some(q);
        }
        if let Err(e) = self.apply_rows(plan.rows, plan.inserted) {
            return self.poison(e);
        }
        self.snapshot_seq = snapshot;
        self.after_write(store);
        Ok(UpsertOutcome {
            upserted: self.write_seq - plan.base_seq,
            write_seq: self.write_seq,
            delta: plan.delta,
            dry_run: false,
        })
    }

    /// Mutate slab and index for rows already written.
    fn apply_rows(&mut self, rows: Vec<PlannedRow>, inserted: u64) -> Result<(), StoreError> {
        let cfg = self.config.clone().ok_or(StoreError::Corrupt("config"))?;
        if self.ann.is_none() {
            if let Some(q) = &self.quant {
                self.ann = Some(Ann::new(&cfg, q, rows.len())?);
            }
        }
        if let Some(ann) = self.ann.as_mut() {
            ann.reserve(inserted as usize);
        }
        let dim = cfg.dim as usize;
        for r in rows {
            let iid = self.iid_for(&r.id);
            self.write_seq += 1;
            let seed = Ann::op_seed(self.write_seq, self.salt);
            if let Some(ann) = self.ann.as_mut() {
                ann.upsert(iid, &r.values, seed)?;
            }
            let meta_len = r.meta_text.as_ref().map_or(0, String::len);
            self.slab.put(r.id, iid, dim, meta_len, r.filt);
        }
        Ok(())
    }

    /// Flush the index when [`super::maintain::flush_ops`] ops are
    /// unpersisted ([`FLUSH_OPS`] or fewer). A failure is remembered for
    /// maintenance (which heals an encoder refusal), never surfaced to the
    /// (committed) write.
    pub(crate) fn after_write(&mut self, store: &dyn SqlStore) {
        let limit = self
            .config
            .as_ref()
            .map_or(FLUSH_OPS, super::maintain::flush_ops);
        if self.ann.is_some() && self.pending_ops() >= limit {
            self.flush_failed = self.flush_keep(store).is_err();
        }
    }

    fn write_upsert(
        &self,
        store: &dyn SqlStore,
        plan: &UpsertPlan,
        actor: Actor<'_>,
        ts: i64,
    ) -> Result<u64, StoreError> {
        if let Some((id, cfg)) = &plan.init {
            for (k, v) in id.to_kv().into_iter().chain(cfg.to_kv()) {
                store.exec(schema::META_PUT, &[k.into(), v.into()])?;
            }
        }
        if let (None, Some(q)) = (&self.quant, &plan.new_quant) {
            store.exec(
                schema::META_PUT,
                &[super::codec::keys::QUANT.into(), q.to_meta()?.into()],
            )?;
        }
        let mut seq = self.write_seq;
        let mut next_iid = self.next_iid;
        for r in &plan.rows {
            seq += 1;
            let iid = match self.slab.index.get(&r.id) {
                Some(&slot) => self.slab.iids[slot],
                None => {
                    next_iid += 1;
                    next_iid - 1
                }
            };
            let body = encode_upsert_body(iid, &r.values, r.meta_text.as_deref());
            store.exec(
                schema::OPS_APPEND,
                &ops_row(seq, "upsert", &r.id, ts, actor, Value::Blob(body))?,
            )?;
            let meta = r.meta_text.clone().map_or(Value::Null, Value::Text);
            store.exec(
                schema::VEC_PUT,
                &[
                    r.id.as_str().into(),
                    iid.into(),
                    encode_f32(&r.values).into(),
                    meta,
                    ts.into(),
                    0i64.into(),
                ],
            )?;
            store.exec(schema::FILTER_DELETE_ID, &[r.id.as_str().into()])?;
            for (k, v) in &r.indexed {
                store.exec(
                    schema::FILTER_PUT,
                    &[k.as_str().into(), v.as_str().into(), r.id.as_str().into()],
                )?;
            }
        }
        self.write_counters(store, seq, Some(next_iid))
    }
}

pub(crate) fn ops_row(
    seq: u64,
    op: &str,
    id: &str,
    ts: i64,
    a: Actor<'_>,
    body: Value,
) -> Result<Vec<Value>, StoreError> {
    let seq = i64::try_from(seq).map_err(|_| StoreError::Corrupt("seq overflow"))?;
    Ok(vec![
        seq.into(),
        op.into(),
        id.into(),
        ts.into(),
        a.sub.into(),
        a.jti.into(),
        a.family_id.into(),
        a.act_sub.map_or(Value::Null, Value::from),
        body,
    ])
}

/// Every value finite **and** the squared norm finite in `f32`: the int8
/// index computes in `f32` (`ruvector_edge_index::validate_vector`), so a
/// vector beyond ~1.8e19 in norm is `400 non_finite_value` at M2.
pub(crate) fn f32_norm_ok(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite()) && ruvector_edge_index::norm(v).is_finite()
}
