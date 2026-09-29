//! Shard write path (ADR-351 §6.1): plan (validate + quota delta) in memory,
//! then issue every statement back to back, then mutate resident state.
//!
//! Statement order per row is `ops` → `vectors` → `filter_idx`, with the
//! `meta` counters and the op-log prune last, so a torn write can only
//! leave ops logged ahead of `meta.write_seq` with their row statements
//! partly applied; [`VectorShard::open`] replays those ops write-through
//! (re-issuing the row statements, then the counters), so durable state,
//! cold load and live state agree. After any storage error the shard is
//! **poisoned** and every call fails with `503 shard_unavailable` until it
//! is reopened.
//!
//! Admission checks both the per-shard float cap and the per-shard
//! resident byte cap ([`SHARD_RESIDENT_CAP_BYTES`]); either is `413
//! budget_exceeded`.

use super::codec::{encode_f32, encode_upsert_body, ShardConfig};
use super::slab::row_resident;
use super::{Actor, VectorShard, SHARD_RESIDENT_CAP_BYTES};
use crate::distance::{norm, Metric};
use crate::error::{ErrorCode, OpError};
use crate::filter::{compact, index_rows, validate_metadata, Compact};
use crate::ports::{SqlStore, StoreError, Value};
use crate::schema;
use ruvector_edge_tenancy::quota::limits::MAX_UPSERT_BATCH;
use ruvector_edge_tenancy::{DoMeta, IdentityCheck, VectorId};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use std::collections::BTreeSet;

/// Maximum ids per delete call (§7).
pub const MAX_DELETE_IDS: usize = 1000;

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
    base_seq: u64,
    /// Ids not yet present.
    pub inserted: u64,
    /// Ids replaced.
    pub replaced: u64,
    /// Usage change the ledger must admit before apply.
    pub delta: UsageDelta,
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

/// Result of a delete.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeleteOutcome {
    /// Ids that existed and were (or would be) removed.
    pub deleted: u64,
    /// Shard `write_seq` after the write.
    pub write_seq: u64,
    /// Usage change (non-positive).
    pub delta: UsageDelta,
    /// `true` when nothing was written.
    pub dry_run: bool,
}

fn to_i64(v: u64) -> Result<i64, OpError> {
    i64::try_from(v).map_err(|_| OpError::new(ErrorCode::BudgetExceeded, "counter overflow"))
}

impl VectorShard {
    fn ensure_live(&self) -> Result<(), OpError> {
        if self.poisoned {
            Err(OpError::new(
                ErrorCode::ShardUnavailable,
                "shard must be reopened",
            ))
        } else {
            Ok(())
        }
    }

    fn poison<T>(&mut self, e: StoreError) -> Result<T, OpError> {
        self.poisoned = true;
        Err(e.into())
    }

    /// Validate an upsert and compute its usage delta. `init_config` is used
    /// only when the shard is uninitialised (first write persists identity
    /// and config); otherwise the stored config governs.
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
        if rows.len() > MAX_UPSERT_BATCH as usize {
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
            if r.values.iter().any(|v| !v.is_finite()) {
                return Err(OpError::new(ErrorCode::NonFiniteValue, "non-finite value"));
            }
            let n = norm(&r.values);
            if cfg.metric == Metric::Cosine && n == 0.0 {
                return Err(OpError::invalid("zero vector norm"));
            }
            let meta_text = r.metadata.as_ref().map(validate_metadata).transpose()?;
            let obj = r.metadata.as_ref().and_then(Json::as_object);
            let indexed = obj
                .map(|m| index_rows(m, &cfg.filterable_keys))
                .unwrap_or_default();
            let filt = compact(obj, &cfg.filterable_keys);
            let meta_len = meta_text.as_ref().map_or(0, String::len);
            let new_bytes = (id.len() + dim * 4 + meta_len) as i64;
            let new_res = row_resident(dim, id.len(), meta_len, &filt);
            match self.slab.index.get(&id) {
                Some(&slot) => {
                    replaced += 1;
                    bytes += new_bytes - to_i64(self.slab.row_bytes(slot, dim))?;
                    resident = resident.saturating_sub(self.slab.slot_resident(slot, dim));
                }
                None => {
                    inserted += 1;
                    bytes += new_bytes;
                }
            }
            resident = resident.saturating_add(new_res);
            planned.push(PlannedRow {
                id,
                values: r.values,
                meta_text,
                indexed,
                filt,
            });
        }
        let after_rows = (self.slab.len() as u64)
            .checked_add(inserted)
            .and_then(|n| n.checked_mul(u64::from(cfg.dim)))
            .ok_or(OpError::new(ErrorCode::BudgetExceeded, "shard float cap"))?;
        if after_rows > cfg.float_cap {
            return Err(OpError::new(ErrorCode::BudgetExceeded, "shard float cap"));
        }
        if resident > SHARD_RESIDENT_CAP_BYTES {
            return Err(OpError::new(
                ErrorCode::BudgetExceeded,
                "shard resident memory cap",
            ));
        }
        Ok(UpsertPlan {
            rows: planned,
            init,
            base_seq: self.write_seq,
            inserted,
            replaced,
            delta: UsageDelta {
                vectors: to_i64(inserted)?,
                floats: to_i64(inserted * u64::from(cfg.dim))?,
                bytes,
            },
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
        let ts = to_i64(now)?;
        let snapshot = match self.write_upsert(store, &plan, actor, ts) {
            Ok(s) => s,
            Err(e) => return self.poison(e),
        };
        if let Some((id, cfg)) = plan.init {
            self.identity = Some(id);
            self.config = Some(cfg);
        }
        let n = plan.rows.len() as u64;
        for r in plan.rows {
            let iid = self.iid_for(&r.id);
            self.slab.put(r.id, iid, &r.values, r.meta_text, r.filt);
        }
        self.write_seq += n;
        self.snapshot_seq = snapshot;
        Ok(UpsertOutcome {
            upserted: n,
            write_seq: self.write_seq,
            delta: plan.delta,
            dry_run: false,
        })
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
        let mut seq = self.write_seq;
        let mut next_iid = self.next_iid;
        let mut fresh = std::collections::BTreeMap::new();
        for r in &plan.rows {
            seq += 1;
            let iid = match self.slab.index.get(&r.id) {
                Some(&slot) => self.slab.iids[slot],
                None => *fresh.entry(r.id.as_str()).or_insert_with(|| {
                    next_iid += 1;
                    next_iid - 1
                }),
            };
            let body = encode_upsert_body(&r.values, r.meta_text.as_deref());
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

    /// Delete ids (absent ids are ignored). `dry_run` writes nothing.
    pub fn delete(
        &mut self,
        store: &dyn SqlStore,
        expected: &DoMeta,
        ids: &[String],
        actor: Actor<'_>,
        dry_run: bool,
        now: u64,
    ) -> Result<DeleteOutcome, OpError> {
        self.ensure_live()?;
        if ids.len() > MAX_DELETE_IDS {
            return Err(OpError::new(ErrorCode::PayloadTooLarge, "too many ids"));
        }
        let mut present = Vec::new();
        let mut seen = BTreeSet::new();
        for id in ids {
            let id = VectorId::parse(id)?.as_str().to_string();
            if seen.insert(id.clone()) && self.slab.index.contains_key(&id) {
                present.push(id);
            }
        }
        let initialized = self.check(expected, true)? == IdentityCheck::Matched;
        let dim = self.config.as_ref().map_or(0, |c| c.dim as usize);
        let mut delta = UsageDelta::default();
        for id in &present {
            let slot = self.slab.index[id];
            delta.vectors -= 1;
            delta.floats -= dim as i64;
            delta.bytes -= to_i64(self.slab.row_bytes(slot, dim))?;
        }
        if dry_run || present.is_empty() || !initialized {
            return Ok(DeleteOutcome {
                deleted: present.len() as u64,
                write_seq: self.write_seq,
                delta,
                dry_run,
            });
        }
        let ts = to_i64(now)?;
        let res = (|| -> Result<u64, StoreError> {
            let mut seq = self.write_seq;
            for id in &present {
                seq += 1;
                store.exec(
                    schema::OPS_APPEND,
                    &ops_row(seq, "delete", id, ts, actor, Value::Null)?,
                )?;
                store.exec(schema::VEC_DELETE, &[id.as_str().into()])?;
                store.exec(schema::FILTER_DELETE_ID, &[id.as_str().into()])?;
            }
            self.write_counters(store, seq, None)
        })();
        let snapshot = match res {
            Ok(s) => s,
            Err(e) => return self.poison(e),
        };
        for id in &present {
            self.slab.remove(id, dim);
        }
        self.write_seq += present.len() as u64;
        self.snapshot_seq = snapshot;
        Ok(DeleteOutcome {
            deleted: present.len() as u64,
            write_seq: self.write_seq,
            delta,
            dry_run: false,
        })
    }

    /// Erase all storage of this shard (collection drop: the DO is wiped
    /// before the catalog row is tombstoned). Returns the usage released.
    pub fn wipe(&mut self, store: &dyn SqlStore, expected: &DoMeta) -> Result<UsageDelta, OpError> {
        self.ensure_live()?;
        if self.check(expected, true)? != IdentityCheck::Matched {
            return Ok(UsageDelta::default());
        }
        let dim = self.config.as_ref().map_or(0, |c| c.dim as i64);
        let mut delta = UsageDelta::default();
        for slot in 0..self.slab.len() {
            delta.vectors -= 1;
            delta.floats -= dim;
            delta.bytes -= to_i64(self.slab.row_bytes(slot, dim as usize))?;
        }
        for sql in [
            schema::OPS_DELETE_ALL,
            schema::VEC_DELETE_ALL,
            schema::FILTER_DELETE_ALL,
            schema::META_DELETE_ALL,
        ] {
            if let Err(e) = store.exec(sql, &[]) {
                return self.poison(e);
            }
        }
        *self = VectorShard::default();
        Ok(delta)
    }
}

fn ops_row(
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
