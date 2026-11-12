//! `QuantShard` writes (ADR-351 §3 rv-quant): upsert planning (validate,
//! de-duplicate, report the usage delta), apply (codes first, all-or-nothing
//! under the quant budgets, then rows) and delete, with the same wire
//! contract as `VectorShard` (a write never exceeds what the ledger admitted;
//! a storage failure after codes changed is an unknown effect).

use crate::quant_load as ql;
use crate::quant_shard::QuantHost;
use crate::quant_store::{self as qs, QMeta};
use crate::wire::{CfgWire, DeltaWire, ShardOut, WireErr};
use ruvector_edge_store::filter::MAX_METADATA_BYTES;
use ruvector_edge_store::{ErrorCode, IndexConfig, OpError, SqlStore, UpsertRow};
use ruvector_edge_tenancy::{DoMeta, VectorId};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// The shard's rotation seed: derived from its identity, so a rebuild or a
/// reload regenerates the same rotation.
pub(crate) fn seed_of(dm: &DoMeta) -> u64 {
    let d = Sha256::digest(dm.do_name().as_str().as_bytes());
    u64::from_le_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])
}

pub(crate) fn check_cfg(cfg: &CfgWire, meta: &QMeta) -> Result<(), OpError> {
    if cfg.index != IndexConfig::Rabitq {
        return Err(OpError::invalid("not a rabitq collection"));
    }
    if meta.ident.is_some() && (cfg.dim != meta.dim || Some(cfg.metric) != meta.metric) {
        return Err(OpError::new(ErrorCode::ServerError, "quant shard config"));
    }
    Ok(())
}

/// A validated, de-duplicated upsert.
pub(crate) struct Plan {
    pub(crate) rows: Vec<(UpsertRow, Option<String>, Option<qs::RowRef>)>,
    pub(crate) delta: DeltaWire,
}

pub(crate) fn plan(
    store: &dyn SqlStore,
    cfg: &CfgWire,
    rows: Vec<UpsertRow>,
) -> Result<Plan, OpError> {
    let mut last: BTreeMap<String, usize> = BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        VectorId::parse(&r.id)?;
        if r.values.len() != cfg.dim as usize {
            return Err(OpError::new(
                ErrorCode::DimensionMismatch,
                "dimension mismatch",
            ));
        }
        if r.values.iter().any(|x| !x.is_finite()) {
            return Err(OpError::new(ErrorCode::NonFiniteValue, "non-finite value"));
        }
        last.insert(r.id.clone(), i);
    }
    let ids: Vec<&str> = last.keys().map(String::as_str).collect();
    let existing = qs::refs_by_ids(store, &ids, cfg.dim)?;
    let mut delta = DeltaWire::default();
    let mut out = Vec::with_capacity(last.len());
    for (i, r) in rows.into_iter().enumerate() {
        if last.get(&r.id) != Some(&i) {
            continue;
        }
        let md = match &r.metadata {
            None => None,
            Some(m @ serde_json::Value::Object(_)) => Some(m.to_string()),
            Some(_) => return Err(OpError::invalid("metadata must be an object")),
        };
        if md.as_ref().is_some_and(|m| m.len() > MAX_METADATA_BYTES) {
            return Err(OpError::new(
                ErrorCode::PayloadTooLarge,
                "metadata too large",
            ));
        }
        let bytes = qs::row_bytes(&r.id, cfg.dim, md.as_deref()) as i64;
        let old = existing.get(&r.id).cloned();
        match &old {
            Some(o) => delta.bytes += bytes - o.bytes as i64,
            None => {
                delta.vectors += 1;
                delta.floats += i64::from(cfg.dim);
                delta.bytes += bytes;
            }
        }
        out.push((r, md, old));
    }
    Ok(Plan { rows: out, delta })
}

pub(crate) fn written(count: u64, write_seq: u64, delta: DeltaWire) -> ShardOut {
    ShardOut::Written {
        count,
        write_seq,
        delta,
    }
}

/// Codes first (all-or-nothing, every budget checked before anything is
/// written), then the rows; a storage failure after that drops the
/// resident state and reports an unknown effect (over-count, never under).
pub(crate) fn apply(
    host: &mut QuantHost,
    key: &str,
    store: &dyn SqlStore,
    before: &QMeta,
    mut after: QMeta,
    p: Plan,
) -> Result<ShardOut, OpError> {
    // The row cap is checked from the counters before any load: a shard
    // never grows past what one turn can cold-load (`413`, not a shard
    // that is CPU-killed on every later open).
    let new_rows = p.rows.iter().filter(|(_, _, old)| old.is_none()).count() as u64;
    if before.count.saturating_add(new_rows) > host.budget.max_vectors {
        return Err(OpError::new(
            ErrorCode::BudgetExceeded,
            "quant shard row limit",
        ));
    }
    let r = host.ready(key, store, &after)?;
    let wseq = after.write_seq + 1;
    let mut keyed = Vec::with_capacity(p.rows.len());
    for (_, _, old) in &p.rows {
        let rk = match old {
            Some(o) => o.rk,
            None => {
                after.next_key += 1;
                after.next_key
            }
        };
        keyed.push(rk);
    }
    let codes: Vec<(u64, &[f32])> = keyed
        .iter()
        .zip(&p.rows)
        .map(|(k, (row, _, _))| (*k, row.values.as_slice()))
        .collect();
    r.q.upsert(&codes).map_err(ql::qerr)?;
    r.dirty_rows += codes.len() as u64;
    let res = (|| -> Result<(), OpError> {
        for (rk, (row, md, _)) in keyed.iter().zip(&p.rows) {
            qs::put_row(store, *rk, &row.id, &row.values, md.as_deref(), wseq)?;
        }
        after.write_seq = wseq;
        after.count = (after.count as i64 + p.delta.vectors).max(0) as u64;
        after.bytes = (after.bytes as i64 + p.delta.bytes).max(0) as u64;
        after.write(store, before)
    })();
    match res {
        Ok(()) => Ok(written(p.rows.len() as u64, wseq, p.delta)),
        Err(e) => {
            host.evict(key);
            Ok(ShardOut::Failed {
                err: WireErr::from_op(&e),
                applied: None,
            })
        }
    }
}

pub(crate) fn delete(
    host: &mut QuantHost,
    key: &str,
    store: &dyn SqlStore,
    meta: &QMeta,
    ids: &[String],
    dry_run: bool,
) -> Result<ShardOut, OpError> {
    let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let found = qs::refs_by_ids(store, &refs, meta.dim)?;
    let n = found.len() as u64;
    let delta = DeltaWire {
        vectors: -(n as i64),
        floats: -((n * u64::from(meta.dim)) as i64),
        bytes: -(found.values().map(|r| r.bytes as i64).sum::<i64>()),
    };
    if dry_run || n == 0 {
        return Ok(written(n, meta.write_seq, delta));
    }
    let r = host.ready(key, store, meta)?;
    let rks: Vec<u64> = found.values().map(|r| r.rk).collect();
    r.q.delete(&rks);
    r.dirty_rows += n;
    let mut after = meta.clone();
    let dseq = meta.write_seq + 1;
    let res = (|| -> Result<(), OpError> {
        for rk in &rks {
            qs::delete_row(store, *rk, dseq)?;
        }
        after.write_seq = dseq;
        after.count = after.count.saturating_sub(n);
        after.bytes = (after.bytes as i64 + delta.bytes).max(0) as u64;
        after.write(store, meta)
    })();
    match res {
        Ok(()) => Ok(written(n, dseq, delta)),
        Err(e) => {
            host.evict(key);
            Ok(ShardOut::Failed {
                err: WireErr::from_op(&e),
                applied: None,
            })
        }
    }
}
