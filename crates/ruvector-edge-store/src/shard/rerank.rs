//! Exact f32 rerank and fetch from SQLite (ADR-351 §6.1 "f32 rerank from
//! SQLite"): candidate rows are read `IID_BATCH` at a time by iid (`IN`
//! list, unused slots bound to iid 0), scored with the exact `f64`
//! distance, and only the best `top_k` keep their values/metadata.
//!
//! Every returned row is cross-checked against the resident slab (same id
//! at that iid): a row SQLite still holds for an id the shard no longer
//! has, or at a different iid, is ignored.

use super::codec::{decode_f32, parse_meta, ShardConfig};
use super::read::{Match, ValidQuery};
use super::slab::Slab;
use crate::distance::{distance, norm, to_wire};
use crate::ports::{col_int, col_text, SqlStore, StoreError, Value};
use crate::schema;
use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap};

struct Ranked {
    d: f64,
    id: String,
    values: Option<Vec<f32>>,
    meta: Option<String>,
}
impl PartialEq for Ranked {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Ranked {
    fn cmp(&self, o: &Self) -> Ordering {
        self.d.total_cmp(&o.d).then_with(|| self.id.cmp(&o.id))
    }
}

fn meta_json(text: Option<String>) -> Option<serde_json::Value> {
    text.and_then(|t| parse_meta(&t).ok())
        .map(serde_json::Value::Object)
}

/// Rerank `cands` (iids) exactly and return the best `k` by `(d, id)`.
pub(crate) fn exact_top_k(
    store: &dyn SqlStore,
    slab: &Slab,
    cfg: &ShardConfig,
    q: &[f32],
    v: &ValidQuery,
    cands: &[i64],
    k: usize,
) -> Result<Vec<Match>, StoreError> {
    let dim = cfg.dim as usize;
    let mut uniq: Vec<i64> = cands
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    uniq.retain(|&i| i > 0);
    let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(k + 1);
    for batch in uniq.chunks(schema::IID_BATCH) {
        let mut params: Vec<Value> = batch.iter().map(|&i| Value::Int(i)).collect();
        params.resize(schema::IID_BATCH, Value::Int(0));
        for r in store.query(schema::VEC_BY_IIDS, &params)? {
            let iid = col_int(&r, 0, "vectors.iid")?;
            let id = col_text(&r, 1, "vectors.id")?;
            if slab.slot_of(iid).map(|s| slab.ids[s].as_str()) != Some(id.as_str()) {
                continue;
            }
            let blob = r
                .get(2)
                .and_then(Value::as_blob)
                .ok_or(StoreError::Corrupt("vectors.f32"))?;
            let vals = decode_f32(blob, dim)?;
            let d = distance(cfg.metric, q, v.q_norm, &vals, norm(&vals));
            let worse = heap.len() >= k
                && heap
                    .peek()
                    .is_some_and(|t| d.total_cmp(&t.d).then_with(|| id.cmp(&t.id)).is_ge());
            if worse {
                continue;
            }
            let meta = if v.want_meta {
                r.get(3).and_then(Value::as_text).map(str::to_string)
            } else {
                None
            };
            heap.push(Ranked {
                d,
                id,
                values: v.want_values.then_some(vals),
                meta,
            });
            if heap.len() > k {
                heap.pop();
            }
        }
    }
    Ok(heap
        .into_sorted_vec()
        .into_iter()
        .map(|c| Match {
            distance: to_wire(c.d),
            metadata: meta_json(c.meta),
            values: c.values,
            score: c.d,
            id: c.id,
        })
        .collect())
}

/// Read `ids` (all resident) by id, in the given order.
pub(crate) fn fetch_ids(
    store: &dyn SqlStore,
    slab: &Slab,
    dim: usize,
    ids: &[&String],
    include_values: bool,
) -> Result<Vec<Match>, StoreError> {
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        let rows = store.query(schema::VEC_BY_ID, &[id.as_str().into()])?;
        let Some(r) = rows.first() else { continue };
        let iid = col_int(r, 1, "vectors.iid")?;
        if slab.slot_of(iid).map(|s| &slab.ids[s]) != Some(*id) {
            continue;
        }
        let values = if include_values {
            let blob = r
                .get(2)
                .and_then(Value::as_blob)
                .ok_or(StoreError::Corrupt("vectors.f32"))?;
            Some(decode_f32(blob, dim)?)
        } else {
            None
        };
        out.push(Match {
            id: (*id).clone(),
            distance: 0.0,
            metadata: meta_json(r.get(3).and_then(Value::as_text).map(str::to_string)),
            values,
            score: 0.0,
        });
    }
    Ok(out)
}
