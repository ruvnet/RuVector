//! Rebases of the resident index (ADR-351 §6.1): a rebuild from `vectors`,
//! quantizer (re)training, and dense iid renumbering. Each changes the index
//! in ways the op log does not describe, so each drops the persisted epochs
//! and is followed by a rebase flush ([`VectorShard::flush_rebase`]).

use super::ann::{Ann, QuantState};
use super::codec::{decode_f32, keys};
use super::persist::IndexState;
use super::VectorShard;
use crate::ports::{col_int, SqlStore, StoreError, Value};
use crate::schema;

/// Rows per page of a rebuild read (f32 only: at 1536-d about 1.5 MB).
pub const REBUILD_PAGE_ROWS: i64 = 256;

impl VectorShard {
    /// Build the index from `vectors` in iid order. HNSW first renumbers
    /// densely whenever its iid space is not exactly `1..=len` with
    /// `next_iid = len + 1`: its encoder refuses gaps, and a `next_iid`
    /// left above `len + 1` (the newest rows deleted before the rebuild)
    /// would open one at the next insert. Lowering `next_iid` is safe: the
    /// rebuild is followed by a rebase flush and `catch_up` replays only
    /// ops past `meta.write_seq`.
    pub(crate) fn rebuild_from_vectors(&mut self, store: &dyn SqlStore) -> Result<(), StoreError> {
        let (Some(cfg), Some(q)) = (self.config.clone(), self.quant.clone()) else {
            return Ok(());
        };
        let hnsw = matches!(cfg.index, super::codec::IndexConfig::Hnsw { .. });
        let len = self.slab.len() as i64;
        if hnsw && (self.slab.max_iid() != len || self.next_iid != len + 1) {
            let dense: Vec<(i64, i64)> = {
                let mut v = self.slab.iids.clone();
                v.sort_unstable();
                v.into_iter().zip(1..).collect()
            };
            self.renumber(store, &dense)?;
        }
        let dim = cfg.dim as usize;
        let mut ann = Ann::new(&cfg, &q, self.slab.len())?;
        let mut after = 0i64;
        loop {
            let page = store.query(
                schema::VEC_PAGE_F32,
                &[after.into(), REBUILD_PAGE_ROWS.into()],
            )?;
            for r in &page {
                let iid = col_int(r, 0, "vectors.iid")?;
                let blob = r
                    .get(1)
                    .and_then(Value::as_blob)
                    .ok_or(StoreError::Corrupt("vectors.f32"))?;
                ann.upsert(
                    iid,
                    &decode_f32(blob, dim)?,
                    Ann::rebuild_seed(iid, self.salt),
                )?;
                after = iid;
            }
            if (page.len() as i64) < REBUILD_PAGE_ROWS {
                break;
            }
        }
        if ann.live() != self.slab.len() {
            return Err(StoreError::Corrupt("vectors/slab disagree"));
        }
        self.ann = Some(ann);
        Ok(())
    }

    /// Train (or retrain) the quantizer from an evenly strided sample of
    /// `vectors` (≤ `TRAIN_ROWS`, fetched by iid: the rest of the shard is
    /// never read), persist it, and drop every epoch (codes of the old
    /// quantizer can no longer be replayed onto).
    pub(crate) fn train_from_vectors(
        &mut self,
        store: &dyn SqlStore,
        epoch: u64,
    ) -> Result<(), StoreError> {
        let Some(cfg) = self.config.clone() else {
            return Ok(());
        };
        let dim = cfg.dim as usize;
        let mut iids = self.slab.iids.clone();
        iids.sort_unstable();
        let stride = (iids.len() / super::ann::TRAIN_ROWS).max(1);
        let picked: Vec<i64> = iids
            .into_iter()
            .step_by(stride)
            .take(super::ann::TRAIN_ROWS)
            .collect();
        let mut rows: Vec<(i64, Vec<f32>)> = Vec::with_capacity(picked.len());
        for batch in picked.chunks(schema::IID_BATCH) {
            let mut params: Vec<Value> = batch.iter().map(|&i| Value::Int(i)).collect();
            params.resize(schema::IID_BATCH, Value::Int(0));
            for r in store.query(schema::VEC_F32_BY_IIDS, &params)? {
                let iid = col_int(&r, 0, "vectors.iid")?;
                let blob = r.get(1).and_then(Value::as_blob);
                rows.push((
                    iid,
                    decode_f32(blob.ok_or(StoreError::Corrupt("f32"))?, dim)?,
                ));
            }
        }
        if rows.is_empty() {
            return Ok(());
        }
        // Iid order, whatever order the `IN` lists came back in.
        rows.sort_unstable_by_key(|(iid, _)| *iid);
        let sample: Vec<f32> = rows.into_iter().flat_map(|(_, v)| v).collect();
        let q = QuantState::train(&cfg, &sample, epoch)?;
        self.invalidate_epochs(store)?;
        store.exec(schema::META_PUT, &[keys::QUANT.into(), q.to_meta()?.into()])?;
        self.quant = Some(q);
        Ok(())
    }

    /// Forget every persisted epoch (before a renumber or requantize, so a
    /// torn rebase can only fall back to a rebuild).
    pub(crate) fn invalidate_epochs(&mut self, store: &dyn SqlStore) -> Result<(), StoreError> {
        let empty = IndexState {
            cur: None,
            prev: None,
        };
        let text = serde_json::to_string(&empty).map_err(|_| StoreError::Corrupt("state"))?;
        store.exec(schema::META_PUT, &[keys::INDEX_STATE.into(), text.into()])?;
        self.persisted = empty;
        Ok(())
    }

    /// Rewrite `vectors.iid` (ascending `(old, new)` pairs, `new ≤ old`, so
    /// no two rows ever share an iid) and the slab; `next_iid` follows.
    pub(crate) fn renumber(
        &mut self,
        store: &dyn SqlStore,
        pairs: &[(i64, i64)],
    ) -> Result<(), StoreError> {
        self.invalidate_epochs(store)?;
        for &(old, new) in pairs {
            if old == new {
                continue;
            }
            let slot = self
                .slab
                .slot_of(old)
                .ok_or(StoreError::Corrupt("renumber"))?;
            store.exec(
                schema::VEC_SET_IID,
                &[new.into(), self.slab.ids[slot].as_str().into()],
            )?;
        }
        let map: std::collections::BTreeMap<i64, i64> = pairs.iter().copied().collect();
        self.slab.renumber(|old| map.get(&old).copied());
        self.next_iid = self.slab.max_iid() + 1;
        store.exec(
            schema::META_PUT,
            &[keys::NEXT_IID.into(), self.next_iid.to_string().into()],
        )?;
        Ok(())
    }
}
