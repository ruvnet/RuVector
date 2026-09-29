//! Op-log replay (ADR-351 §6.1 cold start "replays newer `ops`").
//!
//! A write issues `ops` → `vectors` → `filter_idx` → `meta` back to back.
//! On a Durable Object a statement that throws does not roll back the
//! earlier ones, so a torn write can leave ops logged past
//! `meta.write_seq` whose `vectors` / `filter_idx` rows are missing or
//! stale. [`VectorShard::open`] therefore replays those ops
//! **write-through**: every replayed op re-issues its row statements, then
//! `meta.write_seq` / `next_iid` are advanced, so the durable tables agree
//! with the log before any new write is accepted. Re-issuing is idempotent
//! (`INSERT OR REPLACE`, delete-by-id).

use super::codec::{decode_upsert_body, encode_f32, keys, parse_meta};
use super::{VectorShard, PAGE_ROWS};
use crate::filter::{compact, index_rows};
use crate::ports::{col_int, col_text, SqlStore, StoreError, Value};
use crate::schema;

/// Op-log entries kept below `write_seq` (audit/debug tail). Older entries
/// are already reflected in `vectors` and are pruned in the write batch
/// that moves `write_seq` past them; `meta.snapshot_seq` records the cut.
pub const OPS_TAIL: u64 = 256;

impl VectorShard {
    /// Apply every op with `seq > write_seq`, in order, write-through (see
    /// the module docs). Returns the number applied. A gap in the sequence
    /// fails closed.
    pub fn catch_up(&mut self, store: &dyn SqlStore) -> Result<u64, StoreError> {
        let applied = self.replay(store, true)?;
        if applied > 0 {
            self.snapshot_seq = self.write_counters(store, self.write_seq, Some(self.next_iid))?;
        }
        Ok(applied)
    }

    /// Replay ops past `write_seq` into resident state; with
    /// `write_through`, also re-issue each op's row statements.
    pub(crate) fn replay(
        &mut self,
        store: &dyn SqlStore,
        write_through: bool,
    ) -> Result<u64, StoreError> {
        let Some(cfg) = self.config.clone() else {
            return Ok(0);
        };
        let dim = cfg.dim as usize;
        let mut applied = 0u64;
        loop {
            let after = i64::try_from(self.write_seq).map_err(|_| StoreError::Corrupt("seq"))?;
            let page = store.query(
                schema::OPS_PAGE,
                &[Value::Int(after), Value::Int(PAGE_ROWS)],
            )?;
            for r in &page {
                let seq = u64::try_from(col_int(r, 0, "ops.seq")?)
                    .map_err(|_| StoreError::Corrupt("ops.seq"))?;
                if seq != self.write_seq + 1 {
                    return Err(StoreError::Corrupt("op log gap"));
                }
                let op = col_text(r, 1, "ops.op")?;
                let id = col_text(r, 2, "ops.id")?;
                let ts = r.get(4).and_then(Value::as_int).unwrap_or(0);
                match op.as_str() {
                    "upsert" => {
                        let body = r
                            .get(3)
                            .and_then(Value::as_blob)
                            .ok_or(StoreError::Corrupt("ops.body"))?;
                        let (vals, meta) = decode_upsert_body(body, dim)?;
                        let parsed = meta.as_deref().map(parse_meta).transpose()?;
                        let iid = self.iid_for(&id);
                        if write_through {
                            let indexed = parsed
                                .as_ref()
                                .map(|m| index_rows(m, &cfg.filterable_keys))
                                .unwrap_or_default();
                            store.exec(
                                schema::VEC_PUT,
                                &[
                                    id.as_str().into(),
                                    iid.into(),
                                    encode_f32(&vals).into(),
                                    meta.clone().map_or(Value::Null, Value::Text),
                                    ts.into(),
                                    0i64.into(),
                                ],
                            )?;
                            store.exec(schema::FILTER_DELETE_ID, &[id.as_str().into()])?;
                            for (k, v) in indexed {
                                store.exec(
                                    schema::FILTER_PUT,
                                    &[k.into(), v.into(), id.as_str().into()],
                                )?;
                            }
                        }
                        let filt = compact(parsed.as_ref(), &cfg.filterable_keys);
                        self.slab.put(id, iid, &vals, meta, filt);
                    }
                    "delete" => {
                        if write_through {
                            store.exec(schema::VEC_DELETE, &[id.as_str().into()])?;
                            store.exec(schema::FILTER_DELETE_ID, &[id.as_str().into()])?;
                        }
                        self.slab.remove(&id, dim);
                    }
                    _ => return Err(StoreError::Corrupt("ops.op")),
                }
                self.write_seq = seq;
                applied += 1;
            }
            if (page.len() as i64) < PAGE_ROWS {
                return Ok(applied);
            }
        }
    }

    /// Persist `write_seq` (and `next_iid` when given), then prune the op
    /// log below the retained tail. Returns the new `snapshot_seq`. Issued
    /// last in every write batch, after the row statements it covers.
    pub(crate) fn write_counters(
        &self,
        store: &dyn SqlStore,
        seq: u64,
        next_iid: Option<i64>,
    ) -> Result<u64, StoreError> {
        store.exec(
            schema::META_PUT,
            &[keys::WRITE_SEQ.into(), seq.to_string().into()],
        )?;
        if let Some(n) = next_iid {
            store.exec(
                schema::META_PUT,
                &[keys::NEXT_IID.into(), n.to_string().into()],
            )?;
        }
        let cut = seq.saturating_sub(OPS_TAIL);
        if cut <= self.snapshot_seq {
            return Ok(self.snapshot_seq);
        }
        // Record the cut first: if the prune then fails, the log is merely
        // longer than necessary, and `rebuild_from_ops` still fails closed.
        store.exec(
            schema::META_PUT,
            &[keys::SNAPSHOT_SEQ.into(), cut.to_string().into()],
        )?;
        let cut_i = i64::try_from(cut).map_err(|_| StoreError::Corrupt("seq overflow"))?;
        store.exec(schema::OPS_PRUNE, &[cut_i.into()])?;
        Ok(cut)
    }
}
