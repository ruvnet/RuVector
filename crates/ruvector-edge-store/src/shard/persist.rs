//! Index persistence (ADR-351 §6.1 M2): `index_chunks(epoch, part, bytes)`
//! written by a streaming encoder (≤ 1 MiB rows, CRC-32C each, sha256 over
//! the payload in `meta.index_state`), and a lazy load on first use.
//!
//! **Epochs.** `meta.index_state` records the current epoch and, when it is
//! replay-compatible, the previous one: `(epoch, seq, parts, sha256)` where
//! `seq` is the `write_seq` the epoch reflects. A *rebase* (requantize,
//! dense renumbering, HNSW link repair, rebuild) changes the index in ways
//! the op log does not describe, so it drops the previous epoch.
//!
//! **Load order.** Decode the current epoch and replay the op tail past its
//! `seq` (ops carry their iid, codec v2); on any decode, digest or replay
//! failure try the previous epoch; else rebuild from `vectors`. Replaying
//! a decoded epoch reproduces the live index bit for bit (same codes, same
//! per-op HNSW levels), so a corrupted chunk changes no query result. The
//! op log is never pruned past the oldest retained epoch (`prune_floor`).
//!
//! **Writes.** Chunks first (after clearing the target epoch), then
//! `meta.index_state`, then old epochs are dropped: a torn flush leaves the
//! previous state valid.

use super::ann::Ann;
use super::codec::{decode_delete_body, decode_upsert_body, keys};
use super::{VectorShard, PAGE_ROWS, SHARD_RESIDENT_CAP_BYTES};
use crate::error::{ErrorCode, OpError};
use crate::ports::{col_int, col_text, SqlStore, StoreError, Value};
use crate::schema;
use ruvector_edge_index::{EmitError, IndexChunk};
use serde::{Deserialize, Serialize};

/// One persisted epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpochRec {
    /// `index_chunks.epoch`.
    pub epoch: u64,
    /// `write_seq` the epoch reflects.
    pub seq: u64,
    /// Number of parts.
    pub parts: u32,
    /// Payload sha256.
    #[serde(with = "hex32")]
    pub sha256: [u8; 32],
}

/// `meta.index_state`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexState {
    /// Latest epoch.
    pub cur: Option<EpochRec>,
    /// Previous epoch, only while `cur` is reachable from it by replay.
    pub prev: Option<EpochRec>,
}

mod hex32 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(b: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(b))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let s = String::deserialize(d)?;
        let v = hex::decode(s).map_err(serde::de::Error::custom)?;
        v.try_into()
            .map_err(|_| serde::de::Error::custom("sha256 length"))
    }
}

impl IndexState {
    /// Parse `meta.index_state` (absent → empty).
    pub fn from_meta(text: Option<&str>) -> Result<IndexState, StoreError> {
        text.map_or(Ok(IndexState::default()), |t| {
            serde_json::from_str(t).map_err(|_| StoreError::Corrupt("meta.index_state"))
        })
    }

    /// Highest epoch number ever recorded.
    pub fn max_epoch(&self) -> u64 {
        self.cur
            .map_or(0, |c| c.epoch)
            .max(self.prev.map_or(0, |p| p.epoch))
    }

    /// Oldest op sequence a fallback may need (`None`: no epoch retained).
    pub fn floor(&self) -> Option<u64> {
        self.prev.or(self.cur).map(|r| r.seq)
    }
}

/// Where the resident index came from on load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexSource {
    /// The current epoch plus the op tail.
    Current,
    /// The previous epoch plus the op tail (the current one was unusable).
    Previous,
    /// Rebuilt from `vectors` (no usable epoch).
    Rebuilt,
}

/// `index_chunks` rows of one epoch, pulled one per statement. A SQL error
/// ends the stream and is kept apart from decode failures.
struct ChunkRows<'a> {
    store: &'a dyn SqlStore,
    epoch: i64,
    next: i64,
    err: Option<StoreError>,
}

impl Iterator for ChunkRows<'_> {
    type Item = IndexChunk;
    fn next(&mut self) -> Option<IndexChunk> {
        let rows = match self.store.query(
            schema::CHUNK_GET,
            &[self.epoch.into(), self.next.into(), 1i64.into()],
        ) {
            Ok(r) => r,
            Err(e) => {
                self.err = Some(e);
                return None;
            }
        };
        let r = rows.first()?;
        let part = r.first().and_then(Value::as_int)?;
        let bytes = r.get(1).and_then(Value::as_blob)?.to_vec();
        self.next = part + 1;
        Some(IndexChunk {
            epoch: self.epoch as u64,
            part: u32::try_from(part).ok()?,
            bytes,
        })
    }
}

/// `StoreError::Corrupt` detail of an index the encoder refuses (an HNSW
/// iid gap): deterministic, so maintenance heals it by a dense rebuild
/// instead of poisoning the shard.
pub(crate) const ENCODE_REFUSED: &str = "index encode refused";

/// `true` for the deterministic encoder refusal.
pub(crate) fn is_encode_refused(e: &StoreError) -> bool {
    matches!(e, StoreError::Corrupt(m) if *m == ENCODE_REFUSED)
}

pub(crate) fn i(v: u64) -> Result<i64, StoreError> {
    i64::try_from(v).map_err(|_| StoreError::Corrupt("counter overflow"))
}

impl VectorShard {
    /// Load the index if it is not resident (the lazy first-use load):
    /// the current epoch plus its op tail, else the previous one, else a
    /// rebuild from `vectors`. A flat rebuild (quantize only) runs here; an
    /// HNSW rebuild (a full graph build, tens of CPU seconds at the shard
    /// cap) only when `allow_graph_build` — i.e. in the alarm. Otherwise the
    /// shard is marked `rebuild_pending` and left without an index: requests
    /// answer `503` without poisoning it, and maintenance rebuilds it.
    pub(crate) fn ensure_index_with(
        &mut self,
        store: &dyn SqlStore,
        allow_graph_build: bool,
    ) -> Result<(), StoreError> {
        if self.ann.is_some() || self.config.is_none() {
            return Ok(());
        }
        if self.quant.is_none() && self.slab.len() == 0 {
            return Ok(()); // trained from the first write batch
        }
        if self.quant.is_some() && !self.rebuild_pending {
            let st = self.persisted;
            for (rec, src) in [
                (st.cur, IndexSource::Current),
                (st.prev, IndexSource::Previous),
            ] {
                let Some(rec) = rec else { continue };
                if let Some(ann) = self.try_epoch(store, rec)? {
                    self.ann = Some(ann);
                    self.load_source = Some(src);
                    if src == IndexSource::Previous {
                        // Re-persist the current state; keep the good epoch.
                        self.flush(store, Some(rec))?;
                    }
                    return Ok(());
                }
            }
        }
        let hnsw = matches!(
            self.config.as_ref().map(|c| c.index),
            Some(super::codec::IndexConfig::Hnsw { .. })
        );
        if hnsw && !allow_graph_build {
            self.rebuild_pending = true;
            return Ok(());
        }
        if self.quant.is_none() {
            self.train_from_vectors(store, 1)?;
        }
        self.rebuild_from_vectors(store)?;
        self.rebuild_pending = false;
        self.load_source = Some(IndexSource::Rebuilt);
        self.flush_rebase(store)
    }

    /// [`Self::ensure_index_with`] for a request (no graph build).
    pub(crate) fn ensure_index(&mut self, store: &dyn SqlStore) -> Result<(), StoreError> {
        self.ensure_index_with(store, false)
    }

    /// Make the index resident for a request: a storage error poisons the
    /// shard; an HNSW index awaiting its alarm rebuild is `503
    /// shard_unavailable` **without** poisoning (a poisoned shard schedules
    /// no maintenance, so the rebuild would never run).
    pub(crate) fn index_for_request(&mut self, store: &dyn SqlStore) -> Result<(), OpError> {
        if let Err(e) = self.ensure_index(store) {
            return self.poison(e);
        }
        if self.rebuild_pending {
            return Err(OpError::new(
                ErrorCode::ShardUnavailable,
                "index rebuilding",
            ));
        }
        Ok(())
    }

    /// Decode `rec` and replay the op tail onto it; `Ok(None)` when the
    /// epoch or its tail is unusable (fall back), `Err` on a SQL failure.
    fn try_epoch(&self, store: &dyn SqlStore, rec: EpochRec) -> Result<Option<Ann>, StoreError> {
        let (Some(cfg), Some(q)) = (self.config.as_ref(), self.quant.as_ref()) else {
            return Ok(None);
        };
        if rec.seq > self.write_seq {
            return Ok(None);
        }
        let mut rows = ChunkRows {
            store,
            epoch: i(rec.epoch)?,
            next: 0,
            err: None,
        };
        let decoded = Ann::decode(cfg, &mut rows, &rec.sha256, SHARD_RESIDENT_CAP_BYTES);
        if let Some(e) = rows.err {
            return Err(e);
        }
        let Ok(mut ann) = decoded else {
            return Ok(None);
        };
        if *ann.quant() != q.params {
            return Ok(None);
        }
        match self.replay_tail(store, &mut ann, rec.seq) {
            Ok(()) => {}
            Err(StoreError::Corrupt(_)) => return Ok(None),
            Err(e) => return Err(e),
        }
        // The replayed index must hold exactly the slab's rows.
        let consistent =
            ann.live() == self.slab.len() && self.slab.iids.iter().all(|&iid| ann.contains(iid));
        Ok(consistent.then_some(ann))
    }

    /// Apply ops `(from, write_seq]` to `ann`. Any gap, an M1 body without
    /// an iid, or a pruned tail is `Corrupt` (the caller falls back).
    fn replay_tail(
        &self,
        store: &dyn SqlStore,
        ann: &mut Ann,
        from: u64,
    ) -> Result<(), StoreError> {
        let dim = self.config.as_ref().map_or(0, |c| c.dim as usize);
        let mut at = from;
        while at < self.write_seq {
            let page = store.query(schema::OPS_PAGE, &[i(at)?.into(), PAGE_ROWS.into()])?;
            if page.is_empty() {
                return Err(StoreError::Corrupt("op tail missing"));
            }
            for r in &page {
                let seq = u64::try_from(col_int(r, 0, "ops.seq")?)
                    .map_err(|_| StoreError::Corrupt("ops.seq"))?;
                if seq != at + 1 {
                    return Err(StoreError::Corrupt("op tail gap"));
                }
                if seq > self.write_seq {
                    return Ok(());
                }
                let body = r.get(3).and_then(Value::as_blob);
                match col_text(r, 1, "ops.op")?.as_str() {
                    "upsert" => {
                        let (iid, vals, _) =
                            decode_upsert_body(body.ok_or(StoreError::Corrupt("ops.body"))?, dim)?;
                        let iid = iid.ok_or(StoreError::Corrupt("v1 op body"))?;
                        ann.upsert(iid, &vals, Ann::op_seed(seq, self.salt))?;
                    }
                    "delete" => {
                        let iid =
                            decode_delete_body(body)?.ok_or(StoreError::Corrupt("v1 op body"))?;
                        ann.remove(iid);
                    }
                    _ => return Err(StoreError::Corrupt("ops.op")),
                }
                at = seq;
            }
        }
        Ok(())
    }

    /// Persist the resident index as a new epoch at `write_seq`. `prev` is
    /// the epoch to keep as fallback (`None` for a rebase).
    pub(crate) fn flush(
        &mut self,
        store: &dyn SqlStore,
        prev: Option<EpochRec>,
    ) -> Result<(), StoreError> {
        let Some(ann) = self.ann.as_ref() else {
            return Ok(());
        };
        let epoch = self.persisted.max_epoch().max(self.index_epoch) + 1;
        let ep = i(epoch)?;
        store.exec(schema::CHUNK_DELETE_FROM, &[ep.into()])?;
        let digest = ann
            .encode_into(epoch, &mut |c: IndexChunk| {
                store
                    .exec(
                        schema::CHUNK_PUT,
                        &[ep.into(), i64::from(c.part).into(), c.bytes.into()],
                    )
                    .map(|_| ())
            })
            .map_err(|e| match e {
                EmitError::Sink(e) => e,
                EmitError::Encode(_) => StoreError::Corrupt(ENCODE_REFUSED),
            })?;
        let state = IndexState {
            cur: Some(EpochRec {
                epoch,
                seq: self.write_seq,
                parts: digest.parts,
                sha256: digest.sha256,
            }),
            prev,
        };
        let text = serde_json::to_string(&state).map_err(|_| StoreError::Corrupt("state"))?;
        store.exec(schema::META_PUT, &[keys::INDEX_STATE.into(), text.into()])?;
        store.exec(
            schema::META_PUT,
            &[keys::INDEX_EPOCH.into(), epoch.to_string().into()],
        )?;
        let keep = prev.map_or(epoch, |p| p.epoch);
        store.exec(schema::CHUNK_DELETE_BELOW, &[i(keep)?.into()])?;
        self.persisted = state;
        self.index_epoch = epoch;
        Ok(())
    }

    /// Persist a rebase (the index changed in ways the op log does not
    /// describe) as **two** epochs with the same payload at the same `seq`:
    /// the older is the fallback, reached by a zero-length replay, so one
    /// corrupted chunk still loads bit-identically instead of forcing a
    /// rebuild (ADR §6.1 "corrupted chunk → identical results").
    pub(crate) fn flush_rebase(&mut self, store: &dyn SqlStore) -> Result<(), StoreError> {
        self.flush(store, None)?;
        let first = self.persisted.cur;
        self.flush(store, first)
    }

    /// Persist the op tail as a new epoch keeping the current one as the
    /// fallback (a shard's first epoch is written twice, as for a rebase).
    pub(crate) fn flush_keep(&mut self, store: &dyn SqlStore) -> Result<(), StoreError> {
        match self.persisted.cur {
            Some(cur) => self.flush(store, Some(cur)),
            None => self.flush_rebase(store),
        }
    }

    /// Ops logged since the latest persisted epoch.
    pub fn pending_ops(&self) -> u64 {
        self.write_seq
            .saturating_sub(self.persisted.cur.map_or(0, |c| c.seq))
    }

    /// Where the resident index was loaded from (`None` until loaded).
    pub fn index_source(&self) -> Option<IndexSource> {
        self.load_source
    }

    /// Persisted epochs.
    pub fn index_state(&self) -> IndexState {
        self.persisted
    }
}
