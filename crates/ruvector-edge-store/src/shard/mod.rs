//! `VectorShard` (ADR-351 §6.1): one Durable Object per `(tenant,
//! collection_uid, shard)`, M1 index kind `flat` (f32 slab, exact scan).
//!
//! The struct holds only the resident state; every call takes the
//! [`SqlStore`] of its DO and the [`DoMeta`] the caller expects, and asserts
//! that identity first (a mismatch is `404 not_found`, §4.3). Writes follow
//! the §6.1 write path: validate fully in memory (and plan the quota delta),
//! then issue every statement back to back, then mutate the slab.
//!
//! State can be rebuilt two ways, both of which must equal the live state
//! ([`VectorShard::state_digest`]): a paged cold load of `vectors` plus a
//! write-through replay of any op logged past `meta.write_seq`
//! ([`VectorShard::open`]), and — while the op log is unpruned — a replay
//! of the `ops` log alone ([`VectorShard::rebuild_from_ops`]).
//!
//! Resident memory is accounted per row ([`slab::row_resident`]) and capped
//! per shard at [`SHARD_RESIDENT_CAP_BYTES`], so the isolate registry sees
//! real usage.

pub mod codec;
mod read;
mod replay;
pub mod slab;
mod write;

pub use codec::ShardConfig;
pub use read::{
    validate_query, Match, QueryOutcome, QueryRequest, ValidQuery, MAX_FETCH_IDS, MAX_QUERY_STEPS,
};
pub use replay::OPS_TAIL;
pub use write::{DeleteOutcome, UpsertOutcome, UpsertPlan, UpsertRow, UsageDelta, MAX_DELETE_IDS};

use crate::error::OpError;
use crate::filter::compact;
use crate::ports::{col_int, col_text, read_kv, SqlStore, StoreError, Value};
use crate::resident::ISOLATE_RESIDENT_CAP_BYTES;
use crate::schema;
use codec::{counter, decode_f32, keys, parse_meta};
use ruvector_edge_tenancy::{DoMeta, IdentityCheck};
use sha2::{Digest, Sha256};
use slab::Slab;

/// Rows per page for cold load and replay (ADR §6.1: `LIMIT 1024`).
pub const PAGE_ROWS: i64 = 1024;

/// Per-shard resident byte cap (floats, ids, metadata, per-row overhead).
/// Four fit the isolate cap, matching the §6.1 "3–4 shards fit" plan.
pub const SHARD_RESIDENT_CAP_BYTES: u64 = 14_000_000;

const _: () = assert!(SHARD_RESIDENT_CAP_BYTES <= ISOLATE_RESIDENT_CAP_BYTES);

/// Who performed a write (recorded in `ops`).
#[derive(Debug, Clone, Copy)]
pub struct Actor<'a> {
    /// Edge subject.
    pub sub: &'a str,
    /// Token id.
    pub jti: &'a str,
    /// Grant family.
    pub family_id: &'a str,
}

/// Resident state of one `VectorShard`.
#[derive(Debug, Clone)]
pub struct VectorShard {
    pub(crate) identity: Option<DoMeta>,
    pub(crate) config: Option<ShardConfig>,
    pub(crate) slab: Slab,
    pub(crate) write_seq: u64,
    pub(crate) next_iid: i64,
    pub(crate) index_epoch: u64,
    pub(crate) snapshot_seq: u64,
    pub(crate) poisoned: bool,
}

impl Default for VectorShard {
    fn default() -> Self {
        VectorShard {
            identity: None,
            config: None,
            slab: Slab::default(),
            write_seq: 0,
            next_iid: 1,
            index_epoch: 0,
            snapshot_seq: 0,
            poisoned: false,
        }
    }
}

impl VectorShard {
    /// Create the schema if needed and cold-load the shard: `meta`, then
    /// `vectors` paged by `iid` (never a full-table read), then a
    /// write-through replay of ops past `meta.write_seq`.
    pub fn open(store: &dyn SqlStore) -> Result<Self, StoreError> {
        let mut shard = Self::open_meta(store)?;
        let Some(cfg) = shard.config.clone() else {
            return Ok(shard);
        };
        let dim = cfg.dim as usize;
        let mut after = 0i64;
        loop {
            let page = store.query(
                schema::VEC_PAGE,
                &[Value::Int(after), Value::Int(PAGE_ROWS)],
            )?;
            for r in &page {
                let id = col_text(r, 0, "vectors.id")?;
                let iid = col_int(r, 1, "vectors.iid")?;
                let blob = r
                    .get(2)
                    .and_then(Value::as_blob)
                    .ok_or(StoreError::Corrupt("vectors.f32"))?;
                let vals = decode_f32(blob, dim)?;
                let meta = r.get(3).and_then(Value::as_text).map(str::to_string);
                let parsed = meta.as_deref().map(parse_meta).transpose()?;
                let filt = compact(parsed.as_ref(), &cfg.filterable_keys);
                shard.slab.put(id, iid, &vals, meta, filt);
                after = iid;
            }
            if (page.len() as i64) < PAGE_ROWS {
                break;
            }
        }
        // Never hand out an iid already on disk, even if `meta.next_iid`
        // lagged the rows (a torn write outside a coalesced commit).
        shard.next_iid = shard.next_iid.max(after.saturating_add(1));
        shard.catch_up(store)?;
        Ok(shard)
    }

    /// Rebuild resident state from `meta` plus the `ops` log only, ignoring
    /// `vectors` and writing nothing. Must equal [`VectorShard::open`] and
    /// the live state. Fails closed once the log has been pruned
    /// (`snapshot_seq > 0`): the pruned prefix lives only in `vectors`.
    pub fn rebuild_from_ops(store: &dyn SqlStore) -> Result<Self, StoreError> {
        let mut shard = Self::open_meta(store)?;
        if shard.snapshot_seq > 0 {
            return Err(StoreError::Corrupt("op log pruned below snapshot_seq"));
        }
        shard.write_seq = 0;
        shard.next_iid = 1;
        shard.replay(store, false)?;
        Ok(shard)
    }

    fn open_meta(store: &dyn SqlStore) -> Result<Self, StoreError> {
        for ddl in schema::SHARD_SCHEMA {
            store.exec(ddl, &[])?;
        }
        let kv = read_kv(store, schema::META_SELECT_ALL)?;
        let identity = DoMeta::from_kv(kv.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .map_err(|_| StoreError::Corrupt("meta identity"))?;
        let config = ShardConfig::from_kv(&kv)?;
        if identity.is_some() != config.is_some() {
            return Err(StoreError::Corrupt("meta identity/config"));
        }
        let next_iid = i64::try_from(counter(&kv, keys::NEXT_IID, 1)?)
            .map_err(|_| StoreError::Corrupt("meta.next_iid"))?;
        Ok(VectorShard {
            identity,
            config,
            slab: Slab::default(),
            write_seq: counter(&kv, keys::WRITE_SEQ, 0)?,
            next_iid,
            index_epoch: counter(&kv, keys::INDEX_EPOCH, 0)?,
            snapshot_seq: counter(&kv, keys::SNAPSHOT_SEQ, 0)?,
            poisoned: false,
        })
    }

    /// Existing iid for `id`, or allocate the next one.
    pub(crate) fn iid_for(&mut self, id: &str) -> i64 {
        match self.slab.index.get(id) {
            Some(&slot) => self.slab.iids[slot],
            None => {
                let iid = self.next_iid;
                self.next_iid += 1;
                iid
            }
        }
    }

    /// Assert the caller's identity (§4.3): mismatch → `404`.
    pub(crate) fn check(
        &self,
        expected: &DoMeta,
        is_write: bool,
    ) -> Result<IdentityCheck, OpError> {
        Ok(DoMeta::check(self.identity.as_ref(), expected, is_write)?)
    }

    /// Live vector count.
    pub fn len(&self) -> usize {
        self.slab.len()
    }
    /// `true` after a storage error: reopen before further use.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// `true` when no vectors are resident.
    pub fn is_empty(&self) -> bool {
        self.slab.len() == 0
    }
    /// Last applied op sequence (`write_seq`).
    pub fn write_seq(&self) -> u64 {
        self.write_seq
    }
    /// Highest pruned op sequence (`0` while the log is complete).
    pub fn snapshot_seq(&self) -> u64 {
        self.snapshot_seq
    }
    /// Index epoch (M2b chunk generation; `0` for `flat`).
    pub fn index_epoch(&self) -> u64 {
        self.index_epoch
    }
    /// Stored configuration, `None` for an uninitialised shard.
    pub fn config(&self) -> Option<&ShardConfig> {
        self.config.as_ref()
    }
    /// Stored identity, `None` for an uninitialised shard.
    pub fn identity(&self) -> Option<&DoMeta> {
        self.identity.as_ref()
    }
    /// Resident floats (`len × dim`), the unit of the per-shard float cap.
    pub fn resident_floats(&self) -> u64 {
        self.slab.values.len() as u64
    }
    /// Resident bytes of everything the shard holds per row (for the
    /// per-shard cap and the isolate registry).
    pub fn resident_bytes(&self) -> u64 {
        self.slab.resident
    }

    /// Tenant usage this shard accounts for (all non-negative): vectors,
    /// floats and stored bytes. Used to reconcile the ledger after a torn
    /// write, when the applied delta is unknown.
    pub fn usage_totals(&self) -> UsageDelta {
        UsageDelta {
            vectors: i64::try_from(self.slab.len()).unwrap_or(i64::MAX),
            floats: i64::try_from(self.resident_floats()).unwrap_or(i64::MAX),
            bytes: i64::try_from(self.slab.stored).unwrap_or(i64::MAX),
        }
    }

    /// Order-independent digest of the logical state: identity, config,
    /// `write_seq`, `next_iid`, and every row `(id, iid, f32 bits, metadata)`
    /// sorted by id.
    pub fn state_digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        if let Some(id) = &self.identity {
            for (k, v) in id.to_kv() {
                h.update(k.as_bytes());
                h.update(v.as_bytes());
            }
        }
        if let Some(c) = &self.config {
            for (k, v) in c.to_kv() {
                h.update(k.as_bytes());
                h.update(v.as_bytes());
            }
        }
        h.update(self.write_seq.to_le_bytes());
        h.update(self.next_iid.to_le_bytes());
        let dim = self.config.as_ref().map_or(0, |c| c.dim as usize);
        for (id, &slot) in &self.slab.index {
            h.update((id.len() as u64).to_le_bytes());
            h.update(id.as_bytes());
            h.update(self.slab.iids[slot].to_le_bytes());
            for v in self.slab.row(slot, dim) {
                h.update(v.to_bits().to_le_bytes());
            }
            let m = self.slab.meta_text[slot].as_deref().unwrap_or("\u{0}none");
            h.update((m.len() as u64).to_le_bytes());
            h.update(m.as_bytes());
        }
        h.finalize().into()
    }
}
