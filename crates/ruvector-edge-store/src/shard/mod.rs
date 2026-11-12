//! `VectorShard` (ADR-351 §6.1): one Durable Object per `(tenant,
//! collection_uid, shard)`.
//!
//! M2 layout: f32 rows live only in SQLite (`vectors.f32`); what is
//! resident is the row bookkeeping ([`slab`]) and the index ([`ann`]):
//! int8 codes for `flat` (the default: quantized scan, then an exact f32
//! rerank of the best candidates fetched from SQLite by iid batch) or codes
//! plus HNSW links for `hnsw`. The index is persisted in `index_chunks`
//! ([`persist`]) and loaded lazily on first use, falling back to an older
//! epoch plus the op tail, or to a rebuild from `vectors`, on corruption.
//!
//! The struct holds only the resident state; every call takes the
//! [`SqlStore`] of its DO and the [`DoMeta`] the caller expects, and asserts
//! that identity first (a mismatch is `404 not_found`, §4.3). Writes follow
//! the §6.1 write path: validate fully in memory (and plan the quota delta
//! and resident growth), then issue every statement back to back, then
//! mutate the slab and index, then flush the index if ≥ [`FLUSH_OPS`] ops
//! are unpersisted (so a cold load replays fewer than 200 ops).
//!
//! Row state can be rebuilt two ways, both of which must equal the live
//! state ([`VectorShard::state_digest`]): a paged cold load of `vectors`
//! plus a write-through replay of any op logged past `meta.write_seq`
//! ([`VectorShard::open`]), and — while the op log is unpruned — a replay
//! of the `ops` log alone ([`VectorShard::rebuild_from_ops`]).
//!
//! Resident memory is accounted per row plus the index's allocated bytes
//! and capped per shard at [`SHARD_RESIDENT_CAP_BYTES`].

pub mod ann;
pub mod codec;
mod delete;
pub mod maintain;
pub mod persist;
mod read;
mod rebuild;
mod replay;
mod rerank;
pub mod slab;
mod write;

pub use ann::{QuantState, IID_BASE, MAX_SLOTS};
pub use codec::{IndexConfig, ShardConfig};
pub use delete::{DeleteOutcome, MAX_DELETE_IDS};
pub use maintain::{Due, MaintainReport, FLUSH_AFTER_MS, FLUSH_OPS};
pub use persist::{EpochRec, IndexSource, IndexState};
pub use read::{
    validate_query, Match, QueryOutcome, QueryRequest, ValidQuery, DEFAULT_RERANK_MIN,
    EF_DEFAULT_COSINE, EF_DEFAULT_L2_DOT, MAX_EF, MAX_FETCH_IDS, MAX_QUERY_STEPS, MAX_RERANK,
};
pub use replay::OPS_TAIL;
pub use write::{
    UpsertOutcome, UpsertPlan, UpsertRow, UsageDelta, HNSW_SYNC_UPSERT, M2_SHARD_FLOAT_CAP,
};

use crate::error::OpError;
use crate::filter::compact;
use crate::ports::{col_int, col_text, read_kv, SqlStore, StoreError, Value};
use crate::resident::ISOLATE_RESIDENT_CAP_BYTES;
use crate::schema;
use ann::Ann;
use codec::{counter, keys, parse_meta};
use ruvector_edge_tenancy::{DoMeta, IdentityCheck};
use sha2::{Digest, Sha256};
use slab::Slab;

/// Rows per page for cold load and replay (ADR §6.1: `LIMIT 1024`).
pub const PAGE_ROWS: i64 = 1024;

/// Per-shard resident byte cap (index codes/links plus row bookkeeping).
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
    /// `act.sub` of an exchanged token: the adapter that acted for `sub`
    /// (ADR-351 §5.6, §16.3); `None` when the user acted directly.
    pub act_sub: Option<&'a str>,
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
    /// Resident index (`None` until the lazy load, or before training).
    pub(crate) ann: Option<Ann>,
    /// Quantizer (`None` for an l2/dot shard not yet trained).
    pub(crate) quant: Option<QuantState>,
    /// `meta.index_state`.
    pub(crate) persisted: IndexState,
    /// HNSW level salt, derived from the DO identity.
    pub(crate) salt: u64,
    pub(crate) load_source: Option<IndexSource>,
    /// A post-write flush failed; maintenance retries it.
    pub(crate) flush_failed: bool,
    /// An HNSW index with no usable epoch, left for the alarm to rebuild
    /// (requests answer `503` meanwhile, see `persist`).
    pub(crate) rebuild_pending: bool,
    /// Sliced HNSW link repair in progress: the next node to repair
    /// (resident only; an evicted shard restarts the idempotent pass).
    pub(crate) repair_cursor: Option<u32>,
    /// The collection was dropped: every call is `404` (`meta.wiped`).
    pub(crate) wiped: bool,
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
            ann: None,
            quant: None,
            persisted: IndexState::default(),
            salt: 0,
            load_source: None,
            flush_failed: false,
            rebuild_pending: false,
            repair_cursor: None,
            wiped: false,
        }
    }
}

/// HNSW level salt of a shard identity.
pub(crate) fn salt_of(id: &DoMeta) -> u64 {
    let d = Sha256::digest(id.do_name().as_str().as_bytes());
    u64::from_le_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])
}

impl VectorShard {
    /// Create the schema if needed and cold-load the shard's rows: `meta`,
    /// then `vectors` paged by `iid` without the f32 column (never a
    /// full-table read), then a write-through replay of ops past
    /// `meta.write_seq`. The index loads lazily on first use.
    pub fn open(store: &dyn SqlStore) -> Result<Self, StoreError> {
        let mut shard = Self::open_meta(store)?;
        let Some(cfg) = shard.config.clone() else {
            return Ok(shard);
        };
        let dim = cfg.dim as usize;
        let mut after = 0i64;
        loop {
            let page = store.query(
                schema::VEC_PAGE_META,
                &[Value::Int(after), Value::Int(PAGE_ROWS)],
            )?;
            for r in &page {
                let id = col_text(r, 0, "vectors.id")?;
                let iid = col_int(r, 1, "vectors.iid")?;
                let meta = r.get(2).and_then(Value::as_text);
                let parsed = meta.map(parse_meta).transpose()?;
                let filt = compact(parsed.as_ref(), &cfg.filterable_keys);
                shard.slab.put(id, iid, dim, meta.map_or(0, str::len), filt);
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

    /// Rebuild row state from `meta` plus the `ops` log only, ignoring
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
        let get = |k: &str| kv.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str());
        let quant = match (&config, get(keys::QUANT)) {
            (Some(c), Some(t)) => Some(QuantState::from_meta(t, c)?),
            (Some(c), None) => QuantState::fixed(c),
            (None, _) => None,
        };
        Ok(VectorShard {
            salt: identity.as_ref().map_or(0, salt_of),
            identity,
            config,
            write_seq: counter(&kv, keys::WRITE_SEQ, 0)?,
            next_iid,
            index_epoch: counter(&kv, keys::INDEX_EPOCH, 0)?,
            snapshot_seq: counter(&kv, keys::SNAPSHOT_SEQ, 0)?,
            quant,
            persisted: IndexState::from_meta(get(keys::INDEX_STATE))?,
            wiped: get(keys::WIPED).is_some(),
            ..VectorShard::default()
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
        if self.wiped {
            // A dropped collection's DO never re-initialises (§7.2).
            return Err(OpError::not_found());
        }
        Ok(DoMeta::check(self.identity.as_ref(), expected, is_write)?)
    }

    /// Load the index now (normally lazy, on first query or write). An
    /// HNSW index with no usable epoch is not rebuilt here (`503`, not
    /// poisoned; the maintenance alarm rebuilds it).
    pub fn load_index(&mut self, store: &dyn SqlStore) -> Result<(), OpError> {
        self.ensure_live()?;
        self.index_for_request(store)
    }

    /// `true` while an HNSW index awaits its maintenance rebuild.
    pub fn rebuild_pending(&self) -> bool {
        self.rebuild_pending
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
    /// Latest persisted index epoch (`0` before the first flush).
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
    /// Quantizer in use (`None` for an untrained l2/dot shard).
    pub fn quant(&self) -> Option<&QuantState> {
        self.quant.as_ref()
    }
    /// `true` once the index is resident.
    pub fn index_loaded(&self) -> bool {
        self.ann.is_some()
    }
    /// Resident bytes of the index alone (0 until loaded).
    pub fn index_bytes(&self) -> u64 {
        self.ann.as_ref().map_or(0, Ann::memory_bytes)
    }
    /// Resident bytes: row bookkeeping, the iid map and the index (for the
    /// per-shard cap and the isolate registry).
    pub fn resident_bytes(&self) -> u64 {
        self.slab.resident + self.slab.map_bytes() + self.index_bytes()
    }

    /// Tenant usage this shard accounts for (all non-negative): vectors,
    /// floats and stored bytes. Used to reconcile the ledger after a torn
    /// write, when the applied delta is unknown.
    pub fn usage_totals(&self) -> UsageDelta {
        let dim = self.config.as_ref().map_or(0, |c| u64::from(c.dim));
        UsageDelta {
            vectors: i64::try_from(self.slab.len()).unwrap_or(i64::MAX),
            floats: i64::try_from(self.slab.len() as u64 * dim).unwrap_or(i64::MAX),
            bytes: i64::try_from(self.slab.stored).unwrap_or(i64::MAX),
        }
    }

    /// Order-independent digest of the logical row state: identity,
    /// config, `write_seq`, `next_iid`, and every row `(id, iid, metadata
    /// length, filter form)` sorted by id. Vector values are not resident at
    /// M2 (they are compared through query results instead).
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
        for (id, &slot) in &self.slab.index {
            h.update((id.len() as u64).to_le_bytes());
            h.update(id.as_bytes());
            h.update(self.slab.iids[slot].to_le_bytes());
            h.update(self.slab.meta_len[slot].to_le_bytes());
            h.update(format!("{:?}", self.slab.filt[slot]).as_bytes());
        }
        h.finalize().into()
    }
}
