//! The resident M2 index of a shard (ADR-351 §6.1): `flat` (int8 codes,
//! quantized scan) or `hnsw` (graph over the same codes), wrapping
//! `ruvector-edge-index`, plus the quantizer policy.
//!
//! **Quantizer.** Cosine uses the fixed range `[-1, 1]` (no training, epoch
//! 1 forever). l2/dot train per-dimension ranges from the shard's first
//! write batch and retrain once from a ≤ [`TRAIN_ROWS`] sample of `vectors`
//! when the shard has doubled past its sample (a rebase, see
//! `shard::maintain`). Trained parameters are persisted in `meta.quant`,
//! so a rebuild after chunk loss reproduces bit-identical codes.
//!
//! **Iids.** The store allocates iids from 1 and HNSW is built with
//! `iid_base: 1`, so slot 0 is never paid for and the first flush is dense.
//! HNSW levels come from `SplitMix64(seq ^ salt)` per op, so replaying the
//! op tail onto a decoded epoch rebuilds a bit-identical graph.

use super::codec::{IndexConfig, ShardConfig};
use crate::distance::Metric;
use crate::ports::StoreError;
use ruvector_edge_index::{
    memory, DecodeError, EmitError, HnswIndex, HnswParams, IndexChunk, IndexDigest, IndexError,
    Metric as IMetric, QuantFlatIndex, QuantKind, QuantParams, SplitMix64,
};
use serde::{Deserialize, Serialize};

/// First iid the store allocates (HNSW `iid_base`).
pub const IID_BASE: u32 = 1;
/// Slot cap of both index kinds: `max_iid + 1 ≤ MAX_SLOTS`. `2^20 × 1536`
/// still fits a wasm32 `isize`.
pub const MAX_SLOTS: u32 = 1 << 20;
/// Largest quantizer training sample (the ADR's 1k reservoir).
pub const TRAIN_ROWS: usize = 1000;
/// Tag mixed into the level seed of a from-`vectors` rebuild (not an op).
const REBUILD_TAG: u64 = 0x5245_4255_494c_4421;

pub(crate) fn imetric(m: Metric) -> IMetric {
    match m {
        Metric::Cosine => IMetric::Cosine,
        Metric::L2 => IMetric::L2,
        Metric::Dot => IMetric::Dot,
    }
}

fn ix(e: IndexError) -> StoreError {
    match e {
        IndexError::CapacityExceeded { .. } => StoreError::Backend("index slot cap".into()),
        _ => StoreError::Corrupt("index rejected a stored vector"),
    }
}

/// Quantizer parameters plus the sample size they were trained on (`0`
/// for a fixed range).
#[derive(Debug, Clone, PartialEq)]
pub struct QuantState {
    /// Parameters.
    pub params: QuantParams,
    /// Training rows (`0`: fixed range, never retrained).
    pub rows: u64,
}

#[derive(Serialize, Deserialize)]
struct QuantMeta {
    epoch: u64,
    rows: u64,
    /// Hex of a one-chunk empty flat index carrying the parameters (the
    /// index crate's own checksummed serialisation), absent when fixed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    blob: Option<String>,
}

impl QuantState {
    /// The fixed cosine quantizer (`None` for l2/dot, which train).
    pub fn fixed(cfg: &ShardConfig) -> Option<QuantState> {
        (cfg.metric == Metric::Cosine)
            .then(|| QuantParams::cosine_fixed(cfg.dim as usize, 1).ok())
            .flatten()
            .map(|params| QuantState { params, rows: 0 })
    }

    /// Train from row-major `samples` (all validated, finite).
    pub fn train(cfg: &ShardConfig, samples: &[f32], epoch: u64) -> Result<QuantState, StoreError> {
        if let Some(q) = Self::fixed(cfg) {
            return Ok(q);
        }
        let dim = cfg.dim as usize;
        let params = QuantParams::train(imetric(cfg.metric), dim, samples, epoch)
            .map_err(|_| StoreError::Corrupt("quantizer training"))?;
        Ok(QuantState {
            params,
            rows: (samples.len() / dim.max(1)) as u64,
        })
    }

    /// `meta.quant` text.
    pub fn to_meta(&self) -> Result<String, StoreError> {
        let blob = if self.params.kind() == QuantKind::FixedRange {
            None
        } else {
            let carrier = QuantFlatIndex::new(self.params.clone(), 1)
                .and_then(|f| {
                    f.to_chunks(0, ruvector_edge_index::MAX_CHUNK_BYTES)
                        .map_err(|_| IndexError::InvalidParams("encode"))
                })
                .map_err(|_| StoreError::Corrupt("quant encode"))?;
            let c = carrier
                .chunks
                .first()
                .ok_or(StoreError::Corrupt("quant encode"))?;
            Some(hex::encode(&c.bytes))
        };
        serde_json::to_string(&QuantMeta {
            epoch: self.params.epoch(),
            rows: self.rows,
            blob,
        })
        .map_err(|_| StoreError::Corrupt("quant encode"))
    }

    /// Inverse of [`QuantState::to_meta`], checked against the config.
    pub fn from_meta(text: &str, cfg: &ShardConfig) -> Result<QuantState, StoreError> {
        let bad = || StoreError::Corrupt("meta.quant");
        let m: QuantMeta = serde_json::from_str(text).map_err(|_| bad())?;
        let q = match m.blob {
            None => Self::fixed(cfg).ok_or_else(bad)?,
            Some(h) => {
                let bytes = hex::decode(h).map_err(|_| bad())?;
                let chunk = IndexChunk {
                    epoch: 0,
                    part: 0,
                    bytes,
                };
                let f = QuantFlatIndex::from_chunks(&[chunk], None).map_err(|_| bad())?;
                QuantState {
                    params: f.quant().clone(),
                    rows: m.rows,
                }
            }
        };
        if q.params.dim() != cfg.dim as usize
            || q.params.metric() != imetric(cfg.metric)
            || q.params.epoch() != m.epoch
        {
            return Err(bad());
        }
        Ok(q)
    }
}

/// The resident index.
#[derive(Debug, Clone)]
pub enum Ann {
    /// Quantized flat scan.
    Flat(QuantFlatIndex),
    /// HNSW over codes.
    Hnsw(HnswIndex),
}

/// HNSW parameters for a config (`m0 = 2m`, `iid_base = 1`).
pub fn hnsw_params(cfg: &ShardConfig) -> HnswParams {
    let (m, efc) = match cfg.index {
        IndexConfig::Hnsw { m, ef_construction } => (m, ef_construction),
        IndexConfig::Flat | IndexConfig::Rabitq => (16, 128),
    };
    HnswParams {
        m,
        m0: 2 * m,
        ef_construction: efc,
        max_level: 16,
        max_slots: MAX_SLOTS - IID_BASE,
        iid_base: IID_BASE,
    }
}

/// Expected resident bytes of an index with `slots` slots (`max_iid + 1`).
pub fn estimate_bytes(cfg: &ShardConfig, slots: u64) -> u64 {
    let (n, d) = (slots as usize, cfg.dim as usize);
    let b = match cfg.index {
        IndexConfig::Flat | IndexConfig::Rabitq => memory::flat_estimate_bytes(n, d),
        IndexConfig::Hnsw { .. } => memory::hnsw_estimate_bytes(n, d, &hnsw_params(cfg)),
    };
    b as u64
}

pub(crate) fn iid32(iid: i64) -> Result<u32, StoreError> {
    u32::try_from(iid)
        .ok()
        .filter(|i| (IID_BASE..MAX_SLOTS).contains(i))
        .ok_or(StoreError::Corrupt("iid out of index range"))
}

impl Ann {
    /// Empty index with room for `capacity` slots.
    pub fn new(cfg: &ShardConfig, quant: &QuantState, capacity: usize) -> Result<Ann, StoreError> {
        let q = quant.params.clone();
        Ok(match cfg.index {
            IndexConfig::Flat | IndexConfig::Rabitq => {
                Ann::Flat(QuantFlatIndex::with_capacity(q, MAX_SLOTS, capacity).map_err(ix)?)
            }
            IndexConfig::Hnsw { .. } => {
                Ann::Hnsw(HnswIndex::with_capacity(hnsw_params(cfg), q, capacity).map_err(ix)?)
            }
        })
    }

    /// Quantizer in use.
    pub fn quant(&self) -> &QuantParams {
        match self {
            Ann::Flat(f) => f.quant(),
            Ann::Hnsw(h) => h.quant(),
        }
    }

    /// Insert or replace `iid`. HNSW draws a level (only for a new node)
    /// from `SplitMix64(seed)`.
    pub fn upsert(&mut self, iid: i64, v: &[f32], seed: u64) -> Result<(), StoreError> {
        let iid = iid32(iid)?;
        match self {
            Ann::Flat(f) => f.upsert(iid, v).map_err(ix),
            Ann::Hnsw(h) => h.upsert(iid, v, &mut SplitMix64::new(seed)).map_err(ix),
        }
    }

    /// Level seed of op `seq` (ADR §6.1 `splitmix64(seq ^ salt)`).
    pub fn op_seed(seq: u64, salt: u64) -> u64 {
        SplitMix64::mix(seq ^ salt)
    }

    /// Level seed of a from-`vectors` rebuild row.
    pub fn rebuild_seed(iid: i64, salt: u64) -> u64 {
        SplitMix64::mix((iid as u64) ^ salt ^ REBUILD_TAG)
    }

    /// Remove (flat) / tombstone (HNSW) `iid`.
    pub fn remove(&mut self, iid: i64) -> bool {
        let Ok(iid) = iid32(iid) else { return false };
        match self {
            Ann::Flat(f) => f.remove(iid),
            Ann::Hnsw(h) => h.delete(iid),
        }
    }

    /// Whether `iid` is live.
    pub fn contains(&self, iid: i64) -> bool {
        let Ok(iid) = iid32(iid) else { return false };
        match self {
            Ann::Flat(f) => f.contains(iid),
            Ann::Hnsw(h) => h.contains(iid),
        }
    }

    /// Live vectors.
    pub fn live(&self) -> usize {
        match self {
            Ann::Flat(f) => f.len(),
            Ann::Hnsw(h) => h.len(),
        }
    }

    /// Slots paid for.
    pub fn slots(&self) -> u64 {
        match self {
            Ann::Flat(f) => u64::from(f.slots()),
            Ann::Hnsw(h) => u64::from(h.node_count()) + u64::from(IID_BASE),
        }
    }

    /// Dead share (flat: empty slots; HNSW: tombstones).
    pub fn dead_ratio(&self) -> f64 {
        match self {
            Ann::Flat(f) => f.dead_ratio(),
            Ann::Hnsw(h) => h.dead_ratio(),
        }
    }

    /// Resident bytes (capacity).
    pub fn memory_bytes(&self) -> u64 {
        (match self {
            Ann::Flat(f) => f.memory_bytes(),
            Ann::Hnsw(h) => h.memory_bytes(),
        }) as u64
    }

    /// Pre-size for `n` more slots (exact, so accounting stays honest).
    pub fn reserve(&mut self, n: usize) {
        match self {
            Ann::Flat(f) => f.reserve(n),
            Ann::Hnsw(h) => h.reserve(n),
        }
    }

    /// Release spare capacity (after a compaction).
    pub fn shrink_to_fit(&mut self) {
        match self {
            Ann::Flat(f) => f.shrink_to_fit(),
            Ann::Hnsw(h) => h.shrink_to_fit(),
        }
    }

    /// Stream into `index_chunks` rows.
    pub fn encode_into<E>(
        &self,
        epoch: u64,
        emit: &mut dyn FnMut(IndexChunk) -> Result<(), E>,
    ) -> Result<IndexDigest, EmitError<E>> {
        let max = ruvector_edge_index::MAX_CHUNK_BYTES;
        match self {
            Ann::Flat(f) => f.encode_into(epoch, max, emit),
            Ann::Hnsw(h) => h.encode_into(epoch, max, emit),
        }
    }

    /// Streaming decode of one epoch.
    pub fn decode<I: IntoIterator<Item = IndexChunk>>(
        cfg: &ShardConfig,
        chunks: I,
        sha256: &[u8; 32],
        max_payload: u64,
    ) -> Result<Ann, DecodeError> {
        Ok(match cfg.index {
            IndexConfig::Flat | IndexConfig::Rabitq => Ann::Flat(QuantFlatIndex::from_chunk_iter(
                chunks,
                Some(sha256),
                max_payload,
            )?),
            IndexConfig::Hnsw { .. } => Ann::Hnsw(HnswIndex::from_chunk_iter(
                chunks,
                Some(sha256),
                max_payload,
            )?),
        })
    }

    /// Dense renumbering (flat: drop empty slots; HNSW: purge tombstones,
    /// after [`Ann::repair_links`] has run to completion). Returns `old iid
    /// → new iid` as a function table indexed by old iid (`u32::MAX`:
    /// dropped).
    pub fn compact(&mut self) -> Result<Vec<u32>, StoreError> {
        match self {
            Ann::Flat(f) => f.compact_ids(IID_BASE).map_err(ix),
            Ann::Hnsw(h) => {
                let by_slot = h.compact(true);
                // Re-index by old iid (slot + base) for the caller.
                let mut map = vec![u32::MAX; by_slot.len() + IID_BASE as usize];
                for (s, &new) in by_slot.iter().enumerate() {
                    map[s + IID_BASE as usize] = new;
                }
                Ok(map)
            }
        }
    }

    /// One slice of HNSW link repair around tombstones: up to `max_nodes`
    /// from `cursor`; the next cursor, `None` once the pass is complete
    /// (flat: always complete). Slices tolerate interleaved writes.
    pub fn repair_links(&mut self, cursor: u32, max_nodes: u32) -> Option<u32> {
        match self {
            Ann::Flat(_) => None,
            Ann::Hnsw(h) => {
                let next = h.repair_links(cursor, max_nodes);
                (next < h.node_count()).then_some(next)
            }
        }
    }
}

/// Baseline HNSW insert (and link-repair) cost per node, wasm32 in V8:
/// 26k × 384, m 16, ef_construction 128 builds in 29.0 s (≈ 1.1 ms/node).
const NODE_MS_BASE: f64 = 1.1;

/// Modelled wasm cost (ms) of one HNSW insert or repair for `cfg`: the
/// baseline scaled by `√(dim/384) · √(m·efc / (16·128))` — conservative
/// against the second measured point (6.6k × 1536, m 48, efc 200: 26.3 s,
/// ≈ 4.0 ms/node; the model gives ≈ 4.8). `0` for flat.
pub fn hnsw_node_ms(cfg: &ShardConfig) -> f64 {
    match cfg.index {
        IndexConfig::Flat | IndexConfig::Rabitq => 0.0,
        IndexConfig::Hnsw { m, ef_construction } => {
            let d = f64::from(cfg.dim) / 384.0;
            let w = f64::from(m) * f64::from(ef_construction) / 2048.0;
            NODE_MS_BASE * d.sqrt() * w.sqrt()
        }
    }
}

/// CPU budget of one alarm slice of link repair (ms, wasm).
pub const REPAIR_SLICE_MS: f64 = 2000.0;

/// Nodes per link-repair slice for `cfg` (≈ [`REPAIR_SLICE_MS`]).
pub fn repair_slice_nodes(cfg: &ShardConfig) -> u32 {
    let ms = hnsw_node_ms(cfg).max(0.01);
    ((REPAIR_SLICE_MS / ms) as u32).clamp(64, 16_384)
}
