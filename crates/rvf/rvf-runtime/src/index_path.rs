//! HNSW index integration for the runtime query path.
//!
//! Wires the `rvf-index` HNSW implementation into `RvfStore` queries:
//!
//! - **Build strategy**: the index is built lazily on the first eligible
//!   query (so ingest-only workloads pay nothing up front), then maintained
//!   incrementally as new vectors are ingested.
//! - **Persistence**: on `close()`, a dirty index is encoded with the
//!   existing `rvf-index` INDEX_SEG codec and appended as an INDEX_SEG.
//!   The codec stores adjacency for dense ordinals `0..n`; a versioned,
//!   self-delimiting trailer (ignored by readers that only parse the
//!   leading codec bytes) maps ordinals back to the store's sparse vector
//!   IDs and records the entry point, so no change to the codec's wire
//!   layout is needed.
//! - **Load**: on open, the most recent INDEX_SEG is decoded and validated
//!   against the loaded vectors. A stale or corrupt index is discarded and
//!   rebuilt from vectors on the next eligible query.
//!
//! Queries fall back to the exact brute-force scan when the store is small
//! (below [`INDEX_MIN_VECTORS`], where a scan is faster), when a metadata
//! filter or COW membership filter applies, when too many vectors are
//! soft-deleted, or when the index cannot supply `k` live results.

use std::collections::{HashMap, HashSet};

use rvf_index::codec::{decode_varint, encode_varint, NodeAdjacency, DEFAULT_RESTART_INTERVAL};
use rvf_index::{
    cosine_distance, decode_index_seg, dot_product, encode_index_seg, l2_distance, HnswConfig,
    HnswGraph, HnswLayer, IndexSegData, IndexSegHeader, VectorStore,
};

use crate::options::DistanceMetric;
use crate::read_path::VectorData;

/// Minimum number of stored vectors before the HNSW index path activates.
/// Below this, a brute-force scan is faster than graph traversal.
pub(crate) const INDEX_MIN_VECTORS: usize = 1024;

/// Maximum fraction of soft-deleted vectors tolerated by the index path.
/// Above this, oversampling becomes unreliable and the exact scan is used.
pub(crate) const INDEX_MAX_DELETED_FRACTION: f64 = 0.25;

/// Floor applied to `ef_search` on the index path. Measured on 10k random
/// 128-dim vectors (fixed seed, M=16, ef_construction=200): recall@10 is
/// 0.89 at ef=100, 0.96 at ef=200, and 0.97 at ef=256. The floor keeps the
/// production path above the >=0.95 recall@10 contract; callers passing a
/// larger `ef_search` are still honored.
pub(crate) const INDEX_MIN_EF_SEARCH: usize = 256;

/// Trailer magic ("RVIX") marking the ID-mapping extension appended after
/// the standard INDEX_SEG codec payload.
const TRAILER_MAGIC: [u8; 4] = *b"RVIX";

/// Trailer format version. Bump on incompatible trailer layout changes;
/// unknown versions are treated as a stale index (safe rebuild).
const TRAILER_VERSION: u16 = 1;

/// Adapter exposing the runtime's `VectorData` through the `rvf-index`
/// `VectorStore` trait.
struct VecDataStore<'a> {
    data: &'a VectorData,
}

impl VectorStore for VecDataStore<'_> {
    fn get_vector(&self, id: u64) -> Option<&[f32]> {
        self.data.get(id)
    }

    fn dimension(&self) -> usize {
        self.data.dimension as usize
    }
}

/// Distance function matching the store's `compute_distance` semantics
/// exactly: squared L2, negated dot product, and `1 - cosine_similarity`.
fn metric_distance_fn(metric: DistanceMetric) -> fn(&[f32], &[f32]) -> f32 {
    match metric {
        DistanceMetric::L2 => l2_distance,
        DistanceMetric::InnerProduct => dot_product,
        DistanceMetric::Cosine => cosine_distance,
    }
}

/// Deterministic per-ID level randomness (SplitMix64 mixed to uniform (0,1)).
///
/// Seeding level selection from the vector ID makes graph construction
/// reproducible regardless of HashMap iteration order or process restarts.
fn level_rng(id: u64) -> f64 {
    let mut z = id.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (((z >> 11) as f64) / (1u64 << 53) as f64).clamp(1e-12, 1.0 - 1e-12)
}

/// An HNSW graph over the store's vector IDs, plus bookkeeping for
/// incremental maintenance and persistence.
pub(crate) struct VectorIndex {
    graph: HnswGraph,
    /// IDs currently present in the graph (always a subset of the store's
    /// vector IDs).
    members: HashSet<u64>,
    /// True when the in-memory graph differs from the last persisted state.
    dirty: bool,
}

impl VectorIndex {
    /// Build a fresh index over every vector currently in the store.
    ///
    /// IDs are inserted in ascending order for deterministic construction.
    pub(crate) fn build(
        vectors: &VectorData,
        metric: DistanceMetric,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        let config = HnswConfig {
            m,
            m0: m * 2,
            ef_construction,
        };
        let mut index = Self {
            graph: HnswGraph::new(&config),
            members: HashSet::with_capacity(vectors.len()),
            dirty: true,
        };
        let mut ids: Vec<u64> = vectors.ids().copied().collect();
        ids.sort_unstable();
        index.insert_ids(&ids, vectors, metric);
        index
    }

    /// Insert the given IDs (pre-sorted ascending) into the graph,
    /// skipping IDs that are already indexed.
    pub(crate) fn insert_ids(
        &mut self,
        sorted_ids: &[u64],
        vectors: &VectorData,
        metric: DistanceMetric,
    ) {
        let store = VecDataStore { data: vectors };
        let dist = metric_distance_fn(metric);
        for &id in sorted_ids {
            if self.members.contains(&id) {
                continue;
            }
            self.graph.insert(id, level_rng(id), &store, &dist);
            self.members.insert(id);
            self.dirty = true;
        }
    }

    /// Insert any store vectors that are not yet in the graph (e.g. vectors
    /// ingested after the persisted index was written).
    pub(crate) fn sync_missing(&mut self, vectors: &VectorData, metric: DistanceMetric) {
        // Members is always a subset of the store's IDs, so equal sizes
        // mean equal sets and the O(N) scan below can be skipped.
        if self.members.len() >= vectors.len() {
            return;
        }
        let mut missing: Vec<u64> = vectors
            .ids()
            .filter(|id| !self.members.contains(id))
            .copied()
            .collect();
        missing.sort_unstable();
        self.insert_ids(&missing, vectors, metric);
    }

    /// Search the graph for the `k` nearest neighbors of `query`.
    /// Results are sorted by `(distance, id)` ascending.
    pub(crate) fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        vectors: &VectorData,
        metric: DistanceMetric,
    ) -> Vec<(u64, f32)> {
        let store = VecDataStore { data: vectors };
        let dist = metric_distance_fn(metric);
        self.graph.search(query, k, ef_search, &store, &dist)
    }

    /// Returns true if `id` is present in the graph.
    pub(crate) fn contains(&self, id: u64) -> bool {
        self.members.contains(&id)
    }

    /// Number of indexed vectors.
    pub(crate) fn node_count(&self) -> usize {
        self.members.len()
    }

    /// True when the graph has changed since the last persist/load.
    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark the graph as persisted.
    pub(crate) fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Encode the index as an INDEX_SEG payload.
    ///
    /// Layout: standard `rvf-index` codec payload (adjacency over dense
    /// ordinals in ascending-ID order), followed by a self-delimiting
    /// trailer read from the end of the payload:
    ///
    /// ```text
    /// [codec payload]
    /// [version: u16][m0: u16][max_layer: u32][entry_ordinal: u64]
    /// [id_count: u64][delta-varint sorted IDs]
    /// [trailer_body_len: u32]["RVIX"]
    /// ```
    pub(crate) fn encode_payload(&self) -> Vec<u8> {
        let mut ids: Vec<u64> = self.members.iter().copied().collect();
        ids.sort_unstable();
        let ordinal: HashMap<u64, u64> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i as u64))
            .collect();

        let nodes: Vec<NodeAdjacency> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let mut layers = Vec::new();
                for layer in &self.graph.layers {
                    if !layer.contains(id) {
                        break;
                    }
                    let neighbors: Vec<u64> = layer
                        .neighbors(id)
                        .iter()
                        .filter_map(|n| ordinal.get(n).copied())
                        .collect();
                    layers.push(neighbors);
                }
                NodeAdjacency {
                    node_id: i as u64,
                    layers,
                }
            })
            .collect();

        let data = IndexSegData {
            header: IndexSegHeader {
                index_type: 0,  // HNSW
                layer_level: 2, // Layer C: full adjacency
                m: self.graph.m as u16,
                ef_construction: self.graph.ef_construction as u32,
                node_count: ids.len() as u64,
            },
            restart_interval: DEFAULT_RESTART_INTERVAL,
            nodes,
        };
        let mut payload = encode_index_seg(&data);

        // Versioned ID-mapping trailer (see method docs).
        let mut body = Vec::with_capacity(24 + ids.len() * 2);
        body.extend_from_slice(&TRAILER_VERSION.to_le_bytes());
        body.extend_from_slice(&(self.graph.m0.min(u16::MAX as usize) as u16).to_le_bytes());
        body.extend_from_slice(&(self.graph.max_layer as u32).to_le_bytes());
        let entry_ordinal = self
            .graph
            .entry_point
            .and_then(|ep| ordinal.get(&ep).copied())
            .unwrap_or(u64::MAX);
        body.extend_from_slice(&entry_ordinal.to_le_bytes());
        body.extend_from_slice(&(ids.len() as u64).to_le_bytes());
        let mut prev = 0u64;
        for (i, &id) in ids.iter().enumerate() {
            let delta = if i == 0 { id } else { id - prev };
            encode_varint(delta, &mut body);
            prev = id;
        }

        let body_len = body.len() as u32;
        payload.extend_from_slice(&body);
        payload.extend_from_slice(&body_len.to_le_bytes());
        payload.extend_from_slice(&TRAILER_MAGIC);
        payload
    }

    /// Decode an INDEX_SEG payload and validate it against the store's
    /// current vectors.
    ///
    /// Returns `None` when the payload has no (or an unknown-version)
    /// trailer, is corrupt, or is stale (references an ID that no longer
    /// exists in `vectors`). Callers treat `None` as "rebuild from vectors".
    pub(crate) fn decode_payload(payload: &[u8], vectors: &VectorData) -> Option<Self> {
        // 1. Locate and parse the trailer from the end of the payload.
        if payload.len() < 8 || payload[payload.len() - 4..] != TRAILER_MAGIC {
            return None;
        }
        let len_off = payload.len() - 8;
        let body_len = u32::from_le_bytes(payload[len_off..len_off + 4].try_into().ok()?) as usize;
        let body_start = len_off.checked_sub(body_len)?;
        let body = &payload[body_start..len_off];
        if body.len() < 24 {
            return None;
        }
        let version = u16::from_le_bytes([body[0], body[1]]);
        if version != TRAILER_VERSION {
            return None;
        }
        let m0 = u16::from_le_bytes([body[2], body[3]]) as usize;
        let max_layer = u32::from_le_bytes(body[4..8].try_into().ok()?) as usize;
        let entry_ordinal = u64::from_le_bytes(body[8..16].try_into().ok()?);
        let id_count_raw = u64::from_le_bytes(body[16..24].try_into().ok()?);

        // Bound `id_count` by the bytes actually available before
        // allocating: each delta occupies at least one varint byte, so a
        // count exceeding the remaining body length can never decode. A
        // crafted trailer with e.g. id_count = u64::MAX would otherwise
        // panic (capacity overflow) or OOM in `Vec::with_capacity`.
        if id_count_raw > (body.len() - 24) as u64 {
            return None;
        }
        let id_count = id_count_raw as usize;

        let mut ids = Vec::with_capacity(id_count);
        let mut pos = 24;
        let mut current = 0u64;
        for i in 0..id_count {
            let (delta, consumed) = decode_varint(&body[pos..])?;
            pos += consumed;
            current = if i == 0 {
                delta
            } else {
                current.checked_add(delta)?
            };
            ids.push(current);
        }

        // 2. Decode the codec payload (graph adjacency over ordinals).
        let data = decode_index_seg(payload).ok()?;
        if data.header.index_type != 0 {
            return None;
        }
        if data.header.node_count as usize != id_count || data.nodes.len() != id_count {
            return None;
        }

        // 3. Staleness check: every indexed ID must still exist in the
        //    store (soft-deleted vectors remain present until compaction).
        for &id in &ids {
            vectors.get(id)?;
        }

        // 4. Rebuild the graph in vector-ID space.
        let m = (data.header.m as usize).max(2);
        let config = HnswConfig {
            m,
            m0: m0.max(m),
            ef_construction: (data.header.ef_construction as usize).max(1),
        };
        let mut graph = HnswGraph::new(&config);
        let layer_count = data
            .nodes
            .iter()
            .map(|n| n.layers.len())
            .max()
            .unwrap_or(1)
            .max(1);
        while graph.layers.len() < layer_count {
            graph.layers.push(HnswLayer::default());
        }
        for node in &data.nodes {
            let id = *ids.get(node.node_id as usize)?;
            for (l, neighbors) in node.layers.iter().enumerate() {
                let mapped: Option<Vec<u64>> = neighbors
                    .iter()
                    .map(|&o| ids.get(o as usize).copied())
                    .collect();
                graph.layers[l].adjacency.insert(id, mapped?);
            }
        }
        graph.max_layer = max_layer.min(layer_count.saturating_sub(1));
        graph.entry_point = if entry_ordinal == u64::MAX {
            None
        } else {
            Some(*ids.get(entry_ordinal as usize)?)
        };
        if graph.entry_point.is_none() && !ids.is_empty() {
            // Recover by entering through any node on the top layer.
            graph.entry_point = graph.layers[graph.max_layer]
                .adjacency
                .keys()
                .next()
                .copied();
        }

        Some(Self {
            graph,
            members: ids.into_iter().collect(),
            dirty: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(ids: &[u64], dim: usize, seed: u64) -> VectorData {
        let mut data = VectorData::new(dim as u16);
        let mut x = seed;
        for &id in ids {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                v.push(((x >> 33) as f32) / (u32::MAX as f32) - 0.5);
            }
            data.insert(id, v);
        }
        data
    }

    #[test]
    fn level_rng_is_deterministic_and_in_range() {
        for id in [0u64, 1, 42, u64::MAX] {
            let a = level_rng(id);
            let b = level_rng(id);
            assert_eq!(a, b);
            assert!(a > 0.0 && a < 1.0);
        }
    }

    #[test]
    fn encode_decode_round_trip_with_sparse_ids() {
        // Sparse, non-contiguous IDs exercise the ordinal mapping trailer.
        let ids: Vec<u64> = (0..200u64).map(|i| i * 7 + 3).collect();
        let vectors = make_vectors(&ids, 16, 99);
        let index = VectorIndex::build(&vectors, DistanceMetric::L2, 8, 100);

        let payload = index.encode_payload();
        let decoded =
            VectorIndex::decode_payload(&payload, &vectors).expect("round trip should decode");

        assert_eq!(decoded.node_count(), index.node_count());
        assert!(!decoded.is_dirty());

        // The decoded graph must return identical results.
        let query: Vec<f32> = vectors.get(ids[17]).unwrap().to_vec();
        let a = index.search(&query, 10, 100, &vectors, DistanceMetric::L2);
        let b = decoded.search(&query, 10, 100, &vectors, DistanceMetric::L2);
        assert_eq!(a, b);
        assert_eq!(a[0].0, ids[17]);
    }

    #[test]
    fn decode_rejects_stale_index() {
        let ids: Vec<u64> = (0..64u64).collect();
        let vectors = make_vectors(&ids, 8, 7);
        let index = VectorIndex::build(&vectors, DistanceMetric::L2, 8, 100);
        let payload = index.encode_payload();

        // Removing a vector that the index references makes it stale.
        let mut smaller = make_vectors(&ids, 8, 7);
        smaller.remove(13);
        assert!(VectorIndex::decode_payload(&payload, &smaller).is_none());
    }

    #[test]
    fn decode_rejects_corrupt_or_missing_trailer() {
        let ids: Vec<u64> = (0..32u64).collect();
        let vectors = make_vectors(&ids, 8, 3);
        let index = VectorIndex::build(&vectors, DistanceMetric::L2, 8, 100);
        let payload = index.encode_payload();

        // Truncated payload (no trailer magic).
        assert!(VectorIndex::decode_payload(&payload[..payload.len() - 4], &vectors).is_none());
        // Empty payload.
        assert!(VectorIndex::decode_payload(&[], &vectors).is_none());
    }

    #[test]
    fn decode_rejects_huge_id_count_without_panicking() {
        let ids: Vec<u64> = (0..32u64).collect();
        let vectors = make_vectors(&ids, 8, 3);
        let index = VectorIndex::build(&vectors, DistanceMetric::L2, 8, 100);
        let payload = index.encode_payload();

        // Overwrite the trailer's id_count field with adversarial values.
        // Must return None (safe rebuild), never panic or allocate huge.
        let len_off = payload.len() - 8;
        let body_len =
            u32::from_le_bytes(payload[len_off..len_off + 4].try_into().unwrap()) as usize;
        let body_start = len_off - body_len;
        for count in [u64::MAX, u64::MAX / 2, 1u64 << 48, 1u64 << 32] {
            let mut corrupt = payload.clone();
            corrupt[body_start + 16..body_start + 24].copy_from_slice(&count.to_le_bytes());
            assert!(
                VectorIndex::decode_payload(&corrupt, &vectors).is_none(),
                "id_count {count} must be rejected"
            );
        }

        // Minimal crafted payload: valid magic + body_len + 24-byte body
        // claiming u64::MAX ids with zero delta bytes available.
        let mut crafted = Vec::new();
        crafted.extend_from_slice(&TRAILER_VERSION.to_le_bytes());
        crafted.extend_from_slice(&8u16.to_le_bytes()); // m0
        crafted.extend_from_slice(&1u32.to_le_bytes()); // max_layer
        crafted.extend_from_slice(&u64::MAX.to_le_bytes()); // entry ordinal
        crafted.extend_from_slice(&u64::MAX.to_le_bytes()); // id_count
        let body_len = crafted.len() as u32;
        crafted.extend_from_slice(&body_len.to_le_bytes());
        crafted.extend_from_slice(&TRAILER_MAGIC);
        assert!(VectorIndex::decode_payload(&crafted, &vectors).is_none());
    }

    #[test]
    fn sync_missing_inserts_new_vectors() {
        let ids: Vec<u64> = (0..50u64).collect();
        let mut vectors = make_vectors(&ids, 8, 11);
        let mut index = VectorIndex::build(&vectors, DistanceMetric::L2, 8, 100);
        index.mark_clean();

        // Add new vectors to the store; the index should pick them up.
        let extra = make_vectors(&[100, 101], 8, 12);
        for id in [100u64, 101] {
            vectors.insert(id, extra.get(id).unwrap().to_vec());
        }
        index.sync_missing(&vectors, DistanceMetric::L2);
        assert_eq!(index.node_count(), 52);
        assert!(index.contains(100));
        assert!(index.is_dirty());
    }
}
