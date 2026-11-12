//! Producer: a stream-ready store file and its signed descriptor.
//!
//! Delivery stages (a stage's segments are released only after every lower
//! stage is verified, `requires_lower_stages`):
//!
//! | stage | contents |
//! |---|---|
//! | 0 | journals (tombstones, so deletions precede any vector) |
//! | 1 | Layer A (with anchor vectors) and the hot VEC segment |
//! | 2 | Layer B |
//! | 3 | the remaining (cold) VEC segments |
//! | 4 | Layer C, manifests, witnesses, metadata, the root page |
//!
//! Layer C is the store's own persisted INDEX payload, so
//! `RvfStore::open` loads the identical graph.

use super::layers::{encode_a, encode_b, MAGIC_A, MAGIC_B};
use crate::index_path::decode_graph;
use crate::options::{DistanceMetric, RvfOptions, WitnessConfig};
use crate::{QueryOptions, RvfStore};
use ed25519_dalek::SigningKey;
use rvf_index::{build_layer_a, build_layer_b};
use rvf_types::{RvfError, SegmentType, SEGMENT_ALIGNMENT, SEGMENT_HEADER_SIZE};
use rvf_wire::stream::{StreamBundle, StreamDescriptorBuilder};
use rvf_wire::writer::calculate_padded_size;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Stage of Layer A and the hot vectors.
pub const STAGE_A: u8 = 1;
/// Stage of Layer B.
pub const STAGE_B: u8 = 2;
/// Stage of the cold vectors.
pub const STAGE_COLD: u8 = 3;
/// Stage of Layer C and everything else.
pub const STAGE_C: u8 = 4;

/// Synthetic clustered vectors; ids `0..n` are sorted by cluster so each
/// centroid owns a contiguous id range (what Layer A's partition map needs).
pub struct StreamDataset {
    /// Dimension.
    pub dim: usize,
    /// `n * dim` floats.
    pub data: Vec<f32>,
    /// Cluster of each id.
    pub cluster_of: Vec<u32>,
    /// Number of clusters.
    pub clusters: usize,
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn gauss(x: &mut u64) -> f32 {
    let u = ((splitmix(x) >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
    let v = (splitmix(x) >> 11) as f64 / (1u64 << 53) as f64;
    ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
}

impl StreamDataset {
    /// `n` points in `clusters` Gaussian clusters (centres ~ N(0,1), spread
    /// `sigma`), deterministic in `seed`.
    pub fn clustered(n: usize, dim: usize, clusters: usize, sigma: f32, seed: u64) -> Self {
        let mut s = seed;
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|_| (0..dim).map(|_| gauss(&mut s)).collect())
            .collect();
        let mut data = Vec::with_capacity(n * dim);
        let mut cluster_of = Vec::with_capacity(n);
        for i in 0..n {
            let c = i * clusters / n;
            cluster_of.push(c as u32);
            for &centre in &centers[c] {
                data.push(centre + sigma * gauss(&mut s));
            }
        }
        Self {
            dim,
            data,
            cluster_of,
            clusters,
        }
    }

    /// Number of vectors.
    pub fn len(&self) -> usize {
        self.cluster_of.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.cluster_of.is_empty()
    }

    /// Vector `i`.
    pub fn vector(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    fn centroids(&self) -> Vec<Vec<f32>> {
        let mut sums = vec![vec![0f32; self.dim]; self.clusters];
        let mut cnt = vec![0f32; self.clusters];
        for i in 0..self.len() {
            let c = self.cluster_of[i] as usize;
            cnt[c] += 1.0;
            for (s, x) in sums[c].iter_mut().zip(self.vector(i)) {
                *s += x;
            }
        }
        for (s, c) in sums.iter_mut().zip(&cnt) {
            s.iter_mut().for_each(|x| *x /= c.max(1.0));
        }
        sums
    }
}

/// Producer parameters.
#[derive(Clone, Debug)]
pub struct BuildParams {
    /// Every `hot_stride`-th id is a hot vector (delivered with Layer A).
    pub hot_stride: usize,
    /// Vectors per cold VEC segment.
    pub vec_batch: usize,
    /// Descriptor `chunk_log2` (`0xFF` = segment granular).
    pub chunk_log2: u8,
    /// Descriptor epoch.
    pub epoch: u64,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            hot_stride: 5,
            vec_batch: 10_000,
            chunk_log2: 16,
            epoch: 1,
        }
    }
}

/// A stream-ready file on disk.
pub struct BuiltStore {
    /// The `.rvf` path.
    pub path: PathBuf,
    /// The store's `file_id` (also the descriptor's).
    pub file_id: [u8; 16],
    /// Hot vectors.
    pub hot: usize,
    /// Total vectors.
    pub total: usize,
}

/// Write the stream-ready store (see module docs for the layout).
pub fn build_stream_store(
    path: &Path,
    ds: &StreamDataset,
    p: &BuildParams,
    secret: &[u8; 32],
) -> Result<BuiltStore, RvfError> {
    let options = RvfOptions {
        dimension: ds.dim as u16,
        metric: DistanceMetric::L2,
        witness: WitnessConfig {
            witness_ingest: false,
            witness_delete: false,
            witness_compact: false,
            audit_queries: false,
        },
        ..Default::default()
    };
    let mut store = RvfStore::create_signed(path, options, secret)?;
    let n = ds.len();
    let stride = p.hot_stride.max(1);
    let hot: Vec<usize> = (0..n).step_by(stride).collect();
    let mut is_hot = vec![false; n];
    for &i in &hot {
        is_hot[i] = true;
    }
    let batch = |store: &mut RvfStore, idx: &[usize]| -> Result<(), RvfError> {
        let refs: Vec<&[f32]> = idx.iter().map(|&i| ds.vector(i)).collect();
        let ids: Vec<u64> = idx.iter().map(|&i| i as u64).collect();
        store.ingest_batch(&refs, &ids, None).map(|_| ())
    };
    batch(&mut store, &hot)?;
    let cold: Vec<usize> = (0..n).filter(|&i| !is_hot[i]).collect();
    for c in cold.chunks(p.vec_batch.max(1)) {
        batch(&mut store, c)?;
    }

    // Build the store's own index, then derive A and B from the same graph.
    store.query(ds.vector(0), 1, &QueryOptions::default())?;
    let payload = store
        .index_payload()
        .ok_or(RvfError::Code(rvf_types::ErrorCode::InvalidManifest))?;
    let (graph, _) =
        decode_graph(&payload).ok_or(RvfError::Code(rvf_types::ErrorCode::InvalidManifest))?;
    let assignments: Vec<u32> = ds.cluster_of.clone();
    let mut layer_a = build_layer_a(&graph, &ds.centroids(), &assignments, n as u64);
    // The wire format requires sorted, disjoint partitions: ids must be
    // assigned cluster by cluster (`StreamDataset::clustered` does).
    layer_a.partition_map.sort_by_key(|p| p.vector_id_start);
    if !super::layers::partitions_disjoint(&layer_a.partition_map) {
        return Err(RvfError::Code(rvf_types::ErrorCode::InvalidManifest));
    }
    let mut anchor_ids: BTreeSet<u64> = BTreeSet::new();
    for (id, _) in &layer_a.entry_points {
        anchor_ids.insert(*id);
    }
    for l in &layer_a.top_layers {
        anchor_ids.extend(l.adjacency.keys().copied());
    }
    let anchors: Vec<(u64, &[f32])> = anchor_ids
        .iter()
        .map(|&id| (id, ds.vector(id as usize)))
        .collect();
    let hot_ids: BTreeSet<u64> = hot.iter().map(|&i| i as u64).collect();
    let layer_b = build_layer_b(&graph, &hot_ids);

    store.embed_index_segment(&encode_a(&layer_a, ds.dim, &anchors))?;
    store.embed_index_segment(&encode_b(&layer_b))?;
    let file_id = *store.file_id();
    store.close()?; // persists Layer C (the store's own INDEX payload)
    Ok(BuiltStore {
        path: path.to_path_buf(),
        file_id,
        hot: hot.len(),
        total: n,
    })
}

/// Delivery stage of every segment of `body` in body order (root page last).
pub fn stage_plan(body: &[u8]) -> Vec<u8> {
    let mut stages = Vec::new();
    let mut seen_vec = false;
    let mut off = 0usize;
    while off + SEGMENT_HEADER_SIZE <= body.len() {
        let Ok(h) = rvf_wire::read_segment_header(&body[off..]) else {
            break;
        };
        // Hostile lengths end the walk instead of overflowing (F2-08).
        let Some(plen) = usize::try_from(h.payload_length)
            .ok()
            .and_then(|p| p.checked_add(SEGMENT_HEADER_SIZE))
            .and_then(|t| {
                off.checked_add(t)?
                    .checked_add(SEGMENT_ALIGNMENT)
                    .map(|_| t - SEGMENT_HEADER_SIZE)
            })
        else {
            break;
        };
        let payload = body.get(off + SEGMENT_HEADER_SIZE..off + SEGMENT_HEADER_SIZE + plen);
        let st = match h.seg_type {
            t if t == SegmentType::Journal as u8 => 0,
            t if t == SegmentType::Vec as u8 => {
                let s = if seen_vec { STAGE_COLD } else { STAGE_A };
                seen_vec = true;
                s
            }
            t if t == SegmentType::Index as u8 => match payload.map(|p| p.get(..4)) {
                Some(Some(m)) if m == MAGIC_A => STAGE_A,
                Some(Some(m)) if m == MAGIC_B => STAGE_B,
                _ => STAGE_C,
            },
            _ => STAGE_C,
        };
        stages.push(st);
        off += calculate_padded_size(SEGMENT_HEADER_SIZE, plen);
    }
    stages.push(STAGE_C); // the trailing root page pseudo-entry
    stages
}

/// Read `path` and sign a descriptor for it with the staged plan.
pub fn descriptor_for(
    path: &Path,
    file_id: [u8; 16],
    key: SigningKey,
    p: &BuildParams,
) -> std::io::Result<(Vec<u8>, StreamBundle)> {
    let body = std::fs::read(path)?;
    let stages = stage_plan(&body);
    let bundle = StreamDescriptorBuilder::new(file_id, key)
        .chunk_log2(p.chunk_log2)
        .epoch(p.epoch)
        .stages(&stages)
        .requires_lower_stages(true)
        .build(&body)
        .map_err(|e| std::io::Error::other(format!("{e:?}")))?;
    Ok((body, bundle))
}
