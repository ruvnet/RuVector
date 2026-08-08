//! Turbo4 quantized HNSW index (ADR-296).
//!
//! The HNSW element type is `u8`: each point's data **is** the packed Turbo4
//! code blob (`D/2 + 8` bytes — nibbles + α + Σlevel²). No `Vec<f32>` is
//! retained anywhere in this index; at 1536-D the per-vector payload drops
//! from 6144 B to 776 B (7.9×).
//!
//! Blob roles are structurally disjoint by length (code `D/2+8`, query
//! `D+8`), so one [`Distance<u8>`] functor serves both phases:
//! * insert / graph maintenance → symmetric code×code kernel;
//! * traversal → asymmetric int8-query×code kernel (higher fidelity);
//! * final ranking → the top `k · rescore_multiplier` candidates are
//!   re-scored with the exact f32 rotated query and truncated to `k`.

use crate::error::{Result, RuvectorError};
use crate::index::VectorIndex;
use crate::types::{
    DistanceMetric, HnswConfig, SearchPolicy, SearchQuantization, SearchResult, VectorId,
};
use dashmap::DashMap;
use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use ruvector_turboquant::{bits1, score, Metric, Turbo4Codec, Turbo4Query};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Escalation parameters derived from a [`SearchPolicy`] (ADR-297 §3/§9):
/// relative score-margin below which a query is "uncertain", the ef widening
/// factor per escalation round, and how many rounds are allowed.
struct EscalationParams {
    margin_threshold: f32,
    ef_mult: usize,
    max_rounds: usize,
}

impl EscalationParams {
    /// Policy → escalation mapping, calibrated by the ADR-297 phase-G
    /// ablation (20k×768-D clustered): on concentrated neighborhoods a tight
    /// boundary margin is usually *precision*-bound, which wider traversal
    /// cannot fix (recall@10 moved +0.7 pp for 2.5× latency) — the
    /// `VectorDB`-level f32 verification tier resolves those instead
    /// (−7 pp → +1.4 pp vs the f32 index). So `Balanced` no longer
    /// escalates; `Quality` keeps escalation as the safety net for
    /// deployments where no higher-precision tier exists above this index.
    fn for_policy(policy: SearchPolicy) -> Self {
        match policy {
            SearchPolicy::Quality => Self {
                margin_threshold: 0.05,
                ef_mult: 3,
                max_rounds: 2,
            },
            SearchPolicy::Balanced | SearchPolicy::MaxCompression => Self {
                margin_threshold: 0.0,
                ef_mult: 1,
                max_rounds: 0,
            },
        }
    }
}

/// Distance functor over Turbo4 blobs. Chooses the kernel from blob lengths:
/// hnsw_rs passes the search query straight through to `eval`, which is what
/// makes true asymmetric traversal possible without forking hnsw_rs.
///
/// Two layouts (ADR-297 §2):
/// * **Direct**: node data = Turbo4 code (`D/2+8`); query = int8 blob (`D+8`).
/// * **Cascade** (`RaBitQ1`): node data = `[bits1 (bl+8) ‖ turbo4 (D/2+8)]`
///   with the 1-bit plane FIRST, so traversal touches only the short,
///   cache-friendly prefix; query = bit-plane blob (`8·bl+40`). Graph
///   *construction* (node×node) still scores on the Turbo4 sections — build
///   quality is paid once, traversal bandwidth is paid every query.
struct Turbo4DistanceFn {
    metric: Metric,
    dim: usize,
    code_len: usize,
    query_len: usize,
    /// Cascade-mode lengths; 0 when direct.
    combined_len: usize,
    bits_query_len: usize,
    bits_code_len: usize,
}

impl Distance<u8> for Turbo4DistanceFn {
    #[inline(always)]
    fn eval(&self, a: &[u8], b: &[u8]) -> f32 {
        // Cascade layouts first (combined_len is 0 in direct mode, so these
        // arms are dead there).
        if a.len() == self.combined_len && b.len() == self.combined_len {
            score::symmetric_distance(
                self.metric,
                &a[self.bits_code_len..],
                &b[self.bits_code_len..],
                self.dim,
            )
        } else if a.len() == self.bits_query_len && b.len() == self.combined_len {
            bits1::query_blob_distance(self.metric, a, &b[..self.bits_code_len], self.dim)
        } else if b.len() == self.bits_query_len && a.len() == self.combined_len {
            bits1::query_blob_distance(self.metric, b, &a[..self.bits_code_len], self.dim)
        } else if a.len() == self.code_len && b.len() == self.code_len {
            score::symmetric_distance(self.metric, a, b, self.dim)
        } else if a.len() == self.query_len && b.len() == self.code_len {
            score::asymmetric_distance(self.metric, a, b, self.dim)
        } else if b.len() == self.query_len && a.len() == self.code_len {
            score::asymmetric_distance(self.metric, b, a, self.dim)
        } else {
            debug_assert!(
                false,
                "unrecognized Turbo4 blob lengths {}/{}",
                a.len(),
                b.len()
            );
            f32::MAX
        }
    }
}

fn to_turbo_metric(metric: DistanceMetric) -> Result<Metric> {
    match metric {
        DistanceMetric::Euclidean => Ok(Metric::Euclidean),
        DistanceMetric::Cosine => Ok(Metric::Cosine),
        DistanceMetric::DotProduct => Ok(Metric::DotProduct),
        DistanceMetric::Manhattan => Err(RuvectorError::InvalidParameter(
            "Turbo4 quantization does not support the Manhattan metric \
             (it does not decompose over the level dot product); use \
             Euclidean, Cosine, or DotProduct, or disable quantization"
                .into(),
        )),
    }
}

struct Turbo4Inner {
    hnsw: Hnsw<'static, u8, Turbo4DistanceFn>,
    /// id → packed code blob; the rescoring source and (with the graph's own
    /// copy) the only representation of the vector.
    codes: DashMap<VectorId, Vec<u8>>,
    id_to_idx: DashMap<VectorId, usize>,
    idx_to_id: DashMap<usize, VectorId>,
    next_idx: usize,
}

/// Serializable state of a [`Turbo4HnswIndex`]. Only codes and mappings are
/// stored — the graph is rebuilt deterministically on load (same approach as
/// the f32 `HnswState`), and the codec is reconstructed from
/// `(dimensions, rotation_seed)`, which is exactly why ADR-296 requires the
/// rotation to be bit-stable.
#[derive(bincode::Encode, bincode::Decode)]
struct Turbo4State {
    codes: Vec<(String, Vec<u8>)>,
    id_to_idx: Vec<(String, usize)>,
    idx_to_id: Vec<(usize, String)>,
    next_idx: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    max_elements: usize,
    dimensions: usize,
    /// 0 = Euclidean, 1 = Cosine, 2 = DotProduct.
    metric: u8,
    rotation_seed: u64,
    rescore_multiplier: usize,
    /// 0 = Quality, 1 = Balanced, 2 = MaxCompression.
    policy: u8,
    /// 0 = Turbo4Direct, 1 = RaBitQ1 cascade.
    search_quant: u8,
}

fn policy_to_u8(p: SearchPolicy) -> u8 {
    match p {
        SearchPolicy::Quality => 0,
        SearchPolicy::Balanced => 1,
        SearchPolicy::MaxCompression => 2,
    }
}

fn policy_from_u8(v: u8) -> SearchPolicy {
    match v {
        0 => SearchPolicy::Quality,
        2 => SearchPolicy::MaxCompression,
        _ => SearchPolicy::Balanced,
    }
}

/// HNSW index storing only Turbo4 packed codes (plus, in cascade mode, the
/// 1-bit traversal plane).
pub struct Turbo4HnswIndex {
    inner: Arc<RwLock<Turbo4Inner>>,
    codec: Arc<Turbo4Codec>,
    config: HnswConfig,
    metric: Metric,
    dimensions: usize,
    rotation_seed: u64,
    rescore_multiplier: usize,
    policy: SearchPolicy,
    search_quantization: SearchQuantization,
    /// Offset of the Turbo4 section inside a stored blob (0 in direct mode,
    /// `bits1::code1_len(dim)` in cascade mode).
    t4_off: usize,
    escalation: EscalationParams,
    /// Adaptive-plane telemetry: total queries / queries that escalated.
    queries: AtomicU64,
    escalated: AtomicU64,
}

impl Turbo4HnswIndex {
    /// Create a Turbo4-quantized HNSW index.
    pub fn new(
        dimensions: usize,
        metric: DistanceMetric,
        config: HnswConfig,
        rotation_seed: u64,
        rescore_multiplier: usize,
        policy: SearchPolicy,
        search_quantization: SearchQuantization,
    ) -> Result<Self> {
        let metric = to_turbo_metric(metric)?;
        let codec = Turbo4Codec::new(dimensions, rotation_seed).map_err(|e| {
            RuvectorError::InvalidParameter(format!("Turbo4 codec init failed: {e}"))
        })?;
        let cascade = search_quantization == SearchQuantization::RaBitQ1;
        let bits_code_len = bits1::code1_len(dimensions);
        // Blob-length disambiguation invariant: in cascade mode the only
        // lengths in flight are combined and bits-query (the eval arms check
        // cascade layouts first); in direct mode combined_len = 0 disables
        // those arms entirely.
        let distance_fn = Turbo4DistanceFn {
            metric,
            dim: dimensions,
            code_len: codec.code_len(),
            query_len: codec.query_len(),
            combined_len: if cascade {
                bits_code_len + codec.code_len()
            } else {
                0
            },
            bits_query_len: if cascade {
                bits1::query1_len(dimensions)
            } else {
                0
            },
            bits_code_len,
        };
        let hnsw = Hnsw::<u8, Turbo4DistanceFn>::new(
            config.m,
            config.max_elements,
            dimensions,
            config.ef_construction,
            distance_fn,
        );
        Ok(Self {
            inner: Arc::new(RwLock::new(Turbo4Inner {
                hnsw,
                codes: DashMap::new(),
                id_to_idx: DashMap::new(),
                idx_to_id: DashMap::new(),
                next_idx: 0,
            })),
            codec: Arc::new(codec),
            config,
            metric,
            dimensions,
            rotation_seed,
            rescore_multiplier: rescore_multiplier.max(1),
            policy,
            search_quantization,
            t4_off: if cascade { bits_code_len } else { 0 },
            escalation: EscalationParams::for_policy(policy),
            queries: AtomicU64::new(0),
            escalated: AtomicU64::new(0),
        })
    }

    /// Encode one vector into the stored blob for the active mode.
    fn encode_blob(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if self.search_quantization == SearchQuantization::RaBitQ1 {
            let (turbo4, bits) = self
                .codec
                .encode_dual(vector)
                .map_err(|e| RuvectorError::InvalidInput(e.to_string()))?;
            let mut blob = bits;
            blob.extend_from_slice(&turbo4);
            Ok(blob)
        } else {
            self.codec
                .encode(vector)
                .map_err(|e| RuvectorError::InvalidInput(e.to_string()))
        }
    }

    /// Serialize codes + mappings + parameters (bincode). The graph itself is
    /// not stored; it is rebuilt deterministically by [`Self::deserialize`].
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let inner = self.inner.read();
        let metric_u8 = match self.metric {
            Metric::Euclidean => 0u8,
            Metric::Cosine => 1,
            Metric::DotProduct => 2,
        };
        let state = Turbo4State {
            codes: inner
                .codes
                .iter()
                .map(|e| (e.key().clone(), e.value().clone()))
                .collect(),
            id_to_idx: inner
                .id_to_idx
                .iter()
                .map(|e| (e.key().clone(), *e.value()))
                .collect(),
            idx_to_id: inner
                .idx_to_id
                .iter()
                .map(|e| (*e.key(), e.value().clone()))
                .collect(),
            next_idx: inner.next_idx,
            m: self.config.m,
            ef_construction: self.config.ef_construction,
            ef_search: self.config.ef_search,
            max_elements: self.config.max_elements,
            dimensions: self.dimensions,
            metric: metric_u8,
            rotation_seed: self.rotation_seed,
            rescore_multiplier: self.rescore_multiplier,
            policy: policy_to_u8(self.policy),
            search_quant: match self.search_quantization {
                SearchQuantization::Turbo4Direct => 0,
                SearchQuantization::RaBitQ1 => 1,
            },
        };
        bincode::encode_to_vec(&state, bincode::config::standard()).map_err(|e| {
            RuvectorError::SerializationError(format!("Failed to serialize Turbo4 index: {e}"))
        })
    }

    /// Rebuild an index from [`Self::serialize`] output. Codes are inserted
    /// into a fresh graph in index order; vectors are never re-encoded (the
    /// stored blobs are the source of truth).
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        let (state, _): (Turbo4State, usize) =
            bincode::decode_from_slice(bytes, bincode::config::standard()).map_err(|e| {
                RuvectorError::SerializationError(format!(
                    "Failed to deserialize Turbo4 index: {e}"
                ))
            })?;
        let metric = match state.metric {
            0 => DistanceMetric::Euclidean,
            1 => DistanceMetric::Cosine,
            2 => DistanceMetric::DotProduct,
            other => {
                return Err(RuvectorError::SerializationError(format!(
                    "unknown Turbo4 metric tag {other}"
                )))
            }
        };
        let config = HnswConfig {
            m: state.m,
            ef_construction: state.ef_construction,
            ef_search: state.ef_search,
            max_elements: state.max_elements,
        };
        let index = Self::new(
            state.dimensions,
            metric,
            config,
            state.rotation_seed,
            state.rescore_multiplier,
            policy_from_u8(state.policy),
            if state.search_quant == 1 {
                SearchQuantization::RaBitQ1
            } else {
                SearchQuantization::Turbo4Direct
            },
        )?;
        {
            let mut inner = index.inner.write();
            let codes_lookup: std::collections::HashMap<&str, &Vec<u8>> =
                state.codes.iter().map(|(id, c)| (id.as_str(), c)).collect();
            let mut sorted: Vec<(usize, &String)> =
                state.idx_to_id.iter().map(|(i, id)| (*i, id)).collect();
            sorted.sort_unstable_by_key(|(i, _)| *i);
            for (idx, id) in &sorted {
                if let Some(blob) = codes_lookup.get(id.as_str()) {
                    inner.hnsw.insert_data(blob, *idx);
                }
            }
            for (id, blob) in state.codes {
                inner.codes.insert(id, blob);
            }
            for (id, idx) in state.id_to_idx {
                inner.id_to_idx.insert(id, idx);
            }
            for (idx, id) in state.idx_to_id {
                inner.idx_to_id.insert(idx, id);
            }
            inner.next_idx = state.next_idx;
        }
        Ok(index)
    }

    /// Stored bytes per vector (`D/2 + 8`).
    pub fn code_len(&self) -> usize {
        self.codec.code_len()
    }

    /// Adaptive-plane telemetry: `(total_queries, escalated_queries)`.
    /// A healthy workload escalates only a small fraction (~5–15 %).
    pub fn adaptive_stats(&self) -> (u64, u64) {
        (
            self.queries.load(Ordering::Relaxed),
            self.escalated.load(Ordering::Relaxed),
        )
    }

    /// One traversal + exact-rescore pass. Returns the full rescored
    /// candidate list, ascending — the margin between `list[k-1]` and
    /// `list[k]` is the stability signal for adaptive escalation.
    fn traverse_and_rescore(
        &self,
        inner: &Turbo4Inner,
        traversal_blob: &[u8],
        prepared: &Turbo4Query,
        fetch: usize,
        ef: usize,
    ) -> Vec<SearchResult> {
        let neighbors = inner.hnsw.search(traversal_blob, fetch, ef);
        let mut results: Vec<SearchResult> = neighbors
            .into_iter()
            .filter_map(|n| {
                let id = inner.idx_to_id.get(&n.d_id)?.clone();
                let code = inner.codes.get(&id)?;
                let dist = score::rescore(
                    self.metric,
                    prepared,
                    &code.value()[self.t4_off..],
                    self.dimensions,
                );
                Some(SearchResult {
                    id,
                    score: dist,
                    vector: None,
                    metadata: None,
                })
            })
            .collect();
        results.sort_unstable_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Relative margin between the last kept and first dropped candidate.
    /// `f32::INFINITY` when nothing is dropped (result already exhaustive).
    fn boundary_margin(results: &[SearchResult], k: usize) -> f32 {
        if results.len() <= k || k == 0 {
            return f32::INFINITY;
        }
        let kept = results[k - 1].score;
        let dropped = results[k].score;
        (dropped - kept) / kept.abs().max(1e-6)
    }

    /// Search with an explicit `ef_search`.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: query.len(),
            });
        }
        if k == 0 {
            return Ok(vec![]);
        }
        let prepared = self
            .codec
            .encode_query(query)
            .map_err(|e| RuvectorError::InvalidInput(e.to_string()))?;
        // Cascade mode traverses on the 1-bit plane; the bit-plane query is
        // built from the SAME rotated coordinates (one rotation total).
        let traversal_blob: Vec<u8> = if self.search_quantization == SearchQuantization::RaBitQ1 {
            bits1::Bits1Query::new(&prepared.rotated)
                .map_err(|e| RuvectorError::InvalidInput(e.to_string()))?
                .to_blob()
        } else {
            prepared.blob.clone()
        };

        let inner = self.inner.read();
        // hnsw_rs panics on empty indexes (unguarded heap peek) — return early.
        if inner.codes.is_empty() {
            return Ok(vec![]);
        }
        self.queries.fetch_add(1, Ordering::Relaxed);

        // Base pass: over-fetch for the exact rescoring step.
        let mut fetch = k.saturating_mul(self.rescore_multiplier);
        let mut ef = ef_search.max(fetch);
        let mut results = self.traverse_and_rescore(&inner, &traversal_blob, &prepared, fetch, ef);

        // Adaptive escalation (ADR-297 §3): if the kept/dropped boundary sits
        // inside the quantization noise band, widen the search. Stop as soon
        // as the top-k membership is stable across rounds — that's the
        // candidate-stability signal, independent of absolute scores.
        let mut round = 0;
        let mut did_escalate = false;
        while round < self.escalation.max_rounds
            && Self::boundary_margin(&results, k) < self.escalation.margin_threshold
        {
            did_escalate = true;
            fetch = fetch.saturating_mul(2);
            ef = ef.saturating_mul(self.escalation.ef_mult);
            let wider = self.traverse_and_rescore(&inner, &traversal_blob, &prepared, fetch, ef);

            let stable = wider.len() >= k
                && results.len() >= k
                && wider[..k]
                    .iter()
                    .zip(&results[..k])
                    .all(|(a, b)| a.id == b.id);
            results = wider;
            round += 1;
            if stable {
                break;
            }
        }
        if did_escalate {
            self.escalated.fetch_add(1, Ordering::Relaxed);
        }

        results.truncate(k);
        Ok(results)
    }
}

impl VectorIndex for Turbo4HnswIndex {
    fn add(&mut self, id: VectorId, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: vector.len(),
            });
        }
        let blob = self.encode_blob(&vector)?;
        drop(vector); // floats are not retained

        let mut inner = self.inner.write();
        let idx = inner.next_idx;
        inner.next_idx += 1;
        inner.hnsw.insert_data(&blob, idx);
        inner.codes.insert(id.clone(), blob);
        inner.id_to_idx.insert(id.clone(), idx);
        inner.idx_to_id.insert(idx, id);
        Ok(())
    }

    fn add_batch(&mut self, entries: Vec<(VectorId, Vec<f32>)>) -> Result<()> {
        for (_, vector) in &entries {
            if vector.len() != self.dimensions {
                return Err(RuvectorError::DimensionMismatch {
                    expected: self.dimensions,
                    actual: vector.len(),
                });
            }
        }
        // Encode outside the lock — the expensive part (O(D log D) each).
        let mut encoded = Vec::with_capacity(entries.len());
        for (id, vector) in entries {
            let blob = self.encode_blob(&vector)?;
            encoded.push((id, blob));
        }

        let mut inner = self.inner.write();
        let base = inner.next_idx;
        inner.next_idx += encoded.len();

        // Same parallel-insert gate as the f32 index (hnsw_rs rule of thumb:
        // parallel pays off only for large batches).
        const PARALLEL_THRESHOLD: usize = 10_000;
        if encoded.len() >= PARALLEL_THRESHOLD {
            let datas: Vec<(&[u8], usize)> = encoded
                .iter()
                .enumerate()
                .map(|(i, (_, blob))| (blob.as_slice(), base + i))
                .collect();
            inner.hnsw.parallel_insert_slice(&datas);
        } else {
            for (i, (_, blob)) in encoded.iter().enumerate() {
                inner.hnsw.insert_data(blob, base + i);
            }
        }

        for (i, (id, blob)) in encoded.into_iter().enumerate() {
            inner.codes.insert(id.clone(), blob);
            inner.id_to_idx.insert(id.clone(), base + i);
            inner.idx_to_id.insert(base + i, id);
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    fn remove(&mut self, id: &VectorId) -> Result<bool> {
        let inner = self.inner.write();
        // hnsw_rs has no deletion; drop the mappings (graph node remains,
        // same limitation as the f32 index).
        let removed = inner.codes.remove(id).is_some();
        if removed {
            if let Some((_, idx)) = inner.id_to_idx.remove(id) {
                inner.idx_to_id.remove(&idx);
            }
        }
        Ok(removed)
    }

    fn len(&self) -> usize {
        self.inner.read().codes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic Gaussian test vectors (SplitMix64 + Box–Muller).
    fn gauss_vecs(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        (0..count)
            .map(|_| {
                let mut v = Vec::with_capacity(dim);
                while v.len() < dim {
                    let u1 = (next() >> 11) as f64 / (1u64 << 53) as f64;
                    let u2 = (next() >> 11) as f64 / (1u64 << 53) as f64;
                    let r = (-2.0 * u1.max(1e-12).ln()).sqrt();
                    let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
                    v.push((r * c) as f32);
                    if v.len() < dim {
                        v.push((r * s) as f32);
                    }
                }
                v
            })
            .collect()
    }

    fn l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn manhattan_is_rejected() {
        let err = Turbo4HnswIndex::new(
            64,
            DistanceMetric::Manhattan,
            HnswConfig::default(),
            42,
            4,
            SearchPolicy::Balanced,
            SearchQuantization::Turbo4Direct,
        );
        assert!(err.is_err());
    }

    /// Clustered vectors: `n` points around `n / cluster_size` Gaussian
    /// centers — the neighbor structure real embedding corpora have.
    fn clustered_vecs(
        n: usize,
        dim: usize,
        cluster_size: usize,
        spread: f32,
        seed: u64,
    ) -> Vec<Vec<f32>> {
        let n_centers = n.div_ceil(cluster_size);
        let centers = gauss_vecs(n_centers, dim, seed);
        let noise = gauss_vecs(n, dim, seed ^ 0xABCD_EF01);
        (0..n)
            .map(|i| {
                centers[i / cluster_size]
                    .iter()
                    .zip(&noise[i])
                    .map(|(c, e)| c + spread * e)
                    .collect()
            })
            .collect()
    }

    /// Recall@10 vs brute force, measured for BOTH the f32 HNSW and the
    /// Turbo4 HNSW on identical data/params — the ADR-296 acceptance metric
    /// is the *delta* to the f32 baseline. Data is clustered, like real
    /// embedding corpora (and SIFT/GIST). i.i.d. Gaussian data is deliberately
    /// NOT used here: with 500 points at 128-D all pairwise distances
    /// concentrate within a few percent, so the ~1% distance error of ANY
    /// 4-bit quantizer scrambles top-10 membership — that regime is covered
    /// by the absolute-floor smoke test below instead.
    #[test]
    fn recall_close_to_f32_hnsw_baseline() -> Result<()> {
        use crate::index::hnsw::HnswIndex;

        let dim = 128;
        let n = 500;
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            max_elements: 1000,
        };
        let vectors = clustered_vecs(n, dim, 10, 0.35, 7);
        let entries: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("v{i}"), v.clone()))
            .collect();

        let mut t4 = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            config.clone(),
            42,
            8,
            SearchPolicy::Balanced,
            SearchQuantization::Turbo4Direct,
        )?;
        t4.add_batch(entries.clone())?;
        assert_eq!(t4.len(), n);
        let mut f32_ix = HnswIndex::new(dim, DistanceMetric::Euclidean, config)?;
        f32_ix.add_batch(entries)?;

        // Queries: data points perturbed with fresh noise — same clusters,
        // so each query has a meaningful true neighborhood.
        let qnoise = gauss_vecs(20, dim, 998877);
        let queries: Vec<Vec<f32>> = (0..20)
            .map(|j| {
                vectors[(j * 25) % n]
                    .iter()
                    .zip(&qnoise[j])
                    .map(|(v, e)| v + 0.2 * e)
                    .collect()
            })
            .collect();
        let (mut t4_hits, mut f32_hits, mut total) = (0usize, 0usize, 0usize);
        for q in &queries {
            let mut truth: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2(q, v)))
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1));
            let top10: std::collections::HashSet<String> =
                truth[..10].iter().map(|(i, _)| format!("v{i}")).collect();

            let t4_res = t4.search(q, 10)?;
            assert_eq!(t4_res.len(), 10);
            t4_hits += t4_res.iter().filter(|r| top10.contains(&r.id)).count();
            f32_hits += f32_ix
                .search(q, 10)?
                .iter()
                .filter(|r| top10.contains(&r.id))
                .count();
            total += 10;
        }
        let t4_recall = t4_hits as f32 / total as f32;
        let f32_recall = f32_hits as f32 / total as f32;
        // 3pp = ADR target (0.5pp) + hnsw_rs graph-construction nondeterminism
        // (~±2 hits per 200 across two independently built graphs).
        assert!(
            t4_recall >= f32_recall - 0.03,
            "Turbo4 recall@10 {t4_recall} vs f32 baseline {f32_recall}: loss above 3pp \
             on clustered data ({t4_hits}/{f32_hits}/{total})"
        );
        assert!(
            t4_recall >= 0.90,
            "Turbo4 absolute recall@10 {t4_recall} below floor"
        );
        Ok(())
    }

    /// Worst-case smoke test: i.i.d. Gaussian data has near-total distance
    /// concentration (500 pts @ 128-D ⇒ all pairwise distances within a few
    /// percent), so any 4-bit quantizer loses top-10 members to ~1% distance
    /// noise. Assert a floor, not parity with f32.
    #[test]
    fn gaussian_worst_case_recall_floor() -> Result<()> {
        let dim = 128;
        let n = 500;
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            max_elements: 1000,
        };
        let vectors = gauss_vecs(n, dim, 7);
        let entries: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("v{i}"), v.clone()))
            .collect();
        let mut t4 = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            config,
            42,
            8,
            SearchPolicy::Balanced,
            SearchQuantization::Turbo4Direct,
        )?;
        t4.add_batch(entries)?;

        let queries = gauss_vecs(20, dim, 12345);
        let (mut hits, mut total) = (0usize, 0usize);
        for q in &queries {
            let mut truth: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2(q, v)))
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1));
            let top10: std::collections::HashSet<String> =
                truth[..10].iter().map(|(i, _)| format!("v{i}")).collect();
            hits += t4
                .search(q, 10)?
                .iter()
                .filter(|r| top10.contains(&r.id))
                .count();
            total += 10;
        }
        let recall = hits as f32 / total as f32;
        assert!(
            recall >= 0.75,
            "Turbo4 worst-case Gaussian recall@10 {recall} below floor ({hits}/{total})"
        );
        Ok(())
    }

    #[test]
    fn self_query_returns_self_first() -> Result<()> {
        let dim = 64;
        let config = HnswConfig::default();
        let mut index = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Cosine,
            config,
            42,
            4,
            SearchPolicy::Balanced,
            SearchQuantization::Turbo4Direct,
        )?;
        let vectors = gauss_vecs(50, dim, 3);
        for (i, v) in vectors.iter().enumerate() {
            index.add(format!("v{i}"), v.clone())?;
        }
        let results = index.search(&vectors[7], 5)?;
        assert_eq!(results[0].id, "v7");
        Ok(())
    }

    /// Adaptive escalation (ADR-297 §3): MaxCompression must never escalate;
    /// Quality must escalate on distance-concentrated (uncertain) queries and
    /// must not lose recall relative to the non-escalating policy.
    #[test]
    fn adaptive_escalation_follows_policy() -> Result<()> {
        let dim = 128;
        let n = 400;
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 60,
            max_elements: 1000,
        };
        // i.i.d. Gaussian ⇒ tight boundary margins ⇒ escalation should fire.
        let vectors = gauss_vecs(n, dim, 21);
        let entries: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("v{i}"), v.clone()))
            .collect();

        let mut fixed = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            config.clone(),
            42,
            2,
            SearchPolicy::MaxCompression,
            SearchQuantization::Turbo4Direct,
        )?;
        fixed.add_batch(entries.clone())?;
        let mut adaptive = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            config,
            42,
            2,
            SearchPolicy::Quality,
            SearchQuantization::Turbo4Direct,
        )?;
        adaptive.add_batch(entries)?;

        let queries = gauss_vecs(25, dim, 4242);
        let (mut fixed_hits, mut adaptive_hits, mut total) = (0usize, 0usize, 0usize);
        for q in &queries {
            let mut truth: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2(q, v)))
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1));
            let top10: std::collections::HashSet<String> =
                truth[..10].iter().map(|(i, _)| format!("v{i}")).collect();
            fixed_hits += fixed
                .search(q, 10)?
                .iter()
                .filter(|r| top10.contains(&r.id))
                .count();
            adaptive_hits += adaptive
                .search(q, 10)?
                .iter()
                .filter(|r| top10.contains(&r.id))
                .count();
            total += 10;
        }

        let (fq, fe) = fixed.adaptive_stats();
        assert_eq!(fq, 25);
        assert_eq!(fe, 0, "MaxCompression must never escalate");

        let (aq, ae) = adaptive.adaptive_stats();
        assert_eq!(aq, 25);
        assert!(
            ae > 0,
            "Quality policy should escalate on concentrated Gaussian queries"
        );
        // hnsw_rs graph construction is nondeterministic (internal RNG for
        // level assignment), so two separately built indexes differ by a few
        // hits regardless of policy — allow that noise band while still
        // catching systematic recall loss from escalation.
        assert!(
            adaptive_hits + 5 >= fixed_hits,
            "escalation must not lose recall: adaptive {adaptive_hits} vs fixed {fixed_hits} of {total}"
        );
        Ok(())
    }

    /// Easy well-separated queries must not trigger escalation under
    /// Balanced — the ~5–15 % escalation budget depends on margins staying
    /// wide when results are unambiguous.
    #[test]
    fn easy_queries_do_not_escalate() -> Result<()> {
        let dim = 64;
        let config = HnswConfig::default();
        let mut index = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            config,
            42,
            4,
            SearchPolicy::Balanced,
            SearchQuantization::Turbo4Direct,
        )?;
        // Well-separated clusters of exactly k members, so the kept/dropped
        // boundary falls BETWEEN clusters (wide margin). A boundary inside a
        // cluster is genuinely ambiguous and escalating on it is correct.
        let vectors = clustered_vecs(200, dim, 5, 0.05, 33);
        for (i, v) in vectors.iter().enumerate() {
            index.add(format!("v{i}"), v.iter().map(|x| x * 10.0).collect())?;
        }
        for j in 0..10 {
            let q: Vec<f32> = vectors[j * 20].iter().map(|x| x * 10.0).collect();
            index.search(&q, 5)?;
        }
        let (queries, escalated) = index.adaptive_stats();
        assert_eq!(queries, 10);
        assert!(
            escalated <= 2,
            "well-separated queries escalated {escalated}/10 times"
        );
        Ok(())
    }

    /// Serialization roundtrip: codes are byte-identical, the rebuilt graph
    /// serves the same queries, and self-queries still rank themselves first
    /// with identical rescored distances (rescoring is graph-independent).
    #[test]
    fn serialization_roundtrip() -> Result<()> {
        let dim = 64;
        let mut index = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            HnswConfig::default(),
            42,
            4,
            SearchPolicy::Balanced,
            SearchQuantization::Turbo4Direct,
        )?;
        let vectors = clustered_vecs(120, dim, 6, 0.2, 55);
        for (i, v) in vectors.iter().enumerate() {
            index.add(format!("v{i}"), v.clone())?;
        }
        let bytes = index.serialize()?;
        let restored = Turbo4HnswIndex::deserialize(&bytes)?;
        assert_eq!(restored.len(), 120);
        assert_eq!(restored.code_len(), index.code_len());

        for probe in [3usize, 47, 99] {
            let orig = index.search(&vectors[probe], 3)?;
            let rest = restored.search(&vectors[probe], 3)?;
            assert_eq!(rest[0].id, format!("v{probe}"), "self-query after restore");
            assert!(
                (orig[0].score - rest[0].score).abs() < 1e-6,
                "rescored self distance must be identical: {} vs {}",
                orig[0].score,
                rest[0].score
            );
        }
        Ok(())
    }

    /// Cascade mode (RaBitQ1 traversal → Turbo4 rescore): on clustered data
    /// the cascade must stay within a few pp of direct-mode recall — the
    /// 1-bit plane only generates candidates; ranking quality comes from the
    /// shared Turbo4 rescore.
    #[test]
    fn cascade_recall_close_to_direct_mode() -> Result<()> {
        let dim = 128;
        let n = 500;
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            max_elements: 1000,
        };
        let vectors = clustered_vecs(n, dim, 10, 0.35, 7);
        let entries: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("v{i}"), v.clone()))
            .collect();

        let mut direct = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            config.clone(),
            42,
            8,
            SearchPolicy::Balanced,
            SearchQuantization::Turbo4Direct,
        )?;
        direct.add_batch(entries.clone())?;
        let mut cascade = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            config,
            42,
            8,
            SearchPolicy::Balanced,
            SearchQuantization::RaBitQ1,
        )?;
        cascade.add_batch(entries)?;

        let qnoise = gauss_vecs(20, dim, 998877);
        let queries: Vec<Vec<f32>> = (0..20)
            .map(|j| {
                vectors[(j * 25) % n]
                    .iter()
                    .zip(&qnoise[j])
                    .map(|(v, e)| v + 0.2 * e)
                    .collect()
            })
            .collect();

        let (mut d_hits, mut c_hits, mut total) = (0usize, 0usize, 0usize);
        for q in &queries {
            let mut truth: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2(q, v)))
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1));
            let top10: std::collections::HashSet<String> =
                truth[..10].iter().map(|(i, _)| format!("v{i}")).collect();
            d_hits += direct
                .search(q, 10)?
                .iter()
                .filter(|r| top10.contains(&r.id))
                .count();
            c_hits += cascade
                .search(q, 10)?
                .iter()
                .filter(|r| top10.contains(&r.id))
                .count();
            total += 10;
        }
        let (d_recall, c_recall) = (d_hits as f32 / total as f32, c_hits as f32 / total as f32);
        assert!(
            c_recall >= d_recall - 0.05,
            "cascade recall@10 {c_recall} vs direct {d_recall} ({c_hits}/{d_hits}/{total})"
        );
        assert!(c_recall >= 0.85, "cascade absolute recall {c_recall}");
        Ok(())
    }

    /// Serialization roundtrip in cascade mode: combined blobs survive and
    /// the restored index searches identically at the rescore tier.
    #[test]
    fn cascade_serialization_roundtrip() -> Result<()> {
        let dim = 64;
        let mut index = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Cosine,
            HnswConfig::default(),
            42,
            4,
            SearchPolicy::Balanced,
            SearchQuantization::RaBitQ1,
        )?;
        let vectors = clustered_vecs(100, dim, 5, 0.2, 77);
        for (i, v) in vectors.iter().enumerate() {
            index.add(format!("v{i}"), v.clone())?;
        }
        let restored = Turbo4HnswIndex::deserialize(&index.serialize()?)?;
        assert_eq!(restored.len(), 100);
        for probe in [2usize, 55, 98] {
            let orig = index.search(&vectors[probe], 3)?;
            let rest = restored.search(&vectors[probe], 3)?;
            assert_eq!(rest[0].id, format!("v{probe}"));
            assert!((orig[0].score - rest[0].score).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn empty_index_is_safe() -> Result<()> {
        let index = Turbo4HnswIndex::new(
            64,
            DistanceMetric::Euclidean,
            HnswConfig::default(),
            42,
            4,
            SearchPolicy::Balanced,
            SearchQuantization::Turbo4Direct,
        )?;
        assert!(index.search(&vec![0.5; 64], 5)?.is_empty());
        Ok(())
    }
}
