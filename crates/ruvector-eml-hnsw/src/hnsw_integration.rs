//! Thin integration layer that wires [`EmlDistanceModel`] into `hnsw_rs`.
//!
//! PR #353's original crate produced six standalone models but no consumer,
//! so no real HNSW ever saw the learned dimension selection. This module
//! closes that gap.
//!
//! # Usage
//!
//! 1. Collect a representative sample of the vectors you plan to index
//!    (typically 1k–10k).
//! 2. [`EmlHnsw::train_selector`] learns which dimensions discriminate.
//! 3. [`EmlHnsw::add`] projects each vector into the learned subspace and
//!    inserts the projection into the underlying HNSW.
//! 4. [`EmlHnsw::search`] projects the query and searches the reduced index,
//!    returning the original full-dim ids.
//!
//! The full-dim vectors are kept on the side so callers can optionally
//! re-rank the top-K with exact distance.

use crate::cosine_decomp::{cosine_distance_f32, EmlDistanceModel};
use crate::selected_distance::{cosine_distance_simd, project_vector};
use hnsw_rs::prelude::{DistCosine, Hnsw};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Metric used for the reduced-dim HNSW distance.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EmlMetric {
    Cosine,
}

/// HNSW index that computes distance on an EML-selected subset of dimensions.
///
/// Insert / search take full-dim vectors; the wrapper handles projection.
pub struct EmlHnsw {
    selector: EmlDistanceModel,
    reduced_dim: usize,
    hnsw: Hnsw<'static, f32, DistCosine>,
    /// Full-dim store indexed by internal id (1-based, matching `hnsw_rs`).
    full_store: Vec<Vec<f32>>,
    metric: EmlMetric,
}

/// Result of a reduced-dim search. Distance is cosine over the projection.
#[derive(Clone, Debug)]
pub struct EmlSearchResult {
    pub id: usize,
    pub distance: f32,
}

impl EmlHnsw {
    /// Build a new index. Requires the selector to have been trained.
    ///
    /// `m` and `ef_construction` follow the usual HNSW conventions.
    /// `max_elements` caps the underlying graph capacity.
    pub fn new(
        selector: EmlDistanceModel,
        metric: EmlMetric,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self, &'static str> {
        if !selector.is_trained() {
            return Err("selector must be trained before building an index");
        }
        let reduced_dim = selector.selected_dims().len();
        if reduced_dim == 0 {
            return Err("selector produced zero selected dimensions");
        }
        let hnsw = Hnsw::<f32, DistCosine>::new(
            m,
            max_elements,
            16, // default max layer
            ef_construction,
            DistCosine,
        );
        Ok(Self {
            selector,
            reduced_dim,
            hnsw,
            full_store: Vec::with_capacity(max_elements),
            metric,
        })
    }

    /// Train an [`EmlDistanceModel`] from a representative sample of full-dim
    /// vectors, then return a ready-to-build index.
    ///
    /// The sample does not have to be the full corpus — 500–2000 vectors from
    /// the same distribution is typical. Training time is ~10 ms on commodity
    /// hardware for 1k samples at 128 dims.
    pub fn train_and_build(
        samples: &[Vec<f32>],
        selected_k: usize,
        metric: EmlMetric,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self, &'static str> {
        if samples.len() < 100 {
            return Err("need at least 100 samples to train the selector");
        }
        let full_dim = samples[0].len();
        let mut selector = EmlDistanceModel::new(full_dim, selected_k);

        // Pair samples against themselves to generate (a, b, dist) triples.
        // Keep this to ~2n pairs to stay deterministic and fast.
        for chunk in samples.chunks(2) {
            if chunk.len() < 2 {
                break;
            }
            let a = &chunk[0];
            let b = &chunk[1];
            let d = cosine_distance_f32(a, b);
            selector.record(a, b, d);
        }
        // Top up with cross-pairs if the first pass produced fewer than 100.
        let needed = 100usize.saturating_sub(selector.sample_count());
        if needed > 0 {
            let stride = (samples.len() / (needed + 2)).max(1);
            let mut recorded = 0;
            let mut i = 0;
            while recorded < needed && i + stride < samples.len() {
                let a = &samples[i];
                let b = &samples[i + stride];
                let d = cosine_distance_f32(a, b);
                selector.record(a, b, d);
                recorded += 1;
                i += 1;
            }
        }
        let _ = selector.train();
        Self::new(selector, metric, max_elements, m, ef_construction)
    }

    /// Number of dimensions after projection.
    pub fn reduced_dim(&self) -> usize {
        self.reduced_dim
    }

    /// Selected dimension indices (borrowed from the trained model).
    pub fn selected_dims(&self) -> &[usize] {
        self.selector.selected_dims()
    }

    /// Insert a full-dim vector. Returns the 1-based id assigned by `hnsw_rs`.
    pub fn add(&mut self, full: &[f32]) -> usize {
        let reduced = project_vector(full, self.selector.selected_dims());
        let id = self.full_store.len() + 1;
        self.full_store.push(full.to_vec());
        // hnsw_rs wants &[f32] with a caller-supplied id. We use our running id.
        self.hnsw.insert((&reduced, id));
        id
    }

    /// Bulk insert. Returns the vector of assigned ids (order-preserving).
    pub fn add_batch(&mut self, fulls: &[Vec<f32>]) -> Vec<usize> {
        let mut ids = Vec::with_capacity(fulls.len());
        for v in fulls {
            ids.push(self.add(v));
        }
        ids
    }

    /// Parallel bulk insert via hnsw_rs::parallel_insert. At 1M scale this is
    /// order-of-magnitude faster than the serial `add_batch` because the
    /// projections parallelize across cores and the HNSW graph build itself
    /// is rayon-parallel inside `hnsw_rs`. Projection and id assignment run
    /// upfront so ids stay order-preserving relative to `fulls`.
    pub fn add_batch_parallel(&mut self, fulls: &[Vec<f32>]) -> Vec<usize> {
        use rayon::prelude::*;
        let start_id = self.full_store.len() + 1;
        let ids: Vec<usize> = (start_id..start_id + fulls.len()).collect();
        // Project all vectors in parallel.
        let dims = self.selector.selected_dims();
        let projected: Vec<Vec<f32>> = fulls
            .par_iter()
            .map(|v| project_vector(v, dims))
            .collect();
        // Clone fulls into the side store sequentially (cheap compared to HNSW build).
        self.full_store.extend_from_slice(fulls);
        // Feed the projections into hnsw_rs::parallel_insert.
        let refs: Vec<(&Vec<f32>, usize)> = projected
            .iter()
            .zip(ids.iter().copied())
            .map(|(v, id)| (v, id))
            .collect();
        self.hnsw.parallel_insert(&refs);
        ids
    }

    /// Search returning top `k` results by projected-cosine distance.
    pub fn search(&self, query_full: &[f32], k: usize, ef_search: usize) -> Vec<EmlSearchResult> {
        let reduced = project_vector(query_full, self.selector.selected_dims());
        let neighbors = self.hnsw.search(&reduced, k, ef_search);
        neighbors
            .into_iter()
            .map(|n| EmlSearchResult {
                id: n.d_id,
                distance: n.distance,
            })
            .collect()
    }

    /// Search + exact re-rank: pull `fetch_k` candidates from the reduced
    /// index, then re-order with full-dim cosine, and return top `k`.
    ///
    /// This is the "approx then exact" pattern — the reduced index narrows
    /// the candidate set cheaply, the re-rank restores ground-truth ordering.
    pub fn search_with_rerank(
        &self,
        query_full: &[f32],
        k: usize,
        fetch_k: usize,
        ef_search: usize,
    ) -> Vec<EmlSearchResult> {
        self.search_with_rerank_inner(query_full, k, fetch_k, ef_search, true)
    }

    /// Serial-rerank variant retained for A/B benchmarking. Equivalent to the
    /// v2 implementation that walked candidates with a plain `for` loop. Not
    /// intended for production paths — `search_with_rerank` is parallel by
    /// default and should be preferred.
    pub fn search_with_rerank_serial(
        &self,
        query_full: &[f32],
        k: usize,
        fetch_k: usize,
        ef_search: usize,
    ) -> Vec<EmlSearchResult> {
        self.search_with_rerank_inner(query_full, k, fetch_k, ef_search, false)
    }

    /// Search + exact re-rank: pull `fetch_k` candidates from the reduced
    /// index, then re-order with full-dim cosine, and return top `k`.
    ///
    /// This is the "approx then exact" pattern — the reduced index narrows
    /// the candidate set cheaply, the re-rank restores ground-truth ordering.
    ///
    /// When `parallel` is true, the full-dim cosine pass is dispatched across
    /// rayon's global pool. The inner kernel touches only `&self.full_store`
    /// (a `Sync` borrow of `Vec<Vec<f32>>`) and writes back to
    /// `cands[i].distance` through a `par_iter_mut`, so no extra locking is
    /// needed.
    fn search_with_rerank_inner(
        &self,
        query_full: &[f32],
        k: usize,
        fetch_k: usize,
        ef_search: usize,
        parallel: bool,
    ) -> Vec<EmlSearchResult> {
        let fetch = fetch_k.max(k);
        let mut cands = self.search(query_full, fetch, ef_search);
        let full_store: &[Vec<f32>] = &self.full_store;
        let metric = self.metric;
        let rerank = |c: &mut EmlSearchResult| {
            let stored = &full_store[c.id - 1];
            c.distance = match metric {
                // SimSIMD-backed AVX/NEON cosine kernel; falls back to the scalar
                // reference impl if the runtime does not support it.
                EmlMetric::Cosine => cosine_distance_simd(query_full, stored),
            };
        };
        if parallel {
            cands.par_iter_mut().for_each(rerank);
        } else {
            cands.iter_mut().for_each(rerank);
        }
        cands.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(k);
        cands
    }

    /// How many vectors have been inserted.
    pub fn len(&self) -> usize {
        self.full_store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.full_store.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_skewed(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        // Deterministic LCG; variance concentrated in first 32 dims so the
        // correlation-based selector has signal to find.
        let mut s = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v = Vec::with_capacity(dim);
            for d in 0..dim {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f32 / u32::MAX as f32) - 0.5;
                let scale = if d < 32 { 4.0 } else { 0.3 };
                v.push(u * scale);
            }
            out.push(v);
        }
        out
    }

    #[test]
    fn build_and_search_returns_nearest() {
        let data = make_skewed(500, 128, 42);
        let mut idx = EmlHnsw::train_and_build(
            &data[..200],
            32,
            EmlMetric::Cosine,
            1024,
            16,
            100,
        )
        .expect("build");
        idx.add_batch(&data);
        let q = &data[17];
        let hits = idx.search(q, 5, 50);
        assert_eq!(hits.len(), 5);
        // The query itself should be the nearest (id = 18 because 1-based).
        assert_eq!(hits[0].id, 18);
    }

    #[test]
    fn rerank_preserves_top1_identity() {
        let data = make_skewed(400, 128, 7);
        let mut idx = EmlHnsw::train_and_build(
            &data[..200],
            32,
            EmlMetric::Cosine,
            1024,
            16,
            100,
        )
        .expect("build");
        idx.add_batch(&data);
        let q = &data[42];
        let hits = idx.search_with_rerank(q, 3, 10, 50);
        assert_eq!(hits[0].id, 43);
        assert!(hits[0].distance < 1e-5, "self-query distance must be ~0, got {}", hits[0].distance);
    }

    #[test]
    fn selected_dims_length_matches_config() {
        let data = make_skewed(300, 64, 99);
        let idx = EmlHnsw::train_and_build(
            &data[..200],
            16,
            EmlMetric::Cosine,
            512,
            12,
            64,
        )
        .expect("build");
        assert_eq!(idx.reduced_dim(), 16);
        assert_eq!(idx.selected_dims().len(), 16);
    }
}
