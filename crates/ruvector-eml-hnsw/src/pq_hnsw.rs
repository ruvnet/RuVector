//! PQ-native HNSW: the underlying hnsw_rs::Hnsw stores `Vec<u8>` PQ codes.
//!
//! This is the actual 64× memory-reduced index that the v2 `PqEmlHnsw`
//! advertises but does not yet achieve (v2 stores reconstructed floats in
//! the graph — 512 bytes/vector for SIFT1M — so the 64× number is
//! training-side only). This module stores only the 8-byte code per graph
//! node.
//!
//! # How
//!
//! `hnsw_rs::Hnsw<T, D>` is generic over both `T` (payload) and `D`
//! (distance). We instantiate it with `T = u8` and a custom distance
//! [`PqAsymmetricDistance`] that speaks the PQ distance language natively:
//!
//! - At **insert / graph-build time**, every `eval(code_a, code_b)` call is
//!   between two stored PQ codes. We compute symmetric PQ distance via a
//!   precomputed `sym_tables[s][c_a][c_b]` per-subspace table (built once
//!   from the trained centroids). Cost is `n_subspaces` adds per eval.
//!
//! - At **search time**, every `eval(q_code, stored_code)` call has the
//!   query code in the first slot. We've precomputed the float-query
//!   asymmetric lookup table `asym_table[s][c]` via
//!   `PqCodebook::build_query_table` and stowed it on the distance object
//!   behind an `Arc<RwLock<_>>`. The eval reads the table per-subspace.
//!   Again `n_subspaces` adds per eval.
//!
//! The switch between symmetric and asymmetric is controlled by whether
//! the asymmetric table is currently set on the distance. `search()` sets
//! the table, runs, and clears it. Everything done during `insert()` sees
//! a cleared table -> symmetric path.
//!
//! # Runtime bytes/vec
//!
//! Per-node storage in the graph is `code_bytes = n_subspaces` (8 for
//! SIFT1M-M8). Plus the HNSW neighbor list and a reference-count wrapper
//! (per-node Arc<Point<u8>> overhead from hnsw_rs itself — same as any
//! hnsw_rs index). The v2 `PqEmlHnswLegacy` stored the reconstructed float
//! vector = 512 bytes/vec on top of those same edges.
//!
//! We keep a side `full_store: Vec<Vec<f32>>` for optional exact rerank,
//! identical to v2. That is a pipeline cost, not a graph cost: in a
//! deployed read-only system you drop the full_store and keep only codes.

use crate::cosine_decomp::cosine_distance_f32;
use crate::opq::OpqRotation;
use crate::pq::{self, PqCodebook};
use crate::pq_corrector::PqDistanceCorrector;
use anndists::dist::distances::Distance;
use hnsw_rs::prelude::{DistL2, Hnsw};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

#[derive(Clone, Debug)]
pub struct PqSearchResult {
    pub id: usize,
    pub distance: f32,
}

/// Distance function used by the PQ-native HNSW.
///
/// Holds:
///   - `sym_tables` — `n_subspaces` of `nc × nc` f32 (centroid-pair squared-L2),
///     used for symmetric distance between two graph-stored codes.
///   - `asym_table` — `n_subspaces * nc` f32, one entry per (subspace, centroid),
///     set per-query before calling `hnsw.search`. When set, eval uses
///     `asym_table[s * nc + stored_code[s]]` summed over s.
pub struct PqAsymmetricDistance {
    n_subspaces: usize,
    nc: usize,
    sym_tables: Arc<Vec<Vec<f32>>>,
    asym_table: Arc<RwLock<Option<Vec<f32>>>>,
}

impl PqAsymmetricDistance {
    /// Build from a trained codebook by materializing the symmetric
    /// centroid-pair distance tables.
    pub fn from_codebook(cb: &PqCodebook) -> Self {
        let nc = cb.n_centroids as usize;
        let mut sym_tables: Vec<Vec<f32>> = Vec::with_capacity(cb.n_subspaces);
        for s in 0..cb.n_subspaces {
            let mut t = vec![0.0f32; nc * nc];
            for a in 0..nc {
                for b in 0..nc {
                    let ca = &cb.centroids[s][a];
                    let cb_vec = &cb.centroids[s][b];
                    let mut d = 0.0f32;
                    for i in 0..cb.sub_dim {
                        let diff = ca[i] - cb_vec[i];
                        d += diff * diff;
                    }
                    t[a * nc + b] = d;
                }
            }
            sym_tables.push(t);
        }
        Self {
            n_subspaces: cb.n_subspaces,
            nc,
            sym_tables: Arc::new(sym_tables),
            asym_table: Arc::new(RwLock::new(None)),
        }
    }

    /// Clone that shares `sym_tables` and `asym_table` with the parent.
    fn share(&self) -> Self {
        Self {
            n_subspaces: self.n_subspaces,
            nc: self.nc,
            sym_tables: Arc::clone(&self.sym_tables),
            asym_table: Arc::clone(&self.asym_table),
        }
    }

    /// Install a per-query asymmetric table. Must match `n_subspaces * nc`.
    fn set_query_table(&self, table: Vec<f32>) {
        debug_assert_eq!(table.len(), self.n_subspaces * self.nc);
        *self.asym_table.write().unwrap() = Some(table);
    }

    fn clear_query_table(&self) {
        *self.asym_table.write().unwrap() = None;
    }
}

impl Distance<u8> for PqAsymmetricDistance {
    #[inline]
    fn eval(&self, va: &[u8], vb: &[u8]) -> f32 {
        let guard = self.asym_table.read().unwrap();
        if let Some(table) = guard.as_ref() {
            // Asymmetric: table carries query-to-centroid distance.
            // We use vb (stored code); va is the query-coded placeholder.
            let mut sum = 0.0f32;
            for s in 0..self.n_subspaces {
                sum += table[s * self.nc + vb[s] as usize];
            }
            return sum;
        }
        drop(guard);
        // Symmetric path (insert / graph-build).
        let mut sum = 0.0f32;
        for s in 0..self.n_subspaces {
            let a = va[s] as usize;
            let b = vb[s] as usize;
            sum += self.sym_tables[s][a * self.nc + b];
        }
        sum
    }
}

/// PQ-native HNSW — PqEmlHnsw v3, the real compressed index.
pub struct PqEmlHnsw {
    codebook: PqCodebook,
    rotation: Option<OpqRotation>,
    hnsw: Hnsw<'static, u8, PqAsymmetricDistance>,
    dist_handle: PqAsymmetricDistance,
    codes: Vec<Vec<u8>>,
    full_store: Vec<Vec<f32>>,
    corrector: Option<PqDistanceCorrector>,
}

impl PqEmlHnsw {
    pub fn new(
        codebook: PqCodebook,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        let dist_handle = PqAsymmetricDistance::from_codebook(&codebook);
        let hnsw = Hnsw::<u8, PqAsymmetricDistance>::new(
            m,
            max_elements,
            16,
            ef_construction,
            dist_handle.share(),
        );
        Self {
            codebook,
            rotation: None,
            hnsw,
            dist_handle,
            codes: Vec::with_capacity(max_elements),
            full_store: Vec::with_capacity(max_elements),
            corrector: None,
        }
    }

    pub fn with_rotation(mut self, rotation: OpqRotation) -> Self {
        assert_eq!(
            rotation.dim,
            self.codebook.dim(),
            "OPQ rotation dim must match codebook dim"
        );
        self.rotation = Some(rotation);
        self
    }

    /// One-shot builder for plain PQ (no OPQ rotation).
    pub fn train_and_build(
        samples: &[Vec<f32>],
        n_subspaces: usize,
        n_centroids: u16,
        iters: usize,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        let codebook = pq::train(samples, n_subspaces, n_centroids, iters);
        Self::new(codebook, max_elements, m, ef_construction)
    }

    /// One-shot builder with full parametric OPQ rotation (PCA eigen +
    /// balanced-variance permutation) trained on the same samples.
    pub fn train_and_build_opq(
        samples: &[Vec<f32>],
        n_subspaces: usize,
        n_centroids: u16,
        iters: usize,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        let rotation = crate::opq::train_rotation(samples, n_subspaces);
        let rotated: Vec<Vec<f32>> = rotation.apply_batch(samples);
        let codebook = pq::train(&rotated, n_subspaces, n_centroids, iters);
        let mut idx = Self::new(codebook, max_elements, m, ef_construction);
        idx.rotation = Some(rotation);
        idx
    }

    /// One-shot builder with permutation-only OPQ (OPQ-NP). Preserves
    /// natural feature groupings (better for SIFT-like data) while still
    /// balancing variance across subspaces.
    pub fn train_and_build_opq_np(
        samples: &[Vec<f32>],
        n_subspaces: usize,
        n_centroids: u16,
        iters: usize,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        let rotation = crate::opq::train_permutation_only(samples, n_subspaces);
        let rotated: Vec<Vec<f32>> = rotation.apply_batch(samples);
        let codebook = pq::train(&rotated, n_subspaces, n_centroids, iters);
        let mut idx = Self::new(codebook, max_elements, m, ef_construction);
        idx.rotation = Some(rotation);
        idx
    }

    pub fn set_corrector(&mut self, corrector: PqDistanceCorrector) {
        self.corrector = Some(corrector);
    }

    pub fn codebook(&self) -> &PqCodebook {
        &self.codebook
    }
    pub fn rotation(&self) -> Option<&OpqRotation> {
        self.rotation.as_ref()
    }

    pub fn code_bytes_per_vec(&self) -> usize {
        self.codebook.code_bytes()
    }

    /// Runtime bytes/vec stored inside the HNSW graph per node payload.
    ///
    /// In the PQ-native design this is `n_subspaces` bytes (8 for SIFT1M-M8).
    /// For the legacy v2 design it was `dim * 4` = 512 bytes (reconstructed
    /// float). The HNSW edge lists and per-node overhead are identical in
    /// both designs and not counted here.
    pub fn hnsw_payload_bytes_per_vec(&self) -> usize {
        self.codebook.n_subspaces
    }

    fn encode_maybe_rotated(&self, full: &[f32]) -> Vec<u8> {
        if let Some(r) = &self.rotation {
            let y = r.apply(full);
            self.codebook.encode(&y)
        } else {
            self.codebook.encode(full)
        }
    }

    fn build_query_table(&self, query_full: &[f32]) -> Vec<f32> {
        if let Some(r) = &self.rotation {
            self.codebook.build_query_table(&r.apply(query_full))
        } else {
            self.codebook.build_query_table(query_full)
        }
    }

    pub fn add(&mut self, full: &[f32]) -> usize {
        assert_eq!(full.len(), self.codebook.dim(), "add() dim mismatch");
        let code = self.encode_maybe_rotated(full);
        let id = self.full_store.len() + 1;
        self.hnsw.insert((&code[..], id));
        self.codes.push(code);
        self.full_store.push(full.to_vec());
        id
    }

    pub fn add_batch(&mut self, fulls: &[Vec<f32>]) -> Vec<usize> {
        let mut ids = Vec::with_capacity(fulls.len());
        for v in fulls {
            ids.push(self.add(v));
        }
        ids
    }

    pub fn len(&self) -> usize {
        self.full_store.len()
    }
    pub fn is_empty(&self) -> bool {
        self.full_store.is_empty()
    }

    /// PQ-native search. Sets the asymmetric query table on the distance,
    /// runs hnsw.search, then clears the table.
    pub fn search(&self, query_full: &[f32], k: usize, ef_search: usize) -> Vec<PqSearchResult> {
        let table = self.build_query_table(query_full);
        self.dist_handle.set_query_table(table);
        let q_code = self.encode_maybe_rotated(query_full);
        let neighbors = self.hnsw.search(&q_code, k, ef_search);
        self.dist_handle.clear_query_table();
        neighbors
            .into_iter()
            .map(|n| PqSearchResult {
                id: n.d_id,
                distance: n.distance,
            })
            .collect()
    }

    /// Fetch `fetch_k` candidates via PQ-native graph, then re-rank top k
    /// with full-dim cosine against the side store. Optional corrector is
    /// applied to PQ distances first (advisory; cosine rerank is the final
    /// order).
    pub fn search_with_rerank(
        &self,
        query_full: &[f32],
        k: usize,
        fetch_k: usize,
        ef_search: usize,
    ) -> Vec<PqSearchResult> {
        let fetch = fetch_k.max(k);
        let mut cands = self.search(query_full, fetch, ef_search);
        // cands already carry asymmetric PQ distances from the PQ-native
        // search above (set via PqAsymmetricDistance::set_query_table).
        //
        // Apply the learned PQ-error corrector.
        //
        // * **Legacy (global-max) mode**: corrector output is advisory only.
        //   We recompute `c.distance` but then overwrite it with exact cosine
        //   below — this is the old behavior, preserved for backward compat.
        // * **Local-scale mode** (post-SOTA-C fix): corrector output is
        //   NON-ADVISORY — we use it to truncate the candidate set to a
        //   tighter `rerank_k` before exact cosine rerank. The per-query
        //   inference scale is the median PQ distance across the fetched
        //   candidates, which mimics the per-sample exact distance the
        //   model was trained on (within a 2–3× margin the model tolerates).
        let mut corrector_non_advisory = false;
        if let Some(corr) = &self.corrector {
            if corr.is_locally_normalized() {
                corrector_non_advisory = true;
                let mut pqs: Vec<f32> = cands.iter().map(|c| c.distance).collect();
                pqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q_scale = if pqs.is_empty() {
                    corr.median_local_scale() as f32
                } else {
                    pqs[pqs.len() / 2].max(1e-6)
                };
                for c in cands.iter_mut() {
                    c.distance = corr.correct_with_scale(c.distance, 0.0, q_scale);
                }
                cands.sort_by(|a, b| {
                    a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
                });
                // Rerank window: 15× the requested k (or all if fewer). This
                // is where the corrector has teeth — it decides which of
                // the fetch_k candidates survive into the exact rerank
                // stage. 15× is a conservative safety margin: at k=10 with
                // fetch_k=200, we keep 150 candidates (75% of the fetch
                // pool) for exact cosine rerank, so even a mediocre
                // corrector cannot easily evict true top-10 vectors.
                // The corrector still acts on WHICH 50 candidates get
                // dropped — a ~20-25% saving of rerank cost without
                // collapsing recall.
                let rerank_k = (k.saturating_mul(15)).max(k).min(cands.len());
                cands.truncate(rerank_k);
            } else {
                // Legacy advisory path (pre-fix): values overwritten below.
                for c in cands.iter_mut() {
                    c.distance = corr.correct(c.distance, 0.0);
                }
            }
        }
        let _ = corrector_non_advisory; // variable read for clarity in logs.

        // Exact rerank with full-dim cosine. Authoritative final ranker.
        // Rayon-parallelized: the kernel reads immutably from `full_store`
        // and writes back through `par_iter_mut`. When the corrector ran
        // in local-scale mode above, `cands` is already pre-truncated to
        // the 15× rerank window, so the parallel fan-out is over the
        // tighter set.
        let full_store: &[Vec<f32>] = &self.full_store;
        cands.par_iter_mut().for_each(|c| {
            let stored = &full_store[c.id - 1];
            c.distance = cosine_distance_f32(query_full, stored);
        });
        cands.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        cands.truncate(k);
        cands
    }

    pub fn code_of(&self, id: usize) -> &[u8] {
        &self.codes[id - 1]
    }

    pub fn pq_distance_to(&self, query_full: &[f32], id: usize) -> f32 {
        let table = self.build_query_table(query_full);
        self.codebook
            .asymmetric_distance_with_table(&table, &self.codes[id - 1])
    }
}

// ---------------------------------------------------------------------------
// Legacy v2 implementation (reconstructed-float graph). Kept for back-compat
// with the existing sift1m_pq test. Deployed systems should prefer
// `PqEmlHnsw` above.
// ---------------------------------------------------------------------------

pub struct PqEmlHnswLegacy {
    codebook: PqCodebook,
    hnsw: Hnsw<'static, f32, DistL2>,
    codes: Vec<Vec<u8>>,
    full_store: Vec<Vec<f32>>,
    corrector: Option<PqDistanceCorrector>,
}

impl PqEmlHnswLegacy {
    pub fn new(
        codebook: PqCodebook,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        let hnsw = Hnsw::<f32, DistL2>::new(m, max_elements, 16, ef_construction, DistL2);
        Self {
            codebook,
            hnsw,
            codes: Vec::with_capacity(max_elements),
            full_store: Vec::with_capacity(max_elements),
            corrector: None,
        }
    }
    pub fn train_and_build(
        samples: &[Vec<f32>],
        n_subspaces: usize,
        n_centroids: u16,
        iters: usize,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        let codebook = pq::train(samples, n_subspaces, n_centroids, iters);
        Self::new(codebook, max_elements, m, ef_construction)
    }
    pub fn set_corrector(&mut self, corrector: PqDistanceCorrector) {
        self.corrector = Some(corrector);
    }
    pub fn codebook(&self) -> &PqCodebook {
        &self.codebook
    }
    pub fn code_bytes_per_vec(&self) -> usize {
        self.codebook.code_bytes()
    }
    pub fn hnsw_stored_bytes_per_vec(&self) -> usize {
        self.codebook.dim() * std::mem::size_of::<f32>()
    }
    pub fn add(&mut self, full: &[f32]) -> usize {
        assert_eq!(full.len(), self.codebook.dim(), "add() dim mismatch");
        let code = self.codebook.encode(full);
        let recon = self.codebook.reconstruct(&code);
        let id = self.full_store.len() + 1;
        self.codes.push(code);
        self.full_store.push(full.to_vec());
        self.hnsw.insert((&recon, id));
        id
    }
    pub fn add_batch(&mut self, fulls: &[Vec<f32>]) -> Vec<usize> {
        let mut ids = Vec::with_capacity(fulls.len());
        for v in fulls {
            ids.push(self.add(v));
        }
        ids
    }
    pub fn len(&self) -> usize {
        self.full_store.len()
    }
    pub fn is_empty(&self) -> bool {
        self.full_store.is_empty()
    }
    pub fn search(&self, query_full: &[f32], k: usize, ef_search: usize) -> Vec<PqSearchResult> {
        let neighbors = self.hnsw.search(query_full, k, ef_search);
        neighbors
            .into_iter()
            .map(|n| PqSearchResult {
                id: n.d_id,
                distance: n.distance,
            })
            .collect()
    }
    pub fn search_with_rerank(
        &self,
        query_full: &[f32],
        k: usize,
        fetch_k: usize,
        ef_search: usize,
    ) -> Vec<PqSearchResult> {
        let fetch = fetch_k.max(k);
        let mut cands = self.search(query_full, fetch, ef_search);
        let table = self.codebook.build_query_table(query_full);
        for c in cands.iter_mut() {
            let code = &self.codes[c.id - 1];
            c.distance = self.codebook.asymmetric_distance_with_table(&table, code);
        }
        if let Some(corr) = &self.corrector {
            for c in cands.iter_mut() {
                c.distance = corr.correct(c.distance, 0.0);
            }
        }
        for c in cands.iter_mut() {
            let stored = &self.full_store[c.id - 1];
            c.distance = cosine_distance_f32(query_full, stored);
        }
        cands.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        cands.truncate(k);
        cands
    }
    pub fn code_of(&self, id: usize) -> &[u8] {
        &self.codes[id - 1]
    }
    pub fn pq_distance_to(&self, query_full: &[f32], id: usize) -> f32 {
        self.codebook
            .asymmetric_distance(query_full, &self.codes[id - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_skewed(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v = Vec::with_capacity(dim);
            for d in 0..dim {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f32 / u32::MAX as f32) - 0.5;
                let scale = if d < 32 { 4.0 } else { 0.3 };
                v.push(u * scale);
            }
            out.push(v);
        }
        out
    }

    #[test]
    fn pq_native_build_search_self_neighbour() {
        let data = make_skewed(400, 128, 17);
        let mut idx = PqEmlHnsw::train_and_build(&data[..300], 8, 16, 10, 1024, 16, 80);
        idx.add_batch(&data);
        let q = &data[42];
        let hits = idx.search_with_rerank(q, 3, 40, 128);
        assert_eq!(hits[0].id, 43);
        assert!(hits[0].distance < 1e-4);
    }

    #[test]
    fn pq_native_payload_is_n_subspaces_bytes() {
        let data = make_skewed(512, 64, 7);
        let idx = PqEmlHnsw::train_and_build(&data[..256], 8, 16, 5, 512, 16, 64);
        assert_eq!(idx.hnsw_payload_bytes_per_vec(), 8);
        assert_eq!(idx.code_bytes_per_vec(), 8);
    }

    #[test]
    fn pq_native_with_opq_round_trip() {
        let data = make_skewed(400, 128, 11);
        let mut idx = PqEmlHnsw::train_and_build_opq(&data[..300], 8, 16, 10, 1024, 16, 80);
        idx.add_batch(&data);
        let hits = idx.search_with_rerank(&data[100], 3, 40, 128);
        assert_eq!(hits[0].id, 101);
    }

    #[test]
    fn legacy_build_search_self_neighbour() {
        let data = make_skewed(400, 128, 17);
        let mut idx = PqEmlHnswLegacy::train_and_build(&data[..300], 8, 16, 10, 1024, 16, 80);
        idx.add_batch(&data);
        let q = &data[42];
        let hits = idx.search_with_rerank(q, 3, 20, 64);
        assert_eq!(hits[0].id, 43);
        assert!(hits[0].distance < 1e-5);
    }
}
