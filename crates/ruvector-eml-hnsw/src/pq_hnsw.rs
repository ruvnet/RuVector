//! Tier 3B integration: PqCodebook + hnsw_rs::Hnsw + PqDistanceCorrector.
//!
//! # What this does
//!
//! - Trains a PqCodebook on a representative sample.
//! - Encodes every inserted vector to an 8-byte PQ code (for SIFT1M at 8x256).
//! - Builds an HNSW over the PQ-reconstructed float vectors using the stock
//!   DistL2 metric. The reconstructed vectors are transient graph payload;
//!   we keep only codes + full-dim vectors in our side store.
//! - At search time, for each candidate we:
//!   1. Compute true asymmetric PQ distance via a query-side lookup table,
//!   2. Optionally apply PqDistanceCorrector to refine the approximation,
//!   3. Re-rank top k with full-dim cosine against the side store.
//!
//! # Memory accounting
//!
//! At SIFT1M dim=128, 8 subspaces x 256 centroids, each vector encodes to
//! 8 bytes (code_bytes). The full-dim float copy we keep on the side is
//! 128*4=512 bytes (for rerank). The HNSW graph itself also stores the
//! reconstructed float vector, reported as hnsw_stored_bytes_per_vec for
//! transparency. The target storage cost of PQ for a deployed system is
//! only the PQ code (8 B), which is what we compare against the baseline
//! EmlHnsw (which stores the 512 B float directly in the graph).

use crate::cosine_decomp::cosine_distance_f32;
use crate::pq::{self, PqCodebook};
use crate::pq_corrector::PqDistanceCorrector;
use hnsw_rs::prelude::{DistL2, Hnsw};

#[derive(Clone, Debug)]
pub struct PqSearchResult {
    pub id: usize,
    pub distance: f32,
}

pub struct PqEmlHnsw {
    codebook: PqCodebook,
    hnsw: Hnsw<'static, f32, DistL2>,
    codes: Vec<Vec<u8>>,
    full_store: Vec<Vec<f32>>,
    corrector: Option<PqDistanceCorrector>,
}

impl PqEmlHnsw {
    pub fn new(codebook: PqCodebook, max_elements: usize, m: usize, ef_construction: usize) -> Self {
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
            .map(|n| PqSearchResult { id: n.d_id, distance: n.distance })
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
        for c in cands.iter_mut() {
            let stored = &self.full_store[c.id - 1];
            c.distance = cosine_distance_f32(query_full, stored);
        }
        cands.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(k);
        cands
    }

    pub fn code_of(&self, id: usize) -> &[u8] {
        &self.codes[id - 1]
    }

    pub fn pq_distance_to(&self, query_full: &[f32], id: usize) -> f32 {
        self.codebook.asymmetric_distance(query_full, &self.codes[id - 1])
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
    fn build_search_self_neighbour() {
        let data = make_skewed(400, 128, 17);
        let mut idx = PqEmlHnsw::train_and_build(&data[..300], 8, 16, 10, 1024, 16, 80);
        idx.add_batch(&data);
        let q = &data[42];
        let hits = idx.search_with_rerank(q, 3, 20, 64);
        assert_eq!(hits[0].id, 43);
        assert!(hits[0].distance < 1e-5);
    }

    #[test]
    fn code_bytes_matches_n_subspaces() {
        let data = make_skewed(512, 64, 7);
        let idx = PqEmlHnsw::train_and_build(&data[..256], 8, 16, 5, 512, 16, 64);
        assert_eq!(idx.code_bytes_per_vec(), 8);
    }
}
