//! AnisotropicResidual: AQ + f32 residual re-rank.
//!
//! Phase 1 (ADC scan): retrieve `overfetch * k` candidates by AQ ADC score.
//! Phase 2 (re-rank): compute exact inner product against stored f32 vectors
//!   for the candidate set, return true top-k.
//!
//! This trades memory (full f32 copies) for higher recall at low extra latency,
//! since the re-rank computes only overfetch*k exact dot products.

use crate::codebook::{AqCodebook, AqConfig};
use crate::flat_aq::{encode, flat_scan, normalize_rows};
use crate::{dot, l2_norm, AqSearch, SearchResult};

pub struct AnisotropicResidual {
    codebook: AqCodebook,
    codes: Vec<Vec<u8>>,
    /// Stored normalised f32 vectors for exact re-rank.
    vectors: Vec<Vec<f32>>,
    dim: usize,
    /// How many candidates to retrieve before re-ranking.
    overfetch: usize,
}

impl AnisotropicResidual {
    /// `overfetch`: multiplier on k for first-pass candidate retrieval (e.g. 4).
    pub fn new(
        dim: usize,
        m: usize,
        k: usize,
        eta: f32,
        overfetch: usize,
        train_vectors: &[f32],
    ) -> Self {
        let cfg = AqConfig::anisotropic(m, k, eta);
        let normalized = normalize_rows(train_vectors, dim);
        let codebook = AqCodebook::train(cfg, &normalized, dim);
        Self {
            codebook,
            codes: Vec::new(),
            vectors: Vec::new(),
            dim,
            overfetch,
        }
    }
}

impl AqSearch for AnisotropicResidual {
    fn insert(&mut self, vector: &[f32]) {
        let norm = l2_norm(vector);
        let nv: Vec<f32> = vector.iter().map(|x| x / norm).collect();
        let code = encode(&self.codebook, &nv);
        self.codes.push(code);
        self.vectors.push(nv);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let norm = l2_norm(query);
        let nq: Vec<f32> = query.iter().map(|x| x / norm).collect();

        // Phase 1: AQ ADC scan for overfetch*k candidates.
        let fetch = (k * self.overfetch).min(self.codes.len());
        let candidates = flat_scan(&self.codebook, &self.codes, &nq, fetch);

        // Phase 2: exact re-rank over candidates.
        let mut exact: Vec<(usize, f32)> = candidates
            .iter()
            .map(|r| (r.id, dot(&nq, &self.vectors[r.id])))
            .collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1));
        exact
            .into_iter()
            .take(k)
            .map(|(id, score)| SearchResult { id, score })
            .collect()
    }

    fn memory_bytes(&self) -> usize {
        self.codebook.memory_bytes()
            + self.codes.len() * self.codebook.config.m
            + self.vectors.len() * self.dim * 4
    }

    fn name(&self) -> &'static str {
        "AnisotropicResidual"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn norm_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                let nrm = l2_norm(&v);
                v.iter().map(|x| x / nrm).collect()
            })
            .collect()
    }

    #[test]
    fn residual_aq_searches_correctly() {
        let dim = 64;
        let train_flat: Vec<f32> = norm_vecs(200, dim, 30).into_iter().flatten().collect();
        let mut idx = AnisotropicResidual::new(dim, 8, 16, 2.0, 4, &train_flat);
        let db = norm_vecs(100, dim, 31);
        for v in &db {
            idx.insert(v);
        }
        let q = &db[42];
        let results = idx.search(q, 5);
        assert!(results[0].id == 42, "identical query should be top result");
        assert!(results[0].score > 0.99, "score of self should be near 1.0");
    }

    #[test]
    fn residual_memory_grows_with_vectors() {
        let dim = 32;
        let train_flat: Vec<f32> = norm_vecs(100, dim, 40).into_iter().flatten().collect();
        let mut idx = AnisotropicResidual::new(dim, 4, 16, 2.0, 4, &train_flat);
        let v: Vec<f32> = vec![1.0; dim];
        idx.insert(&v);
        let mem1 = idx.memory_bytes();
        idx.insert(&v);
        let mem2 = idx.memory_bytes();
        assert!(mem2 > mem1);
    }

    #[test]
    fn overfetch_greater_than_k_gives_better_recall() {
        let dim = 64;
        let train_flat: Vec<f32> = norm_vecs(300, dim, 50).into_iter().flatten().collect();
        let db = norm_vecs(200, dim, 51);

        let mut idx_low = AnisotropicResidual::new(dim, 8, 16, 2.0, 1, &train_flat);
        let mut idx_high = AnisotropicResidual::new(dim, 8, 16, 2.0, 8, &train_flat);
        for v in &db {
            idx_low.insert(v);
            idx_high.insert(v);
        }

        use crate::{recall_at_k, ExactSearch};
        let mut exact = ExactSearch::new();
        for v in &db {
            exact.insert(v);
        }

        let k = 10;
        let q = &db[5];
        let gt = exact.search_exact(q, k);
        let r_low = idx_low.search(q, k);
        let r_high = idx_high.search(q, k);
        let ids_low: Vec<usize> = r_low.iter().map(|r| r.id).collect();
        let ids_high: Vec<usize> = r_high.iter().map(|r| r.id).collect();
        let recall_low = recall_at_k(&ids_low, &gt);
        let recall_high = recall_at_k(&ids_high, &gt);
        // High overfetch should not be worse (and is typically better).
        assert!(
            recall_high >= recall_low,
            "overfetch=8 recall {} should be >= overfetch=1 recall {}",
            recall_high,
            recall_low
        );
    }
}
