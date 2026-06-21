//! Residual-Corrected PQ Index: PQ scan + exact L2 re-score for top-k candidates.
//!
//! Alternative B: run FlatPQ ADC to shortlist `k × oversampling` candidates,
//! then re-score using the stored full-precision residual vectors.
//!
//! Recall improvement: residuals capture the quantization error, so re-scoring
//! on `residual + reconstructed` approaches exact L2. Extra memory per vector:
//! M bytes (code) + dim × 4 bytes (residual).

use crate::{
    codebook::PqCodebook,
    encoder::{decode_vector, encode_vector},
    l2_sq, PqSearch, SearchResult,
};

pub struct ResidualPqIndex {
    codebook: PqCodebook,
    /// PQ codes for fast ADC prescreening.
    codes: Vec<Vec<u8>>,
    /// Per-vector residuals: v - decode(encode(v)).
    residuals: Vec<Vec<f32>>,
    /// Oversampling factor: scan k × oversampling candidates, then re-rank.
    oversampling: usize,
}

impl ResidualPqIndex {
    /// `oversampling`: how many extra candidates to fetch for exact re-ranking.
    pub fn new(codebook: PqCodebook, oversampling: usize) -> Self {
        Self {
            codebook,
            codes: Vec::new(),
            residuals: Vec::new(),
            oversampling: oversampling.max(1),
        }
    }
}

impl PqSearch for ResidualPqIndex {
    fn insert(&mut self, vector: &[f32]) {
        let code = encode_vector(&self.codebook, vector);
        let reconstructed = decode_vector(&self.codebook, &code);
        // Residual = original - reconstructed.
        let residual: Vec<f32> = vector
            .iter()
            .zip(reconstructed.iter())
            .map(|(v, r)| v - r)
            .collect();
        self.codes.push(code);
        self.residuals.push(residual);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let table = self.codebook.build_adc_table(query);
        let n_candidates = (k * self.oversampling).min(self.codes.len());

        // ADC prescreening: get top n_candidates.
        let mut prescreened: Vec<(usize, f32)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(id, code)| (id, self.codebook.adc_distance(&table, code)))
            .collect();
        prescreened.sort_by(|a, b| a.1.total_cmp(&b.1));
        prescreened.truncate(n_candidates);

        // Exact re-score using residual correction: dist(q, v) ≈ dist(q, r + v̂)
        // where v̂ = decode(code) and r = residual.
        let mut results: Vec<SearchResult> = prescreened
            .into_iter()
            .map(|(id, _)| {
                let residual = &self.residuals[id];
                // Rebuild approximate original: reconstructed + residual = original.
                // Distance to query after residual correction:
                // L2²(q, v) = L2²(q, reconstructed + residual)
                let code = &self.codes[id];
                let reconstructed = decode_vector(&self.codebook, code);
                let corrected: Vec<f32> = reconstructed
                    .iter()
                    .zip(residual.iter())
                    .map(|(r, e)| r + e)
                    .collect();
                SearchResult {
                    id,
                    distance: l2_sq(query, &corrected),
                }
            })
            .collect();

        results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        results.truncate(k);
        results
    }

    fn memory_bytes(&self) -> usize {
        let codes_mem = self.codes.len() * self.codebook.config.m;
        let residuals_mem = self.residuals.len() * self.codebook.dim * 4;
        self.codebook.memory_bytes() + codes_mem + residuals_mem + std::mem::size_of::<Self>()
    }

    fn name(&self) -> &'static str {
        "ResidualPQ"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::{PqCodebook, PqConfig};

    fn build_residual_index(n: usize, dim: usize) -> ResidualPqIndex {
        let config = PqConfig::new(4, 16);
        let train: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.01).cos()).collect();
        let cb = PqCodebook::train(config, &train, dim);
        let mut idx = ResidualPqIndex::new(cb, 4);
        for i in 0..n {
            idx.insert(&train[i * dim..(i + 1) * dim]);
        }
        idx
    }

    #[test]
    fn residual_search_returns_k_results() {
        let idx = build_residual_index(200, 32);
        let query: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let results = idx.search(&query, 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn residual_results_sorted_ascending() {
        let idx = build_residual_index(200, 32);
        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.05).collect();
        let results = idx.search(&query, 10);
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn residual_corrected_distance_is_exact() {
        // For a single vector: residual PQ with residual stored should return
        // the exact original vector, so re-scored distance = exact L2².
        let dim = 16usize;
        let config = PqConfig::new(4, 4); // tiny codebook
        let train: Vec<f32> = (0..50 * dim).map(|i| i as f32).collect();
        let cb = PqCodebook::train(config, &train, dim);
        let mut idx = ResidualPqIndex::new(cb, 1);
        let v: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5).collect();
        idx.insert(&v);
        let results = idx.search(&v, 1);
        // Distance from v to itself is 0.
        assert_eq!(results.len(), 1);
        assert!(
            results[0].distance.abs() < 1e-5,
            "expected ~0, got {}",
            results[0].distance
        );
    }
}
