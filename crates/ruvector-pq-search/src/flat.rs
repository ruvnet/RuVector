//! Flat PQ Index: linear ADC scan over all encoded vectors (baseline variant).
//!
//! Memory: O(n × M) bytes for codes + O(M × K × d) for codebook.
//! Query: O(M × K) for ADC table + O(n × M) for scan.

use crate::{codebook::PqCodebook, encoder::encode_vector, PqSearch, SearchResult};

pub struct FlatPqIndex {
    codebook: PqCodebook,
    /// PQ codes: each vector stored as M bytes.
    codes: Vec<Vec<u8>>,
}

impl FlatPqIndex {
    pub fn new(codebook: PqCodebook) -> Self {
        Self {
            codebook,
            codes: Vec::new(),
        }
    }
}

impl PqSearch for FlatPqIndex {
    fn insert(&mut self, vector: &[f32]) {
        self.codes.push(encode_vector(&self.codebook, vector));
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let table = self.codebook.build_adc_table(query);
        let mut results: Vec<SearchResult> = self
            .codes
            .iter()
            .enumerate()
            .map(|(id, code)| SearchResult {
                id,
                distance: self.codebook.adc_distance(&table, code),
            })
            .collect();
        results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        results.truncate(k);
        results
    }

    fn memory_bytes(&self) -> usize {
        self.codebook.memory_bytes()
            + self.codes.len() * self.codebook.config.m
            + std::mem::size_of::<Self>()
    }

    fn name(&self) -> &'static str {
        "FlatPQ"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::{PqCodebook, PqConfig};

    fn build_index(n: usize, dim: usize) -> FlatPqIndex {
        let config = PqConfig::new(4, 16);
        let train_data: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let cb = PqCodebook::train(config, &train_data, dim);
        let mut idx = FlatPqIndex::new(cb);
        for i in 0..n {
            idx.insert(&train_data[i * dim..(i + 1) * dim]);
        }
        idx
    }

    #[test]
    fn search_returns_k_results() {
        let idx = build_index(200, 32);
        let query: Vec<f32> = (0..32).map(|i| (i as f32).cos()).collect();
        let results = idx.search(&query, 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn results_are_sorted_ascending() {
        let idx = build_index(200, 32);
        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let results = idx.search(&query, 10);
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn memory_bytes_positive() {
        let idx = build_index(100, 32);
        assert!(idx.memory_bytes() > 0);
    }
}
