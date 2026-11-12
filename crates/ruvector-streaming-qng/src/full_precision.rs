//! Baseline: brute-force f32 linear scan (ground truth for recall measurement).

use crate::{sq_l2, AnnVariant, Hit};

pub struct FullPrecision {
    vectors: Vec<Vec<f32>>,
}

impl Default for FullPrecision {
    fn default() -> Self {
        Self::new()
    }
}

impl FullPrecision {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }
}

impl AnnVariant for FullPrecision {
    fn build(&mut self, vectors: &[Vec<f32>]) {
        self.vectors = vectors.to_vec();
    }

    fn insert(&mut self, vector: Vec<f32>) {
        self.vectors.push(vector);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        if query.iter().any(|value| !value.is_finite()) {
            return vec![];
        }
        let mut hits: Vec<Hit> = self
            .vectors
            .iter()
            .enumerate()
            .filter_map(|(id, v)| {
                let dist = sq_l2(query, v);
                dist.is_finite().then_some(Hit { id, dist })
            })
            .collect();
        hits.sort_unstable();
        hits.truncate(k);
        hits
    }

    fn name(&self) -> &str {
        "FullPrecision"
    }
    fn len(&self) -> usize {
        self.vectors.len()
    }
    fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
    fn memory_bytes(&self) -> usize {
        self.vectors.iter().map(|v| v.len() * 4).sum()
    }
}
