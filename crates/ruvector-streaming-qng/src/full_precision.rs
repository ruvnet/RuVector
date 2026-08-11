//! Baseline: brute-force f32 linear scan (ground truth for recall measurement).

use crate::{AnnVariant, Hit, sq_l2};

pub struct FullPrecision {
    vectors: Vec<Vec<f32>>,
}

impl FullPrecision {
    pub fn new() -> Self {
        Self { vectors: Vec::new() }
    }
}

impl AnnVariant for FullPrecision {
    fn build(&mut self, vectors: &[Vec<f32>]) {
        self.vectors.extend_from_slice(vectors);
    }

    fn insert(&mut self, vector: Vec<f32>) {
        self.vectors.push(vector);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut hits: Vec<Hit> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(id, v)| Hit { id, dist: sq_l2(query, v) })
            .collect();
        hits.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        hits.truncate(k);
        hits
    }

    fn name(&self) -> &str { "FullPrecision" }
    fn len(&self) -> usize { self.vectors.len() }
    fn memory_bytes(&self) -> usize {
        self.vectors.iter().map(|v| v.len() * 4).sum()
    }
}
