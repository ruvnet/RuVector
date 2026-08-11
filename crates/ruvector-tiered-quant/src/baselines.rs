//! Baseline index variants used for recall measurement and benchmarking.

use crate::Hit;
use crate::quantizer::ScalarQuantizer;

/// Flat f32 index; ground-truth baseline for recall measurement.
pub struct FlatF32Index {
    dims: usize,
    vectors: Vec<(usize, Vec<f32>)>,
}

impl FlatF32Index {
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            vectors: Vec::new(),
        }
    }

    pub fn insert(&mut self, id: usize, vector: Vec<f32>) {
        self.vectors.push((id, vector));
    }

    pub fn query(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut results: Vec<Hit> = self
            .vectors
            .iter()
            .map(|(id, v)| {
                let dist = query
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                Hit { id: *id, dist }
            })
            .collect();
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        results.truncate(k);
        results
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// All-warm index: every vector stored as u8 scalar quantized. No tiering.
pub struct FlatU8Index {
    dims: usize,
    sq: ScalarQuantizer,
    entries: Vec<(usize, Vec<u8>, Vec<f32>, Vec<f32>)>,
}

impl FlatU8Index {
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            sq: ScalarQuantizer::new(dims),
            entries: Vec::new(),
        }
    }

    pub fn insert(&mut self, id: usize, vector: Vec<f32>) {
        let (bytes, min, scale) = self.sq.quantize(&vector);
        self.entries.push((id, bytes, min, scale));
    }

    pub fn query(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut results: Vec<Hit> = self
            .entries
            .iter()
            .map(|(id, bytes, min, scale)| {
                let dist: f32 = query
                    .iter()
                    .enumerate()
                    .map(|(i, &q)| {
                        let r = min[i] + bytes[i] as f32 * scale[i];
                        (q - r).powi(2)
                    })
                    .sum::<f32>()
                    .sqrt();
                Hit { id: *id, dist }
            })
            .collect();
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        results.truncate(k);
        results
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Compute recall@k: fraction of ground-truth top-k found in candidate top-k.
pub fn recall_at_k(ground_truth: &[Hit], candidates: &[Hit], k: usize) -> f32 {
    let gt_ids: std::collections::HashSet<usize> =
        ground_truth.iter().take(k).map(|h| h.id).collect();
    let found = candidates
        .iter()
        .take(k)
        .filter(|h| gt_ids.contains(&h.id))
        .count();
    found as f32 / k.min(ground_truth.len()) as f32
}
