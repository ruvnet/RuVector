//! IVF+PQ Index: coarse IVF clustering, then PQ within each inverted list.
//!
//! Alternative A: partition the space into `n_lists` Voronoi cells using
//! k-means on raw vectors. At query time, probe the `n_probe` nearest cells
//! and run ADC only within those lists. Reduces scan from O(n) to O(n/n_lists × n_probe).

use crate::{codebook::PqCodebook, encoder::encode_vector, l2_sq, PqSearch, SearchResult};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct IvfPqIndex {
    codebook: PqCodebook,
    /// Coarse centroids: n_lists × dim flat.
    coarse_centroids: Vec<f32>,
    /// Per-list: (original_id, pq_code).
    lists: Vec<Vec<(usize, Vec<u8>)>>,
    n_lists: usize,
    n_probe: usize,
    dim: usize,
    next_id: usize,
}

impl IvfPqIndex {
    /// Build an IVF+PQ index.
    /// `n_lists`: number of Voronoi partitions.
    /// `n_probe`: cells to probe at query time.
    /// `training_vecs`: flat n×dim slice used only to initialise coarse k-means.
    pub fn new(
        codebook: PqCodebook,
        n_lists: usize,
        n_probe: usize,
        training_vecs: &[f32],
    ) -> Self {
        let dim = codebook.dim;
        let n = training_vecs.len() / dim;
        assert!(n >= n_lists, "need at least n_lists training vectors");

        let coarse_centroids = train_coarse(training_vecs, dim, n_lists, 20, 7);

        Self {
            codebook,
            coarse_centroids,
            lists: vec![Vec::new(); n_lists],
            n_lists,
            n_probe,
            dim,
            next_id: 0,
        }
    }

    fn nearest_cell(&self, v: &[f32]) -> usize {
        let mut best = 0;
        let mut best_d = f32::MAX;
        for c in 0..self.n_lists {
            let cent = &self.coarse_centroids[c * self.dim..(c + 1) * self.dim];
            let d = l2_sq(v, cent);
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        best
    }
}

impl PqSearch for IvfPqIndex {
    fn insert(&mut self, vector: &[f32]) {
        let cell = self.nearest_cell(vector);
        let code = encode_vector(&self.codebook, vector);
        self.lists[cell].push((self.next_id, code));
        self.next_id += 1;
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        // Score each coarse centroid against the query.
        let mut cell_scores: Vec<(usize, f32)> = (0..self.n_lists)
            .map(|c| {
                let cent = &self.coarse_centroids[c * self.dim..(c + 1) * self.dim];
                (c, l2_sq(query, cent))
            })
            .collect();
        cell_scores.sort_by(|a, b| a.1.total_cmp(&b.1));

        let table = self.codebook.build_adc_table(query);
        let n_probe = self.n_probe.min(self.n_lists);

        let mut results: Vec<SearchResult> = cell_scores
            .iter()
            .take(n_probe)
            .flat_map(|(c, _)| {
                self.lists[*c].iter().map(|(id, code)| SearchResult {
                    id: *id,
                    distance: self.codebook.adc_distance(&table, code),
                })
            })
            .collect();

        results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        results.dedup_by_key(|r| r.id);
        results.truncate(k);
        results
    }

    fn memory_bytes(&self) -> usize {
        let codes_mem: usize = self
            .lists
            .iter()
            .map(|l| l.len() * self.codebook.config.m)
            .sum();
        self.codebook.memory_bytes()
            + self.coarse_centroids.len() * 4
            + codes_mem
            + std::mem::size_of::<Self>()
    }

    fn name(&self) -> &'static str {
        "IVF+PQ"
    }
}

fn train_coarse(vecs: &[f32], dim: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let n = vecs.len() / dim;
    let mut rng = StdRng::seed_from_u64(seed);

    // Forgy init.
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k.min(n) {
        let j = rng.gen_range(i..n);
        indices.swap(i, j);
    }
    let mut centroids: Vec<f32> = indices
        .iter()
        .take(k)
        .flat_map(|&idx| vecs[idx * dim..(idx + 1) * dim].iter().copied())
        .collect();
    while centroids.len() < k * dim {
        let r = rng.gen_range(0..n);
        centroids.extend_from_slice(&vecs[r * dim..(r + 1) * dim]);
    }
    centroids.truncate(k * dim);

    let mut assignments = vec![0usize; n];

    for _ in 0..iters {
        for i in 0..n {
            let v = &vecs[i * dim..(i + 1) * dim];
            let mut best_c = 0;
            let mut best_d = f32::MAX;
            for c in 0..k {
                let cent = &centroids[c * dim..(c + 1) * dim];
                let d = l2_sq(v, cent);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }

        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dim {
                sums[c * dim + d] += vecs[i * dim + d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c * dim + d] = sums[c * dim + d] * inv;
                }
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::{PqCodebook, PqConfig};

    fn build_ivf_index(n: usize, dim: usize) -> IvfPqIndex {
        let config = PqConfig::new(4, 16);
        let train: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let cb = PqCodebook::train(config, &train, dim);
        let mut idx = IvfPqIndex::new(cb, 8, 2, &train);
        for i in 0..n {
            idx.insert(&train[i * dim..(i + 1) * dim]);
        }
        idx
    }

    #[test]
    fn ivf_search_returns_results() {
        let idx = build_ivf_index(200, 32);
        let query: Vec<f32> = (0..32).map(|i| (i as f32).cos()).collect();
        let results = idx.search(&query, 5);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn ivf_results_sorted_ascending() {
        let idx = build_ivf_index(200, 32);
        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.05).collect();
        let results = idx.search(&query, 10);
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }
}
