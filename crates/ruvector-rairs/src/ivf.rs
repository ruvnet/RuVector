//! Variant 1 — IvfFlat: classic single-assignment IVF with flat list scan.
//!
//! Each vector is assigned to exactly one centroid. Search probes the
//! `nprobe` closest centroids and linearly scans each list.

use crate::error::RairsError;
use crate::index::{l2sq, AnnIndex, SearchResult};
use crate::kmeans;

/// IVF baseline: one list per vector, flat scan.
#[derive(Debug, Clone)]
pub struct IvfFlat {
    dim: usize,
    nclusters: usize,
    max_iter: usize,
    seed: u64,
    /// Trained centroids (nclusters × dim).
    centroids: Vec<Vec<f32>>,
    /// Per-cluster: list of (vector_id, raw_vector).
    lists: Vec<Vec<(usize, Vec<f32>)>>,
    total: usize,
}

impl IvfFlat {
    /// Create a new untrained IvfFlat index.
    ///
    /// * `dim`       — vector dimensionality
    /// * `nclusters` — number of Voronoi cells (Voronoi = k-means clusters)
    /// * `max_iter`  — k-means max iterations
    /// * `seed`      — RNG seed for reproducibility
    pub fn new(dim: usize, nclusters: usize, max_iter: usize, seed: u64) -> Self {
        Self {
            dim,
            nclusters,
            max_iter,
            seed,
            centroids: Vec::new(),
            lists: Vec::new(),
            total: 0,
        }
    }

    /// Train centroids on the given corpus. Must be called before `add`.
    pub fn train(&mut self, corpus: &[Vec<f32>]) -> Result<(), RairsError> {
        if corpus.is_empty() {
            return Err(RairsError::EmptyCorpus);
        }
        if corpus[0].len() != self.dim {
            return Err(RairsError::DimMismatch {
                expected: self.dim,
                got: corpus[0].len(),
            });
        }
        let k = self.nclusters.min(corpus.len());
        let (centroids, _) = kmeans::train(corpus, k, self.max_iter, self.seed);
        self.centroids = centroids;
        self.lists = vec![Vec::new(); k];
        Ok(())
    }
}

impl AnnIndex for IvfFlat {
    fn add(&mut self, vectors: &[Vec<f32>]) -> Result<(), RairsError> {
        if self.centroids.is_empty() {
            return Err(RairsError::NotTrained);
        }
        for v in vectors {
            if v.len() != self.dim {
                return Err(RairsError::DimMismatch {
                    expected: self.dim,
                    got: v.len(),
                });
            }
            let c = kmeans::nearest_centroid(v, &self.centroids);
            self.lists[c].push((self.total, v.clone()));
            self.total += 1;
        }
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<SearchResult>, RairsError> {
        if self.centroids.is_empty() {
            return Err(RairsError::NotTrained);
        }
        if query.len() != self.dim {
            return Err(RairsError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        // Collect candidates from the top-nprobe lists, then partial-select top-k.
        let mut cands: Vec<SearchResult> = Vec::new();
        for ci in crate::index::top_nprobe_centroids(query, &self.centroids, nprobe) {
            for (id, vec) in &self.lists[ci] {
                cands.push(SearchResult {
                    id: *id,
                    distance: l2sq(query, vec).sqrt(),
                });
            }
        }
        Ok(crate::index::finalize_topk(cands, k))
    }

    fn len(&self) -> usize {
        self.total
    }

    fn num_lists(&self) -> usize {
        self.centroids.len()
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn basic_search_returns_k_results() {
        let n = 200;
        let dim = 16;
        let vecs = corpus(n, dim, 1);
        let mut idx = IvfFlat::new(dim, 8, 20, 42);
        idx.train(&vecs).unwrap();
        idx.add(&vecs).unwrap();
        assert_eq!(idx.len(), n);
        let results = idx.search(&vecs[0], 5, 4).unwrap();
        assert!(results.len() <= 5);
        // Exact self-match must be first (distance ≈ 0)
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 1e-5);
    }

    #[test]
    fn full_probe_gives_exact_results() {
        let n = 100;
        let dim = 8;
        let vecs = corpus(n, dim, 7);
        let mut idx = IvfFlat::new(dim, 4, 20, 42);
        idx.train(&vecs).unwrap();
        idx.add(&vecs).unwrap();
        // With nprobe = nclusters, should get exact top-1
        let results = idx.search(&vecs[42], 1, idx.num_lists()).unwrap();
        assert_eq!(results[0].id, 42);
    }
}
