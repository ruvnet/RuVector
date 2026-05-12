//! Variant 2 — RairsStrict: dual RAIR assignment without block deduplication.
//!
//! Each vector is assigned to a **primary** and a **secondary** list.
//! The secondary centroid is chosen by minimising the RAIR score:
//!
//!   score(c_j) = ‖v − c_j‖² + λ · ⟨r_p, v − c_j⟩
//!
//! where r_p = v − c_primary is the primary residual.  When λ > 0 this
//! penalises secondaries in the same direction as the primary residual,
//! favouring those that cover the opposite side of the Voronoi boundary.
//! λ = 1.0 is the default from the RAIRS paper.
//!
//! At search time both lists are scanned for every probed centroid.
//! A simple `HashSet` deduplicates vector IDs so each candidate is
//! scored at most once.

use crate::error::RairsError;
use crate::index::{l2sq, AnnIndex, SearchResult};
use crate::kmeans;

/// RAIRS with dual assignment, flat lists, query-time hash deduplication.
#[derive(Debug, Clone)]
pub struct RairsStrict {
    dim: usize,
    nclusters: usize,
    max_iter: usize,
    seed: u64,
    /// Amplification factor λ for the RAIR scoring metric.
    pub lambda: f32,
    centroids: Vec<Vec<f32>>,
    /// Per-cluster list of (vector_id, raw_vector).
    lists: Vec<Vec<(usize, Vec<f32>)>>,
    total: usize,
}

impl RairsStrict {
    /// Create a new untrained RairsStrict index.
    ///
    /// `lambda` is the RAIR amplification factor (paper default = 1.0).
    pub fn new(dim: usize, nclusters: usize, max_iter: usize, seed: u64, lambda: f32) -> Self {
        Self {
            dim,
            nclusters,
            max_iter,
            seed,
            lambda,
            centroids: Vec::new(),
            lists: Vec::new(),
            total: 0,
        }
    }

    /// Train centroids. Must be called before `add`.
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

    /// Compute the RAIR score for assigning vector `v` to centroid `c_j`,
    /// given primary residual `r_p = v − c_primary`.
    ///
    /// `score = ‖v − c_j‖² + λ · ⟨r_p, v − c_j⟩` — allocation-free single pass.
    #[inline]
    fn rair_score(&self, v: &[f32], c_j: &[f32], r_p: &[f32]) -> f32 {
        let mut l2 = 0.0f32;
        let mut inner = 0.0f32;
        for ((&vi, &cj), &rp) in v.iter().zip(c_j).zip(r_p) {
            let diff = vi - cj;
            l2 += diff * diff;
            inner += rp * diff;
        }
        l2 + self.lambda * inner
    }

    /// Find the best secondary centroid for `v` given primary index `primary`.
    fn secondary_centroid(&self, v: &[f32], primary: usize) -> usize {
        // Primary residual: r_p = v - c_primary
        let r_p: Vec<f32> = v
            .iter()
            .zip(self.centroids[primary].iter())
            .map(|(a, b)| a - b)
            .collect();

        self.centroids
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != primary)
            .map(|(i, c)| (i, self.rair_score(v, c, &r_p)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

impl AnnIndex for RairsStrict {
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
            let primary = kmeans::nearest_centroid(v, &self.centroids);
            let secondary = if self.centroids.len() > 1 {
                self.secondary_centroid(v, primary)
            } else {
                primary
            };
            self.lists[primary].push((self.total, v.clone()));
            if secondary != primary {
                self.lists[secondary].push((self.total, v.clone()));
            }
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
        // A vector can land in two lists (primary + secondary), so dedup by id.
        // A bool-per-vector scratch array is one cheap memset per query — far
        // cheaper than growing a HashMap on every search call.
        let mut seen = vec![false; self.total];
        let mut cands: Vec<SearchResult> = Vec::new();
        for ci in crate::index::top_nprobe_centroids(query, &self.centroids, nprobe) {
            for (id, vec) in &self.lists[ci] {
                if !seen[*id] {
                    seen[*id] = true;
                    cands.push(SearchResult {
                        id: *id,
                        distance: l2sq(query, vec).sqrt(),
                    });
                }
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
    fn each_vector_appears_at_most_twice() {
        let vecs = corpus(100, 16, 99);
        let mut idx = RairsStrict::new(16, 8, 20, 42, 1.0);
        idx.train(&vecs).unwrap();
        idx.add(&vecs).unwrap();

        let mut appearances = vec![0usize; 100];
        for list in &idx.lists {
            for (id, _) in list {
                appearances[*id] += 1;
            }
        }
        for count in &appearances {
            assert!(*count >= 1 && *count <= 2, "count = {count}");
        }
    }

    #[test]
    fn rairs_strict_self_match() {
        let vecs = corpus(200, 16, 5);
        let mut idx = RairsStrict::new(16, 8, 20, 42, 1.0);
        idx.train(&vecs).unwrap();
        idx.add(&vecs).unwrap();
        let results = idx.search(&vecs[17], 1, idx.num_lists()).unwrap();
        assert_eq!(results[0].id, 17);
    }

    #[test]
    fn rair_score_lambda_zero_equals_l2sq() {
        let idx = RairsStrict::new(4, 2, 10, 0, 0.0);
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let c = vec![0.0f32, 0.0, 0.0, 0.0];
        let r = vec![0.5f32, 0.5, 0.5, 0.5];
        let score = idx.rair_score(&v, &c, &r);
        let expected = l2sq(&v, &c);
        assert!(
            (score - expected).abs() < 1e-5,
            "score={score} expected={expected}"
        );
    }
}
