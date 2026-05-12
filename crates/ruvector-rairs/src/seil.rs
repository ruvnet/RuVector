//! Variant 3 — RairsSeil: full RAIRS with SEIL block layout.
//!
//! SEIL (Shared-cell Enhanced IVF Lists) groups each inverted list into
//! 32-vector **blocks**.  When a vector appears in two lists (due to RAIR
//! secondary assignment), its block is stored once in the *lower-indexed*
//! list; the higher-indexed list holds a `BlockRef` pointing to that block
//! instead of duplicating the data.
//!
//! At query time a `u64`-bitset tracks visited blocks so each block is
//! scored at most once, eliminating redundant distance computations and
//! keeping the cache footprint tight.
//!
//! Memory overhead vs. RairsStrict: −(~50 % of secondary copies) because
//! each shared block is stored once.

use crate::error::RairsError;
use crate::index::{l2sq, AnnIndex, SearchResult};
use crate::kmeans;

const BLOCK_SIZE: usize = 32;

/// One block of up to BLOCK_SIZE (vector_id, raw_vector) pairs.
#[derive(Debug, Clone)]
struct Block {
    entries: Vec<(usize, Vec<f32>)>,
}

/// Either owned data (primary list) or a reference into another list.
#[derive(Debug, Clone)]
enum ListBlock {
    Owned(Block),
    Ref { list_idx: usize, block_idx: usize },
}

/// Full RAIRS: SRAIR dual assignment + SEIL shared-block layout.
#[derive(Debug, Clone)]
pub struct RairsSeil {
    dim: usize,
    nclusters: usize,
    max_iter: usize,
    seed: u64,
    /// Amplification factor λ for the RAIR scoring metric (paper default 1.0).
    pub lambda: f32,
    centroids: Vec<Vec<f32>>,
    /// Per-cluster list of blocks.
    lists: Vec<Vec<ListBlock>>,
    total: usize,
}

impl RairsSeil {
    /// Create a new untrained RairsSeil index.
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

    /// Compute the RAIR score (same formula as RairsStrict).
    #[inline]
    fn rair_score(&self, v: &[f32], c_j: &[f32], r_p: &[f32]) -> f32 {
        let mut l2 = 0.0f32;
        let mut inner = 0.0f32;
        for d in 0..v.len() {
            let diff = v[d] - c_j[d];
            l2 += diff * diff;
            inner += r_p[d] * diff;
        }
        l2 + self.lambda * inner
    }

    fn secondary_centroid(&self, v: &[f32], primary: usize) -> usize {
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

    /// Append `entry` to list `list_idx`, creating a new block if the last
    /// block is full. Returns (list_idx, block_idx) of the placement.
    fn append_owned(&mut self, list_idx: usize, entry: (usize, Vec<f32>)) -> (usize, usize) {
        let list = &mut self.lists[list_idx];
        if list.is_empty() {
            list.push(ListBlock::Owned(Block {
                entries: vec![entry],
            }));
        } else {
            let last = list.len() - 1;
            match &mut list[last] {
                ListBlock::Owned(b) if b.entries.len() < BLOCK_SIZE => {
                    b.entries.push(entry);
                }
                _ => {
                    list.push(ListBlock::Owned(Block {
                        entries: vec![entry],
                    }));
                }
            }
        }
        let bidx = self.lists[list_idx].len() - 1;
        (list_idx, bidx)
    }

    /// Append a Ref block to `secondary_list`, pointing at (primary_list, block_idx).
    fn append_ref(&mut self, secondary_list: usize, primary_list: usize, block_idx: usize) {
        self.lists[secondary_list].push(ListBlock::Ref {
            list_idx: primary_list,
            block_idx,
        });
    }

    /// Resolve a block: follow the (at most one-hop) Ref chain to its owned data.
    fn resolve_block(&self, list_idx: usize, block_idx: usize) -> &Block {
        match &self.lists[list_idx][block_idx] {
            ListBlock::Owned(b) => b,
            ListBlock::Ref {
                list_idx: li,
                block_idx: bi,
            } => self.resolve_block(*li, *bi),
        }
    }

    /// Canonical `(owning_list, block)` identity used to dedup visits.
    fn block_key(&self, list_idx: usize, block_idx: usize) -> (usize, usize) {
        match &self.lists[list_idx][block_idx] {
            ListBlock::Owned(_) => (list_idx, block_idx),
            ListBlock::Ref {
                list_idx: li,
                block_idx: bi,
            } => (*li, *bi),
        }
    }

    /// Per-query prefix sums so a canonical `(li, bi)` block key maps to a flat
    /// index into a `Vec<bool>` visited array (cheaper than a `HashSet`).
    fn block_offsets(&self) -> (Vec<usize>, usize) {
        let mut offsets = Vec::with_capacity(self.lists.len() + 1);
        let mut acc = 0usize;
        for list in &self.lists {
            offsets.push(acc);
            acc += list.len();
        }
        offsets.push(acc);
        (offsets, acc)
    }
}

impl AnnIndex for RairsSeil {
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

            // Always store the owned copy in the lower-indexed list.
            let (owned_list, owned_block) = if primary <= secondary {
                let (l, b) = self.append_owned(primary, (self.total, v.clone()));
                if secondary != primary {
                    self.append_ref(secondary, l, b);
                }
                (l, b)
            } else {
                let (l, b) = self.append_owned(secondary, (self.total, v.clone()));
                self.append_ref(primary, l, b);
                (l, b)
            };
            let _ = (owned_list, owned_block);
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
        // Visited-block dedup: each shared block is scored at most once.
        // Flat bool array indexed via per-list prefix sums — one memset per
        // query instead of a growing HashMap.
        let (offsets, n_blocks) = self.block_offsets();
        let mut visited = vec![false; n_blocks];
        let mut cands: Vec<SearchResult> = Vec::new();

        for ci in crate::index::top_nprobe_centroids(query, &self.centroids, nprobe) {
            for bi in 0..self.lists[ci].len() {
                let (kli, kbi) = self.block_key(ci, bi);
                let flat = offsets[kli] + kbi;
                if !visited[flat] {
                    visited[flat] = true;
                    for (id, vec) in &self.resolve_block(ci, bi).entries {
                        cands.push(SearchResult {
                            id: *id,
                            distance: l2sq(query, vec).sqrt(),
                        });
                    }
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
    fn seil_self_match() {
        let vecs = corpus(200, 16, 3);
        let mut idx = RairsSeil::new(16, 8, 20, 42, 1.0);
        idx.train(&vecs).unwrap();
        idx.add(&vecs).unwrap();
        let results = idx.search(&vecs[0], 1, idx.num_lists()).unwrap();
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn seil_block_dedup_no_duplicate_ids() {
        let vecs = corpus(100, 8, 11);
        let mut idx = RairsSeil::new(8, 4, 20, 42, 1.0);
        idx.train(&vecs).unwrap();
        idx.add(&vecs).unwrap();
        // Full-probe search — each vector ID should appear at most once
        let results = idx.search(&vecs[50], 100, idx.num_lists()).unwrap();
        let mut ids: Vec<usize> = results.iter().map(|r| r.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), results.len(), "duplicate IDs found");
    }

    #[test]
    fn seil_matches_rairs_strict_top1() {
        use crate::rairs::RairsStrict;
        let vecs = corpus(200, 16, 77);
        let mut seil = RairsSeil::new(16, 8, 20, 42, 1.0);
        seil.train(&vecs).unwrap();
        seil.add(&vecs).unwrap();
        let mut strict = RairsStrict::new(16, 8, 20, 42, 1.0);
        strict.train(&vecs).unwrap();
        strict.add(&vecs).unwrap();
        for q in &vecs[0..10] {
            let r1 = seil.search(q, 1, seil.num_lists()).unwrap();
            let r2 = strict.search(q, 1, strict.num_lists()).unwrap();
            assert_eq!(r1[0].id, r2[0].id, "SEIL and strict disagree on top-1");
        }
    }
}
