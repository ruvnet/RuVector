//! Self-contained HNSW graph with per-node deletion flags.
//!
//! Implements the HNSW algorithm (Malkov & Yashunin 2018) from scratch so that
//! the internal edge structure is fully accessible to the repair strategies in
//! [`crate::strategy`].

use crate::l2_sq;
use std::collections::{BinaryHeap, HashSet};

/// HNSW construction and search parameters.
#[derive(Clone, Debug)]
pub struct HnswConfig {
    /// Vector dimensionality.
    pub dim: usize,
    /// Max neighbours per node at levels > 0.
    pub m: usize,
    /// Max neighbours per node at level 0 (typically 2*M).
    pub m0: usize,
    /// Candidate-list size during construction.
    pub ef_construction: usize,
    /// Level multiplier: 1 / ln(M).
    pub ml: f64,
}

impl HnswConfig {
    pub fn new(dim: usize) -> Self {
        let m = 16;
        Self {
            dim,
            m,
            m0: m * 2,
            ef_construction: 100,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

/// HNSW graph with deletion flags.
pub struct HnswGraph {
    pub config: HnswConfig,
    /// All inserted vectors (index = node id).
    pub vectors: Vec<Vec<f32>>,
    /// `deleted[id]` is true after the node is removed from search.
    pub deleted: Vec<bool>,
    /// `node_level[id]` = highest layer this node participates in.
    pub node_level: Vec<usize>,
    /// `layers[level][id]` = neighbour list at that level.
    /// Only levels 0..=node_level[id] are populated.
    pub layers: Vec<Vec<Vec<u32>>>,
    /// Current graph entry point (highest-level node id).
    pub entry: Option<u32>,
    /// PRNG state (LCG) for deterministic level sampling.
    rng: u64,
}

impl HnswGraph {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            vectors: Vec::new(),
            deleted: Vec::new(),
            node_level: Vec::new(),
            layers: vec![Vec::new()], // level 0
            entry: None,
            rng: 0xDEAD_BEEF_CAFE_1234,
        }
    }

    /// Insert a vector and return its node id.
    pub fn insert(&mut self, v: Vec<f32>) -> u32 {
        let id = self.vectors.len() as u32;
        self.vectors.push(v);
        self.deleted.push(false);

        let level = self.random_level();
        self.node_level.push(level);

        // Extend layer arrays if necessary.
        while self.layers.len() <= level {
            self.layers.push(Vec::new());
        }
        for l in 0..=level {
            // Extend each level's per-node array.
            while self.layers[l].len() <= id as usize {
                self.layers[l].push(Vec::new());
            }
        }

        if self.entry.is_none() {
            self.entry = Some(id);
            return id;
        }

        let mut ep = self.entry.unwrap();
        let top = self.node_level[ep as usize];

        // Upper layers: greedy descend to find better ep.
        for lc in ((level + 1)..=top).rev() {
            let better = self.greedy_search_one(ep, &self.vectors[id as usize].clone(), lc);
            ep = better;
        }

        // Insert layer by layer from min(level, top) down to 0.
        for lc in (0..=level.min(top)).rev() {
            let m_max = if lc == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let candidates = self.search_layer_ef(
                ep,
                &self.vectors[id as usize].clone(),
                self.config.ef_construction,
                lc,
            );
            let neighbors = self.select_neighbors(&candidates, m_max);

            // Connect id → neighbors and neighbors → id (bidirectional).
            while self.layers[lc].len() <= id as usize {
                self.layers[lc].push(Vec::new());
            }
            self.layers[lc][id as usize] = neighbors.clone();

            for &nb in &neighbors {
                while self.layers[lc].len() <= nb as usize {
                    self.layers[lc].push(Vec::new());
                }
                if !self.layers[lc][nb as usize].contains(&id) {
                    self.layers[lc][nb as usize].push(id);
                    // Prune if over capacity.
                    if self.layers[lc][nb as usize].len() > m_max {
                        self.prune_neighbors(nb as usize, m_max, lc);
                    }
                }
            }

            // Update ep to the closest candidate for next level.
            if let Some(&(_, best)) = candidates.first() {
                ep = best;
            }
        }

        // Update entry point if this node is higher.
        if level > top {
            self.entry = Some(id);
        }

        id
    }

    /// Search for the `k` nearest live neighbours of `query`.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<u32> {
        let Some(ep) = self.entry else {
            return Vec::new();
        };
        if self.deleted[ep as usize] {
            // Find a live entry point.
            let Some(live_ep) = self.find_live_entry() else {
                return Vec::new();
            };
            return self.search_from(live_ep, query, k, ef);
        }
        self.search_from(ep, query, k, ef)
    }

    fn search_from(&self, ep: u32, query: &[f32], k: usize, ef: usize) -> Vec<u32> {
        let top = self.node_level[ep as usize];
        let mut cur_ep = ep;

        // Upper layers: greedy descent.
        for lc in (1..=top).rev() {
            cur_ep = self.greedy_search_one(cur_ep, query, lc);
        }

        // Level 0: full ef search.
        let candidates = self.search_layer_ef(cur_ep, query, ef.max(k), 0);
        candidates.iter().take(k).map(|(_, id)| *id).collect()
    }

    /// Greedy 1-best walk at a given layer (used during insertion).
    pub fn greedy_search_one(&self, start: u32, query: &[f32], level: usize) -> u32 {
        let dim = self.config.dim;
        let mut best = start;
        let mut best_dist = l2_sq(query, &self.vectors[start as usize], dim);

        loop {
            let mut changed = false;
            if level >= self.layers.len() {
                break;
            }
            let neighbors = if (best as usize) < self.layers[level].len() {
                self.layers[level][best as usize].clone()
            } else {
                break;
            };
            for nb in neighbors {
                if self.deleted[nb as usize] {
                    continue;
                }
                let d = l2_sq(query, &self.vectors[nb as usize], dim);
                if d < best_dist {
                    best_dist = d;
                    best = nb;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        best
    }

    /// ef-bounded search at a given layer. Returns (dist, id) sorted ascending by dist.
    pub fn search_layer_ef(
        &self,
        start: u32,
        query: &[f32],
        ef: usize,
        level: usize,
    ) -> Vec<(f32, u32)> {
        let dim = self.config.dim;
        let mut visited = HashSet::new();

        // Min-heap: (dist, id) — candidates to explore.
        let mut candidates: BinaryHeap<_> = BinaryHeap::new();
        // Max-heap: ef-size working set.
        let mut working: BinaryHeap<_> = BinaryHeap::new();

        if self.deleted[start as usize] {
            if let Some(live) = self.find_live_entry() {
                let d = l2_sq(query, &self.vectors[live as usize], dim);
                candidates.push(std::cmp::Reverse(MinEntry(d, live)));
                working.push(MaxEntry(d, live));
                visited.insert(live);
            } else {
                return Vec::new();
            }
        } else {
            let d = l2_sq(query, &self.vectors[start as usize], dim);
            candidates.push(std::cmp::Reverse(MinEntry(d, start)));
            working.push(MaxEntry(d, start));
            visited.insert(start);
        }

        while let Some(std::cmp::Reverse(MinEntry(dist_c, c))) = candidates.pop() {
            let worst = working.peek().map(|MaxEntry(d, _)| *d).unwrap_or(f32::MAX);
            if dist_c > worst && working.len() >= ef {
                break;
            }
            let neighbors = if level < self.layers.len() && (c as usize) < self.layers[level].len()
            {
                self.layers[level][c as usize].clone()
            } else {
                Vec::new()
            };
            for nb in neighbors {
                if self.deleted[nb as usize] {
                    continue;
                }
                if visited.contains(&nb) {
                    continue;
                }
                visited.insert(nb);
                let d = l2_sq(query, &self.vectors[nb as usize], dim);
                let cur_worst = working
                    .peek()
                    .map(|MaxEntry(dw, _)| *dw)
                    .unwrap_or(f32::MAX);
                if d < cur_worst || working.len() < ef {
                    candidates.push(std::cmp::Reverse(MinEntry(d, nb)));
                    working.push(MaxEntry(d, nb));
                    if working.len() > ef {
                        working.pop();
                    }
                }
            }
        }

        let mut result: Vec<(f32, u32)> =
            working.into_iter().map(|MaxEntry(d, id)| (d, id)).collect();
        result.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        result
    }

    fn select_neighbors(&self, candidates: &[(f32, u32)], m: usize) -> Vec<u32> {
        candidates.iter().take(m).map(|(_, id)| *id).collect()
    }

    fn prune_neighbors(&mut self, node: usize, m_max: usize, level: usize) {
        if level >= self.layers.len() || node >= self.layers[level].len() {
            return;
        }
        let neighbors = self.layers[level][node].clone();
        let dim = self.config.dim;
        let qv = self.vectors[node].clone();
        let mut scored: Vec<(f32, u32)> = neighbors
            .into_iter()
            .filter(|&nb| !self.deleted[nb as usize])
            .map(|nb| (l2_sq(&qv, &self.vectors[nb as usize], dim), nb))
            .collect();
        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.layers[level][node] = scored.into_iter().take(m_max).map(|(_, id)| id).collect();
    }

    fn random_level(&mut self) -> usize {
        self.rng = self
            .rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = (self.rng >> 33) as f64 / (u32::MAX as f64);
        let level = (-f.ln() * self.config.ml).floor() as usize;
        level.min(32) // cap at 32 levels
    }

    fn find_live_entry(&self) -> Option<u32> {
        self.vectors.iter().enumerate().find_map(|(i, _)| {
            if !self.deleted[i] {
                Some(i as u32)
            } else {
                None
            }
        })
    }

    /// Count live (non-deleted) nodes.
    pub fn live_count(&self) -> usize {
        self.deleted.iter().filter(|&&d| !d).count()
    }
}

// Newtype wrappers for BinaryHeap ordering.
#[derive(PartialEq)]
struct MinEntry(f32, u32);
impl Eq for MinEntry {}
impl PartialOrd for MinEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[derive(PartialEq)]
struct MaxEntry(f32, u32);
impl Eq for MaxEntry {}
impl PartialOrd for MaxEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: highest distance at top so pop() removes the worst candidate.
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_f32(s: &mut u64) -> f32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 33) as f32 / (u32::MAX as f32)
    }

    #[test]
    fn insert_and_search_returns_results() {
        let cfg = HnswConfig::new(8);
        let mut g = HnswGraph::new(cfg);
        let mut rng = 1u64;
        for _ in 0..200 {
            let v: Vec<f32> = (0..8).map(|_| lcg_f32(&mut rng)).collect();
            g.insert(v);
        }
        let q: Vec<f32> = (0..8).map(|_| lcg_f32(&mut rng)).collect();
        let results = g.search(&q, 5, 20);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn deleted_nodes_absent_from_search() {
        let cfg = HnswConfig::new(8);
        let mut g = HnswGraph::new(cfg);
        let mut rng = 2u64;
        for _ in 0..100 {
            let v: Vec<f32> = (0..8).map(|_| lcg_f32(&mut rng)).collect();
            g.insert(v);
        }
        for i in 0..30 {
            g.deleted[i] = true;
        }
        let q: Vec<f32> = (0..8).map(|_| lcg_f32(&mut rng)).collect();
        let results = g.search(&q, 5, 20);
        for id in results {
            assert!(!g.deleted[id as usize]);
        }
    }
}
