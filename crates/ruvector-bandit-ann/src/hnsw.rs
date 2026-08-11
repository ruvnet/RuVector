//! Two-layer HNSW with runtime-configurable ef_search.
//!
//! Layer 1 (top): random subset of nodes (~sqrt(N)), long-range highway.
//! Layer 0 (bottom): all N nodes, fine-grained neighborhood search.
//!
//! ef_search controls the dynamic candidate list at layer 0 — this is the
//! parameter the bandit tunes at runtime.

use crate::sq_l2;
use std::collections::{BinaryHeap, HashSet};

// ─── node ────────────────────────────────────────────────────────────────────

struct Node {
    vector: Vec<f32>,
    /// neighbors[0] = layer-0 neighbors; neighbors[1] = layer-1 neighbors.
    neighbors: [Vec<usize>; 2],
}

// ─── HNSW ────────────────────────────────────────────────────────────────────

/// Two-layer HNSW index with configurable ef_search at query time.
pub struct Hnsw {
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    nodes: Vec<Node>,
    entry: Option<usize>,
    entry_level: usize,
}

impl Hnsw {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            m,
            m_max0: m * 2,
            ef_construction,
            nodes: Vec::new(),
            entry: None,
            entry_level: 0,
        }
    }

    /// Insert a vector.  Level is determined deterministically from the node index.
    pub fn insert(&mut self, _id: usize, vector: &[f32]) {
        let internal = self.nodes.len();
        // Deterministic level: ~1/m probability of being in layer 1.
        let level = if internal > 0 && internal % self.m == 0 {
            1
        } else {
            0
        };

        self.nodes.push(Node {
            vector: vector.to_vec(),
            neighbors: [Vec::new(), Vec::new()],
        });

        if self.entry.is_none() {
            self.entry = Some(0);
            self.entry_level = level;
            return;
        }

        let entry = self.entry.unwrap();
        let mut ep = entry;

        // Phase 1: greedy descent from entry_level down to level+1.
        for lc in (level + 1..=self.entry_level).rev() {
            ep = self.greedy_layer(vector, ep, lc, 1)[0].0;
        }

        // Phase 2: for each layer from min(level, entry_level) down to 0.
        let min_level = level.min(self.entry_level);
        for lc in (0..=min_level).rev() {
            let m_l = if lc == 0 { self.m_max0 } else { self.m };
            let candidates = self.greedy_layer(vector, ep, lc, self.ef_construction);
            ep = candidates[0].0;

            // Select up to m_l nearest as neighbors.
            let selected: Vec<usize> = candidates.iter().take(m_l).map(|&(id, _)| id).collect();

            self.nodes[internal].neighbors[lc] = selected.clone();

            // Bidirectional back-links.
            for &nid in &selected {
                self.nodes[nid].neighbors[lc].push(internal);
                let max_back = if lc == 0 { self.m_max0 } else { self.m };
                if self.nodes[nid].neighbors[lc].len() > max_back {
                    let pivot = self.nodes[nid].vector.clone();
                    let mut nbrs = self.nodes[nid].neighbors[lc].clone();
                    nbrs.sort_by(|&a, &b| {
                        sq_l2(&self.nodes[a].vector, &pivot)
                            .partial_cmp(&sq_l2(&self.nodes[b].vector, &pivot))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    nbrs.truncate(max_back);
                    self.nodes[nid].neighbors[lc] = nbrs;
                }
            }
        }

        // Update global entry if new node has higher level.
        if level > self.entry_level {
            self.entry = Some(internal);
            self.entry_level = level;
        }
    }

    /// Greedy search at a single layer.  Returns (id, dist) sorted ascending.
    fn greedy_layer(
        &self,
        query: &[f32],
        start: usize,
        layer: usize,
        ef: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited: HashSet<usize> = HashSet::new();
        visited.insert(start);

        let d0 = sq_l2(query, &self.nodes[start].vector);
        let mut candidates: BinaryHeap<MinFirst> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxFirst> = BinaryHeap::new();

        candidates.push(MinFirst(start, d0));
        results.push(MaxFirst(start, d0));

        while let Some(MinFirst(id, dist)) = candidates.pop() {
            if let Some(MaxFirst(_, worst)) = results.peek() {
                if dist > *worst && results.len() >= ef {
                    break;
                }
            }

            for &nid in &self.nodes[id].neighbors[layer] {
                if visited.insert(nid) {
                    let nd = sq_l2(query, &self.nodes[nid].vector);
                    candidates.push(MinFirst(nid, nd));
                    results.push(MaxFirst(nid, nd));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<(usize, f32)> = results.into_iter().map(|e| (e.0, e.1)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Return up to k nearest neighbours with `ef_search` candidates at layer 0.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let entry = self.entry.unwrap();
        let mut ep = entry;

        // Greedy descent through upper layers.
        for lc in (1..=self.entry_level).rev() {
            ep = self.greedy_layer(query, ep, lc, 1)[0].0;
        }

        // Full search at layer 0.
        let ef = ef_search.max(k);
        let mut results = self.greedy_layer(query, ep, 0, ef);
        results.truncate(k);
        results
    }

    pub fn memory_bytes(&self) -> usize {
        self.nodes.iter().fold(0, |acc, n| {
            acc + n.vector.len() * 4 + n.neighbors[0].len() * 8 + n.neighbors[1].len() * 8
        })
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ─── heap orderings ──────────────────────────────────────────────────────────

struct MinFirst(usize, f32);

impl PartialEq for MinFirst {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MinFirst {}
impl PartialOrd for MinFirst {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinFirst {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse so BinaryHeap pops smallest distance first.
        other
            .1
            .partial_cmp(&self.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

struct MaxFirst(usize, f32);

impl PartialEq for MaxFirst {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MaxFirst {}
impl PartialOrd for MaxFirst {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxFirst {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Natural order so BinaryHeap pops largest distance first.
        self.1
            .partial_cmp(&other.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(id: usize, dim: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..dim).map(|d| (id * dim + d) as f32 * 0.001).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.into_iter().map(|x| x / norm).collect()
    }

    fn build(n: usize, dim: usize) -> Hnsw {
        let mut h = Hnsw::new(12, 100);
        for i in 0..n {
            h.insert(i, &unit_vec(i, dim));
        }
        h
    }

    #[test]
    fn search_returns_k_results() {
        let h = build(200, 32);
        let q = vec![0.1_f32; 32];
        let res = h.search(&q, 10, 30);
        assert_eq!(res.len(), 10);
    }

    #[test]
    fn nearest_is_near_zero_for_direct_lookup() {
        let mut h = Hnsw::new(12, 100);
        for i in 0..100_usize {
            h.insert(i, &unit_vec(i, 16));
        }
        let target = unit_vec(50, 16);
        h.insert(100, &target);
        let res = h.search(&target, 1, 40);
        assert!(!res.is_empty());
        assert!(res[0].1 < 1e-6, "dist = {}", res[0].1);
    }

    #[test]
    fn higher_ef_yields_same_count() {
        let h = build(300, 64);
        let q = vec![0.5_f32; 64];
        assert_eq!(h.search(&q, 10, 10).len(), 10);
        assert_eq!(h.search(&q, 10, 50).len(), 10);
    }

    #[test]
    fn memory_bytes_positive() {
        let h = build(100, 32);
        assert!(h.memory_bytes() > 0);
    }
}
