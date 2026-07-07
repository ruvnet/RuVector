//! Compact HNSW graph data structure.
//!
//! `HnswGraph` stores only the graph topology (neighbor lists per layer).
//! All vector data lives outside in the concrete index types; distances are
//! provided via a two-argument closure `dist_fn(i, j) → f32` so the same
//! algorithm serves f32, sq8, and rerank variants without code duplication.
//!
//! Using a two-argument closure (vs. a single-argument "distance to new
//! node" closure) is essential for correct neighbor pruning: when node A's
//! neighbor list exceeds M after a reverse-link insertion, we re-select the
//! M closest neighbors using A's own distances — not the inserting node's.
//!
//! Algorithm follows Malkov & Yashunin 2018 (Algorithm 1-4).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

// ---------------------------------------------------------------------------
// Ordered float wrapper
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Graph topology node
// ---------------------------------------------------------------------------

struct GraphNode {
    neighbors: Vec<Vec<usize>>,
}

// ---------------------------------------------------------------------------
// HNSW graph
// ---------------------------------------------------------------------------

/// HNSW graph topology.
pub struct HnswGraph {
    nodes: Vec<GraphNode>,
    entry_point: Option<usize>,
    max_layer: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,
    rng: u64,
    ml: f64,
}

impl HnswGraph {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            m,
            m0: 2 * m,
            ef_construction,
            rng: 0xDEAD_BEEF_1234_5678_u64,
            ml: 1.0 / (m as f64).ln(),
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    // ------------------------------------------------------------------
    // Insert
    // ------------------------------------------------------------------

    /// Insert a new node with internal ID `new_id`.
    ///
    /// `dist_fn(i, j)` returns the distance between any two stored nodes
    /// `i` and `j` (or between the new node and existing node `j` when
    /// `i == new_id`).
    ///
    /// Node IDs must be assigned sequentially starting from 0.
    pub fn insert_node(&mut self, new_id: usize, dist_fn: impl Fn(usize, usize) -> f32) {
        let level = self.random_level();

        while self.nodes.len() <= new_id {
            self.nodes.push(GraphNode {
                neighbors: Vec::new(),
            });
        }
        self.nodes[new_id].neighbors = vec![vec![]; level + 1];

        let Some(mut ep) = self.entry_point else {
            self.entry_point = Some(new_id);
            self.max_layer = level;
            return;
        };

        // Precompute: distance from new_id to any existing node
        let dist_to_new = |j: usize| dist_fn(new_id, j);

        // Greedy descent through layers above the new node's level
        for lc in (level + 1..=self.max_layer).rev() {
            let candidates = self.search_layer(ep, 1, lc, &dist_to_new);
            if let Some(&(_, best)) = candidates.first() {
                ep = best;
            }
        }

        // Build connections at each layer the new node participates in
        for lc in (0..=level.min(self.max_layer)).rev() {
            let m = if lc == 0 { self.m0 } else { self.m };
            let candidates = self.search_layer(ep, self.ef_construction, lc, &dist_to_new);

            let nb_ids: Vec<usize> = candidates.iter().take(m).map(|&(_, id)| id).collect();
            self.nodes[new_id].neighbors[lc].extend_from_slice(&nb_ids);

            for &nb in &nb_ids {
                let nb_node = &mut self.nodes[nb];
                if nb_node.neighbors.len() > lc {
                    nb_node.neighbors[lc].push(new_id);
                    // Prune to m using distance from nb (not from the new node)
                    if nb_node.neighbors[lc].len() > m {
                        let neighbors_snapshot = nb_node.neighbors[lc].clone();
                        let mut scored: Vec<(f32, usize)> = neighbors_snapshot
                            .into_iter()
                            .map(|x| (dist_fn(nb, x), x))
                            .collect();
                        scored.sort_by(|a, b| {
                            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        scored.truncate(m);
                        self.nodes[nb].neighbors[lc] = scored.into_iter().map(|(_, x)| x).collect();
                    }
                }
            }

            if let Some(&(_, best)) = candidates.first() {
                ep = best;
            }
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(new_id);
        }
    }

    // ------------------------------------------------------------------
    // Search
    // ------------------------------------------------------------------

    /// Approximate k-NN search.
    ///
    /// `query_dist(j)` returns distance from the query to stored node `j`.
    /// Returns `(distance, node_id)` sorted by distance ascending.
    pub fn search(
        &self,
        ef_search: usize,
        k: usize,
        query_dist: impl Fn(usize) -> f32,
    ) -> Vec<(f32, usize)> {
        let Some(mut ep) = self.entry_point else {
            return vec![];
        };

        // Greedy descent to layer 1
        for lc in (1..=self.max_layer).rev() {
            let candidates = self.search_layer(ep, 1, lc, &query_dist);
            if let Some(&(_, best)) = candidates.first() {
                ep = best;
            }
        }

        let mut results = self.search_layer(ep, ef_search, 0, &query_dist);
        results.truncate(k);
        results
    }

    // ------------------------------------------------------------------
    // Internal search layer
    // ------------------------------------------------------------------

    fn search_layer(
        &self,
        ep: usize,
        ef: usize,
        layer: usize,
        dist_fn: &impl Fn(usize) -> f32,
    ) -> Vec<(f32, usize)> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut w: BinaryHeap<(OrdF32, usize)> = BinaryHeap::new(); // max-heap
        let mut c: BinaryHeap<Reverse<(OrdF32, usize)>> = BinaryHeap::new(); // min-heap

        let d_ep = dist_fn(ep);
        w.push((OrdF32(d_ep), ep));
        c.push(Reverse((OrdF32(d_ep), ep)));
        visited.insert(ep);

        while let Some(Reverse((c_dist, c_id))) = c.pop() {
            let f_dist = w.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
            if c_dist.0 > f_dist {
                break;
            }

            let node = match self.nodes.get(c_id) {
                Some(n) => n,
                None => continue,
            };
            let neighbors = match node.neighbors.get(layer) {
                Some(nb) => nb,
                None => continue,
            };

            for &e in neighbors {
                if visited.insert(e) {
                    let e_dist = dist_fn(e);
                    let f_dist = w.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
                    if e_dist < f_dist || w.len() < ef {
                        c.push(Reverse((OrdF32(e_dist), e)));
                        w.push((OrdF32(e_dist), e));
                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(f32, usize)> = w.into_iter().map(|(d, id)| (d.0, id)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    // ------------------------------------------------------------------
    // Level assignment
    // ------------------------------------------------------------------

    fn random_level(&mut self) -> usize {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let u = (self.rng as f64) / (u64::MAX as f64);
        let level = (-u.ln() * self.ml).floor() as usize;
        level.min(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l2_sq;

    fn build_f32_graph(vecs: &[Vec<f32>], m: usize, ef: usize) -> HnswGraph {
        let mut g = HnswGraph::new(m, ef);
        for (i, _) in vecs.iter().enumerate() {
            g.insert_node(i, |a, b| l2_sq(&vecs[a], &vecs[b]));
        }
        g
    }

    #[test]
    fn single_insert_search() {
        let vecs: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();
        let g = build_f32_graph(&vecs, 8, 40);
        let query = vec![25.0_f32, 0.0, 0.0, 0.0];
        let results = g.search(20, 1, |j| l2_sq(&query, &vecs[j]));
        assert!(!results.is_empty());
        assert_eq!(results[0].1, 25, "expected node 25 as NN");
    }

    #[test]
    fn top5_recall_high() {
        let vecs: Vec<Vec<f32>> = (0..500)
            .map(|i| vec![(i % 50) as f32, (i / 50) as f32])
            .collect();
        let g = build_f32_graph(&vecs, 16, 100);

        let mut hits = 0usize;
        let k = 5;
        for i in 0..20 {
            let query = &vecs[i * 25];
            let approx: Vec<usize> = g
                .search(40, k, |j| l2_sq(query, &vecs[j]))
                .into_iter()
                .map(|(_, id)| id)
                .collect();
            let mut exact: Vec<(f32, usize)> = vecs
                .iter()
                .enumerate()
                .map(|(j, v)| (l2_sq(query, v), j))
                .collect();
            exact.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let ground: Vec<usize> = exact.into_iter().take(k).map(|(_, id)| id).collect();
            hits += approx.iter().filter(|id| ground.contains(id)).count();
        }
        let recall = hits as f64 / (20 * k) as f64;
        assert!(recall >= 0.70, "recall@5 = {:.3} < 0.70", recall);
    }
}
