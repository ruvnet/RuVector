//! Minimal HNSW implementation used as the cache-key index.
//!
//! Stores query vectors; nearest-neighbor search finds previously-cached queries
//! that are semantically similar to a new query.

use rand::Rng;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone)]
pub struct HnswConfig {
    pub dim: usize,
    pub m: usize,
    pub ef_construction: usize,
}

impl HnswConfig {
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            dim,
            m,
            ef_construction,
        }
    }
}

#[derive(Clone)]
struct Node {
    vector: Vec<f32>,
    /// Neighbors per layer: layers[0] = bottom layer, layers[max] = top layer.
    layers: Vec<Vec<usize>>,
}

pub struct HnswGraph {
    config: HnswConfig,
    nodes: Vec<Node>,
    entry_point: Option<usize>,
    max_layer: usize,
    level_mult: f64,
}

#[inline(always)]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot // assumes L2-normalised inputs
}

#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

impl HnswGraph {
    pub fn new(config: HnswConfig) -> Self {
        let level_mult = 1.0 / (config.m as f64).ln();
        Self {
            config,
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            level_mult,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen::<f64>();
        ((-r.ln()) * self.level_mult).floor() as usize
    }

    /// Search a single layer for the ef nearest neighbors of query starting from entry.
    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited: HashMap<usize, f32> = HashMap::new();
        // max-heap for candidates (negate distance for min-heap simulation)
        let mut candidates: BinaryHeap<(i64, usize)> = BinaryHeap::new();
        let mut results: BinaryHeap<(i64, usize)> = BinaryHeap::new();

        let d = l2_sq(query, &self.nodes[entry].vector);
        visited.insert(entry, d);
        candidates.push(((-d * 1e9) as i64, entry));
        results.push(((d * 1e9) as i64, entry));

        while let Some((neg_d, current)) = candidates.pop() {
            let current_d = -neg_d as f32 / 1e9;
            let worst_result = results
                .peek()
                .map(|(wd, _)| *wd as f32 / 1e9)
                .unwrap_or(f32::MAX);
            if current_d > worst_result && results.len() >= ef {
                break;
            }
            if let Some(node) = self.nodes.get(current) {
                let neighbors = if layer < node.layers.len() {
                    &node.layers[layer]
                } else {
                    &node.layers[node.layers.len().saturating_sub(1)]
                };
                for &nb in neighbors {
                    if visited.contains_key(&nb) {
                        continue;
                    }
                    let nd = l2_sq(query, &self.nodes[nb].vector);
                    visited.insert(nb, nd);
                    let worst = results
                        .peek()
                        .map(|(wd, _)| *wd as f32 / 1e9)
                        .unwrap_or(f32::MAX);
                    if nd < worst || results.len() < ef {
                        candidates.push(((-nd * 1e9) as i64, nb));
                        results.push(((nd * 1e9) as i64, nb));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(usize, f32)> = results
            .into_iter()
            .map(|(wd, id)| (id, wd as f32 / 1e9))
            .collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        out
    }

    pub fn insert(&mut self, vector: Vec<f32>) -> usize {
        let id = self.nodes.len();
        let level = self.random_level();
        let mut layers: Vec<Vec<usize>> = vec![Vec::new(); level + 1];

        if self.entry_point.is_none() {
            self.nodes.push(Node { vector, layers });
            self.entry_point = Some(0);
            self.max_layer = level;
            return id;
        }

        let entry = self.entry_point.unwrap();
        let mut curr_entry = entry;

        // Greedy descent from top to level+1
        for lc in (level + 1..=self.max_layer).rev() {
            let results = self.search_layer(&vector, curr_entry, 1, lc);
            if let Some((nearest, _)) = results.first() {
                curr_entry = *nearest;
            }
        }

        // Build connections from max(level, max_layer) down to 0
        for lc in (0..=level.min(self.max_layer)).rev() {
            let ef = self.config.ef_construction;
            let candidates = self.search_layer(&vector, curr_entry, ef, lc);
            let m = self.config.m;
            let selected: Vec<usize> = candidates.iter().take(m).map(|(id, _)| *id).collect();

            layers[lc] = selected.clone();

            // Add back-edges
            for &nb in &selected {
                if lc < self.nodes[nb].layers.len() {
                    self.nodes[nb].layers[lc].push(id);
                    // Prune if oversized. Skip `id` itself: it hasn't been pushed
                    // to self.nodes yet, so accessing self.nodes[id] would panic.
                    if self.nodes[nb].layers[lc].len() > m * 2 {
                        let nb_vec = self.nodes[nb].vector.clone();
                        let n_inserted = self.nodes.len();
                        let mut nb_cands: Vec<(usize, f32)> = self.nodes[nb].layers[lc]
                            .iter()
                            .filter(|&&c| c < n_inserted) // exclude not-yet-pushed id
                            .map(|&c| (c, l2_sq(&nb_vec, &self.nodes[c].vector)))
                            .collect();
                        nb_cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        self.nodes[nb].layers[lc] =
                            nb_cands.iter().take(m).map(|(c, _)| *c).collect();
                    }
                } else {
                    self.nodes[nb].layers.push(vec![id]);
                }
                curr_entry = candidates.first().map(|(i, _)| *i).unwrap_or(curr_entry);
            }
        }

        self.nodes.push(Node { vector, layers });

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(id);
        }

        id
    }

    /// Return top-k most similar cached query ids and their L2-squared distances.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let entry = self.entry_point.unwrap();
        let mut curr = entry;

        for lc in (1..=self.max_layer).rev() {
            let r = self.search_layer(query, curr, 1, lc);
            if let Some((best, _)) = r.first() {
                curr = *best;
            }
        }
        let mut results = self.search_layer(query, curr, ef.max(k), 0);
        results.truncate(k);
        results
    }

    /// Cosine similarity wrapper (convert from L2-squared on normalised vectors).
    /// For normalised vectors: cos_sim = 1 - l2_sq/2.
    pub fn search_by_cosine(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        self.search(query, k, ef)
            .into_iter()
            .map(|(id, d)| (id, 1.0 - d / 2.0))
            .collect()
    }
}
