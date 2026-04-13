//! HNSW-backed [`SemanticIndex`] used under `--features hnsw`.
//!
//! This is a minimal, self-contained Hierarchical Navigable Small World
//! implementation — deliberately compact (~300 lines, no external deps).
//! It follows the Malkov & Yashunin design:
//!
//! * Multi-level proximity graph with exponential level decay.
//! * Greedy descent from the top entry point down to level 0.
//! * Beam search at each level with `ef` candidate pool.
//!
//! Shell segmentation is inherited for free: one [`HnswLayer`] per shell,
//! keyed inside [`HnswIndex`]. Retrieval walks the union of layers matching
//! the caller's shell filter.
//!
//! Reference only: we do not claim parity with a production HNSW such as
//! `hnswlib` or `ruvector-hyperbolic-hnsw`. It is fast enough for the
//! acceptance gate at 10× corpus scale while staying zero-dep.

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::model::{Embedding, EmbeddingId, EmbeddingStore, NodeId, Shell};
use crate::storage::SemanticIndex;

/// HNSW tuning knobs.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Max out-degree per node on layer 0.
    pub m: usize,
    /// Max out-degree per node on layers > 0.
    pub m_max: usize,
    /// Beam width for construction.
    pub ef_construction: usize,
    /// Beam width for search (queried per call through `search_ef`).
    pub ef_search: usize,
    /// Level multiplier for the geometric level distribution.
    pub level_mult: f32,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 12,
            m_max: 12,
            ef_construction: 32,
            ef_search: 48,
            level_mult: 1.0 / (2.0_f32).ln(),
        }
    }
}

#[derive(Debug, Clone)]
struct HnswNode {
    id: NodeId,
    embedding: EmbeddingId,
    /// `neighbors[layer]` is the adjacency list for `layer`.
    neighbors: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Default)]
struct HnswLayer {
    nodes: Vec<HnswNode>,
    /// `by_id[node_id] -> index into nodes`.
    by_id: HashMap<NodeId, usize>,
    entry_point: Option<usize>,
    max_level: usize,
    /// Deterministic LCG state for level assignment.
    rng_state: u64,
}

impl HnswLayer {
    fn new(seed: u64) -> Self {
        Self {
            nodes: Vec::new(),
            by_id: HashMap::new(),
            entry_point: None,
            max_level: 0,
            rng_state: seed.max(1),
        }
    }

    fn next_u32(&mut self) -> u32 {
        // Numerical Recipes LCG — deterministic, good enough for level dice.
        self.rng_state = self
            .rng_state
            .wrapping_mul(1664525)
            .wrapping_add(1013904223);
        (self.rng_state >> 16) as u32
    }

    fn assign_level(&mut self, mult: f32) -> usize {
        let u = (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0);
        (-u.ln() * mult).floor() as usize
    }

    fn distance(
        &self,
        store: &EmbeddingStore,
        a: &Embedding,
        b: EmbeddingId,
    ) -> f32 {
        let Some(vb) = store.get(b) else { return f32::MAX };
        // Cosine distance = 1 - cosine similarity
        1.0 - a.cosine(vb)
    }

    fn search_layer(
        &self,
        store: &EmbeddingStore,
        query: &Embedding,
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        // Greedy beam search. `candidates` is a min-heap over distance;
        // `results` is a max-heap so we can prune the worst easily.
        let mut visited: HashSet<usize> = HashSet::new();
        visited.insert(entry);
        let d0 = self.distance(store, query, self.nodes[entry].embedding);
        let mut candidates = BinaryHeap::new();
        let mut results: BinaryHeap<MaxItem> = BinaryHeap::new();
        candidates.push(MinItem(d0, entry));
        results.push(MaxItem(d0, entry));
        while let Some(MinItem(d, idx)) = candidates.pop() {
            let worst = results.peek().map(|r| r.0).unwrap_or(f32::MAX);
            if d > worst {
                break;
            }
            let neighbors = &self.nodes[idx].neighbors.get(layer).cloned().unwrap_or_default();
            for &nb in neighbors {
                if !visited.insert(nb) {
                    continue;
                }
                let dn = self.distance(store, query, self.nodes[nb].embedding);
                let worst = results.peek().map(|r| r.0).unwrap_or(f32::MAX);
                if results.len() < ef || dn < worst {
                    candidates.push(MinItem(dn, nb));
                    results.push(MaxItem(dn, nb));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut out: Vec<(usize, f32)> = results.into_iter().map(|m| (m.1, m.0)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    fn select_neighbors(&self, candidates: Vec<(usize, f32)>, m: usize) -> Vec<usize> {
        let mut c = candidates;
        c.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        c.truncate(m);
        c.into_iter().map(|(i, _)| i).collect()
    }

    fn insert(
        &mut self,
        store: &EmbeddingStore,
        node_id: NodeId,
        embedding: EmbeddingId,
        cfg: &HnswConfig,
    ) {
        // Replace if already present.
        if let Some(&idx) = self.by_id.get(&node_id) {
            self.nodes[idx].embedding = embedding;
            return;
        }
        let Some(query) = store.get(embedding).cloned() else { return };
        let level = self.assign_level(cfg.level_mult);
        let new_idx = self.nodes.len();
        self.nodes.push(HnswNode {
            id: node_id,
            embedding,
            neighbors: vec![Vec::new(); level + 1],
        });
        self.by_id.insert(node_id, new_idx);

        let Some(mut entry) = self.entry_point else {
            self.entry_point = Some(new_idx);
            self.max_level = level;
            return;
        };

        // Descend from max_level to level+1 (greedy).
        let mut curr_level = self.max_level;
        while curr_level > level {
            let hits = self.search_layer(store, &query, entry, 1, curr_level);
            if let Some((best, _)) = hits.into_iter().next() {
                entry = best;
            }
            if curr_level == 0 {
                break;
            }
            curr_level -= 1;
        }

        // Connect on layers 0..=min(level, max_level).
        let mut layer = level.min(self.max_level);
        loop {
            let hits = self.search_layer(store, &query, entry, cfg.ef_construction, layer);
            let m = if layer == 0 { cfg.m } else { cfg.m_max };
            let chosen = self.select_neighbors(hits.clone(), m);
            // Connect bidirectionally.
            for &nb in &chosen {
                self.nodes[new_idx].neighbors[layer].push(nb);
                while self.nodes[nb].neighbors.len() <= layer {
                    self.nodes[nb].neighbors.push(Vec::new());
                }
                self.nodes[nb].neighbors[layer].push(new_idx);
                // Prune back neighbors if over capacity.
                if self.nodes[nb].neighbors[layer].len() > m {
                    let mut rescored: Vec<(usize, f32)> = self.nodes[nb]
                        .neighbors[layer]
                        .iter()
                        .map(|&i| {
                            (
                                i,
                                self.distance(store, &query, self.nodes[i].embedding),
                            )
                        })
                        .collect();
                    rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    rescored.truncate(m);
                    self.nodes[nb].neighbors[layer] =
                        rescored.into_iter().map(|(i, _)| i).collect();
                }
            }
            entry = chosen.first().copied().unwrap_or(entry);
            if layer == 0 {
                break;
            }
            layer -= 1;
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(new_idx);
        }
    }

    fn remove(&mut self, node_id: NodeId) {
        if let Some(idx) = self.by_id.remove(&node_id) {
            // Soft-remove: mark by emptying adjacency and clearing embedding
            // reference so queries skip it. Full compaction is out of scope.
            self.nodes[idx].neighbors.iter_mut().for_each(|v| v.clear());
            if self.entry_point == Some(idx) {
                self.entry_point = self.nodes.iter().enumerate().find_map(|(i, _)| {
                    if self.by_id.values().any(|v| *v == i) {
                        Some(i)
                    } else {
                        None
                    }
                });
            }
        }
    }

    fn search(
        &self,
        store: &EmbeddingStore,
        query: &Embedding,
        k: usize,
        ef: usize,
    ) -> Vec<(NodeId, f32)> {
        let Some(mut entry) = self.entry_point else {
            return Vec::new();
        };
        // Greedy descent to layer 0.
        for layer in (1..=self.max_level).rev() {
            let hits = self.search_layer(store, query, entry, 1, layer);
            if let Some((best, _)) = hits.into_iter().next() {
                entry = best;
            }
        }
        let final_hits = self.search_layer(store, query, entry, ef.max(k), 0);
        final_hits
            .into_iter()
            .filter(|(idx, _)| self.by_id.values().any(|v| v == idx))
            .take(k)
            .map(|(idx, d)| (self.nodes[idx].id, 1.0 - d))
            .collect()
    }
}

/// HNSW wrappers keyed by [`Shell`] so shell filters are a free operation.
#[derive(Debug, Clone)]
pub struct HnswIndex {
    cfg: HnswConfig,
    shells: HashMap<Shell, HnswLayer>,
    /// `which_shell[node] -> shell` to support fast upsert across shells.
    which_shell: HashMap<NodeId, Shell>,
}

impl HnswIndex {
    /// Create an empty HNSW index with the default config.
    pub fn new() -> Self {
        Self::with_config(HnswConfig::default())
    }

    /// Create an empty HNSW index with a custom config.
    pub fn with_config(cfg: HnswConfig) -> Self {
        Self {
            cfg,
            shells: HashMap::new(),
            which_shell: HashMap::new(),
        }
    }

    /// Upsert a node into the index, reshelling if necessary.
    pub fn upsert(
        &mut self,
        store: &EmbeddingStore,
        node: NodeId,
        embedding: EmbeddingId,
        shell: Shell,
    ) {
        if let Some(prev) = self.which_shell.insert(node, shell) {
            if prev != shell {
                if let Some(layer) = self.shells.get_mut(&prev) {
                    layer.remove(node);
                }
            }
        }
        let seed = node.0.wrapping_add(0xdead_beef);
        let layer = self
            .shells
            .entry(shell)
            .or_insert_with(|| HnswLayer::new(seed));
        layer.insert(store, node, embedding, &self.cfg);
    }

    /// Remove a node from the index.
    pub fn remove(&mut self, node: NodeId) {
        if let Some(shell) = self.which_shell.remove(&node) {
            if let Some(layer) = self.shells.get_mut(&shell) {
                layer.remove(node);
            }
        }
    }

    /// Number of indexed nodes.
    pub fn len(&self) -> usize {
        self.which_shell.len()
    }

    /// `true` if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.which_shell.is_empty()
    }
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticIndex for HnswIndex {
    fn search(
        &self,
        store: &EmbeddingStore,
        query: &Embedding,
        shells: &[Shell],
        k: usize,
    ) -> Vec<(NodeId, f32)> {
        let ef = self.cfg.ef_search.max(k);
        let mut merged: Vec<(NodeId, f32)> = Vec::new();
        let iter: Box<dyn Iterator<Item = (&Shell, &HnswLayer)>> = if shells.is_empty() {
            Box::new(self.shells.iter())
        } else {
            Box::new(self.shells.iter().filter(|(s, _)| shells.contains(s)))
        };
        for (_, layer) in iter {
            merged.extend(layer.search(store, query, k, ef));
        }
        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(k);
        merged
    }
}

// --- heap item wrappers -----------------------------------------------

#[derive(Debug, Clone, Copy)]
struct MinItem(f32, usize);
impl PartialEq for MinItem {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MinItem {}
impl PartialOrd for MinItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.0.partial_cmp(&self.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[derive(Debug, Clone, Copy)]
struct MaxItem(f32, usize);
impl PartialEq for MaxItem {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MaxItem {}
impl PartialOrd for MaxItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_small_corpus() {
        let mut store = EmbeddingStore::new();
        let mut idx = HnswIndex::new();
        for i in 0..20 {
            let e = Embedding::new(vec![
                ((i % 5) as f32) * 0.1 + 0.1,
                (i as f32) * 0.05,
                0.3,
            ]);
            let eid = store.intern(e);
            idx.upsert(&store, NodeId(i + 1), eid, Shell::Event);
        }
        let q = Embedding::new(vec![0.3, 0.2, 0.3]);
        let hits = idx.search(&store, &q, &[Shell::Event], 5);
        assert_eq!(hits.len(), 5);
        // All scores should be cosine similarities in [-1, 1].
        for (_, s) in hits {
            assert!(s >= -1.0 && s <= 1.0001, "bad sim {}", s);
        }
    }
}
