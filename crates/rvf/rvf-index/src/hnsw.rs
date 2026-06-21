//! Core HNSW (Hierarchical Navigable Small World) graph implementation.
//!
//! Implements the algorithm from Malkov & Yashunin (2018) with:
//! - Configurable M (max neighbors per layer) and ef_construction
//! - Layer selection via P = 1/ln(M), level = floor(-ln(random) * P)
//! - Greedy search at upper layers, beam search at layer 0
//! - Vamana/DiskANN-style alpha-RNG neighbor pruning (robust prune,
//!   alpha = [`DEFAULT_ALPHA`]) for diverse, navigable neighbor lists

extern crate alloc;

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::traits::VectorStore;

/// `f32` wrapper with a total ordering (via `f32::total_cmp`) so that
/// distances can be stored in heaps with deterministic ordering.
#[derive(Clone, Copy, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(not(feature = "std"))]
type SparseVisited = alloc::collections::BTreeSet<u64>;
#[cfg(feature = "std")]
type SparseVisited = std::collections::HashSet<u64>;

/// Visited-node tracker for graph traversal.
///
/// Builder-produced graphs use dense IDs (`0..n`), where a bitmap is much
/// faster than a hashed set; a set is used as fallback for sparse ID spaces.
enum VisitedSet {
    /// Bitmap indexed by node ID.
    Dense(Vec<bool>),
    /// Fallback for sparse ID spaces.
    Sparse(SparseVisited),
}

impl VisitedSet {
    /// Mark `id` as visited. Returns `true` if it was not visited before.
    #[inline]
    fn insert(&mut self, id: u64) -> bool {
        match self {
            Self::Dense(bits) => {
                let idx = id as usize;
                if idx >= bits.len() {
                    bits.resize(idx + 1, false);
                }
                !core::mem::replace(&mut bits[idx], true)
            }
            Self::Sparse(set) => set.insert(id),
        }
    }
}

/// Configuration for HNSW graph construction.
#[derive(Clone, Debug)]
pub struct HnswConfig {
    /// Maximum number of neighbors per node per layer (layer > 0).
    pub m: usize,
    /// Maximum number of neighbors at layer 0 (typically 2*M).
    pub m0: usize,
    /// Size of the dynamic candidate list during construction.
    pub ef_construction: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
        }
    }
}

/// A single layer of the HNSW graph, mapping node IDs to their neighbor lists.
#[derive(Clone, Debug, Default)]
pub struct HnswLayer {
    /// Node ID -> sorted list of neighbor IDs.
    pub adjacency: BTreeMap<u64, Vec<u64>>,
}

impl HnswLayer {
    /// Returns true if this layer contains the given node.
    #[inline]
    pub fn contains(&self, id: u64) -> bool {
        self.adjacency.contains_key(&id)
    }

    /// Returns the neighbors of a node, or an empty slice if not present.
    #[inline]
    pub fn neighbors(&self, id: u64) -> &[u64] {
        self.adjacency.get(&id).map_or(&[], |v| v.as_slice())
    }

    /// Number of nodes in this layer.
    #[inline]
    pub fn len(&self) -> usize {
        self.adjacency.len()
    }

    /// Returns true if the layer has no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }
}

/// Default alpha for Vamana-style robust pruning. Values > 1 relax the
/// relative-neighborhood-graph rule, keeping longer-range "highway" edges
/// that improve recall (DiskANN uses 1.2 in production).
pub const DEFAULT_ALPHA: f32 = 1.2;

/// The full HNSW graph structure.
#[derive(Clone, Debug)]
pub struct HnswGraph {
    /// Layers from bottom (0) to top.
    pub layers: Vec<HnswLayer>,
    /// The entry point node ID (node at the highest layer).
    pub entry_point: Option<u64>,
    /// The highest occupied layer index.
    pub max_layer: usize,
    /// Max neighbors per layer (> 0).
    pub m: usize,
    /// Max neighbors at layer 0.
    pub m0: usize,
    /// ef_construction parameter.
    pub ef_construction: usize,
    /// Alpha for Vamana-style neighbor pruning (construction only; does
    /// not affect search over an existing graph). `f32::INFINITY`
    /// disables pruning, reproducing plain closest-first selection.
    pub alpha: f32,
    /// Level normalization factor: 1 / ln(M).
    ml: f64,
}

impl HnswGraph {
    /// Create a new empty HNSW graph with the given configuration.
    pub fn new(config: &HnswConfig) -> Self {
        Self {
            layers: vec![HnswLayer::default()],
            entry_point: None,
            max_layer: 0,
            m: config.m,
            m0: config.m0,
            ef_construction: config.ef_construction,
            alpha: DEFAULT_ALPHA,
            ml: 1.0 / (config.m as f64).ln(),
        }
    }

    /// Select a random level for a new node.
    /// Level = floor(-ln(uniform(0,1)) * ml).
    fn random_level(&self, rng_val: f64) -> usize {
        let r = if rng_val <= 0.0 { 1e-10 } else { rng_val };
        (-r.ln() * self.ml).floor() as usize
    }

    /// Insert a new node into the HNSW graph.
    ///
    /// `id`: the node ID to insert.
    /// `rng_val`: a uniform random value in (0, 1) for level selection.
    /// `vectors`: provides access to all vectors by ID.
    /// `distance_fn`: distance function between two vectors.
    pub fn insert(
        &mut self,
        id: u64,
        rng_val: f64,
        vectors: &dyn VectorStore,
        distance_fn: &dyn Fn(&[f32], &[f32]) -> f32,
    ) {
        let level = self.random_level(rng_val);

        // Ensure we have enough layers.
        while self.layers.len() <= level {
            self.layers.push(HnswLayer::default());
        }

        // Add the node to each layer from 0 to `level`.
        for l in 0..=level {
            self.layers[l].adjacency.entry(id).or_default();
        }

        let query_vec = match vectors.get_vector(id) {
            Some(v) => v,
            None => return,
        };

        if self.entry_point.is_none() {
            // First node.
            self.entry_point = Some(id);
            self.max_layer = level;
            return;
        }

        let ep = self.entry_point.unwrap();

        // Phase 1: greedy search from top layer down to level+1.
        let mut current_ep = ep;
        let top = self.max_layer;
        if top > level {
            for l in (level + 1..=top).rev() {
                current_ep = self.greedy_closest(query_vec, current_ep, l, vectors, distance_fn);
            }
        }

        // Phase 2: at each layer from min(level, max_layer) down to 0,
        // do a beam search and connect neighbors.
        let start_layer = level.min(top);
        let mut entry_points = vec![current_ep];

        for l in (0..=start_layer).rev() {
            let max_neighbors = if l == 0 { self.m0 } else { self.m };

            let candidates = self.search_layer(
                query_vec,
                &entry_points,
                self.ef_construction,
                l,
                vectors,
                distance_fn,
            );

            // Vamana-style alpha-RNG selection over the beam candidates
            // (diverse neighbors instead of the plain closest set).
            let selected =
                self.select_neighbors_alpha(&candidates, max_neighbors, vectors, distance_fn);

            // Connect the new node to selected neighbors.
            let neighbor_ids: Vec<u64> = selected.iter().map(|&(nid, _)| nid).collect();
            self.layers[l].adjacency.insert(id, neighbor_ids.clone());

            // Bidirectional: add the new node as a neighbor of each selected node,
            // then prune if over the limit.
            for &nid in &neighbor_ids {
                let nlist = self.layers[l].adjacency.entry(nid).or_default();
                if !nlist.contains(&id) {
                    nlist.push(id);
                }
                if nlist.len() > max_neighbors {
                    // Prune: keep only the closest max_neighbors.
                    self.prune_neighbors(nid, l, max_neighbors, vectors, distance_fn);
                }
            }

            // Use the selected candidates as entry points for the next layer down.
            entry_points = selected.iter().map(|&(nid, _)| nid).collect();
        }

        // Update entry point if the new node is at a higher layer.
        if level > self.max_layer {
            self.entry_point = Some(id);
            self.max_layer = level;
        }
    }

    /// Greedy search: starting from `ep`, walk to the closest node at `layer`.
    fn greedy_closest(
        &self,
        query: &[f32],
        ep: u64,
        layer: usize,
        vectors: &dyn VectorStore,
        distance_fn: &dyn Fn(&[f32], &[f32]) -> f32,
    ) -> u64 {
        let mut current = ep;
        let mut current_dist = match vectors.get_vector(ep) {
            Some(v) => distance_fn(query, v),
            None => return ep,
        };

        loop {
            let mut changed = false;
            let neighbors = self.layers[layer].neighbors(current);
            for &nid in neighbors {
                if let Some(nv) = vectors.get_vector(nid) {
                    let d = distance_fn(query, nv);
                    if d < current_dist {
                        current = nid;
                        current_dist = d;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    /// Beam search at a given layer. Returns candidates sorted by distance (ascending).
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u64],
        ef: usize,
        layer: usize,
        vectors: &dyn VectorStore,
        distance_fn: &dyn Fn(&[f32], &[f32]) -> f32,
    ) -> Vec<(u64, f32)> {
        // Builder-produced graphs have dense node IDs (0..n), so prefer a
        // bitmap sized by the max ID; fall back to a set when IDs are sparse.
        let max_id = self
            .layers
            .first()
            .and_then(|l| l.adjacency.keys().next_back())
            .copied();
        let node_count = self.node_count() as u64;
        let mut visited = match max_id {
            Some(max) if max < node_count.saturating_mul(4).max(1024) => {
                VisitedSet::Dense(vec![false; (max + 1) as usize])
            }
            _ => VisitedSet::Sparse(SparseVisited::new()),
        };

        // Min-heap of candidates: closest first.
        let mut candidates: BinaryHeap<Reverse<(OrderedF32, u64)>> = BinaryHeap::new();
        // Bounded max-heap of results: the worst (farthest) entry on top.
        let mut results: BinaryHeap<(OrderedF32, u64)> = BinaryHeap::new();

        for &ep in entry_points {
            if visited.insert(ep) {
                if let Some(v) = vectors.get_vector(ep) {
                    let d = distance_fn(query, v);
                    candidates.push(Reverse((OrderedF32(d), ep)));
                    results.push((OrderedF32(d), ep));
                }
            }
        }
        while results.len() > ef {
            results.pop();
        }

        while let Some(Reverse((OrderedF32(cdist), cid))) = candidates.pop() {
            // If the closest candidate is farther than the worst result and
            // we already have `ef` results, stop.
            if results.len() >= ef {
                let worst_dist = results.peek().map_or(f32::MAX, |&(OrderedF32(d), _)| d);
                if cdist > worst_dist {
                    break;
                }
            }

            let neighbors = self.layers[layer].neighbors(cid);
            for &nid in neighbors {
                if !visited.insert(nid) {
                    continue;
                }
                if let Some(nv) = vectors.get_vector(nid) {
                    let d = distance_fn(query, nv);
                    let worst_dist = if results.len() >= ef {
                        results.peek().map_or(f32::MAX, |&(OrderedF32(w), _)| w)
                    } else {
                        f32::MAX
                    };

                    if d < worst_dist || results.len() < ef {
                        candidates.push(Reverse((OrderedF32(d), nid)));
                        results.push((OrderedF32(d), nid));

                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Drain results sorted by (distance, id) ascending.
        let mut out: Vec<(u64, f32)> = results
            .into_iter()
            .map(|(OrderedF32(d), id)| (id, d))
            .collect();
        out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// Vamana/DiskANN-style alpha-RNG neighbor selection (robust prune).
    ///
    /// Scans `candidates` (sorted ascending by `(distance, id)` relative
    /// to the base node) and keeps a candidate `c` only if no
    /// already-selected neighbor `s` occludes it, i.e. only if
    /// `alpha * d(s, c) > d(base, c)` for every selected `s`. With
    /// alpha > 1 this preserves a few longer "highway" edges instead of
    /// `max_neighbors` mutually-redundant closest nodes, which improves
    /// graph navigability and recall.
    ///
    /// Occluded candidates are kept in order and used to backfill any
    /// remaining slots (the `keepPrunedConnections` heuristic), so node
    /// degree — and therefore connectivity — is never reduced versus
    /// plain closest-first selection. Fully deterministic for a given
    /// candidate order.
    fn select_neighbors_alpha(
        &self,
        candidates: &[(u64, f32)],
        max_neighbors: usize,
        vectors: &dyn VectorStore,
        distance_fn: &dyn Fn(&[f32], &[f32]) -> f32,
    ) -> Vec<(u64, f32)> {
        if candidates.len() <= max_neighbors {
            return candidates.to_vec();
        }
        let mut selected: Vec<(u64, f32)> = Vec::with_capacity(max_neighbors);
        let mut occluded: Vec<(u64, f32)> = Vec::new();
        for &(cid, cdist) in candidates {
            if selected.len() >= max_neighbors {
                break;
            }
            let cvec = match vectors.get_vector(cid) {
                Some(v) => v,
                None => continue,
            };
            let is_occluded = selected.iter().any(|&(sid, _)| {
                vectors
                    .get_vector(sid)
                    .is_some_and(|sv| self.alpha * distance_fn(sv, cvec) <= cdist)
            });
            if is_occluded {
                occluded.push((cid, cdist));
            } else {
                selected.push((cid, cdist));
            }
        }
        // Backfill remaining slots with the closest occluded candidates.
        let mut spill = occluded.into_iter();
        while selected.len() < max_neighbors {
            match spill.next() {
                Some(p) => selected.push(p),
                None => break,
            }
        }
        selected
    }

    /// Prune neighbors of a node to at most `max_neighbors`, using
    /// alpha-RNG selection relative to the node.
    fn prune_neighbors(
        &mut self,
        node: u64,
        layer: usize,
        max_neighbors: usize,
        vectors: &dyn VectorStore,
        distance_fn: &dyn Fn(&[f32], &[f32]) -> f32,
    ) {
        let node_vec = match vectors.get_vector(node) {
            Some(v) => v,
            None => return,
        };
        let neighbors = match self.layers[layer].adjacency.get(&node) {
            Some(n) => n.clone(),
            None => return,
        };

        let mut scored: Vec<(u64, f32)> = neighbors
            .iter()
            .filter_map(|&nid| {
                vectors
                    .get_vector(nid)
                    .map(|nv| (nid, distance_fn(node_vec, nv)))
            })
            .collect();
        // Deterministic order: (distance, id), independent of input order.
        scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let kept = self.select_neighbors_alpha(&scored, max_neighbors, vectors, distance_fn);
        let pruned: Vec<u64> = kept.into_iter().map(|(nid, _)| nid).collect();
        self.layers[layer].adjacency.insert(node, pruned);
    }

    /// Search the HNSW graph for the `k` nearest neighbors of `query`.
    ///
    /// `ef_search`: size of the dynamic candidate list during search.
    /// Returns a list of `(node_id, distance)` sorted by distance (ascending).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        vectors: &dyn VectorStore,
        distance_fn: &dyn Fn(&[f32], &[f32]) -> f32,
    ) -> Vec<(u64, f32)> {
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let ef = ef_search.max(k);

        // Phase 1: greedy search from top layer down to layer 1.
        let mut current_ep = ep;
        for l in (1..=self.max_layer).rev() {
            current_ep = self.greedy_closest(query, current_ep, l, vectors, distance_fn);
        }

        // Phase 2: beam search at layer 0.
        let mut results = self.search_layer(query, &[current_ep], ef, 0, vectors, distance_fn);
        results.truncate(k);
        results
    }

    /// Returns the total number of nodes across all layers.
    pub fn node_count(&self) -> usize {
        self.layers.first().map_or(0, |l| l.adjacency.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::l2_distance;
    use crate::traits::InMemoryVectorStore;

    fn make_config() -> HnswConfig {
        HnswConfig {
            m: 8,
            m0: 16,
            ef_construction: 100,
        }
    }

    #[test]
    fn empty_graph_search_returns_empty() {
        let config = make_config();
        let graph = HnswGraph::new(&config);
        let store = InMemoryVectorStore::new(vec![vec![0.0; 4]]);
        let results = graph.search(&[0.0; 4], 5, 50, &store, &l2_distance);
        assert!(results.is_empty());
    }

    #[test]
    fn insert_single_node() {
        let config = make_config();
        let mut graph = HnswGraph::new(&config);
        let store = InMemoryVectorStore::new(vec![vec![1.0, 2.0, 3.0]]);
        graph.insert(0, 0.5, &store, &l2_distance);

        assert_eq!(graph.entry_point, Some(0));
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn insert_and_search_small() {
        let config = make_config();
        let mut graph = HnswGraph::new(&config);

        let vectors: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32])
            .collect();
        let store = InMemoryVectorStore::new(vectors);

        // Insert all with deterministic pseudo-random values.
        for i in 0..20u64 {
            let rng = ((i * 7 + 3) % 100) as f64 / 100.0;
            graph.insert(i, rng, &store, &l2_distance);
        }

        // Search for a query near node 10.
        let query = [10.0, 20.0, 30.0];
        let results = graph.search(&query, 3, 50, &store, &l2_distance);
        assert!(!results.is_empty());
        // Node 10 should be the closest (exact match).
        assert_eq!(results[0].0, 10);
    }

    #[test]
    fn alpha_selection_keeps_diverse_neighbors() {
        // Base node at origin; two near-duplicates and one distant point.
        // Plain greedy keeps the two duplicates; alpha-pruning must keep
        // one duplicate plus the distant (diverse) point first, then
        // backfill the occluded duplicate only if a slot remains.
        let vectors = vec![
            vec![0.0, 0.0], // 0: base
            vec![1.0, 0.0], // 1: near
            vec![1.1, 0.0], // 2: near-duplicate of 1
            vec![0.0, 3.0], // 3: distant, diverse
        ];
        let store = InMemoryVectorStore::new(vectors);
        let graph = HnswGraph::new(&make_config());

        // Candidates sorted by (squared distance to node 0, id).
        let candidates = vec![(1u64, 1.0f32), (2, 1.21), (3, 9.0)];

        let selected = graph.select_neighbors_alpha(&candidates, 2, &store, &l2_distance);
        let ids: Vec<u64> = selected.iter().map(|&(id, _)| id).collect();
        // Node 2 is occluded by node 1 (alpha * d(1,2) = 1.2*0.01 <= 1.21),
        // so the diverse node 3 is selected ahead of it.
        assert_eq!(ids, vec![1, 3]);

        // With 3 slots, the occluded candidate is backfilled: full degree.
        let selected = graph.select_neighbors_alpha(&candidates, 3, &store, &l2_distance);
        assert_eq!(selected.len(), 3);

        // Disabling pruning reproduces plain closest-first selection.
        let mut greedy = HnswGraph::new(&make_config());
        greedy.alpha = f32::INFINITY;
        let selected = greedy.select_neighbors_alpha(&candidates, 2, &store, &l2_distance);
        let ids: Vec<u64> = selected.iter().map(|&(id, _)| id).collect();
        assert_eq!(ids, vec![1, 2]);
    }

    /// Shared harness: build a graph over LCG vectors with the given alpha
    /// and measure recall@10 against brute force.
    fn measure_recall(n: usize, dim: usize, ef_search: usize, alpha: f32) -> f64 {
        use alloc::collections::BTreeSet;

        let mut seed: u64 = 42;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (seed >> 33) as f32 / (1u64 << 31) as f32
                    })
                    .collect()
            })
            .collect();
        let store = InMemoryVectorStore::new(vectors.clone());

        let config = HnswConfig {
            m: 16,
            m0: 32,
            ef_construction: 200,
        };
        let mut graph = HnswGraph::new(&config);
        graph.alpha = alpha;
        let mut rng_seed: u64 = 123;
        for i in 0..n as u64 {
            rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rng_val = ((rng_seed >> 33) as f64 / (1u64 << 31) as f64).clamp(0.001, 0.999);
            graph.insert(i, rng_val, &store, &l2_distance);
        }

        let num_queries = 50;
        let k = 10;
        let mut total_recall = 0.0;
        let mut query_seed: u64 = 999;
        for _ in 0..num_queries {
            let query: Vec<f32> = (0..dim)
                .map(|_| {
                    query_seed = query_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (query_seed >> 33) as f32 / (1u64 << 31) as f32
                })
                .collect();

            let mut all_dists: Vec<(u64, f32)> = (0..n as u64)
                .map(|i| (i, l2_distance(&query, &vectors[i as usize])))
                .collect();
            all_dists.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let gt_set: BTreeSet<u64> = all_dists.iter().take(k).map(|&(id, _)| id).collect();

            let results = graph.search(&query, k, ef_search, &store, &l2_distance);
            let result_set: BTreeSet<u64> = results.iter().map(|&(id, _)| id).collect();
            total_recall += gt_set.intersection(&result_set).count() as f64 / k as f64;
        }
        total_recall / num_queries as f64
    }

    /// Alpha-pruned construction must not regress recall versus plain
    /// greedy selection (alpha = inf), measured at a deliberately low
    /// ef_search where graph quality dominates.
    #[test]
    fn alpha_pruning_does_not_regress_recall() {
        let recall_alpha = measure_recall(1000, 32, 30, DEFAULT_ALPHA);
        let recall_greedy = measure_recall(1000, 32, 30, f32::INFINITY);
        #[cfg(feature = "std")]
        std::eprintln!(
            "recall@10 ef=30: alpha=1.2 -> {recall_alpha:.3}, greedy -> {recall_greedy:.3}"
        );
        assert!(
            recall_alpha + 1e-9 >= recall_greedy,
            "alpha-pruned recall {recall_alpha:.3} regressed vs greedy {recall_greedy:.3}"
        );
        assert!(
            recall_alpha >= 0.80,
            "alpha-pruned recall@10 {recall_alpha:.3} below sanity floor at ef=30"
        );
    }

    /// Build HNSW with 1000 random vectors, verify recall@10 >= 0.95.
    #[test]
    fn recall_at_10_1000_vectors() {
        use alloc::collections::BTreeSet;

        let n = 1000;
        let dim = 32;

        // Generate deterministic pseudo-random vectors using a simple LCG.
        let mut seed: u64 = 42;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (seed >> 33) as f32 / (1u64 << 31) as f32
                    })
                    .collect()
            })
            .collect();
        let store = InMemoryVectorStore::new(vectors.clone());

        // Build the graph.
        let config = HnswConfig {
            m: 16,
            m0: 32,
            ef_construction: 200,
        };
        let mut graph = HnswGraph::new(&config);
        let mut rng_seed: u64 = 123;
        for i in 0..n as u64 {
            rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rng_val = (rng_seed >> 33) as f64 / (1u64 << 31) as f64;
            let rng_val = rng_val.clamp(0.001, 0.999);
            graph.insert(i, rng_val, &store, &l2_distance);
        }

        // Compute brute-force ground truth and measure recall.
        let num_queries = 50;
        let k = 10;
        let ef_search = 200;
        let mut total_recall = 0.0;

        let mut query_seed: u64 = 999;
        for _ in 0..num_queries {
            // Generate a random query.
            let query: Vec<f32> = (0..dim)
                .map(|_| {
                    query_seed = query_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (query_seed >> 33) as f32 / (1u64 << 31) as f32
                })
                .collect();

            // Brute-force top-k.
            let mut all_dists: Vec<(u64, f32)> = (0..n as u64)
                .map(|i| (i, l2_distance(&query, &vectors[i as usize])))
                .collect();
            all_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt_set: BTreeSet<u64> = all_dists.iter().take(k).map(|&(id, _)| id).collect();

            // HNSW search.
            let results = graph.search(&query, k, ef_search, &store, &l2_distance);
            let result_set: BTreeSet<u64> = results.iter().map(|&(id, _)| id).collect();

            let overlap = gt_set.intersection(&result_set).count();
            total_recall += overlap as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall >= 0.95,
            "Recall@10 = {:.3}, expected >= 0.95",
            avg_recall
        );
    }
}
