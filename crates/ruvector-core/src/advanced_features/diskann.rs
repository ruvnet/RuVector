//! DiskANN / Vamana SSD-Backed Approximate Nearest Neighbor Index
//!
//! Implements the Vamana graph index from the DiskANN paper (Subramanya et al., 2019).
//! The core idea is a navigable graph where each node connects to R neighbors chosen
//! via **alpha-RNG pruning**—a relaxed variant of the Relative Neighborhood Graph that
//! balances proximity and angular diversity.
//!
//! # Why DiskANN achieves 95%+ recall at sub-10ms latency
//!
//! 1. **Vamana graph structure**: The alpha parameter (typically 1.2) controls how
//!    aggressively long-range edges are retained. Values > 1.0 keep shortcuts that
//!    let greedy search traverse the graph in O(log n) hops.
//! 2. **SSD-friendly layout**: Each node's vector + neighbor list is packed into
//!    aligned disk pages, so a single read fetches everything needed to evaluate
//!    and expand a node.
//! 3. **Beam search with page cache**: Hot pages stay in an LRU cache, reducing
//!    SSD reads to only cold nodes. Typical workloads see 80-95% cache hit rates.
//! 4. **Filtered search during traversal**: Predicates are evaluated as the graph
//!    is explored, pruning ineligible branches early instead of post-filtering.
//!
//! # Alpha-RNG Pruning
//!
//! Given a candidate neighbor set for node p, the robust prune procedure greedily
//! selects neighbors: a candidate c is kept only if for every already-selected
//! neighbor n, `dist(p, c) <= alpha * dist(n, c)`. This ensures angular diversity—
//! neighbors are spread around p rather than clustered in one direction.

use crate::error::{Result, RuvectorError};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;

/// Configuration for the Vamana graph index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VamanaConfig {
    /// Maximum out-degree per node (R in the paper). Typical values: 32-64.
    pub max_degree: usize,
    /// Search list size (L). Larger values improve recall at the cost of latency.
    pub search_list_size: usize,
    /// Pruning parameter. Values > 1.0 retain long-range edges for faster traversal.
    /// Typical value: 1.2.
    pub alpha: f32,
    /// Number of threads for parallel graph construction (unused in this impl).
    pub num_build_threads: usize,
    /// Page size for SSD-aligned layout in bytes. Default: 4096.
    pub ssd_page_size: usize,
}

impl Default for VamanaConfig {
    fn default() -> Self {
        Self {
            max_degree: 32,
            search_list_size: 64,
            alpha: 1.2,
            num_build_threads: 1,
            ssd_page_size: 4096,
        }
    }
}

impl VamanaConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.max_degree == 0 {
            return Err(RuvectorError::InvalidParameter(
                "max_degree must be > 0".into(),
            ));
        }
        if self.search_list_size < 1 {
            return Err(RuvectorError::InvalidParameter(
                "search_list_size must be >= 1".into(),
            ));
        }
        if self.alpha < 1.0 {
            return Err(RuvectorError::InvalidParameter(
                "alpha must be >= 1.0".into(),
            ));
        }
        Ok(())
    }
}

/// In-memory Vamana graph for building and searching.
#[derive(Debug, Clone)]
pub struct VamanaGraph {
    /// Adjacency lists: `neighbors[i]` holds the neighbor IDs of node i.
    pub neighbors: Vec<Vec<u32>>,
    /// All vectors, row-major: `vectors[i]` is the embedding for node i.
    pub vectors: Vec<Vec<f32>>,
    /// Index of the medoid (entry point).
    pub medoid: u32,
    /// Build configuration.
    pub config: VamanaConfig,
}

impl VamanaGraph {
    /// Build a Vamana graph over the given vectors.
    ///
    /// The algorithm:
    /// 1. Find the geometric medoid as the entry point.
    /// 2. Initialize each node with random neighbors.
    /// 3. For each node, run greedy search to find its natural neighbors,
    ///    then apply robust pruning to select up to R diverse neighbors.
    pub fn build(vectors: Vec<Vec<f32>>, config: VamanaConfig) -> Result<Self> {
        config.validate()?;
        let n = vectors.len();
        if n == 0 {
            return Ok(Self {
                neighbors: vec![],
                vectors: vec![],
                medoid: 0,
                config,
            });
        }
        let dim = vectors[0].len();
        for v in vectors.iter() {
            if v.len() != dim {
                return Err(RuvectorError::DimensionMismatch {
                    expected: dim,
                    actual: v.len(),
                });
            }
        }

        let medoid = MedoidFinder::find_medoid(&vectors);
        let mut graph = Self {
            neighbors: vec![vec![]; n],
            vectors,
            medoid,
            config,
        };

        // Initialize with simple sequential neighbors (will be refined).
        for i in 0..n {
            let mut init_neighbors = Vec::new();
            for j in 0..n.min(graph.config.max_degree + 1) {
                if j as u32 != i as u32 {
                    init_neighbors.push(j as u32);
                }
                if init_neighbors.len() >= graph.config.max_degree {
                    break;
                }
            }
            graph.neighbors[i] = init_neighbors;
        }

        // Iterative refinement: for each node, search and prune.
        for i in 0..n {
            let query = graph.vectors[i].clone();
            let (candidates, _) =
                graph.greedy_search_internal(&query, graph.config.search_list_size);
            let mut candidate_set: Vec<u32> = candidates
                .into_iter()
                .filter(|&c| c != i as u32)
                .collect();
            // Merge existing neighbors into candidates.
            for &nb in &graph.neighbors[i] {
                if !candidate_set.contains(&nb) {
                    candidate_set.push(nb);
                }
            }
            let pruned =
                graph.robust_prune(i as u32, &candidate_set);
            graph.neighbors[i] = pruned.clone();

            // Add reverse edges and prune if needed.
            for &nb in &pruned {
                let nb_idx = nb as usize;
                if !graph.neighbors[nb_idx].contains(&(i as u32)) {
                    graph.neighbors[nb_idx].push(i as u32);
                    if graph.neighbors[nb_idx].len() > graph.config.max_degree {
                        let nb_neighbors = graph.neighbors[nb_idx].clone();
                        graph.neighbors[nb_idx] =
                            graph.robust_prune(nb, &nb_neighbors);
                    }
                }
            }
        }

        Ok(graph)
    }

    /// Greedy beam search from the medoid.
    ///
    /// Returns `(visited_in_order, distances)` for the `top_k` closest nodes.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(u32, f32)> {
        if self.vectors.is_empty() {
            return vec![];
        }
        let beam = self.config.search_list_size.max(top_k);
        let (candidates, dists) = self.greedy_search_internal(query, beam);
        candidates
            .into_iter()
            .zip(dists)
            .take(top_k)
            .collect()
    }

    /// Internal greedy search returning sorted candidates and distances.
    fn greedy_search_internal(&self, query: &[f32], list_size: usize) -> (Vec<u32>, Vec<f32>) {
        let mut visited = HashSet::new();
        // Min-heap of (distance, node_id) for the search frontier.
        let mut frontier: BinaryHeap<Reverse<OrdF32Pair>> = BinaryHeap::new();
        // Best results seen so far.
        let mut results: Vec<(f32, u32)> = Vec::new();

        let start = self.medoid;
        let d = l2_distance(&self.vectors[start as usize], query);
        frontier.push(Reverse(OrdF32Pair(d, start)));
        visited.insert(start);
        results.push((d, start));

        while let Some(Reverse(OrdF32Pair(_, node))) = frontier.pop() {
            for &nb in &self.neighbors[node as usize] {
                if visited.insert(nb) {
                    let dist = l2_distance(&self.vectors[nb as usize], query);
                    results.push((dist, nb));
                    frontier.push(Reverse(OrdF32Pair(dist, nb)));
                }
            }
            // Keep results bounded.
            if results.len() > list_size * 2 {
                results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                results.truncate(list_size);
            }
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(list_size);
        let ids: Vec<u32> = results.iter().map(|r| r.1).collect();
        let dists: Vec<f32> = results.iter().map(|r| r.0).collect();
        (ids, dists)
    }

    /// Robust pruning (alpha-RNG rule).
    ///
    /// From a candidate set, greedily picks neighbors for `node_id` such that
    /// each selected candidate c satisfies: for every already-selected neighbor n,
    /// `dist(node, c) <= alpha * dist(n, c)`. This promotes angular diversity.
    fn robust_prune(&self, node_id: u32, candidates: &[u32]) -> Vec<u32> {
        let node_vec = &self.vectors[node_id as usize];
        let mut scored: Vec<(f32, u32)> = candidates
            .iter()
            .filter(|&&c| c != node_id)
            .map(|&c| (l2_distance(node_vec, &self.vectors[c as usize]), c))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut selected: Vec<u32> = Vec::new();
        for (dist_to_node, cand) in scored {
            if selected.len() >= self.config.max_degree {
                break;
            }
            let cand_vec = &self.vectors[cand as usize];
            let keep = selected.iter().all(|&s| {
                let dist_s_c = l2_distance(&self.vectors[s as usize], cand_vec);
                dist_to_node <= self.config.alpha * dist_s_c
            });
            if keep {
                selected.push(cand);
            }
        }
        selected
    }
}

/// A node stored in the SSD-backed disk layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskNode {
    /// Node identifier.
    pub node_id: u32,
    /// Neighbor list.
    pub neighbors: Vec<u32>,
    /// The node's vector.
    pub vector: Vec<f32>,
}

/// IO statistics for disk-based search.
#[derive(Debug, Clone, Default)]
pub struct IOStats {
    /// Number of page-aligned reads performed.
    pub pages_read: usize,
    /// Total bytes read from disk.
    pub bytes_read: usize,
    /// Number of reads served from the page cache.
    pub cache_hits: usize,
}

/// Simulated SSD-backed disk index. Stores nodes in page-aligned slots and
/// provides beam search with IO accounting.
#[derive(Debug)]
pub struct DiskIndex {
    /// All nodes, indexed by node_id.
    nodes: Vec<DiskNode>,
    /// Page size in bytes.
    page_size: usize,
    /// Medoid entry point.
    medoid: u32,
    /// LRU page cache.
    cache: PageCache,
}

impl DiskIndex {
    /// Create a DiskIndex from a built VamanaGraph.
    pub fn from_graph(graph: &VamanaGraph, cache_size_pages: usize) -> Self {
        let nodes: Vec<DiskNode> = (0..graph.vectors.len())
            .map(|i| DiskNode {
                node_id: i as u32,
                neighbors: graph.neighbors[i].clone(),
                vector: graph.vectors[i].clone(),
            })
            .collect();
        Self {
            nodes,
            page_size: graph.config.ssd_page_size,
            medoid: graph.medoid,
            cache: PageCache::new(cache_size_pages),
        }
    }

    /// Beam search on the disk index, tracking IO statistics.
    ///
    /// Each node access simulates a page-aligned SSD read unless the page is
    /// cached.
    pub fn search_disk(
        &mut self,
        query: &[f32],
        top_k: usize,
        beam_width: usize,
    ) -> (Vec<(u32, f32)>, IOStats) {
        let mut stats = IOStats::default();
        if self.nodes.is_empty() {
            return (vec![], stats);
        }

        let mut visited = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<OrdF32Pair>> = BinaryHeap::new();
        let mut results: Vec<(f32, u32)> = Vec::new();

        let start = self.medoid;
        let node = self.read_node(start, &mut stats);
        let d = l2_distance(&node.vector, query);
        frontier.push(Reverse(OrdF32Pair(d, start)));
        visited.insert(start);
        results.push((d, start));

        while let Some(Reverse(OrdF32Pair(_, current))) = frontier.pop() {
            let node = self.read_node(current, &mut stats);
            let nb_list = node.neighbors.clone();
            for nb in nb_list {
                if visited.insert(nb) {
                    let nb_node = self.read_node(nb, &mut stats);
                    let dist = l2_distance(&nb_node.vector, query);
                    results.push((dist, nb));
                    frontier.push(Reverse(OrdF32Pair(dist, nb)));
                }
            }
            if results.len() > beam_width * 2 {
                results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                results.truncate(beam_width);
            }
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(top_k);
        let output = results.iter().map(|r| (r.1, r.0)).collect();
        (output, stats)
    }

    /// Simulate reading a node from disk, using the page cache.
    fn read_node(&mut self, node_id: u32, stats: &mut IOStats) -> &DiskNode {
        let page_id = node_id as usize; // One node per page (simplified).
        if self.cache.get(page_id) {
            stats.cache_hits += 1;
        } else {
            stats.pages_read += 1;
            stats.bytes_read += self.page_size;
            self.cache.insert(page_id);
        }
        &self.nodes[node_id as usize]
    }

    /// Search with a filter predicate applied during graph traversal.
    ///
    /// Unlike post-filtering, this evaluates the predicate as nodes are visited,
    /// so ineligible nodes still expand the search frontier but are excluded
    /// from results. This preserves graph connectivity while filtering.
    pub fn search_with_filter<F>(
        &mut self,
        query: &[f32],
        filter_fn: F,
        top_k: usize,
    ) -> Vec<(u32, f32)>
    where
        F: Fn(u32) -> bool,
    {
        if self.nodes.is_empty() {
            return vec![];
        }
        let mut visited = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<OrdF32Pair>> = BinaryHeap::new();
        let mut results: Vec<(f32, u32)> = Vec::new();
        let mut dummy_stats = IOStats::default();

        let start = self.medoid;
        let node = self.read_node(start, &mut dummy_stats);
        let d = l2_distance(&node.vector, query);
        frontier.push(Reverse(OrdF32Pair(d, start)));
        visited.insert(start);
        if filter_fn(start) {
            results.push((d, start));
        }

        while let Some(Reverse(OrdF32Pair(_, current))) = frontier.pop() {
            let node = self.read_node(current, &mut dummy_stats);
            let nb_list = node.neighbors.clone();
            for nb in nb_list {
                if visited.insert(nb) {
                    let nb_node = self.read_node(nb, &mut dummy_stats);
                    let dist = l2_distance(&nb_node.vector, query);
                    // Always expand the frontier (preserves connectivity).
                    frontier.push(Reverse(OrdF32Pair(dist, nb)));
                    // Only add to results if filter passes.
                    if filter_fn(nb) {
                        results.push((dist, nb));
                    }
                }
            }
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(top_k);
        results.iter().map(|r| (r.1, r.0)).collect()
    }
}

/// LRU page cache for the disk index.
///
/// Uses a simple ordered map to track access recency. Pages are evicted in
/// least-recently-used order when the cache exceeds its capacity.
#[derive(Debug)]
pub struct PageCache {
    /// Maximum number of pages to cache.
    capacity: usize,
    /// Access order counter.
    clock: u64,
    /// page_id -> last access time.
    entries: HashMap<usize, u64>,
    /// Total hits and accesses for hit rate tracking.
    total_hits: u64,
    total_accesses: u64,
}

impl PageCache {
    /// Create a new page cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            clock: 0,
            entries: HashMap::new(),
            total_hits: 0,
            total_accesses: 0,
        }
    }

    /// Check if a page is cached, updating recency on hit.
    pub fn get(&mut self, page_id: usize) -> bool {
        self.total_accesses += 1;
        self.clock += 1;
        if let Some(ts) = self.entries.get_mut(&page_id) {
            *ts = self.clock;
            self.total_hits += 1;
            true
        } else {
            false
        }
    }

    /// Insert a page, evicting the LRU entry if at capacity.
    pub fn insert(&mut self, page_id: usize) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.len() >= self.capacity {
            // Evict LRU.
            let lru = self
                .entries
                .iter()
                .min_by_key(|&(_, ts)| *ts)
                .map(|(&k, _)| k);
            if let Some(k) = lru {
                self.entries.remove(&k);
            }
        }
        self.clock += 1;
        self.entries.insert(page_id, self.clock);
    }

    /// Return the cache hit rate as a fraction in [0.0, 1.0].
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_accesses == 0 {
            0.0
        } else {
            self.total_hits as f64 / self.total_accesses as f64
        }
    }
}

/// Utility to find the geometric medoid of a dataset.
pub struct MedoidFinder;

impl MedoidFinder {
    /// Find the medoid—the point with the minimum sum of distances to all others.
    ///
    /// This is the natural entry point for the Vamana graph because it
    /// minimises the expected number of hops to any target.
    pub fn find_medoid(vectors: &[Vec<f32>]) -> u32 {
        if vectors.is_empty() {
            return 0;
        }
        let n = vectors.len();
        let mut best_idx = 0u32;
        let mut best_sum = f32::MAX;
        for i in 0..n {
            let sum: f32 = (0..n)
                .map(|j| l2_distance(&vectors[i], &vectors[j]))
                .sum();
            if sum < best_sum {
                best_sum = sum;
                best_idx = i as u32;
            }
        }
        best_idx
    }
}

/// L2 (Euclidean) squared distance between two vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

/// Helper for ordering f32 values in BinaryHeap.
#[derive(Debug, Clone, PartialEq)]
struct OrdF32Pair(f32, u32);

impl Eq for OrdF32Pair {}

impl PartialOrd for OrdF32Pair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32Pair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.1.cmp(&other.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..dim).map(|d| (i * dim + d) as f32).collect())
            .collect()
    }

    #[test]
    fn test_build_graph_basic() {
        let vecs = make_vectors(10, 4);
        let cfg = VamanaConfig { max_degree: 4, search_list_size: 8, ..Default::default() };
        let graph = VamanaGraph::build(vecs.clone(), cfg).unwrap();
        assert_eq!(graph.vectors.len(), 10);
        assert_eq!(graph.neighbors.len(), 10);
        for nb in &graph.neighbors {
            assert!(nb.len() <= 4);
        }
    }

    #[test]
    fn test_search_accuracy() {
        let mut vecs = make_vectors(20, 4);
        // Insert a known nearest neighbor at index 20.
        let query = vec![0.0, 0.0, 0.0, 0.0];
        vecs.push(vec![0.1, 0.1, 0.1, 0.1]); // very close to query
        let cfg = VamanaConfig { max_degree: 8, search_list_size: 30, ..Default::default() };
        let graph = VamanaGraph::build(vecs, cfg).unwrap();
        let results = graph.search(&query, 3);
        assert!(!results.is_empty());
        // The closest vector (index 20 = [0.1,0.1,0.1,0.1]) should be in top results.
        assert!(results.iter().any(|&(id, _)| id == 20));
    }

    #[test]
    fn test_robust_pruning_limits_degree() {
        let vecs = make_vectors(50, 4);
        let cfg = VamanaConfig { max_degree: 5, search_list_size: 16, ..Default::default() };
        let graph = VamanaGraph::build(vecs, cfg).unwrap();
        for nb in &graph.neighbors {
            assert!(nb.len() <= 5, "degree {} exceeds max 5", nb.len());
        }
    }

    #[test]
    fn test_disk_layout_roundtrip() {
        let vecs = make_vectors(10, 4);
        let cfg = VamanaConfig::default();
        let graph = VamanaGraph::build(vecs.clone(), cfg).unwrap();
        let disk = DiskIndex::from_graph(&graph, 16);
        for i in 0..10 {
            assert_eq!(disk.nodes[i].node_id, i as u32);
            assert_eq!(disk.nodes[i].vector, vecs[i]);
            assert_eq!(disk.nodes[i].neighbors, graph.neighbors[i]);
        }
    }

    #[test]
    fn test_page_cache_hits_and_misses() {
        let mut cache = PageCache::new(2);
        assert!(!cache.get(0)); // miss
        cache.insert(0);
        assert!(cache.get(0)); // hit
        cache.insert(1);
        cache.insert(2); // evicts page 0 (LRU)
        assert!(!cache.get(0)); // miss after eviction
        assert!(cache.get(1)); // still cached
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = PageCache::new(4);
        cache.insert(0);
        cache.insert(1);
        assert!(cache.get(0)); // hit
        assert!(cache.get(1)); // hit
        assert!(!cache.get(2)); // miss
        // 2 hits out of 3 accesses
        let rate = cache.cache_hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_filtered_search() {
        let mut vecs = make_vectors(15, 4);
        vecs.push(vec![0.1, 0.1, 0.1, 0.1]);
        let cfg = VamanaConfig { max_degree: 8, search_list_size: 20, ..Default::default() };
        let graph = VamanaGraph::build(vecs, cfg).unwrap();
        let mut disk = DiskIndex::from_graph(&graph, 32);
        // Filter: only even node IDs.
        let results = disk.search_with_filter(&[0.0, 0.0, 0.0, 0.0], |id| id % 2 == 0, 5);
        for &(id, _) in &results {
            assert_eq!(id % 2, 0, "filtered result {} is odd", id);
        }
    }

    #[test]
    fn test_medoid_selection() {
        let vecs = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5], // closest to center
        ];
        let medoid = MedoidFinder::find_medoid(&vecs);
        assert_eq!(medoid, 3, "medoid should be the most central point");
    }

    #[test]
    fn test_empty_dataset() {
        let cfg = VamanaConfig::default();
        let graph = VamanaGraph::build(vec![], cfg).unwrap();
        assert!(graph.vectors.is_empty());
        assert!(graph.neighbors.is_empty());
        let results = graph.search(&[1.0, 2.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_vector() {
        let vecs = vec![vec![1.0, 2.0, 3.0]];
        let cfg = VamanaConfig::default();
        let graph = VamanaGraph::build(vecs, cfg).unwrap();
        assert_eq!(graph.vectors.len(), 1);
        assert!(graph.neighbors[0].is_empty());
        let results = graph.search(&[1.0, 2.0, 3.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_io_stats_tracking() {
        let vecs = make_vectors(10, 4);
        let cfg = VamanaConfig { max_degree: 4, search_list_size: 10, ..Default::default() };
        let graph = VamanaGraph::build(vecs, cfg).unwrap();
        let mut disk = DiskIndex::from_graph(&graph, 2); // tiny cache
        let (_, stats) = disk.search_disk(&[0.0, 0.0, 0.0, 0.0], 3, 10);
        assert!(stats.pages_read > 0, "should have read pages from disk");
        assert_eq!(stats.bytes_read, stats.pages_read * 4096);
    }

    #[test]
    fn test_disk_search_returns_results() {
        let vecs = make_vectors(20, 4);
        let cfg = VamanaConfig { max_degree: 8, search_list_size: 20, ..Default::default() };
        let graph = VamanaGraph::build(vecs, cfg).unwrap();
        let mut disk = DiskIndex::from_graph(&graph, 32);
        let (results, stats) = disk.search_disk(&[0.0; 4], 5, 20);
        assert_eq!(results.len(), 5);
        // Results should be sorted by distance.
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted by distance");
        }
        assert!(stats.pages_read + stats.cache_hits > 0);
    }

    #[test]
    fn test_config_validation() {
        let bad = VamanaConfig { max_degree: 0, ..Default::default() };
        assert!(bad.validate().is_err());
        let bad_alpha = VamanaConfig { alpha: 0.5, ..Default::default() };
        assert!(bad_alpha.validate().is_err());
        let good = VamanaConfig::default();
        assert!(good.validate().is_ok());
    }
}
