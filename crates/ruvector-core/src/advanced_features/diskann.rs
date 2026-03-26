//! DiskANN / Vamana SSD-Backed Approximate Nearest Neighbor Index
//!
//! Implements the Vamana graph index from the DiskANN paper (Subramanya et al., 2019).
//! Each node connects to R neighbors chosen via **alpha-RNG pruning** -- a relaxed
//! Relative Neighborhood Graph balancing proximity and angular diversity.
//!
//! # Why DiskANN achieves 95%+ recall at sub-10ms
//!
//! - **Vamana graph**: alpha > 1.0 retains long-range shortcuts for O(log n) hops.
//! - **SSD layout**: node vector + neighbors packed in aligned pages; one read per hop.
//! - **Page cache**: LRU cache keeps hot pages in memory (80-95% hit rates typical).
//! - **Filtered traversal**: predicates evaluated during search, not post-filter.
//!
//! # Alpha-RNG Pruning
//!
//! A candidate c is kept only if for every already-selected neighbor n,
//! `dist(p, c) <= alpha * dist(n, c)`, ensuring angular diversity.

use crate::error::{Result, RuvectorError};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;

/// Configuration for the Vamana graph index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VamanaConfig {
    /// Maximum out-degree per node (R). Typical: 32-64.
    pub max_degree: usize,
    /// Search list size (L). Larger = better recall, slower search.
    pub search_list_size: usize,
    /// Pruning parameter (>= 1.0). Typical: 1.2.
    pub alpha: f32,
    /// Thread count for build (reserved for future parallel builds).
    pub num_build_threads: usize,
    /// Page size for SSD-aligned layout in bytes.
    pub ssd_page_size: usize,
}

impl Default for VamanaConfig {
    fn default() -> Self {
        Self { max_degree: 32, search_list_size: 64, alpha: 1.2, num_build_threads: 1, ssd_page_size: 4096 }
    }
}

impl VamanaConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.max_degree == 0 {
            return Err(RuvectorError::InvalidParameter("max_degree must be > 0".into()));
        }
        if self.search_list_size < 1 {
            return Err(RuvectorError::InvalidParameter("search_list_size must be >= 1".into()));
        }
        if self.alpha < 1.0 {
            return Err(RuvectorError::InvalidParameter("alpha must be >= 1.0".into()));
        }
        Ok(())
    }
}

/// In-memory Vamana graph for building and searching.
#[derive(Debug, Clone)]
pub struct VamanaGraph {
    /// Adjacency lists per node.
    pub neighbors: Vec<Vec<u32>>,
    /// Vectors, row-major.
    pub vectors: Vec<Vec<f32>>,
    /// Medoid (entry point) index.
    pub medoid: u32,
    /// Build config.
    pub config: VamanaConfig,
}

impl VamanaGraph {
    /// Build a Vamana graph: find medoid, init neighbors, then refine via greedy search + robust prune.
    pub fn build(vectors: Vec<Vec<f32>>, config: VamanaConfig) -> Result<Self> {
        config.validate()?;
        let n = vectors.len();
        if n == 0 {
            return Ok(Self { neighbors: vec![], vectors: vec![], medoid: 0, config });
        }
        let dim = vectors[0].len();
        for v in &vectors {
            if v.len() != dim {
                return Err(RuvectorError::DimensionMismatch { expected: dim, actual: v.len() });
            }
        }
        let medoid = MedoidFinder::find_medoid(&vectors);
        let mut graph = Self { neighbors: vec![vec![]; n], vectors, medoid, config };
        // Initialize with sequential neighbors.
        for i in 0..n {
            let mut nb = Vec::new();
            for j in 0..n.min(graph.config.max_degree + 1) {
                if j != i { nb.push(j as u32); }
                if nb.len() >= graph.config.max_degree { break; }
            }
            graph.neighbors[i] = nb;
        }
        // Refine: search, prune, add reverse edges.
        for i in 0..n {
            let query = graph.vectors[i].clone();
            let (cands, _) = graph.greedy_search_internal(&query, graph.config.search_list_size);
            let mut cset: Vec<u32> = cands.into_iter().filter(|&c| c != i as u32).collect();
            for &nb in &graph.neighbors[i] {
                if !cset.contains(&nb) { cset.push(nb); }
            }
            let pruned = graph.robust_prune(i as u32, &cset);
            graph.neighbors[i] = pruned.clone();
            for &nb in &pruned {
                let ni = nb as usize;
                if !graph.neighbors[ni].contains(&(i as u32)) {
                    graph.neighbors[ni].push(i as u32);
                    if graph.neighbors[ni].len() > graph.config.max_degree {
                        let nbs = graph.neighbors[ni].clone();
                        graph.neighbors[ni] = graph.robust_prune(nb, &nbs);
                    }
                }
            }
        }
        Ok(graph)
    }

    /// Greedy beam search returning top_k (node_id, distance) pairs.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(u32, f32)> {
        if self.vectors.is_empty() { return vec![]; }
        let beam = self.config.search_list_size.max(top_k);
        let (ids, dists) = self.greedy_search_internal(query, beam);
        ids.into_iter().zip(dists).take(top_k).collect()
    }

    fn greedy_search_internal(&self, query: &[f32], list_size: usize) -> (Vec<u32>, Vec<f32>) {
        let mut visited = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<OrdF32Pair>> = BinaryHeap::new();
        let mut results: Vec<(f32, u32)> = Vec::new();
        let start = self.medoid;
        let d = l2_sq(&self.vectors[start as usize], query);
        frontier.push(Reverse(OrdF32Pair(d, start)));
        visited.insert(start);
        results.push((d, start));
        while let Some(Reverse(OrdF32Pair(_, node))) = frontier.pop() {
            for &nb in &self.neighbors[node as usize] {
                if visited.insert(nb) {
                    let dist = l2_sq(&self.vectors[nb as usize], query);
                    results.push((dist, nb));
                    frontier.push(Reverse(OrdF32Pair(dist, nb)));
                }
            }
            if results.len() > list_size * 2 {
                results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                results.truncate(list_size);
            }
        }
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(list_size);
        (results.iter().map(|r| r.1).collect(), results.iter().map(|r| r.0).collect())
    }

    /// Robust prune: greedily select diverse neighbors via the alpha-RNG rule.
    fn robust_prune(&self, node_id: u32, candidates: &[u32]) -> Vec<u32> {
        let nv = &self.vectors[node_id as usize];
        let mut scored: Vec<(f32, u32)> = candidates.iter()
            .filter(|&&c| c != node_id)
            .map(|&c| (l2_sq(nv, &self.vectors[c as usize]), c))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut sel: Vec<u32> = Vec::new();
        for (d2n, cand) in scored {
            if sel.len() >= self.config.max_degree { break; }
            let cv = &self.vectors[cand as usize];
            if sel.iter().all(|&s| d2n <= self.config.alpha * l2_sq(&self.vectors[s as usize], cv)) {
                sel.push(cand);
            }
        }
        sel
    }
}

/// A node stored in SSD-backed layout: id + neighbors + vector in one page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskNode {
    pub node_id: u32,
    pub neighbors: Vec<u32>,
    pub vector: Vec<f32>,
}

/// IO statistics for disk-based search.
#[derive(Debug, Clone, Default)]
pub struct IOStats {
    pub pages_read: usize,
    pub bytes_read: usize,
    pub cache_hits: usize,
}

/// Simulated SSD-backed index with page-aligned reads and LRU cache.
#[derive(Debug)]
pub struct DiskIndex {
    nodes: Vec<DiskNode>,
    page_size: usize,
    medoid: u32,
    cache: PageCache,
}

impl DiskIndex {
    /// Create from a built VamanaGraph.
    pub fn from_graph(graph: &VamanaGraph, cache_size_pages: usize) -> Self {
        let nodes = (0..graph.vectors.len()).map(|i| DiskNode {
            node_id: i as u32, neighbors: graph.neighbors[i].clone(), vector: graph.vectors[i].clone(),
        }).collect();
        Self { nodes, page_size: graph.config.ssd_page_size, medoid: graph.medoid, cache: PageCache::new(cache_size_pages) }
    }

    /// Beam search with IO accounting.
    pub fn search_disk(&mut self, query: &[f32], top_k: usize, beam_width: usize) -> (Vec<(u32, f32)>, IOStats) {
        let mut stats = IOStats::default();
        if self.nodes.is_empty() { return (vec![], stats); }
        let mut visited = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<OrdF32Pair>> = BinaryHeap::new();
        let mut results: Vec<(f32, u32)> = Vec::new();
        let start = self.medoid;
        let d = l2_sq(&self.read_node(start, &mut stats).vector.clone(), query);
        frontier.push(Reverse(OrdF32Pair(d, start)));
        visited.insert(start);
        results.push((d, start));
        while let Some(Reverse(OrdF32Pair(_, cur))) = frontier.pop() {
            let nbs = self.read_node(cur, &mut stats).neighbors.clone();
            for nb in nbs {
                if visited.insert(nb) {
                    let v = self.read_node(nb, &mut stats).vector.clone();
                    let dist = l2_sq(&v, query);
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
        (results.iter().map(|r| (r.1, r.0)).collect(), stats)
    }

    fn read_node(&mut self, node_id: u32, stats: &mut IOStats) -> &DiskNode {
        let page_id = node_id as usize;
        if self.cache.get(page_id) { stats.cache_hits += 1; }
        else { stats.pages_read += 1; stats.bytes_read += self.page_size; self.cache.insert(page_id); }
        &self.nodes[node_id as usize]
    }

    /// Filtered search: predicates evaluated during traversal (not post-filter).
    /// Ineligible nodes still expand the frontier to preserve graph connectivity.
    pub fn search_with_filter<F>(&mut self, query: &[f32], filter_fn: F, top_k: usize) -> Vec<(u32, f32)>
    where F: Fn(u32) -> bool {
        if self.nodes.is_empty() { return vec![]; }
        let mut visited = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<OrdF32Pair>> = BinaryHeap::new();
        let mut results: Vec<(f32, u32)> = Vec::new();
        let mut io = IOStats::default();
        let start = self.medoid;
        let d = l2_sq(&self.read_node(start, &mut io).vector.clone(), query);
        frontier.push(Reverse(OrdF32Pair(d, start)));
        visited.insert(start);
        if filter_fn(start) { results.push((d, start)); }
        while let Some(Reverse(OrdF32Pair(_, cur))) = frontier.pop() {
            let nbs = self.read_node(cur, &mut io).neighbors.clone();
            for nb in nbs {
                if visited.insert(nb) {
                    let v = self.read_node(nb, &mut io).vector.clone();
                    let dist = l2_sq(&v, query);
                    frontier.push(Reverse(OrdF32Pair(dist, nb)));
                    if filter_fn(nb) { results.push((dist, nb)); }
                }
            }
        }
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(top_k);
        results.iter().map(|r| (r.1, r.0)).collect()
    }
}

/// LRU page cache tracking access recency via a clock counter.
#[derive(Debug)]
pub struct PageCache {
    capacity: usize,
    clock: u64,
    entries: HashMap<usize, u64>,
    total_hits: u64,
    total_accesses: u64,
}

impl PageCache {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, clock: 0, entries: HashMap::new(), total_hits: 0, total_accesses: 0 }
    }

    /// Returns true on cache hit, updating recency.
    pub fn get(&mut self, page_id: usize) -> bool {
        self.total_accesses += 1;
        self.clock += 1;
        if let Some(ts) = self.entries.get_mut(&page_id) {
            *ts = self.clock; self.total_hits += 1; true
        } else { false }
    }

    /// Insert a page, evicting LRU if at capacity.
    pub fn insert(&mut self, page_id: usize) {
        if self.capacity == 0 { return; }
        if self.entries.len() >= self.capacity {
            let lru = self.entries.iter().min_by_key(|&(_, ts)| *ts).map(|(&k, _)| k);
            if let Some(k) = lru { self.entries.remove(&k); }
        }
        self.clock += 1;
        self.entries.insert(page_id, self.clock);
    }

    /// Cache hit rate in [0.0, 1.0].
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_accesses == 0 { 0.0 } else { self.total_hits as f64 / self.total_accesses as f64 }
    }
}

/// Finds the geometric medoid (point minimising sum of distances to all others).
pub struct MedoidFinder;

impl MedoidFinder {
    pub fn find_medoid(vectors: &[Vec<f32>]) -> u32 {
        if vectors.is_empty() { return 0; }
        let (mut best_idx, mut best_sum) = (0u32, f32::MAX);
        for i in 0..vectors.len() {
            let sum: f32 = (0..vectors.len()).map(|j| l2_sq(&vectors[i], &vectors[j])).sum();
            if sum < best_sum { best_sum = sum; best_idx = i as u32; }
        }
        best_idx
    }
}

/// L2 squared distance.
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[derive(Debug, Clone, PartialEq)]
struct OrdF32Pair(f32, u32);
impl Eq for OrdF32Pair {}
impl PartialOrd for OrdF32Pair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for OrdF32Pair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal).then(self.1.cmp(&other.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vecs(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n).map(|i| (0..dim).map(|d| (i * dim + d) as f32).collect()).collect()
    }
    fn default_cfg(r: usize, l: usize) -> VamanaConfig {
        VamanaConfig { max_degree: r, search_list_size: l, ..Default::default() }
    }

    #[test]
    fn build_graph_basic() {
        let g = VamanaGraph::build(make_vecs(10, 4), default_cfg(4, 8)).unwrap();
        assert_eq!(g.vectors.len(), 10);
        for nb in &g.neighbors { assert!(nb.len() <= 4); }
    }

    #[test]
    fn search_accuracy() {
        let mut v = make_vecs(20, 4);
        v.push(vec![0.1, 0.1, 0.1, 0.1]);
        let g = VamanaGraph::build(v, default_cfg(8, 30)).unwrap();
        let r = g.search(&[0.0; 4], 3);
        assert!(r.iter().any(|&(id, _)| id == 20));
    }

    #[test]
    fn robust_pruning_limits_degree() {
        let g = VamanaGraph::build(make_vecs(50, 4), default_cfg(5, 16)).unwrap();
        for nb in &g.neighbors { assert!(nb.len() <= 5); }
    }

    #[test]
    fn disk_layout_roundtrip() {
        let v = make_vecs(10, 4);
        let g = VamanaGraph::build(v.clone(), VamanaConfig::default()).unwrap();
        let d = DiskIndex::from_graph(&g, 16);
        for i in 0..10 {
            assert_eq!(d.nodes[i].node_id, i as u32);
            assert_eq!(d.nodes[i].vector, v[i]);
            assert_eq!(d.nodes[i].neighbors, g.neighbors[i]);
        }
    }

    #[test]
    fn page_cache_hits_and_misses() {
        let mut c = PageCache::new(2);
        assert!(!c.get(0));
        c.insert(0);
        assert!(c.get(0));
        c.insert(1);
        c.insert(2); // evicts 0
        assert!(!c.get(0));
        assert!(c.get(1));
    }

    #[test]
    fn cache_hit_rate() {
        let mut c = PageCache::new(4);
        c.insert(0); c.insert(1);
        assert!(c.get(0)); assert!(c.get(1)); assert!(!c.get(2));
        assert!((c.cache_hit_rate() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn filtered_search() {
        let mut v = make_vecs(15, 4);
        v.push(vec![0.1; 4]);
        let g = VamanaGraph::build(v, default_cfg(8, 20)).unwrap();
        let mut d = DiskIndex::from_graph(&g, 32);
        let r = d.search_with_filter(&[0.0; 4], |id| id % 2 == 0, 5);
        for &(id, _) in &r { assert_eq!(id % 2, 0); }
    }

    #[test]
    fn medoid_selection() {
        let v = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        assert_eq!(MedoidFinder::find_medoid(&v), 3);
    }

    #[test]
    fn empty_dataset() {
        let g = VamanaGraph::build(vec![], VamanaConfig::default()).unwrap();
        assert!(g.vectors.is_empty());
        assert!(g.search(&[1.0, 2.0], 5).is_empty());
    }

    #[test]
    fn single_vector() {
        let g = VamanaGraph::build(vec![vec![1.0, 2.0, 3.0]], VamanaConfig::default()).unwrap();
        assert!(g.neighbors[0].is_empty());
        let r = g.search(&[1.0, 2.0, 3.0], 1);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0);
    }

    #[test]
    fn io_stats_tracking() {
        let g = VamanaGraph::build(make_vecs(10, 4), default_cfg(4, 10)).unwrap();
        let mut d = DiskIndex::from_graph(&g, 2);
        let (_, s) = d.search_disk(&[0.0; 4], 3, 10);
        assert!(s.pages_read > 0);
        assert_eq!(s.bytes_read, s.pages_read * 4096);
    }

    #[test]
    fn disk_search_sorted_results() {
        let g = VamanaGraph::build(make_vecs(20, 4), default_cfg(8, 20)).unwrap();
        let mut d = DiskIndex::from_graph(&g, 32);
        let (r, s) = d.search_disk(&[0.0; 4], 5, 20);
        assert_eq!(r.len(), 5);
        for w in r.windows(2) { assert!(w[0].1 <= w[1].1); }
        assert!(s.pages_read + s.cache_hits > 0);
    }

    #[test]
    fn config_validation() {
        assert!(VamanaConfig { max_degree: 0, ..Default::default() }.validate().is_err());
        assert!(VamanaConfig { alpha: 0.5, ..Default::default() }.validate().is_err());
        assert!(VamanaConfig::default().validate().is_ok());
    }
}
