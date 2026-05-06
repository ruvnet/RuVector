//! # ruvector-fresh-diskann
//!
//! Streaming online index maintenance for Vamana/DiskANN graphs.
//! Implements the FreshDiskANN lazy-consolidation approach:
//!
//! 1. New vectors land in an in-memory buffer — immediately searchable via brute-force scan.
//! 2. When the buffer hits the configured threshold `T`, consolidation fires:
//!    each buffered vector is beam-inserted into the Vamana graph with
//!    α-robust pruning + backlink repair. No full rebuild required.
//! 3. Deleted IDs are tracked in a tombstone set and filtered at search time.
//!
//! Reference: Jayaram Subramanya et al., "FreshDiskANN: A Fast and Accurate
//! Graph-Based ANN Index for Streaming Similarity Search" (arXiv:2105.09613).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FreshConfig {
    pub dim: usize,
    /// Maximum out-degree in the Vamana graph (R).
    pub max_degree: usize,
    /// Beam width used during Vamana build and beam-insert (L_build).
    pub build_beam: usize,
    /// Beam width used at query time (L_search).
    pub search_beam: usize,
    /// Alpha parameter for α-robust pruning (≥ 1.0).
    pub alpha: f32,
    /// When and how to consolidate the buffer.
    pub policy: ConsolidationPolicy,
}

#[derive(Debug, Clone)]
pub enum ConsolidationPolicy {
    /// Only consolidate when `consolidate()` is called explicitly.
    Manual,
    /// Consolidate immediately after every `insert()`.
    Eager,
    /// Consolidate automatically once the buffer reaches size `T`.
    Lazy(usize),
}

impl Default for FreshConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            max_degree: 32,
            build_beam: 64,
            search_beam: 64,
            alpha: 1.2,
            policy: ConsolidationPolicy::Lazy(1000),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub dist: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum FreshError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },
    #[error("index not built — call build() first")]
    NotBuilt,
    #[error("duplicate ID: {0}")]
    DuplicateId(String),
    #[error("empty index")]
    Empty,
}

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub consolidations: usize,
    pub consolidation_ms: u64,
    pub vectors_consolidated: usize,
}

// ---------------------------------------------------------------------------
// Core index
// ---------------------------------------------------------------------------

pub struct FreshDiskAnn {
    pub config: FreshConfig,

    // Flat contiguous vector storage for all nodes (consolidated + buffered).
    // Layout: [v0[0..dim], v1[0..dim], ...] so get_vec(i) = &store[i*dim..(i+1)*dim].
    store: Vec<f32>,

    // Adjacency list.  Consolidated nodes have non-empty lists; buffer nodes
    // start empty and are wired during consolidation.
    adj: Vec<Vec<u32>>,

    // Graph entry-point (medoid of consolidated vectors).
    medoid: u32,

    // ID mappings.
    ext_ids: Vec<String>,
    id_lookup: HashMap<String, u32>,
    next_id: u32,

    // Internal IDs that have been stored but not yet wired into the graph.
    buffer_ids: Vec<u32>,

    // Soft-deleted internal IDs.
    tombstones: HashSet<u32>,

    // True after at least one successful `build()`.
    built: bool,

    pub stats: Stats,
}

// ---------------------------------------------------------------------------
// Internal distance helper
// ---------------------------------------------------------------------------

#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ---------------------------------------------------------------------------
// Heap wrappers (min-heap by distance for frontier; max-heap for best-set)
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
struct MinEntry {
    id: u32,
    dist: f32,
}
impl Eq for MinEntry {}
impl PartialOrd for MinEntry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
impl Ord for MinEntry {
    fn cmp(&self, o: &Self) -> Ordering {
        o.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

#[derive(PartialEq)]
struct MaxEntry {
    id: u32,
    dist: f32,
}
impl Eq for MaxEntry {}
impl PartialOrd for MaxEntry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
impl Ord for MaxEntry {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.partial_cmp(&o.dist).unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// FreshDiskAnn implementation
// ---------------------------------------------------------------------------

impl FreshDiskAnn {
    pub fn new(config: FreshConfig) -> Self {
        Self {
            config,
            store: Vec::new(),
            adj: Vec::new(),
            medoid: 0,
            ext_ids: Vec::new(),
            id_lookup: HashMap::new(),
            next_id: 0,
            buffer_ids: Vec::new(),
            tombstones: HashSet::new(),
            built: false,
            stats: Stats::default(),
        }
    }

    // ---- Accessors --------------------------------------------------------

    #[inline]
    fn get_vec(&self, id: u32) -> &[f32] {
        let s = id as usize * self.config.dim;
        &self.store[s..s + self.config.dim]
    }

    pub fn len(&self) -> usize {
        self.adj.len() - self.tombstones.len()
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn buffer_len(&self) -> usize { self.buffer_ids.len() }

    // ---- Ingest -----------------------------------------------------------

    /// Stage a vector without wiring it into the graph.  Used for bulk
    /// pre-loading before `build()`.
    pub fn preload(&mut self, id: String, vector: Vec<f32>) -> Result<u32, FreshError> {
        let dim = self.config.dim;
        if vector.len() != dim {
            return Err(FreshError::DimMismatch { expected: dim, actual: vector.len() });
        }
        if self.id_lookup.contains_key(&id) {
            return Err(FreshError::DuplicateId(id));
        }
        let iid = self.next_id;
        self.next_id += 1;
        self.id_lookup.insert(id.clone(), iid);
        self.ext_ids.push(id);
        self.store.extend_from_slice(&vector);
        self.adj.push(Vec::new());
        Ok(iid)
    }

    /// Streaming insert — lands in buffer, consolidates according to policy.
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<(), FreshError> {
        let iid = self.preload(id, vector)?;
        self.buffer_ids.push(iid);

        match self.config.policy.clone() {
            ConsolidationPolicy::Eager => { self.consolidate(); }
            ConsolidationPolicy::Lazy(t) => {
                if self.buffer_ids.len() >= t { self.consolidate(); }
            }
            ConsolidationPolicy::Manual => {}
        }
        Ok(())
    }

    /// Soft-delete: marks ID as tombstone; filtered at search time.
    pub fn delete(&mut self, id: &str) -> bool {
        if let Some(&iid) = self.id_lookup.get(id) {
            self.tombstones.insert(iid);
            true
        } else {
            false
        }
    }

    // ---- Build (batch Vamana) ---------------------------------------------

    pub fn build(&mut self) -> Result<(), FreshError> {
        let n = self.adj.len();
        if n == 0 { return Err(FreshError::Empty); }

        let dim = self.config.dim;
        self.medoid = self.compute_medoid(n, dim);

        self.random_init(n);

        // Two-pass Vamana: pass 0 with alpha=1.0, pass 1 with configured alpha.
        let passes = if self.config.alpha > 1.0 { 2 } else { 1 };
        let mut order: Vec<u32> = (0..n as u32).collect();
        use rand::{SeedableRng, seq::SliceRandom};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEADBEEF);

        for pass in 0..passes {
            let alpha = if pass == 0 { 1.0f32 } else { self.config.alpha };
            order.shuffle(&mut rng);

            for &node in &order {
                let q = self.get_vec(node).to_vec();
                let cands = self.graph_beam_search(&q, self.config.build_beam, Some(node));
                let pruned = self.robust_prune(node, &cands, alpha);
                self.adj[node as usize] = pruned.clone();

                for &nbr in &pruned {
                    let ni = nbr as usize;
                    if !self.adj[ni].contains(&node) {
                        if self.adj[ni].len() < self.config.max_degree {
                            self.adj[ni].push(node);
                        } else {
                            let mut combined = self.adj[ni].clone();
                            combined.push(node);
                            self.adj[ni] = self.robust_prune(nbr, &combined, alpha);
                        }
                    }
                }
            }
        }

        self.built = true;
        Ok(())
    }

    // ---- Consolidation (FreshDiskANN beam-insert) -------------------------

    /// Wire all buffered vectors into the Vamana graph.
    pub fn consolidate(&mut self) {
        if self.buffer_ids.is_empty() || !self.built { return; }
        let t0 = Instant::now();
        let count = self.buffer_ids.len();
        let ids = std::mem::take(&mut self.buffer_ids);
        for &iid in &ids {
            self.beam_insert_node(iid);
        }
        self.stats.consolidations += 1;
        self.stats.consolidation_ms += t0.elapsed().as_millis() as u64;
        self.stats.vectors_consolidated += count;
    }

    fn beam_insert_node(&mut self, node: u32) {
        let q = self.get_vec(node).to_vec();
        let cands = self.graph_beam_search(&q, self.config.build_beam, Some(node));
        let pruned = self.robust_prune(node, &cands, self.config.alpha);
        self.adj[node as usize] = pruned.clone();

        for &nbr in &pruned {
            let ni = nbr as usize;
            if !self.adj[ni].contains(&node) {
                if self.adj[ni].len() < self.config.max_degree {
                    self.adj[ni].push(node);
                } else {
                    let mut combined = self.adj[ni].clone();
                    combined.push(node);
                    let repruned = self.robust_prune(nbr, &combined, self.config.alpha);
                    self.adj[ni] = repruned;
                }
            }
        }
    }

    // ---- Search -----------------------------------------------------------

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, FreshError> {
        let dim = self.config.dim;
        if query.len() != dim {
            return Err(FreshError::DimMismatch { expected: dim, actual: query.len() });
        }

        let beam = self.config.search_beam.max(k);
        let buf_set: HashSet<u32> = self.buffer_ids.iter().copied().collect();

        // Graph search over consolidated portion.
        let mut cands: Vec<(u32, f32)> = if self.built {
            self.graph_beam_search(query, beam, None)
                .into_iter()
                .filter(|id| !self.tombstones.contains(id) && !buf_set.contains(id))
                .map(|id| (id, l2sq(self.get_vec(id), query)))
                .collect()
        } else {
            // Pre-build fallback: brute-force all stored vectors.
            (0..self.adj.len() as u32)
                .filter(|id| !self.tombstones.contains(id) && !buf_set.contains(id))
                .map(|id| (id, l2sq(self.get_vec(id), query)))
                .collect()
        };

        // Brute-force scan of buffer vectors.
        for &bid in &self.buffer_ids {
            if !self.tombstones.contains(&bid) {
                cands.push((bid, l2sq(self.get_vec(bid), query)));
            }
        }

        cands.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        cands.dedup_by_key(|(id, _)| *id);

        Ok(cands.into_iter().take(k)
            .map(|(id, dist)| SearchResult { id: self.ext_ids[id as usize].clone(), dist })
            .collect())
    }

    // ---- Internal graph algorithms ----------------------------------------

    /// Greedy beam search on the Vamana graph (only follows wired edges).
    fn graph_beam_search(&self, query: &[f32], beam: usize, skip: Option<u32>) -> Vec<u32> {
        let n = self.adj.len();
        if n == 0 { return Vec::new(); }

        let mut visited = vec![false; n];
        let mut frontier = BinaryHeap::<MinEntry>::new();
        let mut best = BinaryHeap::<MaxEntry>::new();

        let sd = l2sq(self.get_vec(self.medoid), query);
        frontier.push(MinEntry { id: self.medoid, dist: sd });
        best.push(MaxEntry { id: self.medoid, dist: sd });
        visited[self.medoid as usize] = true;

        while let Some(cur) = frontier.pop() {
            if best.len() >= beam {
                if best.peek().map_or(false, |w| cur.dist > w.dist) { break; }
            }
            for &nbr in &self.adj[cur.id as usize] {
                if visited[nbr as usize] { continue; }
                if self.tombstones.contains(&nbr) { continue; }
                if skip.map_or(false, |s| s == nbr) { continue; }
                visited[nbr as usize] = true;
                let d = l2sq(self.get_vec(nbr), query);
                let dominated = best.len() >= beam
                    && best.peek().map_or(false, |w| d >= w.dist);
                if !dominated {
                    frontier.push(MinEntry { id: nbr, dist: d });
                    best.push(MaxEntry { id: nbr, dist: d });
                    if best.len() > beam { best.pop(); }
                }
            }
        }

        let mut result: Vec<(u32, f32)> = best.into_iter()
            .map(|e| (e.id, e.dist)).collect();
        result.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result.into_iter().map(|(id, _)| id).collect()
    }

    /// α-robust pruning: retain at most `max_degree` candidates such that no
    /// selected candidate is α-dominated by another selected candidate.
    fn robust_prune(&self, node: u32, candidates: &[u32], alpha: f32) -> Vec<u32> {
        let node_vec = self.get_vec(node).to_vec();
        let r = self.config.max_degree;

        let mut sorted: Vec<(u32, f32)> = candidates.iter()
            .filter(|&&c| c != node && !self.tombstones.contains(&c))
            .map(|&c| (c, l2sq(self.get_vec(c), &node_vec)))
            .collect();
        sorted.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let mut result: Vec<u32> = Vec::with_capacity(r);
        for (cid, cdist) in sorted {
            if result.len() >= r { break; }
            let cid_vec = self.get_vec(cid).to_vec();
            let dominated = result.iter().any(|&sel| {
                alpha * l2sq(self.get_vec(sel), &cid_vec) <= cdist
            });
            if !dominated { result.push(cid); }
        }
        result
    }

    fn compute_medoid(&self, n: usize, dim: usize) -> u32 {
        let mut centroid = vec![0.0f32; dim];
        for i in 0..n {
            let v = self.get_vec(i as u32);
            for d in 0..dim { centroid[d] += v[d]; }
        }
        for x in &mut centroid { *x /= n as f32; }
        (0..n as u32)
            .min_by(|&a, &b| {
                l2sq(self.get_vec(a), &centroid)
                    .partial_cmp(&l2sq(self.get_vec(b), &centroid))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0)
    }

    fn random_init(&mut self, n: usize) {
        use rand::prelude::*;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEADBEEF);
        let r = self.config.max_degree.min(n.saturating_sub(1));
        for i in 0..n {
            let mut nbrs = Vec::with_capacity(r);
            let mut tries = 0usize;
            while nbrs.len() < r && tries < r * 4 {
                let j = rng.gen_range(0..n) as u32;
                if j != i as u32 && !nbrs.contains(&j) { nbrs.push(j); }
                tries += 1;
            }
            self.adj[i] = nbrs;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn seeded_vecs(n: usize, dim: usize, seed: u64) -> Vec<(String, Vec<f32>)> {
        seeded_vecs_pfx(n, dim, seed, "v")
    }

    fn seeded_vecs_pfx(n: usize, dim: usize, seed: u64, pfx: &str) -> Vec<(String, Vec<f32>)> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|i| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
            (format!("{pfx}{i}"), v)
        }).collect()
    }

    #[test]
    fn test_build_and_search_finds_self() {
        let data = seeded_vecs(300, 32, 1);
        let query = data[42].1.clone();
        let mut idx = FreshDiskAnn::new(FreshConfig { dim: 32, max_degree: 16, build_beam: 32, search_beam: 32, alpha: 1.2, policy: ConsolidationPolicy::Manual });
        for (id, v) in &data { idx.preload(id.clone(), v.clone()).unwrap(); }
        idx.build().unwrap();
        let results = idx.search(&query, 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "v42");
        assert!(results[0].dist < 1e-5);
    }

    #[test]
    fn test_streaming_eager_finds_new_vector() {
        let base = seeded_vecs(200, 32, 10);
        let mut idx = FreshDiskAnn::new(FreshConfig { dim: 32, max_degree: 16, build_beam: 32, search_beam: 32, alpha: 1.2, policy: ConsolidationPolicy::Eager });
        for (id, v) in &base { idx.preload(id.clone(), v.clone()).unwrap(); }
        idx.build().unwrap();

        let new_vec: Vec<f32> = vec![0.0f32; 32];
        idx.insert("new_zero".to_string(), new_vec.clone()).unwrap();
        let results = idx.search(&new_vec, 1).unwrap();
        assert_eq!(results[0].id, "new_zero");
    }

    #[test]
    fn test_buffer_scan_finds_before_consolidation() {
        let base = seeded_vecs(100, 16, 20);
        let mut idx = FreshDiskAnn::new(FreshConfig { dim: 16, max_degree: 8, build_beam: 16, search_beam: 16, alpha: 1.2, policy: ConsolidationPolicy::Manual });
        for (id, v) in &base { idx.preload(id.clone(), v.clone()).unwrap(); }
        idx.build().unwrap();

        let stream = seeded_vecs_pfx(50, 16, 99, "s");
        for (id, v) in &stream { idx.insert(id.clone(), v.clone()).unwrap(); }
        assert_eq!(idx.buffer_len(), 50);

        // Buffer-only search should still find a stream vector
        let target = stream[7].1.clone();
        let results = idx.search(&target, 1).unwrap();
        assert_eq!(results[0].id, stream[7].0);
    }

    #[test]
    fn test_lazy_consolidation_empties_buffer() {
        let base = seeded_vecs(200, 16, 5);
        let mut idx = FreshDiskAnn::new(FreshConfig { dim: 16, max_degree: 8, build_beam: 16, search_beam: 16, alpha: 1.2, policy: ConsolidationPolicy::Lazy(50) });
        for (id, v) in &base { idx.preload(id.clone(), v.clone()).unwrap(); }
        idx.build().unwrap();

        let stream = seeded_vecs_pfx(50, 16, 77, "s");
        for (id, v) in &stream { idx.insert(id.clone(), v.clone()).unwrap(); }
        // Exactly 50 => should have triggered consolidation
        assert_eq!(idx.buffer_len(), 0);
        assert_eq!(idx.stats.consolidations, 1);
    }

    #[test]
    fn test_delete_hides_vector() {
        let data = seeded_vecs(200, 16, 7);
        let mut idx = FreshDiskAnn::new(FreshConfig { dim: 16, max_degree: 8, build_beam: 16, search_beam: 16, alpha: 1.2, policy: ConsolidationPolicy::Manual });
        for (id, v) in &data { idx.preload(id.clone(), v.clone()).unwrap(); }
        idx.build().unwrap();

        let target_id = "v0";
        let target_vec = data[0].1.clone();
        assert!(idx.delete(target_id));
        let results = idx.search(&target_vec, 5).unwrap();
        assert!(!results.iter().any(|r| r.id == target_id));
    }

    #[test]
    fn test_recall_at_10_after_streaming() {
        let n_base = 800usize;
        let n_stream = 200usize;
        let k = 10usize;
        let dim = 32usize;

        let base = seeded_vecs(n_base, dim, 42);
        let stream = seeded_vecs_pfx(n_stream, dim, 99, "s");
        let queries = seeded_vecs_pfx(30, dim, 123, "q");

        let all: Vec<(String, Vec<f32>)> = base.iter().chain(stream.iter()).cloned().collect();

        let mut idx = FreshDiskAnn::new(FreshConfig { dim, max_degree: 24, build_beam: 48, search_beam: 48, alpha: 1.2, policy: ConsolidationPolicy::Lazy(100) });
        for (id, v) in &base { idx.preload(id.clone(), v.clone()).unwrap(); }
        idx.build().unwrap();
        for (id, v) in &stream { idx.insert(id.clone(), v.clone()).unwrap(); }
        idx.consolidate();

        let mut total_recall = 0.0f64;
        for (_, qvec) in &queries {
            let mut brute: Vec<(usize, f32)> = all.iter().enumerate()
                .map(|(i, (_, v))| (i, l2sq(v, qvec))).collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt: HashSet<&str> = brute[..k].iter().map(|(i, _)| all[*i].0.as_str()).collect();

            let results = idx.search(qvec, k).unwrap();
            let found: HashSet<&str> = results.iter().map(|r| r.id.as_str()).collect();
            total_recall += gt.intersection(&found).count() as f64 / k as f64;
        }
        let avg = total_recall / queries.len() as f64;
        println!("recall@{k} = {avg:.3}");
        assert!(avg >= 0.70, "recall@{k} = {avg:.3}, want >= 0.70");
    }

    #[test]
    fn test_dim_mismatch_rejected() {
        let mut idx = FreshDiskAnn::new(FreshConfig { dim: 16, ..Default::default() });
        assert!(idx.preload("x".to_string(), vec![0.0; 32]).is_err());
    }

    #[test]
    fn test_duplicate_id_rejected() {
        let mut idx = FreshDiskAnn::new(FreshConfig { dim: 4, ..Default::default() });
        idx.preload("a".to_string(), vec![1.0; 4]).unwrap();
        assert!(idx.preload("a".to_string(), vec![2.0; 4]).is_err());
    }
}
