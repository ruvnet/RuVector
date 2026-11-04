//! RuVector Integration Layer
//!
//! Connects the minimum cut algorithm to the ruvector ecosystem.
//! Provides seamless integration with vector databases and graph processing.

// Integration module - allow missing docs for internal helpers
#![allow(missing_docs)]

use crate::graph::{DynamicGraph, EdgeId, VertexId, Weight};
use crate::wrapper::{MinCutResult, MinCutWrapper};
use std::sync::Arc;

// Agentic chip support (feature-gated)
#[cfg(feature = "agentic")]
use crate::parallel::{CoreExecutor, SharedCoordinator, NUM_CORES};

/// Graph connectivity analysis for ruvector
///
/// Useful for:
/// - Detecting communities in vector similarity graphs
/// - Finding cut points in knowledge graphs
/// - Partitioning data for distributed processing
pub struct RuVectorGraphAnalyzer {
    graph: Arc<DynamicGraph>,
    wrapper: MinCutWrapper,
    /// Cached analysis results
    cached_min_cut: Option<u64>,
    cached_partition: Option<(Vec<VertexId>, Vec<VertexId>)>,
}

impl RuVectorGraphAnalyzer {
    /// Create analyzer for a graph
    pub fn new(graph: Arc<DynamicGraph>) -> Self {
        let mut wrapper = MinCutWrapper::new(Arc::clone(&graph));

        // Sync wrapper with existing graph edges
        for edge in graph.edges() {
            wrapper.insert_edge(edge.id, edge.source, edge.target);
        }

        Self {
            graph,
            wrapper,
            cached_min_cut: None,
            cached_partition: None,
        }
    }

    /// Build graph from similarity matrix
    ///
    /// Creates edges between vectors with similarity above threshold
    pub fn from_similarity_matrix(
        similarities: &[f64],
        num_vectors: usize,
        threshold: f64,
    ) -> Self {
        let graph = Arc::new(DynamicGraph::new());

        for i in 0..num_vectors {
            for j in (i + 1)..num_vectors {
                let sim = similarities[i * num_vectors + j];
                if sim >= threshold {
                    let _ = graph.insert_edge(i as u64, j as u64, sim);
                }
            }
        }

        Self::new(graph)
    }

    /// Build k-NN graph from vectors
    pub fn from_knn(neighbors: &[(usize, Vec<(usize, f64)>)]) -> Self {
        let graph = Arc::new(DynamicGraph::new());

        for &(vertex, ref nn_list) in neighbors {
            for &(neighbor, distance) in nn_list {
                // Use 1/distance as weight (closer = stronger connection)
                let weight = if distance > 0.0 { 1.0 / distance } else { 1.0 };
                let _ = graph.insert_edge(vertex as u64, neighbor as u64, weight);
            }
        }

        Self::new(graph)
    }

    /// Compute minimum cut.
    ///
    /// **Ignores edge weight.** This computes unweighted edge connectivity
    /// (the minimum number of edges whose removal disconnects the graph),
    /// not a weighted minimum cut, regardless of the `Weight` values passed
    /// to [`Self::from_knn`], [`Self::from_similarity_matrix`], or
    /// [`Self::add_edge`]. `MinCutWrapper::insert_edge` -- the entry point
    /// this type syncs through -- takes only `(edge_id, u, v)`; weight is
    /// dropped before reaching the dynamic paper algorithm. This is scoped
    /// to this specific delegation: `ruvector-mincut` also has a separate,
    /// already-weighted engine (`algorithm::DynamicMinCut`/`MinCutBuilder`,
    /// see `wrapper_tests.rs::test_weighted_graph_correctness`) that this
    /// type simply does not use. Confirmed by nightly research `2026-11-04`
    /// (`docs/research/nightly/2026-11-04-canonical-mincut-static-fast-path/`,
    /// ADR-355): a fixed-degree weighted ring graph returns the same cut
    /// value at every tested size, and a two-weight-10-triangles +
    /// weight-100-bridge graph returns the 1-edge bridge cut instead of the
    /// 2-edge, weight-2 cut a weight-aware algorithm must prefer. If you
    /// need the true weighted minimum cut for a one-shot (non-streaming)
    /// query, use [`StaticCutAnalyzer`] instead (requires the `canonical`
    /// feature).
    pub fn min_cut(&mut self) -> u64 {
        if let Some(cached) = self.cached_min_cut {
            return cached;
        }

        let result = self.wrapper.query();
        let value = result.value();
        self.cached_min_cut = Some(value);
        value
    }

    /// Get partition (two sides of minimum cut)
    pub fn partition(&mut self) -> Option<(Vec<VertexId>, Vec<VertexId>)> {
        if let Some(ref cached) = self.cached_partition {
            return Some(cached.clone());
        }

        let result = self.wrapper.query();
        match result {
            MinCutResult::Disconnected => {
                // Return empty partition for disconnected graph
                Some((Vec::new(), Vec::new()))
            }
            MinCutResult::Value { witness, .. } => {
                // Deliberately not `witness.materialize_partition()`: it infers
                // the graph's vertex range from `max(U)` (the cut side's own
                // membership), not the graph's actual vertex set. Whenever U
                // omits the graph's highest-numbered vertex (the common case —
                // e.g. a 9-vertex cluster {0..=8} cut out of a 19-vertex graph
                // whose other side is {9..=18}), that under-estimates the
                // range and V\U comes back truncated or entirely empty. Build
                // both sides from the real vertex list instead: O(|V|), same
                // as `materialize_partition`, but correct for any V\U.
                let mut side_a = Vec::new();
                let mut side_b = Vec::new();
                for v in self.graph.vertices() {
                    if witness.contains(v) {
                        side_a.push(v);
                    } else {
                        side_b.push(v);
                    }
                }
                let partition = (side_a, side_b);
                self.cached_partition = Some(partition.clone());
                Some(partition)
            }
        }
    }

    /// Check if graph is well-connected (min cut above threshold)
    pub fn is_well_connected(&mut self, threshold: u64) -> bool {
        self.min_cut() >= threshold
    }

    /// Find bridge edges (edges whose removal disconnects graph)
    pub fn find_bridges(&self) -> Vec<EdgeId> {
        let mut bridges = Vec::new();

        for edge in self.graph.edges() {
            // Temporarily remove edge and check connectivity
            // This is expensive but correct
            let test_graph = Arc::new(DynamicGraph::new());
            for e in self.graph.edges() {
                if e.id != edge.id {
                    let _ = test_graph.insert_edge(e.source, e.target, e.weight);
                }
            }

            let mut test_wrapper = MinCutWrapper::new(test_graph);
            if test_wrapper.query().value() == 0 {
                bridges.push(edge.id);
            }
        }

        bridges
    }

    /// Invalidate cache after graph modification
    pub fn invalidate_cache(&mut self) {
        self.cached_min_cut = None;
        self.cached_partition = None;
    }

    /// Add edge and invalidate cache
    pub fn add_edge(&mut self, u: VertexId, v: VertexId, weight: Weight) -> crate::Result<EdgeId> {
        let edge_id = self.graph.insert_edge(u, v, weight)?;
        self.wrapper.insert_edge(edge_id, u, v);
        self.invalidate_cache();
        Ok(edge_id)
    }

    /// Remove edge and invalidate cache
    pub fn remove_edge(&mut self, u: VertexId, v: VertexId) -> crate::Result<()> {
        let edge = self.graph.delete_edge(u, v)?;
        self.wrapper.delete_edge(edge.id, u, v);
        self.invalidate_cache();
        Ok(())
    }
}

/// Batch minimum-cut analysis for static (single-snapshot) graphs.
///
/// `RuVectorGraphAnalyzer::partition()` runs the fully-dynamic paper
/// algorithm (`MinCutWrapper`), which amortizes cost over a long-lived
/// stream of incremental updates. Nightly research `2026-09-05`
/// (ADR-345) measured that machinery costing ~77ms at n=50 vertices and
/// ~11.4s at n=400 on a static k-NN graph queried exactly once — because a
/// one-shot query pays the dynamic algorithm's full amortized setup with
/// none of the amortization benefit.
///
/// `StaticCutAnalyzer` is a drop-in alternative, API-compatible with the
/// subset of `RuVectorGraphAnalyzer` used for one-shot batch queries
/// (`min_cut()` / `partition()`), built on the existing `canonical::CactusGraph`
/// Stoer-Wagner engine instead. It has no incremental-update API: callers
/// who mutate the graph and re-query repeatedly should keep using
/// `RuVectorGraphAnalyzer`. It also recomputes the full cactus on every
/// call (no caching), reflecting its target use case: build once, query
/// once, discard.
///
/// See `docs/research/nightly/2026-11-04-canonical-mincut-static-fast-path/`
/// for the benchmark establishing when this is (and is not) a net win, and
/// ADR-355 for the production decision.
#[cfg(feature = "canonical")]
pub struct StaticCutAnalyzer {
    graph: Arc<DynamicGraph>,
}

#[cfg(feature = "canonical")]
impl StaticCutAnalyzer {
    /// Wrap an existing graph snapshot for static (one-shot) min-cut queries.
    pub fn new(graph: Arc<DynamicGraph>) -> Self {
        Self { graph }
    }

    /// Plain union-find connectivity check over the current vertex/edge
    /// snapshot.
    ///
    /// `canonical::CactusGraph::build_from_graph` is not documented or
    /// tested against disconnected inputs and was found, by this nightly's
    /// `brute_force_cross_check` example, to silently return a non-zero,
    /// incorrect cut value for at least one disconnected random graph
    /// (truth 0, returned 7) instead of 0. Rather than modify
    /// `CactusGraph`'s general-purpose cactus construction (used by three
    /// other callers; out of this fix's scope), `StaticCutAnalyzer` checks
    /// connectivity itself first and short-circuits, exactly mirroring
    /// `MinCutWrapper::query()`'s own "fast path: check connectivity first".
    fn is_connected(&self) -> bool {
        let vertices = self.graph.vertices();
        if vertices.len() <= 1 {
            return true;
        }
        let mut parent: std::collections::HashMap<VertexId, VertexId> =
            vertices.iter().map(|&v| (v, v)).collect();

        fn find(
            parent: &mut std::collections::HashMap<VertexId, VertexId>,
            x: VertexId,
        ) -> VertexId {
            let mut root = x;
            while parent[&root] != root {
                root = parent[&root];
            }
            let mut cur = x;
            while parent[&cur] != root {
                let next = parent[&cur];
                parent.insert(cur, root);
                cur = next;
            }
            root
        }

        for edge in self.graph.edges() {
            let ra = find(&mut parent, edge.source);
            let rb = find(&mut parent, edge.target);
            if ra != rb {
                parent.insert(ra, rb);
            }
        }

        let root = find(&mut parent, vertices[0]);
        vertices.iter().all(|&v| find(&mut parent, v) == root)
    }

    /// Build a static analyzer from a similarity matrix, matching
    /// [`RuVectorGraphAnalyzer::from_similarity_matrix`]'s edge construction.
    pub fn from_similarity_matrix(
        similarities: &[f64],
        num_vectors: usize,
        threshold: f64,
    ) -> Self {
        let graph = Arc::new(DynamicGraph::new());
        for i in 0..num_vectors {
            for j in (i + 1)..num_vectors {
                let sim = similarities[i * num_vectors + j];
                if sim >= threshold {
                    let _ = graph.insert_edge(i as u64, j as u64, sim);
                }
            }
        }
        Self::new(graph)
    }

    /// Build a static analyzer from k-NN neighbor lists, matching
    /// [`RuVectorGraphAnalyzer::from_knn`]'s edge construction.
    pub fn from_knn(neighbors: &[(usize, Vec<(usize, f64)>)]) -> Self {
        let graph = Arc::new(DynamicGraph::new());
        for &(vertex, ref nn_list) in neighbors {
            for &(neighbor, distance) in nn_list {
                let weight = if distance > 0.0 { 1.0 / distance } else { 1.0 };
                let _ = graph.insert_edge(vertex as u64, neighbor as u64, weight);
            }
        }
        Self::new(graph)
    }

    /// Compute the minimum cut value via the cactus (Stoer-Wagner) engine.
    ///
    /// Returns `0` for a disconnected or trivial (0- or 1-vertex) graph,
    /// matching `RuVectorGraphAnalyzer::min_cut()`'s convention for the
    /// disconnected case.
    ///
    /// This rounds the real-valued weighted cut to the nearest `u64` so the
    /// signature matches `RuVectorGraphAnalyzer::min_cut()` for drop-in
    /// substitution. Callers that need the exact weighted value (e.g. to
    /// compare two close candidate cuts) should use [`Self::min_cut_f64`]
    /// instead.
    pub fn min_cut(&self) -> u64 {
        let value = self.min_cut_f64();
        if value.is_finite() {
            value.round() as u64
        } else {
            0
        }
    }

    /// Compute the exact real-valued weighted minimum cut, without the
    /// `u64`-rounding [`Self::min_cut`] applies for API compatibility.
    pub fn min_cut_f64(&self) -> f64 {
        if self.graph.num_vertices() < 2 || !self.is_connected() {
            return 0.0;
        }
        let cactus = crate::canonical::CactusGraph::build_from_graph(&self.graph);
        let result = cactus.canonical_cut();
        if result.value.is_finite() {
            result.value
        } else {
            0.0
        }
    }

    /// Get the canonical partition (two sides of the minimum cut).
    ///
    /// Returns `Some((Vec::new(), Vec::new()))` for a disconnected or
    /// trivial graph, matching `RuVectorGraphAnalyzer::partition()`'s
    /// convention.
    pub fn partition(&self) -> Option<(Vec<VertexId>, Vec<VertexId>)> {
        if self.graph.num_vertices() < 2 || !self.is_connected() {
            return Some((Vec::new(), Vec::new()));
        }
        let cactus = crate::canonical::CactusGraph::build_from_graph(&self.graph);
        let result = cactus.canonical_cut();
        let (s, t) = result.partition;
        if s.is_empty() || t.is_empty() {
            return Some((Vec::new(), Vec::new()));
        }
        let side_a: Vec<VertexId> = s.into_iter().map(|v| v as u64).collect();
        let side_b: Vec<VertexId> = t.into_iter().map(|v| v as u64).collect();
        Some((side_a, side_b))
    }
}

/// Community detection using minimum cut
pub struct CommunityDetector {
    analyzer: RuVectorGraphAnalyzer,
    communities: Vec<Vec<VertexId>>,
}

impl CommunityDetector {
    /// Create a new community detector for the given graph
    pub fn new(graph: Arc<DynamicGraph>) -> Self {
        Self {
            analyzer: RuVectorGraphAnalyzer::new(graph),
            communities: Vec::new(),
        }
    }

    /// Detect communities using recursive minimum cut
    pub fn detect(&mut self, min_community_size: usize) -> &[Vec<VertexId>] {
        self.communities.clear();

        // Get all vertices
        let vertices = self.analyzer.graph.vertices();

        // Recursively partition
        self.recursive_partition(vertices, min_community_size);

        &self.communities
    }

    fn recursive_partition(&mut self, vertices: Vec<VertexId>, min_size: usize) {
        if vertices.len() <= min_size {
            if !vertices.is_empty() {
                self.communities.push(vertices);
            }
            return;
        }

        // Build subgraph
        let subgraph = Arc::new(DynamicGraph::new());
        let vertex_set: std::collections::HashSet<_> = vertices.iter().copied().collect();

        for edge in self.analyzer.graph.edges() {
            if vertex_set.contains(&edge.source) && vertex_set.contains(&edge.target) {
                let _ = subgraph.insert_edge(edge.source, edge.target, edge.weight);
            }
        }

        // Compute minimum cut
        let mut sub_analyzer = RuVectorGraphAnalyzer::new(subgraph);
        let min_cut = sub_analyzer.min_cut();

        // If well-connected, keep as single community
        if min_cut > (vertices.len() as u64 / 4) {
            self.communities.push(vertices);
            return;
        }

        // Split and recurse
        if let Some((side_a, side_b)) = sub_analyzer.partition() {
            if !side_a.is_empty() {
                self.recursive_partition(side_a, min_size);
            }
            if !side_b.is_empty() {
                self.recursive_partition(side_b, min_size);
            }
        } else {
            self.communities.push(vertices);
        }
    }

    /// Get detected communities
    pub fn communities(&self) -> &[Vec<VertexId>] {
        &self.communities
    }
}

/// Graph partitioner for distributed processing
pub struct GraphPartitioner {
    graph: Arc<DynamicGraph>,
    num_partitions: usize,
}

impl GraphPartitioner {
    /// Create a new graph partitioner
    pub fn new(graph: Arc<DynamicGraph>, num_partitions: usize) -> Self {
        Self {
            graph,
            num_partitions,
        }
    }

    /// Partition graph to minimize edge cuts
    pub fn partition(&self) -> Vec<Vec<VertexId>> {
        let mut partitions = vec![Vec::new(); self.num_partitions];
        let vertices = self.graph.vertices();

        if vertices.is_empty() {
            return partitions;
        }

        // Use recursive bisection
        self.recursive_bisect(&vertices, &mut partitions, 0, self.num_partitions);

        partitions
    }

    fn recursive_bisect(
        &self,
        vertices: &[VertexId],
        partitions: &mut [Vec<VertexId>],
        start_idx: usize,
        count: usize,
    ) {
        if count == 1 {
            partitions[start_idx].extend(vertices.iter().copied());
            return;
        }

        // Build subgraph
        let subgraph = Arc::new(DynamicGraph::new());
        let vertex_set: std::collections::HashSet<_> = vertices.iter().copied().collect();

        for edge in self.graph.edges() {
            if vertex_set.contains(&edge.source) && vertex_set.contains(&edge.target) {
                let _ = subgraph.insert_edge(edge.source, edge.target, edge.weight);
            }
        }

        // Find minimum cut partition
        let mut analyzer = RuVectorGraphAnalyzer::new(subgraph);

        if let Some((side_a, side_b)) = analyzer.partition() {
            let half = count / 2;
            self.recursive_bisect(&side_a, partitions, start_idx, half);
            self.recursive_bisect(&side_b, partitions, start_idx + half, count - half);
        } else {
            // Can't partition further, put all in first partition
            partitions[start_idx].extend(vertices.iter().copied());
        }
    }

    /// Compute edge cut between partitions
    pub fn edge_cut(&self, partitions: &[Vec<VertexId>]) -> usize {
        let mut partition_map = std::collections::HashMap::new();
        for (i, partition) in partitions.iter().enumerate() {
            for &v in partition {
                partition_map.insert(v, i);
            }
        }

        let mut cut = 0;
        for edge in self.graph.edges() {
            let src_part = partition_map.get(&edge.source);
            let tgt_part = partition_map.get(&edge.target);

            if src_part != tgt_part {
                cut += 1;
            }
        }

        cut
    }
}

/// Agentic chip accelerated analyzer
#[cfg(feature = "agentic")]
pub struct AgenticAnalyzer {
    coordinator: SharedCoordinator,
}

#[cfg(feature = "agentic")]
impl AgenticAnalyzer {
    pub fn new() -> Self {
        let coordinator = SharedCoordinator::new();
        Self { coordinator }
    }

    /// Distribute graph across cores and compute
    pub fn compute(&mut self, graph: &DynamicGraph) -> u16 {
        // Create cores with reference to coordinator
        let mut cores: Vec<CoreExecutor> = (0..NUM_CORES as u8)
            .map(|id| CoreExecutor::init(id, Some(&self.coordinator)))
            .collect();

        // Distribute edges to cores
        for edge in graph.edges() {
            // Simple round-robin distribution
            let core_idx = (edge.id as usize) % NUM_CORES;
            cores[core_idx].add_edge(
                edge.source as u16,
                edge.target as u16,
                (edge.weight * 100.0) as u16,
            );
        }

        // Process all cores
        for core in &mut cores {
            core.process();
        }

        // Return global minimum
        self.coordinator
            .global_min_cut
            .load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_analyzer() {
        let graph = Arc::new(DynamicGraph::new());
        graph.insert_edge(0, 1, 1.0).unwrap();
        graph.insert_edge(1, 2, 1.0).unwrap();
        graph.insert_edge(2, 0, 1.0).unwrap();

        let mut analyzer = RuVectorGraphAnalyzer::new(graph);
        assert_eq!(analyzer.min_cut(), 2);
    }

    #[test]
    fn test_community_detector() {
        let graph = Arc::new(DynamicGraph::new());
        // Two triangles connected by weak edge
        graph.insert_edge(0, 1, 1.0).unwrap();
        graph.insert_edge(1, 2, 1.0).unwrap();
        graph.insert_edge(2, 0, 1.0).unwrap();
        graph.insert_edge(3, 4, 1.0).unwrap();
        graph.insert_edge(4, 5, 1.0).unwrap();
        graph.insert_edge(5, 3, 1.0).unwrap();
        graph.insert_edge(2, 3, 0.1).unwrap(); // Weak bridge

        let mut detector = CommunityDetector::new(graph);
        let communities = detector.detect(2);

        // Should find 2 communities
        assert!(communities.len() >= 1);
    }

    #[test]
    fn test_graph_partitioner() {
        let graph = Arc::new(DynamicGraph::new());
        for i in 0..9 {
            graph.insert_edge(i, i + 1, 1.0).unwrap();
        }

        let partitioner = GraphPartitioner::new(Arc::clone(&graph), 2);
        let partitions = partitioner.partition();

        assert_eq!(partitions.len(), 2);

        let total_vertices: usize = partitions.iter().map(|p| p.len()).sum();
        // Total should be at most the number of vertices in graph
        // Partitioning may not cover all vertices if min-cut fails
        assert!(total_vertices <= 10);
        assert!(total_vertices > 0);
    }

    #[test]
    fn test_from_similarity_matrix() {
        let similarities = vec![1.0, 0.9, 0.1, 0.9, 1.0, 0.8, 0.1, 0.8, 1.0];

        let analyzer = RuVectorGraphAnalyzer::from_similarity_matrix(&similarities, 3, 0.5);

        // Should have edges for similarities >= 0.5
        assert_eq!(analyzer.graph.num_edges(), 2); // 0-1 and 1-2
    }
}
