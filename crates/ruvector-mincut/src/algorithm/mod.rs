//! Core Dynamic Minimum Cut Algorithm
//!
//! Provides the main algorithm with:
//! - Exact global cuts with a matching cached partition
//! - Support for edge insertions and deletions
//! - Both exact and approximate modes
//!
//! ## Modules
//!
//! - [`replacement`]: Replacement edge index for tree edge deletions
//! - [`approximate`]: (1+ε)-approximate min-cut for all cut sizes (SODA 2025)

pub mod approximate;
mod exact;
pub mod replacement;

pub use replacement::{ReplacementEdgeIndex, ReplacementIndexStats};

use crate::error::{MinCutError, Result};
use crate::graph::{DynamicGraph, Edge, EdgeId, VertexId, Weight};
use crate::time_compat::PortableInstant;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Configuration for the minimum cut algorithm
#[derive(Debug, Clone)]
pub struct MinCutConfig {
    /// Maximum cut size supported for exact algorithm
    pub max_exact_cut_size: usize,
    /// Epsilon for approximate algorithm (0 < ε ≤ 1)
    pub epsilon: f64,
    /// Whether to use approximate mode
    pub approximate: bool,
    /// Enable parallel computation
    pub parallel: bool,
    /// Cache size for intermediate results
    pub cache_size: usize,
}

impl Default for MinCutConfig {
    fn default() -> Self {
        Self {
            max_exact_cut_size: 1000,
            epsilon: 0.1,
            approximate: false,
            parallel: true,
            cache_size: 10000,
        }
    }
}

/// Result of a minimum cut query
#[derive(Debug, Clone)]
pub struct MinCutResult {
    /// The minimum cut value
    pub value: f64,
    /// Edges in the cut (if requested)
    pub cut_edges: Option<Vec<Edge>>,
    /// Partition (if requested): (S, T) where S and T are vertex sets
    pub partition: Option<(Vec<VertexId>, Vec<VertexId>)>,
    /// Whether this is an exact or approximate result
    pub is_exact: bool,
    /// Approximation ratio (1.0 for exact)
    pub approximation_ratio: f64,
}

/// Statistics about algorithm performance
#[derive(Debug, Clone, Default)]
pub struct AlgorithmStats {
    /// Total number of insertions
    pub insertions: u64,
    /// Total number of deletions
    pub deletions: u64,
    /// Total number of queries
    pub queries: u64,
    /// Average update time in microseconds
    pub avg_update_time_us: f64,
    /// Average query time in microseconds
    pub avg_query_time_us: f64,
    /// Number of full cut recomputations (including initial construction)
    pub restructures: u64,
}

/// The main dynamic minimum cut structure
pub struct DynamicMinCut {
    /// The underlying graph
    graph: Arc<RwLock<DynamicGraph>>,
    /// One side of the same cut that supplies current_min_cut.
    cut_side: HashSet<VertexId>,
    /// Sorted vertices and crossing edges from the same solver snapshot.
    cut_vertices: Vec<VertexId>,
    current_cut_edges: Vec<Edge>,
    /// Current minimum cut value
    current_min_cut: f64,
    /// Configuration
    config: MinCutConfig,
    /// Statistics
    stats: Arc<RwLock<AlgorithmStats>>,
}

impl DynamicMinCut {
    /// Create a new dynamic minimum cut structure
    pub fn new(config: MinCutConfig) -> Self {
        Self {
            graph: Arc::new(RwLock::new(DynamicGraph::new())),
            cut_side: HashSet::new(),
            cut_vertices: Vec::new(),
            current_cut_edges: Vec::new(),
            current_min_cut: f64::INFINITY,
            config,
            stats: Arc::new(RwLock::new(AlgorithmStats::default())),
        }
    }

    /// Build from an existing graph without cloning its adjacency maps.
    pub fn from_graph(graph: DynamicGraph, config: MinCutConfig) -> Result<Self> {
        let mut mincut = Self::new(config);
        mincut.graph = Arc::new(RwLock::new(graph));
        mincut.recompute_min_cut();
        Ok(mincut)
    }

    /// Insert an edge. Non-crossing insertions between existing vertices
    /// preserve the current minimum cut, so no solver recomputation is needed.
    pub fn insert_edge(&mut self, u: VertexId, v: VertexId, weight: Weight) -> Result<f64> {
        let start_time = PortableInstant::now();
        let preserves_cut = {
            let graph = self.graph.write();
            let existing = graph.has_vertex(u) && graph.has_vertex(v);
            graph.insert_edge(u, v, weight)?;
            existing && self.cut_side.contains(&u) == self.cut_side.contains(&v)
        };
        // Nonnegative insertion cannot lower any cut. If this cut's weight
        // is unchanged and the vertex set is unchanged, it remains optimal.
        if !preserves_cut {
            self.recompute_min_cut();
        }
        self.record_update(true, start_time.elapsed().as_secs_f64() * 1_000_000.0);
        Ok(self.current_min_cut)
    }

    /// Delete an edge. Deletions can expose a competing minimum cut, including
    /// when the deleted edge does not cross the previously selected partition.
    pub fn delete_edge(&mut self, u: VertexId, v: VertexId) -> Result<f64> {
        let start_time = PortableInstant::now();
        self.graph.write().delete_edge(u, v)?;
        self.recompute_min_cut();
        self.record_update(false, start_time.elapsed().as_secs_f64() * 1_000_000.0);
        Ok(self.current_min_cut)
    }

    /// Insert a batch with one final solve. Invalid input leaves the graph unchanged.
    /// As with other updates, callers must not mutate graph() directly.
    pub fn insert_edges(&mut self, edges: &[(VertexId, VertexId, Weight)]) -> Result<f64> {
        let start = PortableInstant::now();
        let mut preserves = true;
        {
            let graph = self.graph.write();
            let mut seen = HashSet::with_capacity(edges.len());
            for &(u, v, weight) in edges {
                if u == v { return Err(MinCutError::InvalidEdge(u, v)); }
                if !weight.is_finite() || weight < 0.0 {
                    return Err(MinCutError::InvalidParameter("edge weight must be finite and nonnegative".into()));
                }
                if graph.has_edge(u, v) || !seen.insert((u.min(v), u.max(v))) {
                    return Err(MinCutError::EdgeExists(u, v));
                }
                preserves &= graph.has_vertex(u) && graph.has_vertex(v)
                    && self.cut_side.contains(&u) == self.cut_side.contains(&v);
            }
            for &(u, v, weight) in edges { graph.insert_edge(u, v, weight)?; }
        }
        if !preserves { self.recompute_min_cut(); }
        self.record_batch(true, edges.len(), start.elapsed().as_secs_f64() * 1_000_000.0);
        Ok(self.current_min_cut)
    }

    /// Delete a validated batch with one final solve; invalid input makes no changes.
    pub fn delete_edges(&mut self, edges: &[(VertexId, VertexId)]) -> Result<f64> {
        let start = PortableInstant::now();
        {
            let graph = self.graph.write();
            let mut seen = HashSet::with_capacity(edges.len());
            for &(u, v) in edges {
                if !graph.has_edge(u, v) || !seen.insert((u.min(v), u.max(v))) {
                    return Err(MinCutError::EdgeNotFound(u, v));
                }
            }
            for &(u, v) in edges { graph.delete_edge(u, v)?; }
        }
        if !edges.is_empty() { self.recompute_min_cut(); }
        self.record_batch(false, edges.len(), start.elapsed().as_secs_f64() * 1_000_000.0);
        Ok(self.current_min_cut)
    }

    /// Set an edge weight, inserting a missing edge. Validation precedes mutation.
    /// Increases inside the cached partition cannot change its optimality.
    pub fn update_edge(&mut self, u: VertexId, v: VertexId, weight: Weight) -> Result<f64> {
        let old = self.graph.read().edge_weight(u, v);
        let Some(old) = old else { return self.insert_edge(u, v, weight); };
        let start = PortableInstant::now();
        self.graph.write().update_edge_weight(u, v, weight)?;
        let preserves = weight == old || (weight > old
            && self.cut_side.contains(&u) == self.cut_side.contains(&v));
        if !preserves { self.recompute_min_cut(); }
        // A replacement counts as a deletion and an insertion, as in the legacy bindings.
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        self.record_batch(false, 1, elapsed / 2.0);
        self.record_batch(true, 1, elapsed / 2.0);
        Ok(self.current_min_cut)
    }

    fn record_batch(&self, insertion: bool, count: usize, elapsed_us: f64) {
        if count == 0 { return; }
        let mut stats = self.stats.write();
        let old_count = (stats.insertions + stats.deletions) as f64;
        if insertion { stats.insertions += count as u64; }
        else { stats.deletions += count as u64; }
        stats.avg_update_time_us = (stats.avg_update_time_us * old_count + elapsed_us)
            / (old_count + count as f64);
    }

    /// Get the current minimum cut value (O(1))
    pub fn min_cut_value(&self) -> f64 {
        let start_time = PortableInstant::now();

        let value = self.current_min_cut;

        // Update query statistics
        let elapsed = start_time.elapsed().as_secs_f64() * 1_000_000.0;
        let mut stats = self.stats.write();
        stats.queries += 1;
        let n = stats.queries as f64;
        stats.avg_query_time_us = (stats.avg_query_time_us * (n - 1.0) + elapsed) / n;

        value
    }

    /// Get detailed minimum cut result
    pub fn min_cut(&self) -> MinCutResult {
        let value = self.min_cut_value();
        let (partition_s, partition_t) = self.partition();
        let edges = self.cut_edges();

        MinCutResult {
            value,
            cut_edges: Some(edges),
            partition: Some((partition_s, partition_t)),
            is_exact: !self.config.approximate,
            approximation_ratio: if self.config.approximate {
                1.0 + self.config.epsilon
            } else {
                1.0
            },
        }
    }

    /// Get the cut partition in sorted vertex order.
    pub fn partition(&self) -> (Vec<VertexId>, Vec<VertexId>) {
        self.cut_vertices
            .iter()
            .copied()
            .partition(|v| self.cut_side.contains(v))
    }

    /// Get cached crossing edges without scanning graph adjacency again.
    pub fn cut_edges(&self) -> Vec<Edge> {
        self.current_cut_edges.clone()
    }

    /// Check if graph is connected
    pub fn is_connected(&self) -> bool {
        let graph = self.graph.read();
        graph.is_connected()
    }

    /// Get algorithm statistics
    pub fn stats(&self) -> AlgorithmStats {
        self.stats.read().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        *self.stats.write() = AlgorithmStats::default();
    }

    /// Get configuration
    pub fn config(&self) -> &MinCutConfig {
        &self.config
    }

    /// Get reference to underlying graph. Mutate through this solver's update
    /// methods; direct graph mutation bypasses cached cut maintenance.
    pub fn graph(&self) -> Arc<RwLock<DynamicGraph>> {
        Arc::clone(&self.graph)
    }

    /// Number of vertices
    pub fn num_vertices(&self) -> usize {
        self.graph.read().num_vertices()
    }

    /// Number of edges
    pub fn num_edges(&self) -> usize {
        self.graph.read().num_edges()
    }

    fn record_update(&self, insertion: bool, elapsed_us: f64) {
        let mut stats = self.stats.write();
        if insertion {
            stats.insertions += 1;
        } else {
            stats.deletions += 1;
        }
        let count = (stats.insertions + stats.deletions) as f64;
        stats.avg_update_time_us += (elapsed_us - stats.avg_update_time_us) / count;
    }

    fn recompute_min_cut(&mut self) {
        self.stats.write().restructures += 1;
        let graph = self.graph.read();
        let mut vertices = graph.vertices();
        vertices.sort_unstable();
        let indices: HashMap<_, _> = vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut original_edges = graph.edges();
        original_edges.sort_unstable_by_key(|edge| edge.canonical_endpoints());
        let edges: Vec<_> = original_edges
            .iter()
            .map(|edge| {
                let (u, v) = edge.canonical_endpoints();
                (indices[&u], indices[&v], edge.weight)
            })
            .collect();
        let (_, side) = exact::minimum_cut(vertices.len(), &edges);
        self.cut_side = side.into_iter().map(|i| vertices[i]).collect();
        self.current_cut_edges = original_edges
            .into_iter()
            .filter(|edge| {
                self.cut_side.contains(&edge.source) != self.cut_side.contains(&edge.target)
            })
            .collect();
        // Sum the original weights in the same order returned to callers.
        self.current_min_cut = if vertices.len() < 2 {
            f64::INFINITY
        } else {
            self.current_cut_edges.iter().map(|edge| edge.weight).sum()
        };
        self.cut_vertices = vertices;
    }
}

/// Builder for DynamicMinCut
pub struct MinCutBuilder {
    config: MinCutConfig,
    initial_edges: Vec<(VertexId, VertexId, Weight)>,
}

impl MinCutBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: MinCutConfig::default(),
            initial_edges: Vec::new(),
        }
    }

    /// Use exact algorithm
    pub fn exact(mut self) -> Self {
        self.config.approximate = false;
        self
    }

    /// Use approximate algorithm with given epsilon
    pub fn approximate(mut self, epsilon: f64) -> Self {
        assert!(epsilon > 0.0 && epsilon <= 1.0, "Epsilon must be in (0, 1]");
        self.config.approximate = true;
        self.config.epsilon = epsilon;
        self
    }

    /// Set maximum cut size for exact algorithm
    pub fn max_cut_size(mut self, size: usize) -> Self {
        self.config.max_exact_cut_size = size;
        self
    }

    /// Enable or disable parallel computation
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.config.parallel = enabled;
        self
    }

    /// Add initial edges
    pub fn with_edges(mut self, edges: Vec<(VertexId, VertexId, Weight)>) -> Self {
        self.initial_edges = edges;
        self
    }

    /// Build the minimum cut structure
    pub fn build(self) -> Result<DynamicMinCut> {
        if self.initial_edges.is_empty() {
            Ok(DynamicMinCut::new(self.config))
        } else {
            // Create graph with initial edges
            let graph = DynamicGraph::new();
            for (u, v, weight) in &self.initial_edges {
                graph.insert_edge(*u, *v, *weight)?;
            }

            DynamicMinCut::from_graph(graph, self.config)
        }
    }
}

impl Default for MinCutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let mincut = DynamicMinCut::new(MinCutConfig::default());
        assert_eq!(mincut.min_cut_value(), f64::INFINITY);
        assert_eq!(mincut.num_vertices(), 0);
        assert_eq!(mincut.num_edges(), 0);
    }

    #[test]
    fn test_single_edge() {
        let mincut = MinCutBuilder::new()
            .with_edges(vec![(1, 2, 1.0)])
            .build()
            .unwrap();

        assert_eq!(mincut.num_vertices(), 2);
        assert_eq!(mincut.num_edges(), 1);
        assert_eq!(mincut.min_cut_value(), 1.0);
        assert!(mincut.is_connected());
    }

    #[test]
    fn test_triangle() {
        let edges = vec![(1, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0)];

        let mincut = MinCutBuilder::new().with_edges(edges).build().unwrap();

        assert_eq!(mincut.num_vertices(), 3);
        assert_eq!(mincut.num_edges(), 3);
        assert_eq!(mincut.min_cut_value(), 2.0); // Minimum cut is 2
        assert!(mincut.is_connected());
    }

    #[test]
    fn test_insert_edge() {
        let mut mincut = MinCutBuilder::new()
            .with_edges(vec![(1, 2, 1.0)])
            .build()
            .unwrap();

        let cut_value = mincut.insert_edge(2, 3, 1.0).unwrap();
        assert_eq!(mincut.num_edges(), 2);
        assert_eq!(cut_value, 1.0);
    }

    #[test]
    fn test_delete_edge() {
        let mut mincut = MinCutBuilder::new()
            .with_edges(vec![(1, 2, 1.0), (2, 3, 1.0)])
            .build()
            .unwrap();

        assert!(mincut.is_connected());

        let cut_value = mincut.delete_edge(1, 2).unwrap();
        assert_eq!(mincut.num_edges(), 1);
        // After deleting edge, graph becomes disconnected
        assert_eq!(cut_value, 0.0);
    }

    #[test]
    fn test_disconnected_graph() {
        let mincut = MinCutBuilder::new()
            .with_edges(vec![(1, 2, 1.0), (3, 4, 1.0)])
            .build()
            .unwrap();

        assert!(!mincut.is_connected());
        assert_eq!(mincut.min_cut_value(), 0.0);
    }

    #[test]
    fn test_weighted_edges() {
        let edges = vec![(1, 2, 2.0), (2, 3, 3.0), (3, 1, 1.0)];

        let mincut = MinCutBuilder::new().with_edges(edges).build().unwrap();

        // Minimum cut should be 2.0 (cutting {1} from {2,3} or similar)
        assert_eq!(mincut.min_cut_value(), 3.0);
    }

    #[test]
    fn test_partition() {
        let edges = vec![(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)];

        let mincut = MinCutBuilder::new().with_edges(edges).build().unwrap();

        let (s, t) = mincut.partition();
        assert!(!s.is_empty());
        assert!(!t.is_empty());
        assert_eq!(s.len() + t.len(), 4);
    }

    #[test]
    fn test_cut_edges() {
        let edges = vec![(1, 2, 1.0), (2, 3, 1.0)];

        let mincut = MinCutBuilder::new().with_edges(edges).build().unwrap();

        let cut = mincut.cut_edges();
        assert!(!cut.is_empty());
        assert!(cut.len() <= 2);
    }

    #[test]
    fn test_min_cut_result() {
        let edges = vec![(1, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0)];

        let mincut = MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build()
            .unwrap();

        let result = mincut.min_cut();
        assert!(result.is_exact);
        assert_eq!(result.approximation_ratio, 1.0);
        assert!(result.cut_edges.is_some());
        assert!(result.partition.is_some());
    }

    #[test]
    fn test_approximate_mode() {
        let mincut = MinCutBuilder::new().approximate(0.1).build().unwrap();

        let result = mincut.min_cut();
        assert!(!result.is_exact);
        assert_eq!(result.approximation_ratio, 1.1);
    }

    #[test]
    fn test_statistics() {
        let mut mincut = MinCutBuilder::new()
            .with_edges(vec![(1, 2, 1.0)])
            .build()
            .unwrap();

        mincut.insert_edge(2, 3, 1.0).unwrap();
        mincut.delete_edge(1, 2).unwrap();
        let _ = mincut.min_cut_value();

        let stats = mincut.stats();
        assert_eq!(stats.insertions, 1);
        assert_eq!(stats.deletions, 1);
        assert_eq!(stats.queries, 1);
        assert!(stats.avg_update_time_us > 0.0);
    }

    #[test]
    fn test_reset_stats() {
        let mut mincut = MinCutBuilder::new()
            .with_edges(vec![(1, 2, 1.0)])
            .build()
            .unwrap();

        mincut.insert_edge(2, 3, 1.0).unwrap();
        assert_eq!(mincut.stats().insertions, 1);

        mincut.reset_stats();
        assert_eq!(mincut.stats().insertions, 0);
    }

    #[test]
    fn test_builder_pattern() {
        let mincut = MinCutBuilder::new()
            .exact()
            .max_cut_size(500)
            .parallel(true)
            .with_edges(vec![(1, 2, 1.0)])
            .build()
            .unwrap();

        assert!(!mincut.config().approximate);
        assert_eq!(mincut.config().max_exact_cut_size, 500);
        assert!(mincut.config().parallel);
    }

    #[test]
    fn test_large_graph() {
        let mut edges = Vec::new();

        // Create a chain: 0 - 1 - 2 - ... - 99
        for i in 0..99 {
            edges.push((i, i + 1, 1.0));
        }

        let mincut = MinCutBuilder::new().with_edges(edges).build().unwrap();

        assert_eq!(mincut.num_vertices(), 100);
        assert_eq!(mincut.num_edges(), 99);
        assert_eq!(mincut.min_cut_value(), 1.0); // Minimum cut is 1
        assert!(mincut.is_connected());
    }

    #[test]
    fn test_tree_edge_deletion_with_replacement() {
        let mut mincut = MinCutBuilder::new()
            .with_edges(vec![
                (1, 2, 1.0),
                (2, 3, 1.0),
                (1, 3, 1.0), // Creates a cycle
            ])
            .build()
            .unwrap();

        assert!(mincut.is_connected());

        // Delete one edge - graph should remain connected due to replacement
        mincut.delete_edge(1, 2).unwrap();

        // Still has 2 edges
        assert_eq!(mincut.num_edges(), 2);
    }

    #[test]
    fn test_multiple_components() {
        let edges = vec![(1, 2, 1.0), (3, 4, 1.0), (5, 6, 1.0)];

        let mincut = MinCutBuilder::new().with_edges(edges).build().unwrap();

        assert!(!mincut.is_connected());
        assert_eq!(mincut.min_cut_value(), 0.0);
    }

    #[test]
    fn test_dynamic_updates() {
        let mut mincut = MinCutBuilder::new().build().unwrap();

        // Start empty
        assert_eq!(mincut.min_cut_value(), f64::INFINITY);

        // Add first edge (creates two vertices)
        mincut.insert_edge(1, 2, 2.0).unwrap();
        assert_eq!(mincut.min_cut_value(), 2.0);

        // Complete the triangle
        mincut.insert_edge(2, 3, 3.0).unwrap();
        mincut.insert_edge(3, 1, 1.0).unwrap();
        assert_eq!(mincut.min_cut_value(), 3.0); // min cut

        // Delete heaviest edge
        mincut.delete_edge(2, 3).unwrap();
        assert_eq!(mincut.min_cut_value(), 1.0); // Now path graph
    }

    #[test]
    fn test_config_access() {
        let mincut = MinCutBuilder::new()
            .approximate(0.2)
            .max_cut_size(2000)
            .build()
            .unwrap();

        let config = mincut.config();
        assert_eq!(config.epsilon, 0.2);
        assert_eq!(config.max_exact_cut_size, 2000);
        assert!(config.approximate);
    }

    #[test]
    fn test_graph_access() {
        let mincut = MinCutBuilder::new()
            .with_edges(vec![(1, 2, 1.0)])
            .build()
            .unwrap();

        let graph = mincut.graph();
        let g = graph.read();
        assert_eq!(g.num_vertices(), 2);
        assert_eq!(g.num_edges(), 1);
    }

    #[test]
    fn test_bridge_graph() {
        // Two triangles connected by a bridge
        let edges = vec![
            // Triangle 1: 1-2-3-1
            (1, 2, 2.0),
            (2, 3, 2.0),
            (3, 1, 2.0),
            // Bridge
            (3, 4, 1.0),
            // Triangle 2: 4-5-6-4
            (4, 5, 2.0),
            (5, 6, 2.0),
            (6, 4, 2.0),
        ];

        let mincut = MinCutBuilder::new().with_edges(edges).build().unwrap();

        // Minimum cut should be the bridge with weight 1.0
        assert_eq!(mincut.min_cut_value(), 1.0);
        assert!(mincut.is_connected());
    }

    #[test]
    fn test_complete_graph_k4() {
        // Complete graph on 4 vertices
        let mut edges = Vec::new();
        for i in 1..=4 {
            for j in (i + 1)..=4 {
                edges.push((i, j, 1.0));
            }
        }

        let mincut = MinCutBuilder::new().with_edges(edges).build().unwrap();

        assert_eq!(mincut.num_vertices(), 4);
        assert_eq!(mincut.num_edges(), 6);
        // Minimum cut of K4 is 3 (degree of any vertex)
        assert_eq!(mincut.min_cut_value(), 3.0);
    }

    #[test]
    fn test_sequential_insertions() {
        let mut mincut = MinCutBuilder::new().build().unwrap();

        // Build graph incrementally
        mincut.insert_edge(1, 2, 1.0).unwrap();
        assert_eq!(mincut.min_cut_value(), 1.0);

        mincut.insert_edge(2, 3, 1.0).unwrap();
        assert_eq!(mincut.min_cut_value(), 1.0);

        mincut.insert_edge(3, 4, 1.0).unwrap();
        assert_eq!(mincut.min_cut_value(), 1.0);

        // Add cycle closure
        mincut.insert_edge(4, 1, 1.0).unwrap();
        assert_eq!(mincut.min_cut_value(), 2.0);
    }

    #[test]
    fn test_sequential_deletions() {
        let mut mincut = MinCutBuilder::new()
            .with_edges(vec![(1, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0)])
            .build()
            .unwrap();

        assert_eq!(mincut.min_cut_value(), 2.0);

        // Delete one edge
        mincut.delete_edge(1, 2).unwrap();
        assert_eq!(mincut.min_cut_value(), 1.0);

        // Delete another edge - disconnects graph
        mincut.delete_edge(2, 3).unwrap();
        assert_eq!(mincut.min_cut_value(), 0.0);
    }
}
