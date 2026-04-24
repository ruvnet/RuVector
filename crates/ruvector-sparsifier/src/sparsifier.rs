//! Main adaptive spectral sparsifier (AdaptiveGeoSpar).
//!
//! Maintains a compressed shadow graph `g_spec` that preserves the Laplacian
//! energy of the full graph `g_full` within `(1 +/- epsilon)`. Supports
//! dynamic edge insertions, deletions, and periodic spectral audits.

use std::collections::HashSet;

use parking_lot::RwLock;
use tracing::{debug, info, warn};

use crate::audit::SpectralAuditor;
use crate::backbone::Backbone;
use crate::error::{Result, SparsifierError};
use crate::graph::SparseGraph;
use crate::importance::LocalImportanceScorer;
use crate::sampler::SpectralSampler;
use crate::traits::{BackboneStrategy, ImportanceScorer, Sparsifier};
use crate::types::{AuditResult, SparsifierConfig, SparsifierStats};

// ---------------------------------------------------------------------------
// AdaptiveGeoSpar
// ---------------------------------------------------------------------------

/// Dynamic spectral graph sparsifier implementing the ADKKP16 approach.
///
/// Maintains:
/// - `g_full`: the full weighted graph (receives all updates)
/// - `g_spec`: the compressed sparsifier (~O(n log n / eps^2) edges)
/// - `backbone`: spanning forest guaranteeing connectivity
/// - `scorer`: random-walk importance estimator
/// - `auditor`: periodic spectral quality check
///
/// # Thread safety
///
/// The sparsifier wraps its state in [`RwLock`] internally. The public API
/// takes `&mut self` to make ownership clear; concurrent readers can
/// access the sparsifier graph via [`sparsifier`](Self::sparsifier) which
/// clones the current snapshot.
pub struct AdaptiveGeoSpar {
    /// The full graph receiving all dynamic updates.
    g_full: SparseGraph,
    /// The compressed spectral sparsifier.
    g_spec: SparseGraph,
    /// Backbone spanning forest.
    backbone: Backbone,
    /// Edge importance scorer.
    scorer: LocalImportanceScorer,
    /// Spectral sampler.
    sampler: SpectralSampler,
    /// Spectral auditor.
    auditor: SpectralAuditor,
    /// Configuration.
    config: SparsifierConfig,
    /// Runtime statistics.
    stats: SparsifierStats,
    /// Set of backbone edge keys for the sampler.
    backbone_edge_set: HashSet<(usize, usize)>,
    /// Thread-safe snapshot for readers (updated after rebuilds).
    snapshot: RwLock<SparseGraph>,
    /// Cached total importance (avoids O(m) re-computation per insert).
    cached_total_importance: f64,
}

impl std::fmt::Debug for AdaptiveGeoSpar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveGeoSpar")
            .field("vertices", &self.g_full.num_vertices())
            .field("full_edges", &self.g_full.num_edges())
            .field("spec_edges", &self.g_spec.num_edges())
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

impl AdaptiveGeoSpar {
    // ----- construction ----------------------------------------------------

    /// Create a new empty sparsifier with the given configuration.
    pub fn new(config: SparsifierConfig) -> Self {
        let scorer = LocalImportanceScorer::new(config.walk_length, config.num_walks);
        let sampler = SpectralSampler::new(config.epsilon);
        let auditor = SpectralAuditor::new(config.n_audit_probes, config.epsilon);

        Self {
            g_full: SparseGraph::new(),
            g_spec: SparseGraph::new(),
            backbone: Backbone::new(0),
            scorer,
            sampler,
            auditor,
            config,
            stats: SparsifierStats::default(),
            backbone_edge_set: HashSet::new(),
            snapshot: RwLock::new(SparseGraph::new()),
            cached_total_importance: 0.0,
        }
    }

    /// Build a sparsifier from an existing static graph.
    ///
    /// This is the primary entry point for initial construction. It scores
    /// all edges, samples according to importance, and sets up the backbone.
    pub fn build(graph: &SparseGraph, config: SparsifierConfig) -> Result<Self> {
        if graph.num_vertices() == 0 {
            return Err(SparsifierError::EmptyGraph);
        }

        let mut spar = Self::new(config);
        spar.g_full = graph.clone();
        spar.backbone = Backbone::new(graph.num_vertices());

        // Insert all edges into backbone.
        for (u, v, w) in graph.edges() {
            let is_bb = spar.backbone.insert_edge(u, v, w);
            if is_bb {
                spar.backbone_edge_set.insert(Self::edge_key(u, v));
            }
        }

        // Score and sample.
        spar.do_full_rebuild()?;

        info!(
            vertices = graph.num_vertices(),
            full_edges = graph.num_edges(),
            spec_edges = spar.g_spec.num_edges(),
            compression = %format!("{:.2}x", spar.compression_ratio()),
            "Built initial sparsifier"
        );

        Ok(spar)
    }

    // ----- dynamic updates -------------------------------------------------

    /// Handle the insertion of an edge into the full graph.
    ///
    /// The edge is added to `g_full`, the backbone is updated, and the
    /// sparsifier is incrementally updated. Periodic audits may trigger
    /// a local or full rebuild.
    pub fn handle_insert(&mut self, u: usize, v: usize, weight: f64) -> Result<()> {
        // Validate.
        if !weight.is_finite() || weight <= 0.0 {
            return Err(SparsifierError::InvalidWeight(weight));
        }

        // Insert into full graph.
        self.g_full.insert_edge(u, v, weight)?;

        // Update backbone.
        let is_bb = self.backbone.insert_edge(u, v, weight);
        if is_bb {
            self.backbone_edge_set.insert(Self::edge_key(u, v));
            // Backbone edges always go into the sparsifier.
            let _ = self.g_spec.insert_or_update_edge(u, v, weight);
        } else {
            // Score and probabilistically add to sparsifier.
            let importance = self.scorer.score(&self.g_full, u, v, weight);
            let budget = self.edge_budget();
            // Incrementally update cached total importance instead of O(m) recompute.
            self.cached_total_importance += importance.score;
            let total_imp = self.cached_total_importance.max(importance.score);

            if let Some((su, sv, sw)) = self.sampler.sample_single_edge(
                &importance,
                self.g_full.num_vertices(),
                total_imp,
                budget,
            ) {
                let _ = self.g_spec.insert_or_update_edge(su, sv, sw);
            }
        }

        self.stats.insertions += 1;
        self.stats.updates_since_audit += 1;
        self.refresh_stats();
        self.maybe_audit();

        Ok(())
    }

    /// Handle the deletion of an edge from the full graph.
    pub fn handle_delete(&mut self, u: usize, v: usize) -> Result<()> {
        // Delete from full graph.
        let weight = self.g_full.delete_edge(u, v)?;

        // Delete from sparsifier if present.
        let _ = self.g_spec.delete_edge(u, v);

        // Update backbone.
        let key = Self::edge_key(u, v);
        if self.backbone_edge_set.remove(&key) {
            self.backbone.delete_edge(u, v, weight);
        }

        self.stats.deletions += 1;
        self.stats.updates_since_audit += 1;
        self.refresh_stats();
        self.maybe_audit();

        Ok(())
    }

    /// Handle a point-move operation: a node's neighbourhood changes.
    ///
    /// `old_neighbors` are edges to remove, `new_neighbors` are edges to add.
    pub fn update_embedding(
        &mut self,
        node: usize,
        old_neighbors: &[(usize, f64)],
        new_neighbors: &[(usize, f64)],
    ) -> Result<()> {
        // Remove old edges.
        for &(v, _) in old_neighbors {
            let _ = self.handle_delete(node, v);
        }
        // Add new edges.
        for &(v, w) in new_neighbors {
            let _ = self.handle_insert(node, v, w);
        }
        Ok(())
    }

    // ----- audit -----------------------------------------------------------

    /// Run a spectral audit comparing `g_spec` against `g_full`.
    pub fn run_audit(&self) -> AuditResult {
        self.auditor
            .audit_quadratic_form(&self.g_full, &self.g_spec, self.config.n_audit_probes)
    }

    /// Check if an audit is due, and if so, run it and optionally rebuild.
    fn maybe_audit(&mut self) {
        if self.stats.updates_since_audit < self.config.audit_interval as u64 {
            return;
        }

        let result = self.run_audit();
        self.stats.audit_count += 1;
        self.stats.updates_since_audit = 0;

        if result.passed {
            self.stats.audit_pass_count += 1;
            debug!(
                max_error = result.max_error,
                avg_error = result.avg_error,
                "Spectral audit passed"
            );
        } else {
            warn!(
                max_error = result.max_error,
                threshold = result.threshold,
                "Spectral audit failed"
            );
            if self.config.auto_rebuild_on_audit_failure {
                let _ = self.do_full_rebuild();
            }
        }
    }

    // ----- rebuild ---------------------------------------------------------

    /// Rebuild the sparsifier around specific vertices.
    ///
    /// Re-scores and re-samples edges incident to the given nodes.
    pub fn do_local_rebuild(&mut self, nodes: &[usize]) -> Result<()> {
        debug!(n_nodes = nodes.len(), "Local rebuild");

        // Collect edges incident to the target nodes.
        let node_set: HashSet<usize> = nodes.iter().copied().collect();
        let incident_edges: Vec<(usize, usize, f64)> = self
            .g_full
            .edges()
            .filter(|(u, v, _)| node_set.contains(u) || node_set.contains(v))
            .collect();

        // Remove these edges from the sparsifier.
        for &(u, v, _) in &incident_edges {
            let _ = self.g_spec.delete_edge(u, v);
        }

        // Re-score and re-sample.
        let scores: Vec<_> = incident_edges
            .iter()
            .map(|&(u, v, w)| self.scorer.score(&self.g_full, u, v, w))
            .collect();

        let budget = self.edge_budget();
        let sampled = self
            .sampler
            .sample_edges(&scores, budget, &self.backbone_edge_set);

        // Merge sampled edges back.
        for (u, v, w) in sampled.edges() {
            let _ = self.g_spec.insert_or_update_edge(u, v, w);
        }

        self.stats.local_rebuilds += 1;
        self.refresh_stats();
        self.update_snapshot();

        Ok(())
    }

    /// Full reconstruction of the sparsifier from scratch.
    fn do_full_rebuild(&mut self) -> Result<()> {
        debug!("Full sparsifier rebuild");

        let scores = self.scorer.score_all(&self.g_full);
        // Refresh cached total importance from fresh scores.
        self.cached_total_importance = scores.iter().map(|s| s.score).sum();
        let budget = self.edge_budget();
        self.g_spec = self
            .sampler
            .sample_edges(&scores, budget, &self.backbone_edge_set);

        self.stats.full_rebuilds += 1;
        self.refresh_stats();
        self.update_snapshot();

        Ok(())
    }

    // ----- accessors -------------------------------------------------------

    /// Get a reference to the full graph.
    pub fn full_graph(&self) -> &SparseGraph {
        &self.g_full
    }

    /// Get a reference to the current sparsifier graph.
    pub fn sparsifier_graph(&self) -> &SparseGraph {
        &self.g_spec
    }

    /// Get a clone of the thread-safe sparsifier snapshot.
    pub fn snapshot(&self) -> SparseGraph {
        self.snapshot.read().clone()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &SparsifierConfig {
        &self.config
    }

    // ----- helpers ---------------------------------------------------------

    /// Target edge budget for the sparsifier.
    fn edge_budget(&self) -> usize {
        self.config.edge_budget_factor * self.g_full.num_vertices().max(1)
    }

    /// Canonical edge key.
    fn edge_key(u: usize, v: usize) -> (usize, usize) {
        if u <= v {
            (u, v)
        } else {
            (v, u)
        }
    }

    /// Refresh derived statistics.
    fn refresh_stats(&mut self) {
        self.stats.vertex_count = self.g_full.num_vertices();
        self.stats.full_edge_count = self.g_full.num_edges();
        self.stats.edge_count = self.g_spec.num_edges();
        self.stats.refresh_ratio();
    }

    /// Push the current `g_spec` into the thread-safe snapshot.
    fn update_snapshot(&self) {
        let mut snap = self.snapshot.write();
        *snap = self.g_spec.clone();
    }
}

// ---------------------------------------------------------------------------
// Trait implementation
// ---------------------------------------------------------------------------

impl Sparsifier for AdaptiveGeoSpar {
    fn insert_edge(&mut self, u: usize, v: usize, weight: f64) -> Result<()> {
        self.handle_insert(u, v, weight)
    }

    fn delete_edge(&mut self, u: usize, v: usize) -> Result<()> {
        self.handle_delete(u, v)
    }

    fn audit(&self) -> AuditResult {
        self.run_audit()
    }

    fn rebuild_local(&mut self, nodes: &[usize]) -> Result<()> {
        self.do_local_rebuild(nodes)
    }

    fn rebuild_full(&mut self) -> Result<()> {
        self.do_full_rebuild()
    }

    fn sparsifier(&self) -> &SparseGraph {
        &self.g_spec
    }

    fn compression_ratio(&self) -> f64 {
        if self.g_spec.num_edges() == 0 {
            return 0.0;
        }
        self.g_full.num_edges() as f64 / self.g_spec.num_edges() as f64
    }

    fn stats(&self) -> &SparsifierStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_graph() -> SparseGraph {
        SparseGraph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
    }

    fn path_graph(n: usize) -> SparseGraph {
        let edges: Vec<_> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
        SparseGraph::from_edges(&edges)
    }

    #[test]
    fn test_build_triangle() {
        let g = triangle_graph();
        let config = SparsifierConfig::default();
        let spar = AdaptiveGeoSpar::build(&g, config).unwrap();

        assert_eq!(spar.full_graph().num_vertices(), 3);
        assert_eq!(spar.full_graph().num_edges(), 3);
        assert!(spar.sparsifier_graph().num_edges() > 0);
        assert!(spar.compression_ratio() > 0.0);
    }

    #[test]
    fn test_build_path() {
        let g = path_graph(10);
        let config = SparsifierConfig::default();
        let spar = AdaptiveGeoSpar::build(&g, config).unwrap();

        // A path has n-1 edges; the sparsifier should keep most/all of them
        // since they are all bridges.
        assert!(spar.sparsifier_graph().num_edges() >= 5);
    }

    #[test]
    fn test_dynamic_insert() {
        let g = triangle_graph();
        let config = SparsifierConfig::default();
        let mut spar = AdaptiveGeoSpar::build(&g, config).unwrap();

        spar.handle_insert(0, 3, 2.0).unwrap();
        assert_eq!(spar.full_graph().num_edges(), 4);
        assert_eq!(spar.stats().insertions, 1);
    }

    #[test]
    fn test_dynamic_delete() {
        let g = triangle_graph();
        let config = SparsifierConfig::default();
        let mut spar = AdaptiveGeoSpar::build(&g, config).unwrap();

        spar.handle_delete(0, 1).unwrap();
        assert_eq!(spar.full_graph().num_edges(), 2);
        assert_eq!(spar.stats().deletions, 1);
    }

    #[test]
    fn test_audit_passes_on_identical() {
        let g = triangle_graph();
        let mut config = SparsifierConfig::default();
        config.epsilon = 0.5; // generous threshold
        let spar = AdaptiveGeoSpar::build(&g, config).unwrap();

        let result = spar.run_audit();
        // For a tiny graph the sparsifier should be very close.
        assert!(result.avg_error < 1.0);
    }

    #[test]
    fn test_empty_graph_error() {
        let g = SparseGraph::new();
        let config = SparsifierConfig::default();
        let result = AdaptiveGeoSpar::build(&g, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_embedding() {
        let g = triangle_graph();
        let config = SparsifierConfig::default();
        let mut spar = AdaptiveGeoSpar::build(&g, config).unwrap();

        // Move vertex 0: remove edge to 1, add edge to new vertex 3.
        spar.update_embedding(0, &[(1, 1.0)], &[(3, 2.0)]).unwrap();

        assert!(!spar.full_graph().has_edge(0, 1));
        assert!(spar.full_graph().has_edge(0, 3));
    }

    #[test]
    fn test_rebuild_full() {
        let g = path_graph(5);
        let config = SparsifierConfig::default();
        let mut spar = AdaptiveGeoSpar::build(&g, config).unwrap();

        spar.rebuild_full().unwrap();
        assert_eq!(spar.stats().full_rebuilds, 2); // initial build + explicit
        assert!(spar.sparsifier_graph().num_edges() > 0);
    }

    #[test]
    fn test_stats_tracking() {
        let g = triangle_graph();
        let config = SparsifierConfig::default();
        let spar = AdaptiveGeoSpar::build(&g, config).unwrap();

        let stats = spar.stats();
        assert_eq!(stats.vertex_count, 3);
        assert_eq!(stats.full_edge_count, 3);
        assert!(stats.edge_count > 0);
    }
}
