//! Trait definitions for the sparsifier framework.
//!
//! These traits allow pluggable strategies for backbone maintenance,
//! importance scoring, and the sparsifier itself.

use crate::error::Result;
use crate::graph::SparseGraph;
use crate::types::{AuditResult, EdgeImportance, SparsifierStats};

// ---------------------------------------------------------------------------
// Sparsifier trait
// ---------------------------------------------------------------------------

/// A dynamic spectral sparsifier that maintains a compressed shadow graph
/// preserving the Laplacian energy of the original within `(1 +/- epsilon)`.
pub trait Sparsifier: Send + Sync {
    /// Insert an edge into the full graph and update the sparsifier.
    fn insert_edge(&mut self, u: usize, v: usize, weight: f64) -> Result<()>;

    /// Delete an edge from the full graph and update the sparsifier.
    fn delete_edge(&mut self, u: usize, v: usize) -> Result<()>;

    /// Run a spectral audit comparing the sparsifier against the full graph.
    fn audit(&self) -> AuditResult;

    /// Rebuild the sparsifier around specific vertices.
    fn rebuild_local(&mut self, nodes: &[usize]) -> Result<()>;

    /// Fully reconstruct the sparsifier from scratch.
    fn rebuild_full(&mut self) -> Result<()>;

    /// Return a reference to the current sparsifier graph.
    fn sparsifier(&self) -> &SparseGraph;

    /// Return the current compression ratio.
    fn compression_ratio(&self) -> f64;

    /// Return the current statistics.
    fn stats(&self) -> &SparsifierStats;
}

// ---------------------------------------------------------------------------
// Importance scorer trait
// ---------------------------------------------------------------------------

/// Strategy for scoring the importance of edges.
///
/// Higher importance means the edge is more critical for preserving
/// spectral properties and should be sampled with higher probability.
pub trait ImportanceScorer: Send + Sync {
    /// Score a single edge.
    fn score(&self, graph: &SparseGraph, u: usize, v: usize, weight: f64) -> EdgeImportance;

    /// Score all edges in the graph, returning a vector of importance scores.
    fn score_all(&self, graph: &SparseGraph) -> Vec<EdgeImportance>;
}

// ---------------------------------------------------------------------------
// Backbone strategy trait
// ---------------------------------------------------------------------------

/// Strategy for maintaining a backbone (spanning forest) that ensures
/// global connectivity in the sparsifier.
pub trait BackboneStrategy: Send + Sync {
    /// Insert an edge, potentially adding it to the backbone.
    ///
    /// Returns `true` if the edge was added to the backbone forest.
    fn insert_edge(&mut self, u: usize, v: usize, weight: f64) -> bool;

    /// Delete an edge, repairing the backbone if needed.
    ///
    /// Returns `true` if the backbone was modified.
    fn delete_edge(&mut self, u: usize, v: usize, weight: f64) -> bool;

    /// Check whether an edge is currently in the backbone.
    fn is_backbone_edge(&self, u: usize, v: usize) -> bool;

    /// Return the number of connected components.
    fn num_components(&self) -> usize;

    /// Check whether two vertices are in the same component.
    fn connected(&mut self, u: usize, v: usize) -> bool;

    /// Return the number of edges in the backbone.
    fn backbone_edge_count(&self) -> usize;

    /// Ensure the backbone can accommodate at least `n` vertices.
    fn ensure_capacity(&mut self, n: usize);
}
