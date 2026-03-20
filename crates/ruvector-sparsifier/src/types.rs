//! Core types for spectral graph sparsification.
//!
//! This module defines the configuration, scoring, audit, and statistics
//! types used throughout the crate. The main graph type [`SparseGraph`] lives
//! in the [`graph`](crate::graph) module.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the adaptive spectral sparsifier.
///
/// Controls approximation quality (`epsilon`), memory budget, audit frequency,
/// and random-walk parameters.
///
/// # Defaults
///
/// | Parameter          | Default   | Notes                            |
/// |--------------------|-----------|----------------------------------|
/// | `epsilon`          | 0.2       | (1 +/- eps) spectral guarantee   |
/// | `edge_budget_factor`| 8        | budget = factor * n              |
/// | `audit_interval`   | 1000      | updates between audits           |
/// | `walk_length`      | 6         | random-walk hops for importance  |
/// | `num_walks`        | 10        | walks per edge for scoring       |
/// | `n_audit_probes`   | 30        | random vectors per audit         |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsifierConfig {
    /// Spectral approximation factor in `(0, 1)`.
    ///
    /// Smaller values yield a more faithful sparsifier but require more
    /// edges. A value of 0.2 means the Laplacian quadratic form is
    /// preserved within a factor of `(1 +/- 0.2)`.
    pub epsilon: f64,

    /// Edge budget expressed as a multiple of `n` (number of vertices).
    ///
    /// The sparsifier will target at most `edge_budget_factor * n` edges.
    /// Typical values range from 4 (aggressive) to 12 (conservative).
    pub edge_budget_factor: usize,

    /// Number of dynamic updates between automatic spectral audits.
    pub audit_interval: usize,

    /// Maximum number of hops in importance random walks.
    pub walk_length: usize,

    /// Number of random walks per edge for importance estimation.
    pub num_walks: usize,

    /// Number of random probe vectors used in spectral audits.
    pub n_audit_probes: usize,

    /// Whether to automatically trigger a rebuild when an audit fails.
    pub auto_rebuild_on_audit_failure: bool,

    /// Fraction of the graph to rebuild during a local rebuild (0..1].
    pub local_rebuild_fraction: f64,
}

impl Default for SparsifierConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.2,
            edge_budget_factor: 8,
            audit_interval: 1000,
            walk_length: 6,
            num_walks: 10,
            n_audit_probes: 30,
            auto_rebuild_on_audit_failure: true,
            local_rebuild_fraction: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Edge importance
// ---------------------------------------------------------------------------

/// Importance score for a single edge combining effective resistance and
/// weight information.
///
/// The importance `score` is proportional to `weight * effective_resistance`
/// and determines the sampling probability during sparsification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EdgeImportance {
    /// Source vertex.
    pub u: usize,
    /// Target vertex.
    pub v: usize,
    /// Edge weight in the original graph.
    pub weight: f64,
    /// Estimated effective resistance between `u` and `v`.
    pub effective_resistance: f64,
    /// Combined importance score = `weight * effective_resistance`.
    pub score: f64,
}

impl EdgeImportance {
    /// Create a new importance score.
    pub fn new(u: usize, v: usize, weight: f64, effective_resistance: f64) -> Self {
        let score = weight * effective_resistance;
        Self {
            u,
            v,
            weight,
            effective_resistance,
            score,
        }
    }
}

// ---------------------------------------------------------------------------
// Audit result
// ---------------------------------------------------------------------------

/// Result of a spectral audit comparing the sparsifier against the full graph.
///
/// An audit samples random vectors `x` and compares `x^T L_full x` against
/// `x^T L_spec x`. The sparsifier passes when all relative errors stay
/// within `epsilon`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResult {
    /// Maximum relative error observed across all probe vectors.
    pub max_error: f64,
    /// Average relative error across all probe vectors.
    pub avg_error: f64,
    /// Whether the audit passed (max_error <= epsilon).
    pub passed: bool,
    /// Number of probe vectors used.
    pub n_probes: usize,
    /// The epsilon threshold used for pass/fail.
    pub threshold: f64,
}

impl AuditResult {
    /// Create a passing audit result with zero error.
    pub fn trivial_pass(threshold: f64) -> Self {
        Self {
            max_error: 0.0,
            avg_error: 0.0,
            passed: true,
            n_probes: 0,
            threshold,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for the sparsifier.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SparsifierStats {
    /// Number of edges in the current sparsifier.
    pub edge_count: usize,
    /// Number of edges in the full graph.
    pub full_edge_count: usize,
    /// Number of vertices.
    pub vertex_count: usize,
    /// Compression ratio: `full_edge_count / edge_count`.
    pub compression_ratio: f64,
    /// Total number of edge insertions processed.
    pub insertions: u64,
    /// Total number of edge deletions processed.
    pub deletions: u64,
    /// Total number of spectral audits performed.
    pub audit_count: u64,
    /// Number of audits that passed.
    pub audit_pass_count: u64,
    /// Number of local rebuilds triggered.
    pub local_rebuilds: u64,
    /// Number of full rebuilds triggered.
    pub full_rebuilds: u64,
    /// Updates since the last audit.
    pub updates_since_audit: u64,
}

impl SparsifierStats {
    /// Recompute the compression ratio from current edge counts.
    pub fn refresh_ratio(&mut self) {
        if self.edge_count > 0 {
            self.compression_ratio = self.full_edge_count as f64 / self.edge_count as f64;
        } else {
            self.compression_ratio = 0.0;
        }
    }
}
