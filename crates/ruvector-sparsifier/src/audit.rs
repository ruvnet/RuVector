//! Spectral audit system for verifying sparsifier quality.
//!
//! Compares the Laplacian quadratic form `x^T L x` of the full graph against
//! the sparsifier on random probe vectors. If the relative error exceeds
//! `epsilon`, the audit fails and a rebuild is recommended.

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::graph::SparseGraph;
use crate::types::AuditResult;

// ---------------------------------------------------------------------------
// SpectralAuditor
// ---------------------------------------------------------------------------

/// Generates random probe vectors and compares Laplacian quadratic forms
/// between the full graph and its sparsifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAuditor {
    /// Number of random probes per audit.
    pub n_probes: usize,
    /// Acceptable relative error threshold.
    pub threshold: f64,
}

impl SpectralAuditor {
    /// Create a new auditor.
    pub fn new(n_probes: usize, threshold: f64) -> Self {
        Self {
            n_probes,
            threshold,
        }
    }

    /// Audit by comparing `x^T L_full x` vs `x^T L_spec x` for random `x`.
    ///
    /// Each probe vector has i.i.d. standard-normal entries. The relative
    /// error for each probe is `|full - spec| / max(full, tiny)`.
    pub fn audit_quadratic_form(
        &self,
        g_full: &SparseGraph,
        g_spec: &SparseGraph,
        n_probes: usize,
    ) -> AuditResult {
        let n = g_full.num_vertices();
        if n == 0 {
            return AuditResult::trivial_pass(self.threshold);
        }

        let n_spec = g_spec.num_vertices();
        let dim = n.max(n_spec);

        let mut rng = rand::thread_rng();
        let mut max_error = 0.0f64;
        let mut sum_error = 0.0f64;
        let probes = if n_probes > 0 { n_probes } else { self.n_probes };

        for _ in 0..probes {
            // Generate random probe vector.
            let x: Vec<f64> = (0..dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();

            let val_full = g_full.laplacian_quadratic_form(&x[..n.max(1)]);
            let val_spec = if n_spec > 0 {
                g_spec.laplacian_quadratic_form(&x[..n_spec.max(1)])
            } else {
                0.0
            };

            let denom = val_full.abs().max(1e-15);
            let rel_error = (val_full - val_spec).abs() / denom;

            max_error = max_error.max(rel_error);
            sum_error += rel_error;
        }

        let avg_error = if probes > 0 {
            sum_error / probes as f64
        } else {
            0.0
        };

        AuditResult {
            max_error,
            avg_error,
            passed: max_error <= self.threshold,
            n_probes: probes,
            threshold: self.threshold,
        }
    }

    /// Audit by comparing random cut values.
    ///
    /// Generates `n_cuts` random binary partitions and checks that the
    /// cut weight in the sparsifier is within `(1 +/- threshold)` of the
    /// full graph cut weight.
    pub fn audit_cuts(
        &self,
        g_full: &SparseGraph,
        g_spec: &SparseGraph,
        n_cuts: usize,
    ) -> AuditResult {
        let n = g_full.num_vertices();
        if n == 0 {
            return AuditResult::trivial_pass(self.threshold);
        }

        let mut rng = rand::thread_rng();
        let mut max_error = 0.0f64;
        let mut sum_error = 0.0f64;

        for _ in 0..n_cuts {
            // Random partition encoded as {-1, +1} indicator.
            let x: Vec<f64> = (0..n)
                .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
                .collect();

            // The quadratic form with {-1,+1} indicator gives 4 * cut_weight.
            let cut_full = g_full.laplacian_quadratic_form(&x);
            let cut_spec = if g_spec.num_vertices() >= n {
                g_spec.laplacian_quadratic_form(&x)
            } else {
                // Pad the spec graph query.
                let mut x_padded = x.clone();
                x_padded.resize(g_spec.num_vertices().max(n), 0.0);
                g_spec.laplacian_quadratic_form(&x_padded)
            };

            let denom = cut_full.abs().max(1e-15);
            let rel_error = (cut_full - cut_spec).abs() / denom;
            max_error = max_error.max(rel_error);
            sum_error += rel_error;
        }

        let avg_error = if n_cuts > 0 {
            sum_error / n_cuts as f64
        } else {
            0.0
        };

        AuditResult {
            max_error,
            avg_error,
            passed: max_error <= self.threshold,
            n_probes: n_cuts,
            threshold: self.threshold,
        }
    }

    /// Audit via random cluster conductance estimation.
    ///
    /// Generates `k_clusters` random cluster indicator vectors and checks
    /// spectral consistency.
    pub fn audit_conductance(
        &self,
        g_full: &SparseGraph,
        g_spec: &SparseGraph,
        k_clusters: usize,
    ) -> AuditResult {
        let n = g_full.num_vertices();
        if n == 0 {
            return AuditResult::trivial_pass(self.threshold);
        }

        let mut rng = rand::thread_rng();
        let mut max_error = 0.0f64;
        let mut sum_error = 0.0f64;

        for _ in 0..k_clusters {
            // Assign each vertex to one of k_clusters clusters.
            let cluster_id: Vec<usize> = (0..n).map(|_| rng.gen_range(0..k_clusters.max(2))).collect();

            // For each cluster, create indicator and measure quadratic form.
            for c in 0..k_clusters.max(2) {
                let x: Vec<f64> = cluster_id
                    .iter()
                    .map(|&cid| if cid == c { 1.0 } else { 0.0 })
                    .collect();

                let val_full = g_full.laplacian_quadratic_form(&x);
                let val_spec = if g_spec.num_vertices() >= n {
                    g_spec.laplacian_quadratic_form(&x)
                } else {
                    let mut x_padded = x.clone();
                    x_padded.resize(g_spec.num_vertices().max(n), 0.0);
                    g_spec.laplacian_quadratic_form(&x_padded)
                };

                let denom = val_full.abs().max(1e-15);
                let rel_error = (val_full - val_spec).abs() / denom;
                max_error = max_error.max(rel_error);
                sum_error += rel_error;
            }
        }

        let total_probes = k_clusters * k_clusters.max(2);
        let avg_error = if total_probes > 0 {
            sum_error / total_probes as f64
        } else {
            0.0
        };

        AuditResult {
            max_error,
            avg_error,
            passed: max_error <= self.threshold,
            n_probes: total_probes,
            threshold: self.threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_identical_graphs() {
        let g = SparseGraph::from_edges(&[
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
        ]);
        let auditor = SpectralAuditor::new(20, 0.01);
        let result = auditor.audit_quadratic_form(&g, &g, 20);
        assert!(result.passed);
        assert!(result.max_error < 1e-10);
    }

    #[test]
    fn test_audit_empty_graph() {
        let g = SparseGraph::new();
        let auditor = SpectralAuditor::new(10, 0.2);
        let result = auditor.audit_quadratic_form(&g, &g, 10);
        assert!(result.passed);
    }

    #[test]
    fn test_audit_cuts_identical() {
        let g = SparseGraph::from_edges(&[
            (0, 1, 2.0),
            (1, 2, 3.0),
        ]);
        let auditor = SpectralAuditor::new(10, 0.01);
        let result = auditor.audit_cuts(&g, &g, 10);
        assert!(result.passed);
    }

    #[test]
    fn test_audit_conductance_identical() {
        let g = SparseGraph::from_edges(&[
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
        ]);
        let auditor = SpectralAuditor::new(10, 0.01);
        let result = auditor.audit_conductance(&g, &g, 3);
        assert!(result.passed);
    }
}
