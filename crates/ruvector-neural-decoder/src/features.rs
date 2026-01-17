//! Structural Features Module
//!
//! Extracts structural features from detector graphs using min-cut algorithms.
//! These features enhance the neural decoder with graph-theoretic information
//! about error patterns and syndrome connectivity.
//!
//! ## Features
//!
//! - **Min-Cut Value**: Global minimum cut as a measure of graph fragility
//! - **Local Cuts**: Per-node cut values for localized structure
//! - **Conductance**: Graph expansion properties
//! - **Edge Weights**: Error probability-derived features
//!
//! ## Integration with ruvector-mincut
//!
//! This module leverages the O(n^{o(1)}) dynamic min-cut algorithm from
//! `ruvector-mincut` for efficient structural analysis.

use ruvector_mincut::{
    DynamicMinCut, MinCutBuilder, MinCutResult, Weight,
};
use serde::{Deserialize, Serialize};

use crate::error::{NqedError, Result};
use crate::graph::{DetectorGraph, DetectorId};

/// Configuration for structural feature extraction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Whether to compute global min-cut.
    pub compute_global_cut: bool,

    /// Whether to compute local cuts per node.
    pub compute_local_cuts: bool,

    /// Whether to compute conductance.
    pub compute_conductance: bool,

    /// Whether to normalize features.
    pub normalize: bool,

    /// Epsilon for approximate min-cut (None for exact).
    pub approximation_epsilon: Option<f64>,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            compute_global_cut: true,
            compute_local_cuts: true,
            compute_conductance: true,
            normalize: true,
            approximation_epsilon: None,
        }
    }
}

/// Structural features extracted from a detector graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuralFeatures {
    /// Global minimum cut value.
    pub global_min_cut: f64,

    /// Partition from the min-cut (node indices in each side).
    pub partition: Option<(Vec<usize>, Vec<usize>)>,

    /// Edges in the minimum cut.
    pub cut_edges: Option<Vec<(usize, usize)>>,

    /// Local cut values per node.
    pub local_cuts: Vec<f64>,

    /// Graph conductance (expansion property).
    pub conductance: f64,

    /// Average weighted degree.
    pub avg_weighted_degree: f64,

    /// Spectral gap estimate.
    pub spectral_gap: f64,

    /// Node centrality scores.
    pub centrality: Vec<f64>,

    /// Cluster assignment for each node.
    pub cluster_assignment: Vec<usize>,

    /// Feature vector for each node (aggregated features).
    pub node_features: Vec<Vec<f32>>,
}

impl Default for StructuralFeatures {
    fn default() -> Self {
        Self {
            global_min_cut: f64::INFINITY,
            partition: None,
            cut_edges: None,
            local_cuts: Vec::new(),
            conductance: 0.0,
            avg_weighted_degree: 0.0,
            spectral_gap: 0.0,
            centrality: Vec::new(),
            cluster_assignment: Vec::new(),
            node_features: Vec::new(),
        }
    }
}

impl StructuralFeatures {
    /// Returns the feature dimension per node.
    #[inline]
    #[must_use]
    pub fn feature_dim(&self) -> usize {
        if self.node_features.is_empty() {
            0
        } else {
            self.node_features[0].len()
        }
    }

    /// Returns true if the graph is likely disconnected.
    #[inline]
    #[must_use]
    pub fn is_disconnected(&self) -> bool {
        self.global_min_cut == 0.0
    }

    /// Returns the number of nodes.
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.local_cuts.len()
    }
}

/// Extractor for structural features from detector graphs.
///
/// Uses `ruvector-mincut` for efficient O(n^{o(1)}) structural analysis.
///
/// ## Example
///
/// ```rust,no_run
/// use ruvector_neural_decoder::features::{StructuralFeatureExtractor, FeatureConfig};
/// use ruvector_neural_decoder::graph::{DetectorGraph, DetectorGraphConfig};
///
/// let config = FeatureConfig::default();
/// let extractor = StructuralFeatureExtractor::new(config);
///
/// let graph_config = DetectorGraphConfig::default();
/// let mut graph = DetectorGraph::new(graph_config);
/// graph.add_detector(0, 0.0, 0.0, 0, true);
/// graph.add_detector(1, 1.0, 0.0, 0, true);
/// graph.build_edges().unwrap();
///
/// let features = extractor.extract(&graph).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct StructuralFeatureExtractor {
    /// Extraction configuration.
    config: FeatureConfig,
}

impl StructuralFeatureExtractor {
    /// Creates a new feature extractor.
    #[must_use]
    pub fn new(config: FeatureConfig) -> Self {
        Self { config }
    }

    /// Extracts structural features from a detector graph.
    pub fn extract(&self, graph: &DetectorGraph) -> Result<StructuralFeatures> {
        let node_count = graph.node_count();
        if node_count == 0 {
            return Ok(StructuralFeatures::default());
        }

        // Convert detector graph to min-cut format
        let edges = self.graph_to_edges(graph)?;

        // Build min-cut structure
        let mut mincut = if let Some(eps) = self.config.approximation_epsilon {
            MinCutBuilder::new()
                .approximate(eps)
                .with_edges(edges.clone())
                .build()
                .map_err(|e| NqedError::MinCutError(e.to_string()))?
        } else {
            MinCutBuilder::new()
                .exact()
                .with_edges(edges.clone())
                .build()
                .map_err(|e| NqedError::MinCutError(e.to_string()))?
        };

        // Compute global min-cut
        let global_result = if self.config.compute_global_cut {
            mincut.min_cut()
        } else {
            MinCutResult {
                value: f64::INFINITY,
                cut_edges: None,
                partition: None,
                is_exact: true,
                approximation_ratio: 1.0,
            }
        };

        // Compute local cuts
        let local_cuts = if self.config.compute_local_cuts {
            self.compute_local_cuts(graph, &edges)?
        } else {
            vec![0.0; node_count]
        };

        // Compute conductance
        let conductance = if self.config.compute_conductance {
            self.compute_conductance(&edges, node_count)
        } else {
            0.0
        };

        // Compute average weighted degree
        let avg_weighted_degree = self.compute_avg_weighted_degree(&edges, node_count);

        // Compute centrality
        let centrality = self.compute_centrality(graph, &edges);

        // Spectral gap estimate
        let spectral_gap = self.estimate_spectral_gap(conductance);

        // Cluster assignment (simple connected components approximation)
        let cluster_assignment = self.compute_clusters(&global_result, node_count);

        // Build per-node feature vectors
        let node_features = self.build_node_features(
            graph,
            &local_cuts,
            &centrality,
            &cluster_assignment,
            conductance,
        )?;

        let mut features = StructuralFeatures {
            global_min_cut: global_result.value,
            partition: global_result.partition.map(|(s, t)| {
                (s.into_iter().map(|v| v as usize).collect(),
                 t.into_iter().map(|v| v as usize).collect())
            }),
            cut_edges: global_result.cut_edges.map(|edges| {
                edges.into_iter().map(|(u, v)| (u as usize, v as usize)).collect()
            }),
            local_cuts,
            conductance,
            avg_weighted_degree,
            spectral_gap,
            centrality,
            cluster_assignment,
            node_features,
        };

        // Normalize if requested
        if self.config.normalize {
            self.normalize_features(&mut features);
        }

        Ok(features)
    }

    /// Converts detector graph to edge list format.
    fn graph_to_edges(&self, graph: &DetectorGraph) -> Result<Vec<(u64, u64, Weight)>> {
        let mut edges = Vec::new();
        let node_indices: Vec<_> = graph.node_indices().collect();

        // Create index mapping
        let idx_map: std::collections::HashMap<_, _> = node_indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i as u64))
            .collect();

        for &idx in &node_indices {
            for (neighbor_idx, edge) in graph.neighbors(idx) {
                let from = idx_map[&idx];
                let to = idx_map[&neighbor_idx];

                // Use error probability as weight (higher prob = lower weight)
                // This makes min-cut find the most likely error patterns
                let weight = 1.0 / (edge.error_probability + 1e-10);

                // Only add edge once (avoid duplicates)
                if from < to {
                    edges.push((from, to, weight));
                }
            }
        }

        Ok(edges)
    }

    /// Computes local cut values for each node.
    fn compute_local_cuts(
        &self,
        graph: &DetectorGraph,
        edges: &[(u64, u64, Weight)],
    ) -> Result<Vec<f64>> {
        let node_count = graph.node_count();
        let mut local_cuts = vec![0.0; node_count];

        // For each node, compute the sum of weights of incident edges
        // This is a proxy for the node's local cut value
        for &(u, v, w) in edges {
            local_cuts[u as usize] += w;
            local_cuts[v as usize] += w;
        }

        Ok(local_cuts)
    }

    /// Computes graph conductance.
    fn compute_conductance(&self, edges: &[(u64, u64, Weight)], node_count: usize) -> f64 {
        if edges.is_empty() || node_count < 2 {
            return 0.0;
        }

        // Conductance = min_S |E(S, S')| / min(vol(S), vol(S'))
        // Approximate using simple degree-based estimation
        let total_weight: f64 = edges.iter().map(|(_, _, w)| w).sum();
        let avg_degree = 2.0 * total_weight / node_count as f64;

        // Simple approximation: conductance ~ 1 / avg_degree
        if avg_degree > 0.0 {
            (1.0 / avg_degree).min(1.0)
        } else {
            0.0
        }
    }

    /// Computes average weighted degree.
    fn compute_avg_weighted_degree(&self, edges: &[(u64, u64, Weight)], node_count: usize) -> f64 {
        if node_count == 0 {
            return 0.0;
        }

        let total_weight: f64 = edges.iter().map(|(_, _, w)| w).sum();
        2.0 * total_weight / node_count as f64
    }

    /// Computes node centrality (degree centrality).
    fn compute_centrality(
        &self,
        graph: &DetectorGraph,
        edges: &[(u64, u64, Weight)],
    ) -> Vec<f64> {
        let node_count = graph.node_count();
        if node_count == 0 {
            return Vec::new();
        }

        let mut degrees = vec![0.0; node_count];

        for &(u, v, w) in edges {
            degrees[u as usize] += w;
            degrees[v as usize] += w;
        }

        // Normalize by maximum possible degree
        let max_degree = degrees.iter().cloned().fold(0.0, f64::max);
        if max_degree > 0.0 {
            degrees.iter().map(|&d| d / max_degree).collect()
        } else {
            degrees
        }
    }

    /// Estimates spectral gap from conductance.
    fn estimate_spectral_gap(&self, conductance: f64) -> f64 {
        // Cheeger inequality: h^2 / 2 <= lambda_2 <= 2h
        // where h is conductance and lambda_2 is spectral gap
        conductance.powi(2) / 2.0
    }

    /// Computes cluster assignment based on min-cut partition.
    fn compute_clusters(&self, mincut_result: &MinCutResult, node_count: usize) -> Vec<usize> {
        if let Some((ref s, ref t)) = mincut_result.partition {
            let mut assignment = vec![0usize; node_count];
            for &v in t {
                if (v as usize) < node_count {
                    assignment[v as usize] = 1;
                }
            }
            assignment
        } else {
            // All nodes in same cluster
            vec![0; node_count]
        }
    }

    /// Builds per-node feature vectors.
    fn build_node_features(
        &self,
        graph: &DetectorGraph,
        local_cuts: &[f64],
        centrality: &[f64],
        cluster_assignment: &[usize],
        conductance: f64,
    ) -> Result<Vec<Vec<f32>>> {
        let node_indices: Vec<_> = graph.node_indices().collect();
        let mut features = Vec::with_capacity(node_indices.len());

        for (i, &idx) in node_indices.iter().enumerate() {
            let node = graph.node(idx).ok_or_else(|| {
                NqedError::FeatureError("node not found".to_string())
            })?;

            // Build feature vector
            let mut f = Vec::with_capacity(8);

            // Position features
            f.push(node.x);
            f.push(node.y);

            // Firing state
            f.push(if node.fired { 1.0 } else { 0.0 });

            // Structural features
            f.push(local_cuts.get(i).copied().unwrap_or(0.0) as f32);
            f.push(centrality.get(i).copied().unwrap_or(0.0) as f32);
            f.push(cluster_assignment.get(i).copied().unwrap_or(0) as f32);
            f.push(conductance as f32);

            // Neighbor count
            f.push(graph.neighbors(idx).len() as f32);

            features.push(f);
        }

        Ok(features)
    }

    /// Normalizes features to [0, 1] range.
    fn normalize_features(&self, features: &mut StructuralFeatures) {
        // Normalize local cuts
        let max_local = features.local_cuts.iter().cloned().fold(0.0, f64::max);
        if max_local > 0.0 {
            features.local_cuts.iter_mut().for_each(|c| *c /= max_local);
        }

        // Node features are already position-normalized; normalize structural features
        for f in &mut features.node_features {
            // Normalize local cut feature (index 3)
            if f.len() > 3 && max_local > 0.0 {
                f[3] /= max_local as f32;
            }
        }
    }

    /// Returns the configuration.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &FeatureConfig {
        &self.config
    }
}

/// Structural feature summary for quick analysis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureSummary {
    /// Global min-cut value.
    pub min_cut: f64,

    /// Graph conductance.
    pub conductance: f64,

    /// Average centrality.
    pub avg_centrality: f64,

    /// Number of clusters.
    pub num_clusters: usize,

    /// Whether graph is connected.
    pub connected: bool,
}

impl StructuralFeatures {
    /// Creates a summary of the structural features.
    #[must_use]
    pub fn summary(&self) -> FeatureSummary {
        let avg_centrality = if self.centrality.is_empty() {
            0.0
        } else {
            self.centrality.iter().sum::<f64>() / self.centrality.len() as f64
        };

        let num_clusters = if self.cluster_assignment.is_empty() {
            0
        } else {
            *self.cluster_assignment.iter().max().unwrap_or(&0) + 1
        };

        FeatureSummary {
            min_cut: self.global_min_cut,
            conductance: self.conductance,
            avg_centrality,
            num_clusters,
            connected: !self.is_disconnected(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DetectorGraphConfig;

    #[test]
    fn test_feature_config_default() {
        let config = FeatureConfig::default();
        assert!(config.compute_global_cut);
        assert!(config.normalize);
    }

    #[test]
    fn test_empty_graph() {
        let config = FeatureConfig::default();
        let extractor = StructuralFeatureExtractor::new(config);

        let graph_config = DetectorGraphConfig::default();
        let graph = DetectorGraph::new(graph_config);

        let features = extractor.extract(&graph).unwrap();
        assert_eq!(features.node_count(), 0);
    }

    #[test]
    fn test_simple_graph() {
        let config = FeatureConfig::default();
        let extractor = StructuralFeatureExtractor::new(config);

        let graph_config = DetectorGraphConfig::builder()
            .code_distance(3)
            .error_rate(0.01)
            .build()
            .unwrap();

        let mut graph = DetectorGraph::new(graph_config);
        graph.add_detector(0, 0.0, 0.0, 0, true);
        graph.add_detector(1, 1.0, 0.0, 0, true);
        graph.add_detector(2, 0.5, 1.0, 0, true);
        graph.build_edges().unwrap();

        let features = extractor.extract(&graph).unwrap();

        assert_eq!(features.node_count(), 3);
        assert!(!features.is_disconnected());
        assert_eq!(features.node_features.len(), 3);
    }

    #[test]
    fn test_feature_summary() {
        let features = StructuralFeatures {
            global_min_cut: 2.0,
            conductance: 0.5,
            centrality: vec![0.3, 0.5, 0.7],
            cluster_assignment: vec![0, 0, 1],
            ..Default::default()
        };

        let summary = features.summary();
        assert_eq!(summary.min_cut, 2.0);
        assert_eq!(summary.num_clusters, 2);
        assert!(summary.connected);
    }

    #[test]
    fn test_disconnected_detection() {
        let features = StructuralFeatures {
            global_min_cut: 0.0,
            ..Default::default()
        };

        assert!(features.is_disconnected());
    }
}
