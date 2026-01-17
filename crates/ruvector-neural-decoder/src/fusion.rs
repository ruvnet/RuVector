//! Feature Fusion for Neural Quantum Error Decoding
//!
//! This module fuses multiple sources of information for error prediction:
//! - GNN embeddings from the graph attention encoder
//! - Min-cut features from graph algorithms
//! - Boundary proximity weighting
//! - Coherence confidence scaling
//!
//! ## Fusion Strategy
//!
//! The fusion combines neural and algorithmic features:
//!
//! 1. **GNN Features**: Rich learned representations of syndrome patterns
//! 2. **Min-Cut Features**: Graph-theoretic error chain likelihood
//! 3. **Boundary Features**: Distance-based corrections for edge effects
//! 4. **Confidence Weighting**: Adaptive fusion based on prediction certainty

use crate::error::{NeuralDecoderError, Result};
use ndarray::{Array1, Array2, Axis};
use ruvector_mincut::{DynamicGraph, MinCutBuilder, Weight};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for feature fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// GNN embedding dimension
    pub gnn_dim: usize,
    /// MinCut feature dimension
    pub mincut_dim: usize,
    /// Output dimension after fusion
    pub output_dim: usize,
    /// Weight for GNN features (0-1)
    pub gnn_weight: f32,
    /// Weight for MinCut features (0-1)
    pub mincut_weight: f32,
    /// Weight for boundary features (0-1)
    pub boundary_weight: f32,
    /// Enable adaptive weighting based on confidence
    pub adaptive_weights: bool,
    /// Temperature for softmax confidence scaling
    pub temperature: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            gnn_dim: 64,
            mincut_dim: 16,
            output_dim: 32,
            gnn_weight: 0.5,
            mincut_weight: 0.3,
            boundary_weight: 0.2,
            adaptive_weights: true,
            temperature: 1.0,
        }
    }
}

impl FusionConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        let total_weight = self.gnn_weight + self.mincut_weight + self.boundary_weight;
        if (total_weight - 1.0).abs() > 1e-6 {
            return Err(NeuralDecoderError::ConfigError(format!(
                "Fusion weights must sum to 1.0, got {}",
                total_weight
            )));
        }
        if self.temperature <= 0.0 {
            return Err(NeuralDecoderError::ConfigError(
                "Temperature must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Min-cut features extracted from detector graph
#[derive(Debug, Clone)]
pub struct MinCutFeatures {
    /// Global minimum cut value
    pub global_mincut: f64,
    /// Local cut values for each node
    pub local_cuts: Vec<f64>,
    /// Edge participation in min-cut
    pub edge_in_cut: HashMap<(usize, usize), bool>,
    /// Cut-based error chain probability
    pub error_chain_prob: Vec<f64>,
}

impl MinCutFeatures {
    /// Extract min-cut features from a detector graph
    ///
    /// # Arguments
    /// * `adjacency` - Adjacency list of detector graph
    /// * `edge_weights` - Edge weights (error probabilities)
    /// * `num_nodes` - Number of detector nodes
    pub fn extract(
        adjacency: &HashMap<usize, Vec<usize>>,
        edge_weights: &HashMap<(usize, usize), f32>,
        num_nodes: usize,
    ) -> Result<Self> {
        if num_nodes == 0 {
            return Err(NeuralDecoderError::EmptyGraph);
        }

        // Build graph for min-cut computation
        let graph = DynamicGraph::new();

        for (&node, neighbors) in adjacency {
            for &neighbor in neighbors {
                if node < neighbor {
                    let weight = edge_weights
                        .get(&(node, neighbor))
                        .or_else(|| edge_weights.get(&(neighbor, node)))
                        .copied()
                        .unwrap_or(1.0);
                    // Use 1/weight as edge capacity (higher prob = lower capacity)
                    let _ = graph.insert_edge(node as u64, neighbor as u64, 1.0 / (weight + 1e-6) as Weight);
                }
            }
        }

        // Compute global min-cut
        let mincut = MinCutBuilder::new()
            .exact()
            .build()
            .map_err(|e| NeuralDecoderError::MinCutError(e.to_string()))?;

        let global_mincut = if graph.num_edges() > 0 {
            mincut.min_cut_value()
        } else {
            f64::INFINITY
        };

        // Compute local cuts (simplified: use node degree as proxy)
        let mut local_cuts = vec![0.0; num_nodes];
        for (node, neighbors) in adjacency {
            let total_weight: f32 = neighbors
                .iter()
                .map(|&n| {
                    edge_weights
                        .get(&(*node, n))
                        .or_else(|| edge_weights.get(&(n, *node)))
                        .copied()
                        .unwrap_or(1.0)
                })
                .sum();
            local_cuts[*node] = total_weight as f64;
        }

        // Estimate error chain probability based on local structure
        let max_cut = local_cuts.iter().cloned().fold(0.0f64, f64::max).max(1e-6);
        let error_chain_prob: Vec<f64> = local_cuts
            .iter()
            .map(|&cut| 1.0 - (cut / max_cut))
            .collect();

        // Track which edges are likely in a cut (high weight / degree ratio)
        let mut edge_in_cut = HashMap::new();
        for (&node, neighbors) in adjacency {
            for &neighbor in neighbors {
                if node < neighbor {
                    let weight = edge_weights
                        .get(&(node, neighbor))
                        .or_else(|| edge_weights.get(&(neighbor, node)))
                        .copied()
                        .unwrap_or(1.0);
                    let avg_degree = (local_cuts[node] + local_cuts[neighbor]) / 2.0;
                    // Edge is likely in cut if it has high relative weight
                    edge_in_cut.insert((node, neighbor), (weight as f64) > avg_degree * 0.3);
                }
            }
        }

        Ok(Self {
            global_mincut,
            local_cuts,
            edge_in_cut,
            error_chain_prob,
        })
    }

    /// Convert to feature vector for each node
    pub fn to_features(&self, num_nodes: usize, feature_dim: usize) -> Array2<f32> {
        let mut features = Array2::zeros((num_nodes, feature_dim));
        let global_norm = self.global_mincut.max(1e-6);

        for i in 0..num_nodes {
            if feature_dim >= 1 {
                // Normalized local cut
                features[[i, 0]] = (self.local_cuts.get(i).copied().unwrap_or(0.0) / global_norm) as f32;
            }
            if feature_dim >= 2 {
                // Error chain probability
                features[[i, 1]] = self.error_chain_prob.get(i).copied().unwrap_or(0.5) as f32;
            }
            if feature_dim >= 3 {
                // Global context
                features[[i, 2]] = (global_norm.ln() / 10.0).tanh() as f32;
            }
            // Pad remaining dimensions with normalized local features
            for j in 3..feature_dim {
                features[[i, j]] = features[[i, j % 3]];
            }
        }

        features
    }
}

/// Boundary proximity features
#[derive(Debug, Clone)]
pub struct BoundaryFeatures {
    /// Distance from each node to nearest boundary
    pub distances: Vec<f32>,
    /// Boundary type for each node (0=inner, 1=X-boundary, 2=Z-boundary)
    pub boundary_types: Vec<u8>,
    /// Normalized boundary weights
    pub weights: Vec<f32>,
}

impl BoundaryFeatures {
    /// Compute boundary features from node positions
    ///
    /// # Arguments
    /// * `positions` - (x, y) coordinates for each node
    /// * `grid_size` - Size of the syndrome grid
    pub fn compute(positions: &[(f32, f32)], grid_size: usize) -> Self {
        let num_nodes = positions.len();
        let mut distances = Vec::with_capacity(num_nodes);
        let mut boundary_types = Vec::with_capacity(num_nodes);
        let mut weights = Vec::with_capacity(num_nodes);

        let size = grid_size as f32;

        for &(x, y) in positions {
            // Normalize to [0, 1]
            let x_norm = x / size.max(1.0);
            let y_norm = y / size.max(1.0);

            // Distance to nearest boundary
            let d_left = x_norm;
            let d_right = 1.0 - x_norm;
            let d_bottom = y_norm;
            let d_top = 1.0 - y_norm;

            let min_x_dist = d_left.min(d_right);
            let min_y_dist = d_bottom.min(d_top);
            let min_dist = min_x_dist.min(min_y_dist);

            distances.push(min_dist);

            // Determine boundary type
            // In surface codes: X-boundaries are left/right, Z-boundaries are top/bottom
            let boundary_type = if min_dist < 0.1 {
                if min_x_dist < min_y_dist {
                    1 // X-boundary
                } else {
                    2 // Z-boundary
                }
            } else {
                0 // Inner
            };
            boundary_types.push(boundary_type);

            // Weight based on distance (closer to boundary = higher weight for boundary effects)
            let weight = 1.0 - min_dist;
            weights.push(weight);
        }

        // Normalize weights
        let max_weight: f32 = weights.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
        for w in &mut weights {
            *w /= max_weight;
        }

        Self {
            distances,
            boundary_types,
            weights,
        }
    }

    /// Convert to feature matrix
    pub fn to_features(&self, feature_dim: usize) -> Array2<f32> {
        let num_nodes = self.distances.len();
        let mut features = Array2::zeros((num_nodes, feature_dim));

        for i in 0..num_nodes {
            if feature_dim >= 1 {
                features[[i, 0]] = self.distances[i];
            }
            if feature_dim >= 2 {
                features[[i, 1]] = self.boundary_types[i] as f32 / 2.0;
            }
            if feature_dim >= 3 {
                features[[i, 2]] = self.weights[i];
            }
            // Additional boundary-derived features
            if feature_dim >= 4 {
                // Sin/cos encoding of boundary type
                let angle = self.boundary_types[i] as f32 * std::f32::consts::PI / 3.0;
                features[[i, 3]] = angle.sin();
            }
            if feature_dim >= 5 {
                let angle = self.boundary_types[i] as f32 * std::f32::consts::PI / 3.0;
                features[[i, 4]] = angle.cos();
            }
            // Pad remaining with distance decay
            for j in 5..feature_dim {
                features[[i, j]] = (-(self.distances[i] * (j - 4) as f32)).exp();
            }
        }

        features
    }
}

/// Coherence-based confidence estimation
#[derive(Debug, Clone)]
pub struct CoherenceEstimator {
    /// Window size for local coherence
    window_size: usize,
    /// Minimum confidence threshold
    min_confidence: f32,
}

impl CoherenceEstimator {
    /// Create a new coherence estimator
    pub fn new(window_size: usize, min_confidence: f32) -> Self {
        Self {
            window_size,
            min_confidence: min_confidence.max(0.01),
        }
    }

    /// Estimate confidence scores based on prediction coherence
    ///
    /// # Arguments
    /// * `predictions` - Raw predictions (num_nodes, output_dim)
    /// * `adjacency` - Graph adjacency
    ///
    /// # Returns
    /// Confidence score for each node
    pub fn estimate(
        &self,
        predictions: &Array2<f32>,
        adjacency: &HashMap<usize, Vec<usize>>,
    ) -> Vec<f32> {
        let num_nodes = predictions.shape()[0];
        let output_dim = predictions.shape()[1];
        let mut confidences = vec![self.min_confidence; num_nodes];

        for node in 0..num_nodes {
            let neighbors = adjacency.get(&node).cloned().unwrap_or_default();

            if neighbors.is_empty() {
                // No neighbors: use prediction entropy as confidence
                let entropy = self.compute_entropy(&predictions.row(node).to_vec());
                confidences[node] = 1.0 - entropy;
                continue;
            }

            // Local coherence: similarity of prediction to neighbors
            let mut total_sim = 0.0;
            let node_pred: Vec<f32> = predictions.row(node).to_vec();

            for &neighbor in &neighbors {
                let neighbor_pred: Vec<f32> = predictions.row(neighbor).to_vec();
                let sim = self.cosine_similarity(&node_pred, &neighbor_pred);
                total_sim += sim;
            }

            let avg_sim = total_sim / neighbors.len() as f32;

            // High similarity to neighbors = high coherence = high confidence
            // Low entropy in predictions = high certainty = high confidence
            let entropy = self.compute_entropy(&node_pred);
            let certainty = 1.0 - entropy;

            // Combine coherence and certainty
            confidences[node] = (0.6 * avg_sim + 0.4 * certainty).max(self.min_confidence);
        }

        confidences
    }

    /// Compute normalized entropy of a probability distribution
    fn compute_entropy(&self, probs: &[f32]) -> f32 {
        let eps = 1e-10;
        let mut entropy = 0.0;
        for &p in probs {
            let p = p.clamp(eps as f32, 1.0 - eps as f32);
            entropy -= p * p.ln();
        }
        // Normalize by max entropy (uniform distribution)
        let max_entropy = (probs.len() as f32).ln();
        if max_entropy > eps as f32 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 1e-10 && norm_b > 1e-10 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Feature fusion module
///
/// Combines GNN embeddings, min-cut features, and boundary features
/// into a unified representation for error prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFusion {
    config: FusionConfig,
    /// GNN projection weights
    gnn_proj: Array2<f32>,
    /// MinCut projection weights
    mincut_proj: Array2<f32>,
    /// Boundary projection weights
    boundary_proj: Array2<f32>,
    /// Output projection
    output_proj: Array2<f32>,
    /// Biases
    bias: Array1<f32>,
}

impl FeatureFusion {
    /// Create a new feature fusion module
    pub fn new(config: FusionConfig) -> Result<Self> {
        config.validate()?;

        let combined_dim = config.gnn_dim + config.mincut_dim + 8; // 8 for boundary features

        // Initialize projection matrices with Xavier initialization
        let gnn_proj = Self::init_weights(config.gnn_dim, config.output_dim);
        let mincut_proj = Self::init_weights(config.mincut_dim, config.output_dim);
        let boundary_proj = Self::init_weights(8, config.output_dim);
        let output_proj = Self::init_weights(config.output_dim * 3, config.output_dim);
        let bias = Array1::zeros(config.output_dim);

        Ok(Self {
            config,
            gnn_proj,
            mincut_proj,
            boundary_proj,
            output_proj,
            bias,
        })
    }

    /// Xavier initialization for weight matrices
    fn init_weights(input_dim: usize, output_dim: usize) -> Array2<f32> {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};

        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        let normal = Normal::new(0.0, scale as f64).unwrap();
        let mut rng = rand::thread_rng();

        Array2::from_shape_fn((output_dim, input_dim), |_| {
            normal.sample(&mut rng) as f32
        })
    }

    /// Simple fuse for GNN and MinCut features only
    ///
    /// # Arguments
    /// * `gnn_features` - GNN embeddings (num_nodes, gnn_dim)
    /// * `mincut_features` - MinCut features (num_nodes, mincut_dim)
    ///
    /// # Returns
    /// Fused features (num_nodes, output_dim)
    pub fn fuse_simple(
        &self,
        gnn_features: &Array2<f32>,
        mincut_features: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        let num_nodes = gnn_features.shape()[0];

        // Create default boundary features (zeros)
        let boundary_features = Array2::zeros((num_nodes, 8));

        self.fuse(gnn_features, mincut_features, &boundary_features, None)
    }

    /// Fuse features from multiple sources
    ///
    /// # Arguments
    /// * `gnn_features` - GNN embeddings (num_nodes, gnn_dim)
    /// * `mincut_features` - MinCut features (num_nodes, mincut_dim)
    /// * `boundary_features` - Boundary features (num_nodes, 8)
    /// * `confidences` - Optional confidence scores for adaptive weighting
    ///
    /// # Returns
    /// Fused features (num_nodes, output_dim)
    pub fn fuse(
        &self,
        gnn_features: &Array2<f32>,
        mincut_features: &Array2<f32>,
        boundary_features: &Array2<f32>,
        confidences: Option<&[f32]>,
    ) -> Result<Array2<f32>> {
        let num_nodes = gnn_features.shape()[0];

        if mincut_features.shape()[0] != num_nodes || boundary_features.shape()[0] != num_nodes {
            return Err(NeuralDecoderError::shape_mismatch(
                vec![num_nodes],
                vec![mincut_features.shape()[0]],
            ));
        }

        // Project each feature set
        let gnn_proj = gnn_features.dot(&self.gnn_proj.t());
        let mincut_proj = mincut_features.dot(&self.mincut_proj.t());
        let boundary_proj = boundary_features.dot(&self.boundary_proj.t());

        // Determine weights (adaptive or fixed)
        let (gnn_w, mincut_w, boundary_w) = if self.config.adaptive_weights {
            if let Some(conf) = confidences {
                // Higher confidence -> trust GNN more
                let avg_conf: f32 = conf.iter().sum::<f32>() / conf.len() as f32;
                let gnn_w = self.config.gnn_weight * (1.0 + avg_conf);
                let mincut_w = self.config.mincut_weight * (2.0 - avg_conf);
                let boundary_w = self.config.boundary_weight;
                let total = gnn_w + mincut_w + boundary_w;
                (gnn_w / total, mincut_w / total, boundary_w / total)
            } else {
                (self.config.gnn_weight, self.config.mincut_weight, self.config.boundary_weight)
            }
        } else {
            (self.config.gnn_weight, self.config.mincut_weight, self.config.boundary_weight)
        };

        // Weighted combination
        let mut combined = Array2::zeros((num_nodes, self.config.output_dim * 3));
        for i in 0..num_nodes {
            // Per-node confidence scaling if available
            let node_scale = confidences.map(|c| c[i]).unwrap_or(1.0);

            for j in 0..self.config.output_dim {
                combined[[i, j]] = gnn_proj[[i, j]] * gnn_w * node_scale;
                combined[[i, self.config.output_dim + j]] = mincut_proj[[i, j]] * mincut_w;
                combined[[i, 2 * self.config.output_dim + j]] = boundary_proj[[i, j]] * boundary_w;
            }
        }

        // Final projection with ReLU and residual
        let output = combined.dot(&self.output_proj.t());
        let activated = output.mapv(|v| v.max(0.0)); // ReLU
        let with_bias = activated + &self.bias;

        Ok(with_bias)
    }

    /// Convenience method to compute all features and fuse them
    ///
    /// # Arguments
    /// * `gnn_embeddings` - GNN node embeddings
    /// * `adjacency` - Graph adjacency list
    /// * `edge_weights` - Edge weights for min-cut
    /// * `positions` - Node positions
    /// * `grid_size` - Grid size for boundary computation
    pub fn fuse_all(
        &self,
        gnn_embeddings: &Array2<f32>,
        adjacency: &HashMap<usize, Vec<usize>>,
        edge_weights: &HashMap<(usize, usize), f32>,
        positions: &[(f32, f32)],
        grid_size: usize,
    ) -> Result<Array2<f32>> {
        let num_nodes = gnn_embeddings.shape()[0];

        // Extract min-cut features
        let mincut_features = MinCutFeatures::extract(adjacency, edge_weights, num_nodes)?;
        let mincut_array = mincut_features.to_features(num_nodes, self.config.mincut_dim);

        // Compute boundary features
        let boundary_features = BoundaryFeatures::compute(positions, grid_size);
        let boundary_array = boundary_features.to_features(8);

        // Estimate confidences based on GNN predictions
        let coherence = CoherenceEstimator::new(3, 0.1);
        let confidences = coherence.estimate(gnn_embeddings, adjacency);

        // Fuse all features
        self.fuse(gnn_embeddings, &mincut_array, &boundary_array, Some(&confidences))
    }

    /// Get configuration
    pub fn config(&self) -> &FusionConfig {
        &self.config
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> (HashMap<usize, Vec<usize>>, HashMap<(usize, usize), f32>) {
        let mut adjacency = HashMap::new();
        adjacency.insert(0, vec![1, 2]);
        adjacency.insert(1, vec![0, 2, 3]);
        adjacency.insert(2, vec![0, 1, 3]);
        adjacency.insert(3, vec![1, 2]);

        let mut edge_weights = HashMap::new();
        edge_weights.insert((0, 1), 0.1);
        edge_weights.insert((0, 2), 0.2);
        edge_weights.insert((1, 2), 0.15);
        edge_weights.insert((1, 3), 0.1);
        edge_weights.insert((2, 3), 0.1);

        (adjacency, edge_weights)
    }

    #[test]
    fn test_mincut_features() {
        let (adjacency, edge_weights) = create_test_graph();
        let features = MinCutFeatures::extract(&adjacency, &edge_weights, 4).unwrap();

        assert_eq!(features.local_cuts.len(), 4);
        assert_eq!(features.error_chain_prob.len(), 4);
        assert!(features.global_mincut > 0.0);
    }

    #[test]
    fn test_boundary_features() {
        let positions = vec![
            (0.0, 0.0),  // Corner
            (0.5, 0.5),  // Center
            (1.0, 0.5),  // Right edge
            (0.5, 1.0),  // Top edge
        ];

        let features = BoundaryFeatures::compute(&positions, 1);

        assert_eq!(features.distances.len(), 4);
        assert!(features.distances[0] < features.distances[1]); // Corner closer to boundary
        assert_eq!(features.boundary_types[1], 0); // Center is inner
    }

    #[test]
    fn test_coherence_estimator() {
        let predictions = Array2::from_shape_fn((4, 2), |(i, j)| {
            if j == 0 { 0.8 } else { 0.2 }
        });

        let (adjacency, _) = create_test_graph();
        let estimator = CoherenceEstimator::new(3, 0.1);
        let confidences = estimator.estimate(&predictions, &adjacency);

        assert_eq!(confidences.len(), 4);
        for &c in &confidences {
            assert!(c >= 0.1 && c <= 1.0);
        }
    }

    #[test]
    fn test_fusion_config_validation() {
        let mut config = FusionConfig::default();
        assert!(config.validate().is_ok());

        config.gnn_weight = 0.8; // Now sum > 1
        assert!(config.validate().is_err());

        config.gnn_weight = 0.5;
        config.temperature = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_feature_fusion() {
        let config = FusionConfig {
            gnn_dim: 16,
            mincut_dim: 8,
            output_dim: 8,
            gnn_weight: 0.5,
            mincut_weight: 0.3,
            boundary_weight: 0.2,
            adaptive_weights: false,
            temperature: 1.0,
        };

        let fusion = FeatureFusion::new(config).unwrap();

        let num_nodes = 4;
        let gnn_features = Array2::from_shape_fn((num_nodes, 16), |(i, j)| {
            ((i + j) as f32) / 100.0
        });
        let mincut_features = Array2::from_shape_fn((num_nodes, 8), |(i, j)| {
            ((i * j) as f32) / 50.0
        });
        let boundary_features = Array2::from_shape_fn((num_nodes, 8), |(i, _)| {
            (i as f32) / 4.0
        });

        let fused = fusion.fuse(
            &gnn_features,
            &mincut_features,
            &boundary_features,
            None,
        ).unwrap();

        assert_eq!(fused.shape(), &[num_nodes, 8]);
    }

    #[test]
    fn test_fuse_all() {
        let config = FusionConfig {
            gnn_dim: 8,
            mincut_dim: 4,
            output_dim: 4,
            gnn_weight: 0.5,
            mincut_weight: 0.3,
            boundary_weight: 0.2,
            adaptive_weights: true,
            temperature: 1.0,
        };

        let fusion = FeatureFusion::new(config).unwrap();
        let (adjacency, edge_weights) = create_test_graph();

        let gnn_embeddings = Array2::from_shape_fn((4, 8), |(i, j)| {
            ((i + j) as f32) / 10.0
        });

        let positions = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ];

        let result = fusion.fuse_all(
            &gnn_embeddings,
            &adjacency,
            &edge_weights,
            &positions,
            2,
        );

        assert!(result.is_ok());
        let fused = result.unwrap();
        assert_eq!(fused.shape(), &[4, 4]);
    }

    #[test]
    fn test_mincut_features_to_array() {
        let (adjacency, edge_weights) = create_test_graph();
        let features = MinCutFeatures::extract(&adjacency, &edge_weights, 4).unwrap();

        let array = features.to_features(4, 8);
        assert_eq!(array.shape(), &[4, 8]);
    }

    #[test]
    fn test_boundary_features_to_array() {
        let positions = vec![(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (0.5, 1.0)];
        let features = BoundaryFeatures::compute(&positions, 2);

        let array = features.to_features(8);
        assert_eq!(array.shape(), &[4, 8]);
    }

    #[test]
    fn test_empty_graph_error() {
        let adjacency = HashMap::new();
        let edge_weights = HashMap::new();

        let result = MinCutFeatures::extract(&adjacency, &edge_weights, 0);
        assert!(matches!(result, Err(NeuralDecoderError::EmptyGraph)));
    }
}
