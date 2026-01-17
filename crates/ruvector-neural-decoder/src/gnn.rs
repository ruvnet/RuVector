//! GNN Encoder for Syndrome Graphs
//!
//! Implements graph neural network layers for encoding detector graphs
//! into fixed-dimensional representations suitable for the Mamba decoder.

use crate::error::{NeuralDecoderError, Result};
use crate::graph::DetectorGraph;
use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Configuration for the GNN encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GNNConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of GNN layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout rate
    pub dropout: f32,
}

impl Default for GNNConfig {
    fn default() -> Self {
        Self {
            input_dim: 5,
            embed_dim: 64,
            hidden_dim: 128,
            num_layers: 3,
            num_heads: 4,
            dropout: 0.1,
        }
    }
}

/// Linear layer for projections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    weights: Array2<f32>,
    bias: Array1<f32>,
}

impl Linear {
    /// Create a new linear layer with Xavier initialization
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        let normal = Normal::new(0.0, scale as f64).unwrap();

        let weights = Array2::from_shape_fn(
            (output_dim, input_dim),
            |_| normal.sample(&mut rng) as f32
        );
        let bias = Array1::zeros(output_dim);

        Self { weights, bias }
    }

    /// Forward pass
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let x = ArrayView1::from(input);
        let output = self.weights.dot(&x) + &self.bias;
        output.to_vec()
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.weights.shape()[0]
    }
}

/// Layer normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    eps: f32,
}

impl LayerNorm {
    /// Create new layer normalization
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            eps,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let x = ArrayView1::from(input);
        let mean = x.mean().unwrap_or(0.0);
        let variance = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;

        let normalized = x.mapv(|v| (v - mean) / (variance + self.eps).sqrt());
        let output = &self.gamma * &normalized + &self.beta;
        output.to_vec()
    }
}

/// Multi-head attention layer for graph attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionLayer {
    num_heads: usize,
    head_dim: usize,
    q_linear: Linear,
    k_linear: Linear,
    v_linear: Linear,
    out_linear: Linear,
    norm: LayerNorm,
}

impl AttentionLayer {
    /// Create a new attention layer
    pub fn new(embed_dim: usize, num_heads: usize) -> Result<Self> {
        if embed_dim % num_heads != 0 {
            return Err(NeuralDecoderError::attention_heads(embed_dim, num_heads));
        }

        let head_dim = embed_dim / num_heads;

        Ok(Self {
            num_heads,
            head_dim,
            q_linear: Linear::new(embed_dim, embed_dim),
            k_linear: Linear::new(embed_dim, embed_dim),
            v_linear: Linear::new(embed_dim, embed_dim),
            out_linear: Linear::new(embed_dim, embed_dim),
            norm: LayerNorm::new(embed_dim, 1e-5),
        })
    }

    /// Forward pass with attention
    pub fn forward(&self, query: &[f32], keys: &[Vec<f32>], values: &[Vec<f32>]) -> Vec<f32> {
        if keys.is_empty() || values.is_empty() {
            return self.norm.forward(query);
        }

        // Project query, keys, values
        let q = self.q_linear.forward(query);
        let k: Vec<Vec<f32>> = keys.iter().map(|k| self.k_linear.forward(k)).collect();
        let v: Vec<Vec<f32>> = values.iter().map(|v| self.v_linear.forward(v)).collect();

        // Multi-head attention
        let q_heads = self.split_heads(&q);
        let k_heads: Vec<Vec<Vec<f32>>> = k.iter().map(|kv| self.split_heads(kv)).collect();
        let v_heads: Vec<Vec<Vec<f32>>> = v.iter().map(|vv| self.split_heads(vv)).collect();

        let mut head_outputs = Vec::new();
        for h in 0..self.num_heads {
            let q_h = &q_heads[h];
            let k_h: Vec<&Vec<f32>> = k_heads.iter().map(|heads| &heads[h]).collect();
            let v_h: Vec<&Vec<f32>> = v_heads.iter().map(|heads| &heads[h]).collect();

            let head_output = self.scaled_dot_product_attention(q_h, &k_h, &v_h);
            head_outputs.push(head_output);
        }

        // Concatenate heads
        let concat: Vec<f32> = head_outputs.into_iter().flatten().collect();

        // Output projection and residual
        let projected = self.out_linear.forward(&concat);
        let residual: Vec<f32> = query.iter().zip(projected.iter())
            .map(|(q, p)| q + p)
            .collect();

        self.norm.forward(&residual)
    }

    /// Split vector into heads
    fn split_heads(&self, x: &[f32]) -> Vec<Vec<f32>> {
        (0..self.num_heads)
            .map(|h| {
                let start = h * self.head_dim;
                let end = start + self.head_dim;
                x[start..end].to_vec()
            })
            .collect()
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        query: &[f32],
        keys: &[&Vec<f32>],
        values: &[&Vec<f32>],
    ) -> Vec<f32> {
        if keys.is_empty() {
            return query.to_vec();
        }

        let scale = (self.head_dim as f32).sqrt();

        // Compute scores
        let scores: Vec<f32> = keys
            .iter()
            .map(|k| {
                let dot: f32 = query.iter().zip(k.iter()).map(|(q, k)| q * k).sum();
                dot / scale
            })
            .collect();

        // Softmax
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum::<f32>().max(1e-10);
        let weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Weighted sum
        let mut output = vec![0.0; self.head_dim];
        for (weight, value) in weights.iter().zip(values.iter()) {
            for (out, &val) in output.iter_mut().zip(value.iter()) {
                *out += weight * val;
            }
        }

        output
    }

    /// Get attention scores (for interpretation)
    pub fn attention_scores(&self, query: &[f32], keys: &[Vec<f32>]) -> Vec<f32> {
        if keys.is_empty() {
            return Vec::new();
        }

        let q = self.q_linear.forward(query);
        let k: Vec<Vec<f32>> = keys.iter().map(|k| self.k_linear.forward(k)).collect();

        let scale = (self.head_dim as f32).sqrt() * (self.num_heads as f32);

        let scores: Vec<f32> = k
            .iter()
            .map(|kv| {
                let dot: f32 = q.iter().zip(kv.iter()).map(|(q, k)| q * k).sum();
                dot / scale
            })
            .collect();

        // Softmax
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum::<f32>().max(1e-10);
        exp_scores.iter().map(|&e| e / sum_exp).collect()
    }
}

/// GNN Encoder for syndrome graphs
#[derive(Debug, Clone)]
pub struct GNNEncoder {
    config: GNNConfig,
    input_projection: Linear,
    layers: Vec<AttentionLayer>,
    output_projection: Linear,
}

impl GNNEncoder {
    /// Create a new GNN encoder
    pub fn new(config: GNNConfig) -> Self {
        let input_projection = Linear::new(config.input_dim, config.embed_dim);

        let layers: Vec<AttentionLayer> = (0..config.num_layers)
            .map(|_| AttentionLayer::new(config.embed_dim, config.num_heads).unwrap())
            .collect();

        let output_projection = Linear::new(config.embed_dim, config.hidden_dim);

        Self {
            config,
            input_projection,
            layers,
            output_projection,
        }
    }

    /// Encode a detector graph
    pub fn encode(&self, graph: &DetectorGraph) -> Result<Array2<f32>> {
        if graph.nodes.is_empty() {
            return Err(NeuralDecoderError::EmptyGraph);
        }

        let num_nodes = graph.num_nodes();

        // Project input features
        let mut embeddings: Vec<Vec<f32>> = graph.nodes
            .iter()
            .map(|n| self.input_projection.forward(&n.features))
            .collect();

        // Message passing layers
        for layer in &self.layers {
            let mut new_embeddings = Vec::with_capacity(num_nodes);

            for (node_id, embedding) in embeddings.iter().enumerate() {
                // Get neighbor embeddings
                let neighbor_ids = graph.neighbors(node_id)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);

                let neighbor_embeddings: Vec<Vec<f32>> = neighbor_ids
                    .iter()
                    .filter_map(|&nid| embeddings.get(nid).cloned())
                    .collect();

                // Apply attention
                let updated = layer.forward(embedding, &neighbor_embeddings, &neighbor_embeddings);
                new_embeddings.push(updated);
            }

            embeddings = new_embeddings;
        }

        // Output projection
        let output_embeddings: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|e| self.output_projection.forward(e))
            .collect();

        // Convert to array
        let mut result = Array2::zeros((num_nodes, self.config.hidden_dim));
        for (i, emb) in output_embeddings.iter().enumerate() {
            for (j, &val) in emb.iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        Ok(result)
    }

    /// Get node embeddings without output projection (for debugging)
    pub fn get_intermediate_embeddings(&self, graph: &DetectorGraph, layer_idx: usize) -> Result<Vec<Vec<f32>>> {
        if graph.nodes.is_empty() {
            return Err(NeuralDecoderError::EmptyGraph);
        }

        let num_nodes = graph.num_nodes();
        let layer_count = layer_idx.min(self.layers.len());

        // Project input features
        let mut embeddings: Vec<Vec<f32>> = graph.nodes
            .iter()
            .map(|n| self.input_projection.forward(&n.features))
            .collect();

        // Message passing layers up to layer_idx
        for layer in self.layers.iter().take(layer_count) {
            let mut new_embeddings = Vec::with_capacity(num_nodes);

            for (node_id, embedding) in embeddings.iter().enumerate() {
                let neighbor_ids = graph.neighbors(node_id)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);

                let neighbor_embeddings: Vec<Vec<f32>> = neighbor_ids
                    .iter()
                    .filter_map(|&nid| embeddings.get(nid).cloned())
                    .collect();

                let updated = layer.forward(embedding, &neighbor_embeddings, &neighbor_embeddings);
                new_embeddings.push(updated);
            }

            embeddings = new_embeddings;
        }

        Ok(embeddings)
    }

    /// Get the configuration
    pub fn config(&self) -> &GNNConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    #[test]
    fn test_gnn_config_default() {
        let config = GNNConfig::default();
        assert_eq!(config.input_dim, 5);
        assert_eq!(config.embed_dim, 64);
        assert_eq!(config.num_heads, 4);
    }

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(4, 8);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = linear.forward(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_layer_norm() {
        let norm = LayerNorm::new(4, 1e-5);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = norm.forward(&input);
        assert_eq!(output.len(), 4);

        // Check zero mean (approximately)
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_attention_layer_creation() {
        let layer = AttentionLayer::new(64, 4);
        assert!(layer.is_ok());

        // Invalid: embed_dim not divisible by num_heads
        let layer = AttentionLayer::new(64, 3);
        assert!(layer.is_err());
    }

    #[test]
    fn test_attention_forward() {
        let layer = AttentionLayer::new(8, 2).unwrap();
        let query = vec![0.5; 8];
        let keys = vec![vec![0.3; 8], vec![0.7; 8]];
        let values = vec![vec![0.2; 8], vec![0.8; 8]];

        let output = layer.forward(&query, &keys, &values);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_attention_empty_neighbors() {
        let layer = AttentionLayer::new(8, 2).unwrap();
        let query = vec![0.5; 8];
        let keys: Vec<Vec<f32>> = vec![];
        let values: Vec<Vec<f32>> = vec![];

        let output = layer.forward(&query, &keys, &values);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_attention_scores() {
        let layer = AttentionLayer::new(8, 2).unwrap();
        let query = vec![0.5; 8];
        let keys = vec![vec![0.3; 8], vec![0.7; 8]];

        let scores = layer.attention_scores(&query, &keys);
        assert_eq!(scores.len(), 2);

        // Scores should sum to 1.0
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gnn_encoder_creation() {
        let config = GNNConfig::default();
        let encoder = GNNEncoder::new(config);
        assert_eq!(encoder.config().num_layers, 3);
    }

    #[test]
    fn test_gnn_encode_small_graph() {
        let config = GNNConfig {
            input_dim: 5,
            embed_dim: 16,
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            dropout: 0.0,
        };
        let encoder = GNNEncoder::new(config);

        let graph = GraphBuilder::from_surface_code(3)
            .build()
            .unwrap();

        let embeddings = encoder.encode(&graph).unwrap();
        assert_eq!(embeddings.shape(), &[9, 32]);
    }

    #[test]
    fn test_gnn_encode_with_syndrome() {
        let config = GNNConfig {
            input_dim: 5,
            embed_dim: 16,
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            dropout: 0.0,
        };
        let encoder = GNNEncoder::new(config);

        let syndrome = vec![true, false, true, false, false, false, true, false, false];
        let graph = GraphBuilder::from_surface_code(3)
            .with_syndrome(&syndrome)
            .unwrap()
            .build()
            .unwrap();

        let embeddings = encoder.encode(&graph).unwrap();
        assert_eq!(embeddings.shape(), &[9, 32]);
    }

    #[test]
    fn test_gnn_encode_empty_graph() {
        let config = GNNConfig::default();
        let encoder = GNNEncoder::new(config);

        let graph = crate::graph::DetectorGraph::new(3);
        let result = encoder.encode(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_intermediate_embeddings() {
        let config = GNNConfig {
            input_dim: 5,
            embed_dim: 16,
            hidden_dim: 32,
            num_layers: 3,
            num_heads: 4,
            dropout: 0.0,
        };
        let encoder = GNNEncoder::new(config);

        let graph = GraphBuilder::from_surface_code(3)
            .build()
            .unwrap();

        // Get embeddings at different layers
        let layer0 = encoder.get_intermediate_embeddings(&graph, 0).unwrap();
        let layer1 = encoder.get_intermediate_embeddings(&graph, 1).unwrap();
        let layer2 = encoder.get_intermediate_embeddings(&graph, 2).unwrap();

        assert_eq!(layer0.len(), 9);
        assert_eq!(layer1.len(), 9);
        assert_eq!(layer2.len(), 9);

        // Each embedding should have embed_dim dimensions
        assert_eq!(layer0[0].len(), 16);
        assert_eq!(layer1[0].len(), 16);
        assert_eq!(layer2[0].len(), 16);
    }

    #[test]
    fn test_gnn_deterministic_structure() {
        // Test that different syndromes produce different embeddings
        let config = GNNConfig {
            input_dim: 5,
            embed_dim: 16,
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            dropout: 0.0,
        };
        let encoder = GNNEncoder::new(config);

        let syndrome1 = vec![true, false, false, false, false, false, false, false, false];
        let syndrome2 = vec![false, false, false, false, true, false, false, false, false];

        let graph1 = GraphBuilder::from_surface_code(3)
            .with_syndrome(&syndrome1)
            .unwrap()
            .build()
            .unwrap();

        let graph2 = GraphBuilder::from_surface_code(3)
            .with_syndrome(&syndrome2)
            .unwrap()
            .build()
            .unwrap();

        let emb1 = encoder.encode(&graph1).unwrap();
        let emb2 = encoder.encode(&graph2).unwrap();

        // Embeddings should differ
        let diff: f32 = (emb1.clone() - emb2.clone())
            .iter()
            .map(|x| x.abs())
            .sum();
        assert!(diff > 0.0);
    }
}
