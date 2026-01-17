//! Graph Attention Encoder for Neural Quantum Error Decoding
//!
//! This module implements a Graph Neural Network encoder with:
//! - Message passing between detector nodes
//! - GraphRoPE-style positional encoding
//! - Multi-head attention aggregation
//! - Support for variable-size syndrome graphs
//!
//! ## Architecture
//!
//! The encoder operates on detector graphs derived from quantum error syndromes:
//!
//! 1. **Positional Encoding**: GraphRoPE encodes structural position in the graph
//! 2. **Message Passing**: Each node aggregates information from neighbors
//! 3. **Attention**: Multi-head attention weights message importance
//! 4. **Layer Norm**: Stabilizes training with layer normalization

use crate::error::{NeuralDecoderError, Result};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the Graph Attention Encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Input feature dimension per node
    pub input_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Output embedding dimension
    pub output_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of message passing layers
    pub num_layers: usize,
    /// Dropout rate (0.0 to 1.0)
    pub dropout: f32,
    /// Maximum number of nodes supported
    pub max_nodes: usize,
    /// Use positional encoding
    pub use_positional_encoding: bool,
    /// Positional encoding dimension
    pub pos_encoding_dim: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            input_dim: 3,      // (syndrome_bit, x_coord, y_coord)
            hidden_dim: 64,
            output_dim: 64,
            num_heads: 4,
            num_layers: 3,
            dropout: 0.1,
            max_nodes: 1024,
            use_positional_encoding: true,
            pos_encoding_dim: 16,
        }
    }
}

impl EncoderConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.hidden_dim % self.num_heads != 0 {
            return Err(NeuralDecoderError::attention_heads(
                self.hidden_dim,
                self.num_heads,
            ));
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(NeuralDecoderError::ConfigError(format!(
                "Dropout must be in [0, 1], got {}",
                self.dropout
            )));
        }
        Ok(())
    }
}

/// Linear transformation layer with Xavier initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    weights: Array2<f32>,
    bias: Array1<f32>,
}

impl Linear {
    /// Create a new linear layer with Xavier/Glorot initialization
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        let normal = Normal::new(0.0, scale as f64).unwrap();

        let weights = Array2::from_shape_fn((output_dim, input_dim), |_| {
            normal.sample(&mut rng) as f32
        });
        let bias = Array1::zeros(output_dim);

        Self { weights, bias }
    }

    /// Forward pass: y = Wx + b
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        self.weights.dot(input) + &self.bias
    }

    /// Forward pass for batched input
    pub fn forward_batch(&self, input: &Array2<f32>) -> Array2<f32> {
        // input: (batch, input_dim), output: (batch, output_dim)
        input.dot(&self.weights.t()) + &self.bias
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.weights.shape()[0]
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.weights.shape()[1]
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
    /// Create a new layer normalization
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            eps,
        }
    }

    /// Normalize a single vector
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mean = input.mean().unwrap_or(0.0);
        let variance = input.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / input.len() as f32;
        let normalized = input.mapv(|v| (v - mean) / (variance + self.eps).sqrt());
        &self.gamma * &normalized + &self.beta
    }

    /// Normalize batch of vectors (along last axis)
    pub fn forward_batch(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = Array2::zeros(input.raw_dim());
        for (i, row) in input.axis_iter(Axis(0)).enumerate() {
            let normalized = self.forward(&row.to_owned());
            output.row_mut(i).assign(&normalized);
        }
        output
    }
}

/// GraphRoPE-style positional encoding for graph nodes
///
/// Encodes node position using sinusoidal functions based on:
/// - Graph distance from boundary
/// - Local neighborhood structure
/// - Coordinate position in lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPositionalEncoding {
    dim: usize,
    max_seq_len: usize,
    /// Precomputed sin/cos tables
    sin_table: Array2<f32>,
    cos_table: Array2<f32>,
}

impl GraphPositionalEncoding {
    /// Create a new positional encoding
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let half_dim = dim / 2;
        let mut sin_table = Array2::zeros((max_seq_len, dim));
        let mut cos_table = Array2::zeros((max_seq_len, dim));

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let angle = pos as f32 / (10000_f32.powf(2.0 * i as f32 / dim as f32));
                sin_table[[pos, 2 * i]] = angle.sin();
                sin_table[[pos, 2 * i + 1]] = angle.cos();
                cos_table[[pos, 2 * i]] = angle.cos();
                cos_table[[pos, 2 * i + 1]] = (-angle).sin();
            }
        }

        Self {
            dim,
            max_seq_len,
            sin_table,
            cos_table,
        }
    }

    /// Encode node positions based on graph structure
    ///
    /// # Arguments
    /// * `positions` - (x, y) coordinates for each node
    /// * `distances_to_boundary` - Distance from each node to nearest boundary
    pub fn encode(
        &self,
        positions: &[(f32, f32)],
        distances_to_boundary: &[f32],
    ) -> Array2<f32> {
        let n_nodes = positions.len();
        let mut encoding = Array2::zeros((n_nodes, self.dim));

        for (i, ((x, y), dist)) in positions.iter().zip(distances_to_boundary.iter()).enumerate() {
            // Encode x-position
            let x_idx = ((*x * 100.0) as usize).min(self.max_seq_len - 1);
            // Encode y-position
            let y_idx = ((*y * 100.0) as usize).min(self.max_seq_len - 1);
            // Encode boundary distance
            let d_idx = ((*dist * 50.0) as usize).min(self.max_seq_len - 1);

            let third = self.dim / 3;

            // X-position encoding
            for j in 0..third.min(self.dim) {
                encoding[[i, j]] = self.sin_table[[x_idx, j % self.dim]];
            }

            // Y-position encoding
            for j in third..(2 * third).min(self.dim) {
                encoding[[i, j]] = self.sin_table[[y_idx, j % self.dim]];
            }

            // Boundary distance encoding
            for j in (2 * third)..self.dim {
                encoding[[i, j]] = self.sin_table[[d_idx, j % self.dim]];
            }
        }

        encoding
    }

    /// Apply rotary position encoding to queries and keys
    pub fn apply_rope(&self, x: &Array2<f32>, positions: &[usize]) -> Array2<f32> {
        let mut output = x.clone();

        for (i, &pos) in positions.iter().enumerate() {
            let pos = pos.min(self.max_seq_len - 1);
            for j in 0..x.shape()[1] {
                let sin_val = self.sin_table[[pos, j % self.dim]];
                let cos_val = self.cos_table[[pos, j % self.dim]];

                // Rotary encoding: x' = x * cos + rotate(x) * sin
                if j + 1 < x.shape()[1] {
                    let x_val = x[[i, j]];
                    let x_rotated = if j % 2 == 0 {
                        -x[[i, j + 1]]
                    } else {
                        x[[i, j - 1]]
                    };
                    output[[i, j]] = x_val * cos_val + x_rotated * sin_val;
                }
            }
        }

        output
    }
}

/// Multi-head graph attention mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    scale: f32,
}

impl GraphMultiHeadAttention {
    /// Create a new multi-head attention layer
    pub fn new(embed_dim: usize, num_heads: usize) -> Result<Self> {
        if embed_dim % num_heads != 0 {
            return Err(NeuralDecoderError::attention_heads(embed_dim, num_heads));
        }

        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            num_heads,
            head_dim,
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            scale,
        })
    }

    /// Forward pass with graph adjacency mask
    ///
    /// # Arguments
    /// * `x` - Node features (n_nodes, embed_dim)
    /// * `adjacency` - Adjacency list: node_idx -> Vec<neighbor_idx>
    /// * `edge_weights` - Optional edge weights
    pub fn forward(
        &self,
        x: &Array2<f32>,
        adjacency: &HashMap<usize, Vec<usize>>,
        edge_weights: Option<&HashMap<(usize, usize), f32>>,
    ) -> Array2<f32> {
        let n_nodes = x.shape()[0];
        let embed_dim = x.shape()[1];

        // Project queries, keys, values
        let q = self.q_proj.forward_batch(x);
        let k = self.k_proj.forward_batch(x);
        let v = self.v_proj.forward_batch(x);

        let mut output = Array2::zeros((n_nodes, embed_dim));

        // Process each node
        for node in 0..n_nodes {
            let neighbors = adjacency.get(&node).cloned().unwrap_or_default();

            if neighbors.is_empty() {
                // No neighbors: keep original features
                output.row_mut(node).assign(&x.row(node));
                continue;
            }

            // Compute attention for each head
            let mut head_outputs = Vec::with_capacity(self.num_heads);

            for h in 0..self.num_heads {
                let start = h * self.head_dim;
                let end = start + self.head_dim;

                // Query for this node and head
                let q_h: Vec<f32> = q.row(node).slice(ndarray::s![start..end]).to_vec();

                // Compute attention scores with neighbors
                let mut scores = Vec::with_capacity(neighbors.len());
                for &neighbor in &neighbors {
                    let k_h: Vec<f32> = k.row(neighbor).slice(ndarray::s![start..end]).to_vec();
                    let score: f32 = q_h.iter().zip(k_h.iter()).map(|(a, b)| a * b).sum();

                    // Apply edge weight if available
                    let edge_weight = edge_weights
                        .and_then(|w| w.get(&(node, neighbor)).or_else(|| w.get(&(neighbor, node))))
                        .copied()
                        .unwrap_or(1.0);

                    scores.push(score * self.scale * edge_weight);
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum::<f32>().max(1e-10);
                let attention_weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

                // Weighted sum of values
                let mut head_out = vec![0.0f32; self.head_dim];
                for (weight, &neighbor) in attention_weights.iter().zip(neighbors.iter()) {
                    let v_h: Vec<f32> = v.row(neighbor).slice(ndarray::s![start..end]).to_vec();
                    for (out, &val) in head_out.iter_mut().zip(v_h.iter()) {
                        *out += weight * val;
                    }
                }
                head_outputs.extend(head_out);
            }

            // Project output
            let concat = Array1::from_vec(head_outputs);
            let projected = self.out_proj.forward(&concat);
            output.row_mut(node).assign(&projected);
        }

        output
    }
}

/// Message passing layer for graph neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePassingLayer {
    /// Message transformation
    msg_linear: Linear,
    /// Update transformation
    update_linear: Linear,
    /// Layer normalization
    layer_norm: LayerNorm,
    /// Attention mechanism
    attention: GraphMultiHeadAttention,
    /// Dropout rate
    dropout: f32,
}

impl MessagePassingLayer {
    /// Create a new message passing layer
    pub fn new(hidden_dim: usize, num_heads: usize, dropout: f32) -> Result<Self> {
        Ok(Self {
            msg_linear: Linear::new(hidden_dim, hidden_dim),
            update_linear: Linear::new(hidden_dim * 2, hidden_dim),
            layer_norm: LayerNorm::new(hidden_dim, 1e-5),
            attention: GraphMultiHeadAttention::new(hidden_dim, num_heads)?,
            dropout,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Node features (n_nodes, hidden_dim)
    /// * `adjacency` - Graph adjacency list
    /// * `edge_weights` - Optional edge weights
    pub fn forward(
        &self,
        x: &Array2<f32>,
        adjacency: &HashMap<usize, Vec<usize>>,
        edge_weights: Option<&HashMap<(usize, usize), f32>>,
    ) -> Array2<f32> {
        // Attention-based message aggregation
        let messages = self.attention.forward(x, adjacency, edge_weights);

        // Concatenate original features with aggregated messages
        let n_nodes = x.shape()[0];
        let hidden_dim = x.shape()[1];
        let mut concat = Array2::zeros((n_nodes, hidden_dim * 2));

        for i in 0..n_nodes {
            for j in 0..hidden_dim {
                concat[[i, j]] = x[[i, j]];
                concat[[i, hidden_dim + j]] = messages[[i, j]];
            }
        }

        // Update transformation
        let updated = self.update_linear.forward_batch(&concat);

        // Residual connection and layer norm
        let mut output = Array2::zeros((n_nodes, hidden_dim));
        for i in 0..n_nodes {
            for j in 0..hidden_dim {
                output[[i, j]] = x[[i, j]] + updated[[i, j]] * (1.0 - self.dropout);
            }
        }

        self.layer_norm.forward_batch(&output)
    }
}

/// Graph Attention Encoder for syndrome decoding
///
/// Encodes detector graphs into fixed-size embeddings using:
/// - Positional encoding of graph structure
/// - Multiple message passing layers
/// - Multi-head attention aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAttentionEncoder {
    config: EncoderConfig,
    /// Input projection
    input_proj: Linear,
    /// Positional encoding
    pos_encoding: GraphPositionalEncoding,
    /// Message passing layers
    layers: Vec<MessagePassingLayer>,
    /// Output projection
    output_proj: Linear,
    /// Final layer norm
    final_norm: LayerNorm,
}

impl GraphAttentionEncoder {
    /// Create a new graph attention encoder
    pub fn new(config: EncoderConfig) -> Result<Self> {
        config.validate()?;

        let actual_input_dim = if config.use_positional_encoding {
            config.input_dim + config.pos_encoding_dim
        } else {
            config.input_dim
        };

        let input_proj = Linear::new(actual_input_dim, config.hidden_dim);
        let pos_encoding = GraphPositionalEncoding::new(config.pos_encoding_dim, config.max_nodes);

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(MessagePassingLayer::new(
                config.hidden_dim,
                config.num_heads,
                config.dropout,
            )?);
        }

        let output_proj = Linear::new(config.hidden_dim, config.output_dim);
        let final_norm = LayerNorm::new(config.output_dim, 1e-5);

        Ok(Self {
            config,
            input_proj,
            pos_encoding,
            layers,
            output_proj,
            final_norm,
        })
    }

    /// Encode a detector graph
    ///
    /// # Arguments
    /// * `node_features` - Features for each node (n_nodes, input_dim)
    /// * `adjacency` - Adjacency list: node_idx -> Vec<neighbor_idx>
    /// * `positions` - (x, y) coordinates for each node
    /// * `boundary_distances` - Distance from each node to nearest boundary
    /// * `edge_weights` - Optional edge weights
    ///
    /// # Returns
    /// Node embeddings (n_nodes, output_dim)
    pub fn encode(
        &self,
        node_features: &Array2<f32>,
        adjacency: &HashMap<usize, Vec<usize>>,
        positions: &[(f32, f32)],
        boundary_distances: &[f32],
        edge_weights: Option<&HashMap<(usize, usize), f32>>,
    ) -> Result<Array2<f32>> {
        let n_nodes = node_features.shape()[0];

        if n_nodes == 0 {
            return Err(NeuralDecoderError::EmptyGraph);
        }

        if node_features.shape()[1] != self.config.input_dim {
            return Err(NeuralDecoderError::embed_dim(
                self.config.input_dim,
                node_features.shape()[1],
            ));
        }

        // Add positional encoding
        let features = if self.config.use_positional_encoding {
            let pos_enc = self.pos_encoding.encode(positions, boundary_distances);

            // Concatenate node features with positional encoding
            let mut combined = Array2::zeros((n_nodes, self.config.input_dim + self.config.pos_encoding_dim));
            for i in 0..n_nodes {
                for j in 0..self.config.input_dim {
                    combined[[i, j]] = node_features[[i, j]];
                }
                for j in 0..self.config.pos_encoding_dim {
                    combined[[i, self.config.input_dim + j]] = pos_enc[[i, j]];
                }
            }
            combined
        } else {
            node_features.clone()
        };

        // Input projection
        let mut x = self.input_proj.forward_batch(&features);

        // Message passing layers
        for layer in &self.layers {
            x = layer.forward(&x, adjacency, edge_weights);
        }

        // Output projection and normalization
        let output = self.output_proj.forward_batch(&x);
        Ok(self.final_norm.forward_batch(&output))
    }

    /// Get the output dimension
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
    }

    /// Get the configuration
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Pool node embeddings into a single graph embedding
    ///
    /// # Arguments
    /// * `node_embeddings` - Node embeddings (n_nodes, output_dim)
    /// * `attention_weights` - Optional attention weights for weighted pooling
    ///
    /// # Returns
    /// Graph-level embedding (output_dim,)
    pub fn pool(
        &self,
        node_embeddings: &Array2<f32>,
        attention_weights: Option<&[f32]>,
    ) -> Array1<f32> {
        let n_nodes = node_embeddings.shape()[0];

        if n_nodes == 0 {
            return Array1::zeros(self.config.output_dim);
        }

        match attention_weights {
            Some(weights) => {
                // Weighted mean pooling
                let sum: f32 = weights.iter().sum::<f32>().max(1e-10);
                let normalized: Vec<f32> = weights.iter().map(|&w| w / sum).collect();

                let mut pooled = Array1::zeros(self.config.output_dim);
                for (i, &weight) in normalized.iter().enumerate() {
                    for j in 0..self.config.output_dim {
                        pooled[j] += weight * node_embeddings[[i, j]];
                    }
                }
                pooled
            }
            None => {
                // Mean pooling
                node_embeddings.mean_axis(Axis(0)).unwrap()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer() {
        let linear = Linear::new(4, 8);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = linear.forward(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_linear_batch() {
        let linear = Linear::new(4, 8);
        let input = Array2::from_shape_vec((3, 4), vec![1.0; 12]).unwrap();
        let output = linear.forward_batch(&input);
        assert_eq!(output.shape(), &[3, 8]);
    }

    #[test]
    fn test_layer_norm() {
        let norm = LayerNorm::new(4, 1e-5);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = norm.forward(&input);

        // Check normalized mean is approximately 0
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        assert!((mean).abs() < 1e-4);
    }

    #[test]
    fn test_positional_encoding() {
        let pe = GraphPositionalEncoding::new(16, 100);
        let positions = vec![(0.0, 0.0), (0.1, 0.2), (0.5, 0.5)];
        let distances = vec![0.0, 0.5, 1.0];

        let encoding = pe.encode(&positions, &distances);
        assert_eq!(encoding.shape(), &[3, 16]);
    }

    #[test]
    fn test_graph_attention() {
        let attention = GraphMultiHeadAttention::new(8, 2).unwrap();
        let x = Array2::from_shape_vec((4, 8), vec![0.5; 32]).unwrap();

        let mut adjacency = HashMap::new();
        adjacency.insert(0, vec![1, 2]);
        adjacency.insert(1, vec![0, 2, 3]);
        adjacency.insert(2, vec![0, 1, 3]);
        adjacency.insert(3, vec![1, 2]);

        let output = attention.forward(&x, &adjacency, None);
        assert_eq!(output.shape(), &[4, 8]);
    }

    #[test]
    fn test_message_passing_layer() {
        let layer = MessagePassingLayer::new(8, 2, 0.1).unwrap();
        let x = Array2::from_shape_vec((4, 8), vec![0.5; 32]).unwrap();

        let mut adjacency = HashMap::new();
        adjacency.insert(0, vec![1, 2]);
        adjacency.insert(1, vec![0, 2, 3]);
        adjacency.insert(2, vec![0, 1, 3]);
        adjacency.insert(3, vec![1, 2]);

        let output = layer.forward(&x, &adjacency, None);
        assert_eq!(output.shape(), &[4, 8]);
    }

    #[test]
    fn test_encoder_config_validation() {
        let mut config = EncoderConfig::default();
        assert!(config.validate().is_ok());

        config.num_heads = 5; // Not divisible
        assert!(config.validate().is_err());

        config.num_heads = 4;
        config.dropout = 1.5; // Out of range
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_graph_attention_encoder() {
        let config = EncoderConfig {
            input_dim: 3,
            hidden_dim: 16,
            output_dim: 16,
            num_heads: 2,
            num_layers: 2,
            dropout: 0.1,
            max_nodes: 100,
            use_positional_encoding: true,
            pos_encoding_dim: 8,
        };

        let encoder = GraphAttentionEncoder::new(config).unwrap();

        // Create test graph
        let n_nodes = 4;
        let node_features = Array2::from_shape_fn((n_nodes, 3), |(i, j)| {
            ((i + j) as f32) / 10.0
        });

        let mut adjacency = HashMap::new();
        adjacency.insert(0, vec![1, 2]);
        adjacency.insert(1, vec![0, 2, 3]);
        adjacency.insert(2, vec![0, 1, 3]);
        adjacency.insert(3, vec![1, 2]);

        let positions = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let distances = vec![0.0, 0.5, 0.5, 1.0];

        let embeddings = encoder.encode(
            &node_features,
            &adjacency,
            &positions,
            &distances,
            None,
        ).unwrap();

        assert_eq!(embeddings.shape(), &[n_nodes, 16]);
    }

    #[test]
    fn test_encoder_pooling() {
        let config = EncoderConfig {
            input_dim: 3,
            hidden_dim: 8,
            output_dim: 8,
            num_heads: 2,
            num_layers: 1,
            dropout: 0.0,
            max_nodes: 100,
            use_positional_encoding: false,
            pos_encoding_dim: 0,
        };

        let encoder = GraphAttentionEncoder::new(config).unwrap();
        let embeddings = Array2::from_shape_vec((3, 8), vec![1.0; 24]).unwrap();

        // Test mean pooling
        let pooled = encoder.pool(&embeddings, None);
        assert_eq!(pooled.len(), 8);

        // Test weighted pooling
        let weights = vec![0.5, 0.3, 0.2];
        let weighted_pooled = encoder.pool(&embeddings, Some(&weights));
        assert_eq!(weighted_pooled.len(), 8);
    }

    #[test]
    fn test_empty_graph_error() {
        let config = EncoderConfig::default();
        let encoder = GraphAttentionEncoder::new(config).unwrap();

        let empty_features = Array2::zeros((0, 3));
        let adjacency = HashMap::new();

        let result = encoder.encode(
            &empty_features,
            &adjacency,
            &[],
            &[],
            None,
        );

        assert!(matches!(result, Err(NeuralDecoderError::EmptyGraph)));
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let config = EncoderConfig::default();
        let encoder = GraphAttentionEncoder::new(config).unwrap();

        let wrong_features = Array2::zeros((4, 5)); // Wrong input dim
        let mut adjacency = HashMap::new();
        adjacency.insert(0, vec![1]);

        let result = encoder.encode(
            &wrong_features,
            &adjacency,
            &[(0.0, 0.0); 4],
            &[0.0; 4],
            None,
        );

        assert!(matches!(result, Err(NeuralDecoderError::InvalidEmbeddingDimension { .. })));
    }
}
