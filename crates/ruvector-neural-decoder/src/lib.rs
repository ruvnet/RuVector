//! # Neural Quantum Error Decoder (NQED)
//!
//! This crate implements a neural-network-based quantum error decoder that combines
//! Graph Neural Networks (GNN) with Mamba state-space models for efficient syndrome
//! decoding.
//!
//! ## Architecture
//!
//! The NQED pipeline consists of:
//! 1. **Syndrome Graph Construction**: Converts syndrome bitmaps to graph structures
//! 2. **GNN Encoding**: Multi-layer graph attention for syndrome representation
//! 3. **Mamba Decoder**: State-space model for sequential decoding
//! 4. **Feature Fusion**: Integrates min-cut structural information
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use ruvector_neural_decoder::{NeuralDecoder, DecoderConfig};
//!
//! let config = DecoderConfig::default();
//! let mut decoder = NeuralDecoder::new(config);
//!
//! // Create syndrome from measurements
//! let syndrome = vec![true, false, true, false, false];
//! let correction = decoder.decode(&syndrome)?;
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod graph;
pub mod gnn;
pub mod mamba;
pub mod fusion;

// Re-exports
pub use error::{NeuralDecoderError, Result};
pub use graph::{DetectorGraph, GraphBuilder, Node, Edge};
pub use gnn::{GNNEncoder, GNNConfig, AttentionLayer};
pub use mamba::{MambaDecoder, MambaConfig, MambaState};
pub use fusion::{FeatureFusion, FusionConfig};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Configuration for the neural decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Code distance (determines graph size)
    pub distance: usize,
    /// Embedding dimension for node features
    pub embed_dim: usize,
    /// Hidden dimension for internal representations
    pub hidden_dim: usize,
    /// Number of GNN layers
    pub num_gnn_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Mamba state dimension
    pub mamba_state_dim: usize,
    /// Whether to use min-cut fusion
    pub use_mincut_fusion: bool,
    /// Dropout rate (0.0 to 1.0)
    pub dropout: f32,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            distance: 5,
            embed_dim: 64,
            hidden_dim: 128,
            num_gnn_layers: 3,
            num_heads: 4,
            mamba_state_dim: 64,
            use_mincut_fusion: false,
            dropout: 0.1,
        }
    }
}

/// Correction output from the decoder
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Correction {
    /// X-type corrections (bit flips)
    pub x_corrections: Vec<usize>,
    /// Z-type corrections (phase flips)
    pub z_corrections: Vec<usize>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Decode time in nanoseconds
    pub decode_time_ns: u64,
}

/// Neural Quantum Error Decoder
///
/// Combines GNN-based syndrome encoding with Mamba state-space decoding.
pub struct NeuralDecoder {
    config: DecoderConfig,
    gnn: GNNEncoder,
    mamba: MambaDecoder,
    fusion: Option<FeatureFusion>,
}

impl NeuralDecoder {
    /// Create a new neural decoder with the given configuration
    pub fn new(config: DecoderConfig) -> Self {
        let gnn_config = GNNConfig {
            input_dim: 5, // Node features: [fired, row_norm, col_norm, node_type_x, node_type_z]
            embed_dim: config.embed_dim,
            hidden_dim: config.hidden_dim,
            num_layers: config.num_gnn_layers,
            num_heads: config.num_heads,
            dropout: config.dropout,
        };

        let mamba_config = MambaConfig {
            input_dim: config.hidden_dim,
            state_dim: config.mamba_state_dim,
            output_dim: config.distance * config.distance,
        };

        let fusion = if config.use_mincut_fusion {
            let fusion_config = FusionConfig {
                gnn_dim: config.hidden_dim,
                mincut_dim: 16,
                output_dim: config.hidden_dim,
                gnn_weight: 0.5,
                mincut_weight: 0.3,
                boundary_weight: 0.2,
                adaptive_weights: true,
                temperature: 1.0,
            };
            FeatureFusion::new(fusion_config).ok()
        } else {
            None
        };

        Self {
            config,
            gnn: GNNEncoder::new(gnn_config),
            mamba: MambaDecoder::new(mamba_config),
            fusion,
        }
    }

    /// Decode a syndrome bitmap and produce corrections
    pub fn decode(&mut self, syndrome: &[bool]) -> Result<Correction> {
        let start = std::time::Instant::now();

        // Build detector graph from syndrome
        let graph = GraphBuilder::from_surface_code(self.config.distance)
            .with_syndrome(syndrome)?
            .build()?;

        // GNN encoding
        let node_embeddings = self.gnn.encode(&graph)?;

        // Optional: fuse with min-cut features (requires graph with edge weights and positions)
        // For now, use the raw GNN embeddings. Full fusion requires:
        // fusion.fuse(&node_embeddings, &mincut_features, &boundary_features, confidences)
        let fused = node_embeddings;

        // Mamba decoding
        let output = self.mamba.decode(&fused)?;

        // Convert output to corrections
        let corrections = self.output_to_corrections(&output)?;

        let elapsed = start.elapsed();

        Ok(Correction {
            x_corrections: corrections.0,
            z_corrections: corrections.1,
            confidence: corrections.2,
            decode_time_ns: elapsed.as_nanos() as u64,
        })
    }

    /// Convert model output to correction indices
    fn output_to_corrections(&self, output: &Array1<f32>) -> Result<(Vec<usize>, Vec<usize>, f64)> {
        let threshold = 0.5;
        let mut x_corrections = Vec::new();

        for (i, &val) in output.iter().enumerate() {
            if val > threshold {
                x_corrections.push(i);
            }
        }

        // Compute confidence as average certainty
        let confidence = output.iter()
            .map(|&v| (v - 0.5).abs() * 2.0)
            .sum::<f32>() / output.len() as f32;

        Ok((x_corrections, Vec::new(), confidence as f64))
    }

    /// Get the decoder configuration
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// Reset the decoder state
    pub fn reset(&mut self) {
        self.mamba.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_config_default() {
        let config = DecoderConfig::default();
        assert_eq!(config.distance, 5);
        assert_eq!(config.embed_dim, 64);
        assert_eq!(config.hidden_dim, 128);
        assert!(config.dropout >= 0.0 && config.dropout <= 1.0);
    }

    #[test]
    fn test_decoder_creation() {
        let config = DecoderConfig::default();
        let decoder = NeuralDecoder::new(config);
        assert_eq!(decoder.config().distance, 5);
    }

    #[test]
    fn test_correction_default() {
        let correction = Correction::default();
        assert!(correction.x_corrections.is_empty());
        assert!(correction.z_corrections.is_empty());
        assert_eq!(correction.confidence, 0.0);
    }

    #[test]
    fn test_decoder_empty_syndrome() {
        let config = DecoderConfig {
            distance: 3,
            ..Default::default()
        };
        let mut decoder = NeuralDecoder::new(config);

        // Empty syndrome (all zeros)
        let syndrome = vec![false; 9];
        let result = decoder.decode(&syndrome);

        // Should succeed even with empty syndrome
        assert!(result.is_ok());
    }
}
