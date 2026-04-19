//! GATLayerBurn: Single Graph Attention Layer with burn autodiff.

use burn::nn::Linear;
use burn::module::Module;
use burn::config::Config;
use burn::prelude::{Backend, Tensor};

// ============================================================================
// Struct Definitions (Phase 2 implementation)
// ============================================================================

/// GATLayerBurn config.
/// 
/// **Phase 2**: Populate fields from GATLayerConfig.
#[derive(Debug, Config)]
pub struct GATLayerBurnConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub heads: usize,
    pub dropout: f32,
    pub negative_slope: f32,
}

/// GATLayerBurn: Single GAT layer with learnable projections per head.
/// 
/// **Phase 2**: Replace `todo!()` with actual burn-nn modules.
#[derive(Debug, Module)]
pub struct GATLayerBurn<B: Backend> {
    // TODO(Phase 2): Replace with burn-nn Linear for Q/K/V projections
    // pub q_proj: Linear<B>,
    // pub k_proj: Linear<B>,
    // pub v_proj: Linear<B>,
    // pub concat_proj: Linear<B>,
    // pub leaky_relu: burn::nn::LeakyReLU,
    // pub dropout: burn::nn::Dropout,
    // pub attn_dropout: burn::nn::Dropout,
    // pub layernorm: burn::nn::LayerNorm<B>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> GATLayerBurn<B> {
    /// Create a new GATLayerBurn from config.
    /// 
    /// **Phase 2**: Initialize burn-nn modules here.
    pub fn new(_config: &GATLayerBurnConfig) -> Self {
        todo!("Phase 2: Initialize burn-nn::Linear modules (Q/K/V projections, concat_proj, LeakyReLU, Dropout, LayerNorm)")
    }

    /// Forward pass: multi-head graph attention.
    ///
    /// **Phase 2**: Implement:
    ///   1. Q/K/V from input features via Linear projections
    ///   2. Split into `heads` for multi-head attention
    ///   3. Compute attention scores: LeakyReLU(Q * K^T / sqrt(d_k))
    ///   4. Mask zeros in adj (no edge) with -inf before softmax
    ///   5. softmax over neighbors
    ///   6. Weighted sum: A * V
    ///   7. Concat heads -> Linear
    ///   8. Residual + LayerNorm
    pub fn forward(
        &self,
        _features: Tensor<B, 2>,
        _adj: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        todo!("Phase 2: Multi-head attention forward pass")
    }

    /// Compute attention scores (Q * K^T / sqrt(d_k)) — separated for gradcheck.
    /// 
    /// **Phase 2**: Return raw attention scores for numerical gradient verification.
    pub fn attention_scores(
        &self,
        _features: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        todo!("Phase 2: Return (num_heads, N, N) attention score tensor")
    }
}

// ============================================================================
// Tests (Phase 2)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Phase 2: Implement burn backend test"]
    fn test_gat_layer_forward_shapes() {
        todo!("Phase 2: Verify output shape = (num_nodes, out_features)")
    }

    #[test]
    #[ignore = "Phase 2: Implement burn backend test"]
    fn test_gat_layer_multi_head_split() {
        todo!("Phase 2: Verify head splitting produces correct num_heads dimension")
    }
}