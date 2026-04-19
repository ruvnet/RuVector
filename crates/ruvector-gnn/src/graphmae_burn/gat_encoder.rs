//! GATEncoderBurn: Stacked GAT layers via `.fold()`.

use super::gat_layer::GATLayerBurn;
use burn::config::Config;
use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::Tensor;

// ============================================================================
// Struct Definitions (Phase 2/3 implementation)
// ============================================================================

/// GATEncoderBurn config.
///
/// **Phase 3**: Populate with layer count and per-layer config.
#[derive(Debug, Config)]
pub struct GATEncoderBurnConfig {
    pub num_layers: usize,
    pub hidden_channels: usize,
    pub out_channels: usize,
    pub heads: usize,
    pub dropout: f32,
}

/// GATEncoderBurn: Stack of GATLayerBurn layers with residual connections.
///
/// **Phase 3**: Replace `todo!()` with actual layer stacking.
#[derive(Debug, Module)]
pub struct GATEncoderBurn<B: Backend> {
    // TODO(Phase 3): Stack of GATLayerBurn modules
    // pub layers: burn::nn::ModuleContainer,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> GATEncoderBurn<B> {
    /// Create encoder from config.
    ///
    /// **Phase 3**: Build layers via `.fold()` or explicit Vec.
    pub fn new(_config: &GATEncoderBurnConfig) -> Self {
        todo!("Phase 3: Initialize stacked GATLayerBurn layers")
    }

    /// Encode graph features.
    ///
    /// **Phase 3**: Pass through all layers with residual connections.
    pub fn encode(
        &self,
        _features: &[Vec<f32>],
        _adj: &[Vec<usize>],
    ) -> Vec<Vec<f32>> {
        todo!("Phase 3: Forward through all layers, return Vec<Vec<f32>>")
    }

    /// Encode from burn tensor (internal use).
    ///
    /// **Phase 3**: Return burn tensor directly for training loop.
    pub fn encode_t(
        &self,
        _features: Tensor<B, 2>,
        _adj: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        todo!("Phase 3: Forward pass, return burn tensor")
    }
}

// ============================================================================
// Tests (Phase 3)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Phase 3: Implement burn backend test"]
    fn test_encoder_output_shape() {
        todo!("Phase 3: Verify output matches (num_nodes, out_channels)")
    }

    #[test]
    #[ignore = "Phase 3: Implement burn backend test"]
    fn test_encoder_stack_layers() {
        todo!("Phase 3: Verify multi-layer stacking with residual connections")
    }
}