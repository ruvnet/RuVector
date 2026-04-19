//! Tensor conversion between Vec<Vec<f32>> and burn::Tensor.
//!
//! **Phase 2**: Actual conversion deferred until tensor shapes are known.

use burn::prelude::Backend;
use burn::tensor::Tensor;

// ============================================================================
// Vec<Vec<f32>> <-> burn::Tensor (Phase 2)
// ============================================================================

/// Convert `Vec<Vec<f32>>` to burn `Tensor<Rank2>`.
/// 
/// **Phase 2**: Assert consistent inner dimension (num_nodes, in_features).
pub fn to_burn_tensor<B: Backend>(
    _features: &[Vec<f32>],
) -> Tensor<B, 2> {
    todo!("Phase 2: Convert [[f32]] -> Tensor<B, 2> with shape (num_nodes, in_features)")
}

/// Convert burn `Tensor<Rank2>` to `Vec<Vec<f32>>`.
/// 
/// **Phase 2**: Detach from compute graph, clone to host.
pub fn from_burn_tensor<B: Backend>(
    _tensor: Tensor<B, 2>,
) -> Vec<Vec<f32>> {
    todo!("Phase 2: Tensor -> Vec<Vec<f32>>, detach + to_data")
}

// ============================================================================
// Adjacency matrix conversions (Phase 2)
// ============================================================================

/// Convert adjacency list `[Vec<usize>]` to burn `Tensor<Rank2>` (sparse optional).
/// 
/// **Phase 2**: Represent as (N, N) mask tensor, or use sparse representation.
pub fn adj_to_burn_tensor<B: Backend>(
    _adj: &[Vec<usize>],
    _num_nodes: usize,
) -> Tensor<B, 2> {
    todo!("Phase 2: Convert adjacency list to (N, N) binary or weighted tensor")
}

// ============================================================================
// Tests (Phase 2)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Phase 2: Requires burn backend"]
    fn test_to_burn_tensor_shape() {
        todo!("Phase 2: Verify (num_nodes, feature_dim) shape is correct")
    }

    #[test]
    #[ignore = "Phase 2: Requires burn backend"]
    fn test_from_burn_tensor_roundtrip() {
        todo!("Phase 2: Verify Vec<Vec<f32>> -> Tensor -> Vec<Vec<f32>> preserves values")
    }
}