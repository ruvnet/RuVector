//! Numeric gradient checking for GATLayerBurn.
//!
//! **Phase 2**: Required verification before claiming backward correctness.

use burn::prelude::Backend;
use burn::tensor::Tensor;

// ============================================================================
// Core gradcheck function (Phase 2)
// ============================================================================

/// Run finite-difference gradcheck on GATLayerBurn.
/// 
/// **Phase 2**: Compare analytical gradients (burn autodiff) vs numerical gradients.
/// 
/// Tolerance: 1e-5 relative error.
pub fn gradcheck_layer<B: Backend>(
    _layer: &super::gat_layer::GATLayerBurn<B>,
    _features: Tensor<B, 2>,
    _adj: Tensor<B, 2>,
    _epsilon: f32,
    _atol: f32,
) -> bool {
    todo!("Phase 2: Finite-difference gradcheck — compute numerical gradient via (f(x+h) - f(x-h)) / 2h, compare with .backward() gradient")
}

// ============================================================================
// Numerical gradient helpers (Phase 2)
// ============================================================================

/// Compute numerical gradient for a single parameter via finite differences.
/// 
/// **Phase 2**: Implement (f(x+h) - f(x-h)) / (2*h) for each parameter.
pub fn numerical_gradient<B: Backend>(
    _layer: &super::gat_layer::GATLayerBurn<B>,
    _features: Tensor<B, 2>,
    _adj: Tensor<B, 2>,
    _param_name: &str,
    _epsilon: f32,
) -> Vec<f32> {
    todo!("Phase 2: Return flat gradient vector for named parameter")
}

// ============================================================================
// Verification (Phase 2)
// ============================================================================

/// Verify all parameters pass gradcheck.
/// 
/// **Phase 2**: Run over all Linear layer parameters, assert max relative error < atol.
pub fn verify_gradients<B: Backend>(
    _layer: &super::gat_layer::GATLayerBurn<B>,
    _features: Tensor<B, 2>,
    _adj: Tensor<B, 2>,
) -> Result<(), String> {
    todo!("Phase 2: Assert all parameter gradients within 1e-5 relative error")
}

// ============================================================================
// Tests (Phase 2)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Phase 2: Requires burn backend"]
    fn test_gradcheck_single_layer() {
        todo!("Phase 2: Create small GATLayerBurn, run gradcheck, assert pass")
    }

    #[test]
    #[ignore = "Phase 2: Requires burn backend"]
    fn test_gradcheck_attention_scores() {
        todo!("Phase 2: Verify attention_score gradients numerically")
    }

    #[test]
    #[ignore = "Phase 2: Requires burn backend"]
    fn test_gradcheck_layernorm() {
        todo!("Phase 2: Verify LayerNorm gradients numerically")
    }
}