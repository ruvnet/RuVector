//! # ruvector-attention
//!
//! Attention mechanisms for ruvector, including geometric, graph, and sparse attention.
//!
//! This crate provides efficient implementations of various attention mechanisms:
//! - Scaled dot-product attention
//! - Multi-head attention with parallel processing
//! - Graph attention for GNN applications
//! - Geometric attention in hyperbolic spaces
//! - Sparse attention patterns
//!
//! ## Features
//!
//! - **SIMD Acceleration**: Optional SIMD optimizations for performance
//! - **Parallel Processing**: Rayon-based parallel head computation
//! - **WASM Support**: WebAssembly compilation support
//! - **NAPI Bindings**: Node.js bindings for JavaScript integration
//!
//! ## Example
//!
//! ```rust
//! use ruvector_attention::{
//!     attention::ScaledDotProductAttention,
//!     traits::Attention,
//! };
//!
//! // Create scaled dot-product attention
//! let attention = ScaledDotProductAttention::new(512);
//!
//! // Prepare inputs
//! let query = vec![1.0; 512];
//! let keys = vec![vec![0.5; 512], vec![0.3; 512]];
//! let values = vec![vec![1.0; 512], vec![2.0; 512]];
//!
//! let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
//! let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
//!
//! // Compute attention
//! let output = attention.compute(&query, &keys_refs, &values_refs).unwrap();
//! assert_eq!(output.len(), 512);
//! ```

pub mod attention;
pub mod config;
pub mod error;
pub mod graph;
pub mod hyperbolic;
pub mod moe;
pub mod sdk;
pub mod sparse;
pub mod training;
pub mod traits;
pub mod utils;

// Re-export main types
pub use attention::{MultiHeadAttention, ScaledDotProductAttention};
pub use config::{AttentionConfig, GraphAttentionConfig, SparseAttentionConfig};
pub use error::{AttentionError, AttentionResult};
pub use hyperbolic::{
    exp_map, log_map, mobius_add, poincare_distance, project_to_ball, HyperbolicAttention,
    HyperbolicAttentionConfig, MixedCurvatureAttention, MixedCurvatureConfig,
};
pub use traits::{
    Attention, EdgeInfo, GeometricAttention, Gradients, GraphAttention, SparseAttention,
    SparseMask, TrainableAttention,
};

// Sparse attention exports
pub use sparse::{
    AttentionMask, FlashAttention, LinearAttention, LocalGlobalAttention, SparseMaskBuilder,
};

// MoE exports
pub use moe::{
    Expert, ExpertType, HyperbolicExpert, LearnedRouter, LinearExpert, MoEAttention, MoEConfig,
    Router, StandardExpert, TopKRouting,
};

// Graph attention exports
pub use graph::{
    DualSpaceAttention, DualSpaceConfig, EdgeFeaturedAttention, EdgeFeaturedConfig, GraphRoPE,
    RoPEConfig,
};

// Training exports
pub use training::{
    Adam, AdamW, CurriculumScheduler, CurriculumStage, DecayType, HardNegativeMiner, InfoNCELoss,
    LocalContrastiveLoss, Loss, MiningStrategy, NegativeMiner, Optimizer, Reduction,
    SpectralRegularization, TemperatureAnnealing, SGD,
};

// SDK exports
pub use sdk::{presets, AttentionBuilder, AttentionPipeline};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_basic_attention_workflow() {
        let config = AttentionConfig::builder()
            .dim(64)
            .num_heads(4)
            .build()
            .unwrap();

        assert_eq!(config.dim, 64);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.head_dim(), 16);
    }
}
