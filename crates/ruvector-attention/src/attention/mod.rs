//! Attention mechanism implementations.
//!
//! This module provides concrete implementations of various attention mechanisms
//! including scaled dot-product attention and multi-head attention.

pub mod kv_cache;
pub mod mla;
pub mod multi_head;
pub mod scaled_dot_product;
pub mod ssm;

pub use mla::{MLACache, MLAConfig, MLALayer, MemoryComparison};
pub use multi_head::MultiHeadAttention;
pub use scaled_dot_product::ScaledDotProductAttention;
pub use ssm::{
    HybridBlock, HybridConfig, LayerKind, MambaBlock, SSMConfig, SSMState, SelectiveSSM,
};
