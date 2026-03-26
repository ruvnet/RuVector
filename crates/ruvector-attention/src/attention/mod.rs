//! Attention mechanism implementations.
//!
//! This module provides concrete implementations of various attention mechanisms
//! including scaled dot-product attention and multi-head attention.

pub mod flash;
pub mod kv_cache;
pub mod mla;
pub mod multi_head;
pub mod scaled_dot_product;
pub mod speculative;
pub mod ssm;

pub use flash::{
    causal_block_mask, FlashAttention3, FlashConfig, FlashOutput, IOStats, RingAttention,
    RingDeviceOutput,
};
pub use mla::{MLACache, MLAConfig, MLALayer, MemoryComparison};
pub use multi_head::MultiHeadAttention;
pub use scaled_dot_product::ScaledDotProductAttention;
pub use speculative::{
    medusa_decode, theoretical_speedup, AcceptedTokens, DecodingStats, DraftModel, MedusaHead,
    MedusaResult, SimpleDraftModel, SimpleMedusaHead, SimpleTargetModel, SpeculativeConfig,
    SpeculativeDecoder, TargetModel, TokenId,
};
pub use ssm::{
    HybridBlock, HybridConfig, LayerKind, MambaBlock, SSMConfig, SSMState, SelectiveSSM,
};
