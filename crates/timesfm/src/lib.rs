//! # timesfm
//!
//! Rust/candle inference path for **TimesFM 1.0 200M** — Google's decoder-only,
//! patched, causal time-series Transformer
//! ([google-research/timesfm](https://github.com/google-research/timesfm),
//! HF model card `google/timesfm-1.0-200m`).
//!
//! This crate is architecturally faithful to the reference
//! `pytorch_patched_decoder.py`, including the non-obvious deviations from a
//! vanilla LLM transformer (post-norm-ish residual flow, per-dim learnable
//! query scaling, a `LayerNorm` *inside* the MLP, `ResidualBlock` patch
//! embed/output, an additive frequency embedding, and RevIN-style per-series
//! instance normalization).
//!
//! ## Feature gating
//!
//! The numeric path lives behind the **`candle`** feature so a stock
//! `cargo build --workspace` stays light (mirroring `ruvector-hailo`). The
//! [`config`] module is always available.
//!
//! ```ignore
//! cargo build  -p timesfm --features candle
//! cargo test   -p timesfm --features candle
//! ```
//!
//! ## Status
//!
//! Architecturally faithful, dimensionally correct, and **weight-parity
//! validated** against the official PyTorch reference
//! (`google/timesfm-1.0-200m-pytorch`'s `torch_model.ckpt` driven through the
//! reference `PatchedTimeSeriesDecoder`). On a deterministic 512-point series
//! with a 128-step horizon, the candle forecast reproduces the reference point
//! forecast to **max-abs-diff 8.58e-6 / MAE 3.25e-6 / rel-error 5.83e-7**
//! (f32 CPU, 2026-06-24) — i.e. agreement at the f32 accumulation-order floor
//! across the full 20-layer stack and autoregressive decode.
//!
//! Reproduce with `tests/parity.rs` (gated on the converted artifacts) or
//! `examples/parity.rs`. The conversion bridge is `scripts/convert_weights.py`
//! (PyTorch state_dict -> candle VarBuilder safetensors; all 253 params map
//! 1:1 with zero unmapped/missing keys).

pub mod config;

pub use config::{TimesfmConfig, QUANTILES};

pub use prune::PruneDecision;

#[cfg(feature = "candle")]
pub mod model;

/// Predictive pruning: forecast an optimization curve's plateau from its first
/// K points and decide PRUNE vs CONTINUE (ADR-191 §2). The numeric path needs
/// `candle`, but the [`prune::PruneDecision`] type is always available.
pub mod prune;

#[cfg(feature = "candle")]
pub use model::{
    ForecastOutput, PatchedTimeSeriesDecoder, PositionalEmbedding, ResidualBlock, StackedDecoder,
    TimesFMAttention, TimesFMDecoderLayer, TransformerMLP,
};

/// Crate-level error type. Wraps candle errors when the `candle` feature is on.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    Config(String),

    #[cfg(feature = "candle")]
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, Error>;
