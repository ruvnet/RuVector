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
//! Architecturally faithful + dimensionally correct. Real numerical
//! weight-parity against the published safetensors is **not** claimed here —
//! the modules load via [`candle_nn::VarBuilder`] so real weights drop in
//! later, but the shape tests run on `VarBuilder::zeros`/`randn`.

pub mod config;

pub use config::{TimesfmConfig, QUANTILES};

#[cfg(feature = "candle")]
pub mod model;

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
