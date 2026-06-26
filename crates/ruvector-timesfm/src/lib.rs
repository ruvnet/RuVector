//! # ruvector-timesfm
//!
//! RuVector-facing integration for the [`timesfm`] TimesFM 1.0 200M time-series
//! foundation model. The base `timesfm` crate is a faithful, parity-validated
//! candle port of the model; this crate wraps it in the three things RuVector
//! and ruflo actually call:
//!
//! 1. [`Forecaster`] — a one-call quantile forecaster: load weights once,
//!    `forecast(series, horizon)` → point forecast **plus calibrated p10..p90
//!    bands** ([`Forecast`]).
//! 2. [`anomaly`] — forecast-band anomaly detection: forecast the expected
//!    window, then flag observed points that fall outside their p10/p90 band
//!    (host/vector-db telemetry: disk-fill, GPU memory, query load).
//! 3. [`sweep`] — TimesFM-driven early stopping for optimization sweeps
//!    (ADR-191 §2): an [`sweep::EarlyStopper`] that wraps
//!    [`timesfm::prune::decide_prune`] with `min_history` + a confidence gate so
//!    ruflo/Darwin runs can kill doomed genomes early.
//!
//! ## Feature gating
//!
//! The numeric path lives behind the **`candle`** feature (and `cuda`/`metal`
//! which imply it), mirroring `timesfm`. Without it, only the plain data types
//! ([`Forecast`], [`anomaly::AnomalyReport`], [`sweep::EarlyStopper`]) compile —
//! the inference methods are gated. This keeps a stock
//! `cargo build --workspace` light.
//!
//! ```ignore
//! use ruvector_timesfm::Forecaster;
//! let f = Forecaster::load("/path/timesfm.safetensors", timesfm::select_device()?)?;
//! let forecast = f.forecast(&history, 64)?;
//! let (lo, hi) = (forecast.p10(), forecast.p90());
//! ```

pub mod anomaly;
mod forecast_types;
pub mod rebuild;
pub mod sweep;

pub use forecast_types::Forecast;

#[cfg(feature = "candle")]
mod forecaster;
#[cfg(feature = "candle")]
pub use forecaster::{Forecaster, Quant};

// Re-export the underlying model crate so callers can reach config/prune types
// (and `select_device`) without a second dependency.
pub use timesfm;

/// Crate error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Input validation failure (empty series, zero horizon, …).
    #[error("invalid input: {0}")]
    Invalid(String),

    /// Error bubbling up from the underlying `timesfm` crate.
    #[error("timesfm error: {0}")]
    Timesfm(#[from] timesfm::Error),

    /// candle tensor error (numeric path only).
    #[cfg(feature = "candle")]
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// I/O error (weight loading, CLI).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON (de)serialization error (CLI / serde boundary).
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;
