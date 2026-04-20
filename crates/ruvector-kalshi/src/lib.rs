//! # ruvector-kalshi
//!
//! Kalshi exchange integration for the RuVector Neural Trader.
//!
//! See ADR-151 for design. This crate provides:
//! - RSA-PSS-SHA256 request signing against Kalshi's REST API
//! - Typed DTOs for Kalshi market/event/order/fill payloads
//! - Normalization from Kalshi payloads into
//!   [`neural_trader_core::MarketEvent`] so downstream coherence, attention,
//!   and replay pipelines work unchanged.
//! - A REST client scaffold (live calls are gated behind a runtime flag)
//! - Secret loading from Google Cloud Secret Manager or a local PEM file

pub mod auth;
pub mod models;
pub mod normalize;
pub mod rate_limit;
pub mod rest;
pub mod secrets;
pub mod strategy_adapter;
pub mod ws;

use thiserror::Error;

/// Venue identifier for Kalshi in [`neural_trader_core::MarketEvent::venue_id`].
pub const KALSHI_VENUE_ID: u16 = 1001;

/// Default Kalshi REST base URL (production).
pub const KALSHI_API_URL: &str = "https://trading-api.kalshi.com/trade-api/v2";

/// Default Kalshi WebSocket URL (production).
pub const KALSHI_WS_URL: &str = "wss://trading-api.kalshi.com/trade-api/ws/v2";

/// Kalshi fixed-point scale for price. Kalshi quotes in cents 0..=100;
/// we upscale into the canonical `price_fp` scale (value × 1e8) so
/// that a 24¢ YES becomes 24_000_000.
pub const KALSHI_PRICE_FP_SCALE: i64 = 1_000_000;

/// Crate-wide error type.
#[derive(Debug, Error)]
pub enum KalshiError {
    #[error("invalid PEM key material: {0}")]
    InvalidPem(String),

    #[error("signing failed: {0}")]
    Signing(String),

    #[error("secret source error: {0}")]
    Secret(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("decode error: {0}")]
    Decode(#[from] serde_json::Error),

    #[error("api error {status}: {body}")]
    Api { status: u16, body: String },

    #[error("normalization error: {0}")]
    Normalize(String),
}

pub type Result<T> = std::result::Result<T, KalshiError>;
