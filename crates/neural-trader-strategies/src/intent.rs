//! Canonical strategy intent — venue-agnostic.

use serde::{Deserialize, Serialize};

/// Buy (open) or sell (close) side of an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Buy,
    Sell,
}

/// For binary event markets like Kalshi: which side of the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Yes,
    No,
}

/// A strategy's desired trade, before the risk gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Intent {
    /// Canonical symbol id (`neural_trader_core::MarketEvent::symbol_id`).
    pub symbol_id: u32,
    /// Which side of the binary contract.
    pub side: Side,
    /// Buy to open, sell to close.
    pub action: Action,
    /// Limit price in Kalshi cents (0..=100). Strategies quote in cents
    /// because event markets settle in cents; the adapter upscales into
    /// `price_fp` if needed downstream.
    pub limit_price_cents: i64,
    /// Number of contracts.
    pub quantity: i64,
    /// Expected edge in basis points over the mid (1 bp = 0.01%).
    pub edge_bps: i64,
    /// Model confidence in `[0, 1]`. Used by the risk gate as a multiplier
    /// on position sizing (higher conviction → larger fraction).
    pub confidence: f64,
    /// Strategy name for attribution / audit.
    pub strategy: &'static str,
}

impl Intent {
    /// Notional in Kalshi cents (`price × quantity`).
    pub fn notional_cents(&self) -> i64 {
        self.limit_price_cents.saturating_mul(self.quantity)
    }
}
