//! # neural-trader-strategies
//!
//! Venue-agnostic strategy runtime for the RuVector Neural Trader.
//!
//! Key types:
//! - [`Intent`]: what a strategy wants to do (canonical, venue-agnostic).
//! - [`RiskGate`]: the mandatory wrapper around every `Intent` that enforces
//!   position cap, daily-loss kill, concentration, min-edge, and the
//!   live-trade env flag.
//! - [`Strategy`]: the trait strategies implement.
//! - [`ev_kelly::ExpectedValueKelly`]: first concrete strategy.
//!
//! Strategies consume [`neural_trader_core::MarketEvent`] and hold no
//! venue-specific state, so the same strategy runs in paper replay and
//! against live Kalshi (or any future venue that normalizes to
//! `MarketEvent`).

pub mod attention_scalper;
pub mod coherence_arb;
pub mod coherence_bridge;
pub mod ev_kelly;
pub mod intent;
pub mod risk;

pub use attention_scalper::{AttentionScalper, AttentionScalperConfig};
pub use coherence_arb::{CoherenceArb, CoherenceArbConfig};
pub use coherence_bridge::{
    simple_context, CoherenceChecker, CoherenceDecision, CoherenceGate, CoherenceOutcome,
    GateConfig, GateContext, RegimeLabel, ThresholdGate,
};
pub use ev_kelly::{ExpectedValueKelly, ExpectedValueKellyConfig};
pub use intent::{Action, Intent, Side};
pub use risk::{PortfolioState, Position, RejectReason, RiskConfig, RiskDecision, RiskGate};

use neural_trader_core::MarketEvent;

/// Strategy trait. Stateless strategies can be `&self`; stateful ones use
/// `&mut self`. Implementers return at most one [`Intent`] per call — if a
/// composite decision is needed, emit the primary intent and let the next
/// event drive follow-ups.
pub trait Strategy {
    fn name(&self) -> &'static str;
    fn on_event(&mut self, event: &MarketEvent) -> Option<Intent>;
}
