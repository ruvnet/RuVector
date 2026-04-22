//! Expected-Value Kelly strategy.
//!
//! Given a model probability `p` for the YES side of a binary contract
//! and a current market price `m` in cents (0..=100), the canonical
//! fractional Kelly sizing for a YES bet is:
//!
//!     edge   = p - m / 100
//!     kelly  = edge / (1 - m / 100)
//!
//! The strategy multiplies `kelly` by a conservative `kelly_fraction`
//! (default 0.25) and by `intent.confidence`, then converts the resulting
//! fraction of cash into a contract count at the current ask.
//!
//! The priors are supplied externally (e.g. by a ruvllm-based news model);
//! this crate simply consumes them via [`ExpectedValueKelly::set_prior`].
//!
//! Strategies are venue-agnostic — they emit [`Intent`]s; the RiskGate
//! plus a venue adapter decide whether to send an order.

use std::collections::HashMap;

use neural_trader_core::{EventType, MarketEvent};

use crate::intent::{Action, Intent, Side};
use crate::Strategy;

/// Per-symbol state tracked by the strategy.
#[derive(Debug, Clone)]
struct SymbolState {
    /// Latest YES-side mid in cents.
    latest_mid_cents: Option<i64>,
    /// External YES-side probability prior, `[0, 1]`.
    prior: Option<f64>,
    /// Last emission sequence — throttle to at most one intent per event kind.
    last_emit_seq: u64,
}

impl Default for SymbolState {
    fn default() -> Self {
        Self {
            latest_mid_cents: None,
            prior: None,
            last_emit_seq: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExpectedValueKellyConfig {
    /// Kelly fraction (0, 1]. Quarter-Kelly (0.25) is a common default.
    pub kelly_fraction: f64,
    /// Total bankroll in cents used for sizing.
    pub bankroll_cents: i64,
    /// Minimum edge in basis points to emit an intent. Defensive — the
    /// RiskGate enforces the final bar, but we avoid spamming intents we
    /// know will be rejected.
    pub min_edge_bps: i64,
    /// Attribution name.
    pub strategy_name: &'static str,
}

impl Default for ExpectedValueKellyConfig {
    fn default() -> Self {
        Self {
            kelly_fraction: 0.25,
            bankroll_cents: 100_000,
            min_edge_bps: 100,
            strategy_name: "ev-kelly",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExpectedValueKelly {
    pub config: ExpectedValueKellyConfig,
    symbols: HashMap<u32, SymbolState>,
}

impl ExpectedValueKelly {
    pub fn new(config: ExpectedValueKellyConfig) -> Self {
        Self {
            config,
            symbols: HashMap::new(),
        }
    }

    /// Install or update the probability prior for a symbol. Priors are
    /// clamped into `[0.01, 0.99]` to avoid divide-by-zero at the limits.
    pub fn set_prior(&mut self, symbol_id: u32, prior: f64) {
        let clamped = prior.clamp(0.01, 0.99);
        self.symbols.entry(symbol_id).or_default().prior = Some(clamped);
    }

    fn handle_snapshot(&mut self, event: &MarketEvent) -> Option<Intent> {
        // Use BookSnapshot YES-bid as the mid proxy. The normalizer already
        // emits separate events per level; we track the most recent bid.
        let cents = fp_to_cents(event.price_fp);
        let entry = self.symbols.entry(event.symbol_id).or_default();
        entry.latest_mid_cents = Some(cents);
        self.maybe_emit(event.symbol_id, event.seq)
    }

    fn handle_ticker_or_trade(&mut self, event: &MarketEvent) -> Option<Intent> {
        let cents = fp_to_cents(event.price_fp);
        if cents <= 0 {
            return None;
        }
        let entry = self.symbols.entry(event.symbol_id).or_default();
        entry.latest_mid_cents = Some(cents);
        self.maybe_emit(event.symbol_id, event.seq)
    }

    fn maybe_emit(&mut self, symbol_id: u32, seq: u64) -> Option<Intent> {
        let state = self.symbols.get(&symbol_id)?;
        let mid = state.latest_mid_cents?;
        let prior = state.prior?;
        if mid <= 0 || mid >= 100 {
            return None;
        }
        let m = mid as f64 / 100.0;
        let edge = prior - m;
        let edge_bps = (edge * 10_000.0).round() as i64;
        if edge_bps < self.config.min_edge_bps {
            return None;
        }
        // Quarter-Kelly for YES side: fraction of bankroll = k × edge / (1 - m).
        let kelly = (self.config.kelly_fraction * edge / (1.0 - m)).clamp(0.0, 1.0);
        let alloc_cents = (self.config.bankroll_cents as f64 * kelly) as i64;
        let contracts = alloc_cents / mid.max(1);
        if contracts <= 0 {
            return None;
        }
        let entry = self.symbols.get_mut(&symbol_id)?;
        entry.last_emit_seq = seq;
        Some(Intent {
            symbol_id,
            side: Side::Yes,
            action: Action::Buy,
            limit_price_cents: mid,
            quantity: contracts,
            edge_bps,
            confidence: prior,
            strategy: self.config.strategy_name,
        })
    }
}

impl Strategy for ExpectedValueKelly {
    fn name(&self) -> &'static str {
        self.config.strategy_name
    }

    fn on_event(&mut self, event: &MarketEvent) -> Option<Intent> {
        match event.event_type {
            EventType::Trade | EventType::VenueStatus => self.handle_ticker_or_trade(event),
            EventType::BookSnapshot => self.handle_snapshot(event),
            _ => None,
        }
    }
}

/// The canonical price-fp scale used by `ruvector-kalshi` is `cents × 1e6`.
/// Reverse that here. If other venues use a different scale and want to
/// share this strategy, they should normalize into the same scale first.
fn fp_to_cents(fp: i64) -> i64 {
    fp / 1_000_000
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_trader_core::{EventType, MarketEvent, Side as NtSide};

    fn snapshot(sym: u32, price_cents: i64, seq: u64) -> MarketEvent {
        MarketEvent {
            event_id: [0u8; 16],
            ts_exchange_ns: 0,
            ts_ingest_ns: 0,
            venue_id: 1001,
            symbol_id: sym,
            event_type: EventType::BookSnapshot,
            side: Some(NtSide::Bid),
            price_fp: price_cents * 1_000_000,
            qty_fp: 1_000_000,
            order_id_hash: None,
            participant_id_hash: None,
            flags: 0,
            seq,
        }
    }

    #[test]
    fn no_prior_no_intent() {
        let mut s = ExpectedValueKelly::new(ExpectedValueKellyConfig::default());
        assert!(s.on_event(&snapshot(1, 24, 0)).is_none());
    }

    #[test]
    fn fair_price_no_edge() {
        let mut s = ExpectedValueKelly::new(ExpectedValueKellyConfig::default());
        s.set_prior(1, 0.25);
        // Mid exactly at prior → edge ~ 0 bps → no intent.
        let intent = s.on_event(&snapshot(1, 25, 0));
        assert!(intent.is_none(), "mid==prior should produce no intent");
    }

    #[test]
    fn positive_edge_sizes_under_bankroll() {
        let mut s = ExpectedValueKelly::new(ExpectedValueKellyConfig {
            kelly_fraction: 0.25,
            bankroll_cents: 100_000,
            min_edge_bps: 100,
            strategy_name: "test",
        });
        s.set_prior(1, 0.40);
        let intent = s.on_event(&snapshot(1, 24, 0)).expect("should emit");
        assert_eq!(intent.symbol_id, 1);
        assert!(matches!(intent.side, Side::Yes));
        assert_eq!(intent.limit_price_cents, 24);
        assert!(intent.quantity > 0);
        // Notional must not exceed bankroll.
        assert!(intent.notional_cents() <= 100_000);
        // Edge = (0.40 - 0.24) × 10_000 = 1600 bps.
        assert_eq!(intent.edge_bps, 1600);
    }

    #[test]
    fn extreme_prior_is_clamped() {
        let mut s = ExpectedValueKelly::new(ExpectedValueKellyConfig::default());
        s.set_prior(1, 1.5); // nonsensical; clamp to 0.99
        let intent = s.on_event(&snapshot(1, 50, 0)).expect("should still emit");
        assert!((intent.confidence - 0.99).abs() < 1e-9);
    }

    #[test]
    fn mid_at_boundary_rejected() {
        let mut s = ExpectedValueKelly::new(ExpectedValueKellyConfig::default());
        s.set_prior(1, 0.80);
        // mid = 0 → boundary
        assert!(s.on_event(&snapshot(1, 0, 0)).is_none());
        // mid = 100 → boundary
        assert!(s.on_event(&snapshot(1, 100, 1)).is_none());
    }
}
