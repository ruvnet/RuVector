//! Cross-market coherence arbitrage.
//!
//! When two Kalshi contracts represent equivalent or near-equivalent
//! outcomes (e.g. "Fed holds rates in Dec" on one expiry vs. another),
//! their YES prices should track each other. When the market price
//! diverges by more than `min_divergence_bps` from the reference symbol,
//! buy the cheaper side.
//!
//! This is a canonical-MarketEvent-only strategy: it consumes nothing
//! venue-specific and makes no assumptions beyond the YES-side mid being
//! carried on `MarketEvent`. The operator configures a small list of
//! `(reference, mirror)` pairs at startup.
//!
//! Not a true ML coherence model — that requires the
//! `neural-trader-coherence` crate, which focuses on the neural trader's
//! own memory gates rather than market correlation. This strategy is
//! deliberately simple and testable so it can run the moment tickers are
//! configured; the ML hook remains open for later work.

use std::collections::HashMap;

use neural_trader_core::{EventType, MarketEvent};

use crate::intent::{Action, Intent, Side};
use crate::Strategy;

#[derive(Debug, Clone)]
pub struct CoherenceArbConfig {
    /// Pairs of `(reference, mirror)` symbol ids. The mirror is bought
    /// (YES) if it trades below the reference by at least
    /// `min_divergence_bps`.
    pub pairs: Vec<(u32, u32)>,
    /// Divergence threshold in basis points (1% = 100 bps).
    pub min_divergence_bps: i64,
    /// Fractional Kelly sizing applied to bankroll.
    pub kelly_fraction: f64,
    pub bankroll_cents: i64,
    pub strategy_name: &'static str,
}

impl Default for CoherenceArbConfig {
    fn default() -> Self {
        Self {
            pairs: Vec::new(),
            min_divergence_bps: 300,
            kelly_fraction: 0.25,
            bankroll_cents: 100_000,
            strategy_name: "coherence-arb",
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CoherenceArb {
    pub config: CoherenceArbConfig,
    latest_mid_cents: HashMap<u32, i64>,
}

impl CoherenceArb {
    pub fn new(config: CoherenceArbConfig) -> Self {
        Self {
            config,
            latest_mid_cents: HashMap::new(),
        }
    }

    fn update_mid(&mut self, event: &MarketEvent) {
        let cents = event.price_fp / 1_000_000;
        if cents > 0 && cents < 100 {
            self.latest_mid_cents.insert(event.symbol_id, cents);
        }
    }

    fn try_arb_for(&self, mirror_sym: u32) -> Option<Intent> {
        // Find the pair this symbol participates in.
        let &(reference, mirror) = self.config.pairs.iter().find(|(_, m)| *m == mirror_sym)?;
        let ref_cents = *self.latest_mid_cents.get(&reference)?;
        let mirror_cents = *self.latest_mid_cents.get(&mirror)?;
        let divergence_cents = ref_cents - mirror_cents;
        let divergence_bps = divergence_cents * 100;
        if divergence_bps < self.config.min_divergence_bps {
            return None;
        }
        // Quarter-Kelly on the divergence: fraction = k × (div / (100-mirror)).
        let denom = (100 - mirror_cents).max(1);
        let kelly = self.config.kelly_fraction * divergence_cents as f64 / denom as f64;
        let kelly = kelly.clamp(0.0, 1.0);
        let alloc_cents = (self.config.bankroll_cents as f64 * kelly) as i64;
        let quantity = alloc_cents / mirror_cents.max(1);
        if quantity <= 0 {
            return None;
        }
        Some(Intent {
            symbol_id: mirror,
            side: Side::Yes,
            action: Action::Buy,
            limit_price_cents: mirror_cents,
            quantity,
            edge_bps: divergence_bps,
            confidence: (divergence_cents as f64 / 100.0).clamp(0.0, 1.0),
            strategy: self.config.strategy_name,
        })
    }
}

impl Strategy for CoherenceArb {
    fn name(&self) -> &'static str {
        self.config.strategy_name
    }

    fn on_event(&mut self, event: &MarketEvent) -> Option<Intent> {
        if !matches!(
            event.event_type,
            EventType::Trade | EventType::VenueStatus | EventType::BookSnapshot
        ) {
            return None;
        }
        self.update_mid(event);
        self.try_arb_for(event.symbol_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_trader_core::{EventType, MarketEvent, Side as NtSide};

    fn ticker_event(sym: u32, cents: i64, seq: u64) -> MarketEvent {
        MarketEvent {
            event_id: [0u8; 16],
            ts_exchange_ns: 0,
            ts_ingest_ns: 0,
            venue_id: 1001,
            symbol_id: sym,
            event_type: EventType::VenueStatus,
            side: Some(NtSide::Bid),
            price_fp: cents * 1_000_000,
            qty_fp: 0,
            order_id_hash: None,
            participant_id_hash: None,
            flags: 0,
            seq,
        }
    }

    #[test]
    fn converged_pair_no_intent() {
        let mut s = CoherenceArb::new(CoherenceArbConfig {
            pairs: vec![(1, 2)],
            min_divergence_bps: 300,
            ..Default::default()
        });
        assert!(s.on_event(&ticker_event(1, 50, 0)).is_none());
        let intent = s.on_event(&ticker_event(2, 50, 1));
        assert!(intent.is_none(), "converged prices should not trigger");
    }

    #[test]
    fn diverged_pair_emits_intent_on_cheaper() {
        let mut s = CoherenceArb::new(CoherenceArbConfig {
            pairs: vec![(1, 2)],
            min_divergence_bps: 300,
            bankroll_cents: 100_000,
            kelly_fraction: 0.25,
            ..Default::default()
        });
        assert!(s.on_event(&ticker_event(1, 50, 0)).is_none());
        // Mirror (2) at 40, reference (1) at 50 → 10¢ = 1000 bps divergence.
        let intent = s.on_event(&ticker_event(2, 40, 1)).expect("should emit");
        assert_eq!(intent.symbol_id, 2);
        assert_eq!(intent.limit_price_cents, 40);
        assert_eq!(intent.edge_bps, 1000);
        assert!(intent.quantity > 0);
    }

    #[test]
    fn missing_reference_is_silent() {
        let mut s = CoherenceArb::new(CoherenceArbConfig {
            pairs: vec![(1, 2)],
            ..Default::default()
        });
        // Only the mirror has data.
        assert!(s.on_event(&ticker_event(2, 40, 0)).is_none());
    }

    #[test]
    fn sub_threshold_divergence_is_silent() {
        let mut s = CoherenceArb::new(CoherenceArbConfig {
            pairs: vec![(1, 2)],
            min_divergence_bps: 500,
            ..Default::default()
        });
        s.on_event(&ticker_event(1, 50, 0));
        // 2¢ = 200 bps < 500 bps threshold.
        assert!(s.on_event(&ticker_event(2, 48, 1)).is_none());
    }
}
