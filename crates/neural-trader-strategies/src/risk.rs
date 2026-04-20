//! Risk gate — mandatory wrapper around every [`Intent`].
//!
//! All checks are in Kalshi cents so we never cross scale boundaries on
//! the hot path. The gate is pure: it never mutates portfolio state, only
//! returns a decision. Updating the portfolio happens after a fill is
//! observed.

use std::collections::HashMap;

use crate::intent::{Action, Intent, Side};

/// Per-ticker open position.
#[derive(Debug, Clone, Default)]
pub struct Position {
    pub symbol_id: u32,
    /// Signed contract quantity (YES = +, NO = -). For Kalshi we store
    /// YES/NO on separate symbol ids using the same ticker so this stays
    /// unsigned in practice, but we keep i64 for future flexibility.
    pub quantity: i64,
    /// Average fill price in cents.
    pub avg_price_cents: i64,
}

impl Position {
    pub fn notional_cents(&self) -> i64 {
        self.quantity.saturating_mul(self.avg_price_cents)
    }
}

/// Cluster identifier — a group of correlated contracts (e.g. "all Fed
/// December strikes"). Strategy authors assign cluster ids; the neural-
/// trader-coherence crate produces them in production but tests can use
/// any `u32`.
pub type ClusterId = u32;

/// Portfolio state observed by the gate. Pure snapshot — the gate never
/// mutates it.
#[derive(Debug, Clone, Default)]
pub struct PortfolioState {
    pub cash_cents: i64,
    pub starting_cash_cents: i64,
    /// Realized P&L for the current UTC day, in cents. Negative = loss.
    pub day_pnl_cents: i64,
    pub positions: HashMap<u32, Position>,
    /// Optional symbol→cluster mapping. When populated, concentration is
    /// enforced by cluster; otherwise by symbol.
    pub clusters: HashMap<u32, ClusterId>,
}

impl PortfolioState {
    pub fn total_notional_cents(&self) -> i64 {
        self.positions
            .values()
            .map(|p| p.notional_cents().abs())
            .fold(0i64, |a, b| a.saturating_add(b))
    }

    fn cluster_notional_cents(&self, cluster: ClusterId) -> i64 {
        self.positions
            .values()
            .filter(|p| self.clusters.get(&p.symbol_id).copied() == Some(cluster))
            .map(|p| p.notional_cents().abs())
            .fold(0i64, |a, b| a.saturating_add(b))
    }
}

/// Gate configuration. All limits are hard — breach → reject.
#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// Max notional for a single new position, as a fraction of cash.
    /// Default: 0.10 (10%).
    pub max_position_frac: f64,
    /// Stop trading when realized day P&L ≤ `-max_daily_loss_frac × starting_cash`.
    /// Default: 0.03 (3%).
    pub max_daily_loss_frac: f64,
    /// Minimum edge to open in basis points. Default: 300 bps.
    pub min_edge_bps: i64,
    /// Max fraction of cash allocated to one cluster. Default: 0.40 (40%).
    pub max_cluster_frac: f64,
    /// If true, live orders require `KALSHI_ENABLE_LIVE=1`. If false, the
    /// gate allows orders regardless (paper mode).
    pub require_live_flag: bool,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position_frac: 0.10,
            max_daily_loss_frac: 0.03,
            min_edge_bps: 300,
            max_cluster_frac: 0.40,
            require_live_flag: true,
        }
    }
}

/// Outcome of the gate for a single intent.
#[derive(Debug, Clone, PartialEq)]
pub enum RiskDecision {
    Approve(Intent),
    Reject { reason: RejectReason, intent: Intent },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    EdgeTooThin,
    PositionTooLarge,
    DailyLossKill,
    ClusterConcentration,
    LiveTradingDisabled,
    InsufficientCash,
    NonPositiveQuantity,
    PriceOutOfRange,
}

#[derive(Debug, Clone, Default)]
pub struct RiskGate {
    pub config: RiskConfig,
}

impl RiskGate {
    pub fn new(config: RiskConfig) -> Self {
        Self { config }
    }

    /// Evaluate a single intent against the portfolio snapshot and env.
    pub fn evaluate(&self, intent: Intent, portfolio: &PortfolioState) -> RiskDecision {
        // 1. Basic sanity (cheapest checks first — failures are immediate).
        if intent.quantity <= 0 {
            return RiskDecision::Reject {
                reason: RejectReason::NonPositiveQuantity,
                intent,
            };
        }
        if intent.limit_price_cents <= 0 || intent.limit_price_cents >= 100 {
            return RiskDecision::Reject {
                reason: RejectReason::PriceOutOfRange,
                intent,
            };
        }

        // 2. Daily loss kill — stop *all* opening trades after breach.
        let max_loss = (portfolio.starting_cash_cents as f64
            * self.config.max_daily_loss_frac)
            .round() as i64;
        if intent.action == Action::Buy && portfolio.day_pnl_cents <= -max_loss {
            return RiskDecision::Reject {
                reason: RejectReason::DailyLossKill,
                intent,
            };
        }

        // 3. Edge threshold.
        if intent.edge_bps < self.config.min_edge_bps {
            return RiskDecision::Reject {
                reason: RejectReason::EdgeTooThin,
                intent,
            };
        }

        // 4. Hard cash check — can we actually pay for the contracts?
        //    Checked before policy caps so the "not enough money" reason is
        //    distinguishable from "over the configured cap."
        let notional = intent.notional_cents();
        if intent.action == Action::Buy && notional > portfolio.cash_cents {
            return RiskDecision::Reject {
                reason: RejectReason::InsufficientCash,
                intent,
            };
        }

        // 5. Single-position notional cap (policy).
        let position_cap = (portfolio.cash_cents as f64
            * self.config.max_position_frac) as i64;
        if intent.action == Action::Buy && notional > position_cap {
            return RiskDecision::Reject {
                reason: RejectReason::PositionTooLarge,
                intent,
            };
        }

        // 6. Cluster concentration (policy).
        if let Some(cluster) = portfolio.clusters.get(&intent.symbol_id).copied() {
            let cluster_cap = (portfolio.cash_cents as f64
                * self.config.max_cluster_frac) as i64;
            let existing = portfolio.cluster_notional_cents(cluster);
            let projected = existing.saturating_add(notional);
            if intent.action == Action::Buy && projected > cluster_cap {
                return RiskDecision::Reject {
                    reason: RejectReason::ClusterConcentration,
                    intent,
                };
            }
        }

        // 7. Live-trade env flag (last; cheap but most commonly hit in dev).
        if self.config.require_live_flag && is_opening_live_order(&intent) {
            if std::env::var("KALSHI_ENABLE_LIVE").ok().as_deref() != Some("1") {
                return RiskDecision::Reject {
                    reason: RejectReason::LiveTradingDisabled,
                    intent,
                };
            }
        }

        RiskDecision::Approve(intent)
    }
}

fn is_opening_live_order(intent: &Intent) -> bool {
    // Opening buys require live; closing sells can always run (we want to
    // be able to exit positions even after the kill switch trips).
    intent.action == Action::Buy && matches!(intent.side, Side::Yes | Side::No)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::{Action, Side};

    fn sample_intent(edge: i64, price: i64, qty: i64) -> Intent {
        Intent {
            symbol_id: 1,
            side: Side::Yes,
            action: Action::Buy,
            limit_price_cents: price,
            quantity: qty,
            edge_bps: edge,
            confidence: 0.9,
            strategy: "test",
        }
    }

    fn portfolio(cash: i64) -> PortfolioState {
        PortfolioState {
            cash_cents: cash,
            starting_cash_cents: cash,
            ..Default::default()
        }
    }

    #[test]
    fn thin_edge_rejected() {
        let gate = RiskGate::default();
        let d = gate.evaluate(sample_intent(100, 24, 10), &portfolio(100_000));
        assert!(matches!(d, RiskDecision::Reject { reason: RejectReason::EdgeTooThin, .. }));
    }

    #[test]
    fn oversize_position_rejected() {
        let gate = RiskGate::default();
        // 24¢ × 600 = 14_400¢ vs cap 10% × 100_000 = 10_000¢
        let d = gate.evaluate(sample_intent(500, 24, 600), &portfolio(100_000));
        assert!(matches!(d, RiskDecision::Reject { reason: RejectReason::PositionTooLarge, .. }));
    }

    #[test]
    fn daily_loss_kill_triggers() {
        let gate = RiskGate::default();
        let mut p = portfolio(100_000);
        p.day_pnl_cents = -3_001; // just past 3%
        let d = gate.evaluate(sample_intent(500, 24, 10), &p);
        assert!(matches!(d, RiskDecision::Reject { reason: RejectReason::DailyLossKill, .. }));
    }

    #[test]
    fn concentration_rejected() {
        // Raise the per-position cap so we're testing concentration, not
        // position size. Disable live flag so we don't trip on env.
        let gate = RiskGate::new(RiskConfig {
            require_live_flag: false,
            max_position_frac: 0.20,
            max_cluster_frac: 0.40,
            ..Default::default()
        });
        let mut p = portfolio(100_000);
        p.clusters.insert(1, 42);
        p.clusters.insert(2, 42);
        p.positions.insert(
            2,
            Position {
                symbol_id: 2,
                quantity: 1000,
                avg_price_cents: 35, // 35_000¢ already in cluster 42
            },
        );
        // 24¢ × 500 = 12_000 < 20% cap (20_000) but existing 35_000 + 12_000
        // = 47_000 > 40% cap (40_000) → concentration rejection.
        let d = gate.evaluate(sample_intent(500, 24, 500), &p);
        assert!(matches!(d, RiskDecision::Reject { reason: RejectReason::ClusterConcentration, .. }));
    }

    #[test]
    fn approves_under_paper_config() {
        let gate = RiskGate::new(RiskConfig { require_live_flag: false, ..Default::default() });
        let d = gate.evaluate(sample_intent(500, 24, 10), &portfolio(100_000));
        assert!(matches!(d, RiskDecision::Approve(_)));
    }

    #[test]
    fn live_gate_rejects_without_env() {
        // Ensure env flag is off.
        std::env::remove_var("KALSHI_ENABLE_LIVE");
        let gate = RiskGate::new(RiskConfig { require_live_flag: true, ..Default::default() });
        let d = gate.evaluate(sample_intent(500, 24, 10), &portfolio(100_000));
        assert!(matches!(d, RiskDecision::Reject { reason: RejectReason::LiveTradingDisabled, .. }));
    }

    #[test]
    fn rejects_non_positive_qty_and_bad_price() {
        let gate = RiskGate::new(RiskConfig { require_live_flag: false, ..Default::default() });
        let d = gate.evaluate(sample_intent(500, 24, 0), &portfolio(100_000));
        assert!(matches!(d, RiskDecision::Reject { reason: RejectReason::NonPositiveQuantity, .. }));
        let d = gate.evaluate(sample_intent(500, 0, 10), &portfolio(100_000));
        assert!(matches!(d, RiskDecision::Reject { reason: RejectReason::PriceOutOfRange, .. }));
        let d = gate.evaluate(sample_intent(500, 100, 10), &portfolio(100_000));
        assert!(matches!(d, RiskDecision::Reject { reason: RejectReason::PriceOutOfRange, .. }));
    }

    #[test]
    fn insufficient_cash_rejected_even_under_cap() {
        // Oversized cap so the position-size check does not mask the cash
        // check.
        let gate = RiskGate::new(RiskConfig {
            require_live_flag: false,
            max_position_frac: 5.0,
            ..Default::default()
        });
        // 50¢ × 200 = 10_000; cash only 5_000.
        let d = gate.evaluate(sample_intent(500, 50, 200), &portfolio(5_000));
        assert!(matches!(d, RiskDecision::Reject { reason: RejectReason::InsufficientCash, .. }));
    }
}
