//! Glue between `neural_trader_strategies::Intent` and Kalshi
//! `NewOrder` payloads. Kept in this crate (not the strategies crate) so
//! strategies remain venue-agnostic.

use neural_trader_strategies::{Action, Intent, Side};

use crate::models::{NewOrder, OrderAction, OrderSide, OrderType};

/// Convert a strategy `Intent` into a Kalshi limit `NewOrder`. The caller
/// must supply a ticker — the `Intent` holds a canonical `symbol_id`
/// (FNV-1a of the ticker) and recovering the original string is not
/// possible, so the caller, who knows which ticker it's submitting for,
/// provides it directly.
pub fn intent_to_order(
    ticker: impl Into<String>,
    intent: &Intent,
    client_order_id: impl Into<String>,
) -> NewOrder {
    let action = match intent.action {
        Action::Buy => OrderAction::Buy,
        Action::Sell => OrderAction::Sell,
    };
    let side = match intent.side {
        Side::Yes => OrderSide::Yes,
        Side::No => OrderSide::No,
    };
    // Kalshi expects the YES/NO price only on the matching side.
    let (yes_price, no_price) = match intent.side {
        Side::Yes => (Some(intent.limit_price_cents), None),
        Side::No => (None, Some(intent.limit_price_cents)),
    };
    NewOrder {
        ticker: ticker.into(),
        action,
        side,
        order_type: OrderType::Limit,
        count: intent.quantity,
        yes_price,
        no_price,
        client_order_id: client_order_id.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_trader_strategies::{Action, Intent, Side};

    #[test]
    fn yes_intent_maps_to_yes_price() {
        let intent = Intent {
            symbol_id: 42,
            side: Side::Yes,
            action: Action::Buy,
            limit_price_cents: 24,
            quantity: 10,
            edge_bps: 1600,
            confidence: 0.4,
            strategy: "ev-kelly",
        };
        let order = intent_to_order("FED-DEC23", &intent, "cli-1");
        assert_eq!(order.ticker, "FED-DEC23");
        assert_eq!(order.client_order_id, "cli-1");
        assert!(matches!(order.side, OrderSide::Yes));
        assert!(matches!(order.action, OrderAction::Buy));
        assert_eq!(order.yes_price, Some(24));
        assert_eq!(order.no_price, None);
        assert_eq!(order.count, 10);
    }

    #[test]
    fn no_intent_maps_to_no_price() {
        let intent = Intent {
            symbol_id: 42,
            side: Side::No,
            action: Action::Buy,
            limit_price_cents: 76,
            quantity: 5,
            edge_bps: 500,
            confidence: 0.3,
            strategy: "t",
        };
        let order = intent_to_order("FED-DEC23", &intent, "cli-2");
        assert_eq!(order.yes_price, None);
        assert_eq!(order.no_price, Some(76));
    }

    #[test]
    fn sell_action_is_preserved() {
        let intent = Intent {
            symbol_id: 42,
            side: Side::Yes,
            action: Action::Sell,
            limit_price_cents: 30,
            quantity: 7,
            edge_bps: 0,
            confidence: 0.0,
            strategy: "t",
        };
        let order = intent_to_order("X", &intent, "c");
        assert!(matches!(order.action, OrderAction::Sell));
    }
}
