//! End-to-end paper-trade demo (no network).
//!
//! A canned sequence of Kalshi WebSocket frames is piped through the
//! full neural-trader pipeline:
//!
//!   raw WS JSON
//!      │
//!      ▼
//!   FeedDecoder (ruvector-kalshi::ws)
//!      │  produces MarketEvent
//!      ▼
//!   ExpectedValueKelly (neural-trader-strategies)
//!      │  emits Intent when edge > threshold
//!      ▼
//!   RiskGate (neural-trader-strategies)
//!      │  approves / rejects with reason
//!      ▼
//!   intent_to_order (ruvector-kalshi::strategy_adapter)
//!      │  produces NewOrder (not sent; KALSHI_ENABLE_LIVE off)
//!      ▼
//!   Paper ledger accumulates fills at the quoted limit price.
//!
//! Run:
//!   cargo run -p ruvector-kalshi --example paper_trade

use std::collections::HashMap;

use neural_trader_core::MarketEvent;
use neural_trader_strategies::{
    ExpectedValueKelly, ExpectedValueKellyConfig, PortfolioState, Position, RiskConfig,
    RiskDecision, RiskGate, Strategy,
};
use ruvector_kalshi::normalize::symbol_id_for;
use ruvector_kalshi::strategy_adapter::intent_to_order;
use ruvector_kalshi::ws::FeedDecoder;

/// Hardcoded scenario: a Fed-rate market drifts from 28¢ down to 24¢
/// while our prior says 40%. The strategy should emit a BUY YES intent.
const FRAMES: &[&str] = &[
    r#"{"type":"orderbook_snapshot","msg":{"market_ticker":"FED-DEC26","yes":[[28,100]],"no":[[72,100]],"ts":1700000000000}}"#,
    r#"{"type":"trade","msg":{"market_ticker":"FED-DEC26","yes_price":27,"count":15,"taker_side":"yes","ts":1700000001000}}"#,
    r#"{"type":"ticker","msg":{"market_ticker":"FED-DEC26","yes_bid":24,"yes_ask":26,"ts":1700000002000}}"#,
    // Heartbeat should be silently ignored.
    r#"{"type":"heartbeat","msg":{}}"#,
    // A thin-edge market that should NOT trigger (prior ~ mid).
    r#"{"type":"ticker","msg":{"market_ticker":"BTC-100K","yes_bid":49,"yes_ask":51,"ts":1700000003000}}"#,
];

fn main() -> anyhow::Result<()> {
    let ticker = "FED-DEC26";
    let ticker_btc = "BTC-100K";
    let sym = symbol_id_for(ticker);
    let sym_btc = symbol_id_for(ticker_btc);

    // Strategy with a 40% YES prior on the Fed market; 50% on BTC (mid).
    let mut strat = ExpectedValueKelly::new(ExpectedValueKellyConfig {
        kelly_fraction: 0.25,
        bankroll_cents: 100_000,
        min_edge_bps: 100,
        strategy_name: "ev-kelly-paper",
    });
    strat.set_prior(sym, 0.40);
    strat.set_prior(sym_btc, 0.50);

    // Risk gate in paper mode (no env flag required).
    let gate = RiskGate::new(RiskConfig {
        require_live_flag: false,
        ..Default::default()
    });
    let mut portfolio = PortfolioState {
        cash_cents: 100_000,
        starting_cash_cents: 100_000,
        ..Default::default()
    };

    let mut decoder = FeedDecoder::new();
    let mut fills: Vec<(String, i64, i64)> = Vec::new();
    let mut intent_count = 0usize;
    let mut approved_count = 0usize;
    let mut rejected: HashMap<&'static str, u32> = HashMap::new();

    for (idx, frame) in FRAMES.iter().enumerate() {
        let events: Vec<MarketEvent> = decoder.decode(frame)?;
        for evt in &events {
            let Some(intent) = strat.on_event(evt) else { continue };
            intent_count += 1;

            // Resolve symbol_id back to ticker for the adapter. Paper trader
            // keeps its own tiny map from sym_id → ticker.
            let out_ticker = if intent.symbol_id == sym {
                ticker
            } else if intent.symbol_id == sym_btc {
                ticker_btc
            } else {
                "UNKNOWN"
            };

            match gate.evaluate(intent.clone(), &portfolio) {
                RiskDecision::Approve(approved) => {
                    approved_count += 1;
                    let client_id = format!("paper-{idx}-{approved_count}");
                    let order = intent_to_order(out_ticker, &approved, client_id);
                    println!(
                        "frame[{idx}] APPROVE {strategy} {side:?} {qty} @ {px}¢ ({bps}bps) → {order:?}",
                        strategy = approved.strategy,
                        side = approved.side,
                        qty = approved.quantity,
                        px = approved.limit_price_cents,
                        bps = approved.edge_bps,
                        order = order,
                    );
                    // Simulate an immediate full fill at the limit.
                    let cost = order.count.saturating_mul(approved.limit_price_cents);
                    portfolio.cash_cents = portfolio.cash_cents.saturating_sub(cost);
                    portfolio
                        .positions
                        .entry(approved.symbol_id)
                        .and_modify(|p: &mut Position| {
                            let new_qty = p.quantity.saturating_add(order.count);
                            let new_cost = p
                                .quantity
                                .saturating_mul(p.avg_price_cents)
                                .saturating_add(cost);
                            p.avg_price_cents = if new_qty > 0 { new_cost / new_qty } else { 0 };
                            p.quantity = new_qty;
                        })
                        .or_insert(Position {
                            symbol_id: approved.symbol_id,
                            quantity: order.count,
                            avg_price_cents: approved.limit_price_cents,
                        });
                    fills.push((out_ticker.into(), approved.limit_price_cents, order.count));
                }
                RiskDecision::Reject { reason, intent: rej } => {
                    let key = match reason {
                        neural_trader_strategies::RejectReason::EdgeTooThin => "thin-edge",
                        neural_trader_strategies::RejectReason::PositionTooLarge => "position-too-large",
                        neural_trader_strategies::RejectReason::DailyLossKill => "daily-loss-kill",
                        neural_trader_strategies::RejectReason::ClusterConcentration => "cluster-concentration",
                        neural_trader_strategies::RejectReason::LiveTradingDisabled => "live-disabled",
                        neural_trader_strategies::RejectReason::InsufficientCash => "insufficient-cash",
                        neural_trader_strategies::RejectReason::NonPositiveQuantity => "bad-qty",
                        neural_trader_strategies::RejectReason::PriceOutOfRange => "bad-price",
                    };
                    *rejected.entry(key).or_insert(0) += 1;
                    println!(
                        "frame[{idx}] REJECT[{key}] {strategy} {side:?} {qty} @ {px}¢",
                        strategy = rej.strategy,
                        side = rej.side,
                        qty = rej.quantity,
                        px = rej.limit_price_cents,
                    );
                }
            }
        }
    }

    println!("\n--- summary ---");
    println!("intents emitted: {intent_count}");
    println!("approved:        {approved_count}");
    println!("rejected:        {rejected:?}");
    println!("fills:           {fills:?}");
    println!(
        "remaining cash:  {}¢ (starting {}¢)",
        portfolio.cash_cents, portfolio.starting_cash_cents
    );
    println!(
        "open positions:  {} symbol(s), total notional {}¢",
        portfolio.positions.len(),
        portfolio.total_notional_cents()
    );

    // Minimal sanity asserts so this runs as a smoke test.
    assert!(intent_count >= 1, "at least one intent should have been emitted");
    assert!(approved_count >= 1, "at least one should have passed risk gate");
    assert!(
        portfolio.cash_cents < portfolio.starting_cash_cents,
        "cash must drop after a fill"
    );
    Ok(())
}
