//! Live execution runner — full pipeline driving real Kalshi orders.
//!
//! **SAFETY GATES** — this binary refuses to run unless ALL of:
//!   * `KALSHI_ENABLE_LIVE=1`    — same gate the REST client enforces
//!   * `KALSHI_CONFIRM_LIVE=yes` — second human-readable confirmation
//!   * `KALSHI_MAX_ORDERS` set   — explicit cap on total orders in a session
//!   * A ticker + prior on the command line
//!
//! Without all four, execution is blocked before any signed request is
//! emitted. Absent `KALSHI_ENABLE_LIVE=1`, the RiskGate and REST client
//! both refuse orders independently.
//!
//! Usage:
//!   KALSHI_ENABLE_LIVE=1 KALSHI_CONFIRM_LIVE=yes KALSHI_MAX_ORDERS=3 \
//!     cargo run -p ruvector-kalshi --example live_trade -- \
//!       FED-24DEC-T3.00 0.35
//!
//! Pipeline (mirrors paper_trade with two differences: events come from
//! the live WS, and approved orders are sent via RestClient):
//!   WS → FeedDecoder → Strategy → CoherenceChecker → RiskGate
//!     → intent_to_order → RestClient::post_order

use std::collections::HashMap;

use neural_trader_core::MarketEvent;
use neural_trader_strategies::{
    coherence_bridge::simple_context, CoherenceChecker, CoherenceOutcome, ExpectedValueKelly,
    ExpectedValueKellyConfig, GateConfig, PortfolioState, RiskConfig, RiskDecision, RiskGate,
    Strategy, ThresholdGate,
};
use ruvector_kalshi::auth::Signer;
use ruvector_kalshi::normalize::symbol_id_for;
use ruvector_kalshi::rest::RestClient;
use ruvector_kalshi::secrets::SecretLoader;
use ruvector_kalshi::strategy_adapter::intent_to_order;
use ruvector_kalshi::ws_client::{reconnect_forever, Subscribe};
use ruvector_kalshi::KALSHI_WS_URL;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ---- Safety gates (fail-closed) ----
    if std::env::var("KALSHI_ENABLE_LIVE").ok().as_deref() != Some("1") {
        anyhow::bail!("refusing to run: KALSHI_ENABLE_LIVE=1 required");
    }
    if std::env::var("KALSHI_CONFIRM_LIVE").ok().as_deref() != Some("yes") {
        anyhow::bail!("refusing to run: KALSHI_CONFIRM_LIVE=yes required");
    }
    let max_orders: usize = std::env::var("KALSHI_MAX_ORDERS")
        .map_err(|_| anyhow::anyhow!("KALSHI_MAX_ORDERS must be set (integer)"))?
        .parse()
        .map_err(|e| anyhow::anyhow!("KALSHI_MAX_ORDERS must be an integer: {e}"))?;
    if max_orders == 0 {
        anyhow::bail!("KALSHI_MAX_ORDERS=0 blocks all trading; exit");
    }

    let mut args = std::env::args().skip(1);
    let ticker = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: live_trade TICKER PRIOR"))?;
    let prior: f64 = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: live_trade TICKER PRIOR"))?
        .parse()
        .map_err(|e| anyhow::anyhow!("PRIOR must be a float 0..1: {e}"))?;

    // ---- Credentials ----
    let project = std::env::var("KALSHI_GCP_PROJECT").unwrap_or_else(|_| "ruv-dev".into());
    let creds = SecretLoader::new(project).load().await?;
    let signer = Signer::from_pem(&creds.api_key, &creds.private_key_pem)?;
    let rest = RestClient::new(&creds.api_url, signer.clone())?;

    // ---- Strategy + Gates ----
    let sym = symbol_id_for(&ticker);
    let mut strat = ExpectedValueKelly::new(ExpectedValueKellyConfig {
        kelly_fraction: 0.10,                   // conservative on live
        bankroll_cents: 10_000,                 // tiny for first live run
        min_edge_bps: 500,
        strategy_name: "ev-kelly-live",
    });
    strat.set_prior(sym, prior);
    let coherence = CoherenceChecker::new(ThresholdGate::new(GateConfig {
        mincut_floor_calm: 3,
        mincut_floor_normal: 2,
        mincut_floor_volatile: 2,
        cusum_threshold: 50.0,
        boundary_stability_windows: 1,
        max_drift_score: 0.9,
    }));
    let gate = RiskGate::new(RiskConfig {
        require_live_flag: true,
        max_position_frac: 0.05,    // ≤ 5% of cash per position
        max_daily_loss_frac: 0.02,  // tight daily kill
        min_edge_bps: 500,
        max_cluster_frac: 0.10,
    });
    let mut portfolio = PortfolioState {
        cash_cents: 10_000,
        starting_cash_cents: 10_000,
        ..Default::default()
    };

    // ---- Live feed ----
    let ws_url = std::env::var("KALSHI_WS_URL").unwrap_or_else(|_| KALSHI_WS_URL.to_string());
    let sub = Subscribe::new(
        1,
        vec![
            "ticker".into(),
            "trade".into(),
            "orderbook_snapshot".into(),
        ],
        vec![ticker.clone()],
    );
    let (tx, mut rx) = mpsc::channel::<MarketEvent>(256);
    let pump = tokio::spawn(reconnect_forever(signer, Some(ws_url), sub, tx));

    println!(
        "live_trade: ticker={ticker} prior={prior:.2} max_orders={max_orders} \
         cash={}¢ — live orders enabled",
        portfolio.cash_cents
    );

    let mut recent_prices: HashMap<u32, Vec<i64>> = HashMap::new();
    let mut observed_depth: HashMap<u32, u64> = HashMap::new();
    let mut orders_sent = 0usize;

    while let Some(evt) = rx.recv().await {
        use neural_trader_core::EventType;
        // Track depth / prices for the coherence context.
        if evt.event_type == EventType::BookSnapshot {
            *observed_depth.entry(evt.symbol_id).or_insert(0) += 1;
        }
        let cents = evt.price_fp / 1_000_000;
        if matches!(evt.event_type, EventType::Trade | EventType::VenueStatus)
            && cents > 0 && cents < 100
        {
            let w = recent_prices.entry(evt.symbol_id).or_default();
            w.push(cents);
            if w.len() > 40 { w.drain(0..w.len() - 40); }
        }

        let Some(intent) = strat.on_event(&evt) else { continue };

        let window = recent_prices.get(&intent.symbol_id).cloned().unwrap_or_default();
        let depth = observed_depth.get(&intent.symbol_id).copied().unwrap_or(1);
        let ctx = simple_context(intent.symbol_id, evt.venue_id, evt.ts_exchange_ns, depth, &window);
        let intent = match coherence.check(intent, &ctx) {
            CoherenceOutcome::Pass(i) => i,
            CoherenceOutcome::Block { decision, .. } => {
                eprintln!("  coherence block: {:?}", decision.reasons);
                continue;
            }
        };
        let approved = match gate.evaluate(intent, &portfolio) {
            RiskDecision::Approve(a) => a,
            RiskDecision::Reject { reason, .. } => {
                eprintln!("  risk reject: {reason:?}");
                continue;
            }
        };

        // ---- Send live order ----
        let client_id = format!("live-{}-{}", chrono::Utc::now().timestamp_millis(), orders_sent);
        let order = intent_to_order(&ticker, &approved, client_id);
        match rest.post_order(&order).await {
            Ok(ack) => {
                orders_sent += 1;
                println!(
                    "  ORDER {} accepted id={} status={}",
                    orders_sent, ack.order.order_id, ack.order.status
                );
                // Simulate the order sitting in the book — we don't yet
                // consume fills here. Update cash optimistically; a real
                // deployment reconciles against the fill stream.
                let cost = order.count.saturating_mul(approved.limit_price_cents);
                portfolio.cash_cents = portfolio.cash_cents.saturating_sub(cost);
            }
            Err(e) => eprintln!("  ORDER FAILED: {e}"),
        }

        if orders_sent >= max_orders {
            println!("reached KALSHI_MAX_ORDERS={max_orders}; stopping");
            break;
        }
    }

    drop(rx);
    let _ = pump.await;
    Ok(())
}
