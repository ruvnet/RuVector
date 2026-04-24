//! End-to-end paper-trade demo (no network) — full neural-trader pipeline.
//!
//! ```text
//!   raw WS JSON
//!      │
//!      ▼
//!   FeedDecoder                      (ruvector-kalshi::ws)
//!      │  produces MarketEvent
//!      ▼
//!   ExpectedValueKelly               (neural-trader-strategies)
//!      │  emits Intent when edge > threshold
//!      ▼
//!   CoherenceChecker (ThresholdGate) (neural-trader-coherence)
//!      │  allow_act gate on mincut/CUSUM/drift
//!      ▼
//!   RiskGate                         (neural-trader-strategies)
//!      │  cash / position / concentration / daily-loss / live-flag
//!      ▼
//!   intent_to_order                  (ruvector-kalshi::strategy_adapter)
//!      │  produces NewOrder (not sent; KALSHI_ENABLE_LIVE off)
//!      ▼
//!   ReservoirStore + InMemoryReceiptLog (neural-trader-replay)
//!      │  append ReplaySegment per event window + WitnessReceipt per fill
//!      ▼
//!   Paper ledger accumulates fills at the quoted limit price.
//! ```
//!
//! Run:
//!   cargo run -p ruvector-kalshi --example paper_trade

use std::collections::HashMap;

use neural_trader_coherence::WitnessLogger;
use neural_trader_coherence::WitnessReceipt;
use neural_trader_core::MarketEvent;
use neural_trader_replay::{
    CoherenceStats, InMemoryReceiptLog, MemoryStore, ReplaySegment, ReservoirStore, SegmentKind,
    SegmentLineage,
};
use neural_trader_strategies::{
    coherence_bridge::simple_context, CoherenceChecker, CoherenceOutcome, ExpectedValueKelly,
    ExpectedValueKellyConfig, GateConfig, PortfolioState, Position, RiskConfig, RiskDecision,
    RiskGate, Strategy, ThresholdGate,
};
use ruvector_kalshi::brain::{BrainClient, Resolution, SharedMemory};
use ruvector_kalshi::normalize::symbol_id_for;
use ruvector_kalshi::strategy_adapter::intent_to_order;
use ruvector_kalshi::ws::FeedDecoder;

/// Load a brain API key from env first, then gcloud Secret Manager.
async fn load_brain_key() -> Option<String> {
    if let Ok(k) = std::env::var("BRAIN_API_KEY") {
        return Some(k);
    }
    let out = tokio::process::Command::new("gcloud")
        .args([
            "secrets",
            "versions",
            "access",
            "latest",
            "--secret",
            "BRAIN_SYSTEM_KEY",
            "--project",
            "ruv-dev",
        ])
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let trimmed = s.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

const FRAMES: &[&str] = &[
    r#"{"type":"orderbook_snapshot","msg":{"market_ticker":"FED-DEC26","yes":[[28,100],[27,80],[26,70],[25,60],[24,50]],"no":[[72,100],[73,80],[74,70],[75,60],[76,50]],"ts":1700000000000}}"#,
    r#"{"type":"trade","msg":{"market_ticker":"FED-DEC26","yes_price":27,"count":15,"taker_side":"yes","ts":1700000001000}}"#,
    r#"{"type":"ticker","msg":{"market_ticker":"FED-DEC26","yes_bid":24,"yes_ask":26,"ts":1700000002000}}"#,
    r#"{"type":"heartbeat","msg":{}}"#,
    r#"{"type":"ticker","msg":{"market_ticker":"BTC-100K","yes_bid":49,"yes_ask":51,"ts":1700000003000}}"#,
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ticker = "FED-DEC26";
    let ticker_btc = "BTC-100K";
    let sym = symbol_id_for(ticker);
    let sym_btc = symbol_id_for(ticker_btc);

    // Optional brain client. Enable by setting BRAIN_ENABLE=1 and either
    // BRAIN_API_KEY directly or a reachable gcloud CLI (secret name:
    // BRAIN_SYSTEM_KEY in project ruv-dev).
    let brain = if std::env::var("BRAIN_ENABLE").ok().as_deref() == Some("1") {
        match load_brain_key().await {
            Some(key) => match BrainClient::new(key) {
                Ok(c) => {
                    println!("brain: share-on-fill enabled");
                    Some(c)
                }
                Err(e) => {
                    eprintln!("brain: disabled (client init failed: {e})");
                    None
                }
            },
            None => {
                eprintln!("brain: disabled (no key; set BRAIN_API_KEY or gcloud auth)");
                None
            }
        }
    } else {
        None
    };

    // --- Strategy ---
    let mut strat = ExpectedValueKelly::new(ExpectedValueKellyConfig {
        kelly_fraction: 0.25,
        bankroll_cents: 100_000,
        min_edge_bps: 100,
        strategy_name: "ev-kelly-paper",
    });
    strat.set_prior(sym, 0.40);
    strat.set_prior(sym_btc, 0.50);

    // --- Coherence gate (pre-RiskGate actuation check) ---
    // Defaults are tuned for deep equity books (mincut floor 12 in Calm).
    // Kalshi binary contracts are shallow — 5–10 levels per side — so
    // relax floors and CUSUM to match the venue. Operators should
    // calibrate against recorded depth/regime statistics.
    let coherence = CoherenceChecker::new(ThresholdGate::new(GateConfig {
        mincut_floor_calm: 3,
        mincut_floor_normal: 2,
        mincut_floor_volatile: 2,
        cusum_threshold: 50.0,
        boundary_stability_windows: 1,
        max_drift_score: 0.9,
    }));

    // --- Risk gate (paper mode) ---
    let gate = RiskGate::new(RiskConfig {
        require_live_flag: false,
        ..Default::default()
    });
    let mut portfolio = PortfolioState {
        cash_cents: 100_000,
        starting_cash_cents: 100_000,
        ..Default::default()
    };

    // --- Replay store + witness log (neural-trader-replay) ---
    let mut replay = ReservoirStore::new(1024);
    let mut receipts = InMemoryReceiptLog::new();

    let mut decoder = FeedDecoder::new();
    let mut recent_prices: HashMap<u32, Vec<i64>> = HashMap::new();
    // Observed top-of-book depth per symbol — carried across frames so a
    // trade/ticker event inherits the last-known mincut proxy from the
    // most recent snapshot.
    let mut observed_depth: HashMap<u32, u64> = HashMap::new();
    let mut fills: Vec<(String, i64, i64)> = Vec::new();
    let mut intent_count = 0usize;
    let mut approved_count = 0usize;
    let mut coherence_blocks = 0usize;
    let mut rejected: HashMap<&'static str, u32> = HashMap::new();

    for (idx, frame) in FRAMES.iter().enumerate() {
        let events: Vec<MarketEvent> = decoder.decode(frame)?;
        if events.is_empty() {
            continue;
        }

        // --- Record a ReplaySegment for the observed window ---
        let first = events.first().unwrap();
        let last = events.last().unwrap();
        // Any approved fills will flip `allow_write`; for admission we
        // always allow in paper mode so we can audit the whole stream.
        let allow_write_decision = neural_trader_coherence::CoherenceDecision {
            allow_retrieve: true,
            allow_write: true,
            allow_learn: true,
            allow_act: true,
            mincut_value: events.len() as u64,
            partition_hash: [0u8; 16],
            drift_score: 0.0,
            cusum_score: 0.0,
            reasons: vec![],
        };
        let _ = replay.maybe_write(
            ReplaySegment {
                segment_id: idx as u64,
                symbol_id: first.symbol_id,
                start_ts_ns: first.ts_exchange_ns,
                end_ts_ns: last.ts_exchange_ns,
                segment_kind: SegmentKind::Routine,
                events: events.clone(),
                embedding: None,
                labels: serde_json::json!({ "frame_index": idx }),
                coherence_stats: CoherenceStats {
                    mincut_value: events.len() as u64,
                    partition_hash: [0u8; 16],
                    drift_score: 0.0,
                    cusum_score: 0.0,
                },
                lineage: SegmentLineage {
                    model_id: "ev-kelly-paper".into(),
                    policy_version: "1".into(),
                    ingest_batch_id: Some(format!("paper-{idx}")),
                },
                witness_hash: [0u8; 16],
            },
            &allow_write_decision,
        )?;

        // A snapshot frame produces one event per level per side; count
        // them to seed the coherence mincut proxy.
        {
            use neural_trader_core::EventType;
            let snap_count = events
                .iter()
                .filter(|e| e.event_type == EventType::BookSnapshot)
                .count() as u64;
            if snap_count > 0 {
                *observed_depth.entry(first.symbol_id).or_insert(0) = snap_count;
            }
        }

        for evt in &events {
            // Track recent *mid-proxy* prices per symbol for the coherence
            // context. Only trades and tickers give us a single mid; book
            // snapshots fire per level and would otherwise inflate CUSUM
            // with spread width rather than price movement.
            use neural_trader_core::EventType;
            let cents = evt.price_fp / 1_000_000;
            if matches!(evt.event_type, EventType::Trade | EventType::VenueStatus)
                && cents > 0
                && cents < 100
            {
                let window = recent_prices.entry(evt.symbol_id).or_default();
                window.push(cents);
                if window.len() > 20 {
                    window.drain(0..window.len() - 20);
                }
            }

            let Some(intent) = strat.on_event(evt) else {
                continue;
            };
            intent_count += 1;

            let out_ticker = if intent.symbol_id == sym {
                ticker
            } else if intent.symbol_id == sym_btc {
                ticker_btc
            } else {
                "UNKNOWN"
            };

            // --- Coherence pre-check ---
            let window = recent_prices
                .get(&intent.symbol_id)
                .cloned()
                .unwrap_or_default();
            let depth = observed_depth.get(&intent.symbol_id).copied().unwrap_or(1);
            let ctx = simple_context(
                intent.symbol_id,
                evt.venue_id,
                evt.ts_exchange_ns,
                depth,
                &window,
            );
            let intent = match coherence.check(intent, &ctx) {
                CoherenceOutcome::Pass(i) => i,
                CoherenceOutcome::Block { intent, decision } => {
                    coherence_blocks += 1;
                    println!(
                        "frame[{idx}] COHERENCE BLOCK: {} | reasons={:?}",
                        intent.strategy, decision.reasons
                    );
                    continue;
                }
            };

            // --- Risk gate ---
            match gate.evaluate(intent.clone(), &portfolio) {
                RiskDecision::Approve(approved) => {
                    approved_count += 1;
                    let client_id = format!("paper-{idx}-{approved_count}");
                    let order = intent_to_order(out_ticker, &approved, client_id);
                    println!(
                        "frame[{idx}] APPROVE {} {:?} {} @ {}¢ ({}bps) → NewOrder {}",
                        approved.strategy,
                        approved.side,
                        approved.quantity,
                        approved.limit_price_cents,
                        approved.edge_bps,
                        order.client_order_id,
                    );
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

                    // --- Witness receipt (audit trail) ---
                    receipts.append_receipt(WitnessReceipt {
                        ts_ns: evt.ts_exchange_ns,
                        model_id: approved.strategy.to_string(),
                        input_segment_hash: [0u8; 16],
                        coherence_witness_hash: [0u8; 16],
                        policy_hash: [0u8; 16],
                        action_intent: format!(
                            "buy-yes {out_ticker} {} @ {}c",
                            approved.quantity, approved.limit_price_cents
                        ),
                        verified_token_id: [0u8; 16],
                        resulting_state_hash: [0u8; 16],
                    })?;

                    // --- Brain share (optional) ---
                    // Paper fills don't resolve real markets, so we share
                    // a provisional note so the brain can correlate
                    // strategy behavior across runs. Real resolutions
                    // would use Resolution::Yes/No with actual P&L.
                    if let Some(brain) = brain.as_ref() {
                        let notional = order.count.saturating_mul(approved.limit_price_cents);
                        let memory = SharedMemory::market_resolution(
                            out_ticker,
                            Resolution::Void, // paper → void until real settlement
                            approved.strategy,
                            0, // P&L unknown at fill time
                            notional,
                        );
                        match brain.share(&memory).await {
                            Ok(id) => println!("    brain: shared id={id}"),
                            Err(e) => eprintln!("    brain: share failed: {e}"),
                        }
                    }
                }
                RiskDecision::Reject {
                    reason,
                    intent: rej,
                } => {
                    let key = match reason {
                        neural_trader_strategies::RejectReason::EdgeTooThin => "thin-edge",
                        neural_trader_strategies::RejectReason::PositionTooLarge => {
                            "position-too-large"
                        }
                        neural_trader_strategies::RejectReason::DailyLossKill => "daily-loss-kill",
                        neural_trader_strategies::RejectReason::ClusterConcentration => {
                            "cluster-concentration"
                        }
                        neural_trader_strategies::RejectReason::LiveTradingDisabled => {
                            "live-disabled"
                        }
                        neural_trader_strategies::RejectReason::InsufficientCash => {
                            "insufficient-cash"
                        }
                        neural_trader_strategies::RejectReason::NonPositiveQuantity => "bad-qty",
                        neural_trader_strategies::RejectReason::PriceOutOfRange => "bad-price",
                    };
                    *rejected.entry(key).or_insert(0) += 1;
                    println!(
                        "frame[{idx}] REJECT[{key}] {} {:?} {} @ {}¢",
                        rej.strategy, rej.side, rej.quantity, rej.limit_price_cents,
                    );
                }
            }
        }
    }

    println!("\n--- summary ---");
    println!("intents emitted:    {intent_count}");
    println!("coherence blocked:  {coherence_blocks}");
    println!("risk approved:      {approved_count}");
    println!("risk rejected:      {rejected:?}");
    println!("fills:              {fills:?}");
    println!(
        "cash:               {}¢ (starting {}¢)",
        portfolio.cash_cents, portfolio.starting_cash_cents
    );
    println!(
        "open positions:     {} symbol(s), notional {}¢",
        portfolio.positions.len(),
        portfolio.total_notional_cents()
    );
    println!(
        "replay segments:    {} stored in ReservoirStore",
        replay.len()
    );
    println!(
        "witness receipts:   {} in InMemoryReceiptLog",
        receipts.len()
    );

    // Quick retrieval check: the replay store must be queryable per symbol.
    let retrieved = replay.retrieve(&neural_trader_replay::MemoryQuery {
        symbol_id: sym,
        embedding: vec![],
        regime: None,
        limit: 10,
    })?;
    println!("replay retrieve(FED-DEC26): {} segments", retrieved.len());

    assert!(intent_count >= 1);
    assert!(approved_count >= 1);
    assert!(
        receipts.len() >= 1,
        "at least one fill must produce a receipt"
    );
    assert!(
        replay.len() >= 1,
        "replay store must capture at least one segment"
    );
    Ok(())
}
