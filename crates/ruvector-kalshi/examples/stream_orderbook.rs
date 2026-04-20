//! Live WebSocket subscriber — prints normalized `MarketEvent`s as they
//! arrive from Kalshi.
//!
//! Run:
//!   cargo run -p ruvector-kalshi --example stream_orderbook -- TICKER [TICKER...]
//!
//! Example:
//!   cargo run -p ruvector-kalshi --example stream_orderbook -- FED-24DEC-T3.00
//!
//! The stream exits on Ctrl-C or after `KALSHI_MAX_EVENTS` frames
//! (default 50). Channels subscribed: `ticker`, `trade`,
//! `orderbook_snapshot`, `orderbook_delta`.

use neural_trader_core::MarketEvent;
use ruvector_kalshi::auth::Signer;
use ruvector_kalshi::secrets::SecretLoader;
use ruvector_kalshi::ws_client::{reconnect_forever, Subscribe};
use ruvector_kalshi::KALSHI_WS_URL;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tickers: Vec<String> = std::env::args().skip(1).collect();
    if tickers.is_empty() {
        anyhow::bail!("usage: stream_orderbook TICKER [TICKER...]");
    }
    let max_events: usize = std::env::var("KALSHI_MAX_EVENTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let project = std::env::var("KALSHI_GCP_PROJECT").unwrap_or_else(|_| "ruv-dev".into());
    let creds = SecretLoader::new(project).load().await?;
    let signer = Signer::from_pem(&creds.api_key, &creds.private_key_pem)?;

    let ws_url = std::env::var("KALSHI_WS_URL").unwrap_or_else(|_| KALSHI_WS_URL.to_string());
    let sub = Subscribe::new(
        1,
        vec![
            "ticker".into(),
            "trade".into(),
            "orderbook_snapshot".into(),
            "orderbook_delta".into(),
        ],
        tickers.clone(),
    );

    println!("subscribing to {tickers:?} at {ws_url}");
    let (tx, mut rx) = mpsc::channel::<MarketEvent>(256);

    let pump = tokio::spawn(reconnect_forever(signer, Some(ws_url), sub, tx));

    let mut n = 0usize;
    while let Some(evt) = rx.recv().await {
        n += 1;
        println!(
            "[{:>4}] sym={:010} type={:?} side={:?} price_fp={:>12} qty_fp={:>12} seq={}",
            n, evt.symbol_id, evt.event_type, evt.side, evt.price_fp, evt.qty_fp, evt.seq
        );
        if n >= max_events {
            break;
        }
    }
    drop(rx); // tells reconnect_forever to stop
    let _ = pump.await;
    println!("done — {n} events received");
    Ok(())
}
