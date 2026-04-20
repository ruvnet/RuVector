//! Authenticated GET /markets against the live Kalshi REST API.
//!
//! Secrets are loaded from Google Cloud Secret Manager by default (project
//! `ruv-dev`, secrets `KALSHI_API_KEY` / `KALSHI_PRIVATE_KEY_PEM` /
//! `KALSHI_API_URL`). Override with `KALSHI_SECRET_SOURCE=local` + a
//! local PEM, or `KALSHI_SECRET_SOURCE=env`.
//!
//! Run:
//!   cargo run -p ruvector-kalshi --example list_markets
//!
//! Optional:
//!   KALSHI_MARKETS_LIMIT=10          — cap the print count (default 5)
//!   KALSHI_MARKETS_STATUS=open       — filter (open / closed / settled)

use ruvector_kalshi::auth::Signer;
use ruvector_kalshi::rest::RestClient;
use ruvector_kalshi::secrets::SecretLoader;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let project = std::env::var("KALSHI_GCP_PROJECT").unwrap_or_else(|_| "ruv-dev".into());
    let loader = SecretLoader::new(project);
    let creds = loader.load().await?;
    let signer = Signer::from_pem(&creds.api_key, &creds.private_key_pem)?;
    let client = RestClient::new(&creds.api_url, signer)?;

    let limit: usize = std::env::var("KALSHI_MARKETS_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let status_filter = std::env::var("KALSHI_MARKETS_STATUS").ok();

    println!("GET {}/markets{}", creds.api_url, status_filter
        .as_deref()
        .map(|s| format!("?status={s}"))
        .unwrap_or_default());

    let markets = client.list_markets(status_filter.as_deref()).await?;
    println!("received {} market(s); showing first {}:\n", markets.len(), limit.min(markets.len()));

    for m in markets.iter().take(limit) {
        let yes = m
            .yes_bid
            .zip(m.yes_ask)
            .map(|(b, a)| format!("{b}/{a}¢"))
            .unwrap_or_else(|| "-/-".into());
        let vol = m.volume.map(|v| v.to_string()).unwrap_or_else(|| "-".into());
        let title = m.title.as_deref().unwrap_or("");
        let status = m.status.as_deref().unwrap_or("-");
        println!(
            "  {:30} [{:>7}] yes={:10} vol={:>10}  {}",
            m.ticker, status, yes, vol, title
        );
    }
    Ok(())
}
