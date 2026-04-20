//! Offline validator: loads the real Kalshi credentials via `SecretLoader`,
//! signs a sample request, verifies the signature round-trips.
//!
//! No network calls are made. This proves the actual PEM on disk (or in
//! GCS) is loadable by this crate and produces valid RSA-PSS-SHA256
//! signatures.
//!
//! Run:
//!   # Offline, local PEM:
//!   KALSHI_SECRET_SOURCE=local cargo run -p ruvector-kalshi --example validate
//!
//!   # From Google Cloud Secret Manager (needs `gcloud` auth):
//!   cargo run -p ruvector-kalshi --example validate

use ruvector_kalshi::{
    auth::{self, Signer},
    secrets::SecretLoader,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let project = std::env::var("KALSHI_GCP_PROJECT").unwrap_or_else(|_| "ruv-dev".into());
    let loader = SecretLoader::new(project);
    let creds = loader.load().await?;
    println!("loaded credentials: {creds:?}");

    let signer = Signer::from_pem(&creds.api_key, &creds.private_key_pem)?;
    println!("signer ready: {signer:?}");

    let ts_ms = 1_700_000_000_000u64;
    let method = "GET";
    let path = "/trade-api/v2/exchange/status";

    let headers = signer.sign_with_ts(ts_ms, method, path);
    println!(
        "signed (ts={}, method={method}, path={path}):\n  \
         KALSHI-ACCESS-KEY:       <len {} chars>\n  \
         KALSHI-ACCESS-TIMESTAMP: {}\n  \
         KALSHI-ACCESS-SIGNATURE: <len {} chars, base64>",
        ts_ms,
        headers.access_key.len(),
        headers.timestamp_ms,
        headers.signature_b64.len(),
    );

    auth::verify(
        &signer.verifying_key(),
        ts_ms,
        method,
        path,
        &headers.signature_b64,
    )?;
    println!("signature verified against derived public key — OK");

    // Tampered verification must fail.
    let tamper = auth::verify(
        &signer.verifying_key(),
        ts_ms,
        method,
        "/trade-api/v2/portfolio",
        &headers.signature_b64,
    );
    if tamper.is_ok() {
        anyhow::bail!("tampered path unexpectedly verified — signing is broken");
    }
    println!("tamper check rejected — OK");

    println!("\nall offline checks passed.");
    Ok(())
}
