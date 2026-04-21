//! Live smoke test against the public Kalshi `/exchange/status` endpoint.
//!
//! This endpoint does NOT require authentication, so the test can run in
//! CI without any secrets. It is gated behind `#[ignore]` so the default
//! `cargo test` stays hermetic; run explicitly with:
//!
//!     cargo test -p ruvector-kalshi --test live_smoke -- --ignored --nocapture
//!
//! The nightly CI workflow passes `--ignored` to exercise this.

#[tokio::test]
#[ignore]
async fn exchange_status_is_reachable() {
    let url = std::env::var("KALSHI_API_URL")
        .unwrap_or_else(|_| ruvector_kalshi::KALSHI_API_URL.to_string());
    let full = format!("{}/exchange/status", url.trim_end_matches('/'));
    let resp = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("client")
        .get(&full)
        .send()
        .await
        .expect("network");
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    println!("GET {full} -> {status}\n{body}");
    // /exchange/status on the current Kalshi deployment requires auth and
    // returns 401 without credentials. We only care that the endpoint is
    // reachable and returns a well-formed HTTP response — any 2xx, 401,
    // or 403 counts as "server up, TLS good."
    let code = status.as_u16();
    assert!(
        (200..300).contains(&code) || code == 401 || code == 403,
        "expected 2xx/401/403 from /exchange/status, got {status}: {body}"
    );
}
