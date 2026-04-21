//! Kalshi REST client. All authenticated endpoints sign the request using
//! [`crate::auth::Signer`] and propagate typed errors.
//!
//! # Live-trade gate
//!
//! [`RestClient::post_order`] is the only method that can move money and it
//! refuses to run unless the `KALSHI_ENABLE_LIVE` environment variable is
//! set to `1`. Any other value (including unset) returns an error without
//! making the HTTP call. This is a belt-and-braces backstop on top of any
//! strategy-level `RiskGate`.

use std::sync::Arc;

use crate::auth::Signer;
use crate::models::{
    AmendOrder, CancelResponse, Market, MarketsResponse, NewOrder, OrderAck, OrderbookResponse,
    OrderbookSnapshot,
};
use crate::rate_limit::RateLimiter;
use crate::{KalshiError, Result};

#[derive(Clone)]
pub struct RestClient {
    /// Base URL string (for `reqwest` to consume). `Arc<str>` keeps clone O(1).
    base_url: Arc<str>,
    /// Pre-computed URL path component of `base_url` (e.g. `/trade-api/v2`)
    /// — prepended to the caller path to build the signature payload
    /// without re-parsing the URL on every request.
    base_path: Arc<str>,
    signer: Signer,
    http: reqwest::Client,
    limiter: Arc<RateLimiter>,
}

impl RestClient {
    pub fn new(base_url: impl Into<String>, signer: Signer) -> Result<Self> {
        // Kalshi's public rate limits are conservative; 10 req/s sustained
        // with a burst of 20 is well under any documented cap.
        Self::with_rate_limit(base_url, signer, 20, 10.0)
    }

    /// Construct with an explicit rate-limit (useful for tests and high-
    /// frequency read-only workloads).
    pub fn with_rate_limit(
        base_url: impl Into<String>,
        signer: Signer,
        burst: u32,
        per_sec: f64,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent("ruvector-kalshi/0.1")
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
        let base: String = base_url.into();
        // Parse once at construction. For malformed URLs fall back to the
        // caller-supplied string (same behavior as the old path lookup).
        let base_path: String = reqwest::Url::parse(&base)
            .map(|u| u.path().trim_end_matches('/').to_string())
            .unwrap_or_else(|_| "".to_string());
        Ok(Self {
            base_url: Arc::from(base.into_boxed_str()),
            base_path: Arc::from(base_path.into_boxed_str()),
            signer,
            http,
            limiter: Arc::new(RateLimiter::new(burst, per_sec)),
        })
    }

    /// Path used in the signature must be the full `/trade-api/v2/...` path,
    /// not the host-relative fragment, per Kalshi's spec.
    ///
    /// Uses the pre-computed `base_path` so there is no URL parse per call.
    fn sig_path_for(&self, path: &str) -> String {
        let p = if path.starts_with('/') { path } else { &format!("/{path}")[..] };
        // Strip any query string for the signature base — Kalshi signs only
        // the path component.
        let path_only = match p.find('?') {
            Some(i) => &p[..i],
            None => p,
        };
        format!("{}{}", self.base_path, path_only)
    }

    async fn send<R: for<'de> serde::Deserialize<'de>>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&impl serde::Serialize>,
    ) -> Result<R> {
        self.limiter.acquire().await;
        let url = url_join(&self.base_url, path);
        let sig_path = self.sig_path_for(path);
        let headers = self.signer.sign_now(method.as_str(), &sig_path);

        let mut rb = self.http.request(method, &url);
        rb = headers.apply(rb);
        if let Some(b) = body {
            rb = rb.json(b);
        }

        let resp = rb.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(KalshiError::Api {
                status: status.as_u16(),
                body,
            });
        }
        let parsed = resp.json::<R>().await?;
        Ok(parsed)
    }

    pub async fn list_markets(&self, status: Option<&str>) -> Result<Vec<Market>> {
        let path = match status {
            Some(s) => format!("/markets?status={s}"),
            None => "/markets".into(),
        };
        let resp: MarketsResponse = self.send(reqwest::Method::GET, &path, NO_BODY).await?;
        Ok(resp.markets)
    }

    pub async fn orderbook(&self, ticker: &str) -> Result<OrderbookSnapshot> {
        let path = format!("/markets/{ticker}/orderbook");
        let resp: OrderbookResponse = self.send(reqwest::Method::GET, &path, NO_BODY).await?;
        Ok(resp.orderbook)
    }

    /// Place a new order. Refuses to run unless `KALSHI_ENABLE_LIVE=1`.
    pub async fn post_order(&self, order: &NewOrder) -> Result<OrderAck> {
        require_live_flag()?;
        self.send(reqwest::Method::POST, "/portfolio/orders", Some(order))
            .await
    }

    /// Cancel an open order. Refuses to run unless `KALSHI_ENABLE_LIVE=1`.
    pub async fn cancel_order(&self, order_id: &str) -> Result<CancelResponse> {
        require_live_flag()?;
        let path = format!("/portfolio/orders/{order_id}");
        self.send(reqwest::Method::DELETE, &path, NO_BODY).await
    }

    /// Amend an existing open order's price or size. Refuses unless
    /// `KALSHI_ENABLE_LIVE=1`. Kalshi's endpoint is a PATCH.
    pub async fn amend_order(&self, order_id: &str, amend: &AmendOrder) -> Result<OrderAck> {
        require_live_flag()?;
        let path = format!("/portfolio/orders/{order_id}/amend");
        self.send(reqwest::Method::POST, &path, Some(amend)).await
    }
}

fn require_live_flag() -> Result<()> {
    if std::env::var("KALSHI_ENABLE_LIVE").ok().as_deref() == Some("1") {
        Ok(())
    } else {
        Err(KalshiError::Api {
            status: 0,
            body: "live trading disabled (set KALSHI_ENABLE_LIVE=1 to enable)".into(),
        })
    }
}

const NO_BODY: Option<&()> = None;

fn url_join(base: &str, path: &str) -> String {
    if path.starts_with("http://") || path.starts_with("https://") {
        return path.to_string();
    }
    let b = base.trim_end_matches('/');
    let p = if path.starts_with('/') { path.to_string() } else { format!("/{path}") };
    format!("{b}{p}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Signer;
    use rsa::pkcs1::EncodeRsaPrivateKey;
    use rsa::RsaPrivateKey;

    fn test_signer() -> Signer {
        let mut rng = rand::thread_rng();
        let key = RsaPrivateKey::new(&mut rng, 2048).unwrap();
        let pem = key.to_pkcs1_pem(rsa::pkcs1::LineEnding::LF).unwrap().to_string();
        Signer::from_pem("test-key", &pem).unwrap()
    }

    #[test]
    fn url_join_handles_trailing_and_leading_slashes() {
        assert_eq!(
            url_join("https://example.com/trade-api/v2/", "/markets"),
            "https://example.com/trade-api/v2/markets"
        );
        assert_eq!(
            url_join("https://example.com/trade-api/v2", "markets"),
            "https://example.com/trade-api/v2/markets"
        );
    }

    #[test]
    fn sig_path_uses_full_url_path() {
        let client = RestClient::new(
            "https://trading-api.kalshi.com/trade-api/v2",
            test_signer(),
        )
        .unwrap();
        let p = client.sig_path_for("/markets");
        assert_eq!(p, "/trade-api/v2/markets");
    }

    #[tokio::test]
    async fn post_order_refuses_without_live_flag() {
        // Ensure the flag is not set.
        std::env::remove_var("KALSHI_ENABLE_LIVE");
        let client = RestClient::new(
            "https://trading-api.kalshi.com/trade-api/v2",
            test_signer(),
        )
        .unwrap();
        let order = NewOrder {
            ticker: "X".into(),
            action: crate::models::OrderAction::Buy,
            side: crate::models::OrderSide::Yes,
            order_type: crate::models::OrderType::Limit,
            count: 1,
            yes_price: Some(24),
            no_price: None,
            client_order_id: "t-1".into(),
        };
        let err = client.post_order(&order).await.unwrap_err();
        match err {
            KalshiError::Api { status: 0, body } => {
                assert!(body.contains("live trading disabled"));
            }
            other => panic!("expected Api status=0 error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn cancel_order_refuses_without_live_flag() {
        std::env::remove_var("KALSHI_ENABLE_LIVE");
        let client = RestClient::new(
            "https://trading-api.kalshi.com/trade-api/v2",
            test_signer(),
        )
        .unwrap();
        let err = client.cancel_order("some-order-id").await.unwrap_err();
        assert!(matches!(err, KalshiError::Api { status: 0, .. }));
    }

    #[tokio::test]
    async fn amend_order_refuses_without_live_flag() {
        std::env::remove_var("KALSHI_ENABLE_LIVE");
        let client = RestClient::new(
            "https://trading-api.kalshi.com/trade-api/v2",
            test_signer(),
        )
        .unwrap();
        let amend = crate::models::AmendOrder {
            yes_price: Some(25),
            ..Default::default()
        };
        let err = client.amend_order("some-order-id", &amend).await.unwrap_err();
        assert!(matches!(err, KalshiError::Api { status: 0, .. }));
    }
}
