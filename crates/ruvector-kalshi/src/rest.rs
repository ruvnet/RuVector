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

use crate::auth::Signer;
use crate::models::{
    Market, MarketsResponse, NewOrder, OrderAck, OrderbookResponse, OrderbookSnapshot,
};
use crate::{KalshiError, Result};

#[derive(Clone)]
pub struct RestClient {
    base_url: String,
    signer: Signer,
    http: reqwest::Client,
}

impl RestClient {
    pub fn new(base_url: impl Into<String>, signer: Signer) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent("ruvector-kalshi/0.1")
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
        Ok(Self {
            base_url: base_url.into(),
            signer,
            http,
        })
    }

    /// Path used in the signature must be the full `/trade-api/v2/...` path,
    /// not the host-relative fragment, per Kalshi's spec.
    fn sig_path_for(&self, path: &str) -> String {
        // `base_url` already ends with `/trade-api/v2`; the sig path is the
        // full URL path component.
        let url = url_join(&self.base_url, path);
        match reqwest::Url::parse(&url) {
            Ok(u) => u.path().to_string(),
            Err(_) => path.to_string(),
        }
    }

    async fn send<R: for<'de> serde::Deserialize<'de>>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&impl serde::Serialize>,
    ) -> Result<R> {
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
        if std::env::var("KALSHI_ENABLE_LIVE").ok().as_deref() != Some("1") {
            return Err(KalshiError::Api {
                status: 0,
                body: "live trading disabled (set KALSHI_ENABLE_LIVE=1 to enable)".into(),
            });
        }
        self.send(reqwest::Method::POST, "/portfolio/orders", Some(order))
            .await
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
}
