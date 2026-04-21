//! pi.ruv.io brain integration — persist market resolutions and strategy
//! P&L as shared memories so other ruvector components can learn from
//! Kalshi outcomes.
//!
//! The brain runs at `https://pi.ruv.io/v1/memories` and expects a bearer
//! token stored in the GCS secret `BRAIN_SYSTEM_KEY`. This module never
//! sends raw credentials, order IDs, or counterparty info — payloads are
//! restricted to market outcomes and strategy metadata.

use serde::{Deserialize, Serialize};

use crate::{KalshiError, Result};

/// Default brain base URL.
pub const BRAIN_URL: &str = "https://pi.ruv.io";

/// Memory payload posted to the brain. Mirrors the `brain_share` tool
/// shape described in CLAUDE.md (category + title + content + tags).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMemory {
    pub category: &'static str,
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
}

impl SharedMemory {
    /// Build a `pattern`-category memory describing a market outcome and
    /// strategy P&L. The content is plain Markdown and contains no key
    /// material or private identifiers.
    pub fn market_resolution(
        ticker: &str,
        resolution: Resolution,
        strategy: &str,
        pnl_cents: i64,
        notional_cents: i64,
    ) -> Self {
        let resolution_str = match resolution {
            Resolution::Yes => "YES",
            Resolution::No => "NO",
            Resolution::Void => "VOID",
        };
        let pnl_str = format!("{:+} cents", pnl_cents);
        let roi_bps = if notional_cents > 0 {
            (pnl_cents as f64 / notional_cents as f64 * 10_000.0).round() as i64
        } else {
            0
        };
        let content = format!(
            "Kalshi market `{ticker}` resolved {resolution_str}.\n\n\
             - Strategy: `{strategy}`\n\
             - Realized P&L: {pnl_str}\n\
             - Notional traded: {notional_cents} cents\n\
             - ROI: {roi_bps} bps\n"
        );
        let tags = vec![
            "kalshi".into(),
            "event-market".into(),
            format!("resolution-{}", resolution_str.to_lowercase()),
            format!("strategy-{strategy}"),
        ];
        Self {
            category: "pattern",
            title: format!("Kalshi {ticker} resolved {resolution_str}"),
            content,
            tags,
        }
    }
}

/// Binary outcome of a Kalshi market at settlement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    Yes,
    No,
    /// Market was voided (e.g., canceled, uncleared).
    Void,
}

#[derive(Clone)]
pub struct BrainClient {
    base_url: String,
    api_key: String,
    http: reqwest::Client,
}

impl std::fmt::Debug for BrainClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrainClient")
            .field("base_url", &self.base_url)
            .field("api_key_len", &self.api_key.len())
            .finish()
    }
}

impl BrainClient {
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::with_base(BRAIN_URL, api_key)
    }

    pub fn with_base(base_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent("ruvector-kalshi-brain/0.1")
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
        Ok(Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            http,
        })
    }

    /// Share a memory with the brain. Returns the brain-assigned id on
    /// success or a typed API error.
    pub async fn share(&self, memory: &SharedMemory) -> Result<String> {
        let url = format!("{}/v1/memories", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(memory)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(KalshiError::Api {
                status: status.as_u16(),
                body,
            });
        }
        #[derive(Deserialize)]
        struct Ack {
            id: Option<String>,
        }
        let ack = resp.json::<Ack>().await?;
        Ok(ack.id.unwrap_or_else(|| "ok".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_memory_contains_expected_fields() {
        let m = SharedMemory::market_resolution(
            "FED-DEC26",
            Resolution::Yes,
            "ev-kelly",
            420,
            10_000,
        );
        assert_eq!(m.category, "pattern");
        assert!(m.title.contains("FED-DEC26"));
        assert!(m.title.contains("YES"));
        assert!(m.content.contains("ev-kelly"));
        assert!(m.content.contains("+420"));
        assert!(m.tags.contains(&"kalshi".to_string()));
        assert!(m.tags.contains(&"resolution-yes".to_string()));
    }

    #[test]
    fn client_debug_does_not_leak_key() {
        let c = BrainClient::new("super-secret-token-9999").unwrap();
        let s = format!("{c:?}");
        assert!(!s.contains("super-secret-token"));
    }

    #[test]
    fn memory_serializes_cleanly() {
        let m = SharedMemory::market_resolution("X", Resolution::No, "s", -100, 5_000);
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("\"category\":\"pattern\""));
        assert!(json.contains("\"tags\""));
        // No stray fields.
        assert!(!json.contains("api_key"));
    }

    #[test]
    fn zero_notional_rocks_roi_zero() {
        let m = SharedMemory::market_resolution("X", Resolution::Void, "s", 0, 0);
        assert!(m.content.contains("ROI: 0 bps"));
    }
}
