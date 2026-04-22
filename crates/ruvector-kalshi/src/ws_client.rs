//! Live WebSocket transport wiring the [`crate::ws::FeedDecoder`] to
//! Kalshi's market-data stream via `tokio-tungstenite`.
//!
//! Keeps three concerns separate:
//! - **Connect + handshake**: sign the WS `GET` path with [`crate::auth::Signer`]
//!   and open a TLS WebSocket.
//! - **Subscribe**: send the initial subscribe command over the socket.
//! - **Pump**: stream each text frame through `FeedDecoder` and forward
//!   canonical `MarketEvent`s into a `tokio::sync::mpsc` channel.
//!
//! Reconnection is caller-driven: the pump returns when the socket closes
//! and the caller decides whether to back off and retry. A
//! [`reconnect_forever`] convenience helper implements exponential backoff
//! up to a ceiling.
//!
//! Offline build/test path: all network I/O is isolated in
//! [`connect`] and [`pump_frames`]; unit tests exercise the subscribe
//! envelope + frame routing through mocked inputs.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use neural_trader_core::MarketEvent;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, Message},
    MaybeTlsStream, WebSocketStream,
};
use tracing::{debug, warn};

use crate::auth::Signer;
use crate::ws::FeedDecoder;
use crate::{KalshiError, Result, KALSHI_WS_URL};

/// Subscription request envelope (Kalshi REST-style over WS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscribe {
    pub id: u64,
    pub cmd: &'static str,
    pub params: SubscribeParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeParams {
    pub channels: Vec<String>,
    pub market_tickers: Vec<String>,
}

impl Subscribe {
    /// Build a subscribe command for a set of tickers and channels.
    pub fn new(id: u64, channels: Vec<String>, tickers: Vec<String>) -> Self {
        Self {
            id,
            cmd: "subscribe",
            params: SubscribeParams {
                channels,
                market_tickers: tickers,
            },
        }
    }

    pub fn to_frame(&self) -> Result<String> {
        serde_json::to_string(self).map_err(KalshiError::from)
    }
}

pub type WsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

/// Open an authenticated WS connection to Kalshi.
pub async fn connect(signer: &Signer, url: &str) -> Result<WsStream> {
    // Kalshi accepts the same RSA-PSS signature scheme on WS upgrade;
    // sign path + method "GET" just like REST.
    let path = reqwest::Url::parse(url)
        .map(|u| u.path().to_string())
        .unwrap_or_else(|_| "/trade-api/ws/v2".to_string());
    let headers = signer.sign_now("GET", &path);

    let mut req = url
        .into_client_request()
        .map_err(|e| KalshiError::Secret(format!("ws request build: {e}")))?;
    let h = req.headers_mut();
    h.insert(
        "KALSHI-ACCESS-KEY",
        headers
            .access_key
            .parse()
            .map_err(|e| KalshiError::Secret(format!("access-key header: {e}")))?,
    );
    h.insert(
        "KALSHI-ACCESS-TIMESTAMP",
        headers
            .timestamp_ms
            .parse()
            .map_err(|e| KalshiError::Secret(format!("ts header: {e}")))?,
    );
    h.insert(
        "KALSHI-ACCESS-SIGNATURE",
        headers
            .signature_b64
            .parse()
            .map_err(|e| KalshiError::Secret(format!("sig header: {e}")))?,
    );

    let (stream, resp) = connect_async(req)
        .await
        .map_err(|e| KalshiError::Secret(format!("ws connect: {e}")))?;
    debug!(?resp, "kalshi ws connected");
    Ok(stream)
}

/// Send a subscribe command on an open socket.
pub async fn subscribe(stream: &mut WsStream, sub: &Subscribe) -> Result<()> {
    let frame = sub.to_frame()?;
    stream
        .send(Message::Text(frame))
        .await
        .map_err(|e| KalshiError::Secret(format!("ws send: {e}")))?;
    Ok(())
}

/// Pump text frames from `stream` through `FeedDecoder` into the channel.
/// Returns when the socket closes or an error occurs.
pub async fn pump_frames(mut stream: WsStream, tx: mpsc::Sender<MarketEvent>) -> Result<()> {
    let mut decoder = FeedDecoder::new();
    while let Some(next) = stream.next().await {
        let msg = match next {
            Ok(m) => m,
            Err(e) => {
                warn!("ws recv error: {e}");
                return Err(KalshiError::Secret(format!("ws recv: {e}")));
            }
        };
        match msg {
            Message::Text(txt) => match decoder.decode(&txt) {
                Ok(events) => {
                    for evt in events {
                        if tx.send(evt).await.is_err() {
                            debug!("consumer dropped; stopping pump");
                            return Ok(());
                        }
                    }
                }
                Err(e) => warn!("decode error: {e} frame={txt}"),
            },
            Message::Ping(payload) => {
                let _ = stream.send(Message::Pong(payload)).await;
            }
            Message::Close(_) => {
                debug!("server sent close");
                return Ok(());
            }
            _ => {}
        }
    }
    Ok(())
}

/// Convenience helper: connect, subscribe, and pump with exponential
/// backoff reconnection. Returns only when the `tx` end of the channel
/// is closed by the consumer.
pub async fn reconnect_forever(
    signer: Signer,
    url: Option<String>,
    subscribe_req: Subscribe,
    tx: mpsc::Sender<MarketEvent>,
) {
    let url = url.unwrap_or_else(|| KALSHI_WS_URL.to_string());
    let mut backoff = Duration::from_millis(500);
    let cap = Duration::from_secs(30);
    loop {
        match connect(&signer, &url).await {
            Ok(mut stream) => {
                if let Err(e) = subscribe(&mut stream, &subscribe_req).await {
                    warn!("subscribe failed: {e}");
                } else {
                    backoff = Duration::from_millis(500);
                    if let Err(e) = pump_frames(stream, tx.clone()).await {
                        warn!("pump ended with error: {e}");
                    }
                }
            }
            Err(e) => warn!("connect failed: {e}"),
        }
        if tx.is_closed() {
            debug!("consumer closed channel; stopping reconnect loop");
            return;
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscribe_serializes_correctly() {
        let sub = Subscribe::new(
            1,
            vec!["ticker".into(), "trade".into()],
            vec!["FED-DEC26".into()],
        );
        let frame = sub.to_frame().unwrap();
        assert!(frame.contains("\"cmd\":\"subscribe\""));
        assert!(frame.contains("\"ticker\""));
        assert!(frame.contains("FED-DEC26"));
        assert!(frame.contains("\"id\":1"));
    }
}
