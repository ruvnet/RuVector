//! Kalshi REST + WebSocket DTOs. Only the fields we actually consume are
//! declared; everything else is ignored by `#[serde(deny_unknown_fields)]`
//! being *off* — we intentionally accept forward-compatible payloads.

use serde::{Deserialize, Serialize};

/// Market metadata (GET /markets, /markets/{ticker}).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Market {
    pub ticker: String,
    pub event_ticker: Option<String>,
    pub title: Option<String>,
    pub status: Option<String>,
    pub yes_bid: Option<i64>,
    pub yes_ask: Option<i64>,
    pub no_bid: Option<i64>,
    pub no_ask: Option<i64>,
    pub last_price: Option<i64>,
    pub volume: Option<i64>,
    pub open_interest: Option<i64>,
    pub close_time: Option<String>,
    pub expiration_time: Option<String>,
}

/// Envelope for list-markets response.
#[derive(Debug, Clone, Deserialize)]
pub struct MarketsResponse {
    pub markets: Vec<Market>,
    pub cursor: Option<String>,
}

/// Single trade print (GET /markets/trades).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KalshiTrade {
    pub ticker: String,
    pub trade_id: String,
    pub yes_price: Option<i64>,
    pub no_price: Option<i64>,
    pub count: i64,
    pub taker_side: Option<String>, // "yes" | "no"
    pub created_time: String,       // ISO-8601
}

/// Orderbook snapshot (GET /markets/{ticker}/orderbook).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OrderbookSnapshot {
    /// Each entry is `[price_cents, contracts]`.
    pub yes: Vec<[i64; 2]>,
    pub no: Vec<[i64; 2]>,
}

/// Wrapper for orderbook GET envelope.
#[derive(Debug, Clone, Deserialize)]
pub struct OrderbookResponse {
    pub orderbook: OrderbookSnapshot,
}

/// Order placement payload (POST /portfolio/orders).
#[derive(Debug, Clone, Serialize)]
pub struct NewOrder {
    pub ticker: String,
    pub action: OrderAction,
    pub side: OrderSide,
    #[serde(rename = "type")]
    pub order_type: OrderType,
    pub count: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub yes_price: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_price: Option<i64>,
    pub client_order_id: String,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderAction {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderSide {
    Yes,
    No,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderType {
    Limit,
    Market,
}

/// Response from POST /portfolio/orders.
#[derive(Debug, Clone, Deserialize)]
pub struct OrderAck {
    pub order: OrderRecord,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OrderRecord {
    pub order_id: String,
    pub client_order_id: Option<String>,
    pub status: String,
    pub ticker: String,
    pub count: i64,
    pub filled_count: Option<i64>,
    pub remaining_count: Option<i64>,
}

/// Raw Kalshi WS envelope. `{"type": "...", "msg": {...}}`. `msg` is
/// kept as a `Value` so unknown `type` tags don't fail the parse — the
/// decoder routes on `msg_type`.
#[derive(Debug, Clone, Deserialize)]
pub struct WsEnvelope {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub msg: serde_json::Value,
}

/// Typed WS messages routed from [`WsEnvelope`] by the decoder.
#[derive(Debug, Clone)]
pub enum WsMessage {
    Ticker(WsTicker),
    Trade(WsTrade),
    OrderbookSnapshot(WsOrderbook),
    OrderbookDelta(WsOrderbookDelta),
    Fill(WsFill),
    /// Any non-data frame (heartbeat, ack, etc.) or unknown `type` tag.
    Other,
}

impl WsMessage {
    /// Decode an envelope into a typed message. Unknown type tags produce
    /// [`WsMessage::Other`] rather than an error so forward-compatible
    /// payloads don't kill the feed.
    pub fn from_envelope(env: WsEnvelope) -> serde_json::Result<Self> {
        Ok(match env.msg_type.as_str() {
            "ticker" => Self::Ticker(serde_json::from_value(env.msg)?),
            "trade" => Self::Trade(serde_json::from_value(env.msg)?),
            "orderbook_snapshot" => Self::OrderbookSnapshot(serde_json::from_value(env.msg)?),
            "orderbook_delta" => Self::OrderbookDelta(serde_json::from_value(env.msg)?),
            "fill" => Self::Fill(serde_json::from_value(env.msg)?),
            _ => Self::Other,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WsTicker {
    pub market_ticker: String,
    pub yes_bid: Option<i64>,
    pub yes_ask: Option<i64>,
    pub price: Option<i64>,
    pub ts: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WsTrade {
    pub market_ticker: String,
    pub yes_price: Option<i64>,
    pub no_price: Option<i64>,
    pub count: i64,
    pub taker_side: Option<String>,
    pub ts: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WsOrderbook {
    pub market_ticker: String,
    pub yes: Vec<[i64; 2]>,
    pub no: Vec<[i64; 2]>,
    pub ts: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WsOrderbookDelta {
    pub market_ticker: String,
    pub side: String,       // "yes" | "no"
    pub price: i64,
    pub delta: i64,
    pub ts: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WsFill {
    pub market_ticker: String,
    pub order_id: String,
    pub yes_price: Option<i64>,
    pub no_price: Option<i64>,
    pub count: i64,
    pub side: String,
    pub ts: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn market_deserializes_with_optional_fields() {
        let json = r#"{
            "ticker": "FED-23DEC-T3.00",
            "title": "Fed raises rates",
            "status": "active",
            "yes_bid": 24,
            "yes_ask": 26,
            "volume": 1200
        }"#;
        let m: Market = serde_json::from_str(json).unwrap();
        assert_eq!(m.ticker, "FED-23DEC-T3.00");
        assert_eq!(m.yes_bid, Some(24));
        assert!(m.no_bid.is_none());
    }

    #[test]
    fn ws_message_dispatch() {
        let json = r#"{"type":"ticker","msg":{"market_ticker":"X","yes_bid":10,"yes_ask":12}}"#;
        let env: WsEnvelope = serde_json::from_str(json).unwrap();
        let msg = WsMessage::from_envelope(env).unwrap();
        assert!(matches!(msg, WsMessage::Ticker(ref t) if t.market_ticker == "X"));
    }

    #[test]
    fn ws_message_unknown_kind_does_not_error() {
        let json = r#"{"type":"heartbeat","msg":{}}"#;
        let env: WsEnvelope = serde_json::from_str(json).unwrap();
        let msg = WsMessage::from_envelope(env).unwrap();
        assert!(matches!(msg, WsMessage::Other));
    }

    #[test]
    fn new_order_limit_serializes() {
        let o = NewOrder {
            ticker: "X".into(),
            action: OrderAction::Buy,
            side: OrderSide::Yes,
            order_type: OrderType::Limit,
            count: 10,
            yes_price: Some(24),
            no_price: None,
            client_order_id: "abc".into(),
        };
        let s = serde_json::to_string(&o).unwrap();
        assert!(s.contains("\"type\":\"limit\""));
        assert!(s.contains("\"action\":\"buy\""));
        assert!(s.contains("\"yes_price\":24"));
        // None field must be omitted.
        assert!(!s.contains("no_price"));
    }
}
