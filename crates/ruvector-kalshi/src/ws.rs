//! WebSocket ingest scaffolding for Kalshi's market-data stream.
//!
//! This module intentionally keeps the network layer *pluggable*. The
//! default build does not pull in `tokio-tungstenite` — callers feed raw
//! JSON frames into [`FeedDecoder::decode`], which produces
//! [`neural_trader_core::MarketEvent`] records ready for the coherence /
//! replay pipelines.
//!
//! When a tungstenite-based transport is needed, add
//! `tokio-tungstenite` to this crate and wire `connect` + `pump` inside a
//! `#[cfg(feature = "tungstenite")]` module. Keeping the decoder feature-
//! free means unit tests cover the parsing path without requiring a
//! network runtime.

use neural_trader_core::MarketEvent;

use crate::models::{WsEnvelope, WsMessage};
use crate::normalize::{
    ws_fill_to_event, ws_orderbook_delta_to_event, ws_orderbook_to_events, ws_ticker_to_event,
    ws_trade_to_event,
};
use crate::Result;

/// Decodes raw JSON WebSocket frames into canonical `MarketEvent`s.
///
/// Stateful only in its monotonically increasing sequence counter, so it
/// is cheap to construct per connection.
#[derive(Debug, Default)]
pub struct FeedDecoder {
    next_seq: u64,
}

impl FeedDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode one JSON frame. Unknown message kinds yield an empty vector
    /// (not an error) so forward-compatible payloads don't kill the feed.
    pub fn decode(&mut self, frame: &str) -> Result<Vec<MarketEvent>> {
        let env: WsEnvelope = serde_json::from_str(frame)?;
        let msg = WsMessage::from_envelope(env)?;
        Ok(self.decode_msg(&msg))
    }

    fn decode_msg(&mut self, msg: &WsMessage) -> Vec<MarketEvent> {
        match msg {
            WsMessage::Ticker(t) => vec![self.tick(|s| ws_ticker_to_event(t, s))],
            WsMessage::Trade(t) => vec![self.tick(|s| ws_trade_to_event(t, s))],
            WsMessage::OrderbookSnapshot(ob) => {
                let events = ws_orderbook_to_events(ob, self.next_seq);
                self.next_seq = self
                    .next_seq
                    .wrapping_add(events.len() as u64);
                events
            }
            WsMessage::OrderbookDelta(d) => {
                vec![self.tick(|s| ws_orderbook_delta_to_event(d, s))]
            }
            WsMessage::Fill(f) => vec![self.tick(|s| ws_fill_to_event(f, s))],
            WsMessage::Other => Vec::new(),
        }
    }

    fn tick<F: FnOnce(u64) -> MarketEvent>(&mut self, f: F) -> MarketEvent {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        f(seq)
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_trader_core::EventType;

    #[test]
    fn decodes_ticker_then_trade_monotonically() {
        let mut dec = FeedDecoder::new();
        let events = dec
            .decode(r#"{"type":"ticker","msg":{"market_ticker":"X","yes_bid":10,"yes_ask":12}}"#)
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].seq, 0);
        assert_eq!(events[0].event_type, EventType::VenueStatus);

        let events = dec
            .decode(r#"{"type":"trade","msg":{"market_ticker":"X","yes_price":11,"count":3}}"#)
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].seq, 1);
        assert_eq!(events[0].event_type, EventType::Trade);
    }

    #[test]
    fn decodes_snapshot_with_multiple_levels() {
        let mut dec = FeedDecoder::new();
        let frame = r#"{
            "type":"orderbook_snapshot",
            "msg":{
                "market_ticker":"X",
                "yes":[[24,100],[23,200]],
                "no":[[76,150]]
            }
        }"#;
        let events = dec.decode(frame).unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(dec.next_seq(), 3);
    }

    #[test]
    fn unknown_message_is_silent() {
        let mut dec = FeedDecoder::new();
        let events = dec.decode(r#"{"type":"heartbeat","msg":{}}"#).unwrap();
        assert!(events.is_empty());
        assert_eq!(dec.next_seq(), 0);
    }

    #[test]
    fn malformed_json_returns_error() {
        let mut dec = FeedDecoder::new();
        let err = dec.decode("not json").unwrap_err();
        assert!(matches!(err, crate::KalshiError::Decode(_)));
    }
}
