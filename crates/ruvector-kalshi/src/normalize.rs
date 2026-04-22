//! Normalize Kalshi payloads into [`neural_trader_core::MarketEvent`] so the
//! existing coherence, attention, CNN, and replay pipelines can consume
//! Kalshi as just another venue.
//!
//! Symbol-id assignment is deterministic via FNV-1a over the Kalshi ticker
//! so the same contract always maps to the same `symbol_id` across runs
//! without needing an external registry.

use chrono::DateTime;
use neural_trader_core::{EventType, MarketEvent, Side};

use crate::models::{
    KalshiTrade, OrderbookSnapshot, WsFill, WsOrderbook, WsOrderbookDelta, WsTicker, WsTrade,
};
use crate::{KalshiError, Result, KALSHI_PRICE_FP_SCALE, KALSHI_VENUE_ID};

/// Stable, deterministic symbol id from a Kalshi ticker. FNV-1a 32-bit,
/// ignoring case so `"FED-DEC23"` and `"fed-dec23"` collide (desired).
pub fn symbol_id_for(ticker: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for b in ticker.as_bytes() {
        let c = b.to_ascii_lowercase();
        hash ^= c as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// Convert a Kalshi cent price (0..=100) into canonical fixed-point.
pub fn cents_to_fp(cents: i64) -> i64 {
    cents.saturating_mul(KALSHI_PRICE_FP_SCALE)
}

fn parse_iso_ns(ts: &str) -> Result<u64> {
    let dt = DateTime::parse_from_rfc3339(ts)
        .map_err(|e| KalshiError::Normalize(format!("parse ts {ts}: {e}")))?;
    let ns = dt
        .timestamp_nanos_opt()
        .ok_or_else(|| KalshiError::Normalize(format!("ts {ts} out of range for i64 ns")))?;
    Ok(ns.max(0) as u64)
}

fn ms_to_ns(ts_ms: Option<i64>) -> u64 {
    ts_ms
        .map(|t| (t.max(0) as u64).saturating_mul(1_000_000))
        .unwrap_or_else(now_ns)
}

fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn empty_event_id() -> [u8; 16] {
    [0u8; 16]
}

fn event_id_from_str(key: &str) -> [u8; 16] {
    // 128-bit FNV-1a. Produces a stable id from trade_id / order_id strings.
    let mut hash: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    for b in key.as_bytes() {
        hash ^= *b as u128;
        hash = hash.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013B);
    }
    hash.to_le_bytes()
}

// ---------------------------------------------------------------------------
// REST normalizers
// ---------------------------------------------------------------------------

/// Normalize a REST `KalshiTrade` into a canonical Trade event.
pub fn trade_to_event(t: &KalshiTrade, seq: u64) -> Result<MarketEvent> {
    let ts_ns = parse_iso_ns(&t.created_time)?;
    let price_cents = t.yes_price.unwrap_or(0);
    let side = match t.taker_side.as_deref() {
        Some("yes") => Some(Side::Bid),
        Some("no") => Some(Side::Ask),
        _ => None,
    };
    Ok(MarketEvent {
        event_id: event_id_from_str(&t.trade_id),
        ts_exchange_ns: ts_ns,
        ts_ingest_ns: now_ns(),
        venue_id: KALSHI_VENUE_ID,
        symbol_id: symbol_id_for(&t.ticker),
        event_type: EventType::Trade,
        side,
        price_fp: cents_to_fp(price_cents),
        qty_fp: t.count.saturating_mul(KALSHI_PRICE_FP_SCALE),
        order_id_hash: None,
        participant_id_hash: None,
        flags: 0,
        seq,
    })
}

/// Expand an orderbook snapshot into per-level MarketEvents. Each Kalshi
/// level becomes one `BookSnapshot` event with the level's price/size.
pub fn orderbook_to_events(
    ticker: &str,
    ts_ns: u64,
    ob: &OrderbookSnapshot,
    starting_seq: u64,
) -> Vec<MarketEvent> {
    let sym = symbol_id_for(ticker);
    let mut out = Vec::with_capacity(ob.yes.len() + ob.no.len());
    let mut seq = starting_seq;
    for [price, size] in &ob.yes {
        out.push(MarketEvent {
            event_id: empty_event_id(),
            ts_exchange_ns: ts_ns,
            ts_ingest_ns: now_ns(),
            venue_id: KALSHI_VENUE_ID,
            symbol_id: sym,
            event_type: EventType::BookSnapshot,
            side: Some(Side::Bid),
            price_fp: cents_to_fp(*price),
            qty_fp: size.saturating_mul(KALSHI_PRICE_FP_SCALE),
            order_id_hash: None,
            participant_id_hash: None,
            flags: 0,
            seq,
        });
        seq = seq.wrapping_add(1);
    }
    for [price, size] in &ob.no {
        out.push(MarketEvent {
            event_id: empty_event_id(),
            ts_exchange_ns: ts_ns,
            ts_ingest_ns: now_ns(),
            venue_id: KALSHI_VENUE_ID,
            symbol_id: sym,
            event_type: EventType::BookSnapshot,
            side: Some(Side::Ask),
            price_fp: cents_to_fp(*price),
            qty_fp: size.saturating_mul(KALSHI_PRICE_FP_SCALE),
            order_id_hash: None,
            participant_id_hash: None,
            flags: 0,
            seq,
        });
        seq = seq.wrapping_add(1);
    }
    out
}

// ---------------------------------------------------------------------------
// WebSocket normalizers
// ---------------------------------------------------------------------------

pub fn ws_ticker_to_event(t: &WsTicker, seq: u64) -> MarketEvent {
    // Use the mid as the canonical price. Fallback to `price` if bids/asks
    // absent (end-of-day tickers).
    let price = match (t.yes_bid, t.yes_ask, t.price) {
        (Some(b), Some(a), _) => (b + a) / 2,
        (_, _, Some(p)) => p,
        _ => 0,
    };
    MarketEvent {
        event_id: empty_event_id(),
        ts_exchange_ns: ms_to_ns(t.ts),
        ts_ingest_ns: now_ns(),
        venue_id: KALSHI_VENUE_ID,
        symbol_id: symbol_id_for(&t.market_ticker),
        event_type: EventType::VenueStatus,
        side: None,
        price_fp: cents_to_fp(price),
        qty_fp: 0,
        order_id_hash: None,
        participant_id_hash: None,
        flags: 0,
        seq,
    }
}

pub fn ws_trade_to_event(t: &WsTrade, seq: u64) -> MarketEvent {
    let side = match t.taker_side.as_deref() {
        Some("yes") => Some(Side::Bid),
        Some("no") => Some(Side::Ask),
        _ => None,
    };
    MarketEvent {
        event_id: empty_event_id(),
        ts_exchange_ns: ms_to_ns(t.ts),
        ts_ingest_ns: now_ns(),
        venue_id: KALSHI_VENUE_ID,
        symbol_id: symbol_id_for(&t.market_ticker),
        event_type: EventType::Trade,
        side,
        price_fp: cents_to_fp(t.yes_price.unwrap_or(0)),
        qty_fp: t.count.saturating_mul(KALSHI_PRICE_FP_SCALE),
        order_id_hash: None,
        participant_id_hash: None,
        flags: 0,
        seq,
    }
}

pub fn ws_orderbook_to_events(ob: &WsOrderbook, starting_seq: u64) -> Vec<MarketEvent> {
    let ts_ns = ms_to_ns(ob.ts);
    orderbook_to_events(
        &ob.market_ticker,
        ts_ns,
        &OrderbookSnapshot {
            yes: ob.yes.clone(),
            no: ob.no.clone(),
        },
        starting_seq,
    )
}

pub fn ws_orderbook_delta_to_event(d: &WsOrderbookDelta, seq: u64) -> MarketEvent {
    let side = match d.side.as_str() {
        "yes" => Some(Side::Bid),
        "no" => Some(Side::Ask),
        _ => None,
    };
    let event_type = if d.delta > 0 {
        EventType::NewOrder
    } else {
        EventType::CancelOrder
    };
    MarketEvent {
        event_id: empty_event_id(),
        ts_exchange_ns: ms_to_ns(d.ts),
        ts_ingest_ns: now_ns(),
        venue_id: KALSHI_VENUE_ID,
        symbol_id: symbol_id_for(&d.market_ticker),
        event_type,
        side,
        price_fp: cents_to_fp(d.price),
        qty_fp: d.delta.abs().saturating_mul(KALSHI_PRICE_FP_SCALE),
        order_id_hash: None,
        participant_id_hash: None,
        flags: 0,
        seq,
    }
}

pub fn ws_fill_to_event(f: &WsFill, seq: u64) -> MarketEvent {
    let side = match f.side.as_str() {
        "yes" => Some(Side::Bid),
        "no" => Some(Side::Ask),
        _ => None,
    };
    MarketEvent {
        event_id: event_id_from_str(&f.order_id),
        ts_exchange_ns: ms_to_ns(f.ts),
        ts_ingest_ns: now_ns(),
        venue_id: KALSHI_VENUE_ID,
        symbol_id: symbol_id_for(&f.market_ticker),
        event_type: EventType::Trade,
        side,
        price_fp: cents_to_fp(f.yes_price.unwrap_or(0)),
        qty_fp: f.count.saturating_mul(KALSHI_PRICE_FP_SCALE),
        order_id_hash: Some(event_id_from_str(&f.order_id)),
        participant_id_hash: None,
        flags: 1, // flag bit 0 = own-fill
        seq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{KalshiTrade, OrderbookSnapshot, WsOrderbookDelta};

    #[test]
    fn symbol_id_is_deterministic_and_case_insensitive() {
        assert_eq!(symbol_id_for("FED-DEC23"), symbol_id_for("fed-dec23"));
        assert_ne!(symbol_id_for("FED-DEC23"), symbol_id_for("FED-DEC24"));
    }

    #[test]
    fn cents_to_fp_scales() {
        assert_eq!(cents_to_fp(24), 24_000_000);
        assert_eq!(cents_to_fp(0), 0);
    }

    #[test]
    fn trade_to_event_maps_side_and_price() {
        let t = KalshiTrade {
            ticker: "FED-DEC23".into(),
            trade_id: "trade-abc".into(),
            yes_price: Some(34),
            no_price: Some(66),
            count: 5,
            taker_side: Some("yes".into()),
            created_time: "2026-04-20T18:00:00Z".into(),
        };
        let e = trade_to_event(&t, 7).unwrap();
        assert_eq!(e.event_type, EventType::Trade);
        assert_eq!(e.side, Some(Side::Bid));
        assert_eq!(e.venue_id, KALSHI_VENUE_ID);
        assert_eq!(e.price_fp, cents_to_fp(34));
        assert_eq!(e.qty_fp, 5 * KALSHI_PRICE_FP_SCALE);
        assert_eq!(e.seq, 7);
        assert_ne!(e.event_id, [0u8; 16], "trade id must seed event_id");
    }

    #[test]
    fn orderbook_levels_expanded() {
        let ob = OrderbookSnapshot {
            yes: vec![[24, 100], [23, 200]],
            no: vec![[76, 150]],
        };
        let events = orderbook_to_events("FED-DEC23", 1_000_000_000, &ob, 0);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].side, Some(Side::Bid));
        assert_eq!(events[0].price_fp, cents_to_fp(24));
        assert_eq!(events[0].qty_fp, 100 * KALSHI_PRICE_FP_SCALE);
        assert_eq!(events[2].side, Some(Side::Ask));
        // Seq monotonic.
        assert_eq!(events[0].seq, 0);
        assert_eq!(events[2].seq, 2);
    }

    #[test]
    fn ws_delta_positive_is_new_order_negative_is_cancel() {
        let add = WsOrderbookDelta {
            market_ticker: "FED-DEC23".into(),
            side: "yes".into(),
            price: 24,
            delta: 5,
            ts: Some(1_700_000_000_000),
        };
        let remove = WsOrderbookDelta {
            market_ticker: "FED-DEC23".into(),
            side: "no".into(),
            price: 76,
            delta: -3,
            ts: None,
        };
        assert_eq!(
            ws_orderbook_delta_to_event(&add, 0).event_type,
            EventType::NewOrder
        );
        assert_eq!(
            ws_orderbook_delta_to_event(&remove, 1).event_type,
            EventType::CancelOrder
        );
        // qty always positive.
        assert_eq!(
            ws_orderbook_delta_to_event(&remove, 1).qty_fp,
            3 * KALSHI_PRICE_FP_SCALE
        );
    }
}
