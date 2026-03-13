//! # neural-trader-wasm
//!
//! WASM bindings for the Neural Trader crates: market events,
//! coherence gates, and replay memory (ADR-085 / ADR-086).

use serde::Serialize;
use wasm_bindgen::prelude::*;

use neural_trader_coherence::{
    CoherenceDecision, CoherenceGate, GateConfig, GateContext, RegimeLabel, ThresholdGate,
};
use neural_trader_core::{EventType, MarketEvent, Side};
use neural_trader_replay::{MemoryStore, ReplaySegment, ReservoirStore};

// ---------------------------------------------------------------------------
// Init & utilities
// ---------------------------------------------------------------------------

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Returns the crate version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Smoke-test that the WASM module loaded correctly.
#[wasm_bindgen(js_name = "healthCheck")]
pub fn health_check() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Hex helpers for [u8; 16]
// ---------------------------------------------------------------------------

fn bytes16_to_hex(b: &[u8; 16]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn hex_to_bytes16_inner(s: &str) -> Result<[u8; 16], String> {
    let s = s.trim();
    // Strip optional 0x prefix for JS ergonomics.
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    if !s.is_ascii() || s.len() != 32 {
        return Err(
            "hex string must be exactly 32 ASCII hex chars (optional 0x prefix)".to_string(),
        );
    }
    let mut out = [0u8; 16];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

fn hex_to_bytes16(s: &str) -> Result<[u8; 16], JsValue> {
    hex_to_bytes16_inner(s).map_err(|e| JsValue::from_str(&e))
}

/// Serialize using BigInt-aware serializer to avoid u64 precision loss.
fn to_js<T: Serialize>(v: &T) -> Result<JsValue, JsValue> {
    let ser = serde_wasm_bindgen::Serializer::new().serialize_large_number_types_as_bigints(true);
    v.serialize(&ser)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ---------------------------------------------------------------------------
// Enum conversion macro
// ---------------------------------------------------------------------------

macro_rules! enum_convert {
    ($wasm:ident <=> $inner:path { $($variant:ident),+ $(,)? }) => {
        impl From<$inner> for $wasm {
            fn from(v: $inner) -> Self {
                match v { $( <$inner>::$variant => $wasm::$variant, )+ }
            }
        }
        impl From<$wasm> for $inner {
            fn from(v: $wasm) -> Self {
                match v { $( $wasm::$variant => <$inner>::$variant, )+ }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum EventTypeWasm {
    NewOrder = 0,
    ModifyOrder = 1,
    CancelOrder = 2,
    Trade = 3,
    BookSnapshot = 4,
    SessionMarker = 5,
    VenueStatus = 6,
}
enum_convert!(EventTypeWasm <=> EventType {
    NewOrder, ModifyOrder, CancelOrder, Trade, BookSnapshot, SessionMarker, VenueStatus
});

#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum SideWasm {
    Bid = 0,
    Ask = 1,
}
enum_convert!(SideWasm <=> Side { Bid, Ask });

#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum RegimeLabelWasm {
    Calm = 0,
    Normal = 1,
    Volatile = 2,
}
enum_convert!(RegimeLabelWasm <=> RegimeLabel { Calm, Normal, Volatile });

#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum SegmentKindWasm {
    HighUncertainty = 0,
    LargeImpact = 1,
    RegimeTransition = 2,
    StructuralAnomaly = 3,
    RareQueuePattern = 4,
    HeadDisagreement = 5,
    Routine = 6,
}
enum_convert!(SegmentKindWasm <=> neural_trader_replay::SegmentKind {
    HighUncertainty, LargeImpact, RegimeTransition, StructuralAnomaly,
    RareQueuePattern, HeadDisagreement, Routine
});

#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum NodeKindWasm {
    Symbol = 0,
    Venue = 1,
    PriceLevel = 2,
    Order = 3,
    Trade = 4,
    Event = 5,
    Participant = 6,
    TimeBucket = 7,
    Regime = 8,
    StrategyState = 9,
}
enum_convert!(NodeKindWasm <=> neural_trader_core::NodeKind {
    Symbol, Venue, PriceLevel, Order, Trade, Event, Participant,
    TimeBucket, Regime, StrategyState
});

#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum EdgeKindWasm {
    AtLevel = 0,
    NextTick = 1,
    Generated = 2,
    Matched = 3,
    ModifiedFrom = 4,
    CanceledBy = 5,
    BelongsToSymbol = 6,
    OnVenue = 7,
    InWindow = 8,
    CorrelatedWith = 9,
    InRegime = 10,
    AffectsState = 11,
}
enum_convert!(EdgeKindWasm <=> neural_trader_core::EdgeKind {
    AtLevel, NextTick, Generated, Matched, ModifiedFrom, CanceledBy,
    BelongsToSymbol, OnVenue, InWindow, CorrelatedWith, InRegime, AffectsState
});

#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum PropertyKeyWasm {
    VisibleDepth = 0,
    EstimatedHiddenDepth = 1,
    QueueLength = 2,
    LocalImbalance = 3,
    RefillRate = 4,
    DepletionRate = 5,
    SpreadDistance = 6,
    LocalRealizedVol = 7,
    CancelHazard = 8,
    FillHazard = 9,
    SlippageToMid = 10,
    PostTradeImpact = 11,
    InfluenceScore = 12,
    CoherenceContribution = 13,
    QueueEstimate = 14,
    Age = 15,
    ModifyCount = 16,
}
enum_convert!(PropertyKeyWasm <=> neural_trader_core::PropertyKey {
    VisibleDepth, EstimatedHiddenDepth, QueueLength, LocalImbalance,
    RefillRate, DepletionRate, SpreadDistance, LocalRealizedVol,
    CancelHazard, FillHazard, SlippageToMid, PostTradeImpact,
    InfluenceScore, CoherenceContribution, QueueEstimate, Age, ModifyCount
});

// ---------------------------------------------------------------------------
// MarketEventWasm
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct MarketEventWasm {
    inner: MarketEvent,
}

#[wasm_bindgen]
impl MarketEventWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        event_type: EventTypeWasm,
        symbol_id: u32,
        venue_id: u16,
        price_fp: i64,
        qty_fp: i64,
    ) -> Self {
        Self {
            inner: MarketEvent {
                event_id: [0u8; 16],
                ts_exchange_ns: 0,
                ts_ingest_ns: 0,
                venue_id,
                symbol_id,
                event_type: event_type.into(),
                side: None,
                price_fp,
                qty_fp,
                order_id_hash: None,
                participant_id_hash: None,
                flags: 0,
                seq: 0,
            },
        }
    }

    // --- Getters ---

    #[wasm_bindgen(getter, js_name = "eventId")]
    pub fn event_id(&self) -> String {
        bytes16_to_hex(&self.inner.event_id)
    }

    #[wasm_bindgen(getter, js_name = "tsExchangeNs")]
    pub fn ts_exchange_ns(&self) -> u64 {
        self.inner.ts_exchange_ns
    }

    #[wasm_bindgen(getter, js_name = "tsIngestNs")]
    pub fn ts_ingest_ns(&self) -> u64 {
        self.inner.ts_ingest_ns
    }

    #[wasm_bindgen(getter, js_name = "venueId")]
    pub fn venue_id(&self) -> u16 {
        self.inner.venue_id
    }

    #[wasm_bindgen(getter, js_name = "symbolId")]
    pub fn symbol_id(&self) -> u32 {
        self.inner.symbol_id
    }

    #[wasm_bindgen(getter, js_name = "eventType")]
    pub fn event_type(&self) -> EventTypeWasm {
        self.inner.event_type.into()
    }

    #[wasm_bindgen(getter)]
    pub fn side(&self) -> Option<SideWasm> {
        self.inner.side.map(|s| s.into())
    }

    #[wasm_bindgen(getter, js_name = "priceFp")]
    pub fn price_fp(&self) -> i64 {
        self.inner.price_fp
    }

    #[wasm_bindgen(getter, js_name = "qtyFp")]
    pub fn qty_fp(&self) -> i64 {
        self.inner.qty_fp
    }

    #[wasm_bindgen(getter)]
    pub fn flags(&self) -> u32 {
        self.inner.flags
    }

    #[wasm_bindgen(getter)]
    pub fn seq(&self) -> u64 {
        self.inner.seq
    }

    // --- Setters ---

    #[wasm_bindgen(setter, js_name = "eventId")]
    pub fn set_event_id(&mut self, hex: &str) -> Result<(), JsValue> {
        self.inner.event_id = hex_to_bytes16(hex)?;
        Ok(())
    }

    #[wasm_bindgen(setter, js_name = "tsExchangeNs")]
    pub fn set_ts_exchange_ns(&mut self, v: u64) {
        self.inner.ts_exchange_ns = v;
    }

    #[wasm_bindgen(setter, js_name = "tsIngestNs")]
    pub fn set_ts_ingest_ns(&mut self, v: u64) {
        self.inner.ts_ingest_ns = v;
    }

    #[wasm_bindgen(setter)]
    pub fn set_side(&mut self, side: SideWasm) {
        self.inner.side = Some(side.into());
    }

    #[wasm_bindgen(setter)]
    pub fn set_flags(&mut self, v: u32) {
        self.inner.flags = v;
    }

    #[wasm_bindgen(setter)]
    pub fn set_seq(&mut self, v: u64) {
        self.inner.seq = v;
    }

    #[wasm_bindgen(js_name = "setOrderIdHash")]
    pub fn set_order_id_hash(&mut self, hex: &str) -> Result<(), JsValue> {
        self.inner.order_id_hash = Some(hex_to_bytes16(hex)?);
        Ok(())
    }

    #[wasm_bindgen(js_name = "setParticipantIdHash")]
    pub fn set_participant_id_hash(&mut self, hex: &str) -> Result<(), JsValue> {
        self.inner.participant_id_hash = Some(hex_to_bytes16(hex)?);
        Ok(())
    }

    // --- JSON round-trip (BigInt-safe) ---

    #[wasm_bindgen(js_name = "toJson")]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner)
    }

    #[wasm_bindgen(js_name = "fromJson")]
    pub fn from_json(val: JsValue) -> Result<MarketEventWasm, JsValue> {
        let inner: MarketEvent =
            serde_wasm_bindgen::from_value(val).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }
}

// ---------------------------------------------------------------------------
// GraphDeltaWasm
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct GraphDeltaWasm {
    inner: neural_trader_core::GraphDelta,
}

#[wasm_bindgen]
impl GraphDeltaWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: neural_trader_core::GraphDelta::default(),
        }
    }

    #[wasm_bindgen(js_name = "nodesAdded")]
    pub fn nodes_added(&self) -> Result<JsValue, JsValue> {
        let pairs: Vec<(u8, u64)> = self
            .inner
            .nodes_added
            .iter()
            .map(|(k, id)| (*k as u8, *id))
            .collect();
        to_js(&pairs)
    }

    #[wasm_bindgen(js_name = "edgesAdded")]
    pub fn edges_added(&self) -> Result<JsValue, JsValue> {
        let triples: Vec<(u8, u64, u64)> = self
            .inner
            .edges_added
            .iter()
            .map(|(k, src, dst)| (*k as u8, *src, *dst))
            .collect();
        to_js(&triples)
    }

    #[wasm_bindgen(js_name = "propertiesUpdated")]
    pub fn properties_updated(&self) -> Result<JsValue, JsValue> {
        let props: Vec<(u64, u8, f64)> = self
            .inner
            .properties_updated
            .iter()
            .map(|(id, k, v)| (*id, *k as u8, *v))
            .collect();
        to_js(&props)
    }
}

impl Default for GraphDeltaWasm {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GateConfigWasm
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct GateConfigWasm {
    inner: GateConfig,
}

#[wasm_bindgen]
impl GateConfigWasm {
    /// Creates a `GateConfig` with sensible defaults.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: GateConfig::default(),
        }
    }

    #[wasm_bindgen(getter, js_name = "mincutFloorCalm")]
    pub fn mincut_floor_calm(&self) -> u64 {
        self.inner.mincut_floor_calm
    }
    #[wasm_bindgen(setter, js_name = "mincutFloorCalm")]
    pub fn set_mincut_floor_calm(&mut self, v: u64) {
        self.inner.mincut_floor_calm = v;
    }

    #[wasm_bindgen(getter, js_name = "mincutFloorNormal")]
    pub fn mincut_floor_normal(&self) -> u64 {
        self.inner.mincut_floor_normal
    }
    #[wasm_bindgen(setter, js_name = "mincutFloorNormal")]
    pub fn set_mincut_floor_normal(&mut self, v: u64) {
        self.inner.mincut_floor_normal = v;
    }

    #[wasm_bindgen(getter, js_name = "mincutFloorVolatile")]
    pub fn mincut_floor_volatile(&self) -> u64 {
        self.inner.mincut_floor_volatile
    }
    #[wasm_bindgen(setter, js_name = "mincutFloorVolatile")]
    pub fn set_mincut_floor_volatile(&mut self, v: u64) {
        self.inner.mincut_floor_volatile = v;
    }

    #[wasm_bindgen(getter, js_name = "cusumThreshold")]
    pub fn cusum_threshold(&self) -> f32 {
        self.inner.cusum_threshold
    }
    #[wasm_bindgen(setter, js_name = "cusumThreshold")]
    pub fn set_cusum_threshold(&mut self, v: f32) {
        self.inner.cusum_threshold = v;
    }

    #[wasm_bindgen(getter, js_name = "boundaryStabilityWindows")]
    pub fn boundary_stability_windows(&self) -> usize {
        self.inner.boundary_stability_windows
    }
    #[wasm_bindgen(setter, js_name = "boundaryStabilityWindows")]
    pub fn set_boundary_stability_windows(&mut self, v: usize) {
        self.inner.boundary_stability_windows = v;
    }

    #[wasm_bindgen(getter, js_name = "maxDriftScore")]
    pub fn max_drift_score(&self) -> f32 {
        self.inner.max_drift_score
    }
    #[wasm_bindgen(setter, js_name = "maxDriftScore")]
    pub fn set_max_drift_score(&mut self, v: f32) {
        self.inner.max_drift_score = v;
    }
}

impl Default for GateConfigWasm {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GateContextWasm
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct GateContextWasm {
    symbol_id: u32,
    venue_id: u16,
    ts_ns: u64,
    mincut_value: u64,
    partition_hash: String,
    cusum_score: f32,
    drift_score: f32,
    regime: RegimeLabelWasm,
    boundary_stable_count: usize,
}

#[wasm_bindgen]
impl GateContextWasm {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        symbol_id: u32,
        venue_id: u16,
        ts_ns: u64,
        mincut_value: u64,
        partition_hash: &str,
        cusum_score: f32,
        drift_score: f32,
        regime: RegimeLabelWasm,
        boundary_stable_count: usize,
    ) -> Self {
        Self {
            symbol_id,
            venue_id,
            ts_ns,
            mincut_value,
            partition_hash: partition_hash.to_string(),
            cusum_score,
            drift_score,
            regime,
            boundary_stable_count,
        }
    }

    #[wasm_bindgen(getter, js_name = "symbolId")]
    pub fn symbol_id(&self) -> u32 {
        self.symbol_id
    }
    #[wasm_bindgen(getter, js_name = "venueId")]
    pub fn venue_id(&self) -> u16 {
        self.venue_id
    }
    #[wasm_bindgen(getter, js_name = "tsNs")]
    pub fn ts_ns(&self) -> u64 {
        self.ts_ns
    }
    #[wasm_bindgen(getter, js_name = "mincutValue")]
    pub fn mincut_value(&self) -> u64 {
        self.mincut_value
    }
    #[wasm_bindgen(getter, js_name = "partitionHash")]
    pub fn partition_hash(&self) -> String {
        self.partition_hash.clone()
    }
    #[wasm_bindgen(getter, js_name = "cusumScore")]
    pub fn cusum_score(&self) -> f32 {
        self.cusum_score
    }
    #[wasm_bindgen(getter, js_name = "driftScore")]
    pub fn drift_score(&self) -> f32 {
        self.drift_score
    }
    #[wasm_bindgen(getter)]
    pub fn regime(&self) -> RegimeLabelWasm {
        self.regime
    }
    #[wasm_bindgen(getter, js_name = "boundaryStableCount")]
    pub fn boundary_stable_count(&self) -> usize {
        self.boundary_stable_count
    }

    fn to_inner(&self) -> Result<GateContext, JsValue> {
        Ok(GateContext {
            symbol_id: self.symbol_id,
            venue_id: self.venue_id,
            ts_ns: self.ts_ns,
            mincut_value: self.mincut_value,
            partition_hash: hex_to_bytes16(&self.partition_hash)?,
            cusum_score: self.cusum_score,
            drift_score: self.drift_score,
            regime: self.regime.into(),
            boundary_stable_count: self.boundary_stable_count,
        })
    }
}

// ---------------------------------------------------------------------------
// ThresholdGateWasm
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct ThresholdGateWasm {
    inner: ThresholdGate,
}

#[wasm_bindgen]
impl ThresholdGateWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(config: &GateConfigWasm) -> Self {
        Self {
            inner: ThresholdGate::new(config.inner.clone()),
        }
    }

    /// Evaluate the coherence gate, returning a decision.
    pub fn evaluate(&self, ctx: &GateContextWasm) -> Result<CoherenceDecisionWasm, JsValue> {
        let inner_ctx = ctx.to_inner()?;
        let decision = self
            .inner
            .evaluate(&inner_ctx)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(CoherenceDecisionWasm { inner: decision })
    }
}

// ---------------------------------------------------------------------------
// CoherenceDecisionWasm
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct CoherenceDecisionWasm {
    inner: CoherenceDecision,
}

#[wasm_bindgen]
impl CoherenceDecisionWasm {
    #[wasm_bindgen(getter, js_name = "allowRetrieve")]
    pub fn allow_retrieve(&self) -> bool {
        self.inner.allow_retrieve
    }

    #[wasm_bindgen(getter, js_name = "allowWrite")]
    pub fn allow_write(&self) -> bool {
        self.inner.allow_write
    }

    #[wasm_bindgen(getter, js_name = "allowLearn")]
    pub fn allow_learn(&self) -> bool {
        self.inner.allow_learn
    }

    #[wasm_bindgen(getter, js_name = "allowAct")]
    pub fn allow_act(&self) -> bool {
        self.inner.allow_act
    }

    #[wasm_bindgen(getter, js_name = "mincutValue")]
    pub fn mincut_value(&self) -> u64 {
        self.inner.mincut_value
    }

    #[wasm_bindgen(getter, js_name = "partitionHash")]
    pub fn partition_hash(&self) -> String {
        bytes16_to_hex(&self.inner.partition_hash)
    }

    #[wasm_bindgen(getter, js_name = "driftScore")]
    pub fn drift_score(&self) -> f32 {
        self.inner.drift_score
    }

    #[wasm_bindgen(getter, js_name = "cusumScore")]
    pub fn cusum_score(&self) -> f32 {
        self.inner.cusum_score
    }

    #[wasm_bindgen(js_name = "allAllowed")]
    pub fn all_allowed(&self) -> bool {
        self.inner.all_allowed()
    }

    #[wasm_bindgen(js_name = "fullyBlocked")]
    pub fn fully_blocked(&self) -> bool {
        self.inner.fully_blocked()
    }

    pub fn reasons(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner.reasons)
    }

    #[wasm_bindgen(js_name = "toJson")]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner)
    }
}

// ---------------------------------------------------------------------------
// ReplaySegmentWasm
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct ReplaySegmentWasm {
    inner: ReplaySegment,
}

#[wasm_bindgen]
impl ReplaySegmentWasm {
    #[wasm_bindgen(getter, js_name = "segmentId")]
    pub fn segment_id(&self) -> u64 {
        self.inner.segment_id
    }

    #[wasm_bindgen(getter, js_name = "symbolId")]
    pub fn symbol_id(&self) -> u32 {
        self.inner.symbol_id
    }

    #[wasm_bindgen(getter, js_name = "startTsNs")]
    pub fn start_ts_ns(&self) -> u64 {
        self.inner.start_ts_ns
    }

    #[wasm_bindgen(getter, js_name = "endTsNs")]
    pub fn end_ts_ns(&self) -> u64 {
        self.inner.end_ts_ns
    }

    /// BigInt-safe JSON serialization.
    #[wasm_bindgen(js_name = "toJson")]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner)
    }

    #[wasm_bindgen(js_name = "fromJson")]
    pub fn from_json(val: JsValue) -> Result<ReplaySegmentWasm, JsValue> {
        let inner: ReplaySegment =
            serde_wasm_bindgen::from_value(val).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }
}

// ---------------------------------------------------------------------------
// ReservoirStoreWasm
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct ReservoirStoreWasm {
    inner: ReservoirStore,
}

#[wasm_bindgen]
impl ReservoirStoreWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(max_size: usize) -> Result<ReservoirStoreWasm, JsValue> {
        if max_size == 0 {
            return Err(JsValue::from_str("max_size must be > 0"));
        }
        Ok(Self {
            inner: ReservoirStore::new(max_size),
        })
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[wasm_bindgen(js_name = "isEmpty")]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Attempt to write a segment. Returns `true` if the gate allowed it.
    #[wasm_bindgen(js_name = "maybeWrite")]
    pub fn maybe_write(
        &mut self,
        segment_json: JsValue,
        decision_json: JsValue,
    ) -> Result<bool, JsValue> {
        let seg: ReplaySegment = serde_wasm_bindgen::from_value(segment_json)
            .map_err(|e| JsValue::from_str(&format!("segment: {e}")))?;
        let gate: CoherenceDecision = serde_wasm_bindgen::from_value(decision_json)
            .map_err(|e| JsValue::from_str(&format!("decision: {e}")))?;
        self.inner
            .maybe_write(seg, &gate)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Retrieve segments matching a symbol, returned as JSON array.
    #[wasm_bindgen(js_name = "retrieveBySymbol")]
    pub fn retrieve_by_symbol(&self, symbol_id: u32, limit: usize) -> Result<JsValue, JsValue> {
        let query = neural_trader_replay::MemoryQuery {
            symbol_id,
            embedding: vec![],
            regime: None,
            limit,
        };
        let results = self
            .inner
            .retrieve(&query)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_js(&results)
    }
}

// ---------------------------------------------------------------------------
// Tests (rlib target only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_not_empty() {
        assert!(!version().is_empty());
    }

    #[test]
    fn health_check_returns_true() {
        assert!(health_check());
    }

    #[test]
    fn hex_roundtrip() {
        let orig = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let hex = bytes16_to_hex(&orig);
        let back = hex_to_bytes16_inner(&hex).unwrap();
        assert_eq!(orig, back);
    }

    #[test]
    fn hex_rejects_non_ascii() {
        let non_ascii = "\u{00e9}".repeat(16);
        assert!(hex_to_bytes16_inner(&non_ascii).is_err());
    }

    #[test]
    fn hex_strips_0x_prefix() {
        let orig = [0xABu8, 0xCD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let hex = format!("0x{}", bytes16_to_hex(&orig));
        let back = hex_to_bytes16_inner(&hex).unwrap();
        assert_eq!(orig, back);
    }

    #[test]
    fn hex_rejects_wrong_length() {
        assert!(hex_to_bytes16_inner("abcd").is_err());
        assert!(hex_to_bytes16_inner("").is_err());
    }

    #[test]
    fn gate_config_defaults() {
        let cfg = GateConfigWasm::new();
        assert_eq!(cfg.mincut_floor_calm(), 12);
        assert_eq!(cfg.mincut_floor_normal(), 9);
        assert_eq!(cfg.mincut_floor_volatile(), 6);
    }

    #[test]
    fn market_event_basic() {
        let evt = MarketEventWasm::new(EventTypeWasm::Trade, 42, 1, 500_000_000, 10_000);
        assert_eq!(evt.symbol_id(), 42);
        assert_eq!(evt.venue_id(), 1);
        assert_eq!(evt.price_fp(), 500_000_000);
        assert!(evt.side().is_none());
    }

    #[test]
    fn enum_conversions() {
        let et: EventType = EventTypeWasm::Trade.into();
        assert_eq!(et, EventType::Trade);

        let back: EventTypeWasm = et.into();
        assert_eq!(back as u8, EventTypeWasm::Trade as u8);

        let rl: RegimeLabel = RegimeLabelWasm::Volatile.into();
        assert_eq!(rl, RegimeLabel::Volatile);
    }

    #[test]
    fn property_key_conversions() {
        let pk: neural_trader_core::PropertyKey = PropertyKeyWasm::CancelHazard.into();
        assert_eq!(pk, neural_trader_core::PropertyKey::CancelHazard);
    }

    // ReservoirStoreWasm::new(0) returns Err(JsValue) which panics in
    // native tests. The zero-size guard is exercised in WASM integration tests.
}
