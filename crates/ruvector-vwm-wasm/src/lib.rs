//! # RuVector VWM WASM Bindings
//!
//! WASM bindings for the RuVector Visual World Model, providing browser-ready
//! access to 4D Gaussian splatting, coherence gating, entity graphs, lineage
//! logging, and streaming primitives.
//!
//! ## Quick Start (JavaScript)
//!
//! ```javascript
//! import { initVwm, WasmGaussian4D, WasmDrawList, WasmCoherenceGate } from '@ruvector/vwm-wasm';
//!
//! initVwm();
//!
//! const g = new WasmGaussian4D(0.0, 1.0, -5.0, 42);
//! g.setVelocity(0.1, 0.0, 0.0);
//! const pos = g.positionAt(2.5);
//! console.log('position:', pos);
//! ```

use wasm_bindgen::prelude::*;

use ruvector_vwm::coherence::{
    CoherenceDecision, CoherenceGate, CoherenceInput, CoherencePolicy, PermissionLevel,
};
use ruvector_vwm::draw_list::{DrawList, OpacityMode};
use ruvector_vwm::entity::{
    AttributeValue, Edge, EdgeType, Entity, EntityGraph, EntityType,
};
use ruvector_vwm::gaussian::Gaussian4D;
use ruvector_vwm::lineage::{
    LineageEventType, LineageLog, Provenance, ProvenanceSource,
};
use ruvector_vwm::streaming::{ActiveMask, BandwidthBudget};
use ruvector_vwm::tile::QuantTier;

// ---------------------------------------------------------------------------
// Top-level functions
// ---------------------------------------------------------------------------

/// Initialize the VWM WASM module.
///
/// Sets up the console error panic hook for better error messages and logs
/// the library version to the browser console.
#[wasm_bindgen(js_name = initVwm)]
pub fn init_vwm() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    web_sys::console::log_1(
        &format!("ruvector-vwm-wasm v{} initialized", env!("CARGO_PKG_VERSION")).into(),
    );
}

/// Return the crate version string.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ---------------------------------------------------------------------------
// Helpers (not exported)
//
// Internal conversion functions return Result<T, String> so they can be tested
// in native (non-WASM) mode. A thin `to_js_err` wrapper converts to JsValue
// at the WASM boundary.
// ---------------------------------------------------------------------------

fn to_js_err(msg: String) -> JsValue {
    JsValue::from_str(&msg)
}

fn quant_tier_from_u8(v: u8) -> Result<QuantTier, String> {
    match v {
        0 => Ok(QuantTier::Hot8),
        1 => Ok(QuantTier::Warm7),
        2 => Ok(QuantTier::Warm5),
        3 => Ok(QuantTier::Cold3),
        _ => Err(format!("invalid QuantTier: {} (expected 0-3)", v)),
    }
}

fn opacity_mode_from_u8(v: u8) -> Result<OpacityMode, String> {
    match v {
        0 => Ok(OpacityMode::AlphaBlend),
        1 => Ok(OpacityMode::Additive),
        2 => Ok(OpacityMode::Opaque),
        _ => Err(format!("invalid OpacityMode: {} (expected 0-2)", v)),
    }
}

fn permission_level_from_u8(v: u8) -> Result<PermissionLevel, String> {
    match v {
        0 => Ok(PermissionLevel::ReadOnly),
        1 => Ok(PermissionLevel::Standard),
        2 => Ok(PermissionLevel::Elevated),
        3 => Ok(PermissionLevel::Admin),
        _ => Err(format!("invalid PermissionLevel: {} (expected 0-3)", v)),
    }
}

fn edge_type_from_str(s: &str) -> Result<EdgeType, String> {
    match s.to_ascii_lowercase().as_str() {
        "adjacency" | "spatial" => Ok(EdgeType::Adjacency),
        "containment" => Ok(EdgeType::Containment),
        "continuity" | "temporal" => Ok(EdgeType::Continuity),
        "causality" | "causal" => Ok(EdgeType::Causality),
        "same_identity" | "sameidentity" | "identity" | "semantic" => {
            Ok(EdgeType::SameIdentity)
        }
        _ => Err(format!(
            "unknown edge type: '{}' (expected adjacency|containment|continuity|causality|same_identity)",
            s
        )),
    }
}

fn decision_to_str(d: CoherenceDecision) -> &'static str {
    match d {
        CoherenceDecision::Accept => "accept",
        CoherenceDecision::Defer => "defer",
        CoherenceDecision::Freeze => "freeze",
        CoherenceDecision::Rollback => "rollback",
    }
}

fn parse_embedding_json(json: &str) -> Result<Vec<f32>, String> {
    serde_json::from_str::<Vec<f32>>(json)
        .map_err(|e| format!("failed to parse embedding JSON: {}", e))
}

// ---------------------------------------------------------------------------
// WasmGaussian4D
// ---------------------------------------------------------------------------

/// A 4D Gaussian primitive exposed to JavaScript.
///
/// Represents a volumetric element with position, velocity, color, opacity,
/// and temporal activity range.
#[wasm_bindgen]
pub struct WasmGaussian4D {
    inner: Gaussian4D,
}

#[wasm_bindgen]
impl WasmGaussian4D {
    /// Create a new Gaussian at position (x, y, z) with the given ID.
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32, z: f32, id: u32) -> Self {
        Self {
            inner: Gaussian4D::new([x, y, z], id),
        }
    }

    /// Evaluate the position at time `t` using the linear motion model.
    ///
    /// Returns a `Float32Array` of `[x, y, z]`.
    #[wasm_bindgen(js_name = positionAt)]
    pub fn position_at(&self, t: f32) -> js_sys::Float32Array {
        let pos = self.inner.position_at(t);
        js_sys::Float32Array::from(&pos[..])
    }

    /// Check whether this Gaussian is active at time `t`.
    #[wasm_bindgen(js_name = isActiveAt)]
    pub fn is_active_at(&self, t: f32) -> bool {
        self.inner.is_active_at(t)
    }

    /// Set the temporal activity range `[start, end]`.
    #[wasm_bindgen(js_name = setTimeRange)]
    pub fn set_time_range(&mut self, start: f32, end: f32) {
        self.inner.time_range = [start, end];
    }

    /// Set the per-axis velocity for the linear motion model.
    #[wasm_bindgen(js_name = setVelocity)]
    pub fn set_velocity(&mut self, vx: f32, vy: f32, vz: f32) {
        self.inner.velocity = [vx, vy, vz];
    }

    /// Set the opacity (clamped to `[0, 1]`).
    #[wasm_bindgen(js_name = setOpacity)]
    pub fn set_opacity(&mut self, opacity: f32) {
        self.inner.opacity = opacity.clamp(0.0, 1.0);
    }

    /// Set the RGB color via spherical harmonics degree-0 coefficients.
    #[wasm_bindgen(js_name = setColor)]
    pub fn set_color(&mut self, r: f32, g: f32, b: f32) {
        self.inner.sh_coeffs = [r, g, b];
    }
}

// ---------------------------------------------------------------------------
// WasmActiveMask
// ---------------------------------------------------------------------------

/// A compact bitmask tracking which Gaussians are active.
#[wasm_bindgen]
pub struct WasmActiveMask {
    inner: ActiveMask,
}

#[wasm_bindgen]
impl WasmActiveMask {
    /// Create a mask where all Gaussians start inactive.
    #[wasm_bindgen(constructor)]
    pub fn new(total_count: u32) -> Self {
        Self {
            inner: ActiveMask::new(total_count),
        }
    }

    /// Set the active state of a Gaussian by index.
    pub fn set(&mut self, index: u32, active: bool) {
        self.inner.set(index, active);
    }

    /// Check if a Gaussian is active.
    #[wasm_bindgen(js_name = isActive)]
    pub fn is_active(&self, index: u32) -> bool {
        self.inner.is_active(index)
    }

    /// Count the number of active Gaussians.
    #[wasm_bindgen(js_name = activeCount)]
    pub fn active_count(&self) -> u32 {
        self.inner.active_count()
    }

    /// Return the byte size of the backing storage.
    #[wasm_bindgen(js_name = byteSize)]
    pub fn byte_size(&self) -> usize {
        self.inner.byte_size()
    }
}

// ---------------------------------------------------------------------------
// WasmDrawList
// ---------------------------------------------------------------------------

/// A packed draw list for GPU submission.
///
/// Accumulates tile-bind, budget, and draw-block commands, then serializes
/// to a compact byte buffer for GPU upload or network transport.
#[wasm_bindgen]
pub struct WasmDrawList {
    inner: DrawList,
}

#[wasm_bindgen]
impl WasmDrawList {
    /// Create a new empty draw list.
    #[wasm_bindgen(constructor)]
    pub fn new(epoch: u64, sequence: u32, budget_profile_id: u32) -> Self {
        Self {
            inner: DrawList::new(epoch, sequence, budget_profile_id),
        }
    }

    /// Bind a tile to a GPU block with the given quantization tier.
    ///
    /// `quant_tier`: 0 = Hot8, 1 = Warm7, 2 = Warm5, 3 = Cold3.
    #[wasm_bindgen(js_name = bindTile)]
    pub fn bind_tile(
        &mut self,
        tile_id: u64,
        block_ref: u32,
        quant_tier: u8,
    ) -> Result<(), JsValue> {
        let tier = quant_tier_from_u8(quant_tier).map_err(to_js_err)?;
        self.inner.bind_tile(tile_id, block_ref, tier);
        Ok(())
    }

    /// Set a per-tile rendering budget.
    #[wasm_bindgen(js_name = setBudget)]
    pub fn set_budget(&mut self, screen_tile_id: u32, max_gaussians: u32, max_overdraw: f32) {
        self.inner.set_budget(screen_tile_id, max_gaussians, max_overdraw);
    }

    /// Append a draw-block command.
    ///
    /// `opacity_mode`: 0 = AlphaBlend, 1 = Additive, 2 = Opaque.
    #[wasm_bindgen(js_name = drawBlock)]
    pub fn draw_block(
        &mut self,
        block_ref: u32,
        sort_key: f32,
        opacity_mode: u8,
    ) -> Result<(), JsValue> {
        let mode = opacity_mode_from_u8(opacity_mode).map_err(to_js_err)?;
        self.inner.draw_block(block_ref, sort_key, mode);
        Ok(())
    }

    /// Finalize the draw list and return the integrity checksum.
    pub fn finalize(&mut self) -> u32 {
        self.inner.finalize()
    }

    /// Return the number of commands (excluding the End sentinel).
    #[wasm_bindgen(js_name = commandCount)]
    pub fn command_count(&self) -> usize {
        self.inner.command_count()
    }

    /// Serialize the draw list to bytes.
    ///
    /// Returns a `Uint8Array` containing the packed binary representation.
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> js_sys::Uint8Array {
        let bytes = self.inner.to_bytes();
        js_sys::Uint8Array::from(&bytes[..])
    }
}

// ---------------------------------------------------------------------------
// WasmCoherenceGate
// ---------------------------------------------------------------------------

/// Coherence gate for evaluating world-model update proposals.
///
/// Produces a decision string: `"accept"`, `"defer"`, `"freeze"`, or
/// `"rollback"` based on disagreement, confidence, freshness, and budget
/// pressure.
#[wasm_bindgen]
pub struct WasmCoherenceGate {
    inner: CoherenceGate,
}

impl Default for WasmCoherenceGate {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmCoherenceGate {
    /// Create a gate with the default policy.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: CoherenceGate::with_defaults(),
        }
    }

    /// Create a gate with a fully custom policy.
    #[wasm_bindgen(js_name = withPolicy)]
    pub fn with_policy(
        accept_threshold: f32,
        defer_threshold: f32,
        freeze_disagreement: f32,
        rollback_disagreement: f32,
        max_staleness_ms: u64,
        budget_freeze_threshold: f32,
    ) -> Self {
        let policy = CoherencePolicy {
            accept_threshold,
            defer_threshold,
            freeze_disagreement,
            rollback_disagreement,
            max_staleness_ms,
            budget_freeze_threshold,
        };
        Self {
            inner: CoherenceGate::new(policy),
        }
    }

    /// Evaluate an update proposal and return a decision string.
    ///
    /// `permission_level`: 0 = ReadOnly, 1 = Standard, 2 = Elevated, 3 = Admin.
    ///
    /// Returns one of: `"accept"`, `"defer"`, `"freeze"`, `"rollback"`.
    pub fn evaluate(
        &self,
        tile_disagreement: f32,
        entity_continuity: f32,
        sensor_confidence: f32,
        sensor_freshness_ms: u64,
        budget_pressure: f32,
        permission_level: u8,
    ) -> Result<String, JsValue> {
        let perm = permission_level_from_u8(permission_level).map_err(to_js_err)?;
        let input = CoherenceInput {
            tile_disagreement,
            entity_continuity,
            sensor_confidence,
            sensor_freshness_ms,
            budget_pressure,
            permission_level: perm,
        };
        let decision = self.inner.evaluate(&input);
        Ok(decision_to_str(decision).to_string())
    }
}

// ---------------------------------------------------------------------------
// WasmEntityGraph
// ---------------------------------------------------------------------------

/// A semantic entity graph for scene understanding.
///
/// Stores typed entities (objects, tracks, regions, events) and weighted
/// edges (adjacency, containment, continuity, causality, identity).
#[wasm_bindgen]
pub struct WasmEntityGraph {
    inner: EntityGraph,
}

impl Default for WasmEntityGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmEntityGraph {
    /// Create an empty entity graph.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: EntityGraph::new(),
        }
    }

    /// Add an object entity.
    ///
    /// `embedding_json`: a JSON array of floats (e.g. `"[0.1, 0.2, 0.3]"`),
    /// or an empty string for no embedding.
    #[wasm_bindgen(js_name = addObject)]
    pub fn add_object(
        &mut self,
        id: u64,
        class_name: &str,
        embedding_json: &str,
        confidence: f32,
    ) -> Result<(), JsValue> {
        let embedding = if embedding_json.is_empty() {
            vec![]
        } else {
            parse_embedding_json(embedding_json).map_err(to_js_err)?
        };

        let entity = Entity {
            id,
            entity_type: EntityType::Object {
                class: class_name.to_string(),
            },
            time_span: [f32::NEG_INFINITY, f32::INFINITY],
            embedding,
            confidence,
            privacy_tags: vec![],
            attributes: vec![],
            gaussian_ids: vec![],
        };
        self.inner.add_entity(entity);
        Ok(())
    }

    /// Add a track entity.
    ///
    /// `embedding_json`: a JSON array of floats, or an empty string.
    #[wasm_bindgen(js_name = addTrack)]
    pub fn add_track(
        &mut self,
        id: u64,
        embedding_json: &str,
        confidence: f32,
    ) -> Result<(), JsValue> {
        let embedding = if embedding_json.is_empty() {
            vec![]
        } else {
            parse_embedding_json(embedding_json).map_err(to_js_err)?
        };

        let entity = Entity {
            id,
            entity_type: EntityType::Track,
            time_span: [f32::NEG_INFINITY, f32::INFINITY],
            embedding,
            confidence,
            privacy_tags: vec![],
            attributes: vec![],
            gaussian_ids: vec![],
        };
        self.inner.add_entity(entity);
        Ok(())
    }

    /// Add an edge between two entities.
    ///
    /// `edge_type_str`: one of `"adjacency"`, `"containment"`, `"continuity"`,
    /// `"causality"`, `"same_identity"` (case-insensitive; aliases like
    /// `"spatial"`, `"temporal"`, `"causal"`, `"semantic"` are also accepted).
    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(
        &mut self,
        source: u64,
        target: u64,
        edge_type_str: &str,
        weight: f32,
    ) -> Result<(), JsValue> {
        let edge_type = edge_type_from_str(edge_type_str).map_err(to_js_err)?;
        self.inner.add_edge(Edge {
            source,
            target,
            edge_type,
            weight,
            time_range: None,
        });
        Ok(())
    }

    /// Get an entity as a JSON string.
    ///
    /// Returns a JSON object with `id`, `type`, `confidence`, `embedding`,
    /// and `attributes` fields. Returns `null` if the entity does not exist.
    #[wasm_bindgen(js_name = getEntityJson)]
    pub fn get_entity_json(&self, id: u64) -> String {
        match self.inner.get_entity(id) {
            Some(e) => entity_to_json(e),
            None => "null".to_string(),
        }
    }

    /// Query entities by type and return a JSON array of matching entity IDs.
    ///
    /// `type_str`: `"object"`, `"track"`, `"region"`, `"event"`, or an object
    /// class name like `"car"`.
    #[wasm_bindgen(js_name = queryByType)]
    pub fn query_by_type(&self, type_str: &str) -> String {
        let entities = self.inner.query_by_type(type_str);
        let ids: Vec<u64> = entities.iter().map(|e| e.id).collect();
        format!(
            "[{}]",
            ids.iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    /// Total number of entities in the graph.
    #[wasm_bindgen(js_name = entityCount)]
    pub fn entity_count(&self) -> usize {
        self.inner.entity_count()
    }

    /// Total number of edges in the graph.
    #[wasm_bindgen(js_name = edgeCount)]
    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }
}

/// Serialize an Entity to a JSON string without serde derives on the core type.
fn entity_to_json(e: &Entity) -> String {
    let type_str = match &e.entity_type {
        EntityType::Object { class } => format!("\"object:{}\"", escape_json_str(class)),
        EntityType::Track => "\"track\"".to_string(),
        EntityType::Region => "\"region\"".to_string(),
        EntityType::Event => "\"event\"".to_string(),
    };

    let embedding_str = format!(
        "[{}]",
        e.embedding
            .iter()
            .map(|v| format!("{}", v))
            .collect::<Vec<_>>()
            .join(",")
    );

    let attrs_str = format!(
        "{{{}}}",
        e.attributes
            .iter()
            .map(|(k, v)| format!("\"{}\":{}", escape_json_str(k), attr_value_to_json(v)))
            .collect::<Vec<_>>()
            .join(",")
    );

    format!(
        "{{\"id\":{},\"type\":{},\"confidence\":{},\"embedding\":{},\"attributes\":{}}}",
        e.id, type_str, e.confidence, embedding_str, attrs_str
    )
}

fn attr_value_to_json(v: &AttributeValue) -> String {
    match v {
        AttributeValue::Float(f) => format!("{}", f),
        AttributeValue::Int(i) => format!("{}", i),
        AttributeValue::Text(s) => format!("\"{}\"", escape_json_str(s)),
        AttributeValue::Bool(b) => format!("{}", b),
        AttributeValue::Vec3(arr) => format!("[{},{},{}]", arr[0], arr[1], arr[2]),
    }
}

fn escape_json_str(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ---------------------------------------------------------------------------
// WasmLineageLog
// ---------------------------------------------------------------------------

/// Append-only lineage log for world-model provenance tracking.
///
/// Records tile creation and update events with source information and
/// confidence scores.
#[wasm_bindgen]
pub struct WasmLineageLog {
    inner: LineageLog,
}

impl Default for WasmLineageLog {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmLineageLog {
    /// Create an empty lineage log.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: LineageLog::new(),
        }
    }

    /// Append a tile-created event. Returns the assigned event ID.
    ///
    /// `source_type`: a descriptive string for the data source
    /// (e.g. `"sensor:cam0"`, `"model:yolo"`, `"manual"`, `"system"`).
    #[wasm_bindgen(js_name = appendTileCreated)]
    pub fn append_tile_created(
        &mut self,
        tile_id: u64,
        timestamp_ms: u64,
        source_type: &str,
        confidence: f32,
    ) -> u64 {
        let provenance = make_provenance(source_type, confidence);
        self.inner.append(
            timestamp_ms,
            tile_id,
            LineageEventType::TileCreated,
            provenance,
            None,
            CoherenceDecision::Accept,
            confidence,
        )
    }

    /// Append a tile-updated event. Returns the assigned event ID.
    ///
    /// `source_type`: same format as in `appendTileCreated`.
    #[wasm_bindgen(js_name = appendTileUpdated)]
    pub fn append_tile_updated(
        &mut self,
        tile_id: u64,
        timestamp_ms: u64,
        delta_size: u32,
        source_type: &str,
        confidence: f32,
    ) -> u64 {
        let provenance = make_provenance(source_type, confidence);
        self.inner.append(
            timestamp_ms,
            tile_id,
            LineageEventType::TileUpdated { delta_size },
            provenance,
            None,
            CoherenceDecision::Accept,
            confidence,
        )
    }

    /// Query all events for a tile, returned as a JSON array.
    ///
    /// Each event is a JSON object with `event_id`, `tile_id`, `timestamp_ms`,
    /// `event_type`, `source`, and `confidence` fields.
    #[wasm_bindgen(js_name = queryTileJson)]
    pub fn query_tile_json(&self, tile_id: u64) -> String {
        let events = self.inner.query_tile(tile_id);
        let items: Vec<String> = events.iter().map(|e| lineage_event_to_json(e)).collect();
        format!("[{}]", items.join(","))
    }

    /// Total number of events in the log.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the log is empty.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

fn make_provenance(source_type: &str, confidence: f32) -> Provenance {
    let source = if let Some(stripped) = source_type.strip_prefix("sensor:") {
        ProvenanceSource::Sensor {
            sensor_id: stripped.to_string(),
        }
    } else if let Some(stripped) = source_type.strip_prefix("model:") {
        ProvenanceSource::Inference {
            model_id: stripped.to_string(),
        }
    } else if source_type == "manual" {
        ProvenanceSource::Manual {
            user_id: "unknown".to_string(),
        }
    } else {
        ProvenanceSource::Sensor {
            sensor_id: source_type.to_string(),
        }
    };

    Provenance {
        source,
        confidence,
        signature: None,
    }
}

fn lineage_event_to_json(e: &ruvector_vwm::lineage::LineageEvent) -> String {
    let event_type_str = match &e.event_type {
        LineageEventType::TileCreated => "\"created\"".to_string(),
        LineageEventType::TileUpdated { delta_size } => {
            format!("\"updated(delta_size={})\"", delta_size)
        }
        LineageEventType::TileMerged { source_tiles } => {
            let tiles_str = source_tiles
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!("\"merged([{}])\"", tiles_str)
        }
        LineageEventType::EntityAdded { entity_id } => {
            format!("\"entity_added({})\"", entity_id)
        }
        LineageEventType::EntityUpdated { entity_id } => {
            format!("\"entity_updated({})\"", entity_id)
        }
        LineageEventType::Rollback { reason } => {
            format!("\"rollback({})\"", escape_json_str(reason))
        }
        LineageEventType::Freeze { reason } => {
            format!("\"freeze({})\"", escape_json_str(reason))
        }
    };

    let source_str = match &e.provenance.source {
        ProvenanceSource::Sensor { sensor_id } => {
            format!("\"sensor:{}\"", escape_json_str(sensor_id))
        }
        ProvenanceSource::Inference { model_id } => {
            format!("\"model:{}\"", escape_json_str(model_id))
        }
        ProvenanceSource::Manual { user_id } => {
            format!("\"manual:{}\"", escape_json_str(user_id))
        }
        ProvenanceSource::Merge { sources } => {
            let ids = sources
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!("\"merge:[{}]\"", ids)
        }
    };

    format!(
        "{{\"event_id\":{},\"tile_id\":{},\"timestamp_ms\":{},\"event_type\":{},\"source\":{},\"confidence\":{}}}",
        e.event_id, e.tile_id, e.timestamp_ms, event_type_str, source_str, e.provenance.confidence
    )
}

// ---------------------------------------------------------------------------
// WasmBandwidthBudget
// ---------------------------------------------------------------------------

/// Token-bucket style bandwidth limiter for stream rate control.
#[wasm_bindgen]
pub struct WasmBandwidthBudget {
    inner: BandwidthBudget,
}

#[wasm_bindgen]
impl WasmBandwidthBudget {
    /// Create a new bandwidth budget with the given rate limit.
    #[wasm_bindgen(constructor)]
    pub fn new(max_bytes_per_second: u64) -> Self {
        Self {
            inner: BandwidthBudget::new(max_bytes_per_second),
        }
    }

    /// Check whether `bytes` can be sent at time `now_ms` without exceeding
    /// the budget.
    #[wasm_bindgen(js_name = canSend)]
    pub fn can_send(&self, bytes: u64, now_ms: u64) -> bool {
        self.inner.can_send(bytes, now_ms)
    }

    /// Record that `bytes` were sent at time `now_ms`.
    #[wasm_bindgen(js_name = recordSent)]
    pub fn record_sent(&mut self, bytes: u64, now_ms: u64) {
        self.inner.record_sent(bytes, now_ms);
    }

    /// Return the current utilization ratio in `[0, 1]`.
    pub fn utilization(&self) -> f32 {
        self.inner.utilization()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_not_empty() {
        assert!(!version().is_empty());
    }

    #[test]
    fn test_gaussian_position_at() {
        let mut g = WasmGaussian4D::new(0.0, 0.0, 0.0, 1);
        g.set_velocity(1.0, 2.0, 3.0);
        g.set_time_range(0.0, 10.0);
        // Test the underlying position_at logic directly via the inner type.
        // t_mid = 5.0, at t=7.0: dt=2.0 -> pos = [2.0, 4.0, 6.0]
        let pos = g.inner.position_at(7.0);
        assert!((pos[0] - 2.0).abs() < 1e-6);
        assert!((pos[1] - 4.0).abs() < 1e-6);
        assert!((pos[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_active() {
        let mut g = WasmGaussian4D::new(0.0, 0.0, 0.0, 1);
        g.set_time_range(1.0, 5.0);
        assert!(!g.is_active_at(0.5));
        assert!(g.is_active_at(3.0));
        assert!(g.is_active_at(5.0));
        assert!(!g.is_active_at(5.5));
    }

    #[test]
    fn test_active_mask() {
        let mut mask = WasmActiveMask::new(128);
        assert_eq!(mask.active_count(), 0);
        mask.set(0, true);
        mask.set(64, true);
        mask.set(127, true);
        assert_eq!(mask.active_count(), 3);
        assert!(mask.is_active(0));
        assert!(!mask.is_active(1));
        assert!(mask.is_active(64));
        mask.set(64, false);
        assert_eq!(mask.active_count(), 2);
    }

    #[test]
    fn test_draw_list() {
        let mut dl = WasmDrawList::new(1, 0, 100);
        dl.bind_tile(42, 1, 0).unwrap(); // Hot8
        dl.set_budget(0, 1024, 2.0);
        dl.draw_block(1, 0.5, 0).unwrap(); // AlphaBlend
        assert_eq!(dl.command_count(), 3);
        let checksum = dl.finalize();
        assert_ne!(checksum, 0);
        // Verify the inner draw list serializes to a non-empty byte buffer.
        // (to_bytes() returns js_sys::Uint8Array which requires a JS runtime,
        // so we test the inner type directly for native tests.)
        let bytes = dl.inner.to_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_coherence_gate_accept() {
        let gate = WasmCoherenceGate::new();
        let result = gate
            .evaluate(0.1, 0.9, 1.0, 100, 0.3, 1)
            .unwrap();
        assert_eq!(result, "accept");
    }

    #[test]
    fn test_coherence_gate_admin_accept() {
        let gate = WasmCoherenceGate::new();
        let result = gate
            .evaluate(1.0, 0.0, 0.0, 100000, 1.0, 3)
            .unwrap();
        assert_eq!(result, "accept");
    }

    #[test]
    fn test_coherence_gate_readonly_defer() {
        let gate = WasmCoherenceGate::new();
        let result = gate
            .evaluate(0.0, 1.0, 1.0, 0, 0.0, 0)
            .unwrap();
        assert_eq!(result, "defer");
    }

    #[test]
    fn test_entity_graph() {
        let mut graph = WasmEntityGraph::new();
        graph
            .add_object(1, "car", "[0.1,0.2,0.3]", 0.95)
            .unwrap();
        graph.add_object(2, "person", "", 0.8).unwrap();
        graph.add_track(3, "", 0.9).unwrap();
        graph.add_edge(1, 2, "adjacency", 0.5).unwrap();

        assert_eq!(graph.entity_count(), 3);
        assert_eq!(graph.edge_count(), 1);

        let json = graph.get_entity_json(1);
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("car"));

        let missing = graph.get_entity_json(999);
        assert_eq!(missing, "null");

        let cars = graph.query_by_type("car");
        assert!(cars.contains('1'));
    }

    #[test]
    fn test_lineage_log() {
        let mut log = WasmLineageLog::new();
        let id0 = log.append_tile_created(10, 1000, "sensor:cam0", 0.9);
        let id1 = log.append_tile_updated(10, 1100, 256, "model:yolo", 0.85);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(log.len(), 2);

        let json = log.query_tile_json(10);
        assert!(json.contains("\"event_id\":0"));
        assert!(json.contains("\"event_id\":1"));
        assert!(json.contains("\"created\""));
    }

    #[test]
    fn test_bandwidth_budget() {
        let mut budget = WasmBandwidthBudget::new(1000);
        assert!(budget.can_send(500, 100));
        budget.record_sent(500, 100);
        assert!((budget.utilization() - 0.5).abs() < 1e-6);
        assert!(budget.can_send(500, 200));
        assert!(!budget.can_send(501, 200));
    }

    #[test]
    fn test_quant_tier_from_u8_invalid() {
        assert!(quant_tier_from_u8(4).is_err());
    }

    #[test]
    fn test_opacity_mode_from_u8_invalid() {
        assert!(opacity_mode_from_u8(5).is_err());
    }

    #[test]
    fn test_permission_level_from_u8_invalid() {
        assert!(permission_level_from_u8(10).is_err());
    }

    #[test]
    fn test_edge_type_from_str_aliases() {
        assert!(edge_type_from_str("spatial").is_ok());
        assert!(edge_type_from_str("temporal").is_ok());
        assert!(edge_type_from_str("causal").is_ok());
        assert!(edge_type_from_str("semantic").is_ok());
        assert!(edge_type_from_str("Adjacency").is_ok());
        assert!(edge_type_from_str("unknown_type").is_err());
    }
}
