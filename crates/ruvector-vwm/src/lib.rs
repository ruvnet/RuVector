//! Visual World Model: 4D Gaussian Splatting Core Library
//!
//! This crate implements the core data structures and logic for a 4D Gaussian
//! splatting visual world model as described in ADR-018. It is designed with zero
//! external dependencies for full WASM compatibility, following the same patterns
//! as `ruvector-temporal-tensor`.
//!
//! # Architecture: Three Loops
//!
//! The VWM operates through three concurrent loops, each with distinct latency
//! targets and responsibilities:
//!
//! ## 1. Render Loop (~16ms / 60Hz)
//!
//! The fastest loop handles frame-by-frame rendering of the Gaussian scene.
//! It consumes packed [`draw_list::DrawList`]s that bind tiles, set per-screen-tile
//! budgets, and issue draw calls. The draw list protocol is designed for minimal
//! per-frame allocation and efficient GPU upload.
//!
//! ```text
//! Camera Pose -> Tile Visibility -> Sort Gaussians -> Build DrawList -> Rasterize
//! ```
//!
//! ## 2. Update Loop (~100ms)
//!
//! The update loop ingests new sensor data, runs the
//! [`coherence::CoherenceGate`], and mutates the world-model tiles when updates
//! are accepted. It manages the [`tile::PrimitiveBlock`] encode/decode cycle,
//! the [`streaming`] protocol for network transport, and entity graph updates.
//!
//! ```text
//! Sensor Data -> Coherence Gate -> Tile Update -> Lineage Log -> Stream Packets
//! ```
//!
//! ## 3. Governance Loop (~1s+)
//!
//! The slowest loop handles policy enforcement, privacy, lineage auditing, and
//! tile lifecycle management (creation, merging, eviction). The
//! [`lineage::LineageLog`] provides an append-only audit trail of all mutations
//! with full provenance.
//!
//! ```text
//! Lineage Audit -> Privacy Check -> Tile Lifecycle -> Policy Update
//! ```
//!
//! # Core Types
//!
//! | Module | Primary Type | Purpose |
//! |--------|-------------|---------|
//! | [`gaussian`] | [`Gaussian4D`](gaussian::Gaussian4D) | Single 4D volumetric primitive |
//! | [`tile`] | [`Tile`](tile::Tile), [`PrimitiveBlock`](tile::PrimitiveBlock) | Spacetime tile with packed Gaussians |
//! | [`draw_list`] | [`DrawList`](draw_list::DrawList) | Packed GPU draw commands |
//! | [`coherence`] | [`CoherenceGate`](coherence::CoherenceGate) | Update acceptance/rejection gate |
//! | [`lineage`] | [`LineageLog`](lineage::LineageLog) | Append-only provenance log |
//! | [`entity`] | [`EntityGraph`](entity::EntityGraph) | Semantic entity graph |
//! | [`streaming`] | [`StreamPacket`](streaming::StreamPacket) | Network transport protocol |
//!
//! # Zero Dependencies
//!
//! This crate has no external dependencies, making it fully WASM-compatible
//! and suitable for embedded or constrained environments.
//!
//! # Quick Start
//!
//! ```rust
//! use ruvector_vwm::gaussian::Gaussian4D;
//! use ruvector_vwm::tile::{PrimitiveBlock, QuantTier};
//! use ruvector_vwm::draw_list::{DrawList, OpacityMode};
//! use ruvector_vwm::coherence::{CoherenceGate, CoherenceInput, PermissionLevel};
//!
//! // Create Gaussians
//! let g1 = Gaussian4D::new([0.0, 0.0, -5.0], 0);
//! let g2 = Gaussian4D::new([1.0, 0.0, -5.0], 1);
//!
//! // Pack into a tile
//! let block = PrimitiveBlock::encode(&[g1, g2], QuantTier::Hot8);
//! assert_eq!(block.count, 2);
//!
//! // Build a draw list
//! let mut dl = DrawList::new(1, 0, 0);
//! dl.bind_tile(42, 0, QuantTier::Hot8);
//! dl.draw_block(0, 0.5, OpacityMode::AlphaBlend);
//! dl.finalize();
//!
//! // Evaluate coherence
//! let gate = CoherenceGate::with_defaults();
//! let input = CoherenceInput {
//!     tile_disagreement: 0.1,
//!     entity_continuity: 0.9,
//!     sensor_confidence: 1.0,
//!     sensor_freshness_ms: 50,
//!     budget_pressure: 0.2,
//!     permission_level: PermissionLevel::Standard,
//! };
//! let decision = gate.evaluate(&input);
//! assert_eq!(decision, ruvector_vwm::coherence::CoherenceDecision::Accept);
//! ```

pub mod attention;
pub mod coherence;
pub mod draw_list;
pub mod entity;
pub mod gaussian;
pub mod layer;
pub mod lineage;
pub mod query;
pub mod runtime;
pub mod streaming;
pub mod tile;

// Re-export primary types for convenience.
pub use coherence::{CoherenceDecision, CoherenceGate, CoherenceInput, CoherencePolicy};
pub use draw_list::{DrawCommand, DrawList, DrawListHeader, OpacityMode};
pub use entity::{Edge, EdgeType, Entity, EntityGraph, EntityType};
pub use gaussian::{Gaussian4D, ScreenGaussian};
pub use lineage::{LineageEvent, LineageLog};
pub use streaming::{
    ActiveMask, BandwidthBudget, DeltaPacket, KeyframePacket, SemanticPacket, StreamPacket,
};
pub use attention::{
    AttentionPipeline, AttentionResult, AttentionStats, SemanticAttention, TemporalAttention,
    ViewAttention, WriteAttention,
};
pub use layer::{GaussianLayer, LayerType, LayeredScene};
pub use query::{QueryResult, SceneQuery};
pub use runtime::{LoopCadence, LoopConfig, LoopMetrics, LoopScheduler};
pub use tile::{PrimitiveBlock, QuantTier, Tile, TileCoord};
