# ruvector-vwm

## What is this?

This crate turns video and 3D data into a persistent, queryable world model. Instead of storing raw pixels, it stores the actual objects, their positions, movements, and relationships as **4D Gaussian primitives** -- volumetric elements that know where they are in space AND time. You get a structured representation of reality that you can query, stream, diff, and render on demand.

## Why does this matter?

- **Query reality like a database.** Ask "show me all forklifts near bay 3 between 2-4pm" instead of scrubbing through hours of footage.
- **Render only what matters.** Stream compact deltas instead of full video frames. 10x bandwidth reduction is the starting point, not the ceiling.
- **Privacy by design.** Store geometric structure and semantic labels. Discard the raw imagery entirely. No faces, no license plates, no liability.
- **Continuous learning without retraining.** The world model updates incrementally from new sensor data. No batch retraining cycles, no model versioning headaches.
- **Audit trail for every change.** Every mutation is logged with full provenance -- who changed what, when, why, and what the coherence score was at the time.

## Quick Start

```rust
use ruvector_vwm::gaussian::Gaussian4D;
use ruvector_vwm::tile::{PrimitiveBlock, QuantTier};
use ruvector_vwm::draw_list::{DrawList, OpacityMode};
use ruvector_vwm::coherence::{CoherenceGate, CoherenceInput, PermissionLevel};

// Create two Gaussians at known positions
let g1 = Gaussian4D::new([0.0, 0.0, -5.0], 0);
let g2 = Gaussian4D::new([1.0, 0.0, -5.0], 1);

// Pack them into a quantized tile block
let block = PrimitiveBlock::encode(&[g1, g2], QuantTier::Hot8);
assert_eq!(block.count, 2);

// Build a draw list for the renderer
let mut draw_list = DrawList::new(1, 0, 0);
draw_list.bind_tile(42, 0, QuantTier::Hot8);
draw_list.draw_block(0, 0.5, OpacityMode::AlphaBlend);
draw_list.finalize();

// Decide whether to accept an incoming update
let gate = CoherenceGate::with_defaults();
let input = CoherenceInput {
    tile_disagreement: 0.1,
    entity_continuity: 0.9,
    sensor_confidence: 1.0,
    sensor_freshness_ms: 50,
    budget_pressure: 0.2,
    permission_level: PermissionLevel::Standard,
};
let decision = gate.evaluate(&input);
assert_eq!(decision, ruvector_vwm::coherence::CoherenceDecision::Accept);
```

## Core Concepts

### The Three Loops

The world model runs on three concurrent loops, each at a different speed, each with a different job:

1. **Render Loop (~60 Hz, ~16ms).** The fast path. Takes a camera pose, figures out which tiles are visible, sorts the Gaussians front-to-back, builds a packed `DrawList`, and hands it to the GPU. This loop never touches the world model directly -- it only reads pre-built draw lists.

2. **Update Loop (~2-10 Hz, ~100ms).** The integration path. New sensor data arrives, gets checked by the `CoherenceGate`, and -- if accepted -- mutates the tile's `PrimitiveBlock`. Entity graph relationships are updated. Stream packets are emitted for remote consumers. The lineage log records what happened and why.

3. **Governance Loop (~0.1-1 Hz, ~1s+).** The policy path. Audits the lineage log, enforces privacy rules, manages tile lifecycle (creation, merging, eviction), and tunes coherence thresholds. This is where the system thinks about whether its own state makes sense.

```text
                         Governance Loop (1 Hz)
                    +---------------------------------+
                    | Lineage Audit -> Privacy Check  |
                    | -> Tile Lifecycle -> Policy      |
                    +---------------------------------+
                              |         ^
                              v         |
                         Update Loop (2-10 Hz)
                    +---------------------------------+
                    | Sensor Data -> Coherence Gate   |
                    | -> Tile Update -> Lineage Log   |
                    | -> Stream Packets               |
                    +---------------------------------+
                              |         ^
                              v         |
                         Render Loop (60 Hz)
                    +---------------------------------+
                    | Camera Pose -> Tile Visibility  |
                    | -> Sort Gaussians -> DrawList   |
                    | -> Rasterize                    |
                    +---------------------------------+
```

---

<details>
<summary><h3>4D Gaussians</h3></summary>

#### What are they?

A Gaussian splat is a soft, fuzzy blob in 3D space. Think of it as a tiny colored cloud with a position, a shape (how stretched or squished it is in each direction), a color, and an opacity. Thousands of these blobs, layered together, can represent a photorealistic scene.

**4D** means each Gaussian also knows about time. It has a `time_range` (when it exists) and a `velocity` (how it moves). This lets you evaluate where any object was at any point in time without storing separate snapshots.

#### Why 4D, not 3D?

3D Gaussians give you a frozen moment. To represent a 10-minute scene, you would need hundreds of separate 3D snapshots. 4D Gaussians encode motion directly: a single primitive can represent a moving object across its entire lifespan. This is dramatically more compact and lets you query arbitrary timestamps.

#### The linear motion model

Position at time `t` is computed as:

```
position(t) = center + velocity * (t - t_mid)
```

where `t_mid` is the midpoint of the Gaussian's time range. This is cheap to evaluate (three multiply-adds) and good enough for most real-world motion over short time windows.

#### Code example

```rust
use ruvector_vwm::gaussian::Gaussian4D;

// Create a Gaussian at position (1, 2, -5) with ID 0
let mut g = Gaussian4D::new([1.0, 2.0, -5.0], 0);

// Give it a time range and velocity
g.time_range = [0.0, 10.0];        // exists from t=0 to t=10
g.velocity = [0.5, 0.0, 0.0];      // moving along X axis

// Where is it at t=7?
// t_mid = 5.0, so position = [1.0 + 0.5*(7-5), 2.0, -5.0] = [2.0, 2.0, -5.0]
let pos = g.position_at(7.0);
assert!((pos[0] - 2.0).abs() < 1e-6);

// Is it active at t=12? No.
assert!(!g.is_active_at(12.0));

// Project to screen space with a view-projection matrix
let view_proj: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];
if let Some(screen_g) = g.project(&view_proj, 5.0) {
    println!("Screen position: {:?}", screen_g.center_screen);
    println!("Depth: {}", screen_g.depth);
    println!("Radius: {}", screen_g.radius);
}
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `center` | `[f32; 3]` | XYZ position in world space |
| `covariance` | `[f32; 6]` | Upper triangle of the 3x3 covariance matrix (shape) |
| `sh_coeffs` | `[f32; 3]` | Spherical harmonics for view-dependent color (degree 0 RGB) |
| `opacity` | `f32` | Transparency, 0.0 (invisible) to 1.0 (solid) |
| `scale` | `[f32; 3]` | Per-axis scale factors |
| `rotation` | `[f32; 4]` | Orientation as a quaternion [w, x, y, z] |
| `time_range` | `[f32; 2]` | When this Gaussian exists [start, end] |
| `velocity` | `[f32; 3]` | Per-axis velocity for the linear motion model |
| `id` | `u32` | Unique ID within its tile |

</details>

---

<details>
<summary><h3>Spacetime Tiles</h3></summary>

#### What are they?

The world is divided into a regular 3D spatial grid, and each grid cell is further divided by time buckets and levels of detail. Each cell is a **tile**, addressed by a `TileCoord`:

```rust
pub struct TileCoord {
    pub x: i32,        // spatial X
    pub y: i32,        // spatial Y
    pub z: i32,        // spatial Z
    pub time_bucket: u32,  // which time window
    pub lod: u8,       // level of detail
}
```

Every tile holds a `PrimitiveBlock` -- a packed binary buffer of Gaussian data -- along with references to the entities it contains, a coherence score, and a last-update epoch.

#### Quantization tiers

Not all tiles need full precision. The system supports four compression tiers:

| Tier | Bits | Compression | Use case |
|------|------|-------------|----------|
| `Hot8` | 8-bit | ~4x | Active tiles, currently being rendered |
| `Warm7` | 7-bit | ~4.57x | Recently used tiles, might be needed soon |
| `Warm5` | 5-bit | ~6.4x | Background tiles, lower priority |
| `Cold3` | 3-bit | ~10.67x | Archived tiles, rarely accessed |

The tier is recorded in the block metadata. The encode/decode pipeline currently stores raw `f32` bytes (quantized packing is planned for a future iteration), but the tier tag travels with the data so downstream consumers know the intended fidelity.

#### Code example

```rust
use ruvector_vwm::gaussian::Gaussian4D;
use ruvector_vwm::tile::{PrimitiveBlock, QuantTier};

// Create some Gaussians
let g1 = Gaussian4D::new([1.0, 2.0, 3.0], 10);
let g2 = Gaussian4D::new([4.0, 5.0, 6.0], 20);

// Encode into a primitive block
let block = PrimitiveBlock::encode(&[g1, g2], QuantTier::Hot8);
assert_eq!(block.count, 2);

// Verify data integrity
assert!(block.verify_checksum());

// Decode back to Gaussians
let decoded = block.decode();
assert_eq!(decoded[0].center, [1.0, 2.0, 3.0]);
assert_eq!(decoded[0].id, 10);
assert_eq!(decoded[1].center, [4.0, 5.0, 6.0]);
assert_eq!(decoded[1].id, 20);
```

Each primitive block includes a `DecodeDescriptor` that records byte offsets and scale factors for every field, so decoders do not need to hardcode the layout.

</details>

---

<details>
<summary><h3>Draw Lists</h3></summary>

#### Why the renderer never queries the world model

Direct queries from a 60fps render loop into a mutable world model would be a recipe for contention and inconsistency. Instead, the update loop pre-builds a **draw list** -- a flat, packed sequence of commands that tells the renderer exactly what to do. The renderer just plays the list forward. No locks, no queries, no surprises.

#### What is in a draw list?

A `DrawList` contains a header (epoch, sequence number, budget profile, checksum) and a stream of `DrawCommand`s:

| Command | Purpose |
|---------|---------|
| `TileBind` | Bind a tile's primitive block for subsequent draws |
| `SetBudget` | Set a per-screen-tile Gaussian budget and overdraw limit |
| `DrawBlock` | Issue a draw call for a bound block with a sort key and blend mode |
| `End` | Sentinel marking the end of the stream |

#### Code example

```rust
use ruvector_vwm::draw_list::{DrawList, OpacityMode};
use ruvector_vwm::tile::QuantTier;

// Create a draw list for epoch 1, frame 0
let mut dl = DrawList::new(1, 0, 0);

// Bind tile 42's primitive block
dl.bind_tile(42, 0, QuantTier::Hot8);

// Set a budget: screen tile 0 gets at most 1000 Gaussians, 2x overdraw
dl.set_budget(0, 1000, 2.0);

// Draw the bound block with alpha blending
dl.draw_block(0, 0.5, OpacityMode::AlphaBlend);

// Finalize computes the checksum and appends an End sentinel
let checksum = dl.finalize();

// Serialize to bytes for GPU upload or network transport
let bytes = dl.to_bytes();
assert!(bytes.len() > 20); // header alone is 20 bytes
```

The wire format is fully little-endian: 20-byte header followed by tagged command payloads. Three blend modes are supported: `AlphaBlend` (standard transparency), `Additive` (glow/emissive), and `Opaque` (no blending).

</details>

---

<details>
<summary><h3>Coherence Gate</h3></summary>

#### What it does

Not every incoming sensor update should be applied. The `CoherenceGate` is the bouncer at the door of the world model. It evaluates each proposed update against a tunable policy and decides:

| Decision | Meaning |
|----------|---------|
| **Accept** | The update is consistent; apply it. |
| **Defer** | Something is off, but not critically. Try again next cycle. |
| **Freeze** | The tile is in an unstable state. Stop all updates until conditions improve. |
| **Rollback** | The tile has diverged too far. Revert to the last known-good state. |

#### How it decides

The gate evaluates inputs in priority order:

1. **Admin permission** always accepts (override for emergencies).
2. **Read-only permission** always defers (no writes allowed).
3. **Stale data** (older than `max_staleness_ms`) is deferred.
4. **Very high tile disagreement** (>= `rollback_disagreement`, default 0.95) triggers rollback.
5. **High tile disagreement** (>= `freeze_disagreement`, default 0.80) triggers freeze.
6. **High budget pressure** (>= `budget_freeze_threshold`, default 0.90) triggers freeze.
7. **Entity continuity** (weighted by sensor confidence) determines accept vs. defer. Elevated permissions get a small continuity boost (+0.1).

#### Code example

```rust
use ruvector_vwm::coherence::{
    CoherenceGate, CoherenceInput, CoherencePolicy, CoherenceDecision, PermissionLevel,
};

// Use default thresholds
let gate = CoherenceGate::with_defaults();

// A solid update: low disagreement, high continuity, fresh data
let good_input = CoherenceInput {
    tile_disagreement: 0.1,
    entity_continuity: 0.9,
    sensor_confidence: 1.0,
    sensor_freshness_ms: 50,
    budget_pressure: 0.2,
    permission_level: PermissionLevel::Standard,
};
assert_eq!(gate.evaluate(&good_input), CoherenceDecision::Accept);

// A suspicious update: high disagreement
let bad_input = CoherenceInput {
    tile_disagreement: 0.96,
    entity_continuity: 0.9,
    sensor_confidence: 1.0,
    sensor_freshness_ms: 50,
    budget_pressure: 0.2,
    permission_level: PermissionLevel::Standard,
};
assert_eq!(gate.evaluate(&bad_input), CoherenceDecision::Rollback);

// Custom policy: make the gate stricter
let mut strict_gate = CoherenceGate::with_defaults();
strict_gate.update_policy(CoherencePolicy {
    accept_threshold: 0.95,   // much harder to accept
    defer_threshold: 0.6,
    freeze_disagreement: 0.7,
    rollback_disagreement: 0.9,
    max_staleness_ms: 2000,
    budget_freeze_threshold: 0.8,
});
```

</details>

---

<details>
<summary><h3>Entity Graph</h3></summary>

#### What it represents

The entity graph is the semantic layer on top of the geometric world. While tiles hold raw Gaussians, the entity graph holds **meaning**: this cluster of Gaussians is a forklift, that region is loading bay 3, and there is a causal relationship between the forklift entering the bay and the alarm going off.

#### Node types

| Type | Description |
|------|-------------|
| `Object` | A physical thing with a class label ("car", "person", "pallet") |
| `Track` | A temporal sequence of observations of the same object |
| `Region` | A named spatial area |
| `Event` | A discrete occurrence (arrival, departure, collision) |

Each entity has an ID, a time span, an optional embedding vector (for similarity search), confidence score, privacy tags, arbitrary key-value attributes, and references to its underlying Gaussian IDs.

#### Edge types

| Type | Description |
|------|-------------|
| `Adjacency` | Two entities are spatially near each other |
| `Containment` | One entity is inside another (pallet inside truck) |
| `Continuity` | Same object observed at different times |
| `Causality` | One event caused another |
| `SameIdentity` | Two observations are the same real-world entity |

All edges are weighted and optionally time-bounded.

#### Code example

```rust
use ruvector_vwm::entity::{
    EntityGraph, Entity, EntityType, Edge, EdgeType, AttributeValue,
};

let mut graph = EntityGraph::new();

// Add a forklift
graph.add_entity(Entity {
    id: 1,
    entity_type: EntityType::Object { class: "forklift".into() },
    time_span: [100.0, 500.0],
    embedding: vec![],
    confidence: 0.95,
    privacy_tags: vec![],
    attributes: vec![
        ("speed_mps".into(), AttributeValue::Float(2.5)),
    ],
    gaussian_ids: vec![10, 11, 12, 13],
});

// Add a loading bay region
graph.add_entity(Entity {
    id: 2,
    entity_type: EntityType::Region,
    time_span: [0.0, f32::INFINITY],
    embedding: vec![],
    confidence: 1.0,
    privacy_tags: vec![],
    attributes: vec![
        ("name".into(), AttributeValue::Text("Bay 3".into())),
    ],
    gaussian_ids: vec![],
});

// Connect them
graph.add_edge(Edge {
    source: 1,
    target: 2,
    edge_type: EdgeType::Containment,
    weight: 1.0,
    time_range: Some([200.0, 400.0]),
});

// Query by type
let forklifts = graph.query_by_type("forklift");
assert_eq!(forklifts.len(), 1);

// Query by time range
let active = graph.query_time_range(150.0, 300.0);
assert_eq!(active.len(), 2); // both entities overlap this window

// Find neighbors
let neighbors = graph.neighbors(1);
assert_eq!(neighbors.len(), 1);
assert_eq!(neighbors[0].id, 2);
```

</details>

---

<details>
<summary><h3>Lineage Log</h3></summary>

#### What it does

The lineage log is an **append-only** record of every mutation to the world model. Every tile creation, update, merge, entity change, freeze, and rollback is captured with:

- A monotonically increasing event ID
- A wall-clock timestamp
- The affected tile ID
- The type of mutation (and relevant metadata like delta sizes or merged tile IDs)
- Full provenance: which sensor, model, or user produced the data, at what confidence, with an optional cryptographic signature
- The coherence decision and score at the time of the event
- An optional rollback pointer to a known-good state

This gives you a complete audit trail. You can answer "what happened to tile 42 between 1pm and 2pm?" or "find the last consistent state of this tile so we can roll back."

#### Code example

```rust
use ruvector_vwm::lineage::{
    LineageLog, LineageEventType, Provenance, ProvenanceSource,
};
use ruvector_vwm::coherence::CoherenceDecision;

let mut log = LineageLog::new();

// Record a tile creation
let event_id = log.append(
    1000,                              // timestamp_ms
    42,                                // tile_id
    LineageEventType::TileCreated,
    Provenance {
        source: ProvenanceSource::Sensor { sensor_id: "cam-north-01".into() },
        confidence: 0.95,
        signature: None,
    },
    None,                              // no rollback pointer
    CoherenceDecision::Accept,
    0.95,                              // coherence score
);

// Record an update
log.append(
    2000,
    42,
    LineageEventType::TileUpdated { delta_size: 1024 },
    Provenance {
        source: ProvenanceSource::Inference { model_id: "yolo-v9".into() },
        confidence: 0.88,
        signature: None,
    },
    Some(event_id),                    // can roll back to the creation event
    CoherenceDecision::Accept,
    0.88,
);

// Query the tile's history
let history = log.query_tile(42);
assert_eq!(history.len(), 2);

// Query by time range
let recent = log.query_range(1500, 2500);
assert_eq!(recent.len(), 1);

// Find the best rollback point
let rollback = log.find_rollback_point(42);
assert_eq!(rollback, Some(0)); // points to the creation event
```

</details>

---

<details>
<summary><h3>Streaming Protocol</h3></summary>

#### How data moves over the network

The streaming protocol defines three packet types for getting world-model data from producers to consumers:

| Packet | Purpose | Size |
|--------|---------|------|
| `KeyframePacket` | Full tile snapshot. Sent periodically or when a consumer joins. | Large (full block) |
| `DeltaPacket` | Only the Gaussians that changed since a base keyframe. | Small (active changes only) |
| `SemanticPacket` | Entity-level updates: new embeddings, attribute changes. | Variable |

In steady state, most traffic is delta packets. Keyframes are expensive but necessary for random access and new subscriber bootstrapping.

#### Active masks

A `DeltaPacket` includes an `ActiveMask` -- a compact bitmask that indicates which Gaussians are active in the current time window. This uses packed `u64` words for O(1) set/get per Gaussian:

```rust
use ruvector_vwm::streaming::ActiveMask;

// Track 1000 Gaussians
let mut mask = ActiveMask::new(1000);

// Mark some as active
mask.set(0, true);
mask.set(42, true);
mask.set(999, true);

assert_eq!(mask.active_count(), 3);
assert!(mask.is_active(42));
assert!(!mask.is_active(500));

// Storage is compact: ceil(1000/64) = 16 words = 128 bytes
assert_eq!(mask.byte_size(), 128);
```

#### Bandwidth budgeting

The `BandwidthBudget` controller prevents producers from flooding the network. It tracks bytes sent in a 1-second sliding window and refuses sends that would exceed the cap:

```rust
use ruvector_vwm::streaming::BandwidthBudget;

// Allow 1 MB/s
let mut budget = BandwidthBudget::new(1_000_000);
budget.reset_window(0);

// Send some data
assert!(budget.can_send(500_000, 0));
budget.record_sent(500_000, 0);

// Check remaining capacity
assert!(budget.can_send(500_000, 0));   // exactly at limit
assert!(!budget.can_send(500_001, 0));  // over limit

// After 1 second, the window resets automatically
assert!(budget.can_send(1_000_000, 1000));

// Check utilization
budget.record_sent(200_000, 1500);
assert!((budget.utilization() - 0.2).abs() < 1e-6);
```

</details>

---

## Use Cases

<details>
<summary><h3>Product Tier</h3></summary>

**1. Searchable, Rewindable Reality**

Instead of watching hours of security footage, query the world model directly. "Show me every time a person entered zone B after 6pm." The entity graph and time-range queries make this a data retrieval problem, not a video analysis problem.

**2. Industrial Digital Twins**

Point cameras at a warehouse floor. The update loop continuously integrates new observations, the coherence gate rejects bad sensor readings, and the entity graph maintains a live map of assets, vehicles, and personnel. No manual model updates.

**3. Bandwidth Collapse**

A full HD video stream at 30fps is roughly 5 Mbps. A delta stream of Gaussian updates for the same scene can be 10-100x smaller because you are only sending what changed, in a structured format the receiver already understands.

**4. Privacy-First Perception**

The world model stores shapes, positions, and semantic labels -- not pixels. You can track that "a person walked from A to B at 3pm" without storing any image of that person. Raw imagery can be discarded at the edge.

</details>

<details>
<summary><h3>Research Tier</h3></summary>

**5. Always-Learning Perception**

New sensor data is integrated through the update loop without retraining a model from scratch. The coherence gate ensures only consistent updates are accepted, so the world model improves incrementally.

**6. Intent-Driven Rendering**

The draw list protocol lets you render only what a particular consumer needs. A safety system might only care about Gaussians near machinery. A navigation system might only need floor-level tiles. Budget profiles make this explicit.

**7. Stable Identity**

Entity continuity edges in the graph, combined with the coherence gate's continuity scoring, prevent object identity from drifting. If a forklift is briefly occluded, its entity persists rather than fragmenting into two new objects.

**8. Perceptual Memory**

The lineage log and entity graph together form a structured memory. The system remembers not just "what is here now" but "what was here, when, and how confident we were about it."

**9. Self-Stabilizing World**

The coherence gate acts as an anti-hallucination mechanism. Updates that disagree too strongly with existing state are frozen or rolled back, preventing the world model from drifting into an inconsistent state.

**10. Multi-Agent Shared Reality**

Multiple robots or systems can subscribe to the same world model via the streaming protocol. They share a common, consistent representation of the environment rather than each maintaining their own potentially contradictory view.

</details>

<details>
<summary><h3>Frontier Tier</h3></summary>

**11. Time as Reasoning**

Because every Gaussian has a temporal extent and velocity, you can detect causal drift -- changes in the temporal geometry of the scene that indicate something unusual is happening. A forklift that normally takes 2 minutes to traverse a bay but is now taking 10 is detectable purely from the temporal structure.

**12. Programmable Substrate**

The same world model serves multiple applications simultaneously. A safety system reads entity proximities. An optimization system analyzes traffic patterns. A simulation system renders what-if scenarios. Different draw list budget profiles, same underlying data.

**13. Memory that Improves**

Long-running systems accumulate lineage data that makes the coherence gate better calibrated over time. The system learns what "normal" disagreement looks like for each tile and sensor, making it progressively more stable and accurate.

</details>

## Architecture

```text
                    +---------------------------+
                    |    Governance Loop (1Hz)   |
                    |  Lineage Log  |  Privacy   |
                    |  Tile Lifecycle | Policy   |
                    +----------+----+-----------+
                               |    ^
                               v    |
                    +----------+----+-----------+
                    |    Update Loop (2-10Hz)    |
                    |                           |
  Sensor Data ---->|  Coherence   Entity Graph  |----> Stream Packets
                    |    Gate    /              |       (Keyframe/Delta/
                    |           /               |        Semantic)
                    |  Tile    /   Lineage Log  |
                    |  Update /                 |
                    +----------+----+-----------+
                               |    ^
                               v    |
                    +----------+----+-----------+
                    |    Render Loop (60Hz)      |
                    |                           |
  Camera Pose ---->|  Tile       Sort     Draw  |----> Rasterized
                    |  Visibility  Gaussians List|      Output
                    +---------------------------+

  Data Structures:
  +-------------+    +----------------+    +------------+
  | Gaussian4D  |--->| PrimitiveBlock |--->| DrawList   |
  | (primitive) |    | (packed tile)  |    | (GPU cmds) |
  +-------------+    +----------------+    +------------+
        |                   |
        v                   v
  +-------------+    +----------------+
  | EntityGraph |    | LineageLog     |
  | (semantics) |    | (provenance)   |
  +-------------+    +----------------+
```

## Performance

- **Zero external dependencies** for full WASM compatibility. The only `use` outside `std` is internal crate modules.
- **FNV-1a checksums** for data integrity on primitive blocks and draw lists. Fast, non-cryptographic, good distribution.
- **Packed binary serialization** for draw lists. The wire format is designed for direct GPU upload without intermediate parsing.
- **Active bit masks** with O(1) set/get via packed `u64` words. Tracking 10,000 Gaussians costs 1,250 bytes.
- **104 bytes per Gaussian** in the raw encoding (25 floats + 1 u32 ID), with quantization tiers ready for future compression down to ~10 bytes per Gaussian at Cold3.

## WASM Support

This crate is designed for full WASM compatibility with zero external dependencies. A companion crate, `ruvector-vwm-wasm`, provides the wasm-bindgen bindings for browser and edge deployment. The same core logic runs natively or in WebAssembly without conditional compilation.

## License

MIT
