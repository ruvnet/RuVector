# ruvector-vwm-wasm

WASM bindings for the RuVector Visual World Model. Run 4D Gaussian splatting, coherence gating, entity graphs, and streaming in the browser.

## Install

```bash
# Build with wasm-pack
wasm-pack build --target web

# Or for Node.js
wasm-pack build --target nodejs
```

## Quick Start (JavaScript)

```javascript
import init, {
  initVwm, WasmGaussian4D, WasmDrawList,
  WasmCoherenceGate, WasmEntityGraph
} from './pkg/ruvector_vwm_wasm.js';

await init();
initVwm();

// Create a Gaussian at position (0, 1, -5) with ID 42
const g = new WasmGaussian4D(0.0, 1.0, -5.0, 42);
g.setVelocity(0.1, 0.0, 0.0);
g.setTimeRange(0.0, 10.0);
g.setColor(1.0, 0.5, 0.0); // orange

// Check position at t=7.5
const pos = g.positionAt(7.5);
console.log('Position:', pos); // Float32Array [0.25, 1.0, -5.0]
```

## API Reference

### `initVwm()`
Initializes the RuVector Visual World Model runtime. Call once at startup.

**Signature:** `initVwm() → void`

### `version() → string`
Returns the current version of the WASM library.

**Signature:** `version() → string`

**Returns:** Semantic version string (e.g., "0.1.0")

### `WasmGaussian4D`
Represents a 4D Gaussian splatting primitive with motion and time bounds.

**Constructor:**
```javascript
new WasmGaussian4D(x: f32, y: f32, z: f32, id: u64)
```

**Methods:**
- `setVelocity(vx: f32, vy: f32, vz: f32) → void` - Set velocity vector
- `setTimeRange(startTime: f32, endTime: f32) → void` - Set temporal bounds
- `setColor(r: f32, g: f32, b: f32) → void` - Set RGB color (0.0-1.0)
- `setOpacity(alpha: f32) → void` - Set opacity (0.0-1.0)
- `setCovariance(cov: Float32Array) → void` - Set 3×3 covariance matrix
- `positionAt(time: f32) → Float32Array` - Compute position at given time
- `getId() → BigInt` - Get unique identifier
- `serialize() → Uint8Array` - Encode to binary format

### `WasmActiveMask`
Manages active/inactive state for Gaussians with spatial or temporal culling.

**Constructor:**
```javascript
new WasmActiveMask(capacity: u32)
```

**Methods:**
- `enable(id: u64) → void` - Mark Gaussian as active
- `disable(id: u64) → void` - Mark Gaussian as inactive
- `isActive(id: u64) → boolean` - Query active status
- `countActive() → u32` - Get number of active Gaussians
- `clear() → void` - Reset all masks

### `WasmDrawList`
Batches Gaussians into sortable draw commands for GPU rendering.

**Constructor:**
```javascript
new WasmDrawList(tileId: u64, startX: u32, endX: u32)
```

**Methods:**
- `bindTile(layerId: u64, quantTier: u8, reserved: u32) → void` - Bind a tile layer
- `setBudget(index: u32, byteLimit: u32, pixelRatio: f32) → void` - Set memory budget
- `drawBlock(mode: u8, depth: f32, flags: u32) → void` - Issue draw command
- `finalize() → u32` - Compute final checksum and lock list
- `toBytes() → Uint8Array` - Serialize to binary (for GPU upload)
- `getCount() → u32` - Get number of draw commands

### `WasmCoherenceGate`
Evaluates temporal coherence decisions for entity streaming.

**Constructor:**
```javascript
new WasmCoherenceGate()
```

**Methods:**
- `evaluate(disagreement: f32, continuity: f32, confidence: f32, freshnessMs: i32, pressure: f32, permission: u8) → string` - Evaluate coherence
  - Returns: "accept", "defer", or "reject"
- `setPressureThreshold(threshold: f32) → void` - Override pressure threshold
- `reset() → void` - Clear internal state

### `WasmEntityGraph`
Stores and queries entity relationships as a directed graph.

**Constructor:**
```javascript
new WasmEntityGraph()
```

**Methods:**
- `addEntity(id: u64, type: string) → void` - Insert an entity node
- `removeEntity(id: u64) → void` - Remove an entity and its edges
- `addEdge(from: u64, to: u64, edgeType: string) → void` - Add directed edge
- `queryByType(type: string) → BigUint64Array` - Get all entities of given type
- `queryNeighbors(id: u64) → BigUint64Array` - Get neighbors of entity
- `queryEdgeType(from: u64, to: u64) → string | null` - Get edge label
- `toJson() → string` - Export graph as JSON
- `clear() → void` - Remove all entities and edges

### `WasmLineageLog`
Tracks historical events and tile mutations for audit/replay.

**Constructor:**
```javascript
new WasmLineageLog(capacity: u32)
```

**Methods:**
- `appendEvent(timestamp: f32, tileId: u64, mutation: string) → void` - Log an event
- `queryTile(tileId: u64) → string` - Get history for a tile (JSON)
- `queryRange(startTime: f32, endTime: f32) → string` - Get events in time window (JSON)
- `size() → u32` - Number of stored events
- `clear() → void` - Clear all logs

### `WasmBandwidthBudget`
Rate-limits streaming to respect network/compute constraints.

**Constructor:**
```javascript
new WasmBandwidthBudget(bytesPerSecond: u32)
```

**Methods:**
- `canTransmit(bytes: u32) → boolean` - Check if transmission fits budget
- `consume(bytes: u32) → void` - Debit from budget
- `refill() → void` - Reset budget for next interval
- `setBudget(bytesPerSecond: u32) → void` - Change bandwidth limit

## Examples

### <details><summary>Build a Draw List</summary>

```javascript
const dl = new WasmDrawList(1n, 0, 100);
dl.bindTile(42n, 1, 0); // Hot8 quantization
dl.setBudget(0, 1024, 2.0); // 1KB limit, 2x pixel ratio
dl.drawBlock(1, 0.5, 0); // Additive blend at depth 0.5
const checksum = dl.finalize();
const bytes = dl.toBytes(); // Uint8Array for GPU upload
console.log('Draw list checksum:', checksum);
console.log('Serialized size:', bytes.length);
```

</details>

### <details><summary>Evaluate Coherence</summary>

```javascript
const gate = new WasmCoherenceGate();

// Args: disagreement, continuity, confidence, freshness_ms, pressure, permission
const decision = gate.evaluate(
  0.1,    // low disagreement
  0.9,    // high continuity
  1.0,    // perfect confidence
  100,    // 100ms fresh data
  0.3,    // low pressure
  1       // Standard permission
);
console.log('Coherence decision:', decision); // "accept"

// High pressure scenario
const stressed = gate.evaluate(0.5, 0.5, 0.8, 500, 0.8, 1);
console.log('Under stress:', stressed); // likely "defer"
```

</details>

### <details><summary>Entity Graph Queries</summary>

```javascript
const graph = new WasmEntityGraph();

// Add entities with types
graph.addEntity(1n, "person");
graph.addEntity(2n, "object");
graph.addEntity(3n, "person");

// Create relationships
graph.addEdge(1n, 2n, "holds");
graph.addEdge(1n, 3n, "interacts_with");
graph.addEdge(2n, 3n, "near");

// Query all people
const people = graph.queryByType("person");
console.log('People:', people); // BigUint64Array(2) [1n, 3n]

// Get neighbors of entity 1
const neighbors = graph.queryNeighbors(1n);
console.log('Entity 1 neighbors:', neighbors); // [2n, 3n]

// Check relationship type
const edgeType = graph.queryEdgeType(1n, 2n);
console.log('Edge (1→2):', edgeType); // "holds"

// Export as JSON
const json = graph.toJson();
console.log(json);
```

</details>

### <details><summary>Lineage Tracking</summary>

```javascript
const log = new WasmLineageLog(1000);

// Append events with timestamps and mutations
log.appendEvent(0.0, 1n, "created");
log.appendEvent(1.5, 1n, "quantized_to_warm7");
log.appendEvent(3.2, 1n, "coherence_accepted");
log.appendEvent(5.0, 1n, "color_updated");

// Query history of a tile
const tileHistory = log.queryTile(1n);
console.log('Tile 1 history:', tileHistory);

// Query time window [1.0, 4.0]
const range = log.queryRange(1.0, 4.0);
console.log('Events in [1.0, 4.0]:', range);

console.log('Total events logged:', log.size());
```

</details>

### <details><summary>Bandwidth Control</summary>

```javascript
const budget = new WasmBandwidthBudget(10000); // 10KB/sec

function tryStream(data) {
  if (budget.canTransmit(data.length)) {
    console.log('Sending', data.length, 'bytes');
    budget.consume(data.length);
    // Actually send data...
  } else {
    console.log('Budget exceeded, deferring transmission');
  }
}

// Simulate streaming
tryStream(new Uint8Array(5000));  // Success
tryStream(new Uint8Array(4000));  // Success
tryStream(new Uint8Array(3000));  // Deferred (would exceed 10KB)

// Reset for next time interval
budget.refill();
tryStream(new Uint8Array(8000));  // Success after refill
```

</details>

## Type Mappings

| Rust | WASM/JS | Notes |
|------|---------|-------|
| QuantTier | u8 (0-3) | 0=Hot8, 1=Warm7, 2=Warm5, 3=Cold3 |
| OpacityMode | u8 (0-2) | 0=AlphaBlend, 1=Additive, 2=Opaque |
| PermissionLevel | u8 (0-3) | 0=ReadOnly, 1=Standard, 2=Elevated, 3=Admin |
| EdgeType | string | "adjacency", "containment", "continuity", "causality", "same_identity" |

## License

MIT
