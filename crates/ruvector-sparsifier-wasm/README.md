# RuVector Sparsifier WASM

[![License](https://img.shields.io/crates/l/ruvector-sparsifier-wasm.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-ruvnet%2Fruvector-blue?logo=github)](https://github.com/ruvnet/ruvector)

**WebAssembly bindings for dynamic spectral graph sparsification.**

Enables browser-side and edge-deployed spectral sparsification with the same guarantees as the native Rust crate.

---

## Quick Start

```typescript
import init, { WasmSparsifier, default_config } from 'ruvector-sparsifier-wasm';

await init();

// Build from edges: [[u, v, weight], ...]
const spar = WasmSparsifier.buildFromEdges(
  '[[0,1,1.0],[1,2,1.0],[2,3,1.0],[3,0,1.0],[0,2,0.5]]',
  default_config()
);

// Dynamic updates
spar.insertEdge(1, 3, 2.0);
spar.deleteEdge(0, 2);

// Audit quality
const audit = JSON.parse(spar.audit());
console.log('Passed:', audit.passed, 'Max error:', audit.max_error);

// Stats
console.log('Compression:', spar.compressionRatio(), 'x');
console.log('Full edges:', spar.numEdges());
console.log('Sparse edges:', spar.sparsifierNumEdges());
```

## API

### `WasmSparsifier`

| Method | Description |
|--------|------------|
| `new(config_json)` | Create empty sparsifier with config |
| `buildFromEdges(edges_json, config_json)` | Build from edge list |
| `insertEdge(u, v, weight)` | Insert edge |
| `deleteEdge(u, v)` | Delete edge |
| `updateEmbedding(node, old_json, new_json)` | Handle point move |
| `audit()` | Run spectral audit (JSON) |
| `sparsifierEdges()` | Get sparsifier edges (JSON) |
| `stats()` | Get statistics (JSON) |
| `compressionRatio()` | Compression ratio |
| `rebuildLocal(nodes_json)` | Rebuild around nodes |
| `rebuildFull()` | Full reconstruction |

### `WasmSparseGraph`

| Method | Description |
|--------|------------|
| `new(n)` | Create graph with n vertices |
| `addEdge(u, v, weight)` | Add edge |
| `removeEdge(u, v)` | Remove edge |
| `degree(u)` | Vertex degree |
| `numEdges()` | Edge count |
| `toJson()` | Serialize to JSON |

### Helpers

| Function | Description |
|----------|------------|
| `init()` | Initialize WASM module |
| `version()` | Crate version |
| `default_config()` | Default config JSON |

## Build

```bash
wasm-pack build crates/ruvector-sparsifier-wasm --target web
```

## Memory

For a 10K-vertex graph: ~1.6 MB WASM linear memory (26 pages).

## License

MIT
