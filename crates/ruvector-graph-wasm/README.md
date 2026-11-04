# @ruvector/graph-wasm

[![npm](https://img.shields.io/npm/v/@ruvector/graph-wasm)](https://www.npmjs.com/package/@ruvector/graph-wasm)
[![license](https://img.shields.io/npm/l/@ruvector/graph-wasm)](https://github.com/ruvnet/ruvector/blob/main/LICENSE)

WebAssembly bindings for the RuVector graph database: a Neo4j-style property graph with basic Cypher support and embedding-backed hyperedges (n-ary relationships). Runs in browsers and Node.js.

## Features

- **Property graph** — nodes with labels and properties, typed directed edges with properties
- **Basic Cypher** — `MATCH` / `CREATE` patterns via `query()`, bulk `importCypher()`, full `exportCypher()`
- **Hyperedges** — one relationship connecting any number of nodes, with an embedding and a confidence score
- **Browser & Node.js** — a single ES module build (`--target web`), no native dependencies

## Installation

```bash
npm install @ruvector/graph-wasm
```

## Quick Start

### Browser (ES modules / bundlers)

```javascript
import init, { GraphDB } from '@ruvector/graph-wasm';

await init(); // fetches ruvector_graph_wasm_bg.wasm next to the JS file

const db = new GraphDB('cosine');

const alice = db.createNode(['Person'], { name: 'Alice', age: 30 });
const bob = db.createNode(['Person'], { name: 'Bob', age: 35 });
db.createEdge(alice, bob, 'KNOWS', { since: 2020 });

const result = await db.query('MATCH (n:Person)-[r:KNOWS]->(m) RETURN n, r, m');
console.log(result.count, result.nodes.length, result.edges.length); // 1 2 1

const stats = db.stats();
console.log(`Nodes: ${stats.nodeCount}, Edges: ${stats.edgeCount}`);
```

### Node.js

The package is an ES module built for the web target, so in Node.js load the `.wasm` bytes yourself and initialise synchronously:

```javascript
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { initSync, GraphDB } from '@ruvector/graph-wasm';

const require = createRequire(import.meta.url);
initSync({ module: readFileSync(require.resolve('@ruvector/graph-wasm/ruvector_graph_wasm_bg.wasm')) });

const db = new GraphDB('cosine');
// ... same API as in the browser
```

## API Reference

### `new GraphDB(metric?: string)`

`metric` is the distance metric for hyperedge embeddings: `'cosine'` (default), `'euclidean'`, `'dotproduct'` or `'manhattan'`. Any other value throws `Unknown metric`.

### Nodes

```typescript
createNode(labels: string[], properties: object): string   // returns the new node id (UUID)
getNode(id: string): JsNode | undefined
deleteNode(id: string): boolean                            // also removes its edges and hyperedges
```

A numeric-array `embedding` property is also exposed as `JsNode.embedding` (`Float32Array`) and is required for the node to take part in a hyperedge.

### Edges

```typescript
createEdge(from: string, to: string, type: string, properties: object): string
getEdge(id: string): JsEdge | undefined
deleteEdge(id: string): boolean
```

### Hyperedges

```typescript
createHyperedge(
  nodes: string[],
  description: string,
  embedding?: Float32Array | null,  // null: derived from the member nodes' embeddings
  confidence?: number | null
): string
getHyperedge(id: string): JsHyperedge | undefined
```

Every member node must have an `embedding` property, otherwise `createHyperedge` throws (`No embeddings found for nodes` / `Entity … not found in hypergraph`).

### Queries

```typescript
query(cypher: string): Promise<QueryResult>
importCypher(statements: string[]): Promise<number>  // number of statements executed
exportCypher(): string                               // the whole graph as one CREATE statement
```

Cypher support is basic: label and relationship patterns such as `MATCH (n:Person) RETURN n`, `MATCH (n)-[r:KNOWS]->(m) RETURN n, r, m`, and `CREATE (n:Person {name: "Alice"})`.

### `stats(): object`

`{ nodeCount, edgeCount, hyperedgeCount, hypergraphEntities, hypergraphEdges, avgEntityDegree }`

### Module functions

- `init()` (default export) — async initialisation for browsers; `initSync({ module })` for synchronous initialisation from bytes
- `version(): string`

### Types

```typescript
class JsNode {
  readonly id: string;
  readonly labels: string[];
  readonly properties: any;
  readonly embedding: Float32Array | undefined;
  getProperty(key: string): any;
  hasLabel(label: string): boolean;
}

class JsEdge {
  readonly id: string;
  readonly from: string;
  readonly to: string;
  readonly type: string;
  readonly properties: any;
  getProperty(key: string): any;
}

class JsHyperedge {
  readonly id: string;
  readonly nodes: string[];
  readonly description: string;
  readonly embedding: Float32Array;
  readonly confidence: number;
  readonly properties: any;
  readonly order: number; // number of connected nodes
}

class QueryResult {
  readonly nodes: JsNode[];
  readonly edges: JsEdge[];
  readonly hyperedges: JsHyperedge[];
  readonly data: any;     // one row object per match, keyed by RETURN variable
  readonly count: number;
  isEmpty(): boolean;
}
```

Wasm-bindgen objects own WASM memory; call `.free()` on objects you no longer need in long-running processes.

The module also exports `AsyncQueryExecutor`, `AsyncTransaction`, `BatchOperations` and `ResultStream`. In this release they are placeholders that are not connected to a `GraphDB` instance, so use `GraphDB.query()` / `importCypher()` instead.

## Hypergraph example

```javascript
const doc1 = db.createNode(['Document'], { title: 'AI Research', embedding: [0.1, 0.2, 0.3] });
const doc2 = db.createNode(['Document'], { title: 'ML Tutorial', embedding: [0.3, 0.2, 0.1] });

const id = db.createHyperedge(
  [doc1, doc2],
  'Documents on related topics',
  null, // embedding derived from the member nodes
  0.95
);

const he = db.getHyperedge(id);
console.log(`Hyperedge connects ${he.order} nodes`); // 2
```

## Example: vector search + graph expansion

[`examples/npm-vector-graph`](https://github.com/ruvnet/ruvector/tree/main/examples/npm-vector-graph) pairs this package with [`@ruvector/core`](https://www.npmjs.com/package/@ruvector/core): HNSW top-k search finds the closest notes, then the graph follows their relationships. It runs offline in Node.js.

## Building from source

The Rust source lives in [`crates/ruvector-graph-wasm`](https://github.com/ruvnet/ruvector/tree/main/crates/ruvector-graph-wasm) and is built with `wasm-pack` (see its `build.sh`).

## Related packages

- [`@ruvector/core`](https://www.npmjs.com/package/@ruvector/core) — native HNSW vector database; pair it with this graph for graph-expanded retrieval.
- [`@ruvector/graph-node`](https://www.npmjs.com/package/@ruvector/graph-node) — the same graph database as a native Node.js module.
- [`rvlite`](https://www.npmjs.com/package/rvlite) — embedded vector database with SQL, SPARQL and Cypher in one package.
- [`ruvector`](https://www.npmjs.com/package/ruvector) — umbrella package and CLI (`npx ruvector`).

## License

MIT — see [LICENSE](https://github.com/ruvnet/ruvector/blob/main/LICENSE).
