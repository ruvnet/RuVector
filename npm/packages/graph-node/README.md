# @ruvector/graph-node

Native Node.js bindings for RuVector Graph Database with hypergraph support, Cypher queries, and persistence. **10x faster than WASM**.

## Features

- **Native Performance**: Direct NAPI-RS bindings - no WASM overhead
- **Hypergraph Support**: Multi-node relationships with vector embeddings
- **Cypher Queries**: a practical subset of Cypher — see [Cypher support](#cypher-support)
- **Persistence**: ACID-compliant storage with redb backend
- **Vector Similarity Search**: Fast k-NN search on embeddings
- **Graph Traversal**: k-hop neighbor discovery
- **Transactions**: Full ACID support with begin/commit/rollback
- **Batch Operations**: High-throughput bulk inserts (131K+ ops/sec)
- **Zero-Copy**: Efficient Float32Array handling
- **TypeScript**: Full type definitions included

## Installation

```bash
npm install @ruvector/graph-node
```

## Quick Start

```javascript
const { GraphDatabase } = require('@ruvector/graph-node');

// Create an in-memory database
const db = new GraphDatabase({
  distanceMetric: 'Cosine',
  dimensions: 384
});

// Or create a persistent database
const persistentDb = new GraphDatabase({
  distanceMetric: 'Cosine',
  dimensions: 384,
  storagePath: './my-graph.db'
});

// Or open an existing database
const existingDb = GraphDatabase.open('./my-graph.db');

// Create nodes
await db.createNode({
  id: 'alice',
  embedding: new Float32Array([1.0, 0.0, 0.0, /* ... */]),
  labels: ['Person', 'Employee'],
  properties: { name: 'Alice', age: '30' }
});

// Create edges
await db.createEdge({
  from: 'alice',
  to: 'bob',
  description: 'KNOWS',
  embedding: new Float32Array([0.5, 0.5, 0.0, /* ... */]),
  confidence: 0.95
});

// Create hyperedges (multi-node relationships)
await db.createHyperedge({
  nodes: ['alice', 'bob', 'charlie'],
  description: 'COLLABORATED_ON_PROJECT',
  embedding: new Float32Array([0.33, 0.33, 0.33, /* ... */]),
  confidence: 0.85
});

// Query with Cypher
const results = await db.query('MATCH (n:Person) RETURN n');

// Vector similarity search
const similar = await db.searchHyperedges({
  embedding: new Float32Array([0.3, 0.3, 0.3, /* ... */]),
  k: 10
});

// Get statistics
const stats = await db.stats();
console.log(\`Nodes: \${stats.totalNodes}, Edges: \${stats.totalEdges}\`);
```

## Cypher support

`query()` and `querySync()` run a single-pattern matcher, not a full planner.
Anything it cannot execute returns an **error naming what was refused** — it
never returns an empty result set to mean "unsupported".

Supported:

```javascript
await db.query('MATCH (n) RETURN n');                        // full scan
await db.query('MATCH (n:Person) RETURN n');                 // label index
await db.query("MATCH (n) WHERE n.id = 'alice' RETURN n");   // point lookup
await db.query('MATCH (n:Person) WHERE n.age > 30 RETURN n');
await db.query('MATCH (n) WHERE n.a > 1 AND n.b < 5 RETURN n');
await db.query("MATCH (n {name: 'alice'}) RETURN n");        // inline props
await db.query('MATCH (a)-[r:knows]->(b) RETURN a, r, b');   // typed edges
await db.query('MATCH (a)-[r]->(b) RETURN r');               // any edge
```

`WHERE` handles `=`, `<>`, `<`, `<=`, `>`, `>=`, `AND`, `OR`, arithmetic, and
property access. `n.id` resolves to the node's identity when no stored property
shadows it. A missing property compares false rather than matching or throwing.

Not supported — these **raise an error**, they do not silently return `[]`:

| Construct | Use instead |
|---|---|
| `CREATE` / `SET` / `DELETE` via `query()` | `createNode()`, `createEdge()`, `deleteNode()` |
| variable-length paths, `[*1..3]` | `kHopNeighbors()` |
| chained patterns, `(a)-[]->(b)<-[]-(c)` | separate queries |
| hyperedge patterns in `MATCH` | `searchHyperedges()` |

Not supported at the parser level, so these fail to parse: `CONTAINS`,
`STARTS WITH`, `ENDS WITH`, `IN`, `IS NULL`, `=~`. Aggregations (`count()`,
`collect()`), `ORDER BY`, `SKIP` and `LIMIT` are parsed but not applied.

Known defect: `NOT` binds tighter than comparison, so `NOT n.age = 30` parses
as `(NOT n.age) = 30` and matches nothing. Use `<>`. Tracked in
[#939](https://github.com/ruvnet/RuVector/issues/939).

Internal properties (`__embedding`, `__confidence`) are not returned in result
rows.

## Benchmarks

| Operation | Throughput | Latency |
|-----------|------------|---------|
| Node Creation | 9.17K ops/sec | 109ms |
| Batch Node Creation | 131.10K ops/sec | 7.63ms |
| Edge Creation | 9.30K ops/sec | 107ms |
| Vector Search (k=10) | 2.35K ops/sec | 42ms |
| k-hop Traversal | 10.28K ops/sec | 9.73ms |

## Platform Support

| Platform | Architecture | Status |
|----------|--------------|--------|
| Linux | x64 (glibc) | Supported |
| Linux | arm64 (glibc) | Supported |
| macOS | x64 | Supported |
| macOS | arm64 (M1/M2) | Supported |
| Windows | x64 | Supported |

## License

MIT
