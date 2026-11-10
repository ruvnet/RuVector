# Vector search + graph expansion (npm)

[All examples](../README.md)

Combines two published packages in one small Node.js script:

- [`@ruvector/core`](../../npm/packages/core/README.md) — native HNSW vector index. Finds the notes closest to a query vector.
- [`@ruvector/graph-wasm`](../../npm/packages/graph-wasm/README.md) — property graph with Cypher, in WebAssembly. Expands each hit to the notes it links to.

Runs offline, no API keys. Embeddings come from a toy hashed bag-of-words function, so matches are lexical rather than semantic; replace `embed()` with a real embedding model for real data.

## Run

```bash
cd examples/npm-vector-graph
npm install
npm start
```

Expected output:

```text
query: "retrieve relevant documents for an LLM answer"

rag  (cosine distance 0.641)
  -[USES]-> hnsw
  -[USES]-> cypher
cypher  (cosine distance 0.673)
  <-[USES]- rag

vectors: 5, graph: 5 nodes, 4 edges, 1 hyperedge
```

Requires Node.js 18+ on a platform with an `@ruvector/core` native binary (Linux x64/ARM64 glibc, macOS x64/ARM64, Windows x64).

## Gotchas this example handles

- `@ruvector/core` only accepts `Float32Array` vectors, its `distanceMetric` values are PascalCase (`'Cosine'`, `'Euclidean'`, `'DotProduct'`, `'Manhattan'`), and it writes `./ruvector.db` when `storagePath` is omitted — the script uses a temp directory and deletes it afterwards.
- With the `Cosine` metric, `search()` returns a distance in `score` (lower is closer).
- `@ruvector/graph-wasm` is a web-target build; in Node.js initialise it with `initSync({ module: readFileSync(...) })`.
- A node can join a hyperedge only if it has an `embedding` property.
