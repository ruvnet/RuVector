# @ruvector/acorn-wasm

**ACORN predicate-agnostic filtered HNSW in WebAssembly.** High-recall vector search with arbitrary metadata filters, in the browser or at the edge.

[![npm](https://img.shields.io/npm/v/@ruvector/acorn-wasm.svg)](https://www.npmjs.com/package/@ruvector/acorn-wasm)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](https://github.com/ruvnet/RuVector#license)

## What is ACORN?

ACORN ([Patel et al., SIGMOD 2024, arXiv:2403.04871](https://arxiv.org/abs/2403.04871)) solves filtered HNSW's **recall-collapse problem**. Standard post-filter HNSW retrieves k candidates and discards the ones that fail your predicate — but at low selectivity (e.g. 1 % of vectors match) you'd need to retrieve thousands of candidates to expect 10 valid hits, and recall drops to near-zero. ACORN fixes this structurally with two changes:

1. **γ-augmented graph construction** — `γ × M` edges per node instead of `M`. The denser graph stays navigable even when the predicate prunes most nodes.
2. **Predicate-agnostic traversal** — expand all neighbors regardless of predicate. A failing node doesn't enter the result set, but its neighbors enter the candidate frontier. The beam never starves.

Net effect: **96 % recall@10 at 1 % selectivity** where post-filter HNSW collapses to near-zero.

## Install

```bash
npm install @ruvector/acorn-wasm
```

## Usage (browser)

```js
import init, { AcornIndex } from "@ruvector/acorn-wasm";

await init();

const dim = 128;
const n = 5_000;
const vectors = new Float32Array(n * dim);
// ... populate `vectors` with embeddings (n × dim, row-major) ...

// gamma=2 → ACORN-γ (best recall at low selectivity)
// gamma=1 → ACORN-1 (smaller index, fine for moderate selectivity)
const idx = AcornIndex.build(vectors, dim, 2);

const query = new Float32Array(dim);
// ... fill query ...

// Predicate is any JS function (id: number) => boolean
const inStock = (id) => products[id].stockCount > 0;
const results = idx.search(query, 10, inStock);
// → [{ id, distance }, ...]
```

## Usage (Node.js / Bun)

```js
import { AcornIndex } from "@ruvector/acorn-wasm/node/ruvector_acorn_wasm.js";
// no `init()` for the node target

const idx = AcornIndex.build(vectors, 128, 2);
const results = idx.search(query, 10, (id) => metadata[id].published);
```

## Usage (bundlers — Vite, Webpack, Rollup)

```js
import { AcornIndex } from "@ruvector/acorn-wasm/bundler/ruvector_acorn_wasm.js";
// the bundler handles the .wasm import transparently
```

## API

### `class AcornIndex`

#### `AcornIndex.build(vectors, dim, gamma)`

Build an index from a flat `Float32Array` of length `n * dim`.

| Parameter | Type | Description |
|---|---|---|
| `vectors` | `Float32Array` | Row-major matrix of `n` vectors, each of length `dim`. |
| `dim` | `number` | Vector dimensionality. |
| `gamma` | `number` | Edge multiplier. `1` → ACORN-1 (M=16). `2` → ACORN-γ (M·γ=32, recommended for low selectivity). |

Throws if `dim == 0`, `vectors` is empty, `vectors.length` is not a multiple of `dim`, or `gamma == 0`.

#### `idx.search(query, k, predicate)`

Find the `k` nearest neighbors of `query` whose `id` satisfies `predicate`. Returns an array of `SearchResult` ordered ascending by distance.

`predicate` is invoked as `predicate(id: number) => boolean` for each node visited during search (≤ ef nodes, ~150 default — bounded). Use it for any metadata filter: equality, range, geo, ACL, composite — there is no schema coupling.

#### `idx.dim` (getter, number)

Vector dimensionality of the index.

#### `idx.memoryBytes` (getter, number)

Approximate heap size — graph edges + raw vectors, in bytes.

#### `idx.name` (getter, string)

Variant label for diagnostics: `"ACORN-1 (γ=1, M=16)"` or `"ACORN-γ (γ=2, M=32)"`.

### `interface SearchResult`

```ts
{
  id: number;       // caller-supplied vector id
  distance: number; // approximate L2² distance
}
```

### `version()`

Returns the crate version baked at build time.

## Recall and performance

Native Rust benchmark (x86_64, n=5K, D=128, k=10):

| Selectivity | ACORN-γ recall@10 | ACORN-γ QPS | Flat scan recall | Flat scan QPS |
|---|---|---|---|---|
| 50 % | 34.5 % | 65 K | 100.0 % | 18 K |
| 10 % | 79.7 % | 47 K | 100.0 % | 60 K |
| **1 %** | **96.0 %** | 18 K | 100.0 % | 151 K |

The structural win is at **low selectivity**: ACORN-γ holds high recall as the predicate gets more selective, while post-filter approaches collapse. WASM throughput is typically 30–60 % of native at the same dataset size.

## Why use this in the browser

- **Filtered RAG without a server.** Query an embedding store with arbitrary metadata filters entirely client-side.
- **Privacy.** User vectors never leave the device.
- **Edge runtimes.** Cloudflare Workers, Deno Deploy, Vercel Edge — same `.wasm`, no native binaries.
- **Predicate is just JS.** Any `(id: number) => boolean` function works — your filter logic stays in JS where you already have it.

## Sister packages

- [`@ruvector/rabitq-wasm`](https://www.npmjs.com/package/@ruvector/rabitq-wasm) — 1-bit quantized vector index (when you need 32× memory reduction more than predicate filtering).
- [`@ruvector/graph-wasm`](https://www.npmjs.com/package/@ruvector/graph-wasm) — Cypher-compatible hypergraph database in WASM.
- [`ruvector`](https://www.npmjs.com/package/ruvector), [`@ruvector/core`](https://www.npmjs.com/package/@ruvector/core) — Node.js NAPI bindings for the full ruvector engine.

## Source

- **Rust crate**: [`crates/ruvector-acorn-wasm/`](https://github.com/ruvnet/RuVector/tree/main/crates/ruvector-acorn-wasm)
- **Algorithm crate**: [`crates/ruvector-acorn/`](https://github.com/ruvnet/RuVector/tree/main/crates/ruvector-acorn)
- **ADR**: [ADR-160 — ACORN predicate-agnostic filtered HNSW](https://github.com/ruvnet/RuVector/blob/main/docs/adr/ADR-160-acorn-filtered-hnsw.md)
- **Packaging ADR**: [ADR-162 — `ruvector-acorn-wasm` npm package](https://github.com/ruvnet/RuVector/blob/main/docs/adr/ADR-162-acorn-wasm-npm-package.md)
- **Paper**: [arXiv:2403.04871](https://arxiv.org/abs/2403.04871)
- **Repository**: [github.com/ruvnet/RuVector](https://github.com/ruvnet/RuVector)

## License

MIT OR Apache-2.0
