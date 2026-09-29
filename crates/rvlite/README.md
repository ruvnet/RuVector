# RvLite - Standalone Vector Database

**Status**: Proof of Concept (v0.1.0)

RvLite is a lightweight, standalone vector database that runs entirely in WebAssembly. It provides SQL, SPARQL, and Cypher query interfaces, along with graph neural networks and self-learning capabilities.

## 🎯 Vision

A complete vector database that runs anywhere JavaScript runs:
- ✅ Browsers (Chrome, Firefox, Safari, Edge)
- ✅ Node.js
- ✅ Deno
- ✅ Bun
- ✅ Cloudflare Workers
- ✅ Vercel Edge Functions

## 🏗️ Architecture

RvLite is a **thin orchestration layer** over battle-tested WASM crates:

```
┌─────────────────────────────────────────┐
│  RvLite (Orchestration)                 │
│  ├─ SQL executor                        │
│  ├─ SPARQL executor                     │
│  ├─ Storage adapter                     │
│  └─ Unified WASM API                    │
└──────────────┬──────────────────────────┘
               │ depends on (100% reuse)
               ▼
┌──────────────────────────────────────────┐
│  Existing WASM Crates                    │
├──────────────────────────────────────────┤
│  • ruvector-core (vectors, SIMD)         │
│  • ruvector-wasm (storage, indexing)     │
│  • ruvector-graph-wasm (Cypher)          │
│  • ruvector-gnn-wasm (GNN layers)        │
│  • sona (ReasoningBank learning)         │
│  • micro-hnsw-wasm (ultra-fast HNSW)     │
└──────────────────────────────────────────┘
```

## 🚀 Quick Start (Future)

```typescript
import { RvLite } from '@rvlite/wasm';

// Create database
const db = await RvLite.create();

// SQL with vector search
await db.sql(`
  CREATE TABLE docs (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(384)
  )
`);

await db.sql(`
  SELECT id, content, embedding <=> $1 AS distance
  FROM docs
  ORDER BY distance
  LIMIT 10
`, [queryVector]);

// Cypher graph queries
await db.cypher(`
  CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})
`);

// SPARQL RDF queries
await db.sparql(`
  SELECT ?name WHERE {
    ?person foaf:name ?name .
  }
`);

// GNN embeddings
const embeddings = await db.gnn.computeEmbeddings('social_network', [
  db.gnn.createLayer('gcn', { inputDim: 128, outputDim: 64 })
]);

// Self-learning with ReasoningBank
await db.learning.recordTrajectory({ state: [0.1], action: 2, reward: 1.0 });
await db.learning.train({ algorithm: 'q-learning', iterations: 1000 });
```

## Cargo features

| Feature | Default | What it adds |
|---------|---------|--------------|
| `browser` | yes | wasm-bindgen exports (`RvLite`, `RvLiteConfig`, `CypherEngine`), IndexedDB persistence, `JsValue` glue for the npm package |
| `rvf-backend` | no | RVF file persistence |

With `default-features = false`, rvlite is a plain Rust library: the `cypher`,
`sql` and `sparql` engines plus the serde state types in `storage`. rvlite itself
then depends on no `wasm-bindgen`, `js-sys` or `web-sys`. On `wasm32` targets,
transitive dependencies (`getrandom`'s `js` backend, and `chrono`/`uuid` via
`ruvector-core`) still link `wasm-bindgen`/`js-sys`; `web-sys` is never pulled
in. Use this to embed the engines server-side, e.g. in a Cloudflare Worker via
workers-rs:

```toml
rvlite = { version = "0.4", default-features = false }
```

**Upgrading from 0.3:** before 0.4.0 the browser bindings were always compiled
and `default-features = false` had no effect. From 0.4.0 it removes them. If you
set `default-features = false` and still use `RvLite`, `IndexedDBStorage` or
`CypherEngine::execute`, add `features = ["browser"]`.

## 📦 Current Status (v0.1.0 - POC)

This is a **proof of concept** to validate:
- ✅ Basic WASM compilation with ruvector-core
- ✅ WASM bindings setup (wasm-bindgen)
- ⏳ Integration with other WASM crates (pending)
- ⏳ Bundle size measurement (pending)
- ⏳ Performance benchmarks (pending)

## 🛠️ Development

### Build

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for web
cd crates/rvlite
wasm-pack build --target web --release

# Build for Node.js
wasm-pack build --target nodejs --release
```

### Test

```bash
# Run Rust unit tests
cargo test

# Run WASM tests (requires Chrome/Firefox)
wasm-pack test --headless --chrome
wasm-pack test --headless --firefox
```

### Size Analysis

```bash
# Build optimized
wasm-pack build --release

# Check size
ls -lh pkg/*.wasm
du -sh pkg/
```

## 📖 Documentation

See `/crates/rvlite/docs/` for comprehensive documentation:
- `00_EXISTING_WASM_ANALYSIS.md` - Analysis of existing WASM infrastructure
- `01_SPECIFICATION.md` - Complete requirements specification
- `02_API_SPECIFICATION.md` - TypeScript API design
- `03_IMPLEMENTATION_ROADMAP.md` - Original 5-week timeline
- `04_REVISED_ARCHITECTURE_MAX_REUSE.md` - Optimized 2-3 week plan
- `05_ARCHITECTURE_REVIEW_AND_VALIDATION.md` - Architecture validation
- `SPARC_OVERVIEW.md` - SPARC methodology overview

## 🎯 Roadmap

### Phase 1: Proof of Concept (Current)
- [x] Create rvlite crate structure
- [x] Set up WASM bindings
- [x] Basic compilation test
- [ ] Measure bundle size
- [ ] Integration with ruvector-wasm
- [ ] Integration with ruvector-graph-wasm

### Phase 2: Core Integration (Week 1)
- [ ] Storage adapter implementation
- [ ] SPARQL extraction from ruvector-postgres
- [ ] SQL parser integration (sqlparser-rs)
- [ ] Basic query routing

### Phase 3: Full Features (Week 2)
- [ ] GNN layer integration
- [ ] ReasoningBank integration
- [ ] Hyperbolic embeddings
- [ ] Comprehensive testing

### Phase 4: Production Release (Week 3)
- [ ] Documentation
- [ ] Examples (browser, Node.js, Deno)
- [ ] Performance benchmarks
- [ ] NPM package publication

## 📊 Size Budget

**Target**: < 3MB gzipped

**Expected breakdown**:
- ruvector-core: ~500KB
- SQL parser: ~200KB
- SPARQL executor: ~300KB
- Cypher (ruvector-graph-wasm): ~600KB
- GNN layers: ~300KB
- ReasoningBank (sona): ~300KB
- Orchestration: ~100KB

**Total estimated**: ~2.3MB gzipped ✅

## 🤝 Contributing

This project reuses existing battle-tested WASM crates. Contributions should focus on:
1. Integration and orchestration
2. SQL/SPARQL/Cypher query routing
3. Storage adapter implementation
4. Testing and benchmarks
5. Documentation and examples

## 📄 License

MIT OR Apache-2.0

## 🙏 Acknowledgments

RvLite is built on the shoulders of:
- `ruvector-core` - Vector operations and SIMD
- `ruvector-wasm` - WASM vector database
- `ruvector-graph` - Cypher and graph database
- `ruvector-gnn` - Graph neural networks
- `sona` - Self-learning and ReasoningBank
- `micro-hnsw-wasm` - Ultra-lightweight HNSW

---

**Status**: Proof of Concept - Architecture Validated ✅
**Next Step**: Build and measure bundle size
