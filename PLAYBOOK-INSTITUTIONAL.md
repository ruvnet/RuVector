# RuVector Institutional Knowledge Playbook

**Last Updated**: 2026-01-14
**Version**: 0.1.31

---

## What is RuVector?

RuVector is a **distributed vector database that learns**. It combines:

- **Vector search** with HNSW indexing (<0.5ms latency)
- **Graph queries** via Neo4j-compatible Cypher
- **Self-learning** through Graph Neural Networks (GNN)
- **Horizontal scaling** with Raft consensus
- **AI routing** via Tiny Dancer (FastGRNN neural inference)

Think of it as: **Pinecone + Neo4j + PyTorch + Postgres + etcd** in one Rust package.

### Why Use RuVector?

| Problem | RuVector Solution |
|---------|-------------------|
| Vector DBs don't get smarter | GNN layers improve search over time |
| No horizontal scaling | Raft consensus + auto-sharding |
| Separate graph DB needed | Native Cypher queries |
| High inference costs | Tiny Dancer routes to optimal LLM |
| Memory bloat | 2-32x automatic compression |
| Python too slow | 10-100x faster native Rust |

---

## Architecture Overview

### Core Components

```
ruvector/
├── crates/                    # 54 Rust crates
│   ├── ruvector-core/         # Vector DB engine (HNSW, storage)
│   ├── ruvector-graph/        # Graph DB + Cypher parser
│   ├── ruvector-gnn/          # GNN layers, compression, training
│   ├── ruvector-raft/         # Raft consensus
│   ├── ruvector-cluster/      # Cluster management
│   ├── ruvector-attention/    # 39 attention mechanisms
│   ├── ruvector-tiny-dancer-core/  # AI agent routing
│   ├── ruvector-postgres/     # PostgreSQL extension
│   ├── ruvector-*-wasm/       # WebAssembly bindings
│   └── ruvector-*-node/       # Node.js bindings (napi-rs)
├── npm/packages/              # 35+ npm packages
├── examples/                  # 18+ production examples
└── docs/                      # Comprehensive documentation
```

### Crate Dependency Map

```
                    ┌─────────────────┐
                    │  ruvector-core  │  <- Foundation: HNSW, storage, SIMD
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌────────────────┐
│ ruvector-graph│   │  ruvector-gnn │   │ruvector-router │
│   (Cypher)    │   │(Neural layers)│   │   (Semantic)   │
└───────────────┘   └───────────────┘   └────────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  ruvector-dag   │  <- Query optimization
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌────────────────┐
│   *-node      │   │    *-wasm     │   │   *-postgres   │
│  (Node.js)    │   │   (Browser)   │   │   (Extension)  │
└───────────────┘   └───────────────┘   └────────────────┘

Distributed:
┌───────────────┐   ┌────────────────┐   ┌─────────────────┐
│ ruvector-raft │ → │ruvector-cluster│ → │ruvector-replicat│
│  (Consensus)  │   │  (Sharding)    │   │  (Multi-master) │
└───────────────┘   └────────────────┘   └─────────────────┘
```

### Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Storage | redb + memmap2 | Memory-mapped embedded DB |
| Indexing | HNSW (hnsw_rs patched) | Sub-millisecond vector search |
| Distance | SimSIMD + custom | AVX-512/AVX2/NEON acceleration |
| Serialization | rkyv + bincode | Zero-copy, fast loading |
| Node.js | napi-rs | Native bindings |
| WASM | wasm-bindgen | Browser support |
| Consensus | Custom Raft | Distributed coordination |

---

## Development Setup

### Prerequisites

```bash
# Rust (1.77+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
rustup target add wasm32-unknown-unknown

# Node.js (18+)
brew install node  # macOS
# or: nvm install 18

# WASM tools
cargo install wasm-pack

# For PostgreSQL extension
cargo install cargo-pgrx --version "0.12.9" --locked
cargo pgrx init  # Initialize pgrx (downloads PG headers)
```

### Clone and Build

```bash
git clone https://github.com/ruvnet/ruvector
cd ruvector

# Build all crates
cargo build --release

# Build with native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Build specific crate
cargo build -p ruvector-core --release
```

### Build Common Targets

```bash
# Node.js native module
cd crates/ruvector-node
npm run build

# WASM module
cd crates/ruvector-wasm
wasm-pack build --target web --release

# CLI
cargo install --path crates/ruvector-cli

# PostgreSQL extension
cd crates/ruvector-postgres
cargo pgrx install --release
```

---

## Testing and Debugging

### Running Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p ruvector-core

# With output
cargo test -p ruvector-gnn -- --nocapture

# Integration tests
cargo test --test '*'
```

### Benchmarks

```bash
# All benchmarks
cargo bench --workspace

# Specific benchmark
cargo bench --bench comprehensive_bench

# Compare before/after
cargo bench -- --save-baseline before
# ... make changes ...
cargo bench -- --baseline before
```

### Debugging Tips

```bash
# Enable debug logging
RUST_LOG=debug cargo run

# More granular logging
RUST_LOG=ruvector_core=trace,ruvector_gnn=debug cargo run

# Profile with flamegraph
cargo flamegraph --bin ruvector-cli -- search ...

# Memory profiling
cargo valgrind --bin ruvector-bench
```

### WASM Debugging

```bash
# Build with debug info
wasm-pack build --target web --dev

# Enable console_error_panic_hook
# (already in lib.rs with feature)

# Browser console shows panic backtraces
```

---

## Common Issues and Solutions

### Build Issue: ndarray Workspace Version

**Problem**: Workspace members have conflicting ndarray versions.

**Symptom**:
```
error: failed to resolve: ndarray version conflicts in workspace
```

**Solution**: Pin ndarray to 0.16 in workspace Cargo.toml:
```toml
[workspace.dependencies]
ndarray = "0.16"
```

**Applied Fix**: The workspace Cargo.toml already includes this.

---

### Build Issue: sparse-inference-wasm API Mismatch

**Problem**: WASM bindings import types that don't exist or have changed.

**Symptom**:
```
error[E0433]: failed to resolve: use of undeclared type or module `GenerationConfig`
```

**Solution**: Verify imports match the actual ruvector-sparse-inference API:
```rust
// Correct imports (check current API)
use ruvector_sparse_inference::{
    InferenceConfig,
    integration::ruvllm::{GenerationConfig, KVCache, SparseInferenceBackend},
};
```

**Verification**:
```bash
# Check what's exported
cargo doc -p ruvector-sparse-inference --open
```

---

### Build Issue: pgrx Setup for PostgreSQL Extension

**Problem**: pgrx not initialized or wrong PostgreSQL version.

**Symptom**:
```
error: could not find `pgrx` headers for pg17
```

**Solution**:
```bash
# Install cargo-pgrx with correct version
cargo install cargo-pgrx --version "0.12.9" --locked

# Initialize pgrx (downloads PostgreSQL headers)
cargo pgrx init

# Build for specific PG version
cargo build -p ruvector-postgres --features pg17

# Or for older versions
cargo build -p ruvector-postgres --features pg16
```

**Note**: The extension defaults to pg17. Use features to target other versions:
- `pg14`, `pg15`, `pg16`, `pg17`

---

### Build Issue: hnsw_rs rand Version Conflict

**Problem**: hnsw_rs uses rand 0.9 but WASM needs rand 0.8 for getrandom compatibility.

**Symptom**:
```
error: getrandom version conflict (0.2 vs 0.3)
```

**Solution**: The repo includes a patched hnsw_rs:
```toml
# In workspace Cargo.toml
[patch.crates-io]
hnsw_rs = { path = "./patches/hnsw_rs" }
```

The patch in `patches/hnsw_rs/Cargo.toml` pins rand to 0.8.

---

### Runtime Issue: WASM Memory Limits

**Problem**: Out of memory in browser with large vector sets.

**Solution**:
```javascript
// Use Web Workers for large datasets
const worker = new Worker('./ruvector-worker.js');

// Or limit vector count
const db = new VectorDB({
  maxElements: 100000,  // Limit for browser
  mmap: false  // Disable mmap in WASM
});
```

---

### Runtime Issue: Slow First Query

**Problem**: First query takes 2-3 seconds, subsequent queries fast.

**Cause**: HNSW index lazy loading from disk.

**Solution**:
```rust
// Pre-warm the index
db.warmup()?;

// Or use mmap for instant access
let options = DbOptions {
    mmap_vectors: true,
    ..Default::default()
};
```

---

### Runtime Issue: Low Recall

**Problem**: Search returns wrong results.

**Solution**: Tune HNSW parameters:
```rust
let options = DbOptions {
    hnsw_m: 32,              // Increase connections (16-64)
    hnsw_ef_construction: 200,  // Increase build quality (100-400)
    hnsw_ef_search: 100,     // Increase search quality (50-500)
    ..Default::default()
};
```

---

## Performance Reference

### Benchmarks (Apple M2 / Intel i7)

| Operation | Dimensions | Time | Throughput |
|-----------|------------|------|------------|
| HNSW Search (k=10) | 384 | 61us | 16,400 QPS |
| HNSW Search (k=100) | 384 | 164us | 6,100 QPS |
| Cosine Distance | 1536 | 143ns | 7M ops/sec |
| Dot Product | 384 | 33ns | 30M ops/sec |
| Batch Distance (1000) | 384 | 237us | 4.2M/sec |

### Compression Ratios

| Format | Compression | Recall | Use Case |
|--------|-------------|--------|----------|
| f32 | 1x | 100% | Hot data |
| f16 | 2x | 99%+ | Warm data |
| PQ8 | 8x | 95%+ | Cool data |
| PQ4 | 16x | 90%+ | Cold data |
| Binary | 32x | 80%+ | Archive |

### Memory Usage (1M vectors)

| Method | Memory |
|--------|--------|
| Uncompressed | 2GB |
| Scalar quant | 500MB |
| PQ8 | 200MB |
| Binary | 60MB |

---

## Repository Structure

```
ruvector/
├── Cargo.toml              # Workspace configuration
├── Cargo.lock              # Locked dependencies
├── CLAUDE.md               # Claude Code configuration
├── README.md               # Main documentation
├── CHANGELOG.md            # Version history
├── install.sh              # One-line installer
├── workers.yaml            # CI/CD configuration
├── patches/                # Dependency patches
│   └── hnsw_rs/            # Patched hnsw for rand 0.8
├── crates/                 # 54 Rust crates
├── npm/                    # 35+ npm packages
│   └── packages/           # Individual npm packages
├── examples/               # 18+ examples
├── docs/                   # Documentation
│   ├── guides/             # Getting started, tutorials
│   ├── api/                # API references
│   ├── optimization/       # Performance tuning
│   ├── postgres/           # PostgreSQL extension docs
│   └── hooks/              # Claude Code hooks
├── tests/                  # Integration tests
├── benches/                # Benchmarks
├── benchmarks/             # Benchmark data & results
├── scripts/                # Build & utility scripts
└── plans/                  # Development plans
```

---

## Key Configuration Files

### Workspace Cargo.toml

```toml
[workspace]
members = [
    "crates/ruvector-core",
    "crates/ruvector-node",
    "crates/ruvector-wasm",
    # ... 54 total crates
]

[workspace.package]
version = "0.1.31"
edition = "2021"
rust-version = "1.77"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true

[patch.crates-io]
hnsw_rs = { path = "./patches/hnsw_rs" }
```

### Environment Variables

```bash
# Runtime
RUST_LOG=info                    # Logging level
RAYON_NUM_THREADS=16             # Thread pool size
RUVECTOR_POSTGRES_URL=...        # PostgreSQL connection

# Build
RUSTFLAGS="-C target-cpu=native" # Enable native SIMD
CARGO_BUILD_JOBS=16              # Parallel compilation
```

---

## Support Resources

- **GitHub**: https://github.com/ruvnet/ruvector
- **Issues**: https://github.com/ruvnet/ruvector/issues
- **Docs**: https://docs.rs/ruvector-core
- **npm**: https://npmjs.com/package/ruvector

---

*Built by [rUv](https://ruv.io) - Vector search that gets smarter over time.*
