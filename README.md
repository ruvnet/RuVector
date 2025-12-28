# RuVector

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Crates.io](https://img.shields.io/crates/v/ruvector-core.svg)](https://crates.io/crates/ruvector-core)
[![postgres](https://img.shields.io/crates/v/ruvector-postgres.svg?label=postgres)](https://crates.io/crates/ruvector-postgres)
[![SONA](https://img.shields.io/crates/v/ruvector-sona.svg?label=sona)](https://crates.io/crates/ruvector-sona)
[![npm](https://img.shields.io/npm/v/ruvector.svg)](https://www.npmjs.com/package/ruvector)
[![@ruvector/sona](https://img.shields.io/npm/v/@ruvector/sona.svg?label=%40ruvector%2Fsona)](https://www.npmjs.com/package/@ruvector/sona)
[![Rust](https://img.shields.io/badge/rust-1.77%2B-orange.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/github/actions/workflow/status/ruvnet/ruvector/ci.yml?branch=main)](https://github.com/ruvnet/ruvector/actions)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](./docs/)

**A distributed vector database that learns.** Store embeddings, query with Cypher, scale horizontally with Raft consensus, and let the index improve itself through Graph Neural Networks.

```bash
npx ruvector
```

> **All-in-One Package**: The core `ruvector` package includes everything — vector search, graph queries, GNN layers, distributed clustering, AI routing, and WASM support. No additional packages needed.

## What Problem Does RuVector Solve?

Traditional vector databases just store and search. When you ask "find similar items," they return results but never get smarter. They don't scale horizontally. They can't route AI requests intelligently.

**RuVector is different:**

1. **Store vectors** like any vector DB (embeddings from OpenAI, Cohere, etc.)
2. **Query with Cypher** like Neo4j (`MATCH (a)-[:SIMILAR]->(b) RETURN b`)
3. **The index learns** — GNN layers make search results improve over time
4. **Scale horizontally** — Raft consensus, multi-master replication, auto-sharding
5. **Route AI requests** — Semantic routing and FastGRNN neural inference for LLM optimization
6. **Compress automatically** — 2-32x memory reduction with adaptive tiered compression
7. **39 attention mechanisms** — Flash, linear, graph, hyperbolic for custom models
8. **Drop into Postgres** — pgvector-compatible extension with SIMD acceleration
9. **Run anywhere** — Node.js, browser (WASM), HTTP server, or native Rust
10. **Continuous learning** — SONA enables runtime adaptation with LoRA, EWC++, and ReasoningBank

Think of it as: **Pinecone + Neo4j + PyTorch + postgres + etcd** in one Rust package.



## How the GNN Works

Traditional vector search:
```
Query → HNSW Index → Top K Results
```

RuVector with GNN:
```
Query → HNSW Index → GNN Layer → Enhanced Results
                ↑                      │
                └──── learns from ─────┘
```

The GNN layer:
1. Takes your query and its nearest neighborsa
2. Applies multi-head attention to weigh which neighbors matter
3. Updates representations based on graph structure
4. Returns better-ranked results

Over time, frequently-accessed paths get reinforced, making common queries faster and more accurate.


## Quick Start

### One-Line Install

```bash
# Vector database
npm install ruvector
npx ruvector

# Self-learning hooks for Claude Code
npx @ruvector/cli hooks init
npx @ruvector/cli hooks install
```

### Node.js / Browser

```bash
# Install
npm install ruvector

# Or try instantly
npx ruvector
```


## Comparison

| Feature | RuVector | Pinecone | Qdrant | Milvus | ChromaDB |
|---------|----------|----------|--------|--------|----------|
| **Latency (p50)** | **61µs** | ~2ms | ~1ms | ~5ms | ~50ms |
| **Memory (1M vec)** | 200MB* | 2GB | 1.5GB | 1GB | 3GB |
| **Graph Queries** | ✅ Cypher | ❌ | ❌ | ❌ | ❌ |
| **SPARQL/RDF** | ✅ W3C 1.1 | ❌ | ❌ | ❌ | ❌ |
| **Hyperedges** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Dynamic Min-Cut** | ✅ n^0.12 | ❌ | ❌ | ❌ | ❌ |
| **Self-Learning (GNN)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Runtime Adaptation (SONA)** | ✅ LoRA+EWC++ | ❌ | ❌ | ❌ | ❌ |
| **AI Agent Routing** | ✅ Tiny Dancer | ❌ | ❌ | ❌ | ❌ |
| **Attention Mechanisms** | ✅ 39 types | ❌ | ❌ | ❌ | ❌ |
| **Hyperbolic Embeddings** | ✅ Poincaré+Lorentz | ❌ | ❌ | ❌ | ❌ |
| **Local Embeddings** | ✅ 6 models | ❌ | ❌ | ❌ | ❌ |
| **PostgreSQL Extension** | ✅ 77+ functions | ❌ | ❌ | ❌ | ❌ |
| **SIMD Optimization** | ✅ AVX-512/NEON | Partial | ✅ | ✅ | ❌ |
| **Metadata Filtering** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Sparse Vectors** | ✅ BM25/TF-IDF | ✅ | ✅ | ✅ | ❌ |
| **Raft Consensus** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Multi-Master Replication** | ✅ | ❌ | ❌ | ✅ | ❌ |
| **Auto-Sharding** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Auto-Compression** | ✅ 2-32x | ❌ | ❌ | ✅ | ❌ |
| **Snapshots/Backups** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Browser/WASM** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Standalone Edge DB** | ✅ rvLite | ❌ | ❌ | ❌ | ❌ |
| **Differentiable** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Multi-Tenancy** | ✅ Collections | ✅ | ✅ | ✅ | ✅ |
| **Open Source** | ✅ MIT | ❌ | ✅ | ✅ | ✅ |

*With PQ8 compression. Benchmarks on Apple M2 / Intel i7.



## Features

### Core Capabilities

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Vector Search** | HNSW index, <0.5ms latency, SIMD acceleration | Fast enough for real-time apps |
| **Cypher Queries** | `MATCH`, `WHERE`, `CREATE`, `RETURN` | Familiar Neo4j syntax |
| **GNN Layers** | Neural network on index topology | Search improves with usage |
| **Hyperedges** | Connect 3+ nodes at once | Model complex relationships |
| **Metadata Filtering** | Filter vectors by properties | Combine semantic + structured search |
| **Collections** | Namespace isolation, multi-tenancy | Organize vectors by project/user |

### Distributed Systems

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Raft Consensus** | Leader election, log replication | Strong consistency for metadata |
| **Auto-Sharding** | Consistent hashing, shard migration | Scale to billions of vectors |
| **Multi-Master Replication** | Write to any node, conflict resolution | High availability, no SPOF |
| **Snapshots** | Point-in-time backups, incremental | Disaster recovery |
| **Cluster Metrics** | Prometheus-compatible monitoring | Observability at scale |

```bash
cargo add ruvector-raft ruvector-cluster ruvector-replication
```

### AI & ML

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Tensor Compression** | f32→f16→PQ8→PQ4→Binary | 2-32x memory reduction |
| **Differentiable Search** | Soft attention k-NN | End-to-end trainable |
| **Semantic Router** | Route queries to optimal endpoints | Multi-model AI orchestration |
| **Tiny Dancer** | FastGRNN neural inference | Optimize LLM inference costs |
| **Adaptive Routing** | Learn optimal routing strategies | Minimize latency, maximize accuracy |
| **SONA** | Two-tier LoRA + EWC++ + ReasoningBank | Runtime learning without retraining |

### Attention Mechanisms (`@ruvector/attention`)

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **39 Mechanisms** | Dot-product, multi-head, flash, linear, sparse, cross-attention | Cover all transformer and GNN use cases |
| **Graph Attention** | RoPE, edge-featured, local-global, neighborhood | Purpose-built for graph neural networks |
| **Hyperbolic Attention** | Poincaré ball operations, curved-space math | Better embeddings for hierarchical data |
| **SIMD Optimized** | Native Rust with AVX2/NEON acceleration | 2-10x faster than pure JS |
| **Streaming & Caching** | Chunk-based processing, KV-cache | Constant memory, 10x faster inference |

> **Documentation**: [Attention Module Docs](./crates/ruvector-attention/README.md)

#### Core Attention Mechanisms

Standard attention layers for sequence modeling and transformers.

| Mechanism | Complexity | Memory | Best For |
|-----------|------------|--------|----------|
| **DotProductAttention** | O(n²) | O(n²) | Basic attention for small-medium sequences |
| **MultiHeadAttention** | O(n²·h) | O(n²·h) | BERT, GPT-style transformers |
| **FlashAttention** | O(n²) | O(n) | Long sequences with limited GPU memory |
| **LinearAttention** | O(n·d) | O(n·d) | 8K+ token sequences, real-time streaming |
| **HyperbolicAttention** | O(n²) | O(n²) | Tree-like data: taxonomies, org charts |
| **MoEAttention** | O(n·k) | O(n·k) | Large models with sparse expert routing |

#### Graph Attention Mechanisms

Attention layers designed for graph-structured data and GNNs.

| Mechanism | Complexity | Best For |
|-----------|------------|----------|
| **GraphRoPeAttention** | O(n²) | Position-aware graph transformers |
| **EdgeFeaturedAttention** | O(n²·e) | Molecules, knowledge graphs with edge data |
| **DualSpaceAttention** | O(n²) | Hybrid flat + hierarchical embeddings |
| **LocalGlobalAttention** | O(n·k + n) | 100K+ node graphs, scalable GNNs |

#### Specialized Mechanisms

Task-specific attention variants for efficiency and multi-modal learning.

| Mechanism | Type | Best For |
|-----------|------|----------|
| **SparseAttention** | Efficiency | Long docs, low-memory inference |
| **CrossAttention** | Multi-modal | Image-text, encoder-decoder models |
| **NeighborhoodAttention** | Graph | Local message passing in GNNs |
| **HierarchicalAttention** | Structure | Multi-level docs (section → paragraph) |

#### Hyperbolic Math Functions

Operations for Poincaré ball embeddings—curved space that naturally represents hierarchies.

| Function | Description | Use Case |
|----------|-------------|----------|
| `expMap(v, c)` | Map to hyperbolic space | Initialize embeddings |
| `logMap(p, c)` | Map to flat space | Compute gradients |
| `mobiusAddition(x, y, c)` | Add vectors in curved space | Aggregate features |
| `poincareDistance(x, y, c)` | Measure hyperbolic distance | Compute similarity |
| `projectToPoincareBall(p, c)` | Ensure valid coordinates | Prevent numerical errors |

#### Async & Batch Operations

Utilities for high-throughput inference and training optimization.

| Operation | Description | Performance |
|-----------|-------------|-------------|
| `asyncBatchCompute()` | Process batches in parallel | 3-5x faster |
| `streamingAttention()` | Process in chunks | Fixed memory usage |
| `HardNegativeMiner` | Find hard training examples | Better contrastive learning |
| `AttentionCache` | Cache key-value pairs | 10x faster inference |

```bash
# Install attention module
npm install @ruvector/attention

# CLI commands
npx ruvector attention list                    # List all 39 mechanisms
npx ruvector attention info flash              # Details on FlashAttention
npx ruvector attention benchmark               # Performance comparison
npx ruvector attention compute -t dot -d 128   # Run attention computation
npx ruvector attention hyperbolic -a distance -v "[0.1,0.2]" -b "[0.3,0.4]"
```

### Deployment

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **HTTP/gRPC Server** | REST API, streaming support | Easy integration |
| **WASM/Browser** | Full client-side support | Run AI search offline |
| **Node.js Bindings** | Native napi-rs bindings | No serialization overhead |
| **FFI Bindings** | C-compatible interface | Use from Python, Go, etc. |
| **CLI Tools** | Benchmarking, testing, management | DevOps-friendly |

## Benchmarks

Real benchmark results on standard hardware:

| Operation | Dimensions | Time | Throughput |
|-----------|------------|------|------------|
| **HNSW Search (k=10)** | 384 | 61µs | 16,400 QPS |
| **HNSW Search (k=100)** | 384 | 164µs | 6,100 QPS |
| **Cosine Distance** | 1536 | 143ns | 7M ops/sec |
| **Dot Product** | 384 | 33ns | 30M ops/sec |
| **Batch Distance (1000)** | 384 | 237µs | 4.2M/sec |

### Global Cloud Performance (500M Streams)

Production-validated metrics at hyperscale:

| Metric | Value | Details |
|--------|-------|---------|
| **Concurrent Streams** | 500M baseline | Burst capacity to 25B (50x) |
| **Global Latency (p50)** | <10ms | Multi-region + CDN edge caching |
| **Global Latency (p99)** | <50ms | Cross-continental with failover |
| **Availability SLA** | 99.99% | 15 regions, automatic failover |
| **Cost per Stream/Month** | $0.0035 | 60% optimized ($1.74M total at 500M) |
| **Regions** | 15 global | Americas, EMEA, APAC coverage |
| **Throughput per Region** | 100K+ QPS | Adaptive batching enabled |
| **Memory Efficiency** | 2-32x compression | Tiered hot/warm/cold storage |
| **Index Build Time** | 1M vectors/min | Parallel HNSW construction |
| **Replication Lag** | <100ms | Multi-master async replication |


## Compression Tiers

**The architecture adapts to your data.** Hot paths get full precision and maximum compute. Cold paths compress automatically and throttle resources. Recent data stays crystal clear; historical data optimizes itself in the background.

Think of it like your computer's memory hierarchy—frequently accessed data lives in fast cache, while older files move to slower, denser storage. RuVector does this automatically for your vectors:

| Access Frequency | Format | Compression | What Happens |
|-----------------|--------|-------------|--------------|
| **Hot** (>80%) | f32 | 1x | Full precision, instant retrieval |
| **Warm** (40-80%) | f16 | 2x | Slight compression, imperceptible latency |
| **Cool** (10-40%) | PQ8 | 8x | Smart quantization, ~1ms overhead |
| **Cold** (1-10%) | PQ4 | 16x | Heavy compression, still fast search |
| **Archive** (<1%) | Binary | 32x | Maximum density, batch retrieval |

**No configuration needed.** RuVector tracks access patterns and automatically promotes/demotes vectors between tiers. Your hot data stays fast; your cold data shrinks.

## Use Cases

**RAG (Retrieval-Augmented Generation)**
```javascript
const context = ruvector.search(questionEmbedding, 5);
const prompt = `Context: ${context.join('\n')}\n\nQuestion: ${question}`;
```

**Recommendation Systems**
```cypher
MATCH (user:User)-[:VIEWED]->(item:Product)
MATCH (item)-[:SIMILAR_TO]->(rec:Product)
RETURN rec ORDER BY rec.score DESC LIMIT 10
```

**Knowledge Graphs**
```cypher
MATCH (concept:Concept)-[:RELATES_TO*1..3]->(related)
RETURN related
```

## Installation

| Platform | Command |
|----------|---------|
| **npm** | `npm install ruvector` |
| **npm (SONA)** | `npm install @ruvector/sona` |
| **Browser/WASM** | `npm install ruvector-wasm` |
| **Rust** | `cargo add ruvector-core ruvector-graph ruvector-gnn` |
| **Rust (SONA)** | `cargo add ruvector-sona` |

## Documentation

| Topic | Link |
|-------|------|
| Getting Started | [docs/guides/GETTING_STARTED.md](./docs/guides/GETTING_STARTED.md) |
| Cypher Reference | [docs/api/CYPHER_REFERENCE.md](./docs/api/CYPHER_REFERENCE.md) |
| GNN Architecture | [docs/gnn/gnn-layer-implementation.md](./docs/gnn/gnn-layer-implementation.md) |
| Node.js API | [crates/ruvector-gnn-node/README.md](./crates/ruvector-gnn-node/README.md) |
| WASM API | [crates/ruvector-gnn-wasm/README.md](./crates/ruvector-gnn-wasm/README.md) |
| Performance Tuning | [docs/optimization/PERFORMANCE_TUNING_GUIDE.md](./docs/optimization/PERFORMANCE_TUNING_GUIDE.md) |
| API Reference | [docs/api/](./docs/api/) |

## Crates

All crates are published to [crates.io](https://crates.io) under the `ruvector-*` namespace.

### Core Crates

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-core](./crates/ruvector-core) | Vector database engine with HNSW indexing | [![crates.io](https://img.shields.io/crates/v/ruvector-core.svg)](https://crates.io/crates/ruvector-core) |
| [ruvector-collections](./crates/ruvector-collections) | Collection and namespace management | [![crates.io](https://img.shields.io/crates/v/ruvector-collections.svg)](https://crates.io/crates/ruvector-collections) |
| [ruvector-filter](./crates/ruvector-filter) | Vector filtering and metadata queries | [![crates.io](https://img.shields.io/crates/v/ruvector-filter.svg)](https://crates.io/crates/ruvector-filter) |
| [ruvector-metrics](./crates/ruvector-metrics) | Performance metrics and monitoring | [![crates.io](https://img.shields.io/crates/v/ruvector-metrics.svg)](https://crates.io/crates/ruvector-metrics) |
| [ruvector-snapshot](./crates/ruvector-snapshot) | Snapshot and persistence management | [![crates.io](https://img.shields.io/crates/v/ruvector-snapshot.svg)](https://crates.io/crates/ruvector-snapshot) |

### Graph & GNN

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-graph](./crates/ruvector-graph) | Hypergraph database with Neo4j-style Cypher | [![crates.io](https://img.shields.io/crates/v/ruvector-graph.svg)](https://crates.io/crates/ruvector-graph) |
| [ruvector-graph-node](./crates/ruvector-graph-node) | Node.js bindings for graph operations | [![crates.io](https://img.shields.io/crates/v/ruvector-graph-node.svg)](https://crates.io/crates/ruvector-graph-node) |
| [ruvector-graph-wasm](./crates/ruvector-graph-wasm) | WASM bindings for browser graph queries | [![crates.io](https://img.shields.io/crates/v/ruvector-graph-wasm.svg)](https://crates.io/crates/ruvector-graph-wasm) |
| [ruvector-gnn](./crates/ruvector-gnn) | Graph Neural Network layers and training | [![crates.io](https://img.shields.io/crates/v/ruvector-gnn.svg)](https://crates.io/crates/ruvector-gnn) |
| [ruvector-gnn-node](./crates/ruvector-gnn-node) | Node.js bindings for GNN inference | [![crates.io](https://img.shields.io/crates/v/ruvector-gnn-node.svg)](https://crates.io/crates/ruvector-gnn-node) |
| [ruvector-gnn-wasm](./crates/ruvector-gnn-wasm) | WASM bindings for browser GNN | [![crates.io](https://img.shields.io/crates/v/ruvector-gnn-wasm.svg)](https://crates.io/crates/ruvector-gnn-wasm) |

### Attention Mechanisms

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-attention](./crates/ruvector-attention) | 39 attention mechanisms (Flash, Hyperbolic, MoE, Graph) | [![crates.io](https://img.shields.io/crates/v/ruvector-attention.svg)](https://crates.io/crates/ruvector-attention) |
| [ruvector-attention-node](./crates/ruvector-attention-node) | Node.js bindings for attention mechanisms | [![crates.io](https://img.shields.io/crates/v/ruvector-attention-node.svg)](https://crates.io/crates/ruvector-attention-node) |
| [ruvector-attention-wasm](./crates/ruvector-attention-wasm) | WASM bindings for browser attention | [![crates.io](https://img.shields.io/crates/v/ruvector-attention-wasm.svg)](https://crates.io/crates/ruvector-attention-wasm) |
| [ruvector-attention-cli](./crates/ruvector-attention-cli) | CLI for attention testing and benchmarking | [![crates.io](https://img.shields.io/crates/v/ruvector-attention-cli.svg)](https://crates.io/crates/ruvector-attention-cli) |

### Distributed Systems

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-cluster](./crates/ruvector-cluster) | Cluster management and coordination | [![crates.io](https://img.shields.io/crates/v/ruvector-cluster.svg)](https://crates.io/crates/ruvector-cluster) |
| [ruvector-raft](./crates/ruvector-raft) | Raft consensus implementation | [![crates.io](https://img.shields.io/crates/v/ruvector-raft.svg)](https://crates.io/crates/ruvector-raft) |
| [ruvector-replication](./crates/ruvector-replication) | Data replication and synchronization | [![crates.io](https://img.shields.io/crates/v/ruvector-replication.svg)](https://crates.io/crates/ruvector-replication) |

### AI Agent Routing (Tiny Dancer)

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-tiny-dancer-core](./crates/ruvector-tiny-dancer-core) | FastGRNN neural inference for AI routing | [![crates.io](https://img.shields.io/crates/v/ruvector-tiny-dancer-core.svg)](https://crates.io/crates/ruvector-tiny-dancer-core) |
| [ruvector-tiny-dancer-node](./crates/ruvector-tiny-dancer-node) | Node.js bindings for AI routing | [![crates.io](https://img.shields.io/crates/v/ruvector-tiny-dancer-node.svg)](https://crates.io/crates/ruvector-tiny-dancer-node) |
| [ruvector-tiny-dancer-wasm](./crates/ruvector-tiny-dancer-wasm) | WASM bindings for browser AI routing | [![crates.io](https://img.shields.io/crates/v/ruvector-tiny-dancer-wasm.svg)](https://crates.io/crates/ruvector-tiny-dancer-wasm) |

### Router (Semantic Routing)

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-router-core](./crates/ruvector-router-core) | Core semantic routing engine | [![crates.io](https://img.shields.io/crates/v/ruvector-router-core.svg)](https://crates.io/crates/ruvector-router-core) |
| [ruvector-router-cli](./crates/ruvector-router-cli) | CLI for router testing and benchmarking | [![crates.io](https://img.shields.io/crates/v/ruvector-router-cli.svg)](https://crates.io/crates/ruvector-router-cli) |
| [ruvector-router-ffi](./crates/ruvector-router-ffi) | FFI bindings for other languages | [![crates.io](https://img.shields.io/crates/v/ruvector-router-ffi.svg)](https://crates.io/crates/ruvector-router-ffi) |
| [ruvector-router-wasm](./crates/ruvector-router-wasm) | WASM bindings for browser routing | [![crates.io](https://img.shields.io/crates/v/ruvector-router-wasm.svg)](https://crates.io/crates/ruvector-router-wasm) |

### Dynamic Min-Cut (December 2025 Breakthrough)

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-mincut](./crates/ruvector-mincut) | Subpolynomial fully-dynamic min-cut ([arXiv:2512.13105](https://arxiv.org/abs/2512.13105)) | [![crates.io](https://img.shields.io/crates/v/ruvector-mincut.svg)](https://crates.io/crates/ruvector-mincut) |
| [ruvector-mincut-node](./crates/ruvector-mincut-node) | Node.js bindings for min-cut | [![crates.io](https://img.shields.io/crates/v/ruvector-mincut-node.svg)](https://crates.io/crates/ruvector-mincut-node) |
| [ruvector-mincut-wasm](./crates/ruvector-mincut-wasm) | WASM bindings for browser min-cut | [![crates.io](https://img.shields.io/crates/v/ruvector-mincut-wasm.svg)](https://crates.io/crates/ruvector-mincut-wasm) |

**First deterministic exact fully-dynamic min-cut** with verified **n^0.12 subpolynomial** update scaling:

- **Brain connectivity** — Detect Alzheimer's markers by tracking neural pathway changes in milliseconds
- **Network resilience** — Predict outages before they happen, route around failures instantly
- **AI agent coordination** — Find communication bottlenecks in multi-agent systems
- **Neural network pruning** — Identify which connections can be removed without losing accuracy
- **448+ tests**, 256-core parallel optimization, 8KB per core (compile-time verified)

```rust
use ruvector_mincut::{DynamicMinCut, Graph};

let mut graph = Graph::new();
graph.add_edge(0, 1, 10.0);
graph.add_edge(1, 2, 5.0);

let mincut = DynamicMinCut::new(&graph);
let (value, cut_edges) = mincut.compute();
// Updates in subpolynomial time as edges change
```

### Standalone Vector Database (rvLite)

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [rvlite](./crates/rvlite) | SQLite-style vector database for browsers & edge | [![crates.io](https://img.shields.io/crates/v/rvlite.svg)](https://crates.io/crates/rvlite) |

**Runs anywhere JavaScript runs** — browsers, Node.js, Deno, Bun, Cloudflare Workers, Vercel Edge:

- **SQL + SPARQL + Cypher** unified query interface
- **Zero dependencies** — thin orchestration over existing WASM crates
- **Self-learning** via SONA ReasoningBank integration

```typescript
import { RvLite } from '@rvlite/wasm';

const db = await RvLite.create();
await db.sql(`CREATE TABLE docs (id SERIAL, embedding VECTOR(384))`);
await db.sparql(`SELECT ?s WHERE { ?s rdf:type ex:Document }`);
await db.cypher(`MATCH (d:Doc)-[:SIMILAR]->(r) RETURN r`);
```

### Self-Optimizing Neural Architecture (SONA)

| Crate | Description | crates.io | npm |
|-------|-------------|-----------|-----|
| [ruvector-sona](./crates/sona) | Runtime-adaptive learning with LoRA, EWC++, and ReasoningBank | [![crates.io](https://img.shields.io/crates/v/ruvector-sona.svg)](https://crates.io/crates/ruvector-sona) | [![npm](https://img.shields.io/npm/v/@ruvector/sona.svg)](https://www.npmjs.com/package/@ruvector/sona) |

**SONA** enables AI systems to continuously improve from user feedback without expensive retraining:

- **Two-tier LoRA**: MicroLoRA (rank 1-2) for instant adaptation, BaseLoRA (rank 4-16) for long-term learning
- **EWC++**: Elastic Weight Consolidation prevents catastrophic forgetting
- **ReasoningBank**: K-means++ clustering stores and retrieves successful reasoning patterns
- **Lock-free Trajectories**: ~50ns overhead per step with crossbeam ArrayQueue
- **Sub-millisecond Learning**: <0.8ms per trajectory processing

```bash
# Rust
cargo add ruvector-sona

# Node.js
npm install @ruvector/sona
```

```rust
use ruvector_sona::{SonaEngine, SonaConfig};

let engine = SonaEngine::new(SonaConfig::default());
let traj_id = engine.start_trajectory(query_embedding);
engine.record_step(traj_id, node_id, 0.85, 150);
engine.end_trajectory(traj_id, 0.90);
engine.learn_from_feedback(LearningSignal::positive(50.0, 0.95));
```

```javascript
// Node.js
const { SonaEngine } = require('@ruvector/sona');

const engine = new SonaEngine(256); // 256 hidden dimensions
const trajId = engine.beginTrajectory([0.1, 0.2, ...]);
engine.addTrajectoryStep(trajId, activations, attention, 0.9);
engine.endTrajectory(trajId, 0.95);
```

### PostgreSQL Extension

| Crate | Description | crates.io | npm |
|-------|-------------|-----------|-----|
| [ruvector-postgres](./crates/ruvector-postgres) | pgvector-compatible PostgreSQL extension with SIMD optimization | [![crates.io](https://img.shields.io/crates/v/ruvector-postgres.svg)](https://crates.io/crates/ruvector-postgres) | [![npm](https://img.shields.io/npm/v/@ruvector/postgres-cli.svg)](https://www.npmjs.com/package/@ruvector/postgres-cli) |

**v0.2.0** — Drop-in replacement for pgvector with **77+ SQL functions**, full **AVX-512/AVX2/NEON SIMD** acceleration (~2x faster than AVX2), HNSW and IVFFlat indexes, 39 attention mechanisms, GNN layers, hyperbolic embeddings (Poincaré + Lorentz), sparse vectors/BM25, **W3C SPARQL 1.1** with 50+ RDF functions, **local embeddings** (6 fastembed models), and self-learning capabilities.

```bash
# Docker (recommended)
docker run -d -e POSTGRES_PASSWORD=secret -p 5432:5432 ruvector/postgres:latest

# From source
cargo install cargo-pgrx --version "0.12.9" --locked
cargo pgrx install --release

# CLI tool for management
npm install -g @ruvector/postgres-cli
ruvector-pg install
ruvector-pg vector create table --dim 1536 --index hnsw
```

See [ruvector-postgres README](./crates/ruvector-postgres/README.md) for full SQL API reference and advanced features.

### Tools & Utilities

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-bench](./crates/ruvector-bench) | Benchmarking suite for vector operations | [![crates.io](https://img.shields.io/crates/v/ruvector-bench.svg)](https://crates.io/crates/ruvector-bench) |
| [profiling](./crates/profiling) | Performance profiling and analysis tools | [![crates.io](https://img.shields.io/crates/v/ruvector-profiling.svg)](https://crates.io/crates/ruvector-profiling) |
| [micro-hnsw-wasm](./crates/micro-hnsw-wasm) | Lightweight HNSW implementation for WASM | [![crates.io](https://img.shields.io/crates/v/micro-hnsw-wasm.svg)](https://crates.io/crates/micro-hnsw-wasm) |

### Self-Learning Intelligence Hooks

**Make your AI assistant smarter over time.**

When you use Claude Code (or any AI coding assistant), it starts fresh every session. It doesn't remember which approaches worked, which files you typically edit together, or what errors you've seen before.

**RuVector Hooks fixes this.** It's a lightweight intelligence layer that:

1. **Remembers what works** — Tracks which agent types succeed for different tasks
2. **Learns from mistakes** — Records error patterns and suggests fixes you've used before
3. **Predicts your workflow** — Knows that after editing `api.rs`, you usually edit `api_test.rs`
4. **Coordinates teams** — Manages multi-agent swarms for complex tasks

Think of it as giving your AI assistant a memory and intuition about your codebase.

#### How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Claude Code    │────▶│  RuVector Hooks  │────▶│   Intelligence  │
│  (PreToolUse)   │     │   (pre-edit)     │     │      Layer      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
         ┌───────────────────────────────────────────────┘
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Q-Learning    │     │  Vector Memory   │     │  Swarm Graph    │
│   α=0.1 γ=0.95  │     │  64-dim embed    │     │  Coordination   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

The hooks integrate with Claude Code's event system:
- **PreToolUse** → Provides guidance before edits (agent routing, related files)
- **PostToolUse** → Records outcomes for learning (success/failure, patterns)
- **SessionStart/Stop** → Manages session state and metrics export

#### Technical Specifications

| Component | Implementation | Details |
|-----------|----------------|---------|
| **Q-Learning** | Temporal Difference | α=0.1, γ=0.95, ε=0.1 (ε-greedy exploration) |
| **Embeddings** | Hash-based vectors | 64 dimensions, normalized, cosine similarity |
| **LRU Cache** | `lru` crate | 1000 entries, ~10x faster Q-value lookups |
| **Compression** | `flate2` gzip | 70-83% storage reduction, fast compression |
| **Storage** | JSON / PostgreSQL | Auto-fallback, 5000 memory entry limit |
| **Cross-platform** | Rust + TypeScript | Windows (USERPROFILE), Unix (HOME) |

#### Performance

| Metric | Value |
|--------|-------|
| Q-value lookup (cached) | <1µs |
| Q-value lookup (uncached) | ~50µs |
| Memory search (1000 entries) | <5ms |
| Storage compression ratio | 70-83% |
| Session start overhead | <10ms |

| Crate/Package | Description | Status |
|---------------|-------------|--------|
| [ruvector-cli hooks](./crates/ruvector-cli) | Rust CLI with 29 hooks commands | [![crates.io](https://img.shields.io/crates/v/ruvector-cli.svg)](https://crates.io/crates/ruvector-cli) |
| [@ruvector/cli hooks](./npm/packages/cli) | npm CLI with 26 hooks commands | [![npm](https://img.shields.io/npm/v/@ruvector/cli.svg)](https://www.npmjs.com/package/@ruvector/cli) |

#### Quick Start

```bash
# Rust CLI
cargo install ruvector-cli
ruvector hooks init
ruvector hooks install

# npm CLI
npx @ruvector/cli hooks init
npx @ruvector/cli hooks install
```

#### Core Capabilities

| Feature | Description | Technical Details |
|---------|-------------|-------------------|
| **Q-Learning Routing** | Routes tasks to best agent with learned confidence scores | TD learning with α=0.1, γ=0.95, ε-greedy exploration |
| **Semantic Memory** | Vector-based memory with embeddings for context retrieval | 64-dim hash embeddings, cosine similarity, top-k search |
| **Error Learning** | Records error patterns and suggests fixes | Pattern matching for E0308, E0433, TS2322, etc. |
| **File Sequences** | Predicts next files to edit based on historical patterns | Markov chain transitions, frequency-weighted suggestions |
| **Swarm Coordination** | Registers agents, tracks coordination edges, optimizes | Graph-based topology, weighted edges, task assignment |
| **LRU Cache** | 1000-entry cache for faster Q-value lookups | ~10x speedup, automatic eviction, RefCell for interior mutability |
| **Gzip Compression** | Storage savings with automatic compression | flate2 fast mode, 70-83% reduction, transparent load/save |
| **Batch Saves** | Dirty flag tracking to reduce disk I/O | Only writes when data changes, force_save() override |
| **Shell Completions** | Tab completion for all commands | bash, zsh, fish, PowerShell support |

#### Supported Error Codes

The intelligence layer has built-in knowledge for common error patterns:

| Language | Error Codes | Auto-Suggested Fixes |
|----------|-------------|---------------------|
| **Rust** | E0308, E0433, E0425, E0277, E0382 | Type mismatches, missing imports, borrow checker |
| **TypeScript** | TS2322, TS2339, TS2345, TS7006 | Type assignments, property access, argument types |
| **Python** | ImportError, AttributeError, TypeError | Module imports, attribute access, type errors |
| **Go** | undefined, cannot use, not enough arguments | Variable scope, type conversion, function calls |

#### Commands Reference

```bash
# Setup
ruvector hooks init [--force]           # Initialize hooks in project
ruvector hooks install                   # Install into Claude settings

# Core
ruvector hooks stats                     # Show intelligence statistics
ruvector hooks session-start             # Start a new session
ruvector hooks session-end               # End session with metrics

# Memory
ruvector hooks remember -t edit "content"  # Store in semantic memory
ruvector hooks recall "query" -k 5         # Search memory semantically

# Learning
ruvector hooks learn <state> <action> --reward 0.8  # Record trajectory
ruvector hooks suggest <state> --actions "a,b,c"    # Get action suggestion
ruvector hooks route "implement caching" --file src/cache.rs  # Route to agent

# Claude Code Hooks
ruvector hooks pre-edit <file>           # Pre-edit intelligence hook
ruvector hooks post-edit <file> --success  # Post-edit learning hook
ruvector hooks pre-command <cmd>         # Pre-command hook
ruvector hooks post-command <cmd> --success  # Post-command hook

# Intelligence
ruvector hooks record-error <cmd> <stderr>  # Record error pattern
ruvector hooks suggest-fix E0308           # Get fix for error code
ruvector hooks suggest-next <file> -n 3    # Predict next files
ruvector hooks should-test <file>          # Check if tests needed

# Swarm
ruvector hooks swarm-register <id> <type>  # Register agent
ruvector hooks swarm-coordinate <src> <tgt>  # Record coordination
ruvector hooks swarm-optimize "task1,task2"  # Optimize distribution
ruvector hooks swarm-recommend "rust"      # Recommend agent for task
ruvector hooks swarm-heal <agent-id>       # Handle agent failure
ruvector hooks swarm-stats                 # Show swarm statistics

# Optimization (Rust only)
ruvector hooks compress                   # Compress storage (70-83% savings)
ruvector hooks cache-stats                # Show LRU cache statistics
ruvector hooks completions bash           # Generate shell completions
```

#### Tutorial: Claude Code Integration

**1. Initialize and install hooks:**

```bash
ruvector hooks init
ruvector hooks install --settings-dir .claude
```

This creates `.claude/settings.json` with hook configurations:

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Edit|Write|MultiEdit", "hooks": ["ruvector hooks pre-edit \"$TOOL_INPUT_FILE_PATH\""] },
      { "matcher": "Bash", "hooks": ["ruvector hooks pre-command \"$TOOL_INPUT_COMMAND\""] }
    ],
    "PostToolUse": [
      { "matcher": "Edit|Write|MultiEdit", "hooks": ["ruvector hooks post-edit ... --success"] },
      { "matcher": "Bash", "hooks": ["ruvector hooks post-command ... --success"] }
    ],
    "SessionStart": ["ruvector hooks session-start"],
    "Stop": ["ruvector hooks session-end --export-metrics"],
    "PreCompact": ["ruvector hooks pre-compact"]
  }
}
```

**All 5 Claude Code hooks covered:**
| Hook | When It Fires | What RuVector Does |
|------|---------------|-------------------|
| `PreToolUse` | Before file edit or command | Suggests agent, shows related files |
| `PostToolUse` | After file edit or command | Records outcome, updates Q-values |
| `SessionStart` | When session begins | Loads intelligence, shows stats |
| `Stop` | When session ends | Saves state, exports metrics |
| `PreCompact` | Before context compaction | Preserves critical memories |

**2. Use routing for intelligent agent selection:**

```bash
# Route a task to the best agent
$ ruvector hooks route "implement vector search" --file src/lib.rs
{
  "recommended": "rust-developer",
  "confidence": 0.85,
  "reasoning": "learned from 47 similar edits"
}
```

**3. Learn from outcomes:**

```bash
# Record successful outcome
ruvector hooks learn "edit-rs-lib" "rust-developer" --reward 1.0

# Record failed outcome
ruvector hooks learn "edit-rs-lib" "typescript-dev" --reward -0.5
```

**4. Get error fix suggestions:**

```bash
$ ruvector hooks suggest-fix E0308
{
  "code": "E0308",
  "type": "type_mismatch",
  "fixes": [
    "Check return type matches function signature",
    "Use .into() or .as_ref() for type conversion",
    "Verify generic type parameters"
  ]
}
```

#### Tutorial: Swarm Coordination

**1. Register agents:**

```bash
ruvector hooks swarm-register agent-1 rust-developer --capabilities "rust,async,testing"
ruvector hooks swarm-register agent-2 typescript-dev --capabilities "ts,react,node"
ruvector hooks swarm-register agent-3 reviewer --capabilities "review,security,performance"
```

**2. Record coordination patterns:**

```bash
# Agent-1 hands off to Agent-3 for review
ruvector hooks swarm-coordinate agent-1 agent-3 --weight 0.9
```

**3. Optimize task distribution:**

```bash
$ ruvector hooks swarm-optimize "implement-api,write-tests,code-review"
{
  "assignments": {
    "implement-api": "agent-1",
    "write-tests": "agent-1",
    "code-review": "agent-3"
  }
}
```

**4. Handle failures with self-healing:**

```bash
# Mark agent as failed and redistribute
ruvector hooks swarm-heal agent-2
```

#### PostgreSQL Storage (Optional)

For production deployments, use PostgreSQL instead of JSON files:

```bash
# Set connection URL
export RUVECTOR_POSTGRES_URL="postgres://user:pass@localhost/ruvector"

# Or use individual variables
export RUVECTOR_PG_HOST=localhost
export RUVECTOR_PG_USER=ruvector
export RUVECTOR_PG_DATABASE=ruvector

# Apply schema
psql $RUVECTOR_POSTGRES_URL -f crates/ruvector-cli/sql/hooks_schema.sql
```

The PostgreSQL backend provides:
- Vector embeddings with native `ruvector` type
- Q-learning functions (`ruvector_hooks_update_q`, `ruvector_hooks_best_action`)
- Swarm coordination tables with foreign key relationships
- Automatic memory cleanup (keeps last 5000 entries)

### Scientific OCR (SciPix)

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-scipix](./examples/scipix) | OCR engine for scientific documents, math equations → LaTeX/MathML | [![crates.io](https://img.shields.io/crates/v/ruvector-scipix.svg)](https://crates.io/crates/ruvector-scipix) |

**SciPix** extracts text and mathematical equations from images, converting them to LaTeX, MathML, or plain text. Features GPU-accelerated ONNX inference, SIMD-optimized preprocessing, REST API server, CLI tool, and MCP integration for AI assistants.

```bash
# Install
cargo add ruvector-scipix

# CLI usage
scipix-cli ocr --input equation.png --format latex
scipix-cli serve --port 3000

# MCP server for Claude/AI assistants
scipix-cli mcp
claude mcp add scipix -- scipix-cli mcp
```

### ONNX Embeddings

| Example | Description | Path |
|---------|-------------|------|
| [ruvector-onnx-embeddings](./examples/onnx-embeddings) | Production-ready ONNX embedding generation in pure Rust | `examples/onnx-embeddings` |

**ONNX Embeddings** provides native embedding generation using ONNX Runtime — no Python required. Supports 8+ pretrained models (all-MiniLM, BGE, E5, GTE), multiple pooling strategies, GPU acceleration (CUDA, TensorRT, CoreML, WebGPU), and direct RuVector index integration for RAG pipelines.

```rust
use ruvector_onnx_embeddings::{Embedder, PretrainedModel};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create embedder with default model (all-MiniLM-L6-v2)
    let mut embedder = Embedder::default_model().await?;

    // Generate embedding (384 dimensions)
    let embedding = embedder.embed_one("Hello, world!")?;

    // Compute semantic similarity
    let sim = embedder.similarity(
        "I love programming in Rust",
        "Rust is my favorite language"
    )?;
    println!("Similarity: {:.4}", sim); // ~0.85

    Ok(())
}
```

**Supported Models:**
| Model | Dimension | Speed | Best For |
|-------|-----------|-------|----------|
| `AllMiniLmL6V2` | 384 | Fast | General purpose (default) |
| `BgeSmallEnV15` | 384 | Fast | Search & retrieval |
| `AllMpnetBaseV2` | 768 | Accurate | Production RAG |

### Bindings & Tools

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-node](./crates/ruvector-node) | Main Node.js bindings (napi-rs) | [![crates.io](https://img.shields.io/crates/v/ruvector-node.svg)](https://crates.io/crates/ruvector-node) |
| [ruvector-wasm](./crates/ruvector-wasm) | Main WASM bindings for browsers | [![crates.io](https://img.shields.io/crates/v/ruvector-wasm.svg)](https://crates.io/crates/ruvector-wasm) |
| [ruvllm-wasm](./crates/ruvllm-wasm) | LLM integration WASM bindings | [![crates.io](https://img.shields.io/crates/v/ruvllm-wasm.svg)](https://crates.io/crates/ruvllm-wasm) |
| [ruvector-cli](./crates/ruvector-cli) | Command-line interface | [![crates.io](https://img.shields.io/crates/v/ruvector-cli.svg)](https://crates.io/crates/ruvector-cli) |
| [ruvector-server](./crates/ruvector-server) | HTTP/gRPC server | [![crates.io](https://img.shields.io/crates/v/ruvector-server.svg)](https://crates.io/crates/ruvector-server) |

### Examples

Production-ready examples demonstrating RuVector integration patterns, from cognitive AI substrates to WASM browser deployments.

| Example | Description | Type |
|---------|-------------|------|
| [mincut](./examples/mincut) | 6 self-organizing network demos: strange loops, time crystals, causal discovery | Rust |
| [exo-ai-2025](./examples/exo-ai-2025) | Cognitive substrate with 9 neural-symbolic crates + 11 research experiments | Rust |
| [ruvLLM](./examples/ruvLLM) | LLM integration patterns for RAG and AI agents | Rust |
| [apify](./examples/apify) | 13 Apify actors: trading, memory engine, synth data, market research | npm |
| [google-cloud](./examples/google-cloud) | GCP deployment templates for Cloud Run, GKE, and Vertex AI | Rust |
| [ultra-low-latency-sim](./examples/ultra-low-latency-sim) | 13+ quadrillion meta-simulations/sec with SIMD | Rust |
| [meta-cognition-spiking-neural-network](./examples/meta-cognition-spiking-neural-network) | Spiking neural network with meta-cognitive learning | npm |
| [onnx-embeddings](./examples/onnx-embeddings) | Production ONNX embedding generation without Python | Rust |
| [refrag-pipeline](./examples/refrag-pipeline) | RAG pipeline with vector search and document processing | Rust |
| [scipix](./examples/scipix) | Scientific OCR: equations → LaTeX/MathML with ONNX inference | Rust |
| [spiking-network](./examples/spiking-network) | Biologically-inspired spiking neural networks | Rust |
| [wasm-react](./examples/wasm-react) | React integration with WASM vector operations | WASM |
| [wasm-vanilla](./examples/wasm-vanilla) | Vanilla JS WASM example for browser vector search | WASM |
| [agentic-jujutsu](./examples/agentic-jujutsu) | Quantum-resistant version control for AI agents | Rust |
| [graph](./examples/graph) | Graph database examples with Cypher queries | Rust |
| [nodejs](./examples/nodejs) | Node.js integration examples | Node.js |
| [rust](./examples/rust) | Core Rust usage examples | Rust |

### npm Packages

#### ✅ Published

| Package | Description | npm |
|---------|-------------|-----|
| [ruvector](https://www.npmjs.com/package/ruvector) | All-in-one CLI & package (vectors, graphs, GNN) | [![npm](https://img.shields.io/npm/v/ruvector.svg)](https://www.npmjs.com/package/ruvector) |
| [@ruvector/core](https://www.npmjs.com/package/@ruvector/core) | Core vector database with native Rust bindings | [![npm](https://img.shields.io/npm/v/@ruvector/core.svg)](https://www.npmjs.com/package/@ruvector/core) |
| [@ruvector/gnn](https://www.npmjs.com/package/@ruvector/gnn) | Graph Neural Network layers & tensor compression | [![npm](https://img.shields.io/npm/v/@ruvector/gnn.svg)](https://www.npmjs.com/package/@ruvector/gnn) |
| [@ruvector/graph-node](https://www.npmjs.com/package/@ruvector/graph-node) | Hypergraph database with Cypher queries | [![npm](https://img.shields.io/npm/v/@ruvector/graph-node.svg)](https://www.npmjs.com/package/@ruvector/graph-node) |
| [@ruvector/tiny-dancer](https://www.npmjs.com/package/@ruvector/tiny-dancer) | FastGRNN neural inference for AI agent routing | [![npm](https://img.shields.io/npm/v/@ruvector/tiny-dancer.svg)](https://www.npmjs.com/package/@ruvector/tiny-dancer) |
| [@ruvector/router](https://www.npmjs.com/package/@ruvector/router) | Semantic router with HNSW vector search | [![npm](https://img.shields.io/npm/v/@ruvector/router.svg)](https://www.npmjs.com/package/@ruvector/router) |
| [@ruvector/agentic-synth](https://www.npmjs.com/package/@ruvector/agentic-synth) | Synthetic data generator for AI/ML | [![npm](https://img.shields.io/npm/v/@ruvector/agentic-synth.svg)](https://www.npmjs.com/package/@ruvector/agentic-synth) |
| [@ruvector/attention](https://www.npmjs.com/package/@ruvector/attention) | 39 attention mechanisms for transformers & GNNs | [![npm](https://img.shields.io/npm/v/@ruvector/attention.svg)](https://www.npmjs.com/package/@ruvector/attention) |
| [@ruvector/postgres-cli](https://www.npmjs.com/package/@ruvector/postgres-cli) | CLI for ruvector-postgres extension management | [![npm](https://img.shields.io/npm/v/@ruvector/postgres-cli.svg)](https://www.npmjs.com/package/@ruvector/postgres-cli) |
| [@ruvector/wasm](https://www.npmjs.com/package/@ruvector/wasm) | WASM fallback for core vector DB | [![npm](https://img.shields.io/npm/v/@ruvector/wasm.svg)](https://www.npmjs.com/package/@ruvector/wasm) |
| [@ruvector/gnn-wasm](https://www.npmjs.com/package/@ruvector/gnn-wasm) | WASM fallback for GNN layers | [![npm](https://img.shields.io/npm/v/@ruvector/gnn-wasm.svg)](https://www.npmjs.com/package/@ruvector/gnn-wasm) |
| [@ruvector/graph-wasm](https://www.npmjs.com/package/@ruvector/graph-wasm) | WASM fallback for graph DB | [![npm](https://img.shields.io/npm/v/@ruvector/graph-wasm.svg)](https://www.npmjs.com/package/@ruvector/graph-wasm) |
| [@ruvector/attention-wasm](https://www.npmjs.com/package/@ruvector/attention-wasm) | WASM fallback for attention mechanisms | [![npm](https://img.shields.io/npm/v/@ruvector/attention-wasm.svg)](https://www.npmjs.com/package/@ruvector/attention-wasm) |
| [@ruvector/tiny-dancer-wasm](https://www.npmjs.com/package/@ruvector/tiny-dancer-wasm) | WASM fallback for AI routing | [![npm](https://img.shields.io/npm/v/@ruvector/tiny-dancer-wasm.svg)](https://www.npmjs.com/package/@ruvector/tiny-dancer-wasm) |
| [@ruvector/router-wasm](https://www.npmjs.com/package/@ruvector/router-wasm) | WASM fallback for semantic router | [![npm](https://img.shields.io/npm/v/@ruvector/router-wasm.svg)](https://www.npmjs.com/package/@ruvector/router-wasm) |
| [@ruvector/sona](https://www.npmjs.com/package/@ruvector/sona) | Self-Optimizing Neural Architecture (SONA) | [![npm](https://img.shields.io/npm/v/@ruvector/sona.svg)](https://www.npmjs.com/package/@ruvector/sona) |
| [@ruvector/cluster](https://www.npmjs.com/package/@ruvector/cluster) | Distributed clustering & sharding | [![npm](https://img.shields.io/npm/v/@ruvector/cluster.svg)](https://www.npmjs.com/package/@ruvector/cluster) |
| [@ruvector/server](https://www.npmjs.com/package/@ruvector/server) | HTTP/gRPC server mode | [![npm](https://img.shields.io/npm/v/@ruvector/server.svg)](https://www.npmjs.com/package/@ruvector/server) |

**Platform-specific native bindings** (auto-detected):
- `@ruvector/node-linux-x64-gnu`, `@ruvector/node-linux-arm64-gnu`, `@ruvector/node-darwin-x64`, `@ruvector/node-darwin-arm64`, `@ruvector/node-win32-x64-msvc`
- `@ruvector/gnn-linux-x64-gnu`, `@ruvector/gnn-linux-arm64-gnu`, `@ruvector/gnn-darwin-x64`, `@ruvector/gnn-darwin-arm64`, `@ruvector/gnn-win32-x64-msvc`
- `@ruvector/tiny-dancer-linux-x64-gnu`, `@ruvector/tiny-dancer-linux-arm64-gnu`, `@ruvector/tiny-dancer-darwin-x64`, `@ruvector/tiny-dancer-darwin-arm64`, `@ruvector/tiny-dancer-win32-x64-msvc`
- `@ruvector/router-linux-x64-gnu`, `@ruvector/router-linux-arm64-gnu`, `@ruvector/router-darwin-x64`, `@ruvector/router-darwin-arm64`, `@ruvector/router-win32-x64-msvc`
- `@ruvector/attention-linux-x64-gnu`, `@ruvector/attention-linux-arm64-gnu`, `@ruvector/attention-darwin-x64`, `@ruvector/attention-darwin-arm64`, `@ruvector/attention-win32-x64-msvc`

#### 🚧 Planned

| Package | Description | Status |
|---------|-------------|--------|
| @ruvector/raft | Raft consensus for distributed ops | Crate ready |
| @ruvector/replication | Multi-master replication | Crate ready |
| @ruvector/scipix | Scientific OCR (LaTeX/MathML) | Crate ready |

See [GitHub Issue #20](https://github.com/ruvnet/ruvector/issues/20) for multi-platform npm package roadmap.

```bash
# Install all-in-one package
npm install ruvector

# Or install individual packages
npm install @ruvector/core @ruvector/gnn @ruvector/graph-node

# List all available packages
npx ruvector install
```


```javascript
const ruvector = require('ruvector');

// Vector search
const db = new ruvector.VectorDB(128);
db.insert('doc1', embedding1);
const results = db.search(queryEmbedding, 10);

// Graph queries (Cypher)
db.execute("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})");
db.execute("MATCH (p:Person)-[:KNOWS]->(friend) RETURN friend.name");

// GNN-enhanced search
const layer = new ruvector.GNNLayer(128, 256, 4);
const enhanced = layer.forward(query, neighbors, weights);

// Compression (2-32x memory savings)
const compressed = ruvector.compress(embedding, 0.3);

// Tiny Dancer: AI agent routing
const router = new ruvector.Router();
const decision = router.route(candidates, { optimize: 'cost' });
```

### Rust

```bash
cargo add ruvector-graph ruvector-gnn
```

```rust
use ruvector_graph::{GraphDB, NodeBuilder};
use ruvector_gnn::{RuvectorLayer, differentiable_search};

let db = GraphDB::new();

let doc = NodeBuilder::new("doc1")
    .label("Document")
    .property("embedding", vec![0.1, 0.2, 0.3])
    .build();
db.create_node(doc)?;

// GNN layer
let layer = RuvectorLayer::new(128, 256, 4, 0.1);
let enhanced = layer.forward(&query, &neighbors, &weights);
```

```rust
use ruvector_raft::{RaftNode, RaftNodeConfig};
use ruvector_cluster::{ClusterManager, ConsistentHashRing};
use ruvector_replication::{SyncManager, SyncMode};

// Configure a 5-node Raft cluster
let config = RaftNodeConfig {
    node_id: "node-1".into(),
    cluster_members: vec!["node-1", "node-2", "node-3", "node-4", "node-5"]
        .into_iter().map(Into::into).collect(),
    election_timeout_min: 150,  // ms
    election_timeout_max: 300,  // ms
    heartbeat_interval: 50,     // ms
};
let raft = RaftNode::new(config);

// Auto-sharding with consistent hashing (150 virtual nodes per real node)
let ring = ConsistentHashRing::new(64, 3); // 64 shards, replication factor 3
let shard = ring.get_shard("my-vector-key");

// Multi-master replication with conflict resolution
let sync = SyncManager::new(SyncMode::SemiSync { min_replicas: 2 });
```

## Project Structure

```
crates/
├── ruvector-core/           # Vector DB engine (HNSW, storage)
├── ruvector-graph/          # Graph DB + Cypher parser + Hyperedges
├── ruvector-gnn/            # GNN layers, compression, training
├── ruvector-tiny-dancer-core/  # AI agent routing (FastGRNN)
├── ruvector-*-wasm/         # WebAssembly bindings
└── ruvector-*-node/         # Node.js bindings (napi-rs)
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./docs/development/CONTRIBUTING.md).

```bash
# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace

# Build WASM
cargo build -p ruvector-gnn-wasm --target wasm32-unknown-unknown
```

## License

MIT License — free for commercial and personal use.

---

<div align="center">

**Built by [rUv](https://ruv.io)** • [GitHub](https://github.com/ruvnet/ruvector) • [npm](https://npmjs.com/package/ruvector) • [Docs](./docs/)

*Vector search that gets smarter over time.*

</div>
