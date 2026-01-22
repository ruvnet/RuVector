# RuVector

[![Crates.io](https://img.shields.io/crates/v/ruvector-core.svg)](https://crates.io/crates/ruvector-core)
[![npm](https://img.shields.io/npm/v/ruvector.svg)](https://www.npmjs.com/package/ruvector)
[![npm Downloads](https://img.shields.io/npm/dt/ruvector.svg?label=total)](https://www.npmjs.com/package/ruvector)
[![npm Downloads](https://img.shields.io/npm/dm/ruvector.svg?label=monthly)](https://www.npmjs.com/package/ruvector)
[![HuggingFace](https://img.shields.io/badge/🤗-RuvLTRA_Models-yellow.svg)](https://huggingface.co/ruv/ruvltra)
[![ruv.io](https://img.shields.io/badge/ruv.io-website-purple.svg)](https://ruv.io)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**The vector database that gets smarter the more you use it.**

```bash
npx ruvector
```

Most vector databases are static—they store embeddings and search them. That's it. RuVector is different: it learns from every query, runs LLMs locally, scales horizontally, and costs nothing to operate.

| | Pinecone/Weaviate | RuVector |
|---|---|---|
| **Search improves over time** | ❌ | ✅ GNN layers learn from usage |
| **Run LLMs locally** | ❌ | ✅ ruvllm + RuvLTRA models ($0) |
| **Graph queries (Cypher)** | ❌ | ✅ `MATCH (a)-[:SIMILAR]->(b)` |
| **Self-learning AI hooks** | ❌ | ✅ Q-learning, HNSW memory |
| **Real-time graph updates** | ❌ Rebuild index | ✅ Dynamic min-cut (no rebuild) |
| **Horizontal scaling** | 💰 Paid | ✅ Raft consensus, free |
| **Works offline** | ❌ | ✅ Browser, edge, embedded |

**One package. Everything included:** vector search, graph queries, GNN learning, distributed clustering, local LLMs, 39 attention mechanisms, and WASM support.

<details>
<summary>📋 See Full Capabilities (30+ features)</summary>

**Core Vector Database**
| # | Capability | What It Does |
|---|------------|--------------|
| 1 | **Store vectors** | Embeddings from OpenAI, Cohere, local ONNX with HNSW indexing |
| 2 | **Query with Cypher** | Graph queries like Neo4j (`MATCH (a)-[:SIMILAR]->(b)`) |
| 3 | **The index learns** | GNN layers make search results improve over time |
| 4 | **Hyperbolic HNSW** | Hierarchical data in hyperbolic space for better tree structures |
| 5 | **Compress automatically** | 2-32x memory reduction with adaptive tiered compression |

**Distributed Systems**
| # | Capability | What It Does |
|---|------------|--------------|
| 6 | **Raft consensus** | Leader election, log replication, fault-tolerant coordination |
| 7 | **Multi-master replication** | Vector clocks, conflict resolution, geo-distributed sync |
| 8 | **Burst scaling** | 10-50x capacity scaling for traffic spikes |
| 9 | **Auto-sharding** | Automatic data partitioning across nodes |

**AI & Machine Learning**
| # | Capability | What It Does |
|---|------------|--------------|
| 10 | **Run LLMs locally** | ruvllm with GGUF, Metal/CUDA/ANE acceleration |
| 11 | **RuvLTRA models** | Pre-trained GGUF for routing & embeddings (<10ms) → [HuggingFace](https://huggingface.co/ruv/ruvltra) |
| 12 | **SONA learning** | Self-Optimizing Neural Architecture with LoRA, EWC++ |
| 13 | **39 attention mechanisms** | Flash, linear, graph, hyperbolic, mincut-gated (50% compute) |
| 14 | **Spiking neural networks** | Event-driven neuromorphic computing |
| 15 | **Mincut-gated transformer** | Dynamic attention via graph min-cut optimization |
| 16 | **Route AI requests** | Semantic routing + FastGRNN for LLM optimization |

**Specialized Processing**
| # | Capability | What It Does |
|---|------------|--------------|
| 17 | **SciPix OCR** | LaTeX/MathML extraction from scientific documents |
| 18 | **DAG workflows** | Self-learning directed acyclic graph execution |
| 19 | **Cognitum Gate** | Cognitive AI gateway with TileZero acceleration |
| 20 | **FPGA transformer** | Hardware-accelerated transformer inference |
| 21 | **Quantum coherence** | ruQu for quantum error correction via dynamic min-cut |

**Platform & Integration**
| # | Capability | What It Does |
|---|------------|--------------|
| 22 | **Run anywhere** | Node.js, browser (WASM), edge (rvLite), HTTP server, Rust |
| 23 | **Drop into Postgres** | pgvector-compatible extension with SIMD acceleration |
| 24 | **MCP integration** | Model Context Protocol server for AI assistant tools |
| 25 | **Cloud deployment** | One-click deploy to Cloud Run, Kubernetes |

**Self-Learning & Adaptation**
| # | Capability | What It Does |
|---|------------|--------------|
| 26 | **Self-learning hooks** | Q-learning, neural patterns, HNSW memory |
| 27 | **ReasoningBank** | Trajectory learning with verdict judgment |
| 28 | **Economy system** | Tokenomics, CRDT-based distributed state |
| 29 | **Nervous system** | Event-driven reactive architecture |
| 30 | **Agentic synthesis** | Multi-agent workflow composition |

</details>

*Think of it as: **Pinecone + Neo4j + PyTorch + llama.cpp + postgres + etcd** — in one Rust package.*

---

### Ecosystem: AI Agent Orchestration

RuVector powers two major AI orchestration platforms:

| Platform | Purpose | Install |
|----------|---------|---------|
| [**Claude-Flow**](https://github.com/ruvnet/claude-flow) | Enterprise multi-agent orchestration for Claude Code | `npx @claude-flow/cli@latest` |
| [**Agentic-Flow**](https://github.com/ruvnet/agentic-flow) | Standalone AI agent framework (any LLM provider) | `npx agentic-flow@latest` |

<details>
<summary><strong>Claude-Flow v3</strong> — Turn Claude Code into a collaborative AI team</summary>

**54+ specialized agents** working together on complex software engineering tasks:

```bash
# Install
npx @claude-flow/cli@latest init --wizard

# Spawn a swarm
npx @claude-flow/cli@latest swarm init --topology hierarchical --max-agents 8
```

**Key Features:**
- **SONA Learning**: Sub-50ms adaptive routing, learns optimal patterns over time
- **Queen-led Swarms**: Byzantine fault-tolerant consensus with 5 protocols (Raft, Gossip, CRDT)
- **HNSW Memory**: 150x-12,500x faster pattern retrieval via RuVector
- **175+ MCP Tools**: Native Model Context Protocol integration
- **Cost Optimization**: 3-tier routing extends Claude Code quota by 2.5x
- **Security**: AIDefence threat detection (<10ms), prompt injection blocking

</details>

<details>
<summary><strong>Agentic-Flow v2</strong> — Production AI agents for any cloud</summary>

**66 self-learning agents** with Claude Agent SDK, deployable to any cloud:

```bash
# Install
npx agentic-flow@latest

# Or with npm
npm install agentic-flow
```

**Key Features:**
- **SONA Architecture**: <1ms adaptive learning, +55% quality improvement
- **Flash Attention**: 2.49x JS speedup, 7.47x with NAPI bindings
- **213 MCP Tools**: Swarm management, memory, GitHub integration
- **Agent Booster**: 352x faster code editing for simple transforms
- **Multi-Provider**: Claude, GPT, Gemini, Cohere, local models with failover
- **Graph Reasoning**: GNN query refinement with +12.4% recall improvement

</details>

---

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
1. Takes your query and its nearest neighbors
2. Applies multi-head attention to weigh which neighbors matter
3. Updates representations based on graph structure
4. Returns better-ranked results

Over time, frequently-accessed paths get reinforced, making common queries faster and more accurate.


## Quick Start

### One-Line Install

```bash
# Interactive installer - lists all packages
npx ruvector install

# Or install directly
npm install ruvector
npx ruvector

# Self-learning hooks for Claude Code
npx @ruvector/cli hooks init
npx @ruvector/cli hooks install

# LLM runtime (SONA learning, HNSW memory)
npm install @ruvector/ruvllm
```

### Node.js / Browser

```bash
# Install
npm install ruvector

# Or try instantly
npx ruvector
```


<details>
<summary>📊 Comparison with Other Vector Databases</summary>

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
| **Local Embeddings** | ✅ 8+ models | ❌ | ❌ | ❌ | ❌ |
| **PostgreSQL Extension** | ✅ 77+ functions | ❌ | ❌ | ❌ | ❌ |
| **SIMD Optimization** | ✅ AVX-512/NEON | Partial | ✅ | ✅ | ❌ |
| **Metadata Filtering** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Sparse Vectors** | ✅ BM25/TF-IDF | ✅ | ✅ | ✅ | ❌ |
| **Raft Consensus** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Multi-Master Replication** | ✅ | ❌ | ❌ | ✅ | ❌ |
| **Auto-Sharding** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Auto-Compression** | ✅ 2-32x | ❌ | ❌ | ✅ | ❌ |
| **Snapshots/Backups** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Browser/WASM** | ✅ WebGPU | ❌ | ❌ | ❌ | ❌ |
| **Standalone Edge DB** | ✅ rvLite | ❌ | ❌ | ❌ | ❌ |
| **LLM Runtime** | ✅ ruvllm | ❌ | ❌ | ❌ | ❌ |
| **Pre-trained Models** | ✅ RuvLTRA (HF) | ❌ | ❌ | ❌ | ❌ |
| **MCP Server** | ✅ mcp-gate | ❌ | ❌ | ❌ | ❌ |
| **Self-Learning Hooks** | ✅ Q-learning+Neural+HNSW | ❌ | ❌ | ❌ | ❌ |
| **Quantum Coherence** | ✅ ruQu | ❌ | ❌ | ❌ | ❌ |
| **MinCut-Gated Attention** | ✅ 50% compute | ❌ | ❌ | ❌ | ❌ |
| **FPGA Acceleration** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Local ONNX Embeddings** | ✅ 8+ models | ❌ | ❌ | ❌ | ❌ |
| **Differentiable** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Multi-Tenancy** | ✅ Collections | ✅ | ✅ | ✅ | ✅ |
| **DAG Workflows** | ✅ Self-learning | ❌ | ❌ | ❌ | ❌ |
| **ReasoningBank** | ✅ Trajectory learning | ❌ | ❌ | ❌ | ❌ |
| **Economy System** | ✅ CRDT tokenomics | ❌ | ❌ | ❌ | ❌ |
| **Nervous System** | ✅ Event-driven | ❌ | ❌ | ❌ | ❌ |
| **Cognitum Gate** | ✅ TileZero | ❌ | ❌ | ❌ | ❌ |
| **SciPix OCR** | ✅ LaTeX/MathML | ❌ | ❌ | ❌ | ❌ |
| **Spiking Neural Nets** | ✅ Neuromorphic | ❌ | ❌ | ❌ | ❌ |
| **Node.js Native** | ✅ napi-rs | ❌ | ❌ | ❌ | ✅ |
| **Burst Scaling** | ✅ 10-50x | ✅ | ❌ | ✅ | ❌ |
| **Streaming API** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Open Source** | ✅ MIT | ❌ | ✅ | ✅ | ✅ |

*With PQ8 compression. Benchmarks on Apple M2 / Intel i7.

</details>

<details>
<summary>⚡ Core Features & Capabilities</summary>

### Core Capabilities

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Vector Search** | HNSW index, <0.5ms latency, SIMD acceleration | Fast enough for real-time apps |
| **Cypher Queries** | `MATCH`, `WHERE`, `CREATE`, `RETURN` | Familiar Neo4j syntax |
| **GNN Layers** | Neural network on index topology | Search improves with usage |
| **Hyperedges** | Connect 3+ nodes at once | Model complex relationships |
| **Metadata Filtering** | Filter vectors by properties | Combine semantic + structured search |
| **Collections** | Namespace isolation, multi-tenancy | Organize vectors by project/user |
| **Hyperbolic HNSW** | Poincaré ball indexing for hierarchies | Better tree/taxonomy embeddings |
| **Sparse Vectors** | BM25/TF-IDF hybrid search | Combine keyword + semantic |

### LLM Runtime

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **ruvllm** | Local LLM inference with GGUF models | Run AI without cloud APIs |
| **Metal/CUDA/ANE** | Hardware acceleration on Mac/NVIDIA/Apple | 10-50x faster inference |
| **ruvllm-wasm** | Browser LLM with WebGPU acceleration | Client-side AI, zero latency |
| **RuvLTRA Models** | Pre-trained GGUF for routing & embeddings | <10ms inference → [HuggingFace](https://huggingface.co/ruv/ruvltra) |
| **Streaming Tokens** | Real-time token generation | Responsive chat UX |
| **Quantization** | Q4, Q5, Q8 model support | Run 7B models in 4GB RAM |

```bash
npm install @ruvector/ruvllm        # Node.js
cargo add ruvllm                    # Rust
```

### Platform & Edge

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **rvLite** | Standalone 2MB edge database | IoT, mobile, embedded |
| **PostgreSQL Extension** | 77+ SQL functions, pgvector replacement | Drop-in upgrade for existing DBs |
| **MCP Server** | Model Context Protocol integration | AI assistant tool calling |
| **WASM/Browser** | Full client-side vector search | Offline-first apps |
| **Node.js Bindings** | Native napi-rs, zero-copy | No serialization overhead |
| **HTTP/gRPC Server** | REST API with streaming | Easy microservice integration |

```bash
docker pull ruvnet/ruvector-postgres    # PostgreSQL
npm install rvlite                       # Edge DB
npx ruvector mcp start                   # MCP Server
```

### Distributed Systems

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Raft Consensus** | Leader election, log replication | Strong consistency for metadata |
| **Auto-Sharding** | Consistent hashing, shard migration | Scale to billions of vectors |
| **Multi-Master Replication** | Write to any node, conflict resolution | High availability, no SPOF |
| **Snapshots** | Point-in-time backups, incremental | Disaster recovery |
| **Cluster Metrics** | Prometheus-compatible monitoring | Observability at scale |
| **Burst Scaling** | 10-50x capacity for traffic spikes | Handle viral moments |

```bash
cargo add ruvector-raft ruvector-cluster ruvector-replication
```

### AI & ML

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Tensor Compression** | f32→f16→PQ8→PQ4→Binary | 2-32x memory reduction |
| **Differentiable Search** | Soft attention k-NN | End-to-end trainable |
| **Semantic Router** | Route queries to optimal endpoints | Multi-model AI orchestration |
| **Hybrid Routing** | Keyword-first + embedding fallback | **90% accuracy** for agent routing |
| **Tiny Dancer** | FastGRNN neural inference | Optimize LLM inference costs |
| **Adaptive Routing** | Learn optimal routing strategies | Minimize latency, maximize accuracy |
| **SONA** | Two-tier LoRA + EWC++ + ReasoningBank | Runtime learning without retraining |
| **Local Embeddings** | 8+ ONNX models built-in | No external API needed |

### Specialized Processing

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **SciPix OCR** | LaTeX/MathML from scientific docs | Index research papers |
| **DAG Workflows** | Self-learning directed acyclic graphs | Complex pipeline orchestration |
| **Cognitum Gate** | Cognitive AI gateway + TileZero | Unified AI model routing |
| **FPGA Transformer** | Hardware-accelerated inference | Ultra-low latency serving |
| **ruQu Quantum** | Quantum error correction via min-cut | Future-proof algorithms |
| **Mincut-Gated Transformer** | Dynamic attention via graph optimization | **50% compute reduction** |
| **Sparse Inference** | Efficient sparse matrix operations | 10x faster for sparse data |

### Self-Learning & Adaptation

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Self-Learning Hooks** | Q-learning + neural patterns + HNSW | System improves automatically |
| **ReasoningBank** | Trajectory learning with verdict judgment | Learn from successes/failures |
| **Economy System** | Tokenomics, CRDT-based distributed state | Incentivize agent behavior |
| **Nervous System** | Event-driven reactive architecture | Real-time adaptation |
| **Agentic Synthesis** | Multi-agent workflow composition | Emergent problem solving |
| **EWC++** | Elastic weight consolidation | Prevent catastrophic forgetting |

```bash
npx @ruvector/cli hooks init      # Install self-learning hooks
npx @ruvector/cli hooks install   # Configure for Claude Code
```

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

</details>

<details>
<summary>🚀 Deployment Options</summary>

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **HTTP/gRPC Server** | REST API, streaming support | Easy integration |
| **WASM/Browser** | Full client-side support | Run AI search offline |
| **Node.js Bindings** | Native napi-rs bindings | No serialization overhead |
| **FFI Bindings** | C-compatible interface | Use from Python, Go, etc. |
| **CLI Tools** | Benchmarking, testing, management | DevOps-friendly |

</details>

<details>
<summary>📈 Performance Benchmarks</summary>

**Measured results** from [`/bench_results/`](./bench_results/):

| Configuration | QPS | p50 Latency | p99 Latency | Recall |
|---------------|-----|-------------|-------------|--------|
| **ruvector (optimized)** | 1,216 | 0.78ms | 0.78ms | 100% |
| **Multi-threaded (16)** | 3,597 | 2.86ms | 8.47ms | 100% |
| **ef_search=50** | 674 | 1.35ms | 1.35ms | 100% |
| Python baseline | 77 | 11.88ms | 11.88ms | 100% |
| Brute force | 12 | 77.76ms | 77.76ms | 100% |

*Dataset: 384D, 10K-50K vectors. See full results in [latency_benchmark.md](./bench_results/latency_benchmark.md).*

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

</details>

<details>
<summary>🗜️ Adaptive Compression Tiers</summary>

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

</details>

<details>
<summary>💡 Use Cases</summary>

### AI & LLM Applications

| Use Case | Features Used | Example |
|----------|---------------|---------|
| **RAG Pipelines** | Vector search, Local embeddings, ruvllm | [examples/ruvLLM](./examples/ruvLLM) |
| **AI Agent Routing** | Tiny Dancer, Semantic router, SONA | [Claude-Flow](https://github.com/ruvnet/claude-flow) |
| **Multi-Agent Orchestration** | GNN, HNSW memory, Consensus | [Agentic-Flow](https://github.com/ruvnet/agentic-flow) |
| **Self-Learning Chatbots** | ReasoningBank, EWC++, Neural patterns | [examples/meta-cognition](./examples/meta-cognition-spiking-neural-network) |

```javascript
// RAG with local LLM (zero cloud costs)
import { RuVector } from 'ruvector';
import { RuvLLM } from '@ruvector/ruvllm';

const db = new RuVector({ dimensions: 384 });
const llm = new RuvLLM({ model: 'ruvltra-small-0.5b-q4_k_m.gguf' });

// Search learns from usage via GNN layers
const context = await db.search(questionEmbedding, { k: 5 });
const response = await llm.generate(`Context: ${context}\n\nQ: ${question}`);
```

### Search & Discovery

| Use Case | Features Used | Example |
|----------|---------------|---------|
| **Semantic Search** | HNSW, Metadata filtering, SIMD | Core feature |
| **Hybrid Search** | BM25 + embeddings, Sparse vectors | [docs/api](./docs/api/) |
| **Image Similarity** | CLIP embeddings, Hyperbolic HNSW | [examples/wasm-react](./examples/wasm-react) |
| **Code Search** | Local ONNX embeddings, Graph queries | [examples/nodejs](./examples/nodejs) |

```javascript
// Hybrid search: keyword + semantic
const results = await db.search(query, {
  k: 10,
  filter: { category: 'electronics', price: { $lt: 500 } },
  hybridAlpha: 0.7,  // 70% semantic, 30% keyword
  rerank: true       // GNN-enhanced reranking
});
```

### Recommendations & Personalization

| Use Case | Features Used | Example |
|----------|---------------|---------|
| **Product Recommendations** | Graph queries, Cypher, GNN | [examples/graph](./examples/graph) |
| **Content Personalization** | User embeddings, Collaborative filtering | Real-time adaptation |
| **Similar Items** | Cosine similarity, Hyperbolic space | Hierarchical taxonomies |

```cypher
// Neo4j-style recommendations with learning
MATCH (user:User {id: $userId})-[:VIEWED]->(item:Product)
MATCH (item)-[:SIMILAR_TO]->(rec:Product)
WHERE NOT (user)-[:PURCHASED]->(rec)
RETURN rec ORDER BY rec.gnn_score DESC LIMIT 10
```

### Knowledge Management

| Use Case | Features Used | Example |
|----------|---------------|---------|
| **Knowledge Graphs** | Hypergraph, Cypher, SPARQL | [docs/api/CYPHER_REFERENCE.md](./docs/api/CYPHER_REFERENCE.md) |
| **Document Q&A** | Chunking, Embeddings, RAG | [examples/refrag-pipeline](./examples/refrag-pipeline) |
| **Scientific Papers** | SciPix OCR, LaTeX extraction | [examples/scipix](./examples/scipix) |
| **Research Discovery** | Citation graphs, Concept linking | Hyperedge relationships |

```cypher
// Multi-hop knowledge graph traversal
MATCH (paper:Paper)-[:CITES*1..3]->(cited:Paper)
WHERE paper.topic = 'machine learning'
MATCH (cited)-[:AUTHORED_BY]->(author:Researcher)
RETURN author, count(cited) as influence
ORDER BY influence DESC LIMIT 20
```

### Real-Time & Edge Computing

| Use Case | Features Used | Example |
|----------|---------------|---------|
| **Browser AI** | WASM, WebGPU, ruvllm-wasm | [examples/wasm-vanilla](./examples/wasm-vanilla) |
| **IoT Sensors** | rvLite, Edge DB, no_std | [examples/edge](./examples/edge) |
| **Mobile Apps** | 2MB footprint, Offline-first | [examples/edge-net](./examples/edge-net) |
| **Streaming Data** | Real-time indexing, Dynamic min-cut | [examples/neural-trader](./examples/neural-trader) |

```javascript
// Browser-based AI (no server required)
import init, { RuvLLMWasm } from '@ruvector/ruvllm-wasm';

await init();
const llm = await RuvLLMWasm.new(true); // WebGPU enabled
await llm.load_model_from_url('https://cdn.example.com/model.gguf');

// Runs entirely in browser
const response = await llm.generate('Explain quantum computing', {
  max_tokens: 200,
  temperature: 0.7
});
```

### Scientific & Research

| Use Case | Features Used | Example |
|----------|---------------|---------|
| **Neural Network Analysis** | Spiking NN, Meta-cognition | [examples/meta-cognition](./examples/meta-cognition-spiking-neural-network) |
| **Algorithmic Trading** | Neural Trader, Time-series | [examples/neural-trader](./examples/neural-trader) |
| **Quantum Computing** | ruQu, Min-cut coherence | [crates/ruQu](./crates/ruQu) |
| **Brain Connectivity** | Dynamic min-cut, Network analysis | [examples/mincut](./examples/mincut) |

```rust
// Neuromorphic computing with spiking networks
use ruvector_nervous_system::{SpikingNetwork, LIFNeuron};

let mut network = SpikingNetwork::new();
network.add_layer(LIFNeuron::new(128));  // 128 spiking neurons
network.enable_stdp();                    // Spike-timing plasticity

// 10-50x more energy efficient than traditional ANNs
let output = network.forward(&input_spikes);
```

### Neuromorphic Computing (micro-hnsw v2.3)

Novel neuromorphic discoveries for brain-inspired vector search in **11.8KB WASM**.

| Discovery | Description | Benefit |
|-----------|-------------|---------|
| **Spike-Timing Vector Encoding** | Convert vectors to temporal spike patterns | Temporal similarity matching |
| **Homeostatic Plasticity** | Self-stabilizing network activity | Prevents runaway activation |
| **Oscillatory Resonance** | Gamma-frequency (40Hz) search amplification | Improved recall via resonance |
| **Winner-Take-All Circuits** | Competitive neural selection with lateral inhibition | Sparse, efficient representations |
| **Dendritic Computation** | Non-linear local processing in dendrites | Complex pattern detection |
| **STDP Learning** | Spike-Timing Dependent Plasticity | Unsupervised Hebbian learning |

```rust
// micro-hnsw: Neuromorphic HNSW in 11.8KB WASM
use micro_hnsw_wasm::{MicroHnsw, LIFNeuron, SpikeTrain};

// 256 cores × 32 vectors = 8K total capacity
let mut hnsw = MicroHnsw::new(16, Metric::Cosine);  // 16-dim vectors

// Spike-timing vector encoding
let spike_train = SpikeTrain::encode(&embedding, 8);  // 8-bit temporal resolution

// LIF neuron with STDP learning
let neuron = LIFNeuron::new(0.8);  // threshold = 0.8
neuron.enable_stdp(0.01, 0.012);   // A+ = 0.01, A- = 0.012
neuron.enable_homeostasis(0.1);    // Target rate: 0.1 spikes/ms

// Winner-take-all search with lateral inhibition
let results = hnsw.search_wta(&query, 10, 0.8);  // WTA inhibition = 0.8
```

### Distributed & Enterprise

| Use Case | Features Used | Example |
|----------|---------------|---------|
| **Multi-Region Deployment** | Raft consensus, Replication | [docs/cloud-architecture](./docs/cloud-architecture/) |
| **High Availability** | Auto-sharding, Failover | 99.99% SLA capable |
| **PostgreSQL Integration** | 230+ SQL functions, pgvector replacement | [crates/ruvector-postgres](./crates/ruvector-postgres) |
| **Burst Traffic** | 10-50x scaling, Load balancing | [examples/google-cloud](./examples/google-cloud) |

```sql
-- PostgreSQL with RuVector extension
CREATE EXTENSION ruvector;

-- Create vector column with GNN-enhanced index
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding VECTOR(384)
);

CREATE INDEX ON documents USING hnsw_gnn (embedding);

-- Self-improving search
SELECT * FROM documents
ORDER BY embedding <-> query_vector
LIMIT 10;
```

### AI Safety & Coherence (Cognitum Gate)

A **256-tile WASM fabric** for real-time AI agent safety decisions with cryptographic verification.

| Component | Description | Memory |
|-----------|-------------|--------|
| **Worker Tiles (255)** | Local graph shards, evidence accumulation, witness fragments | 64KB each |
| **TileZero Arbiter** | Supergraph merging, global decisions, permit tokens | Central |
| **Gate Decisions** | Permit / Defer / Deny with confidence scores | <1ms |
| **Witness Receipts** | Hash-chained cryptographic audit trail | Immutable |

| Feature | Description |
|---------|-------------|
| **Anytime-Valid Testing** | Sequential hypothesis testing with e-values |
| **Min-Cut Aggregation** | Global coherence via distributed min-cut |
| **Signed Permits** | Cryptographic tokens for approved actions |
| **Evidence Filters** | Three-filter decision system (structural, evidence, combined) |

```rust
// Cognitum Gate: AI agent safety in microseconds
use cognitum_gate_tilezero::{GateDecision, ActionContext, PermitToken};

let gate = CoherenceGate::new_256_tiles();

// Evaluate action safety
let context = ActionContext {
    action_id: "deploy-model-v2".into(),
    action_type: "config_change".into(),
    agent_id: "coder-agent-01".into(),
    ..Default::default()
};

let decision = gate.evaluate(&context).await?;

match decision {
    GateDecision::Permit(token) => {
        // Cryptographically signed permit token
        assert!(token.verify(&gate.public_key()));
        execute_action(token);
    }
    GateDecision::Defer(reason) => {
        // Needs more evidence - retry later
        log::info!("Deferred: {}", reason);
    }
    GateDecision::Deny(evidence) => {
        // Action blocked with witness receipt
        log::warn!("Denied: {:?}", evidence);
    }
}
```

```javascript
// Browser: Real-time safety checks via WASM
import { CognitumGate } from '@cognitum/gate';

const gate = await CognitumGate.init();

// Check action in <1ms
const result = await gate.evaluate({
  action: 'modify_user_data',
  agent: 'assistant-v3',
  context: { user_id: '12345' }
});

if (result.permitted) {
  const receipt = result.witnessReceipt;  // Hash-chained audit log
  console.log('Permit token:', result.token);
}
```

### Dynamic Embedding Fine-Tuning

| Use Case | Features Used | Example |
|----------|---------------|---------|
| **Real-Time Adaptation** | MicroLoRA (<1ms), Per-request learning | [docs/ruvllm/FINE_TUNING.md](./docs/ruvllm/FINE_TUNING.md) |
| **Contrastive Training** | Triplet loss, Hard negatives, InfoNCE | [npm/packages/ruvllm](./npm/packages/ruvllm) |
| **Task-Specific Adapters** | 5 pre-defined adapters (Coder, Researcher, Security, Architect, Reviewer) | [docs/training](./docs/training/) |
| **Catastrophic Forgetting Prevention** | EWC++ (Elastic Weight Consolidation) | [crates/sona](./crates/sona) |
| **Browser Fine-Tuning** | MicroLoRA WASM, <50KB adapters | [crates/ruvllm-wasm](./crates/ruvllm-wasm) |

**Three-Tier Adaptation System:**

| Tier | Technique | Latency | Use Case |
|------|-----------|---------|----------|
| **Instant** | MicroLoRA (rank 1-2) | <1ms | Per-request adaptation |
| **Background** | Adapter Merge + EWC++ | ~100ms | Pattern consolidation |
| **Deep** | Full Training Pipeline | Minutes | Periodic optimization |

```javascript
// Real-time embedding fine-tuning with contrastive learning
import { ContrastiveTrainer, tripletLoss } from '@ruvector/ruvllm';

const trainer = new ContrastiveTrainer({
  epochs: 10,
  batchSize: 16,
  margin: 0.5,           // Triplet loss margin
  hardNegativeRatio: 0.7 // 70% hard negatives for better learning
});

// Train with triplets: anchor (task) → positive (correct agent) → negative (wrong agent)
trainer.addTriplet(taskEmb, correctAgentEmb, wrongAgentEmb, isHardNegative);
const results = trainer.train();
trainer.exportTrainingData('./fine-tuned-model');
```

```rust
// Rust: MicroLoRA for per-request adaptation
use ruvllm::lora::{MicroLoRA, MicroLoraConfig, AdaptFeedback};

let lora = MicroLoRA::new(MicroLoraConfig::for_hidden_dim(4096));

// During inference: apply LoRA delta
let output = model.forward(&input)?;
let delta = lora.forward(&input, &TargetModule::QProj);
let enhanced = output.iter().zip(delta.iter()).map(|(o, d)| o + d).collect();

// After response: adapt based on quality feedback
lora.adapt(&input, AdaptFeedback::from_quality(0.85))?;
lora.apply_updates(0.01); // Learning rate
```

```javascript
// Browser: Real-time fine-tuning with MicroLoRA WASM
import init, { MicroLoraWasm, MicroLoraConfigWasm } from 'ruvllm-wasm';

await init();
const config = new MicroLoraConfigWasm();
config.rank = 2;           // Tiny rank for browser (<50KB)
config.alpha = 4.0;
config.inFeatures = 768;

const lora = new MicroLoraWasm(config);
const delta = lora.forward(hiddenStates);  // <1ms latency

// Persist to localStorage/IndexedDB
const json = lora.toJson();
localStorage.setItem('user-adapter', json);
```

### Agentic Workflows

| Use Case | Features Used | Example |
|----------|---------------|---------|
| **Version Control for AI** | Agentic Jujutsu, Branching | [examples/agentic-jujutsu](./examples/agentic-jujutsu) |
| **Data Pipelines** | DAG workflows, Self-learning | [crates/ruvector-dag](./crates/ruvector-dag) |
| **Web Scraping** | Apify integration, Embeddings | [examples/apify](./examples/apify) |
| **Synthetic Data** | Agentic synthesis, Generation | [Agentic-Flow](https://github.com/ruvnet/agentic-flow) |

```javascript
// Self-learning DAG workflow
import { QueryDag, AttentionSelector } from '@ruvector/dag';

const dag = new QueryDag();
dag.addNode({ type: 'fetch', source: 'api' });
dag.addNode({ type: 'embed', model: 'local-onnx' });
dag.addNode({ type: 'index', engine: 'hnsw' });

// DAG learns optimal execution paths over time
dag.enableSonaLearning();
await dag.execute();
```

</details>

## Installation

| Platform | Command |
|----------|---------|
| **npm** | `npm install ruvector` |
| **npm (SONA)** | `npm install @ruvector/sona` |
| **Browser/WASM** | `npm install ruvector-wasm` |
| **Rust** | `cargo add ruvector-core ruvector-graph ruvector-gnn` |
| **Rust (SONA)** | `cargo add ruvector-sona` |
| **Rust (LLM)** | `cargo add ruvllm` |

---

## Package Reference

<details>
<summary>📖 Documentation</summary>

#### Getting Started

| Topic | Link |
|-------|------|
| Getting Started | [docs/guides/GETTING_STARTED.md](./docs/guides/GETTING_STARTED.md) |
| API Reference | [docs/api/](./docs/api/) |
| Cypher Reference | [docs/api/CYPHER_REFERENCE.md](./docs/api/CYPHER_REFERENCE.md) |
| Performance Tuning | [docs/optimization/PERFORMANCE_TUNING_GUIDE.md](./docs/optimization/PERFORMANCE_TUNING_GUIDE.md) |

#### Core Components

| Topic | Link |
|-------|------|
| GNN Architecture | [docs/gnn/](./docs/gnn/) |
| HNSW Indexing | [docs/hnsw/](./docs/hnsw/) |
| DAG System | [docs/dag/](./docs/dag/) |
| Nervous System | [docs/nervous-system/](./docs/nervous-system/) |
| Sparse Inference | [docs/sparse-inference/](./docs/sparse-inference/) |

#### Bindings & Integration

| Topic | Link |
|-------|------|
| Node.js API | [crates/ruvector-gnn-node/README.md](./crates/ruvector-gnn-node/README.md) |
| WASM API | [crates/ruvector-gnn-wasm/README.md](./crates/ruvector-gnn-wasm/README.md) |
| PostgreSQL | [docs/postgres/](./docs/postgres/) |
| Self-Learning Hooks | [docs/hooks/](./docs/hooks/) |
| Integration Guides | [docs/integration/](./docs/integration/) |

#### LLM & AI

| Topic | Link |
|-------|------|
| RuvLLM | [docs/ruvllm/](./docs/ruvllm/) |
| Training Guides | [docs/training/](./docs/training/) |

#### Operations

| Topic | Link |
|-------|------|
| Architecture | [docs/architecture/](./docs/architecture/) |
| Cloud Deployment | [docs/cloud-architecture/](./docs/cloud-architecture/) |
| Security | [docs/security/](./docs/security/) |
| Benchmarks | [docs/benchmarks/](./docs/benchmarks/) |
| Testing | [docs/testing/](./docs/testing/) |

#### Research

| Topic | Link |
|-------|------|
| Research Papers | [docs/research/](./docs/research/) |
| GNN V2 Features | [docs/research/gnn-v2/](./docs/research/gnn-v2/) |
| Min-Cut Algorithms | [docs/research/mincut/](./docs/research/mincut/) |
| SPARQL Support | [docs/research/sparql/](./docs/research/sparql/) |
| Latent Space | [docs/research/latent-space/](./docs/research/latent-space/) |

### Architecture Decision Records (ADRs)

| ADR | Status | Description |
|-----|--------|-------------|
| [ADR-001](./docs/adr/ADR-001-ruvector-core-architecture.md) | Accepted | Core architecture design |
| [ADR-002](./docs/adr/ADR-002-ruvllm-integration.md) | Accepted | RuvLLM integration |
| [ADR-003](./docs/adr/ADR-003-simd-optimization-strategy.md) | Accepted | SIMD optimization strategy |
| [ADR-004](./docs/adr/ADR-004-kv-cache-management.md) | Accepted | KV cache management |
| [ADR-005](./docs/adr/ADR-005-wasm-runtime-integration.md) | Accepted | WASM runtime integration |
| [ADR-006](./docs/adr/ADR-006-memory-management.md) | Accepted | Memory management |
| [ADR-007](./docs/adr/ADR-007-security-review-technical-debt.md) | Accepted | Security review |
| [ADR-008](./docs/adr/ADR-008-mistral-rs-integration.md) | **New** | Mistral-rs backend integration |
| [ADR-009](./docs/adr/ADR-009-structured-output.md) | **New** | Structured output (SOTA) |
| [ADR-010](./docs/adr/ADR-010-function-calling.md) | **New** | Function calling (SOTA) |
| [ADR-011](./docs/adr/ADR-011-prefix-caching.md) | **New** | Prefix caching (SOTA) |
| [ADR-012](./docs/adr/ADR-012-security-remediation.md) | **New** | Security remediation |
| [ADR-013](./docs/adr/ADR-013-huggingface-publishing.md) | **New** | HuggingFace publishing |

</details>


<details>
<summary>📦 npm Packages (45+ Packages)</summary>

#### Core Packages

| Package | Description | Version | Downloads |
|---------|-------------|---------|-----------|
| [ruvector](https://www.npmjs.com/package/ruvector) | All-in-one CLI & package | [![npm](https://img.shields.io/npm/v/ruvector.svg)](https://www.npmjs.com/package/ruvector) | [![downloads](https://img.shields.io/npm/dt/ruvector.svg)](https://www.npmjs.com/package/ruvector) |
| [@ruvector/core](https://www.npmjs.com/package/@ruvector/core) | Core vector database with HNSW | [![npm](https://img.shields.io/npm/v/@ruvector/core.svg)](https://www.npmjs.com/package/@ruvector/core) | [![downloads](https://img.shields.io/npm/dt/@ruvector/core.svg)](https://www.npmjs.com/package/@ruvector/core) |
| [@ruvector/node](https://www.npmjs.com/package/@ruvector/node) | Unified Node.js bindings | [![npm](https://img.shields.io/npm/v/@ruvector/node.svg)](https://www.npmjs.com/package/@ruvector/node) | [![downloads](https://img.shields.io/npm/dt/@ruvector/node.svg)](https://www.npmjs.com/package/@ruvector/node) |
| [ruvector-extensions](https://www.npmjs.com/package/ruvector-extensions) | Advanced features: embeddings, UI | [![npm](https://img.shields.io/npm/v/ruvector-extensions.svg)](https://www.npmjs.com/package/ruvector-extensions) | [![downloads](https://img.shields.io/npm/dt/ruvector-extensions.svg)](https://www.npmjs.com/package/ruvector-extensions) |

#### Graph & GNN

| Package | Description | Version | Downloads |
|---------|-------------|---------|-----------|
| [@ruvector/gnn](https://www.npmjs.com/package/@ruvector/gnn) | Graph Neural Network layers | [![npm](https://img.shields.io/npm/v/@ruvector/gnn.svg)](https://www.npmjs.com/package/@ruvector/gnn) | [![downloads](https://img.shields.io/npm/dt/@ruvector/gnn.svg)](https://www.npmjs.com/package/@ruvector/gnn) |
| [@ruvector/graph-node](https://www.npmjs.com/package/@ruvector/graph-node) | Hypergraph with Cypher queries | [![npm](https://img.shields.io/npm/v/@ruvector/graph-node.svg)](https://www.npmjs.com/package/@ruvector/graph-node) | [![downloads](https://img.shields.io/npm/dt/@ruvector/graph-node.svg)](https://www.npmjs.com/package/@ruvector/graph-node) |
| [@ruvector/graph-wasm](https://www.npmjs.com/package/@ruvector/graph-wasm) | Browser graph queries | [![npm](https://img.shields.io/npm/v/@ruvector/graph-wasm.svg)](https://www.npmjs.com/package/@ruvector/graph-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/graph-wasm.svg)](https://www.npmjs.com/package/@ruvector/graph-wasm) |
| [@ruvector/graph-data-generator](https://www.npmjs.com/package/@ruvector/graph-data-generator) | AI-powered synthetic graph data | [![npm](https://img.shields.io/npm/v/@ruvector/graph-data-generator.svg)](https://www.npmjs.com/package/@ruvector/graph-data-generator) | [![downloads](https://img.shields.io/npm/dt/@ruvector/graph-data-generator.svg)](https://www.npmjs.com/package/@ruvector/graph-data-generator) |

#### AI Routing & Attention

| Package | Description | Version | Downloads |
|---------|-------------|---------|-----------|
| [@ruvector/tiny-dancer](https://www.npmjs.com/package/@ruvector/tiny-dancer) | FastGRNN neural routing | [![npm](https://img.shields.io/npm/v/@ruvector/tiny-dancer.svg)](https://www.npmjs.com/package/@ruvector/tiny-dancer) | [![downloads](https://img.shields.io/npm/dt/@ruvector/tiny-dancer.svg)](https://www.npmjs.com/package/@ruvector/tiny-dancer) |
| [@ruvector/router](https://www.npmjs.com/package/@ruvector/router) | Semantic router + HNSW | [![npm](https://img.shields.io/npm/v/@ruvector/router.svg)](https://www.npmjs.com/package/@ruvector/router) | [![downloads](https://img.shields.io/npm/dt/@ruvector/router.svg)](https://www.npmjs.com/package/@ruvector/router) |
| [@ruvector/attention](https://www.npmjs.com/package/@ruvector/attention) | 39 attention mechanisms | [![npm](https://img.shields.io/npm/v/@ruvector/attention.svg)](https://www.npmjs.com/package/@ruvector/attention) | [![downloads](https://img.shields.io/npm/dt/@ruvector/attention.svg)](https://www.npmjs.com/package/@ruvector/attention) |

#### Learning & Neural

| Package | Description | Version | Downloads |
|---------|-------------|---------|-----------|
| [@ruvector/sona](https://www.npmjs.com/package/@ruvector/sona) | Self-Optimizing Neural Architecture | [![npm](https://img.shields.io/npm/v/@ruvector/sona.svg)](https://www.npmjs.com/package/@ruvector/sona) | [![downloads](https://img.shields.io/npm/dt/@ruvector/sona.svg)](https://www.npmjs.com/package/@ruvector/sona) |
| [@ruvector/spiking-neural](https://www.npmjs.com/package/@ruvector/spiking-neural) | Spiking neural networks (SNN) | [![npm](https://img.shields.io/npm/v/@ruvector/spiking-neural.svg)](https://www.npmjs.com/package/@ruvector/spiking-neural) | [![downloads](https://img.shields.io/npm/dt/@ruvector/spiking-neural.svg)](https://www.npmjs.com/package/@ruvector/spiking-neural) |

#### LLM Runtime

| Package | Description | Version | Downloads |
|---------|-------------|---------|-----------|
| [@ruvector/ruvllm](https://www.npmjs.com/package/@ruvector/ruvllm) | LLM orchestration + SONA | [![npm](https://img.shields.io/npm/v/@ruvector/ruvllm.svg)](https://www.npmjs.com/package/@ruvector/ruvllm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/ruvllm.svg)](https://www.npmjs.com/package/@ruvector/ruvllm) |
| [@ruvector/ruvllm-cli](https://www.npmjs.com/package/@ruvector/ruvllm-cli) | LLM CLI: inference, benchmarks | [![npm](https://img.shields.io/npm/v/@ruvector/ruvllm-cli.svg)](https://www.npmjs.com/package/@ruvector/ruvllm-cli) | [![downloads](https://img.shields.io/npm/dt/@ruvector/ruvllm-cli.svg)](https://www.npmjs.com/package/@ruvector/ruvllm-cli) |
| [@ruvector/ruvllm-wasm](https://www.npmjs.com/package/@ruvector/ruvllm-wasm) | Browser LLM inference | [![npm](https://img.shields.io/npm/v/@ruvector/ruvllm-wasm.svg)](https://www.npmjs.com/package/@ruvector/ruvllm-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/ruvllm-wasm.svg)](https://www.npmjs.com/package/@ruvector/ruvllm-wasm) |

#### Distributed Systems

| Package | Description | Version | Downloads |
|---------|-------------|---------|-----------|
| [@ruvector/cluster](https://www.npmjs.com/package/@ruvector/cluster) | Distributed clustering | [![npm](https://img.shields.io/npm/v/@ruvector/cluster.svg)](https://www.npmjs.com/package/@ruvector/cluster) | [![downloads](https://img.shields.io/npm/dt/@ruvector/cluster.svg)](https://www.npmjs.com/package/@ruvector/cluster) |
| [@ruvector/server](https://www.npmjs.com/package/@ruvector/server) | HTTP/gRPC server | [![npm](https://img.shields.io/npm/v/@ruvector/server.svg)](https://www.npmjs.com/package/@ruvector/server) | [![downloads](https://img.shields.io/npm/dt/@ruvector/server.svg)](https://www.npmjs.com/package/@ruvector/server) |
| [@ruvector/raft](https://www.npmjs.com/package/@ruvector/raft) | Raft consensus | [![npm](https://img.shields.io/npm/v/@ruvector/raft.svg)](https://www.npmjs.com/package/@ruvector/raft) | [![downloads](https://img.shields.io/npm/dt/@ruvector/raft.svg)](https://www.npmjs.com/package/@ruvector/raft) |
| [@ruvector/replication](https://www.npmjs.com/package/@ruvector/replication) | Multi-master replication | [![npm](https://img.shields.io/npm/v/@ruvector/replication.svg)](https://www.npmjs.com/package/@ruvector/replication) | [![downloads](https://img.shields.io/npm/dt/@ruvector/replication.svg)](https://www.npmjs.com/package/@ruvector/replication) |
| [@ruvector/burst-scaling](https://www.npmjs.com/package/@ruvector/burst-scaling) | 10-50x burst scaling | [![npm](https://img.shields.io/npm/v/@ruvector/burst-scaling.svg)](https://www.npmjs.com/package/@ruvector/burst-scaling) | [![downloads](https://img.shields.io/npm/dt/@ruvector/burst-scaling.svg)](https://www.npmjs.com/package/@ruvector/burst-scaling) |

#### Edge & Standalone

| Package | Description | Version | Downloads |
|---------|-------------|---------|-----------|
| [rvlite](https://www.npmjs.com/package/rvlite) | SQLite-style edge DB | [![npm](https://img.shields.io/npm/v/rvlite.svg)](https://www.npmjs.com/package/rvlite) | [![downloads](https://img.shields.io/npm/dt/rvlite.svg)](https://www.npmjs.com/package/rvlite) |
| [@ruvector/rudag](https://www.npmjs.com/package/@ruvector/rudag) | Self-learning DAG | [![npm](https://img.shields.io/npm/v/@ruvector/rudag.svg)](https://www.npmjs.com/package/@ruvector/rudag) | [![downloads](https://img.shields.io/npm/dt/@ruvector/rudag.svg)](https://www.npmjs.com/package/@ruvector/rudag) |

#### Agentic & Synthetic Data

| Package | Description | Version | Downloads |
|---------|-------------|---------|-----------|
| [@ruvector/agentic-synth](https://www.npmjs.com/package/@ruvector/agentic-synth) | AI synthetic data generator | [![npm](https://img.shields.io/npm/v/@ruvector/agentic-synth.svg)](https://www.npmjs.com/package/@ruvector/agentic-synth) | [![downloads](https://img.shields.io/npm/dt/@ruvector/agentic-synth.svg)](https://www.npmjs.com/package/@ruvector/agentic-synth) |
| [@ruvector/agentic-integration](https://www.npmjs.com/package/@ruvector/agentic-integration) | Distributed agent coordination | [![npm](https://img.shields.io/npm/v/@ruvector/agentic-integration.svg)](https://www.npmjs.com/package/@ruvector/agentic-integration) | [![downloads](https://img.shields.io/npm/dt/@ruvector/agentic-integration.svg)](https://www.npmjs.com/package/@ruvector/agentic-integration) |
| [@cognitum/gate](https://www.npmjs.com/package/@cognitum/gate) | AI coherence gate | [![npm](https://img.shields.io/npm/v/@cognitum/gate.svg)](https://www.npmjs.com/package/@cognitum/gate) | [![downloads](https://img.shields.io/npm/dt/@cognitum/gate.svg)](https://www.npmjs.com/package/@cognitum/gate) |

#### CLI Tools

| Package | Description | Version | Downloads |
|---------|-------------|---------|-----------|
| [@ruvector/cli](https://www.npmjs.com/package/@ruvector/cli) | CLI + self-learning hooks | [![npm](https://img.shields.io/npm/v/@ruvector/cli.svg)](https://www.npmjs.com/package/@ruvector/cli) | [![downloads](https://img.shields.io/npm/dt/@ruvector/cli.svg)](https://www.npmjs.com/package/@ruvector/cli) |
| [@ruvector/postgres-cli](https://www.npmjs.com/package/@ruvector/postgres-cli) | PostgreSQL extension CLI | [![npm](https://img.shields.io/npm/v/@ruvector/postgres-cli.svg)](https://www.npmjs.com/package/@ruvector/postgres-cli) | [![downloads](https://img.shields.io/npm/dt/@ruvector/postgres-cli.svg)](https://www.npmjs.com/package/@ruvector/postgres-cli) |
| [@ruvector/scipix](https://www.npmjs.com/package/@ruvector/scipix) | Scientific OCR client | [![npm](https://img.shields.io/npm/v/@ruvector/scipix.svg)](https://www.npmjs.com/package/@ruvector/scipix) | [![downloads](https://img.shields.io/npm/dt/@ruvector/scipix.svg)](https://www.npmjs.com/package/@ruvector/scipix) |

#### WASM Packages

| Package | Description | Version | Downloads |
|---------|-------------|---------|-----------|
| [@ruvector/wasm](https://www.npmjs.com/package/@ruvector/wasm) | Unified WASM meta-package | [![npm](https://img.shields.io/npm/v/@ruvector/wasm.svg)](https://www.npmjs.com/package/@ruvector/wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/wasm.svg)](https://www.npmjs.com/package/@ruvector/wasm) |
| [@ruvector/wasm-unified](https://www.npmjs.com/package/@ruvector/wasm-unified) | Unified TypeScript API | [![npm](https://img.shields.io/npm/v/@ruvector/wasm-unified.svg)](https://www.npmjs.com/package/@ruvector/wasm-unified) | [![downloads](https://img.shields.io/npm/dt/@ruvector/wasm-unified.svg)](https://www.npmjs.com/package/@ruvector/wasm-unified) |
| [@ruvector/gnn-wasm](https://www.npmjs.com/package/@ruvector/gnn-wasm) | GNN WASM bindings | [![npm](https://img.shields.io/npm/v/@ruvector/gnn-wasm.svg)](https://www.npmjs.com/package/@ruvector/gnn-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/gnn-wasm.svg)](https://www.npmjs.com/package/@ruvector/gnn-wasm) |
| [@ruvector/attention-wasm](https://www.npmjs.com/package/@ruvector/attention-wasm) | Attention WASM bindings | [![npm](https://img.shields.io/npm/v/@ruvector/attention-wasm.svg)](https://www.npmjs.com/package/@ruvector/attention-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/attention-wasm.svg)](https://www.npmjs.com/package/@ruvector/attention-wasm) |
| [@ruvector/attention-unified-wasm](https://www.npmjs.com/package/@ruvector/attention-unified-wasm) | All 39 attention mechanisms | [![npm](https://img.shields.io/npm/v/@ruvector/attention-unified-wasm.svg)](https://www.npmjs.com/package/@ruvector/attention-unified-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/attention-unified-wasm.svg)](https://www.npmjs.com/package/@ruvector/attention-unified-wasm) |
| [@ruvector/tiny-dancer-wasm](https://www.npmjs.com/package/@ruvector/tiny-dancer-wasm) | AI routing WASM | [![npm](https://img.shields.io/npm/v/@ruvector/tiny-dancer-wasm.svg)](https://www.npmjs.com/package/@ruvector/tiny-dancer-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/tiny-dancer-wasm.svg)](https://www.npmjs.com/package/@ruvector/tiny-dancer-wasm) |
| [@ruvector/router-wasm](https://www.npmjs.com/package/@ruvector/router-wasm) | Semantic router WASM | [![npm](https://img.shields.io/npm/v/@ruvector/router-wasm.svg)](https://www.npmjs.com/package/@ruvector/router-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/router-wasm.svg)](https://www.npmjs.com/package/@ruvector/router-wasm) |
| [@ruvector/learning-wasm](https://www.npmjs.com/package/@ruvector/learning-wasm) | Learning module WASM | [![npm](https://img.shields.io/npm/v/@ruvector/learning-wasm.svg)](https://www.npmjs.com/package/@ruvector/learning-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/learning-wasm.svg)](https://www.npmjs.com/package/@ruvector/learning-wasm) |
| [@ruvector/economy-wasm](https://www.npmjs.com/package/@ruvector/economy-wasm) | Tokenomics WASM | [![npm](https://img.shields.io/npm/v/@ruvector/economy-wasm.svg)](https://www.npmjs.com/package/@ruvector/economy-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/economy-wasm.svg)](https://www.npmjs.com/package/@ruvector/economy-wasm) |
| [@ruvector/exotic-wasm](https://www.npmjs.com/package/@ruvector/exotic-wasm) | Exotic features WASM | [![npm](https://img.shields.io/npm/v/@ruvector/exotic-wasm.svg)](https://www.npmjs.com/package/@ruvector/exotic-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/exotic-wasm.svg)](https://www.npmjs.com/package/@ruvector/exotic-wasm) |
| [@ruvector/nervous-system-wasm](https://www.npmjs.com/package/@ruvector/nervous-system-wasm) | Nervous system WASM | [![npm](https://img.shields.io/npm/v/@ruvector/nervous-system-wasm.svg)](https://www.npmjs.com/package/@ruvector/nervous-system-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/nervous-system-wasm.svg)](https://www.npmjs.com/package/@ruvector/nervous-system-wasm) |

</details>

<details>
<summary>🦀 Rust Crates (63 Packages)</summary>

All crates are published to [crates.io](https://crates.io) under the `ruvector-*` namespace.

### Core Crates

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-core](./crates/ruvector-core) | Vector database engine with HNSW indexing | [![crates.io](https://img.shields.io/crates/v/ruvector-core.svg)](https://crates.io/crates/ruvector-core) |
| [ruvector-collections](./crates/ruvector-collections) | Collection and namespace management | [![crates.io](https://img.shields.io/crates/v/ruvector-collections.svg)](https://crates.io/crates/ruvector-collections) |
| [ruvector-filter](./crates/ruvector-filter) | Vector filtering and metadata queries | [![crates.io](https://img.shields.io/crates/v/ruvector-filter.svg)](https://crates.io/crates/ruvector-filter) |
| [ruvector-metrics](./crates/ruvector-metrics) | Performance metrics and monitoring | [![crates.io](https://img.shields.io/crates/v/ruvector-metrics.svg)](https://crates.io/crates/ruvector-metrics) |
| [ruvector-snapshot](./crates/ruvector-snapshot) | Snapshot and persistence management | [![crates.io](https://img.shields.io/crates/v/ruvector-snapshot.svg)](https://crates.io/crates/ruvector-snapshot) |
| [ruvector-node](./crates/ruvector-node) | Node.js bindings via NAPI-RS | [![crates.io](https://img.shields.io/crates/v/ruvector-node.svg)](https://crates.io/crates/ruvector-node) |
| [ruvector-wasm](./crates/ruvector-wasm) | WASM bindings for browser/edge | [![crates.io](https://img.shields.io/crates/v/ruvector-wasm.svg)](https://crates.io/crates/ruvector-wasm) |
| [ruvector-cli](./crates/ruvector-cli) | CLI and MCP server | [![crates.io](https://img.shields.io/crates/v/ruvector-cli.svg)](https://crates.io/crates/ruvector-cli) |
| [ruvector-bench](./crates/ruvector-bench) | Benchmarking suite | [![crates.io](https://img.shields.io/crates/v/ruvector-bench.svg)](https://crates.io/crates/ruvector-bench) |

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

### LLM Runtime (ruvllm)

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvllm](./crates/ruvllm) | LLM serving runtime with SONA, paged attention, KV cache | [![crates.io](https://img.shields.io/crates/v/ruvllm.svg)](https://crates.io/crates/ruvllm) |
| [ruvllm-cli](./crates/ruvllm-cli) | CLI for model inference and benchmarking | [![crates.io](https://img.shields.io/crates/v/ruvllm-cli.svg)](https://crates.io/crates/ruvllm-cli) |
| [ruvllm-wasm](./crates/ruvllm-wasm) | WASM bindings for browser LLM inference | [![crates.io](https://img.shields.io/crates/v/ruvllm-wasm.svg)](https://crates.io/crates/ruvllm-wasm) |

**Features:** Candle backend, Metal/CUDA acceleration, Apple Neural Engine, GGUF support, SONA learning integration.

```bash
cargo add ruvllm --features inference-metal  # Mac with Metal
cargo add ruvllm --features inference-cuda   # NVIDIA GPU
```

**RuvLTRA Models** — Pre-trained GGUF models optimized for Claude Code workflows:

| Model | Size | Use Case | Link |
|-------|------|----------|------|
| ruvltra-claude-code-0.5b-q4_k_m | 398 MB | Agent routing | [HuggingFace](https://huggingface.co/ruv/ruvltra) |
| ruvltra-small-0.5b-q4_k_m | 398 MB | Embeddings | [HuggingFace](https://huggingface.co/ruv/ruvltra) |
| ruvltra-medium-1.1b-q4_k_m | 800 MB | Classification | [HuggingFace](https://huggingface.co/ruv/ruvltra) |

```bash
# Download and use
wget https://huggingface.co/ruv/ruvltra/resolve/main/ruvltra-small-0.5b-q4_k_m.gguf
```


### Distributed Systems

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-cluster](./crates/ruvector-cluster) | Cluster management and coordination | [![crates.io](https://img.shields.io/crates/v/ruvector-cluster.svg)](https://crates.io/crates/ruvector-cluster) |
| [ruvector-raft](./crates/ruvector-raft) | Raft consensus implementation | [![crates.io](https://img.shields.io/crates/v/ruvector-raft.svg)](https://crates.io/crates/ruvector-raft) |
| [ruvector-replication](./crates/ruvector-replication) | Data replication and synchronization | [![crates.io](https://img.shields.io/crates/v/ruvector-replication.svg)](https://crates.io/crates/ruvector-replication) |
| [ruvector-server](./crates/ruvector-server) | REST/gRPC API server | [![crates.io](https://img.shields.io/crates/v/ruvector-server.svg)](https://crates.io/crates/ruvector-server) |

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

**Hybrid Routing** achieves **90% accuracy** for agent routing using keyword-first strategy with embedding fallback. See [Issue #122](https://github.com/ruvnet/ruvector/issues/122) for benchmarks and the [training tutorials](#-ruvllm-training--fine-tuning-tutorials) for fine-tuning guides.

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

### Quantum Coherence (ruQu)

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruqu](./crates/ruQu) | Classical nervous system for quantum machines - coherence via min-cut | [![crates.io](https://img.shields.io/crates/v/ruqu.svg)](https://crates.io/crates/ruqu) |
| [cognitum-gate-kernel](./crates/cognitum-gate-kernel) | Anytime-valid coherence gate kernel | [![crates.io](https://img.shields.io/crates/v/cognitum-gate-kernel.svg)](https://crates.io/crates/cognitum-gate-kernel) |
| [cognitum-gate-tilezero](./crates/cognitum-gate-tilezero) | TileZero arbiter for coherence decisions | [![crates.io](https://img.shields.io/crates/v/cognitum-gate-tilezero.svg)](https://crates.io/crates/cognitum-gate-tilezero) |
| [mcp-gate](./crates/mcp-gate) | MCP server for coherence gate integration | [![crates.io](https://img.shields.io/crates/v/mcp-gate.svg)](https://crates.io/crates/mcp-gate) |

**ruQu Features:** Real-time quantum coherence assessment, MWPM decoder integration, mincut-gated attention (50% FLOPs reduction).

```rust
use ruqu::{CoherenceGate, SyndromeFilter};

let gate = CoherenceGate::new();
let syndrome = gate.assess_coherence(&quantum_state)?;
```

### Advanced Math & Inference

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-math](./crates/ruvector-math) | Core math utilities, SIMD operations | [![crates.io](https://img.shields.io/crates/v/ruvector-math.svg)](https://crates.io/crates/ruvector-math) |
| [ruvector-math-wasm](./crates/ruvector-math-wasm) | WASM bindings for math operations | [![crates.io](https://img.shields.io/crates/v/ruvector-math-wasm.svg)](https://crates.io/crates/ruvector-math-wasm) |
| [ruvector-sparse-inference](./crates/ruvector-sparse-inference) | Sparse tensor inference engine | [![crates.io](https://img.shields.io/crates/v/ruvector-sparse-inference.svg)](https://crates.io/crates/ruvector-sparse-inference) |
| [ruvector-sparse-inference-wasm](./crates/ruvector-sparse-inference-wasm) | WASM bindings for sparse inference | [![crates.io](https://img.shields.io/crates/v/ruvector-sparse-inference-wasm.svg)](https://crates.io/crates/ruvector-sparse-inference-wasm) |
| [ruvector-hyperbolic-hnsw](./crates/ruvector-hyperbolic-hnsw) | HNSW in hyperbolic space (Poincaré/Lorentz) | [![crates.io](https://img.shields.io/crates/v/ruvector-hyperbolic-hnsw.svg)](https://crates.io/crates/ruvector-hyperbolic-hnsw) |
| [ruvector-hyperbolic-hnsw-wasm](./crates/ruvector-hyperbolic-hnsw-wasm) | WASM bindings for hyperbolic HNSW | [![crates.io](https://img.shields.io/crates/v/ruvector-hyperbolic-hnsw-wasm.svg)](https://crates.io/crates/ruvector-hyperbolic-hnsw-wasm) |

### FPGA & Hardware Acceleration

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-fpga-transformer](./crates/ruvector-fpga-transformer) | FPGA-optimized transformer inference | [![crates.io](https://img.shields.io/crates/v/ruvector-fpga-transformer.svg)](https://crates.io/crates/ruvector-fpga-transformer) |
| [ruvector-fpga-transformer-wasm](./crates/ruvector-fpga-transformer-wasm) | WASM simulation of FPGA transformer | [![crates.io](https://img.shields.io/crates/v/ruvector-fpga-transformer-wasm.svg)](https://crates.io/crates/ruvector-fpga-transformer-wasm) |
| [ruvector-mincut-gated-transformer](./crates/ruvector-mincut-gated-transformer) | MinCut-gated attention for 50% compute reduction | [![crates.io](https://img.shields.io/crates/v/ruvector-mincut-gated-transformer.svg)](https://crates.io/crates/ruvector-mincut-gated-transformer) |
| [ruvector-mincut-gated-transformer-wasm](./crates/ruvector-mincut-gated-transformer-wasm) | WASM bindings for mincut-gated transformer | [![crates.io](https://img.shields.io/crates/v/ruvector-mincut-gated-transformer-wasm.svg)](https://crates.io/crates/ruvector-mincut-gated-transformer-wasm) |

### Neuromorphic & Bio-Inspired Learning

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-nervous-system](./crates/ruvector-nervous-system) | Spiking neural networks with BTSP learning & EWC plasticity | [![crates.io](https://img.shields.io/crates/v/ruvector-nervous-system.svg)](https://crates.io/crates/ruvector-nervous-system) |
| [ruvector-nervous-system-wasm](./crates/ruvector-nervous-system-wasm) | WASM bindings for neuromorphic learning | [![crates.io](https://img.shields.io/crates/v/ruvector-nervous-system-wasm.svg)](https://crates.io/crates/ruvector-nervous-system-wasm) |
| [ruvector-learning-wasm](./crates/ruvector-learning-wasm) | MicroLoRA adaptation (<100µs latency) | [![crates.io](https://img.shields.io/crates/v/ruvector-learning-wasm.svg)](https://crates.io/crates/ruvector-learning-wasm) |
| [ruvector-economy-wasm](./crates/ruvector-economy-wasm) | CRDT-based autonomous credit economy | [![crates.io](https://img.shields.io/crates/v/ruvector-economy-wasm.svg)](https://crates.io/crates/ruvector-economy-wasm) |
| [ruvector-exotic-wasm](./crates/ruvector-exotic-wasm) | Exotic AI primitives (strange loops, time crystals) | [![crates.io](https://img.shields.io/crates/v/ruvector-exotic-wasm.svg)](https://crates.io/crates/ruvector-exotic-wasm) |
| [ruvector-attention-unified-wasm](./crates/ruvector-attention-unified-wasm) | Unified 18+ attention mechanisms (Neural, DAG, Mamba SSM) | [![crates.io](https://img.shields.io/crates/v/ruvector-attention-unified-wasm.svg)](https://crates.io/crates/ruvector-attention-unified-wasm) |
| [micro-hnsw-wasm](./crates/micro-hnsw-wasm) | Neuromorphic HNSW with spiking neurons (11.8KB WASM) | [![crates.io](https://img.shields.io/crates/v/micro-hnsw-wasm.svg)](https://crates.io/crates/micro-hnsw-wasm) |

**Bio-inspired features:**
- **Spiking Neural Networks (SNNs)** — 10-50x energy efficiency vs traditional ANNs
- **BTSP Learning** — Behavioral Time-Scale Synaptic Plasticity for rapid adaptation
- **MicroLoRA** — Sub-microsecond fine-tuning for per-operator learning
- **Mamba SSM** — State Space Model attention for linear-time sequences

### Self-Learning (SONA)

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [sona](./crates/sona) | Self-Optimizing Neural Architecture - LoRA, EWC++, ReasoningBank | [![crates.io](https://img.shields.io/crates/v/ruvector-sona.svg)](https://crates.io/crates/ruvector-sona) |

**SONA Features:** Two-tier LoRA adaptation, Elastic Weight Consolidation (EWC++), ReasoningBank for trajectory learning, runtime-adaptive learning for LLM routers.

### Standalone Edge Database (rvLite)

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [rvlite](./crates/rvlite) | Standalone 2MB edge database with SQL, SPARQL, Cypher | [![crates.io](https://img.shields.io/crates/v/rvlite.svg)](https://crates.io/crates/rvlite) |

**rvLite Features:** Powered by RuVector WASM, supports SQL/SPARQL/Cypher queries, ideal for IoT, mobile, and embedded systems.

### PostgreSQL Extension

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-postgres](./crates/ruvector-postgres) | pgvector replacement with 230+ SQL functions, SIMD, Flash Attention | [![crates.io](https://img.shields.io/crates/v/ruvector-postgres.svg)](https://crates.io/crates/ruvector-postgres) |

**PostgreSQL Features:** Drop-in pgvector replacement, GNN layers, hybrid search, multi-tenancy, self-healing, self-learning capabilities.

```bash
docker pull ruvnet/ruvector-postgres    # Docker image
cargo add ruvector-postgres             # Rust crate
```

### Self-Learning Query DAG (ruvector-dag)

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-dag](./crates/ruvector-dag) | Neural self-learning DAG for automatic query optimization | [![crates.io](https://img.shields.io/crates/v/ruvector-dag.svg)](https://crates.io/crates/ruvector-dag) |
| [ruvector-dag-wasm](./crates/ruvector-dag-wasm) | WASM bindings for browser DAG optimization (58KB gzipped) | [![crates.io](https://img.shields.io/crates/v/ruvector-dag-wasm.svg)](https://crates.io/crates/ruvector-dag-wasm) |

**Make your queries faster automatically.** RuVector DAG learns from every query execution and continuously optimizes performance—no manual tuning required.

- **7 Attention Mechanisms**: Automatically selects the best strategy (Topological, Causal Cone, Critical Path, MinCut Gated, etc.)
- **SONA Learning**: Self-Optimizing Neural Architecture adapts in <100μs per query
- **MinCut Control**: Rising "tension" triggers automatic strategy switching and predictive healing
- **50-80% Latency Reduction**: Queries improve over time without code changes

```rust
use ruvector_dag::{QueryDag, OperatorNode};
use ruvector_dag::attention::{AttentionSelector, SelectionPolicy};

let mut dag = QueryDag::new();
let scan = dag.add_node(OperatorNode::hnsw_scan(0, "vectors_idx", 64));
let filter = dag.add_node(OperatorNode::filter(1, "score > 0.5"));
dag.add_edge(scan, filter).unwrap();

// System learns which attention mechanism works best
let selector = AttentionSelector::new();
let scores = selector.select_and_apply(SelectionPolicy::Adaptive, &dag)?;
```

See [ruvector-dag README](./crates/ruvector-dag/README.md) for full documentation.

### Distributed Systems (Raft & Replication)

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-raft](./crates/ruvector-raft) | Raft consensus with leader election & log replication | [![crates.io](https://img.shields.io/crates/v/ruvector-raft.svg)](https://crates.io/crates/ruvector-raft) |
| [ruvector-replication](./crates/ruvector-replication) | Multi-master replication with vector clocks | [![crates.io](https://img.shields.io/crates/v/ruvector-replication.svg)](https://crates.io/crates/ruvector-replication) |
| [ruvector-cluster](./crates/ruvector-cluster) | Cluster coordination and sharding | [![crates.io](https://img.shields.io/crates/v/ruvector-cluster.svg)](https://crates.io/crates/ruvector-cluster) |

**Build distributed vector databases** with strong consistency guarantees:

- **Raft Consensus** — Leader election, log replication, automatic failover
- **Vector Clocks** — Causal ordering for conflict detection
- **Conflict Resolution** — Last-Write-Wins, custom merge functions, CRDT support
- **Change Data Capture** — Stream changes to replicas in real-time
- **Automatic Failover** — Promote replicas on primary failure

```typescript
import { RaftNode, ReplicaSet, VectorClock } from '@ruvector/raft';
import { ReplicationManager, ConflictStrategy } from '@ruvector/replication';

// Raft consensus cluster
const node = new RaftNode({
  nodeId: 'node-1',
  peers: ['node-2', 'node-3'],
  electionTimeout: [150, 300],
});

await node.start();
const entry = await node.propose({ op: 'insert', vector: embedding });

// Multi-master replication
const replicaSet = new ReplicaSet();
replicaSet.addReplica('primary', 'localhost:5001', 'primary');
replicaSet.addReplica('replica-1', 'localhost:5002', 'replica');

const manager = new ReplicationManager(replicaSet, {
  conflictStrategy: ConflictStrategy.LastWriteWins,
  syncMode: 'async',
});

await manager.write('vectors', { id: 'v1', data: embedding });
```

See [npm/packages/raft/README.md](./npm/packages/raft/README.md) and [npm/packages/replication/README.md](./npm/packages/replication/README.md) for full documentation.

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

</details>

---

## Platform Features

<details>
<summary><strong>🔀 Self-Learning DAG (Query Optimization)</strong></summary>

[![crates.io](https://img.shields.io/crates/v/ruvector-dag.svg)](https://crates.io/crates/ruvector-dag)
[![npm](https://img.shields.io/npm/v/@ruvector/rudag.svg)](https://www.npmjs.com/package/@ruvector/rudag)

**Make your queries faster automatically.** RuVector DAG learns from every query execution and continuously optimizes performance—no manual tuning required.

### What is RuVector DAG?

A **self-learning query optimization system**—like a "nervous system" for your database queries that:

1. **Watches** how queries execute and identifies bottlenecks
2. **Learns** which optimization strategies work best for different query patterns
3. **Adapts** in real-time, switching strategies when conditions change
4. **Heals** itself by detecting anomalies and fixing problems before they impact users

Unlike traditional query optimizers that use static rules, RuVector DAG learns from actual execution patterns and gets smarter over time.

### Key Benefits

| Benefit | What It Does | Result |
|---------|--------------|--------|
| **Automatic Improvement** | Queries get faster without code changes | **50-80% latency reduction** after learning |
| **Zero-Downtime Adaptation** | Adapts to pattern changes automatically | No manual index rebuilds |
| **Predictive Prevention** | Detects rising "tension" early | Intervenes *before* slowdowns |
| **Works Everywhere** | PostgreSQL, Browser (58KB WASM), Embedded | Universal deployment |

### Use Cases

| Use Case | Why RuVector DAG Helps |
|----------|------------------------|
| **Vector Search Applications** | Optimize similarity searches that traditional databases struggle with |
| **High-Traffic APIs** | Automatically adapt to changing query patterns throughout the day |
| **Real-Time Analytics** | Learn which aggregation paths are fastest for your specific data |
| **Edge/Embedded Systems** | 58KB WASM build runs in browsers and IoT devices |
| **Multi-Tenant Platforms** | Learn per-tenant query patterns without manual tuning |

### How It Works

```
Query comes in → DAG analyzes execution plan → Best attention mechanism selected
                                                          ↓
Query executes → Results returned → Learning system records what worked
                                                          ↓
                    Next similar query benefits from learned optimizations
```

The system maintains a **MinCut tension** score that acts as a health indicator. When tension rises, the system automatically switches to more aggressive optimization strategies and triggers predictive healing.

### 7 DAG Attention Mechanisms

| Mechanism | When to Use | Trigger |
|-----------|-------------|---------|
| **Topological** | Default baseline | Low variance |
| **Causal Cone** | Downstream impact analysis | Write-heavy patterns |
| **Critical Path** | Latency-bound queries | p99 > 2x p50 |
| **MinCut Gated** | Bottleneck-aware weighting | Cut tension rising |
| **Hierarchical Lorentz** | Deep hierarchical queries | Depth > 10 |
| **Parallel Branch** | Wide parallel execution | Branch count > 3 |
| **Temporal BTSP** | Time-series workloads | Temporal patterns |

### Quick Start

**Rust:**
```rust
use ruvector_dag::{QueryDag, OperatorNode, OperatorType};
use ruvector_dag::attention::{TopologicalAttention, DagAttention};

// Build a query DAG
let mut dag = QueryDag::new();
let scan = dag.add_node(OperatorNode::hnsw_scan(0, "vectors_idx", 64));
let filter = dag.add_node(OperatorNode::filter(1, "score > 0.5"));
let result = dag.add_node(OperatorNode::new(2, OperatorType::Result));

dag.add_edge(scan, filter).unwrap();
dag.add_edge(filter, result).unwrap();

// Compute attention scores
let attention = TopologicalAttention::new(Default::default());
let scores = attention.forward(&dag).unwrap();
```

**Node.js:**
```javascript
import { QueryDag, TopologicalAttention } from '@ruvector/rudag';

// Build DAG
const dag = new QueryDag();
const scan = dag.addNode({ type: 'hnsw_scan', table: 'vectors', k: 64 });
const filter = dag.addNode({ type: 'filter', condition: 'score > 0.5' });
dag.addEdge(scan, filter);

// Apply attention
const attention = new TopologicalAttention();
const scores = attention.forward(dag);
console.log('Attention scores:', scores);
```

**Browser (WASM - 58KB):**
```html
<script type="module">
import init, { QueryDag, TopologicalAttention } from '@ruvector/rudag-wasm';

await init();
const dag = new QueryDag();
// ... same API as Node.js
</script>
```

### SONA Learning Integration

SONA (Self-Optimizing Neural Architecture) runs post-query in background, never blocking execution:

```rust
use ruvector_dag::sona::{DagSonaEngine, SonaConfig};

let config = SonaConfig {
    embedding_dim: 256,
    lora_rank: 2,           // Rank-2 for <100μs updates
    ewc_lambda: 5000.0,     // Catastrophic forgetting prevention
    trajectory_capacity: 10_000,
};
let mut sona = DagSonaEngine::new(config);

// Pre-query: Get enhanced embedding (fast path)
let enhanced = sona.pre_query(&dag);

// Execute query...
let execution_time = execute_query(&dag);

// Post-query: Record trajectory (async, background)
sona.post_query(&dag, execution_time, baseline_time, "topological");
```

### Self-Healing

Reactive (Z-score anomaly detection) + Predictive (rising MinCut tension triggers early intervention):

```rust
use ruvector_dag::healing::{HealingOrchestrator, AnomalyConfig, PredictiveConfig};

let mut orchestrator = HealingOrchestrator::new();

// Reactive: Z-score anomaly detection
orchestrator.add_detector("query_latency", AnomalyConfig {
    z_threshold: 3.0,
    window_size: 100,
    min_samples: 10,
});

// Predictive: Rising cut tension triggers early intervention
orchestrator.enable_predictive(PredictiveConfig {
    tension_threshold: 0.6,    // Intervene before 0.7 crisis
    variance_threshold: 1.5,   // Rising variance = trouble coming
    lookahead_window: 50,      // Predict 50 queries ahead
});
```

### Query Convergence Example

A slow query converges over several runs:

```text
[run 1] query: SELECT * FROM vectors WHERE embedding <-> $1 < 0.5
        attention: topological (default)
        mincut_tension: 0.23
        latency: 847ms (improvement: 0.4%)

[run 4] mincut_tension: 0.71 > 0.7 (THRESHOLD)
        --> switching attention: topological -> mincut_gated
        latency: 412ms (improvement: 51.5%)

[run 10] attention: mincut_gated
         mincut_tension: 0.22 (stable)
         latency: 156ms (improvement: 81.6%)
```

### Performance Targets

| Component | Target | Notes |
|-----------|--------|-------|
| Attention (100 nodes) | <100μs | All 7 mechanisms |
| MicroLoRA adaptation | <100μs | Rank-2, per-operator |
| Pattern search (10K) | <2ms | K-means++ indexing |
| MinCut update | O(n^0.12) | Subpolynomial amortized |
| Anomaly detection | <50μs | Z-score, streaming |
| WASM size | 58KB | Gzipped, browser-ready |

### Installation

```bash
# Rust
cargo add ruvector-dag

# Node.js
npm install @ruvector/rudag

# WASM (browser)
npm install @ruvector/rudag-wasm
```

> **Full Documentation**: [ruvector-dag README](./crates/ruvector-dag/README.md)

</details>

<details>
<summary><strong>📦 rvLite - Standalone Edge Database</strong></summary>

[![crates.io](https://img.shields.io/crates/v/rvlite.svg)](https://crates.io/crates/rvlite)
[![npm](https://img.shields.io/npm/v/@ruvector/rvlite.svg)](https://www.npmjs.com/package/@ruvector/rvlite)
[![downloads](https://img.shields.io/npm/dt/@ruvector/rvlite.svg)](https://www.npmjs.com/package/@ruvector/rvlite)

**A complete vector database that runs anywhere JavaScript runs** — browsers, Node.js, Deno, Bun, Cloudflare Workers, Vercel Edge Functions.

### What is rvLite?

rvLite is a **lightweight, standalone vector database** that runs entirely in WebAssembly. It provides SQL, SPARQL, and Cypher query interfaces, along with graph neural networks and self-learning capabilities—all in under 3MB.

### Key Features

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **SQL Interface** | Familiar `SELECT`, `INSERT`, `WHERE` | No learning curve |
| **SPARQL Support** | W3C-compliant RDF queries | Knowledge graphs in browser |
| **Cypher Queries** | Neo4j-style graph queries | Graph traversals anywhere |
| **GNN Embeddings** | Graph neural network layers | Self-learning search |
| **ReasoningBank** | Trajectory learning | Gets smarter over time |
| **SIMD Optimized** | Vector operations accelerated | Native-like performance |

### Runs Everywhere

| Platform | Status | Use Case |
|----------|--------|----------|
| **Browsers** | ✅ | Offline-first apps |
| **Node.js** | ✅ | Server-side |
| **Deno** | ✅ | Edge functions |
| **Bun** | ✅ | Fast runtime |
| **Cloudflare Workers** | ✅ | Edge computing |
| **Vercel Edge** | ✅ | Serverless |

### Architecture

```
┌─────────────────────────────────────────┐
│  RvLite (Orchestration)                 │
│  ├─ SQL executor                        │
│  ├─ SPARQL executor                     │
│  ├─ Cypher executor                     │
│  └─ Unified WASM API                    │
└──────────────┬──────────────────────────┘
               │ depends on (100% reuse)
               ▼
┌──────────────────────────────────────────┐
│  Existing WASM Crates                    │
├──────────────────────────────────────────┤
│  • ruvector-core (vectors, SIMD)         │
│  • ruvector-graph-wasm (Cypher)          │
│  • ruvector-gnn-wasm (GNN layers)        │
│  • sona (ReasoningBank learning)         │
│  • micro-hnsw-wasm (ultra-fast HNSW)     │
└──────────────────────────────────────────┘
```

### Quick Start

```typescript
import { RvLite } from '@ruvector/rvlite';

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

### Size Budget

| Component | Size | Purpose |
|-----------|------|---------|
| ruvector-core | ~500KB | Vectors, SIMD |
| SQL parser | ~200KB | Query parsing |
| SPARQL executor | ~300KB | RDF queries |
| Cypher (graph-wasm) | ~600KB | Graph queries |
| GNN layers | ~300KB | Neural networks |
| ReasoningBank (sona) | ~300KB | Self-learning |
| **Total** | **~2.3MB** | Gzipped |

### Installation

```bash
# npm
npm install @ruvector/rvlite

# Rust
cargo add rvlite

# Build WASM
wasm-pack build --target web --release
```

> **Full Documentation**: [rvlite README](./crates/rvlite/README.md)

</details>

<details>
<summary><strong>🌐 Edge-Net - Collective AI Computing Network</strong></summary>

[![npm](https://img.shields.io/npm/v/@ruvector/edge-net.svg)](https://www.npmjs.com/package/@ruvector/edge-net)

**Share, Contribute, Compute Together** — A distributed computing platform that enables collective resource sharing for AI workloads.

### What is Edge-Net?

Edge-Net creates a **collective computing network** where participants share idle browser resources to power distributed AI workloads:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              EDGE-NET: COLLECTIVE AI COMPUTING NETWORK                      │
├─────────────────────────────────────────────────────────────────────────────┤
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐              │
│   │  Your       │       │  Collective │       │  AI Tasks   │              │
│   │  Browser    │◄─────►│  Network    │◄─────►│  Completed  │              │
│   │  (Idle CPU) │  P2P  │  (1000s)    │       │  for You    │              │
│   └─────────────┘       └─────────────┘       └─────────────┘              │
│         │                     │                     │                       │
│   Contribute ──────► Earn rUv Units ──────► Use for AI Workloads           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How It Works

| Step | What Happens | Result |
|------|--------------|--------|
| 1. **Contribute** | Share unused CPU cycles when browsing | Help the network |
| 2. **Earn** | Accumulate rUv (Resource Utility Vouchers) | Build credits |
| 3. **Use** | Spend rUv to run AI tasks | Access collective power |
| 4. **Network Grows** | More participants = more power | Everyone benefits |

### Why Collective AI Computing?

| Traditional AI | Collective Edge-Net |
|----------------|---------------------|
| Expensive GPU servers | Free idle browser CPUs |
| Centralized data centers | Distributed global network |
| Pay-per-use pricing | Contribution-based access |
| Single point of failure | Resilient P2P mesh |
| Limited by your hardware | Scale with the collective |

### AI Intelligence Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI INTELLIGENCE STACK                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MicroLoRA Adapter Pool (from ruvLLM)              │   │
│  │  • LRU-managed pool (16 slots) • <50µs rank-1 forward               │   │
│  │  • 4-bit/8-bit quantization    • 2,236+ ops/sec                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SONA - Self-Optimizing Neural Architecture         │   │
│  │  • Instant Loop: Per-request MicroLoRA adaptation                    │   │
│  │  • Background Loop: Hourly K-means consolidation                     │   │
│  │  • Deep Loop: Weekly EWC++ (prevents catastrophic forgetting)        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────────┐  │
│  │   HNSW Vector Index  │  │  Federated Learning  │  │ ReasoningBank   │  │
│  │   • 150x faster      │  │  • Byzantine tolerant│  │ • Pattern learn │  │
│  │   • O(log N) search  │  │  • Diff privacy      │  │ • 87x energy    │  │
│  └──────────────────────┘  └──────────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core AI Tasks

| Task Type | Use Case | Performance |
|-----------|----------|-------------|
| **Vector Search** | Find similar items | 150x speedup via HNSW |
| **Embeddings** | Text understanding | Semantic vectors |
| **Semantic Match** | Intent detection | Classify meaning |
| **LoRA Inference** | Task adaptation | <100µs forward |
| **Pattern Learning** | Self-optimization | ReasoningBank trajectories |

### Pi-Key Identity System

Ultra-compact cryptographic identity using mathematical constants:

| Key Type | Size | Purpose |
|----------|------|---------|
| **π (Pi-Key)** | 40 bytes | Permanent identity |
| **e (Session)** | 34 bytes | Encrypted sessions |
| **φ (Genesis)** | 21 bytes | Network origin markers |

### Self-Optimizing Features

| Feature | Mechanism | Benefit |
|---------|-----------|---------|
| **Task Routing** | Multi-head attention | Work goes to best nodes |
| **Topology Optimization** | Self-organizing mesh | Network adapts to load |
| **Q-Learning Security** | Reinforcement learning | Learns to defend threats |
| **Economic Balance** | rUv token system | Self-sustaining economy |

### Quick Start

**Add to Your Website:**
```html
<script type="module">
  import init, { EdgeNetNode, EdgeNetConfig } from '@ruvector/edge-net';

  async function joinCollective() {
    await init();

    // Join the collective
    const node = new EdgeNetConfig('my-website')
      .cpuLimit(0.3)          // Contribute 30% CPU when idle
      .memoryLimit(256 * 1024 * 1024)  // 256MB max
      .respectBattery(true)   // Reduce on battery
      .build();

    node.start();

    // Monitor participation
    setInterval(() => {
      console.log(`Contributed: ${node.ruvBalance()} rUv`);
    }, 10000);
  }

  joinCollective();
</script>
```

**Use the Collective's AI Power:**
```javascript
// Submit an AI task to the collective
const result = await node.submitTask('vector_search', {
  query: embeddings,
  k: 10,
  index: 'shared-knowledge-base'
}, 5);  // Spend up to 5 rUv

console.log('Similar items:', result);
```

**Monitor Your Contribution:**
```javascript
const stats = node.getStats();
console.log(`
  rUv Earned: ${stats.ruv_earned}
  rUv Spent: ${stats.ruv_spent}
  Tasks Completed: ${stats.tasks_completed}
  Reputation: ${(stats.reputation * 100).toFixed(1)}%
`);
```

### Key Features

| Feature | Benefit |
|---------|---------|
| **Idle CPU Utilization** | Use resources that would otherwise be wasted |
| **Browser-Based** | No installation, runs in any modern browser |
| **Adjustable Contribution** | Control how much you share (10-50% CPU) |
| **Battery Aware** | Automatically reduces on battery power |
| **Fair Distribution** | Work routed based on capability matching |
| **Privacy-First** | Pi-Key cryptographic identity |
| **Federated Learning** | Learn collectively without sharing data |
| **Byzantine Tolerance** | Resilient to malicious nodes |

### Installation

```bash
# npm
npm install @ruvector/edge-net

# Or include directly
<script src="https://unpkg.com/@ruvector/edge-net"></script>
```

> **Full Documentation**: [edge-net README](./examples/edge-net/README.md)

</details>

---

## AI & Machine Learning

<details>
<summary><strong>🎲 Agentic-Synth - AI Synthetic Data Generation</strong></summary>

[![npm](https://img.shields.io/npm/v/@ruvector/agentic-synth.svg)](https://www.npmjs.com/package/@ruvector/agentic-synth)
[![downloads](https://img.shields.io/npm/dt/@ruvector/agentic-synth.svg)](https://www.npmjs.com/package/@ruvector/agentic-synth)

**AI-Powered Synthetic Data Generation at Scale** — Generate unlimited, high-quality synthetic data for training AI models, testing systems, and building robust agentic applications.

### Why Agentic-Synth?

| Problem | Solution |
|---------|----------|
| Real data is **expensive** to collect | Generate **unlimited** synthetic data |
| **Privacy-sensitive** with compliance risks | **Fully synthetic**, no PII concerns |
| **Slow** to generate at scale | **10-100x faster** than manual creation |
| **Insufficient** for edge cases | **Customizable** schemas for any scenario |
| **Hard to reproduce** across environments | **Reproducible** with seed values |

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Support** | Gemini, OpenRouter, GPT, Claude, and 50+ models via DSPy.ts |
| **Context Caching** | 95%+ performance improvement with intelligent LRU cache |
| **Smart Model Routing** | Automatic load balancing, failover, and cost optimization |
| **DSPy.ts Integration** | Self-learning optimization with 20-25% quality improvement |
| **Streaming** | AsyncGenerator for real-time data flow |
| **Memory Efficient** | <50MB for datasets up to 10K records |

### Data Generation Types

| Type | Use Cases |
|------|-----------|
| **Time-Series** | Financial data, IoT sensors, metrics |
| **Events** | Logs, user actions, system events |
| **Structured** | JSON, CSV, databases, APIs |
| **Embeddings** | Vector data for RAG systems |

### Quick Start

```bash
# Install
npm install @ruvector/agentic-synth

# Or run instantly with npx
npx @ruvector/agentic-synth generate --count 100

# Interactive mode
npx @ruvector/agentic-synth interactive
```

### Basic Usage

```typescript
import { AgenticSynth } from '@ruvector/agentic-synth';

// Initialize with your preferred model
const synth = new AgenticSynth({
  model: 'gemini-pro',
  apiKey: process.env.GEMINI_API_KEY
});

// Generate structured data
const users = await synth.generate({
  schema: {
    name: 'string',
    email: 'email',
    age: 'number:18-65',
    role: ['admin', 'user', 'guest']
  },
  count: 1000
});

// Generate time-series data
const stockData = await synth.timeSeries({
  fields: ['open', 'high', 'low', 'close', 'volume'],
  interval: '1h',
  count: 500,
  volatility: 0.02
});

// Stream large datasets
for await (const batch of synth.stream({ count: 100000, batchSize: 1000 })) {
  await processData(batch);
}
```

### Self-Learning with DSPy

```typescript
import { AgenticSynth, DSPyOptimizer } from '@ruvector/agentic-synth';

// Enable self-learning optimization
const synth = new AgenticSynth({
  model: 'gemini-pro',
  optimizer: new DSPyOptimizer({
    learningRate: 0.1,
    qualityThreshold: 0.85
  })
});

// Quality improves automatically over time
const data = await synth.generate({
  schema: { ... },
  count: 1000,
  optimize: true  // Enable learning
});

console.log(`Quality score: ${data.metrics.quality}`);
// First run: 0.72
// After 100 runs: 0.94 (+25% improvement)
```

### Performance

| Metric | Value |
|--------|-------|
| **With caching** | 98.2% faster |
| **P99 latency** | 2500ms → 45ms |
| **Memory** | <50MB for 10K records |
| **Throughput** | 1000+ records/sec |

### Ecosystem Integration

| Package | Purpose |
|---------|---------|
| **RuVector** | Native vector database for RAG |
| **DSPy.ts** | Prompt optimization |
| **Agentic-Jujutsu** | Version-controlled generation |

### Installation

```bash
# npm
npm install @ruvector/agentic-synth

# Examples package (50+ production examples)
npm install @ruvector/agentic-synth-examples
```

> **Full Documentation**: [agentic-synth README](./npm/packages/agentic-synth/README.md)

</details>

<details>
<summary><strong>📈 Neural Trader - AI Trading System</strong></summary>

[![npm](https://img.shields.io/npm/v/neural-trader.svg)](https://www.npmjs.com/package/neural-trader)
[![downloads](https://img.shields.io/npm/dt/neural-trader.svg)](https://www.npmjs.com/package/neural-trader)

**Production-ready neural trading system** combining state-of-the-art ML for automated trading, sports betting, and portfolio management. Zero external ML dependencies, sub-millisecond latency.

### Core AI/ML Engines

| Engine | Description | Performance |
|--------|-------------|-------------|
| **Fractional Kelly** | Optimal position sizing with risk-adjusted bet scaling | 588,885 ops/s |
| **LSTM-Transformer** | Deep learning price prediction (temporal + attention) | 1,468 seq/s |
| **DRL Portfolio** | Reinforcement learning ensemble (PPO/SAC/A2C) | 17,043 steps/s |
| **Sentiment Alpha** | Real-time sentiment analysis for alpha generation | 3,764 pipeline/s |

### Why Neural Trader?

| Traditional ML | Neural Trader |
|----------------|---------------|
| TensorFlow/PyTorch required | **Zero dependencies** |
| 1.2MB+ bundle size | **45KB** bundle |
| 2.1ms LSTM inference | **0.68ms** inference |
| Complex deployment | **Works in browser & Node.js** |

### Research-Backed Algorithms

| Algorithm | Research Finding |
|-----------|------------------|
| **Kelly Criterion** | 1/5th fractional achieves 98% ROI with 85% less risk of ruin |
| **LSTM-Transformer** | Temporal + attention fusion outperforms single architectures |
| **DRL Ensemble** | PPO/SAC/A2C voting reduces variance vs single agent |
| **Sentiment Alpha** | 3% annual excess returns documented in academia |

### Quick Start

```javascript
import { KellyCriterion, HybridLSTMTransformer, DRLPortfolioManager } from 'neural-trader';

// Kelly position sizing
const kelly = new KellyCriterion();
const stake = kelly.calculateStake(9000, 0.55, 2.0, 0.2);  // 1/5th Kelly
// → $180 recommended stake (2% of bankroll)

// LSTM-Transformer prediction
const model = new HybridLSTMTransformer({
  lstm: { hiddenSize: 64, layers: 2 },
  transformer: { heads: 4, layers: 2 }
});
const prediction = model.predict(candles);
// → { signal: 'BUY', confidence: 0.73, direction: 'bullish' }

// DRL portfolio allocation
const manager = new DRLPortfolioManager({ numAssets: 10 });
await manager.train(marketData, { episodes: 100 });
const allocation = manager.getAction(currentState);
// → [0.15, 0.12, 0.08, ...] optimal weights
```

### Use Cases

| Use Case | Example |
|----------|---------|
| **Stock Trading** | DAG-based pipeline with parallel execution |
| **Sports Betting** | Kelly sizing with ML calibration |
| **Crypto Trading** | DRL portfolio for 20+ assets |
| **News Trading** | Real-time sentiment stream processing |
| **Portfolio Rebalancing** | Reinforcement learning allocation |

### Package Ecosystem (20+)

| Package | Description |
|---------|-------------|
| `neural-trader` | Core engine with native HNSW, SIMD |
| `@neural-trader/core` | Ultra-low latency Rust + Node.js bindings |
| `@neural-trader/strategies` | Strategy management and backtesting |
| `@neural-trader/execution` | Trade execution and order management |
| `@neural-trader/mcp` | MCP server with 87+ trading tools |
| `@neural-trader/risk` | VaR, stress testing, risk metrics |
| `@neural-trader/portfolio` | Markowitz, Risk Parity optimization |
| `@neural-trader/neural` | Neural network training |
| `@neural-trader/brokers` | Alpaca, Interactive Brokers |
| `@neural-trader/sports-betting` | Arbitrage, Kelly, odds analysis |

### CLI Interface

```bash
# Real-time trading
node cli.js run --strategy=hybrid --symbol=AAPL --capital=100000

# Backtest historical performance
node cli.js backtest --days=252 --capital=50000 --strategy=drl

# Paper trading simulation
node cli.js paper --capital=100000 --strategy=sentiment

# Performance benchmarks
node cli.js benchmark --iterations=100
```

### Exotic Examples

| Example | Description |
|---------|-------------|
| **Multi-Agent Swarm** | Distributed trading intelligence with consensus |
| **GNN Correlation Network** | Graph neural network correlation analysis |
| **Attention Regime Detection** | Transformer-based market regime classification |
| **Quantum Portfolio** | QAOA & quantum annealing optimization |
| **Hyperbolic Embeddings** | Poincaré disk market embeddings |
| **Atomic Arbitrage** | Cross-exchange with MEV protection |

### Performance

| Module | Latency | Throughput | Status |
|--------|---------|------------|--------|
| Kelly Engine | 0.014ms | 71,295/s | ✓ Ready |
| LSTM-Transformer | 0.681ms | 1,468/s | ✓ Ready |
| DRL Portfolio | 0.059ms | 17,043/s | ✓ Ready |
| Sentiment Alpha | 0.266ms | 3,764/s | ✓ Ready |
| Full Pipeline | 4.68ms | 214/s | ✓ Ready |

### Installation

```bash
# npm
npm install neural-trader

# Full ecosystem
npm install @neural-trader/core @neural-trader/strategies @neural-trader/mcp
```

> **Full Documentation**: [neural-trader README](./examples/neural-trader/README.md)

</details>

<details>
<summary><strong>🥋 Agentic-Jujutsu - Quantum-Resistant Version Control</strong></summary>

[![npm](https://img.shields.io/npm/v/agentic-jujutsu.svg)](https://www.npmjs.com/package/agentic-jujutsu)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

### What is Agentic-Jujutsu?

Agentic-Jujutsu is a **quantum-resistant, self-learning version control system** designed for AI agents. It combines lock-free concurrent operations with ReasoningBank trajectory learning for continuous improvement.

| Traditional Git | Agentic-Jujutsu |
|----------------|-----------------|
| Lock-based commits | Lock-free (23x faster) |
| Manual conflict resolution | 87% automatic resolution |
| Static operations | Self-learning from patterns |
| No quantum protection | SHA3-512 + HQC-128 |
| Sequential agents | Concurrent multi-agent |

### Key Features

| Feature | Performance | Description |
|---------|-------------|-------------|
| **Concurrent Commits** | 350 ops/s | 23x faster than Git (15 ops/s) |
| **Context Switching** | <100ms | 10x faster than Git (500-1000ms) |
| **Conflict Resolution** | 87% auto | AI-powered pattern matching |
| **Quantum Security** | <1ms verify | SHA3-512 fingerprints, HQC-128 encryption |
| **ReasoningBank** | Continuous | Trajectory learning with verdicts |

### Quick Start

```bash
# Install
npm install agentic-jujutsu

# Basic usage
npx agentic-jujutsu
```

```typescript
import { JjWrapper } from 'agentic-jujutsu';

const jj = new JjWrapper();

// Start learning trajectory
jj.startTrajectory('Implement feature X');

// Make changes and commit
await jj.newCommit('Add authentication module');
jj.addToTrajectory();

// Finalize with success score
jj.finalizeTrajectory(0.9, 'Feature implemented successfully');

// Get AI-powered suggestions
const suggestions = await jj.getSuggestions();
```

### Multi-Agent Coordination

```typescript
// Concurrent commits without locks
const agents = ['agent-1', 'agent-2', 'agent-3'];
await Promise.all(agents.map(agent =>
  jj.newCommit(`Changes from ${agent}`)
));
// All commits succeed - no lock waiting!
```

> **Full Documentation**: [agentic-jujutsu README](./examples/agentic-jujutsu/README.md)

</details>

<details>
<summary><strong>🔬 SciPix - Scientific Document OCR</strong></summary>

[![crates.io](https://img.shields.io/crates/v/ruvector-scipix.svg)](https://crates.io/crates/ruvector-scipix)
[![docs.rs](https://docs.rs/ruvector-scipix/badge.svg)](https://docs.rs/ruvector-scipix)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### What is SciPix?

SciPix is a **blazing-fast, memory-safe OCR engine** written in pure Rust, purpose-built for scientific documents, mathematical equations, and technical diagrams.

| Feature | SciPix | Tesseract | Mathpix |
|---------|--------|-----------|---------|
| Simple Text OCR | **50ms** | 120ms | 200ms* |
| Math Equation | **80ms** | N/A | 150ms* |
| Batch (100 images) | **2.1s** | 8.5s | N/A |
| Memory Usage | **45MB** | 180MB | Cloud |
| LaTeX Output | Yes | No | Yes |

*API latency, not processing time

### Key Features

| Feature | Description |
|---------|-------------|
| **ONNX Runtime** | GPU-accelerated with CUDA, TensorRT, CoreML |
| **LaTeX Output** | Mathematical equation recognition with LaTeX, MathML, AsciiMath |
| **SIMD Optimized** | 4x faster image preprocessing with AVX2, SSE4, NEON |
| **REST API** | Production-ready HTTP server with rate limiting |
| **MCP Server** | Integrate with Claude, ChatGPT via Model Context Protocol |
| **WebAssembly** | Run OCR directly in browsers |

### Quick Start

```bash
# Add to Cargo.toml
cargo add ruvector-scipix

# With features
ruvector-scipix = { version = "0.1.16", features = ["ocr", "math", "optimize"] }
```

```rust
use ruvector_scipix::{SciPixOcr, OcrConfig};

// Initialize OCR engine
let ocr = SciPixOcr::new(OcrConfig::default())?;

// Process scientific image
let result = ocr.process_image("equation.png")?;
println!("LaTeX: {}", result.latex);
println!("Confidence: {:.2}%", result.confidence * 100.0);
```

### Use Cases

- **Academic Paper Digitization** - Extract text and equations from scanned research papers
- **Math Homework Assistance** - Convert handwritten equations to LaTeX for AI tutoring
- **Technical Documentation** - Process engineering diagrams and scientific charts
- **AI/LLM Integration** - Feed scientific content to language models via MCP

> **Full Documentation**: [scipix README](./examples/scipix/README.md)

</details>

<details>
<summary><strong>🧠 Meta-Cognition SNN - Spiking Neural Networks</strong></summary>

### What is Meta-Cognition SNN?

A hybrid AI architecture combining **Spiking Neural Networks (SNN)**, **SIMD-optimized vector operations**, and **5 attention mechanisms** with meta-cognitive self-discovery capabilities.

| Capability | Performance | Description |
|------------|-------------|-------------|
| **Spiking Neural Networks** | 10-50x faster | LIF neurons + STDP learning with N-API SIMD |
| **SIMD Vector Operations** | 5-54x faster | Loop-unrolled distance/dot product calculations |
| **5 Attention Mechanisms** | Sub-millisecond | Multi-Head, Flash, Linear, Hyperbolic, MoE |
| **Vector Search** | 150x faster | RuVector-powered semantic search |
| **Meta-Cognition** | Autonomous | Self-discovering emergent capabilities |

### SIMD Performance

| Operation | Speedup | Notes |
|-----------|---------|-------|
| LIF Updates | **16.7x** | Leaky integrate-and-fire neurons |
| Synaptic Forward | **14.9x** | Forward propagation |
| STDP Learning | **26.3x** | Spike-timing dependent plasticity |
| Distance (128d) | **54x** | Euclidean distance calculation |
| Full Simulation | **18.4x** | End-to-end SNN simulation |

### 5 Attention Mechanisms

| Mechanism | Best For | Latency |
|-----------|----------|---------|
| **Flash** | Long sequences | 0.023ms |
| **MoE** | Specialized domains | 0.021ms |
| **Multi-Head** | Complex patterns | 0.047ms |
| **Linear** | Real-time processing | 0.075ms |
| **Hyperbolic** | Hierarchical data | 0.222ms |

### Quick Start

```bash
# Install and run demos
cd examples/meta-cognition-spiking-neural-network
npm install
node demos/run-all.js
```

```javascript
const { createFeedforwardSNN, rateEncoding } = require('./demos/snn/lib/SpikingNeuralNetwork');

// Create SNN with SIMD optimization
const snn = createFeedforwardSNN([100, 50, 10], {
  dt: 1.0,
  tau: 20.0,
  a_plus: 0.005,
  lateral_inhibition: true
});

// Train with STDP
const input = rateEncoding(pattern, snn.dt, 100);
snn.step(input);
```

### 6 Emergent Discoveries

1. Multi-Scale Attention Hierarchy (Novelty: 5/5)
2. Spike Synchronization Patterns
3. Attention-Gated Spike Propagation
4. Temporal Coherence Emergence
5. Emergent Sparsity (80% fewer active neurons)
6. Meta-Plasticity (faster learning on later tasks)

> **Full Documentation**: [meta-cognition-snn README](./examples/meta-cognition-spiking-neural-network/README.md)

</details>

<details>
<summary><strong>🤖 RuvLLM - Self-Learning LLM Orchestration</strong></summary>

[![Rust](https://img.shields.io/badge/rust-1.77%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/export-HuggingFace-yellow.svg)](#)

### What is RuvLLM?

RuvLLM is a **self-learning language model orchestration system** that combines frozen foundation models with adaptive memory and intelligent routing. Unlike traditional LLMs that rely solely on static parameters, RuvLLM continuously improves from every interaction.

> *"The intelligence is not in one model anymore. It is in the loop."*

### SONA: 3-Tier Temporal Learning

| Loop | Frequency | Latency | Description |
|------|-----------|---------|-------------|
| **A: Instant** | Per-request | <100μs | MicroLoRA adaptation (rank 1-2) |
| **B: Background** | Hourly | ~1.3ms | K-means++ clustering, base LoRA (rank 4-16) |
| **C: Deep** | Weekly | N/A | EWC++ consolidation, concept hierarchies |

### Core Components

| Component | Description |
|-----------|-------------|
| **LFM2 Cortex** | Frozen reasoning engine (135M-2.6B params) |
| **Ruvector Memory** | Adaptive synaptic mesh with HNSW indexing |
| **FastGRNN Router** | Intelligent model selection circuit |
| **Graph Attention** | 8-head attention with edge features |
| **SONA Engine** | LoRA + EWC++ + ReasoningBank |

### Performance (CPU-Only)

| Metric | Value |
|--------|-------|
| **Initialization** | 3.71ms |
| **Average Query** | 0.09ms |
| **Session Query** | 0.04ms |
| **Throughput** | ~38,000 q/s |
| **Memory** | ~50MB |

### Quick Start

```rust
use ruvllm::{RuvLLMOrchestrator, OrchestratorConfig};

// Initialize orchestrator
let config = OrchestratorConfig::default();
let orchestrator = RuvLLMOrchestrator::new(config)?;

// Query with automatic learning
let response = orchestrator.query("Explain quantum entanglement").await?;
println!("{}", response.text);

// Response improves over time as SONA learns patterns
```

### Federated Learning

```rust
// Ephemeral agents collect trajectories
let agent = EphemeralAgent::new("task-specific-agent");
agent.process_task(&task).await?;
let export = agent.export();

// Central coordinator aggregates learning
coordinator.accept_export(export)?;
coordinator.consolidate();  // Share patterns with new agents
```

### Dynamic Embedding Fine-Tuning

RuvLLM's adaptive learning system enables real-time model improvement without retraining.

| Feature | Description | Latency |
|---------|-------------|---------|
| **MicroLoRA** | Per-request adaptation (rank 1-2), <50KB adapters | <1ms |
| **Contrastive Training** | Triplet loss with hard negatives for embedding optimization | Batch |
| **Task-Specific Adapters** | Pre-tuned for Coder, Researcher, Security, Architect, Reviewer | Hot-swap |
| **EWC++ Consolidation** | Prevents catastrophic forgetting during continuous learning | Background |
| **Adapter Merging** | Average, Weighted, SLERP, TIES, DARE strategies | On-demand |

```javascript
// Contrastive fine-tuning for agent routing
import { ContrastiveTrainer } from '@ruvector/ruvllm';

const trainer = new ContrastiveTrainer({
  margin: 0.5,
  hardNegativeRatio: 0.7
});

// Learn: task → correct agent, not wrong agent
trainer.addTriplet(taskEmb, correctAgentEmb, wrongAgentEmb, true);
const model = trainer.train();
```

```rust
// Task-specific adapter hot-swapping
use ruvllm::lora::RuvLtraAdapters;

let adapters = RuvLtraAdapters::new();
let coder = adapters.create_lora("coder", 768)?;      // Rank 16, code patterns
let security = adapters.create_lora("security", 768)?; // Rank 16, vulnerability detection

// Hot-swap at runtime without model reload
orchestrator.set_adapter(coder);
let code_response = orchestrator.query("Implement binary search").await?;

orchestrator.set_adapter(security);
let audit_response = orchestrator.query("Audit this code for vulnerabilities").await?;
```

### Advanced Features

- **SIMD Inference**: AVX2/AVX512/SSE4.1 optimization
- **Q4 Quantization**: 4-bit weights for memory efficiency
- **HuggingFace Export**: Export LoRA weights and preference pairs
- **Multi-Model Routing**: SmolLM, Qwen2, TinyLlama selection
- **WASM Support**: Run SONA in browsers and edge devices
- **Browser Fine-Tuning**: MicroLoRA WASM with localStorage persistence

> **Full Documentation**: [ruvLLM README](./examples/ruvLLM/README.md) | [Fine-Tuning Guide](./docs/ruvllm/FINE_TUNING.md) | [Task Adapters](./docs/training/task_specific_lora_adapters.md)

</details>

<details>
<summary><strong>🗜️ REFRAG - Compress-Sense-Expand RAG</strong></summary>

### What is REFRAG?

REFRAG implements the **Compress-Sense-Expand architecture** from [arXiv:2509.01092](https://arxiv.org/abs/2509.01092), achieving **~30x RAG latency reduction** by storing pre-computed representation tensors instead of raw text.

### Architecture

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│   COMPRESS     │───▶│     SENSE      │───▶│    EXPAND      │
│    Layer       │    │     Layer      │    │    Layer       │
└────────────────┘    └────────────────┘    └────────────────┘

Binary tensor         Policy network        Dimension projection
storage with          decides COMPRESS      (768 → 4096 dims)
zero-copy access      vs EXPAND
```

### Compression Strategies

| Strategy | Compression | Use Case |
|----------|-------------|----------|
| `None` | 1x | Maximum precision |
| `Float16` | 2x | Good balance |
| `Int8` | 4x | Memory constrained |
| `Binary` | 32x | Extreme compression |

### Policy Networks

| Policy | Latency | Description |
|--------|---------|-------------|
| `ThresholdPolicy` | ~2μs | Cosine similarity threshold |
| `LinearPolicy` | ~5μs | Single layer classifier |
| `MLPPolicy` | ~15μs | Two-layer neural network |

### Quick Start

```bash
# Run demo
cargo run --bin refrag-demo

# Run benchmarks
cargo run --bin refrag-benchmark --release
```

```rust
use refrag_pipeline_example::{RefragStore, RefragEntry};

// Create REFRAG-enabled store
let store = RefragStore::new(384, 768)?;

// Insert with representation tensor
let entry = RefragEntry::new("doc_1", search_vector, "The quick brown fox...")
    .with_tensor(tensor_bytes, "llama3-8b");
store.insert(entry)?;

// Hybrid search (policy-based COMPRESS/EXPAND)
let results = store.search_hybrid(&query, 10, Some(0.85))?;

for result in results {
    match result.response_type {
        RefragResponseType::Compress => {
            // Tensor directly injectable into LLM context
            println!("Tensor: {} dims", result.tensor_dims.unwrap());
        }
        RefragResponseType::Expand => {
            // Original text when full context needed
            println!("Text: {}", result.content.unwrap());
        }
    }
}
```

### Target LLM Dimensions

| Source | Target | LLM |
|--------|--------|-----|
| 768 | 4096 | LLaMA-3 8B |
| 768 | 8192 | LLaMA-3 70B |
| 1536 | 8192 | GPT-4 |

> **Full Documentation**: [refrag-pipeline README](./examples/refrag-pipeline/README.md)

</details>

<details>
<summary><strong>🐦 7sense - Bioacoustic Intelligence Platform</strong></summary>

[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![Tests](https://img.shields.io/badge/tests-329%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-85%25-green.svg)]()

### What is 7sense?

7sense transforms **bird calls into navigable geometric space** using cutting-edge AI and vector search. It converts audio recordings of bird songs into rich, searchable embeddings using Perch 2.0 neural networks and ultra-fast HNSW indexing.

| Traditional Monitoring | 7sense |
|----------------------|--------|
| Expert human listeners | Instant AI species ID |
| Basic spectrogram analysis | Neural embeddings (1536-dim) |
| Limited scale | Millions of recordings |
| Manual pattern finding | Automated discovery |

### Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| HNSW Search Speedup | 150x vs brute force | Achieved |
| Query Latency (p99) | < 50ms | Achieved |
| Recall@10 | >= 0.95 | Achieved |
| Embedding Throughput | > 100 segments/sec | Achieved |
| Memory per 1M vectors | < 6 GB | Achieved |

### 9 Rust Crates

| Crate | Description |
|-------|-------------|
| `sevensense-core` | Species taxonomy, temporal types |
| `sevensense-audio` | WAV/MP3/FLAC, Mel spectrograms |
| `sevensense-embedding` | Perch 2.0 ONNX, 1536-dim vectors |
| `sevensense-vector` | HNSW with 150x speedup |
| `sevensense-learning` | GNN training, EWC regularization |
| `sevensense-analysis` | HDBSCAN clustering, Markov models |
| `sevensense-interpretation` | Evidence packs, species narratives |
| `sevensense-api` | GraphQL, REST, WebSocket streaming |
| `sevensense-benches` | Criterion.rs performance suites |

### Quick Start

```bash
# Build and run
cd examples/vibecast-7sense
cargo build --release
cargo run -p sevensense-api --release
```

```rust
use sevensense_audio::AudioProcessor;
use sevensense_embedding::EmbeddingPipeline;
use sevensense_vector::HnswIndex;

// Load and process audio
let processor = AudioProcessor::new(Default::default());
let segments = processor.process_file("recording.wav").await?;

// Generate Perch 2.0 embeddings
let pipeline = EmbeddingPipeline::new(Default::default()).await?;
let embeddings = pipeline.embed_segments(&segments).await?;

// Search for similar calls (150x faster)
let index = HnswIndex::new(Default::default());
index.add_batch(&embeddings)?;
let neighbors = index.search(&embeddings[0], 10)?;

println!("Found {} similar bird calls", neighbors.len());
```

### Use Cases

- **Species Identification** - Instant predictions with confidence scores
- **Pattern Discovery** - Find similar calls across millions of recordings
- **Behavioral Insights** - Detect singing patterns, dialects, anomalies
- **Conservation Monitoring** - Track biodiversity at scale

> **Full Documentation**: [7sense README](./examples/vibecast-7sense/README.md)

</details>

<details>
<summary><strong>🧬 EXO-AI - Advanced Cognitive Substrate</strong></summary>

[![crates.io](https://img.shields.io/crates/v/exo-core.svg)](https://crates.io/crates/exo-core)
[![docs.rs](https://docs.rs/exo-core/badge.svg)](https://docs.rs/exo-core)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

### What is EXO-AI?

EXO-AI 2025 is a comprehensive **cognitive substrate** implementing cutting-edge theories from neuroscience, physics, and consciousness research. It provides 9 interconnected Rust crates totaling ~15,800+ lines of research-grade code.

> Traditional AI systems process information. EXO-AI aims to understand it.

### SIMD-Accelerated Performance

| Operation | Speedup | Notes |
|-----------|---------|-------|
| Distance (128d) | **54x** | AVX2/NEON optimized |
| Cosine Similarity | **2.73x** | Batch operations |
| Pattern Matching | **8-54x** | Loop-unrolled |
| Meta-Simulation | **13+ quadrillion/s** | From ultra-low-latency-sim |

### 9 Rust Crates

| Crate | Description |
|-------|-------------|
| `exo-core` | IIT consciousness (Φ) measurement & Landauer thermodynamics |
| `exo-temporal` | Temporal memory with causal tracking & consolidation |
| `exo-hypergraph` | Topological analysis with persistent homology |
| `exo-manifold` | SIREN networks + SIMD-accelerated retrieval |
| `exo-exotic` | 10 cutting-edge cognitive experiments |
| `exo-federation` | Post-quantum federated cognitive mesh |
| `exo-backend-classical` | SIMD-accelerated compute backend |
| `exo-wasm` | Browser & edge WASM deployment |
| `exo-node` | Node.js bindings via NAPI-RS |

### Key Theories Implemented

| Theory | Implementation |
|--------|---------------|
| **IIT (Integrated Information Theory)** | Consciousness level (Φ) measurement |
| **Landauer's Principle** | Computational thermodynamics |
| **Free Energy Principle** | Friston's predictive processing |
| **Strange Loops** | Hofstadter's self-referential patterns |
| **Morphogenesis** | Pattern formation emergence |

### Quick Start

```toml
[dependencies]
exo-core = "0.1"
exo-temporal = "0.1"
exo-exotic = "0.1"
exo-manifold = "0.1"  # SIMD acceleration!
```

```rust
use exo_core::consciousness::{ConsciousnessSubstrate, IITConfig};
use exo_core::thermodynamics::CognitiveThermometer;

// Measure integrated information (Φ)
let substrate = ConsciousnessSubstrate::new(IITConfig::default());
substrate.add_pattern(pattern);
let phi = substrate.compute_phi();
println!("Consciousness level (Φ): {:.4}", phi);

// Track computational thermodynamics
let thermo = CognitiveThermometer::new(300.0); // Kelvin
let cost = thermo.landauer_cost_bits(1024);
println!("Landauer cost: {:.2e} J", cost);
```

### SIMD Pattern Retrieval

```rust
use exo_manifold::{ManifoldEngine, cosine_similarity_simd, batch_distances};

// 54x faster similarity search
let query = vec![0.5; 768];
let results = engine.retrieve(&query, 10)?;

// Batch distance computation
let distances = batch_distances(&query, &database);  // 8-54x speedup
```

> **Full Documentation**: [exo-ai README](./examples/exo-ai-2025/README.md)

</details>

---

## Database Extensions

<details>
<summary><strong>🐘 PostgreSQL Extension</strong></summary>

[![crates.io](https://img.shields.io/crates/v/ruvector-postgres.svg)](https://crates.io/crates/ruvector-postgres)
[![npm](https://img.shields.io/npm/v/@ruvector/postgres-cli.svg)](https://www.npmjs.com/package/@ruvector/postgres-cli)
[![Docker Hub](https://img.shields.io/docker/pulls/ruvnet/ruvector-postgres?label=docker%20pulls)](https://hub.docker.com/r/ruvnet/ruvector-postgres)
[![Docker](https://img.shields.io/docker/v/ruvnet/ruvector-postgres?label=docker)](https://hub.docker.com/r/ruvnet/ruvector-postgres)

**The most advanced PostgreSQL vector extension** — a drop-in pgvector replacement with 230+ SQL functions, hardware-accelerated SIMD operations, and built-in AI capabilities. Transform your existing PostgreSQL database into a full-featured vector search engine with GNN layers, attention mechanisms, and self-learning capabilities.

```bash
# Quick Install from Docker Hub
docker run -d --name ruvector \
  -e POSTGRES_PASSWORD=secret \
  -p 5432:5432 \
  ruvnet/ruvector-postgres:latest

# Connect and use
psql -h localhost -U ruvector -d ruvector_test

# Create extension
CREATE EXTENSION ruvector;
```

**Why RuVector Postgres?**
- **Zero Migration** — Works with existing pgvector code, just swap the extension
- **10x More Functions** — 230+ SQL functions vs pgvector's ~20
- **2x Faster** — AVX-512/AVX2/NEON SIMD acceleration
- **AI-Native** — GNN layers, 39 attention mechanisms, local embeddings
- **Self-Learning** — Improves search quality over time with ReasoningBank

| Feature | pgvector | RuVector Postgres |
|---------|----------|-------------------|
| SQL Functions | ~20 | **230+** |
| SIMD Acceleration | Basic | AVX-512/AVX2/NEON (~2x faster) |
| Index Types | HNSW, IVFFlat | HNSW, IVFFlat + Hyperbolic |
| Attention Mechanisms | ❌ | 39 types (Flash, Linear, Graph) |
| GNN Layers | ❌ | GCN, GraphSAGE, GAT, GIN |
| Sparse Vectors | ❌ | BM25, TF-IDF, SPLADE |
| Self-Learning | ❌ | ReasoningBank, trajectory learning |
| Local Embeddings | ❌ | 6 fastembed models built-in |
| Multi-Tenancy | ❌ | Built-in namespace isolation |
| Quantization | ❌ | Scalar, Product, Binary (4-32x compression) |

<details>
<summary><strong>🐳 Docker Hub (Recommended)</strong></summary>

**Pull from Docker Hub:** [hub.docker.com/r/ruvnet/ruvector-postgres](https://hub.docker.com/r/ruvnet/ruvector-postgres)

```bash
# Quick start
docker run -d --name ruvector \
  -e POSTGRES_PASSWORD=secret \
  -p 5432:5432 \
  ruvnet/ruvector-postgres:latest

# Connect
psql -h localhost -U ruvector -d ruvector_test

# Create extension
CREATE EXTENSION ruvector;
```

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `ruvector` | Database user |
| `POSTGRES_PASSWORD` | `ruvector` | Database password |
| `POSTGRES_DB` | `ruvector_test` | Default database |

**Docker Compose:**
```yaml
version: '3.8'
services:
  ruvector-postgres:
    image: ruvnet/ruvector-postgres:latest
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: ruvector_test
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

**Available Tags:**
- `ruvnet/ruvector-postgres:latest` - PostgreSQL + RuVector 2.0
- `ruvnet/ruvector-postgres:2.0.0` - Specific version

</details>

<details>
<summary><strong>📦 npm CLI</strong></summary>

```bash
# Install globally
npm install -g @ruvector/postgres-cli

# Or use npx
npx @ruvector/postgres-cli --help

# Commands available as 'ruvector-pg' or 'rvpg'
ruvector-pg --version
rvpg --help
```

**CLI Commands:**
```bash
# Install extension to existing PostgreSQL
ruvector-pg install

# Create vector table with HNSW index
ruvector-pg vector create table embeddings --dim 1536 --index hnsw

# Import vectors from file
ruvector-pg vector import embeddings data.json

# Search vectors
ruvector-pg vector search embeddings --query "0.1,0.2,..." --limit 10

# Benchmark performance
ruvector-pg bench --iterations 1000

# Check extension status
ruvector-pg status
```

**Programmatic Usage:**
```typescript
import { RuvectorPG } from '@ruvector/postgres-cli';

const client = new RuvectorPG({
  host: 'localhost',
  port: 5432,
  database: 'vectors',
  user: 'postgres',
  password: 'secret'
});

// Create table with HNSW index
await client.createTable('embeddings', {
  dimensions: 1536,
  indexType: 'hnsw',
  distanceMetric: 'cosine'
});

// Insert vectors
await client.insert('embeddings', {
  id: '1',
  vector: [0.1, 0.2, ...],
  metadata: { source: 'openai' }
});

// Search
const results = await client.search('embeddings', queryVector, { limit: 10 });
```

</details>

<details>
<summary><strong>🦀 Rust Crate</strong></summary>

```bash
# Install pgrx (PostgreSQL extension framework)
cargo install cargo-pgrx --version "0.12.9" --locked
cargo pgrx init

# Build and install extension
cd crates/ruvector-postgres
cargo pgrx install --release

# Or install specific PostgreSQL version
cargo pgrx install --release --pg-config /usr/lib/postgresql/17/bin/pg_config
```

**Cargo.toml:**
```toml
[dependencies]
ruvector-postgres = "2.0"

# Optional features
[features]
default = ["pg17"]
pg16 = ["ruvector-postgres/pg16"]
pg15 = ["ruvector-postgres/pg15"]

# AI features (opt-in)
ai-complete = ["ruvector-postgres/ai-complete"]  # All AI features
learning = ["ruvector-postgres/learning"]         # Self-learning
attention = ["ruvector-postgres/attention"]       # 39 attention mechanisms
gnn = ["ruvector-postgres/gnn"]                   # Graph neural networks
hyperbolic = ["ruvector-postgres/hyperbolic"]     # Hyperbolic embeddings
embeddings = ["ruvector-postgres/embeddings"]     # Local embedding generation
```

**Build with all features:**
```bash
cargo pgrx install --release --features "ai-complete,embeddings"
```

</details>

<details>
<summary><strong>📝 SQL Examples</strong></summary>

```sql
-- Enable extension
CREATE EXTENSION ruvector;

-- Create table with vector column
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding VECTOR(1536)
);

-- Create HNSW index
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

-- Insert vectors
INSERT INTO documents (content, embedding)
VALUES ('Hello world', '[0.1, 0.2, ...]'::vector);

-- Semantic search (cosine similarity)
SELECT id, content, embedding <=> '[0.1, 0.2, ...]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- Hybrid search (vector + full-text)
SELECT id, content
FROM documents
WHERE to_tsvector(content) @@ to_tsquery('machine & learning')
ORDER BY embedding <=> query_embedding
LIMIT 10;

-- GNN-enhanced search (with learning)
SELECT * FROM ruvector_gnn_search(
  'documents',
  '[0.1, 0.2, ...]'::vector,
  10,  -- limit
  'gcn' -- gnn_type: gcn, graphsage, gat, gin
);

-- Generate embeddings locally (no API needed)
SELECT ruvector_embed('all-MiniLM-L6-v2', 'Your text here');

-- Flash attention
SELECT ruvector_flash_attention(query, key, value);
```

</details>

See [ruvector-postgres README](./crates/ruvector-postgres/README.md) for full SQL API reference (230+ functions).

</details>

---

## Developer Tools

<details>
<summary>🛠️ Tools & Utilities</summary>

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-bench](./crates/ruvector-bench) | Benchmarking suite for vector operations | [![crates.io](https://img.shields.io/crates/v/ruvector-bench.svg)](https://crates.io/crates/ruvector-bench) |
| [ruvector-metrics](./crates/ruvector-metrics) | Observability, metrics, and monitoring | [![crates.io](https://img.shields.io/crates/v/ruvector-metrics.svg)](https://crates.io/crates/ruvector-metrics) |
| [ruvector-filter](./crates/ruvector-filter) | Metadata filtering and query predicates | [![crates.io](https://img.shields.io/crates/v/ruvector-filter.svg)](https://crates.io/crates/ruvector-filter) |
| [ruvector-collections](./crates/ruvector-collections) | Multi-tenant collection management | [![crates.io](https://img.shields.io/crates/v/ruvector-collections.svg)](https://crates.io/crates/ruvector-collections) |
| [ruvector-snapshot](./crates/ruvector-snapshot) | Point-in-time snapshots and backups | [![crates.io](https://img.shields.io/crates/v/ruvector-snapshot.svg)](https://crates.io/crates/ruvector-snapshot) |
| [micro-hnsw-wasm](./crates/micro-hnsw-wasm) | Lightweight HNSW implementation for WASM | [![crates.io](https://img.shields.io/crates/v/micro-hnsw-wasm.svg)](https://crates.io/crates/micro-hnsw-wasm) |

### Embedded & IoT

| Crate | Description | Target |
|-------|-------------|--------|
| [ruvector-esp32](./examples/edge) | ESP32/ESP-IDF vector search | ESP32, ESP32-S3 |
| [rvlite](./crates/rvlite) | SQLite-style edge DB (no_std compatible) | ARM, RISC-V, WASM |
| [micro-hnsw-wasm](./crates/micro-hnsw-wasm) | <50KB HNSW for constrained devices | WASM, embedded |

```rust
// ESP32 example (no_std)
#![no_std]
use rvlite::RvLite;

let db = RvLite::new(128);  // 128-dim vectors
db.insert(0, &embedding);
let results = db.search(&query, 5);
```

</details>

---

## Browser & Edge (WASM)

<details>
<summary>🌐 WASM Packages (Browser & Edge)</summary>

Specialized WebAssembly modules for browser and edge deployment. These packages bring advanced AI and distributed computing primitives to JavaScript/TypeScript with near-native performance.

### Quick Install (All Browser WASM)

```bash
# Core vector search
npm install ruvector-wasm @ruvector/rvlite

# AI & Neural
npm install @ruvector/gnn-wasm @ruvector/attention-wasm @ruvector/sona-wasm

# Graph & Algorithms
npm install @ruvector/graph-wasm @ruvector/mincut-wasm @ruvector/hyperbolic-hnsw-wasm

# Exotic AI
npm install @ruvector/economy-wasm @ruvector/exotic-wasm @ruvector/nervous-system-wasm

# LLM (browser inference)
npm install @ruvector/ruvllm-wasm
```

| Category | Packages | Total Size |
|----------|----------|------------|
| **Core** | ruvector-wasm, rvlite | ~200KB |
| **AI/Neural** | gnn, attention, sona | ~300KB |
| **Graph** | graph, mincut, hyperbolic-hnsw | ~250KB |
| **Exotic** | economy, exotic, nervous-system | ~350KB |
| **LLM** | ruvllm-wasm | ~500KB |

### Installation

```bash
# Install individual packages
npm install @ruvector/learning-wasm
npm install @ruvector/economy-wasm
npm install @ruvector/exotic-wasm
npm install @ruvector/nervous-system-wasm
npm install @ruvector/attention-unified-wasm

# Or build from source
cd crates/ruvector-learning-wasm
wasm-pack build --target web
```

### ruvector-learning-wasm

**MicroLoRA, BTSP, and HDC for self-learning AI systems.**

Ultra-fast Low-Rank Adaptation (LoRA) optimized for WASM execution with <100us adaptation latency. Designed for real-time per-operator learning in query optimization and AI agent systems.

| Feature | Performance | Description |
|---------|-------------|-------------|
| **MicroLoRA** | <100us latency | Rank-2 LoRA matrices for instant weight adaptation |
| **Per-Operator Scoping** | Zero-allocation hot paths | Separate adapters for different operator types |
| **Trajectory Tracking** | Lock-free buffers | Record learning trajectories for replay |

**Architecture:**

```
Input Embedding (256-dim)
       |
       v
  +---------+
  | A: d x 2 |  Down projection
  +---------+
       |
       v
  +---------+
  | B: 2 x d |  Up projection
  +---------+
       |
       v
Delta W = alpha * (A @ B)
       |
       v
Output = Input + Delta W
```

**JavaScript/TypeScript Example:**

```typescript
import init, { WasmMicroLoRA } from '@ruvector/learning-wasm';

await init();

// Create MicroLoRA engine (256-dim, alpha=0.1, lr=0.01)
const lora = new WasmMicroLoRA(256, 0.1, 0.01);

// Forward pass with adaptation
const input = new Float32Array(256).fill(0.5);
const output = lora.forward_array(input);

// Adapt based on gradient signal
const gradient = new Float32Array(256).fill(0.1);
lora.adapt_array(gradient);

// Adapt with reward signal for RL
lora.adapt_with_reward(0.8);  // 80% improvement

console.log(`Adaptations: ${lora.adapt_count()}`);
console.log(`Delta norm: ${lora.delta_norm()}`);
```

### ruvector-economy-wasm

**CRDT-based autonomous credit economy for distributed compute networks.**

P2P-safe concurrent transactions using Conflict-free Replicated Data Types (CRDTs). Features a 10x-to-1x early adopter contribution curve and stake/slash mechanisms for participation incentives.

| Feature | Description |
|---------|-------------|
| **CRDT Ledger** | G-Counter (earned) + PN-Counter (spent) for P2P consistency |
| **Contribution Curve** | 10x early adopter multiplier decaying to 1x baseline |
| **Stake/Slash** | Participation requirements with slashing for bad actors |
| **Reputation Scoring** | Multi-factor: accuracy * uptime * stake_weight |
| **Merkle Verification** | SHA-256 state root for quick ledger verification |

**Architecture:**

```
+------------------------+
|     CreditLedger       |  <-- CRDT-based P2P-safe ledger
|  +------------------+  |
|  | G-Counter: Earned|  |  <-- Monotonically increasing
|  | PN-Counter: Spent|  |  <-- Can handle disputes/refunds
|  | Stake: Locked    |  |  <-- Participation requirement
|  | State Root       |  |  <-- Merkle root for verification
|  +------------------+  |
+------------------------+
          |
          v
+------------------------+
|  ContributionCurve     |  <-- Exponential decay: 10x -> 1x
+------------------------+
          |
          v
+------------------------+
|   ReputationScore      |  <-- accuracy * uptime * stake_weight
+------------------------+
```

**JavaScript/TypeScript Example:**

```typescript
import init, {
  CreditLedger,
  ReputationScore,
  contribution_multiplier
} from '@ruvector/economy-wasm';

await init();

// Create a new ledger for a node
const ledger = new CreditLedger("node-123");

// Earn credits (with early adopter multiplier)
ledger.creditWithMultiplier(100, "task:abc");
console.log(`Balance: ${ledger.balance()}`);
console.log(`Multiplier: ${ledger.currentMultiplier()}x`);

// Stake for participation
ledger.stake(50);
console.log(`Staked: ${ledger.stakedAmount()}`);

// Check multiplier for network compute hours
const mult = contribution_multiplier(50000.0);  // 50K hours
console.log(`Network multiplier: ${mult}x`);  // ~8.5x

// Track reputation
const rep = new ReputationScore(0.95, 0.98, 1000);
console.log(`Composite score: ${rep.composite_score()}`);

// P2P merge with another ledger (CRDT operation)
const otherEarned = new Uint8Array([/* serialized earned counter */]);
const otherSpent = new Uint8Array([/* serialized spent counter */]);
const mergedCount = ledger.merge(otherEarned, otherSpent);
```

### ruvector-exotic-wasm

**Exotic AI mechanisms for emergent behavior in distributed systems.**

Novel coordination primitives inspired by decentralized governance, developmental biology, and quantum physics.

| Mechanism | Inspiration | Use Case |
|-----------|-------------|----------|
| **Neural Autonomous Organization (NAO)** | DAOs + oscillatory sync | Decentralized AI agent governance |
| **Morphogenetic Network** | Developmental biology | Emergent network topology |
| **Time Crystal Coordinator** | Quantum time crystals | Robust distributed coordination |

**NAO Features:**
- Stake-weighted quadratic voting
- Oscillatory synchronization for coherence
- Quorum-based consensus (configurable threshold)

**Morphogenetic Network Features:**
- Cellular differentiation through morphogen gradients
- Emergent network topology via growth/pruning
- Synaptic pruning for optimization

**Time Crystal Features:**
- Period-doubled oscillations for stable coordination
- Floquet engineering for noise resilience
- Phase-locked agent synchronization

**JavaScript/TypeScript Example:**

```typescript
import init, {
  WasmNAO,
  WasmMorphogeneticNetwork,
  WasmTimeCrystal,
  ExoticEcosystem
} from '@ruvector/exotic-wasm';

await init();

// Neural Autonomous Organization
const nao = new WasmNAO(0.7);  // 70% quorum
nao.addMember("agent_1", 100);  // 100 stake
nao.addMember("agent_2", 50);

const propId = nao.propose("Upgrade memory backend");
nao.vote(propId, "agent_1", 0.9);  // 90% approval weight
nao.vote(propId, "agent_2", 0.6);

if (nao.execute(propId)) {
  console.log("Proposal executed!");
}

// Morphogenetic Network
const net = new WasmMorphogeneticNetwork(100, 100);  // 100x100 grid
net.seedSignaling(50, 50);  // Seed signaling cell at center

for (let i = 0; i < 1000; i++) {
  net.grow(0.1);  // 10% growth rate
}
net.differentiate();
net.prune(0.1);  // 10% pruning threshold

// Time Crystal Coordinator
const crystal = new WasmTimeCrystal(10, 100);  // 10 oscillators, 100ms period
crystal.crystallize();

for (let i = 0; i < 200; i++) {
  const pattern = crystal.tick();
  // Use pattern for coordination decisions
}

console.log(`Synchronization: ${crystal.orderParameter()}`);

// Combined Ecosystem (all three working together)
const eco = new ExoticEcosystem(5, 50, 8);  // 5 agents, 50x50 grid, 8 oscillators
eco.crystallize();

for (let i = 0; i < 100; i++) {
  eco.step();
}

console.log(eco.summaryJson());
```

### ruvector-nervous-system-wasm

**Bio-inspired neural system components for browser execution.**

| Component | Performance | Description |
|-----------|-------------|-------------|
| **BTSP** | Immediate | Behavioral Timescale Synaptic Plasticity for one-shot learning |
| **HDC** | <50ns bind, <100ns similarity | Hyperdimensional Computing with 10,000-bit vectors |
| **WTA** | <1us | Winner-Take-All for instant decisions |
| **K-WTA** | <10us | K-Winner-Take-All for sparse distributed coding |
| **Global Workspace** | <10us | 4-7 item attention bottleneck (Miller's Law) |

**Hyperdimensional Computing:**
- 10,000-bit binary hypervectors
- 10^40 representational capacity
- XOR binding (associative, commutative, self-inverse)
- Hamming distance similarity with SIMD optimization

**Biological References:**
- BTSP: Bittner et al. 2017 - Hippocampal place fields
- HDC: Kanerva 1988, Plate 2003 - Hyperdimensional computing
- WTA: Cortical microcircuits - Lateral inhibition
- Global Workspace: Baars 1988, Dehaene 2014 - Consciousness

**JavaScript/TypeScript Example:**

```typescript
import init, {
  BTSPLayer,
  Hypervector,
  HdcMemory,
  WTALayer,
  KWTALayer,
  GlobalWorkspace,
  WorkspaceItem,
} from '@ruvector/nervous-system-wasm';

await init();

// One-shot learning with BTSP
const btsp = new BTSPLayer(100, 2000.0);  // 100 dim, 2000ms tau
const pattern = new Float32Array(100).fill(0.1);
btsp.one_shot_associate(pattern, 1.0);  // Immediate association
const output = btsp.forward(pattern);

// Hyperdimensional Computing
const apple = Hypervector.random();
const orange = Hypervector.random();
const fruit = apple.bind(orange);  // XOR binding

const similarity = apple.similarity(orange);  // ~0.0 (orthogonal)
console.log(`Similarity: ${similarity}`);  // Random vectors are orthogonal

// HDC Memory
const memory = new HdcMemory();
memory.store("apple", apple);
memory.store("orange", orange);

const results = memory.retrieve(apple, 0.9);  // threshold 0.9
const topK = memory.top_k(fruit, 3);  // top-3 similar

// Instant decisions with WTA
const wta = new WTALayer(1000, 0.5, 0.8);  // 1000 neurons, threshold, inhibition
const activations = new Float32Array(1000);
// ... fill activations ...
const winner = wta.compete(activations);

// Sparse coding with K-WTA
const kwta = new KWTALayer(1000, 50);  // 1000 neurons, k=50 winners
const winners = kwta.select(activations);

// Attention bottleneck with Global Workspace
const workspace = new GlobalWorkspace(7);  // Miller's Law: 7 +/- 2
const item = new WorkspaceItem(
  new Float32Array([1, 2, 3]),  // content
  0.9,  // salience
  1,    // source
  Date.now()  // timestamp
);
workspace.broadcast(item);
```

### ruvector-attention-unified-wasm

**Unified API for 18+ attention mechanisms across Neural, DAG, Graph, and SSM domains.**

A single WASM interface that routes to the appropriate attention implementation based on your data structure and requirements.

| Category | Mechanisms | Best For |
|----------|------------|----------|
| **Neural** | Scaled Dot-Product, Multi-Head, Hyperbolic, Linear, Flash, Local-Global, MoE | Transformers, sequences |
| **DAG** | Topological, Causal Cone, Critical Path, MinCut-Gated, Hierarchical Lorentz, Parallel Branch, Temporal BTSP | Query DAGs, workflows |
| **Graph** | GAT, GCN, GraphSAGE | GNNs, knowledge graphs |
| **SSM** | Mamba | Long sequences, streaming |

**Mechanism Selection:**

```
+------------------+     +-------------------+
|   Your Data      | --> | UnifiedAttention  | --> Optimal Mechanism
+------------------+     +-------------------+
                               |
        +----------------------+----------------------+
        |                      |                      |
   +----v----+           +-----v-----+          +-----v----+
   | Neural  |           |    DAG    |          |  Graph   |
   +---------+           +-----------+          +----------+
   | dot_prod|           | topological|         | gat      |
   | multi_hd|           | causal_cone|         | gcn      |
   | flash   |           | mincut_gtd |         | graphsage|
   +---------+           +-----------+          +----------+
```

**JavaScript/TypeScript Example:**

```typescript
import init, {
  UnifiedAttention,
  availableMechanisms,
  getStats,
  softmax,
  temperatureSoftmax,
  cosineSimilarity,
  // Neural attention
  ScaledDotProductAttention,
  MultiHeadAttention,
  // DAG attention
  TopologicalAttention,
  MinCutGatedAttention,
  // Graph attention
  GraphAttention,
  // SSM
  MambaSSM,
} from '@ruvector/attention-unified-wasm';

await init();

// List all available mechanisms
console.log(availableMechanisms());
// { neural: [...], dag: [...], graph: [...], ssm: [...] }

console.log(getStats());
// { total_mechanisms: 18, neural_count: 7, dag_count: 7, ... }

// Unified selector - routes to appropriate implementation
const attention = new UnifiedAttention("multi_head");
console.log(`Category: ${attention.category}`);  // "neural"
console.log(`Supports sequences: ${attention.supportsSequences()}`);  // true
console.log(`Supports graphs: ${attention.supportsGraphs()}`);  // false

// For DAG structures
const dagAttention = new UnifiedAttention("topological");
console.log(`Category: ${dagAttention.category}`);  // "dag"
console.log(`Supports graphs: ${dagAttention.supportsGraphs()}`);  // true

// Hyperbolic attention for hierarchical data
const hypAttention = new UnifiedAttention("hierarchical_lorentz");
console.log(`Supports hyperbolic: ${hypAttention.supportsHyperbolic()}`);  // true

// Utility functions
const logits = [1.0, 2.0, 3.0, 4.0];
const probs = softmax(logits);
console.log(`Probabilities sum to: ${probs.reduce((a, b) => a + b)}`);  // 1.0

// Temperature-scaled softmax (lower = more peaked)
const sharperProbs = temperatureSoftmax(logits, 0.5);

// Cosine similarity
const vecA = [1.0, 0.0, 0.0];
const vecB = [1.0, 0.0, 0.0];
console.log(`Similarity: ${cosineSimilarity(vecA, vecB)}`);  // 1.0
```

### WASM Package Summary

| Package | Size Target | Key Features |
|---------|-------------|--------------|
| `@ruvector/learning-wasm` | <50KB | MicroLoRA (<100us), trajectory tracking |
| `@ruvector/economy-wasm` | <100KB | CRDT ledger, 10x->1x curve, stake/slash |
| `@ruvector/exotic-wasm` | <150KB | NAO, Morphogenetic, Time Crystal |
| `@ruvector/nervous-system-wasm` | <100KB | BTSP, HDC (10K-bit), WTA, Global Workspace |
| `@ruvector/attention-unified-wasm` | <200KB | 18+ attention mechanisms, unified API |

**Common Patterns:**

```typescript
// All packages follow the same initialization pattern
import init, { /* exports */ } from '@ruvector/<package>-wasm';
await init();

// Version check
import { version } from '@ruvector/<package>-wasm';
console.log(`Version: ${version()}`);

// Feature discovery
import { available_mechanisms } from '@ruvector/<package>-wasm';
console.log(available_mechanisms());
```

</details>

---

## Self-Learning Systems

<details>
<summary>🧠 Self-Learning Intelligence Hooks</summary>

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
| [ruvector-cli hooks](./crates/ruvector-cli) | Rust CLI with 34 hooks commands | [![crates.io](https://img.shields.io/crates/v/ruvector-cli.svg)](https://crates.io/crates/ruvector-cli) |
| [@ruvector/cli hooks](./npm/packages/cli) | npm CLI with 29 hooks commands | [![npm](https://img.shields.io/npm/v/@ruvector/cli.svg)](https://www.npmjs.com/package/@ruvector/cli) |

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
ruvector hooks init [--force] [--postgres]  # Initialize hooks (--postgres for DB schema)
ruvector hooks install                   # Install into Claude settings

# Core
ruvector hooks stats                     # Show intelligence statistics
ruvector hooks session-start [--resume]  # Start/resume a session
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
ruvector hooks suggest-context           # UserPromptSubmit context injection
ruvector hooks track-notification        # Track notification patterns
ruvector hooks pre-compact [--auto]      # Pre-compact hook (auto/manual)

# Claude Code v2.0.55+ Features
ruvector hooks lsp-diagnostic --file <f> --severity error  # LSP diagnostics
ruvector hooks suggest-ultrathink "complex task"  # Recommend extended reasoning
ruvector hooks async-agent --action spawn --agent-id <id>  # Async sub-agents

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

**All 7 Claude Code hooks covered:**
| Hook | When It Fires | What RuVector Does |
|------|---------------|-------------------|
| `PreToolUse` | Before file edit, command, or Task | Suggests agent, shows related files, validates agent assignments |
| `PostToolUse` | After file edit or command | Records outcome, updates Q-values, injects context |
| `SessionStart` | When session begins/resumes | Loads intelligence, shows stats (startup vs resume) |
| `Stop` | When session ends | Saves state, exports metrics |
| `PreCompact` | Before context compaction | Preserves critical memories (auto vs manual) |
| `UserPromptSubmit` | Before processing user prompt | Injects learned patterns as context |
| `Notification` | On system notifications | Tracks notification patterns |

**Advanced Features:**
- **Stdin JSON Parsing**: Hooks receive full JSON via stdin (session_id, tool_input, tool_response)
- **Context Injection**: PostToolUse returns `additionalContext` to inject into Claude's context
- **Timeout Optimization**: All hooks have optimized timeouts (1-5 seconds vs 60s default)

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

# Initialize PostgreSQL schema (automatic)
ruvector hooks init --postgres

# Or apply schema manually
psql $RUVECTOR_POSTGRES_URL -f crates/ruvector-cli/sql/hooks_schema.sql

# Build CLI with postgres feature
cargo build -p ruvector-cli --features postgres
```

The PostgreSQL backend provides:
- Vector embeddings with native `ruvector` type
- Q-learning functions (`ruvector_hooks_update_q`, `ruvector_hooks_best_action`)
- Swarm coordination tables with foreign key relationships
- Automatic memory cleanup (keeps last 5000 entries)

</details>

---

## Additional Modules

<details>
<summary>🔬 Scientific OCR (SciPix)</summary>

| Package | Description | Install |
|---------|-------------|---------|
| [ruvector-scipix](./examples/scipix) | Rust OCR engine for scientific documents | `cargo add ruvector-scipix` |
| [@ruvector/scipix](https://www.npmjs.com/package/@ruvector/scipix) | TypeScript client for SciPix API | `npm install @ruvector/scipix` |

**SciPix** extracts text and mathematical equations from images, converting them to LaTeX, MathML, or plain text.

**Features:**
- **Multi-format output** — LaTeX, MathML, AsciiMath, plain text, structured JSON
- **Batch processing** — Process multiple images with parallel execution
- **Content detection** — Equations, tables, diagrams, mixed content
- **Confidence scoring** — Per-region confidence levels (high/medium/low)
- **PDF support** — Extract from multi-page PDFs with page selection

```typescript
import { SciPixClient, OutputFormat } from '@ruvector/scipix';

const client = new SciPixClient({
  baseUrl: 'http://localhost:8080',
  apiKey: 'your-api-key',
});

// OCR an image file
const result = await client.ocrFile('./equation.png', {
  formats: [OutputFormat.LaTeX, OutputFormat.MathML],
  detectEquations: true,
});

console.log('LaTeX:', result.latex);
console.log('Confidence:', result.confidence);

// Quick LaTeX extraction
const latex = await client.extractLatex('./math.png');

// Batch processing
const batchResult = await client.batchOcr({
  images: [
    { source: 'base64...', id: 'eq1' },
    { source: 'base64...', id: 'eq2' },
  ],
  defaultOptions: { formats: [OutputFormat.LaTeX] },
});
```

```bash
# Rust CLI usage
scipix-cli ocr --input equation.png --format latex
scipix-cli serve --port 3000

# MCP server for Claude/AI assistants
scipix-cli mcp
claude mcp add scipix -- scipix-cli mcp
```

See [npm/packages/scipix/README.md](./npm/packages/scipix/README.md) for full documentation.

</details>

<details>
<summary>🔗 ONNX Embeddings</summary>

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

</details>

<details>
<summary>🔧 Bindings & Tools</summary>

**Native bindings and tools** for integrating RuVector into any environment — Node.js, browsers, CLI, or as an HTTP/gRPC server.

| Crate | Description | crates.io |
|-------|-------------|-----------|
| [ruvector-node](./crates/ruvector-node) | Native Node.js bindings via napi-rs | [![crates.io](https://img.shields.io/crates/v/ruvector-node.svg)](https://crates.io/crates/ruvector-node) |
| [ruvector-wasm](./crates/ruvector-wasm) | WASM bindings for browsers & edge | [![crates.io](https://img.shields.io/crates/v/ruvector-wasm.svg)](https://crates.io/crates/ruvector-wasm) |
| [ruvllm-wasm](./crates/ruvllm-wasm) | Browser LLM inference with WebGPU | [![crates.io](https://img.shields.io/crates/v/ruvllm-wasm.svg)](https://crates.io/crates/ruvllm-wasm) |
| [ruvector-cli](./crates/ruvector-cli) | Command-line interface | [![crates.io](https://img.shields.io/crates/v/ruvector-cli.svg)](https://crates.io/crates/ruvector-cli) |
| [ruvector-server](./crates/ruvector-server) | HTTP/gRPC server | [![crates.io](https://img.shields.io/crates/v/ruvector-server.svg)](https://crates.io/crates/ruvector-server) |

**Node.js (Native Performance)**
```bash
npm install @ruvector/node
```
```javascript
const { RuVector } = require('@ruvector/node');
const db = new RuVector({ dimensions: 1536 });
db.insert('doc1', embedding, { title: 'Hello' });
const results = db.search(queryEmbedding, 10);
```

**Browser (WASM)**
```bash
npm install @ruvector/wasm
```
```javascript
import { RuVectorWasm } from '@ruvector/wasm';
const db = await RuVectorWasm.create({ dimensions: 384 });
await db.insert('doc1', embedding);
const results = await db.search(query, 5);
```

**CLI**
```bash
cargo install ruvector-cli
ruvector init mydb --dim 1536
ruvector insert mydb --file embeddings.json
ruvector search mydb --query "[0.1, 0.2, ...]" --limit 10
```

**HTTP Server**
```bash
cargo install ruvector-server
ruvector-server --port 8080 --data ./vectors

# REST API
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "limit": 10}'
```

</details>

---

## Examples & Tutorials

<details>
<summary>📚 Production Examples</summary>

28 production-ready examples demonstrating RuVector integration patterns.

| Example | Description | Type |
|---------|-------------|------|
| [agentic-jujutsu](./examples/agentic-jujutsu) | Quantum-resistant version control for AI agents (23x faster than Git) | Rust |
| [mincut](./examples/mincut) | 6 self-organizing network demos: strange loops, time crystals, causal discovery | Rust |
| [subpolynomial-time](./examples/subpolynomial-time) | n^0.12 subpolynomial algorithm demos | Rust |
| [exo-ai-2025](./examples/exo-ai-2025) | Cognitive substrate with 9 neural-symbolic crates + 11 research experiments | Rust/TS |
| [neural-trader](./examples/neural-trader) | AI trading with DRL + sentiment analysis + SONA learning | Rust |
| [ultra-low-latency-sim](./examples/ultra-low-latency-sim) | 13+ quadrillion meta-simulations/sec with SIMD | Rust |
| [meta-cognition-spiking-neural-network](./examples/meta-cognition-spiking-neural-network) | Spiking neural network with meta-cognitive learning (10-50x speedup) | npm |
| [spiking-network](./examples/spiking-network) | Biologically-inspired spiking neural networks | Rust |
| [ruvLLM](./examples/ruvLLM) | LLM integration patterns for RAG and AI agents | Rust |
| [onnx-embeddings](./examples/onnx-embeddings) | Production ONNX embedding generation without Python | Rust |
| [onnx-embeddings-wasm](./examples/onnx-embeddings-wasm) | WASM ONNX embeddings for browsers | WASM |
| [refrag-pipeline](./examples/refrag-pipeline) | RAG pipeline with vector search and document processing | Rust |
| [scipix](./examples/scipix) | Scientific OCR: equations → LaTeX/MathML with ONNX inference | Rust |
| [graph](./examples/graph) | Graph database examples with Cypher queries | Rust |
| [edge](./examples/edge) | 364KB WASM edge deployment | Rust |
| [edge-full](./examples/edge-full) | Full-featured edge vector DB | Rust |
| [edge-net](./examples/edge-net) | Networked edge deployment with zero-cost swarms | Rust |
| [vibecast-7sense](./examples/vibecast-7sense) | 7-sense perception AI application | TypeScript |
| [apify](./examples/apify) | 13 Apify actors: trading, memory engine, synth data, market research | npm |
| [google-cloud](./examples/google-cloud) | GCP templates for Cloud Run, GKE, Vertex AI | Terraform |
| [wasm-react](./examples/wasm-react) | React integration with WASM vector operations | WASM |
| [wasm-vanilla](./examples/wasm-vanilla) | Vanilla JS WASM example for browser vector search | WASM |
| [wasm](./examples/wasm) | Core WASM examples and bindings | WASM |
| [nodejs](./examples/nodejs) | Node.js integration examples | Node.js |
| [rust](./examples/rust) | Core Rust usage examples | Rust |

</details>

<details>
<summary>🎓 Tutorials</summary>

### Tutorial 1: Vector Search in 60 Seconds

```javascript
import { VectorDB } from 'ruvector';

// Create DB with 384-dimensional vectors
const db = new VectorDB(384);

// Add vectors
db.insert('doc1', [0.1, 0.2, ...]);  // 384 floats
db.insert('doc2', [0.3, 0.1, ...]);

// Search (returns top 5 nearest neighbors)
const results = db.search(queryVector, 5);
// -> [{ id: 'doc1', score: 0.95 }, { id: 'doc2', score: 0.87 }]
```

### Tutorial 2: Graph Queries with Cypher

```javascript
import { GraphDB } from 'ruvector';

const graph = new GraphDB();

// Create nodes and relationships
graph.query(`
  CREATE (a:Person {name: 'Alice', embedding: $emb1})
  CREATE (b:Person {name: 'Bob', embedding: $emb2})
  CREATE (a)-[:KNOWS {since: 2020}]->(b)
`, { emb1: aliceVector, emb2: bobVector });

// Hybrid query: graph traversal + vector similarity
const results = graph.query(`
  MATCH (p:Person)-[:KNOWS*1..3]->(friend)
  WHERE vector.similarity(friend.embedding, $query) > 0.8
  RETURN friend.name, vector.similarity(friend.embedding, $query) as score
  ORDER BY score DESC
`, { query: queryVector });
```

### Tutorial 3: Self-Learning with SONA

```rust
use ruvector_sona::{SonaEngine, SonaConfig};

// Initialize SONA with LoRA adapters
let sona = SonaEngine::with_config(SonaConfig {
    hidden_dim: 256,
    lora_rank: 8,
    ewc_lambda: 0.4,  // Elastic Weight Consolidation
    ..Default::default()
});

// Record successful action
let mut trajectory = sona.begin_trajectory(query_embedding);
trajectory.add_step(result_embedding, vec![], 1.0);  // reward=1.0
sona.end_trajectory(trajectory, true);  // success=true

// SONA learns and improves future predictions
sona.force_learn();

// Later: get improved predictions
let prediction = sona.predict(&new_query_embedding);
```

### Tutorial 4: Dynamic Min-Cut (n^0.12 Updates)

```rust
use ruvector_mincut::{DynamicMinCut, Graph};

// Build graph
let mut graph = Graph::new(100);  // 100 nodes
graph.add_edge(0, 1, 10.0);
graph.add_edge(1, 2, 5.0);
graph.add_edge(0, 2, 15.0);

// Compute initial min-cut
let mut mincut = DynamicMinCut::new(&graph);
let (value, cut_edges) = mincut.compute();
println!("Min-cut value: {}", value);  // -> 15.0

// Dynamic update - subpolynomial time O(n^0.12)!
graph.update_edge(1, 2, 20.0);
let (new_value, _) = mincut.recompute();  // Much faster than recomputing from scratch
```

### Tutorial 5: 39 Attention Mechanisms

```rust
use ruvector_attention::{
    Attention, FlashAttention, LinearAttention,
    HyperbolicAttention, GraphAttention, MinCutGatedAttention
};

// FlashAttention - O(n) memory, fastest for long sequences
let flash = FlashAttention::new(512, 8);  // dim=512, heads=8
let output = flash.forward(&query, &key, &value);

// LinearAttention - O(n) time complexity
let linear = LinearAttention::new(512, 8);

// HyperbolicAttention - for hierarchical data (Poincaré ball)
let hyper = HyperbolicAttention::new(512, 8, Curvature(-1.0));

// GraphAttention - respects graph structure
let gat = GraphAttention::new(512, 8, &adjacency_matrix);

// MinCutGatedAttention - 50% compute reduction via sparsity
let mincut_gated = MinCutGatedAttention::new(512, 8, sparsity: 0.5);
let sparse_output = mincut_gated.forward(&query, &key, &value);
```

### Tutorial 6: Spiking Neural Networks

```javascript
import { SpikingNetwork, HDCEncoder } from '@ruvector/spiking-neural';

// High-Dimensional Computing encoder (10K-bit vectors)
const encoder = new HDCEncoder(10000);
const encoded = encoder.encode("hello world");

// Spiking network with BTSP learning
const network = new SpikingNetwork({
  layers: [784, 256, 10],
  learning: 'btsp',  // Behavioral Time-Scale Plasticity
  threshold: 1.0
});

// Train with spike timing
network.train(spikes, labels, { epochs: 10 });

// Inference
const output = network.forward(inputSpikes);
```

### Tutorial 7: Claude Code Hooks Integration

```bash
# 1. Initialize hooks
npx @ruvector/cli hooks init

# 2. Install into Claude settings
npx @ruvector/cli hooks install

# 3. Hooks now capture:
#    - File edits (pre/post)
#    - Commands (pre/post)
#    - Sessions (start/end)
#    - Errors and fixes

# 4. Query learned patterns
npx @ruvector/cli hooks recall "authentication error"
# -> Returns similar past solutions

# 5. Get AI routing suggestions
npx @ruvector/cli hooks route "implement caching"
# -> Suggests: rust-developer (confidence: 0.89)
```

### Tutorial 8: Edge Deployment with rvLite

```javascript
import { RvLite } from '@ruvector/rvlite';

// Create persistent edge database (IndexedDB in browser)
const db = await RvLite.create({
  path: 'my-vectors.db',
  dimensions: 384
});

// Works offline - all computation local
await db.insert('doc1', embedding1, { title: 'Hello' });
await db.insert('doc2', embedding2, { title: 'World' });

// Semantic search with metadata filtering
const results = await db.search(queryEmbedding, {
  limit: 10,
  filter: { title: { $contains: 'Hello' } }
});

// Sync when online
await db.sync('https://api.example.com/vectors');
```

</details>

<details>
<summary>🍕 WASM & Utility Packages</summary>

| Package | Description | Version | Downloads |
|---------|-------------|---------|-----------|
| [@ruvector/wasm](https://www.npmjs.com/package/@ruvector/wasm) | WASM core vector DB | [![npm](https://img.shields.io/npm/v/@ruvector/wasm.svg)](https://www.npmjs.com/package/@ruvector/wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/wasm.svg)](https://www.npmjs.com/package/@ruvector/wasm) |
| [@ruvector/gnn-wasm](https://www.npmjs.com/package/@ruvector/gnn-wasm) | WASM GNN layers | [![npm](https://img.shields.io/npm/v/@ruvector/gnn-wasm.svg)](https://www.npmjs.com/package/@ruvector/gnn-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/gnn-wasm.svg)](https://www.npmjs.com/package/@ruvector/gnn-wasm) |
| [@ruvector/graph-wasm](https://www.npmjs.com/package/@ruvector/graph-wasm) | WASM graph DB | [![npm](https://img.shields.io/npm/v/@ruvector/graph-wasm.svg)](https://www.npmjs.com/package/@ruvector/graph-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/graph-wasm.svg)](https://www.npmjs.com/package/@ruvector/graph-wasm) |
| [@ruvector/attention-wasm](https://www.npmjs.com/package/@ruvector/attention-wasm) | WASM attention | [![npm](https://img.shields.io/npm/v/@ruvector/attention-wasm.svg)](https://www.npmjs.com/package/@ruvector/attention-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/attention-wasm.svg)](https://www.npmjs.com/package/@ruvector/attention-wasm) |
| [@ruvector/tiny-dancer-wasm](https://www.npmjs.com/package/@ruvector/tiny-dancer-wasm) | WASM AI routing | [![npm](https://img.shields.io/npm/v/@ruvector/tiny-dancer-wasm.svg)](https://www.npmjs.com/package/@ruvector/tiny-dancer-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/tiny-dancer-wasm.svg)](https://www.npmjs.com/package/@ruvector/tiny-dancer-wasm) |
| [@ruvector/router-wasm](https://www.npmjs.com/package/@ruvector/router-wasm) | WASM semantic router | [![npm](https://img.shields.io/npm/v/@ruvector/router-wasm.svg)](https://www.npmjs.com/package/@ruvector/router-wasm) | [![downloads](https://img.shields.io/npm/dt/@ruvector/router-wasm.svg)](https://www.npmjs.com/package/@ruvector/router-wasm) |
| [@ruvector/postgres-cli](https://www.npmjs.com/package/@ruvector/postgres-cli) | Postgres extension CLI | [![npm](https://img.shields.io/npm/v/@ruvector/postgres-cli.svg)](https://www.npmjs.com/package/@ruvector/postgres-cli) | [![downloads](https://img.shields.io/npm/dt/@ruvector/postgres-cli.svg)](https://www.npmjs.com/package/@ruvector/postgres-cli) |
| [@ruvector/agentic-synth](https://www.npmjs.com/package/@ruvector/agentic-synth) | Synthetic data generator | [![npm](https://img.shields.io/npm/v/@ruvector/agentic-synth.svg)](https://www.npmjs.com/package/@ruvector/agentic-synth) | [![downloads](https://img.shields.io/npm/dt/@ruvector/agentic-synth.svg)](https://www.npmjs.com/package/@ruvector/agentic-synth) |
| [@ruvector/graph-data-generator](https://www.npmjs.com/package/@ruvector/graph-data-generator) | Graph data generation | [![npm](https://img.shields.io/npm/v/@ruvector/graph-data-generator.svg)](https://www.npmjs.com/package/@ruvector/graph-data-generator) | [![downloads](https://img.shields.io/npm/dt/@ruvector/graph-data-generator.svg)](https://www.npmjs.com/package/@ruvector/graph-data-generator) |
| [@ruvector/agentic-integration](https://www.npmjs.com/package/@ruvector/agentic-integration) | Agentic workflows | [![npm](https://img.shields.io/npm/v/@ruvector/agentic-integration.svg)](https://www.npmjs.com/package/@ruvector/agentic-integration) | [![downloads](https://img.shields.io/npm/dt/@ruvector/agentic-integration.svg)](https://www.npmjs.com/package/@ruvector/agentic-integration) |
| [rvlite](https://www.npmjs.com/package/rvlite) | SQLite-style edge DB (SQL/SPARQL/Cypher) | [![npm](https://img.shields.io/npm/v/rvlite.svg)](https://www.npmjs.com/package/rvlite) | [![downloads](https://img.shields.io/npm/dt/rvlite.svg)](https://www.npmjs.com/package/rvlite) |


**Platform-specific native bindings** (auto-detected):
- `@ruvector/node-linux-x64-gnu`, `@ruvector/node-linux-arm64-gnu`, `@ruvector/node-darwin-x64`, `@ruvector/node-darwin-arm64`, `@ruvector/node-win32-x64-msvc`
- `@ruvector/gnn-linux-x64-gnu`, `@ruvector/gnn-linux-arm64-gnu`, `@ruvector/gnn-darwin-x64`, `@ruvector/gnn-darwin-arm64`, `@ruvector/gnn-win32-x64-msvc`
- `@ruvector/tiny-dancer-linux-x64-gnu`, `@ruvector/tiny-dancer-linux-arm64-gnu`, `@ruvector/tiny-dancer-darwin-x64`, `@ruvector/tiny-dancer-darwin-arm64`, `@ruvector/tiny-dancer-win32-x64-msvc`
- `@ruvector/router-linux-x64-gnu`, `@ruvector/router-linux-arm64-gnu`, `@ruvector/router-darwin-x64`, `@ruvector/router-darwin-arm64`, `@ruvector/router-win32-x64-msvc`
- `@ruvector/attention-linux-x64-gnu`, `@ruvector/attention-linux-arm64-gnu`, `@ruvector/attention-darwin-x64`, `@ruvector/attention-darwin-arm64`, `@ruvector/attention-win32-x64-msvc`
- `@ruvector/ruvllm-linux-x64-gnu`, `@ruvector/ruvllm-linux-arm64-gnu`, `@ruvector/ruvllm-darwin-x64`, `@ruvector/ruvllm-darwin-arm64`, `@ruvector/ruvllm-win32-x64-msvc`

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

</details>

<details>
<summary>🦀 Rust Usage Examples</summary>

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

</details>


<details>
<summary>🎓 RuvLLM Training & RLM Fine-Tuning Tutorials </summary>

#### Hybrid Routing (90% Accuracy)

RuvLTRA achieves **90% routing accuracy** using a keyword-first strategy with embedding fallback:

```javascript
// Optimal routing: Keywords first, embeddings as tiebreaker
function routeTask(task, taskEmbedding, agentEmbeddings) {
  const keywordScores = getKeywordScores(task);
  const maxKw = Math.max(...Object.values(keywordScores));

  if (maxKw > 0) {
    const candidates = Object.entries(keywordScores)
      .filter(([_, score]) => score === maxKw)
      .map(([agent]) => agent);

    if (candidates.length === 1) return { agent: candidates[0] };
    return pickByEmbedding(candidates, taskEmbedding, agentEmbeddings);
  }

  return embeddingSimilarity(taskEmbedding, agentEmbeddings);
}
```

Run the benchmark: `node npm/packages/ruvllm/scripts/hybrid-model-compare.js`

#### Generate Training Data

```bash
# Using CLI (recommended)
npx @ruvector/ruvllm train stats              # View dataset statistics
npx @ruvector/ruvllm train dataset            # Export training data
npx @ruvector/ruvllm train contrastive        # Run full training pipeline

# With options
npx @ruvector/ruvllm train dataset --output ./my-training
npx @ruvector/ruvllm train contrastive --epochs 20 --batch-size 32 --lr 0.0001
```

**Programmatic API:**
```javascript
import { ContrastiveTrainer, generateTrainingDataset, getDatasetStats } from '@ruvector/ruvllm';

const stats = getDatasetStats();
console.log(`${stats.totalExamples} examples, ${stats.agentTypes} agent types`);

const trainer = new ContrastiveTrainer({ epochs: 10, margin: 0.5 });
trainer.addTriplet(anchor, anchorEmb, positive, positiveEmb, negative, negativeEmb, true);
const result = trainer.train();
trainer.exportTrainingData('./output');
```

#### Fine-Tune with LoRA

```bash
pip install transformers peft datasets accelerate

python -m peft.lora_train \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset ./data/training/routing-examples.jsonl \
  --output_dir ./ruvltra-routing-lora \
  --lora_r 8 --lora_alpha 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-4
```

#### Convert to GGUF

```bash
# Merge LoRA weights
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
model = PeftModel.from_pretrained(base, './ruvltra-routing-lora')
model.merge_and_unload().save_pretrained('./ruvltra-routing-merged')
"

# Convert and quantize
python llama.cpp/convert_hf_to_gguf.py ./ruvltra-routing-merged --outfile ruvltra-routing-f16.gguf
./llama.cpp/llama-quantize ruvltra-routing-f16.gguf ruvltra-routing-q4_k_m.gguf Q4_K_M
```

#### Contrastive Embedding Training

**Using RuvLLM CLI (recommended):**
```bash
# Full contrastive training pipeline with triplet loss
npx @ruvector/ruvllm train contrastive --output ./training-output

# Exports: triplets.jsonl, embeddings.json, lora_config.json, train.sh
```

**Using Python (for GPU training):**
```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

train_examples = [
    InputExample(texts=["implement login", "build auth component"], label=1.0),
    InputExample(texts=["implement login", "write unit tests"], label=0.0),
]

model = SentenceTransformer("Qwen/Qwen2.5-0.5B-Instruct")
train_loss = losses.CosineSimilarityLoss(model)
model.fit([(DataLoader(train_examples, batch_size=16), train_loss)], epochs=5)
```

**Resources:** [Issue #122](https://github.com/ruvnet/ruvector/issues/122) | [LoRA Paper](https://arxiv.org/abs/2106.09685) | [Sentence Transformers](https://www.sbert.net/docs/training/overview.html)

#### Rust Training Module

For production-scale dataset generation, use the Rust training module ([full docs](./crates/ruvllm/src/training/README.md)):

```rust
use ruvllm::training::{DatasetGenerator, DatasetConfig};

let config = DatasetConfig {
    examples_per_category: 100,
    enable_augmentation: true,
    seed: 42,
    ..Default::default()
};

let dataset = DatasetGenerator::new(config).generate();
let (train, val, test) = dataset.split(0.7, 0.15, 0.15, 42);
dataset.export_jsonl("training.jsonl")?;
```

**Features:**
- **5 agent categories**: Coder, Researcher, Security, Architecture, Reviewer (20% each)
- **Model routing**: Haiku (simple) → Sonnet (moderate) → Opus (complex/security)
- **Data augmentation**: Paraphrasing, complexity variations, domain transfer
- **8 technical domains**: Web, Systems, DataScience, Mobile, DevOps, Security, Database, API
- **Quality scores**: 0.80-0.96 based on template quality and category
- **Performance**: ~10,000 examples/second, ~50 MB/s JSONL export

```bash
cargo run --example generate_claude_dataset --release
# Outputs: train.jsonl, val.jsonl, test.jsonl, stats.json
```

</details>

---

## Project

<details>
<summary>📁 Project Structure</summary>

```
crates/
├── ruvector-core/           # Vector DB engine (HNSW, storage)
├── ruvector-graph/          # Graph DB + Cypher parser + Hyperedges
├── ruvector-gnn/            # GNN layers, compression, training
├── ruvector-tiny-dancer-core/  # AI agent routing (FastGRNN)
├── ruvector-*-wasm/         # WebAssembly bindings
└── ruvector-*-node/         # Node.js bindings (napi-rs)
```

</details>

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
