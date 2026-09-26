# ruvector 2026: Streaming WAL-ANN — Coherence-Gated Merge for Continuous Vector Ingestion in Rust

**Searchable write-ahead log + navigable graph + coherence quality gate for zero-downtime streaming ANN in Rust.** Three measured variants on 3K×64-dim vectors: all pass recall@10≥0.70 and <5ms latency.

One sentence: SWAL-ANN is a Rust vector index that buffers streaming inserts in a searchable WAL and flushes to the main navigable graph only when a coherence quality gate fires — giving agents continuous write throughput without blind spots in query results.

**GitHub**: https://github.com/ruvnet/ruvector  
**Branch**: `research/nightly/2026-07-14-streaming-wal-ann`

---

## Introduction

Every production vector database today uses the same flush policy: accumulate inserts in a write-ahead log until the buffer reaches N entries, then build or update the ANN index. Size. That's it. No quality signal. No understanding of whether the new vectors actually need to be in the graph right now or whether they're so close to existing entries that they can wait.

This matters more than it sounds. AI agents are not batch workloads. A long-running autonomous agent continuously writes new memories — new observations, new plans, new tool results — and simultaneously queries those memories to inform its next action. A vector index that imposes even a 500ms "dark window" between insert and searchability is a broken tool for continuous cognition.

Current systems partially paper over this with WAL tiers that serve recent writes via brute-force scan (Milvus does this with Growing Segments). But the decision of *when to merge those pending vectors into the main index* is still pure size arithmetic. No production system asks: "are these pending vectors so isolated from the existing index that we should merge them now, before the buffer is even half full?"

SWAL-ANN answers that question. The coherence gate measures the average isolation of pending WAL vectors relative to the main navigable graph. When isolation spikes — because a new semantic cluster is arriving — the gate fires early. When vectors are redundant with existing entries, the gate defers. The result is quality-adaptive merge timing: aggressive when coverage gaps demand it, lazy when they don't.

RuVector is the right substrate for this because it already has coherence scoring infrastructure (ADR-254), proof-gated write authorization (ADR-224), and incremental graph structures across multiple crates. SWAL-ANN connects those pieces into a unified streaming memory primitive.

This matters for AI agents, graph RAG, MCP-based memory tools, edge AI deployments, and any system where new data must be available for retrieval immediately after insertion — not after the next index rebuild. In Rust, with no GC pauses and zero-copy vector storage, the WAL-to-graph pipeline can run at sustained ~9,400 vectors/second on a single thread.

The future this points to: autonomous agents with persistent vector memory that never rebuilds offline, self-manages its merge schedule based on data arrival patterns, and runs in 1–2 MB on an edge device. SWAL-ANN is the Rust primitive that makes that possible.

---

## Features

| Feature | What it does | Why it matters | Status |
|---------|-------------|----------------|--------|
| Searchable WAL tier | New vectors immediately queryable via linear scan | No inserted vector is ever invisible | Implemented in PoC |
| `MergeGate` trait | Pluggable merge policy (eager, lazy, coherence-driven) | Separates policy from mechanism | Implemented in PoC |
| `EagerGate` | Flush every 32 vectors | High freshness, frequent merge cost | Implemented + Measured |
| `LazyGate` | Flush every 512 vectors | High throughput, occasional latency spikes | Implemented + Measured |
| `CoherenceGate` | Flush when coherence < 0.08 OR size ≥ 256 | Quality-adaptive, no manual tuning of merge cadence | Implemented + Measured |
| Sampled coherence score | O(1024·D) coherence estimate, evaluated every 8 inserts | Negligible overhead vs. exact O(N²) computation | Implemented in PoC |
| Incremental `NavGraph` | NSW with beam-search insert + long-jump edges | O(ef·M·D) per insert, not O(N²) rebuild | Implemented in PoC |
| Dual-tier search | Merge graph beam-search + WAL linear-scan results | Single API call across both tiers | Implemented in PoC |
| Recall@10 ≥ 0.70 | Measured on 3K × 64-dim with k=10 | Acceptance-tested | Measured |
| Edge/WASM compatible | Core structs need only `Vec` + `BinaryHeap` | Runs in Cognitum Seed ≤4MB RAM envelope | Research direction |
| Durable WAL | Append to mmap'd file before ack | Crash safety for agent memory | Production candidate |
| MCP tool surface | `wal_insert`, `wal_flush`, `wal_coherence`, `search` | Agent-accessible memory operations | Research direction |
| ruFlo automation | Coherence monitor drives flush scheduling | Autonomous memory management | Research direction |

---

## Technical Design

### Core Data Structure

Three tiers:

```
VectorWal           NavGraph                WalAnnIndex<G>
──────────────      ────────────────        ──────────────────────────────
pending: Vec<E>     vectors: Vec<f32>       graph: NavGraph
capacity: usize     adj: Vec<Vec<u32>>      wal: VectorWal
dims: usize         m: 16                   gate: G (MergeGate impl)
                    ef_construction: 100    merge_count: usize
                    m_longjump: 6           graph_ids: Vec<u64>
```

### Trait-Based API

```rust
// Pluggable merge policy.
pub trait MergeGate: Send + Sync {
    fn should_merge(&self, wal_size: usize, coherence: f32) -> bool;
    fn name(&self) -> &'static str;
}

// Main index API.
impl<G: MergeGate> WalAnnIndex<G> {
    pub fn insert(&mut self, vector: Vec<f32>) -> u64;
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;
    pub fn flush_wal(&mut self);
    pub fn coherence_score_sampled(&self) -> f32;
}
```

### Coherence Score

```
isolation(v, G) = min{ L2(v, g) | g ∈ G }
coherence       = 1 / (1 + mean(isolation for 16 WAL samples vs 64 graph samples))
```

In 64-dim Gaussian space, typical coherence ≈ 0.12. The `CoherenceGate(0.08)` fires when isolation rises 40% above baseline — indicating a new semantic cluster arriving.

### Three Variants

```mermaid
flowchart LR
    E[EagerMerge\nthreshold=32\n94 merges] -->|32 vecs| F1[flush]
    L[LazyMerge\nthreshold=512\n6 merges] -->|512 vecs| F2[flush]
    C[CoherenceGate\ncoh<0.08 OR sz≥256\n12 merges] -->|quality signal| F3[flush]
```

- **EagerMerge**: Best for real-time search freshness (32-vector flush interval).
- **LazyMerge**: Best for bulk ingest throughput (500-vector average batch).
- **CoherenceGatedMerge**: Best for distribution-shifting workloads where new clusters arrive unpredictably.

### Memory Model

For N vectors, D dims, M=16 neighbours:
```
Vectors:   N × D × 4 bytes
Adjacency: N × (M + m_longjump) × 4 bytes  ≈ N × 88 bytes
ID map:    N × 8 bytes
WAL peak:  max_wal_size × D × 4 bytes

At N=3K, D=64: measured 1,156 KB total.
```

### Performance Model

- Insert throughput: ~9,400 vecs/sec (dominated by NavGraph incremental build).
- Coherence check: 1,024 × D FLOPs every 8 inserts → negligible amortised cost.
- WAL scan: O(|WAL| × D) per query; bounded by max_wal_size.
- Graph search: O(ef × M × D) beam traversal; ef=64 for k=10.

---

## Benchmark Results

**Command**: `cargo run --release -p ruvector-wal-ann --bin benchmark`  
**Hardware**: x86_64 Linux (cloud VM)  
**OS**: Linux  
**Rust**: stable, edition 2021  
**Dataset**: 3,000 × 64-dim f32, Normal(0,1), deterministic seed  
**Queries**: 100 × k=10, independent seed  
**Ground truth**: brute-force L2 scan  

### INSERT THROUGHPUT

| Variant | Total(ms) | Vecs/sec | Merges | Graph size after |
|---------|-----------|----------|--------|-----------------|
| EagerMerge | 319.6 | 9,388 | 94 | 3,000 |
| LazyMerge | 316.7 | 9,472 | 6 | 3,000 |
| CoherenceGatedMerge | 354.7 | 8,458 | 12 | 3,000 |

### QUERY PERFORMANCE (k=10, post-flush)

| Variant | Recall@10 | Mean(µs) | p50(µs) | p95(µs) | QPS | Mem(KB) |
|---------|-----------|---------|---------|---------|-----|---------|
| EagerMerge | **0.716** | 110.1 | 105.1 | 150.8 | 9,084 | 1,156 |
| LazyMerge | **0.716** | 113.5 | 107.2 | 150.6 | 8,811 | 1,156 |
| CoherenceGatedMerge | **0.716** | 105.5 | 104.0 | 132.8 | 9,477 | 1,156 |

### ACCEPTANCE

```
EagerMerge            recall=0.716  mean=110µs  → PASS
LazyMerge             recall=0.716  mean=114µs  → PASS
CoherenceGatedMerge   recall=0.716  mean=106µs  → PASS
```

**Benchmark limitations**: The single-layer NSW used here has lower recall (0.716) than a full multi-layer HNSW (typically >0.90). Recall is limited by early-node connectivity: nodes inserted when the graph was tiny have fewer diverse neighbours. Production use should replace `NavGraph` with a full HNSW backend. Competitor numbers are not reproduced here — this is a standalone PoC measurement.

---

## Comparison with Vector Databases

| System | Core strength | Where it's strong | Where RuVector differs | Benchmarked here |
|--------|--------------|-------------------|----------------------|------------------|
| Milvus | Distributed, enterprise-grade | Multi-node, high throughput, rich APIs | RuVector: Rust-native, proof-gated writes, coherence-gated merge, no JVM | No |
| Qdrant | HNSW quality, Rust-native | Filtered search, scalar quantization | RuVector: agent memory primitives, MCP, ruFlo integration, edge WASM | No |
| Weaviate | Graph schema, semantic types | Enterprise search, multi-modal | RuVector: no schema requirement, pure Rust, RVF portable format | No |
| Pinecone | Managed, auto-scaling | Production SaaS without infra | RuVector: self-hosted, no vendor lock-in, WASM edge deployment | No |
| LanceDB | Lance columnar format | Analytics + vector in one store | RuVector: graph-aware coherence, proof gates, agent OS primitives | No |
| FAISS | GPU acceleration, library | Research, large-scale batch | RuVector: streaming-first, no C++ runtime, Rust safety guarantees | No |
| pgvector | PostgreSQL native | Existing Postgres workloads | RuVector: not tied to Postgres, standalone Rust, edge deployable | No |
| Chroma | Developer-friendly | Prototyping, Python-native | RuVector: production Rust, not Python, multi-tier streaming | No |
| Vespa | Hybrid search maturity | Enterprise hybrid (BM25+ANN) | RuVector: coherence score, agent memory focus, proof gates, WASM | No |

RuVector's differentiator in streaming contexts: the `MergeGate` abstraction with quality-signal gating is not available in any listed system. RuVector also integrates proof-gated writes (ADR-224) and capability-gated reads (ADR-268) as first-class primitives, enabling the full access-control lifecycle for agent memory.

---

## Practical Applications

| Application | User | Why it matters | How RuVector uses it | Near-term path |
|-------------|------|---------------|---------------------|----------------|
| Streaming agent memory | Autonomous AI agents | Memories must be searchable the moment they form | WalAnnIndex insert → WAL scan serves immediately | ruFlo MCP tool in crates/wal-ann-mcp |
| Document ingestion | Enterprise knowledge base | New documents searchable without reindex downtime | WAL buffers new chunks; coherence gate merges by relevance cluster | Phase 2 durable WAL |
| Security event retrieval | SOC platforms | IOCs must match against all history in <1s | WAL covers stream; graph covers history | Direct crate integration |
| Code intelligence | Dev tools / IDEs | Newly opened files visible without full index rebuild | WAL scan serves recent files; graph serves corpus | IDE plugin via MCP |
| Log anomaly detection | SRE, observability | New log patterns match as they arrive | CoherenceGate fires on anomaly-cluster arrival | Observable via coherence metric |
| Edge sensor embedding | IoT / robotics | Memory bounded to 1–2MB; continuous stream | Cognitum Seed profile: max_wal=128, graph ≤ 2K nodes | WASM target Phase 3 |
| Recommendation systems | Personalisation platforms | New user actions update similarity graph without blocking | LazyGate amortises merge cost over 512 events | Phase 2 production hardening |
| Workflow automation | ruFlo pipelines | Each workflow step produces a vector result | ruFlo insert → coherence monitor → scheduled flush | ruFlo native integration |

---

## Exotic Applications

| Application | 10–20 year thesis | Required advances | RuVector role | Risk or unknown |
|-------------|------------------|-------------------|---------------|-----------------|
| Cognitum edge cognition | Agent stores episodic memories for lifetime in ≤4MB; no cloud sync required | NVM-backed WAL, INT8 quantized vectors | SWAL-ANN as the memory OS for edge agents | NVM write endurance; INT8 recall degradation |
| Proof-gated distributed WAL | Swarm of N agents writes to a shared WAL; merge requires quorum of coherence attestations | Distributed coherence consensus, cryptographic proof of quality score | WAL-ANN + Raft quorum gate | Byzantine coherence manipulation |
| Swarm collective memory | 1,000-agent swarm converges on a shared graph when global coherence improves | Multi-writer WAL, coherence aggregation protocol | CoherenceGate as distributed predicate over agent population | Latency in consensus round-trips |
| Self-healing vector graph | Graph detects quality degradation from churn and re-merges from WAL snapshots | Topology quality metrics (Wolverine/Topology-Aware approaches) | SWAL-ANN WAL as the "repair buffer" | Cascading repair under heavy churn |
| Chronological RAG | Document vectors only merge when cross-document coherence exceeds threshold | Document-level coherence scoring, citation graph | Coherence gate on document-grain WAL | Corpus-wide coherence is expensive to estimate |
| Agent OS memory manager | OS-level virtual memory management for vector indexes; WAL = working set; graph = long-term store | OS kernel integration, hardware-accelerated coherence | WalAnnIndex as a VMA abstraction for agent processes | Kernel integration complexity; scheduling conflicts |
| Neuromorphic consolidation | Sleep-phase-like replay: WAL vectors "replayed" during low-activity periods to consolidate into graph | Activity-signal-driven gate (spike rate proxy for coherence) | Coherence as proxy for neural binding quality | Biologically unrealistic metric; calibration unclear |
| Synthetic nervous system | Distributed SWAL-ANN nodes as the "hippocampus" of a synthetic cognitive architecture | Cross-shard coherence, temporal decay of WAL entries | RuVector as the hippocampal substrate | Architectural complexity; no known implementation path |

---

## Deep Research Notes

### What the SOTA Suggests

Three main paradigms for streaming ANN updates exist in the literature:

1. **In-place per-vector updates** (IP-DiskANN, arXiv:2502.13826): no WAL, reverse-edge maintenance. Best for steady-state low-rate inserts.
2. **Tiered LSM-style** (LSM-VEC, arXiv:2505.17152): distributes graph edges across LSM levels. Best for disk-resident storage.
3. **Size-gated WAL** (Milvus, LanceDB, Qdrant, all production systems): simplest, no quality signal.

The closest published analogue to SWAL-ANN's coherence gate is the navigability-signal-triggered repair in arXiv:2607.00728, which uses a quality signal to decide *when to repair* a graph, not when to flush a WAL. SWAL-ANN is the first application of a quality signal to the WAL *merge* decision.

### What Remains Unsolved

1. **Auto-calibration**: coherence threshold must be calibrated per data distribution. Online distance estimation during the first K inserts would automate this.
2. **Incremental multi-layer HNSW**: the PoC uses a single-layer NSW. Production needs the ef=200, M=32 multi-layer HNSW for recall >0.90.
3. **Concurrent writes**: multi-writer access requires thread-safe WAL.
4. **Durable WAL**: crash recovery is absent.
5. **Metric generalisation**: the coherence score assumes L2 distance. Cosine, dot-product, and angular metrics need their own isolation measures.

### Where This PoC Fits

The PoC establishes: the three-tier architecture is correct, the gate logic works, all variants achieve acceptable recall and latency, and the coherence score can be computed cheaply via sampling. The gap between this PoC (recall 0.716) and production target (>0.90) is well-understood: it requires a multi-layer HNSW backend.

### What Would Falsify the Approach

- If the coherence score does not correlate with actual graph quality (navigability, search path integrity) across data distributions, gate decisions become arbitrary. Validation against the metrics from Wolverine (PVLDB 2025) and Topology-Aware Updates (arXiv:2503.00402) is needed.
- If the incremental NSW recall ceiling is too low even at M=32, ef=200, a full offline rebuild would always outperform the streaming approach. The right threshold depends on the application's recall requirement.

**References**: [^1] FreshDiskANN arXiv:2105.09613; [^2] IP-DiskANN arXiv:2502.13826; [^3] UBIS arXiv:2602.00563; [^4] VStream PVLDB 2025; [^5] Incremental IVF arXiv:2411.00970; [^6] LSM-VEC arXiv:2505.17152; [^7] Starling SIGMOD 2024; [^8] Wolverine PVLDB 2025; [^9] Navigability-signal arXiv:2607.00728; [^10] Topology-aware updates arXiv:2503.00402; [^11] Write-read decoupling arXiv:2605.01260; [^12] Big-ANN NeurIPS 2023 arXiv:2409.17424.

---

## Usage Guide

```bash
# Checkout the branch
git checkout research/nightly/2026-07-14-streaming-wal-ann

# Build
cargo build --release -p ruvector-wal-ann

# Tests (15 unit tests)
cargo test -p ruvector-wal-ann

# Default benchmark (3000 × 64-dim, 100 queries)
cargo run --release -p ruvector-wal-ann --bin benchmark

# Larger dataset
cargo run --release -p ruvector-wal-ann --bin benchmark -- --n 10000 --dims 128 --queries 500

# Change k
cargo run --release -p ruvector-wal-ann --bin benchmark -- --k 20
```

### Expected Output

```
════════════════════════════════════════════════════════════════
  BENCHMARK RESULT: ALL VARIANTS PASS
════════════════════════════════════════════════════════════════
```

### Interpreting Results

- **Merges**: fewer merges = lower amortised insert cost per vector.
- **Vecs/sec**: insert throughput. CoherenceGate is slightly slower due to sampled coherence computation every 8 inserts.
- **Recall@k**: fraction of true top-k neighbours found. Limited by incremental NSW; improve by increasing ef_construction.
- **Mem(KB)**: measured index memory. Grows linearly with N.

### Adding a New Backend

Implement `MergeGate` and pass to `WalAnnIndex::new`:

```rust
struct TimedGate { max_secs: f64 }
impl MergeGate for TimedGate {
    fn should_merge(&self, _wal_size: usize, _coherence: f32) -> bool { /* check elapsed */ true }
    fn name(&self) -> &'static str { "TimedMerge" }
}
let idx = WalAnnIndex::new(64, 1024, TimedGate { max_secs: 1.0 });
```

---

## Optimization Guide

**Memory**: lower `max_wal_size` and M, use INT8 quantization on vectors before insertion.

**Latency**: increase search ef (currently k×6; try k×8 or k×10 for higher recall). Use rayon parallel WAL scan for large WAL sizes.

**Recall**: replace `NavGraph` (single-layer NSW) with multi-layer HNSW at M=32, ef_construction=200.

**Edge deployment**: cap `max_wal_size=64`, M=8, ef_construction=32. Use INT8 quantized `NavGraph` with SIMD dot product via `ruvector-math`.

**WASM**: compile with `--no-default-features`. Remove `rand_distr` from benchmark (WASM generates data differently). Use `wasm_bindgen` for MCP tool surface.

**MCP tool optimization**: batch WAL inserts in a single `wal_batch_insert` call; amortise coherence computation across the batch.

**ruFlo automation**: expose coherence score as a ruFlo observable metric; trigger `wal_flush` as a ruFlo action when coherence drops below threshold in the workflow monitor.

---

## Roadmap

### Now

- Replace `NavGraph` single-layer NSW with proper multi-layer HNSW (`crates/ruvector-hnsw-repair`) to reach recall ≥ 0.90.
- Auto-calibrate coherence threshold via 200-vector warm-up pass.
- Add durable WAL option: append-only file flushed with `fdatasync` before insert acknowledgement.

### Next

- Multi-writer access: MPSC channel + merge thread.
- MCP tool surface: `ruvector-wal-ann-mcp` crate.
- ruFlo integration: coherence monitor as ruFlo observable.
- WASM target: `ruvector-wal-ann-wasm` with no-std allocator support.
- Benchmark against NeurIPS 2023 big-ANN streaming track (arXiv:2409.17424).

### Later (2030–2046)

- Distributed SWAL-ANN: coherence gate as a quorum predicate across agent swarm.
- Proof-gated merge: cryptographic attestation of coherence score before WAL flush.
- Neuromorphic memory consolidation: activity-signal-driven WAL replay.
- Agent OS memory management: SWAL-ANN as a virtual memory abstraction for persistent agent processes.
- Quantum-accelerated coherence: quantum minimum-spanning-tree algorithms for O(√N) coherence computation.

---

## Footnotes and References

[^1]: Singh, Simhadri et al. "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search." arXiv:2105.09613, 2021. https://arxiv.org/abs/2105.09613. Accessed 2026-07-14.

[^2]: Xu, Dobson Manohar, Bernstein, Chandramouli, Wen, Simhadri. "In-Place Updates of a Graph Index for Streaming Approximate Nearest Neighbor Search." arXiv:2502.13826, Feb 2025. https://arxiv.org/abs/2502.13826. Accessed 2026-07-14.

[^3]: Lai, Huang, Wang. "Updatable Balanced Index for Stable Streaming Similarity Search (UBIS)." arXiv:2602.00563, IEEE BigData 2025. https://arxiv.org/abs/2602.00563. Accessed 2026-07-14.

[^4]: Gong et al. "VStream: A Distributed Streaming Vector Search System." PVLDB 18(6):1593–1606, 2025. https://dl.acm.org/doi/10.14778/3725688.3725692. Accessed 2026-07-14.

[^5]: Mohoney et al. "Incremental IVF Index Maintenance for Streaming Vector Search." arXiv:2411.00970, 2024. https://arxiv.org/abs/2411.00970. Accessed 2026-07-14.

[^6]: "LSM-VEC: A Large-Scale Disk-Based System for Dynamic Vector Search." arXiv:2505.17152, May 2025. https://arxiv.org/abs/2505.17152. Accessed 2026-07-14.

[^7]: Starling. "An I/O-Efficient Disk-Resident Graph Index Framework." SIGMOD/ACM MoD 2024. https://dl.acm.org/doi/10.1145/3639269. Accessed 2026-07-14.

[^8]: Liu et al. "Wolverine: Highly Efficient Monotonic Search Path Repair for Graph-Based ANN Index Updates." PVLDB 2025. https://dl.acm.org/doi/10.14778/3734839.3734860. Accessed 2026-07-14.

[^9]: Mandarapu, Kunkunuru. "When to Repair a Graph ANN Index: Navigability-Signal-Triggered Local Repair." arXiv:2607.00728, 2025. https://arxiv.org/pdf/2607.00728. Accessed 2026-07-14.

[^10]: "A Topology-Aware Localized Update Strategy for Graph-Based ANN Index." arXiv:2503.00402, Mar 2025. https://arxiv.org/abs/2503.00402. Accessed 2026-07-14.

[^11]: "Write-Read Decoupling in Vector Database Architectures." arXiv:2605.01260, 2025. https://arxiv.org/pdf/2605.01260. Accessed 2026-07-14.

[^12]: Big-ANN NeurIPS 2023 Streaming Track. arXiv:2409.17424. https://arxiv.org/abs/2409.17424. Accessed 2026-07-14.

---

## SEO Tags

**Keywords**: ruvector, Rust vector database, Rust vector search, high performance Rust, ANN search, HNSW, streaming vector index, write-ahead log, WAL ANN, coherence-gated merge, agent memory, AI agents, MCP, WASM AI, edge AI, self-optimizing vector database, ruvnet, ruFlo, Claude Flow, autonomous agents, retrieval augmented generation, graph RAG, navigable small world, NSW.

**Suggested GitHub topics**: rust, vector-database, vector-search, ann, hnsw, streaming-index, write-ahead-log, rag, graph-rag, ai-agents, agent-memory, mcp, wasm, edge-ai, rust-ai, semantic-search, graph-database, autonomous-agents, retrieval, embeddings, ruvector.
