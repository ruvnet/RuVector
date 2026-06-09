# ruvector 2026: Agent Memory Compaction via Coherence-Gated Graph Clustering in Rust

> Merge semantically redundant AI agent memories using k-NN cosine graphs and coherence-gated clustering — 60% storage reduction at >0.99 recall@10 in pure Rust.

**One sentence**: `ruvector-memory-compact` is the first Rust crate that treats vector database compaction as a *semantic* problem — not just a storage problem — using coherence-gated graph clustering with auditable witness chains.

- Repository: https://github.com/ruvnet/ruvector
- Research branch: `research/nightly/2026-06-09-ruvector-memory-compact`
- Research doc: `docs/research/nightly/2026-06-09-ruvector-memory-compact/README.md`
- ADR: `docs/adr/ADR-199-agent-memory-compaction.md`

---

## Introduction

Autonomous AI agents accumulate memories continuously. A coding agent working
across a week-long project might store thousands of code snippet embeddings,
error messages, documentation fragments, and conversation turns. A customer
support agent might accumulate millions of interaction embeddings across months
of operation. Without compaction, memory grows without bound — and eventually
retrieval latency degrades as the index fills with near-duplicate entries
representing the same concept from slightly different angles.

The naive solution — just delete old memories based on age (TTL expiry) — destroys
useful information. The right solution is *semantic compaction*: identify groups
of near-duplicate memories, replace each group with a single representative
centroid, and record exactly which original memories went into each centroid so
the merge is auditable and reversible.

Current production vector databases (Qdrant, Milvus, LanceDB, Chroma) treat
compaction as a *structural* concern — merging small index segments into large
ones for I/O efficiency. None of them understand that 50 different phrasings of
"the user prefers dark mode" should be stored as one embedding, not 50. This is
the gap that `ruvector-memory-compact` fills.

RuVector is uniquely positioned to solve this because it was built from day one
as a *cognition substrate*, not just a vector store. It already ships coherence
scoring (`ruvector-coherence`), graph clustering (`ruvector-mincut`), and
provenance tracking (`ruvector-verified`). This nightly adds the orchestration
layer that wires those primitives together into a compaction pipeline.

The result is a self-contained Rust crate with no external service dependencies,
deployable to edge devices and WASM targets, producing auditable `WitnessRecord`
chains that let AI safety auditors trace every merge decision. Three strategies
are provided — K-means baseline, threshold graph merge, and coherence-gated
adaptive merge — each measuring real recall@10 against the pre-compaction ground
truth.

---

## Features

| Feature | What it does | Why it matters | Status |
|---|---|---|---|
| K-means compaction | Lloyd's algorithm on cosine similarity | Fastest variant; works on any clustered data | Implemented in PoC |
| Graph-merge compaction | k-NN cosine graph + threshold-based connected components | Discovers natural topic granularity; does not force fixed K | Implemented in PoC |
| Coherence-gated compaction | k-NN graph + per-node coherence score gates each merge | Prevents over-merging of heterogeneous memories | Implemented in PoC |
| WitnessRecord chain | Records which original IDs → centroid for every merge | Enables audit, rollback, and safety provenance | Implemented in PoC |
| Recall@10 measurement | Cluster-aware recall against pre-compaction ground truth | Verifies no catastrophic information loss | Measured |
| `Compactor` trait | Swappable strategy interface | Enables downstream code to be strategy-agnostic | Implemented in PoC |
| Edge / WASM safe | No external deps; compiles to wasm32 | Runs on Cognitum Seed, Pi Zero 2W, browser WASM | Implemented in PoC |
| MCP memory tool | `memory_compact(namespace, ratio)` agent tool | Enables ruFlo agents to self-manage memory | Research direction |
| Approximate k-NN graph | HNSW-backed graph for N > 10K | O(N log N) instead of O(N²) | Production candidate |
| Proof-gated witness | ZK attestation that each merge was coherence-justified | AI safety in regulated industries | Research direction |

---

## Technical Design

### Core data structure

```rust
pub struct MemoryStore {
    pub entries: Vec<MemoryEntry>,
    pub(crate) next_id: u64,
}

pub struct MemoryEntry {
    pub id: u64,
    pub embedding: Vec<f32>,
    pub age: u64,
    pub metadata: String,
}

pub struct WitnessRecord {
    pub centroid_id: u64,
    pub merged_ids: Vec<u64>,    // original IDs absorbed
    pub intra_sim: f32,          // avg cosine similarity within cluster
}
```

### Trait-based API

```rust
pub trait Compactor {
    fn compact(
        &self,
        store: &mut MemoryStore,
        target_ratio: f64,        // fraction of vectors to KEEP
        queries: &[Vec<f32>],     // for recall measurement
        k: usize,
    ) -> CompactionResult;
    fn name(&self) -> &'static str;
}
```

### Baseline: NaiveCompactor

Lloyd's K-means on cosine similarity. K = ⌈N × target_ratio⌉. 30 iterations.
O(N × K × D × 30) per compaction. Fastest at small N.

### Variant A: GraphMergeCompactor

1. Build k-NN cosine similarity graph (k=15 per node).
2. Binary-search for threshold T: connected_components(T) ≈ target_k.
3. Each component → centroid → `WitnessRecord`.

Advantage: discovers natural cluster granularity. With tight topic clusters, may
compact far beyond the target ratio (e.g., 98% instead of 60%) when the data
is extremely well-clustered.

### Variant B: CoherenceGatedCompactor

1. Build k-NN graph.
2. Pre-compute per-node coherence score: `mean(edge_weights) − std_dev(edge_weights)`.
3. Sort edges by weight descending. For each edge (a, b):
   - Compute `coherence = avg(node_coherence[a], node_coherence[b])`.
   - Merge only if `coherence ≥ floor` AND `weight ≥ floor × 0.8` AND `merged_size ≤ max`.
4. Stop when target_k clusters are formed.

The coherence floor prevents merging of heterogeneous memories that happen to
share a noisy edge.

### Memory model

- Raw: N × D × 4 bytes (float32 embeddings)
- Graph: N × k × (4 + 8) bytes (edge weights + neighbour indices) ≈ N × 15 × 12 = 180N bytes
- Compacted: (N × target_ratio) × D × 4 bytes
- Witness chain: one record per centroid ≈ N × (1 − target_ratio) × 16 bytes (amortised)

At N=1000, D=128: raw=0.488MB, graph=0.180MB, compacted=0.195MB.

### Performance model

Graph build: O(N² × D) exact. Dominant cost.
K-means: O(N × K × D × iterations) per iteration.
Graph-merge: O(N² × D) + O(E log E) sort + O(E × α(N)) union-find.
Coherence-gated: same as graph-merge.

### Architecture diagram

```
MemoryStore ──build──► CoherenceGraph ──cluster──► [Cluster₁, ..., ClusterK]
                                                        │
                                               centroid(Cluster_i) → MemoryEntry
                                               WitnessRecord{centroid_id, merged_ids, intra_sim}
                                                        │
                                                CompactionResult{ratio, recall, witnesses}
```

---

## Benchmark Results

**Environment**: OS=linux, Arch=x86_64, Rust=1.94.1 (release build)

```bash
cargo run --release -p ruvector-memory-compact
```

**Dataset**: 20 topic centroids × 50 vectors each = N=1000, dim=128, noise=0.15

### Primary results

| Variant | N→M | Compact% | Recall@10 | Time(ms) | Mem after (MB) | Pass |
|---|---|---|---|---|---|---|
| naive-kmeans | 1000→400 | 60.0% | 0.915 | 72 | 0.195 | ✓ |
| graph-merge | 1000→20 | 98.0% | 1.000 | 119 | 0.010 | ✓ |
| coherence-gated | 1000→400 | 60.0% | 0.990 | 114 | 0.195 | ✓ |

### Latency sweep (5 runs each)

| Variant | Mean (ms) | p50 (ms) | p95 (ms) | Throughput (vecs/s) |
|---|---|---|---|---|
| naive-kmeans | 70.6 | 71 | 71 | 14,164 |
| graph-merge | 120.6 | 121 | 124 | 8,292 |
| coherence-gated | 117.8 | 118 | 120 | 8,489 |

### Memory math

- Raw store: 1000 × 128 × 4 B = **0.488 MB**
- After 60% compaction: 400 × 128 × 4 B = **0.195 MB** (2.5x reduction)
- Graph-merge extreme case: 20 × 128 × 4 B = **0.010 MB** (49x reduction)

### Benchmark limitations

- N=1000 is small; the O(N²) graph construction bottleneck only matters at N > 10K.
- Synthetic clustered data is easier to compact than real agent memory.
- Recall numbers are not directly comparable to any external system benchmark.
- Acceptance threshold (recall@10 ≥ 0.55) is conservative; production would target ≥ 0.80.

---

## Comparison with Vector Databases

| System | Core strength | Where it is strong | Where RuVector differs | Benchmarked here |
|---|---|---|---|---|
| Milvus | Production scale, GPU support | Billion-scale ANN, ANNS-HT benchmarks | Semantic compaction, graph coherence, agent memory | No |
| Qdrant | Rust performance, payload filtering | Filtered ANN, on-disk indexing | Coherence-gated compaction, witness chain, MCP native | No |
| Weaviate | Knowledge graph integration | Multi-modal, hybrid search, GraphQL | Pure Rust, no JVM, edge/WASM deployment | No |
| Pinecone | Managed cloud, serverless | Ease of use, hybrid search SaaS | Local-first, no cloud dependency, agent memory | No |
| LanceDB | Columnar storage, SQL integration | Batch analytics on embeddings | Online compaction, coherence gating | No |
| FAISS | Raw ANN performance | Maximum recall/speed on GPU | Rust-native, no BLAS dependency, graph coherence | No |
| pgvector | PostgreSQL integration | SQL vector queries | Standalone, no PostgreSQL dependency | No |
| Chroma | Ease of use, Python ecosystem | Prototyping, small collections | Production Rust, no Python, edge deployment | No |
| Vespa | Hybrid search, ranking | Structured + vector + BM25 | Agent memory compaction, witness chain | No |

> **Note**: No external competitor benchmarks are claimed or reproduced here.
> All numbers in this document are from the RuVector PoC only.

---

## Practical Applications

| Application | User | Why it matters | RuVector role | Near-term path |
|---|---|---|---|---|
| Agent episodic memory | Long-horizon AI agents (Claude, GPT) | Prevents unbounded memory growth | MemoryStore + CoherenceGatedCompactor | Phase 2 MCP tool |
| RAG index compaction | Enterprise search systems | Removes stale near-duplicate documents | GraphMergeCompactor on doc embeddings | Phase 2 server API |
| MCP memory tools | ruFlo workflows, Claude agents | Bounded memory for multi-session agents | ruvector-server MCP endpoint | Phase 2 |
| Conversation summarisation | Chatbot backends | Compress old turns into topic centroids | NaiveCompactor on turn embeddings | Phase 2 |
| Code intelligence | IDE assistants | Merge near-duplicate code snippets | CoherenceGatedCompactor on code embeds | Phase 3 |
| Log anomaly detection | SRE tooling | Compact normal logs; preserve anomalies | High coherence_floor preserves rare events | Research |
| Scientific literature | Research assistants | Merge near-duplicate abstract clusters | GraphMergeCompactor on paper embeddings | Research |
| Workflow automation | ruFlo orchestrator | Compact step history for context window | MemoryStore compaction hook | Phase 2 |

---

## Exotic Applications

| Application | 10–20 year thesis | Required advances | RuVector role | Risk/unknown |
|---|---|---|---|---|
| Lifelong cognitive substrate | Agents with years of operation need hierarchical memory analogous to human sleep-mediated consolidation | Multi-level recursive compaction | Nested MemoryStore + Compactor hierarchy | Concept drift invalidates old centroids |
| Proof-gated memory surgery | Regulated AI systems need ZK-proof that each merge was coherence-justified | ruvector-verified + ZK witness chain integration | Compaction with cryptographic attestation | ZK overhead at compaction time |
| Swarm collective memory | 1000-agent swarms share one compacted memory namespace | Distributed compaction with Raft consensus | ruvector-raft + distributed MemoryStore | Byzantine merge decisions |
| RVM coherence domains | RVM uses coherence domains as first-class memory GC regions | CoherenceGatedCompactor as domain GC | rvm crate integration | Coherence boundary semantics TBD |
| Self-healing vector graphs | HNSW auto-deduplicates near-identical nodes on insert | Compaction integrated into HNSW insert path | ruvector-core HNSW integration | Breaks layer invariants without care |
| Synthetic long-term memory | Neural-inspired episodic → semantic consolidation | Multi-level + LLM summarisation | MemoryStore + ruvLLM summarisation | Summarisation quality limits recall |
| Agent operating system | OS kernel manages agent memory across processes | Kernel-level MemoryStore + priority queues | ruvix + ruvector-memory-compact | OS-level permission model needed |
| Bio-signal memory bank | EEG/ECG streams compacted by temporal coherence clustering | Real-time compaction at N > 1M | SIMD-accelerated graph build | Temporal ≠ semantic coherence |

---

## Deep Research Notes

### What the SOTA suggests

The 2024–2026 agent memory literature (MemGPT[^1], A-MEM[^6], Zep, Mem0) focuses
on retrieval augmentation and paging. The closest analogue — Microsoft GraphRAG[^2] —
uses community detection on knowledge graphs for summarisation, but requires an
LLM call per merge. Our approach is fully deterministic, sub-second, and LLM-free.

### What remains unsolved

1. Optimal `target_ratio` selection (requires domain-specific calibration).
2. Temporal coherence: geometrically similar memories from different time periods.
3. Multi-modal embeddings: intra- and cross-modal similarity require separate treatment.
4. Online compaction: the current implementation is batch; streaming is needed for
   real-time agents.

### Where this PoC fits

Working demonstration of geometric semantic compaction at N=1000, sub-120ms,
>91% recall retention. Not yet production-grade for N > 10K or adversarial inputs.

### What would make this production-grade

1. HNSW-backed approximate k-NN graph (O(N log N) build).
2. `ruvector-snapshot` integration for pre-compaction checkpointing.
3. Streaming witness chain to `redb`-backed store.
4. Empirical calibration on real agent memory datasets (Claude session logs, etc.).

### What would falsify the approach

If real agent memories are not clustered — each memory is semantically unique —
coherence-gated compaction achieves near-zero compaction ratio and is useless.
If recall cannot be maintained above 0.80 at ≥50% compaction on real data,
a summary-based method (LLM-generated summaries) would be required instead.

---

## Usage Guide

```bash
git checkout research/nightly/2026-06-09-ruvector-memory-compact
cargo build --release -p ruvector-memory-compact
cargo test -p ruvector-memory-compact
cargo run --release -p ruvector-memory-compact

# Larger dataset
N_TOPICS=50 VECS_PER_TOPIC=100 cargo run --release -p ruvector-memory-compact

# Higher dimensions
DIM=256 cargo run --release -p ruvector-memory-compact
```

Expected output ends with:
```
Acceptance threshold : recall@10 ≥ 0.55  →  ALL PASS ✓
```

### Interpreting results

- `Compact%` = fraction of vectors removed. Higher = more aggressive compaction.
- `Recall@10` = cluster-aware recall against pre-compaction ground truth.
- `graph-merge` may compact more aggressively than requested (it finds the natural
  cluster granularity of the data, which may be fewer clusters than target_k).
- `coherence-gated` respects the `max_cluster` limit; adjust `coherence_floor`
  to tune aggressiveness.

### Adding a new backend

```rust
pub struct MyCompactor;
impl Compactor for MyCompactor {
    fn name(&self) -> &'static str { "my-compactor" }
    fn compact(&self, store: &mut MemoryStore, target_ratio: f64,
               queries: &[Vec<f32>], k: usize) -> CompactionResult {
        // your algorithm here
    }
}
```

---

## Optimization Guide

| Dimension | Optimization | Gain |
|---|---|---|
| Memory | Reduce `graph_k` (5 instead of 15) | 3x less graph memory |
| Latency | Use `NaiveCompactor` for N < 500 | 2x faster than graph variants |
| Recall | Increase `graph_k` (20+) | Better cluster boundaries |
| Edge deployment | `default-features = false` (no rayon) | Single-threaded, WASM-safe |
| WASM | Reduce N to ≤ 200 | Sub-50ms on Cortex-A53 |
| MCP throughput | Batch compaction (compact once/hour, not per insert) | Amortises O(N²) cost |
| ruFlo automation | Trigger on `store.len() > threshold` hook | Prevents unbounded growth |

---

## Roadmap

### Now
- Merge `ruvector-memory-compact` crate into workspace
- Expose via `ruvector-server` REST endpoint: `POST /v1/memory/{ns}/compact`
- Add MCP tool: `memory_compact(namespace, target_ratio, strategy, dry_run)`

### Next
- Approximate k-NN graph (HNSW-backed) for N > 10K
- `ruvector-snapshot` integration (pre-compaction checkpoint)
- Streaming `WitnessRecord` persistence to `redb`
- ruFlo hook: auto-compact on memory threshold event
- Age-weighted edges (discount old memories to prevent temporal conflation)

### Later (10–20 years)
- Hierarchical multi-level compaction (episodic → semantic → conceptual)
- ZK-proof witness chains (proof-gated memory surgery for regulated AI)
- Swarm collective memory compaction with Raft consensus
- Integration with ruvix agent OS kernel for process-level memory management
- Synthetic long-term memory with sleep-analogous consolidation cycles

---

## Footnotes and References

[^1]: Packer, C. et al. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560 (2023). https://arxiv.org/abs/2310.08560 — accessed 2026-06-09.

[^2]: Edge, D. et al. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." Microsoft Research (2024). https://arxiv.org/abs/2404.16130 — accessed 2026-06-09.

[^3]: Malkov, Y. & Yashunin, D. "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs." IEEE TPAMI (2018). https://arxiv.org/abs/1603.09320 — accessed 2026-06-09.

[^4]: Qdrant team. "Snapshots and Recovery." Qdrant documentation. https://qdrant.tech/documentation/concepts/snapshots/ — accessed 2026-06-09.

[^5]: Milvus team. "Compaction." Milvus documentation. https://milvus.io/docs/compaction.md — accessed 2026-06-09.

[^6]: Yang, Z. et al. "A-MEM: Agentic Memory for LLM Agents." arXiv:2502.12110 (2025). https://arxiv.org/abs/2502.12110 — accessed 2026-06-09.

[^7]: Shi, J. & Malik, J. "Normalized Cuts and Image Segmentation." IEEE TPAMI 22(8) (2000). https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf — accessed 2026-06-09. The normalised-cut intuition informs why coherence gating (preserving intra-cluster tightness) is preferable to raw threshold cuts.

---

## SEO Tags

**Keywords**: ruvector, Rust vector database, Rust vector search, agent memory,
memory compaction, coherence-gated clustering, k-NN graph, cosine similarity,
graph RAG, ANN search, HNSW, semantic deduplication, witness chain, ruvnet,
ruFlo, MCP memory tools, edge AI, WASM AI, high performance Rust, autonomous
agents, retrieval augmented generation, AI agent memory management.

**Suggested GitHub topics**: rust, vector-database, agent-memory, memory-compaction,
coherence, graph-clustering, ann, cosine-similarity, witness-chain, rag, graph-rag,
mcp, wasm, edge-ai, rust-ai, semantic-search, autonomous-agents, ruvector.
