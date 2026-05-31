# ruvector 2026: Agent Memory Compaction via Graph-Cut Clustering for High-Performance Rust Vector Search

> **150-char summary:** Rust agent memory compaction using cosine-similarity graph clustering achieves 100% cluster recall at 5% budget — 95 pp better than age eviction.

**One sentence:** `ruvector-memcompact` is the first Rust implementation of semantic-aware agent memory compaction using graph-cut clustering, achieving perfect cluster recall while reducing a 500-entry memory store to 20 representatives.

- Repository: https://github.com/ruvnet/ruvector
- Research branch: `research/nightly/2026-05-26-agent-memory-graph-compaction`

---

## Introduction

Every AI agent that runs for more than a few hours faces the same problem: memory accumulates faster than it can be used. A coding assistant remembers 500 conversations but can only hold 25 in working context. A customer support agent builds up thousands of interaction embeddings over weeks. A scientific research agent indexes hundreds of papers and then re-indexes variations of the same papers.

The standard approach to this problem is age-based eviction — delete the oldest memories when the store exceeds a budget. This is fast and simple. It is also semantically blind. A memory from three weeks ago that introduced a critical architectural constraint is just as important as a memory from yesterday, yet age eviction treats them identically based on when they arrived rather than what they contain.

Current vector database systems do not solve this. LanceDB's fragment merge optimises disk I/O but has no semantic awareness. Qdrant, Milvus, and FAISS support index rebuilding but not compaction. MemGPT and Letta, the dominant agent memory frameworks, leave compaction entirely to the agent itself — a manual operation that rarely happens in production deployments. No production system applies graph structure to the compaction decision.

RuVector is a Rust-native cognition substrate for AI agents. It has HNSW graph indexing, dynamic min-cut algorithms, graph neural retrieval, and distributed graph storage. What it has lacked is a semantic-aware compaction primitive that sits between the write path and the index — reducing memory stores without destroying the semantic coverage that makes retrieval useful. This post introduces `ruvector-memcompact`, which fills that gap.

The core insight is straightforward: memories that say roughly the same thing should be merged rather than individually evicted. Agent memories cluster naturally by topic. A cosine-similarity graph over stored embedding vectors, partitioned by Union-Find at a threshold, reveals those clusters. Replacing each cluster with its arithmetic centroid preserves what the cluster "knows" while shrinking the index proportionally to cluster density. At 5% budget (25 entries from 500), this achieves 100% cluster recall — versus 5% for age eviction and 75% for importance-based eviction.

---

## Features

| Feature | What it does | Why it matters | Status |
|---------|-------------|----------------|--------|
| `CompactionStrategy` trait | Shared interface for all strategies | Composable — strategies can be chained or swapped | Implemented in PoC |
| `AgeEviction` | Keep newest N entries by timestamp | Baseline; reflects production default in most agent systems | Implemented in PoC |
| `ImportanceEviction` | Keep highest-importance N entries | Better than age when importance labels are accurate | Implemented in PoC |
| `GraphCutCompaction` | Cluster by cosine similarity, merge to centroids | 100% cluster recall at 5% budget vs 5% for age eviction | Implemented in PoC |
| `UnionFind` clustering | O(n·α(n)) connected components at similarity threshold θ | Near-linear cluster discovery over pairwise similarity graph | Implemented in PoC |
| Pairwise cosine similarity | O(n²·d) similarity graph construction | Foundation for all graph-based strategies | Measured |
| Cluster-level recall@k | Semantic quality metric over cluster IDs | More meaningful than exact-match recall for compacted stores | Measured |
| Deterministic dataset generation | Reproducible synthetic benchmark with Gaussian mixture model | Real benchmarks with no external dependency | Measured |
| `#![forbid(unsafe_code)]` | No unsafe Rust | Memory-safe compaction primitive | Implemented in PoC |
| WASM-compatible dependencies | `rand`, `rand_distr`, `serde` only | Edge deployment with no libc dependency | Production candidate |

---

## Technical Design

### Core data structure

```rust
pub struct MemoryEntry {
    pub id:         MemId,
    pub vector:     Vec<f32>,
    pub timestamp:  u64,
    pub importance: f32,
    pub cluster_id: Option<u64>, // evaluation only
}

pub struct MemoryStore {
    pub entries: Vec<MemoryEntry>,
}
```

### Trait-based API

```rust
pub trait CompactionStrategy {
    fn name(&self) -> &'static str;
    fn compact(&self, store: &MemoryStore, budget: usize) -> MemoryStore;
}
```

All three strategies implement this trait. Callers can swap strategies at runtime, chain them, or benchmark them against each other using the same interface.

### Baseline: AgeEviction

Sort by timestamp descending, truncate to budget. O(n log n). Zero semantic awareness — represents the current production default.

### Variant A: ImportanceEviction

Sort by importance score descending, truncate. O(n log n). Better than age when importance labels accurately reflect semantic value; degrades to random sampling when labels are uniform or noisy.

### Variant B: GraphCutCompaction

```
1. Build pairwise cosine similarity graph: O(n² · d)
2. Union-Find: merge pairs with sim ≥ θ into components: O(n · α(n))
3. For each component: compute centroid vector, inherit max(importance), max(timestamp)
4. If |representatives| > budget: trim by descending importance
5. Return compacted MemoryStore
```

Per-dimension noise is generated with std = σ/√D so ‖ε‖ ≈ σ, ensuring within-cluster cosine similarity ≈ 1/(1+σ²) >> threshold. This cleanly separates within-cluster pairs (sim ≈ 0.978) from between-cluster pairs (sim ≈ 0 for random unit vectors in D=64).

### Memory model

| Component | Memory | Formula |
|-----------|--------|---------|
| Original (N=500, D=64) | 156.2 KB | N × (D×4 + struct overhead) |
| AgeEviction (budget=25) | 7.8 KB | budget × same |
| GraphCutCompaction (K=20 clusters) | 6.2 KB | K × same (K ≤ budget) |

### Performance model

| Step | Complexity | At N=500, D=64 |
|------|------------|----------------|
| Pairwise similarity | O(n²·d) | ~16 M ops, 9.5 ms |
| Union-Find | O(n·α(n)) | ~500 ops |
| Centroid synthesis | O(n·d/K) | ~1.6 K ops |
| Importance trim | O(K log K) | ~80 ops |

Practical limit: ~10 K entries before pairwise becomes too slow for interactive use. Beyond that, use an approximate k-NN graph (as in HNSW construction) to reduce to O(n·k·d).

```mermaid
graph LR
    A[MemoryStore\n500 entries] --> B[Pairwise cosine\nO n²d]
    B --> C[Union-Find\nclusters at θ=0.70]
    C --> D[Centroid synthesis\nper cluster]
    D --> E[Compacted Store\n20 representatives]
    E --> F[Recall@10\n= 100%]
```

---

## Benchmark Results

All numbers from a single `cargo run --release -p ruvector-memcompact` run.
No aspirational numbers. No competitor numbers (direct comparison requires
running the same workload on competitor systems, which was not done).

**Hardware:**
- CPU: Intel(R) Xeon(R) Processor @ 2.80 GHz
- OS: Linux x86_64, kernel 6.18.5
- Rust: 1.87.0-nightly
- Command: `cargo run --release -p ruvector-memcompact`

**Dataset:** N=500 entries, D=64 dimensions, K=20 semantic clusters (25 entries/cluster),
noise σ=0.15 (‖ε‖ scaled to σ/√D per dimension), budget=25.

| Variant | N | D | K | Budget | Compact ms | Query μs (mean) | Query μs (p50) | Query μs (p95) | Memory KB | Reduction | Recall@10 | Accept |
|---------|---|---|---|--------|------------|-----------------|----------------|----------------|-----------|-----------|-----------|--------|
| AgeEviction | 500 | 64 | 20 | 25 | 0.07 | 2.27 | 2.27 | 2.34 | 7.8 | 95.0% | 5.0% | — |
| ImportanceEviction | 500 | 64 | 20 | 25 | 0.04 | 2.26 | 2.25 | 2.31 | 7.8 | 95.0% | 75.0% | — |
| **GraphCutCompaction** | **500** | **64** | **20** | **25** | **9.50** | **1.75** | **1.74** | **1.81** | **6.2** | **96.0%** | **100.0%** | **PASS** |

**Notes on benchmark limitations:**
- Compaction time is a single run, not averaged. For O(n²) the variance is low.
- Query latency is brute-force over the compacted store (no ANN index).
- Synthetic data with well-separated Gaussian clusters; real agent memories may have fuzzier cluster boundaries.
- No competitor systems were directly benchmarked. Claims about MemGPT, LanceDB etc. are based on published documentation and literature, not direct measurement.

---

## Comparison with Vector Databases

| System | Core strength | Where it is strong | Where RuVector differs | Direct benchmarked here |
|--------|--------------|--------------------|-----------------------|------------------------|
| Milvus | IVF-PQ at billion scale | Large-scale production RAG | No agent memory compaction; index rebuild is structural only | No |
| Qdrant | HNSW + payload filtering | Hybrid search, metadata filtering | No semantic-aware compaction; deletions require index rebuild | No |
| Weaviate | HNSW + GraphQL | Knowledge graph integration | No compaction strategy abstraction | No |
| Pinecone | Proprietary IVF-like | Managed cloud, zero ops | Opaque; no compaction control | No |
| LanceDB | Lance fragment merge | Append-only, good for analytics | Fragment merge is I/O structural, not semantic | No |
| FAISS | IVF-PQ, GPU | Academic benchmarks, GPU workloads | No agent lifecycle integration; C++ library | No |
| pgvector | SQL-native vectors | PostgreSQL extensions, SQL joins | No compaction primitive; relies on VACUUM | No |
| Chroma | Developer-friendly | Rapid prototyping, local RAG | No production compaction; Python-first | No |
| Vespa | Tensor fields, BM25+vector | Enterprise hybrid search | No agent memory lifecycle management | No |

RuVector differentiates on: Rust memory safety, semantic-aware compaction, graph-cut primitives, ruFlo automation, MCP tool surface, WASM edge deployment, and composable `CompactionStrategy` trait.

---

## Practical Applications

| Application | User | Why it matters | How RuVector uses it | Near-term path |
|-------------|------|----------------|---------------------|----------------|
| Agent session compaction | AI assistant developers | Prevents context overflow in long sessions | GraphCutCompaction on `ruvector-core` store | Wire `MemoryStore` to `ruvector-core` VecStore |
| Enterprise knowledge base dedup | Enterprise AI teams | Removes near-duplicate embeddings from RAG corpus | GraphCutCompaction on document embedding index | Import documents via `ruvector-server` API |
| MCP memory tool | Agent framework developers | Expose compaction as an MCP-callable primitive | `memory_compact` MCP tool wrapping `ruvector-memcompact` | Implement `src/mcp.rs` |
| ruFlo nightly compaction | ruFlo workflow users | Automatic nightly memory consolidation | ruFlo cron step calling `memory_compact` | ruFlo step type integration |
| Code intelligence compaction | IDE agent users | Keeps relevant file context, evicts stale analysis | GraphCutCompaction on code embedding index | IDE plugin integration |
| Multi-agent shared memory | Swarm system developers | Compact shared memory before distributing to agents | RVF-packaged compacted store | RVF manifest with compacted embeddings |
| Edge AI assistant memory | Cognitum Seed users | On-device compaction without cloud round-trip | WASM-compiled `ruvector-memcompact-wasm` | WASM packaging |
| Scientific literature dedup | Researchers | Merge near-duplicate paper embeddings | High-K clustering on arXiv/PubMed embeddings | Import pipeline via `ruvector-server` |

---

## Exotic Applications

| Application | 10–20 year thesis | Required technical advances | RuVector role | Risk or unknown |
|-------------|-------------------|----------------------------|---------------|-----------------|
| Cognitum autonomous memory consolidation | Agents run for years, self-compacting without human oversight | Stable threshold auto-calibration; on-device compaction daemon | Embedded compaction in Cognitum Seed firmware | Centroid drift after many compaction cycles |
| RVM coherence domain compaction | Coherence boundaries from graph-cut structure of shared agent memory | RVM coherence scoring + compaction integration | MinCut-aware compaction across domain boundaries | Coherence score reliability at scale |
| Proof-gated memory archives | Compacted memories are cryptographically attested; auditable by regulators | ZK-proof over centroid computation | `ruvector-verified` + `ruvector-memcompact` pipeline | Proof overhead per compaction cycle |
| Swarm distributed memory compaction | 100+ agents share compacted memory graphs; writes are consensus-gated | Byzantine-tolerant consensus on cluster boundaries | Raft-based compaction coordinator in `ruvector-raft` | Network partition during compaction |
| Self-healing vector graphs | After hardware failure, reconstruct memory clusters from surviving centroid metadata | Erasure coding over centroid vectors; RVF recovery format | RVF manifest + recovery procedure | Information loss when entire cluster is lost |
| Biological signal memory | Implantable devices compact neural spike trains using learned cosine similarity | Low-power WASM runtime; learned similarity for bio signals | WASM-SIMD compaction kernel | Embedding quality for high-dimensional bio signals |
| Autonomous robotics long-term memory | Robots compact spatial and procedural memories over years of operation | Compaction of pose-conditioned embeddings | RVF-packaged robot memory graphs | Domain shift between environments |
| Agent operating system memory manager | AOS manages agent memory as a resource with compaction as garbage collection | AOS kernel integration; MMU-like memory isolation | `ruvector-memcompact` as AOS GC primitive | Security isolation between agent memory regions |

---

## Deep Research Notes

### What the SOTA suggests

The research literature on agent memory (2024–2026) identifies three major
patterns[^1][^2][^3][^4]:

1. **Read-optimised, not compact-optimised.** Systems like Mnemis[^4] build
   sophisticated retrieval graphs but have no compaction pass. The assumption
   is that storage is cheap; the hard problem is retrieval quality, not storage
   reduction.

2. **LLM-directed, not algorithmic.** A-MEM[^2] and similar systems delegate
   compaction decisions to the LLM itself. This produces human-readable
   compaction reasoning but is non-deterministic, slow, and expensive.

3. **KV-cache clustering is close but at the wrong layer.** SemantiCache[^6]
   uses seed-based clustering to merge semantically coherent KV-cache tokens.
   The algorithm is closely analogous to graph-cut compaction but applied to
   transient inference state rather than persistent agent memory.

### What remains unsolved

1. Threshold auto-calibration for θ without domain knowledge.
2. Incremental online compaction (O(k) per insert vs O(n²) batch).
3. Quality metric beyond cluster-level recall (does compaction hurt downstream
   task performance?).
4. Production integration with HNSW internals to eliminate the O(n²) pass.

### Where this PoC fits

`ruvector-memcompact` is a validated proof of concept with real benchmark numbers.
It demonstrates the algorithm works on synthetic data with realistic parameters.
It does not claim production readiness; the path to production is spelled out in
the ADR implementation plan.

### What would falsify the approach

If a production embedding model produces within-cluster and between-cluster
cosine similarities that are indistinguishable (within 0.05 of each other),
then Union-Find clustering degrades to random component assignment.  This can
be detected cheaply: compute the ratio of mean within-cluster similarity to
mean between-cluster similarity before committing to compaction.  If the ratio
is < 1.2, the embedding space is not suitable for cosine-based compaction.

---

## Usage Guide

```bash
git checkout research/nightly/2026-05-26-agent-memory-graph-compaction

# Build
cargo build --release -p ruvector-memcompact

# Run tests (12 unit tests)
cargo test -p ruvector-memcompact

# Run benchmark (default: N=500, D=64, K=20, budget=25)
cargo run --release -p ruvector-memcompact

# Custom dataset size
BENCH_N=1000 BENCH_CLUSTERS=40 BENCH_BUDGET=50 \
  cargo run --release -p ruvector-memcompact

# Larger stress test
BENCH_N=2000 BENCH_CLUSTERS=80 BENCH_BUDGET=100 BENCH_DIM=128 \
  cargo run --release -p ruvector-memcompact
```

**Expected output (default parameters):**

```
╔══════════════════════════════════════════════════════════════╗
║      ruvector-memcompact  |  Agent Memory Compaction Bench   ║
╚══════════════════════════════════════════════════════════════╝

  Strategy             Entries  Compact  Query  Memory  Reduction  Recall
  ─────────────────────────────────────────────────────────────────────────
  AgeEviction              25    0.07ms  2.27μs   7.8KB    95.0%    5.0%
  ImportanceEviction        25    0.04ms  2.26μs   7.8KB    95.0%   75.0%
  GraphCutCompaction        20    9.50ms  1.75μs   6.2KB    96.0%  100.0%

  Overall: ACCEPT ✓
```

**Interpreting results:**
- Recall@10 is cluster-level: 100% means every semantic topic in the original
  store has a representative in the compacted store.
- Compact time scales O(n²·d) with dataset size. At N=2000 expect ~150 ms.
- Query time scales O(entries) with compacted store size.

**Adding a new compaction strategy:**

```rust
pub struct MyStrategy;

impl CompactionStrategy for MyStrategy {
    fn name(&self) -> &'static str { "MyStrategy" }
    fn compact(&self, store: &MemoryStore, budget: usize) -> MemoryStore {
        // your logic here
        MemoryStore { entries: vec![] }
    }
}
```

**Plugging into RuVector:**

```rust
// Once ruvector-core MemoryStore integration lands:
use ruvector_memcompact::{GraphCutCompaction, CompactionStrategy};
let compacted = GraphCutCompaction::with_threshold(0.70)
    .compact(&core_store.to_memory_store(), budget);
core_store.replace_with(compacted.to_vec_store());
```

---

## Optimization Guide

**Memory optimization:**
- Lower budget or increase threshold θ to produce fewer cluster representatives.
- For very large stores (n > 10 K), use a k-NN graph approximation to avoid
  materialising the full n² similarity matrix.

**Latency optimization:**
- GraphCutCompaction is O(n²·d). For latency-sensitive paths, run compaction
  off the query hot path (background thread or ruFlo cron step).
- Query latency improves automatically as the store shrinks. At K=20 vs N=500,
  queries run 23% faster (1.75 μs vs 2.27 μs).

**Recall optimization:**
- Lower threshold θ if recall is below target (allows more merges).
- Verify within-cluster similarity is >> θ using the similarity distribution
  check before running compaction.

**Edge deployment optimization:**
- Use BENCH_N ≤ 500 for on-device compaction on Pi Zero 2W class hardware.
- Future WASM build will enable browser-side compaction.

**MCP tool optimization:**
- Cache the compacted store; only recompact when store grows by > 20%.
- Use the `CompactionStrategy::name()` field to log which strategy was used
  for observability.

**ruFlo automation optimization:**
- Schedule compaction with a size trigger: `if store.len() > 2 * budget { compact() }`.
- Use `AgeEviction` as a fast pre-filter before `GraphCutCompaction` for
  very large stores.

---

## Roadmap

### Now
- Add `source_ids: Vec<MemId>` to centroid entries for provenance.
- Wire `MemoryStore` adapter to `ruvector-core`'s vector storage.
- Auto-calibrate threshold θ from empirical cosine distribution.
- Add `memory_compact` MCP tool handler.

### Next
- HNSW-graph-reuse compaction (O(E) instead of O(n²)).
- Incremental online compaction — O(k) per new memory insert.
- ruFlo workflow step integration.
- `ruvector-memcompact-wasm` WASM packaging.
- Proof-gated compaction audit trail via `ruvector-verified`.

### Later (10–20 years)
- Self-organising memory with adaptive cluster topology (clusters split and
  merge as the agent's knowledge evolves).
- Cognitum Seed on-device compaction daemon for year-long agent lifespans.
- RVM coherence domain compaction: mincut-based partitioning of shared agent
  memory graphs.
- Agent operating system memory manager using compaction as garbage collection.
- Biological signal memory compaction for embedded neuromorphic devices.

---

## Footnotes and References

[^1]: "MemGPT: Towards LLMs as Operating Systems", Packer et al., arXiv:2310.08560, 2023. https://arxiv.org/abs/2310.08560. Accessed 2026-05-26.

[^2]: "A-MEM: Agentic Memory for LLM Agents", arXiv:2502.12110, Feb 2025. https://arxiv.org/abs/2502.12110. Accessed 2026-05-26.

[^3]: "Mnemosyne: Unsupervised Long-Term Memory for Edge LLMs", arXiv:2510.08601, Oct 2025. https://arxiv.org/abs/2510.08601. Accessed 2026-05-26.

[^4]: "Mnemis: Dual-Route Retrieval on Hierarchical Graphs", arXiv:2602.15313, Apr 2026. https://arxiv.org/abs/2602.15313. Accessed 2026-05-26.

[^5]: "Adaptive Memory Admission Control for LLM Agents", arXiv:2603.04549, Mar 2026. https://arxiv.org/pdf/2603.04549. Accessed 2026-05-26.

[^6]: "SemantiCache: KV Cache Compression via Semantic Chunking and Clustered Merging", arXiv:2603.14303, Mar 2026. https://arxiv.org/pdf/2603.14303. Accessed 2026-05-26.

[^7]: "Down with the Hierarchy: The H in HNSW Stands for Hubs", Lyu et al., arXiv:2412.01940, Dec 2024. https://arxiv.org/pdf/2412.01940. Accessed 2026-05-26.

[^8]: "A Scalable Clustering Algorithm to Approximate Graph Cuts", arXiv:2308.09613, 2023. https://arxiv.org/pdf/2308.09613. Accessed 2026-05-26.

[^9]: LanceDB Data Concepts — Compaction. https://lancedb.com/documentation/concepts/data.html. Accessed 2026-05-26.

[^10]: "Beyond Nearest Neighbors: Semantic Compression and Graph-Augmented Retrieval", arXiv:2507.19715, Jul 2025. https://arxiv.org/abs/2507.19715. Accessed 2026-05-26.

---

## SEO Tags

**Keywords:**
ruvector, Rust vector database, Rust vector search, high performance Rust, agent memory compaction, graph-cut clustering, ANN search, HNSW, filtered vector search, graph RAG, agent memory, AI agents, MCP, WASM AI, edge AI, self learning vector database, ruvnet, ruFlo, Claude Flow, autonomous agents, retrieval augmented generation, memory deduplication, semantic memory, Union-Find clustering, cosine similarity, vector database compaction.

**Suggested GitHub topics:**
rust, vector-database, vector-search, ann, hnsw, agent-memory, memory-compaction, graph-cut, rag, graph-rag, ai-agents, mcp, wasm, edge-ai, rust-ai, semantic-search, graph-database, autonomous-agents, retrieval, embeddings, ruvector.
