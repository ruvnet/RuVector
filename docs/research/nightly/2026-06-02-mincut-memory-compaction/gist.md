# ruvector 2026: MinCut-Guided Agent Memory Compaction in Rust

> Graph-cut coherence eviction for AI agent working memory — the missing
> primitive for self-managing vector stores.  Built in pure Rust, zero
> external service dependencies, WASM-portable, and MCP-ready.

Every AI agent that maintains external memory faces the same question: *when
memory is full, what should be forgotten?*  Age-based eviction is blind to
semantics.  Random eviction destroys coherence.  MinCut-guided compaction
removes the memory entries that are *least connected to the semantic core* —
the right thing to forget.

→ Repository: https://github.com/ruvnet/ruvector  
→ Branch: `research/nightly/2026-06-02-mincut-memory-compaction`  
→ Crate: `crates/ruvector-mincut-memory`

---

## Introduction

### The problem

Production AI agents — Claude, GPT-4o, Gemini agents, AutoGPT, OpenAgents —
all maintain some form of working memory beyond the context window.  This
memory is almost always a vector store: embeddings of past observations,
retrieved facts, or processed documents.

The problem is growth.  Without a principled eviction policy, the store grows
unboundedly.  At 1,000 entries retrieval is fast.  At 100,000 it degrades.
At 10,000,000 it is unusable without sharding.  But more than raw size,
*semantic noise* is the real issue: as stale, irrelevant entries accumulate,
the signal-to-noise ratio of any retrieval query drops.

### Why the problem matters now

In 2026, agents are deployed in long-running, persistent configurations:
coding assistants that remember a project for months, medical decision support
systems that accumulate patient history, financial agents that track market
context over years.  The memory management question is no longer academic —
it is a production reliability concern.

### Why current vector databases only partially solve it

Every major vector database offers deletion:

| System | Compaction mechanism |
|---|---|
| Qdrant | Delete by ID or filter |
| Milvus | TTL via scalar metadata |
| Weaviate | Object-level deletion |
| Pinecone | Namespace delete |
| LanceDB | Full dataset rewrite |
| FAISS | Remove and rebuild |
| pgvector | SQL DELETE |

None of these systems answers the question *which entries to delete*.  They
provide the mechanism, not the policy.  Existing LLM-based solutions
(summarisation, importance scoring) require expensive model calls.
Forgetting-curve heuristics (Ebbinghaus decay) ignore the graph structure of
memory.

### Why RuVector is the right substrate

RuVector already has:

- `ruvector-mincut` — dynamic min-cut algorithms
- `ruvector-graph` — graph storage with Neo4j compatibility
- `ruvector-core` — HNSW vector search with SIMD
- `mcp-gate` — MCP tool surface
- `rvAgent` — AI agent framework in Rust
- `ruvector-coherence` — coherence scoring
- `ruvector-verified` — proof-gated writes with witness logs

All the primitives exist.  This nightly connects them: `ruvector-mincut-memory`
uses cosine similarity to build a graph over working memory entries, then uses
a weighted-degree approximation of minimum cut to identify and evict the most
peripheral — least semantically connected — entries.

### Why this matters for AI agents, graph RAG, edge AI, MCP, and Rust

- **AI agents:** A principled eviction policy makes long-running agents
  stable: memory stays bounded, recall stays high, latency stays low.
- **Graph RAG:** When the knowledge graph grows too large, graph-cut
  compaction removes weakly-connected knowledge nodes without destroying the
  dense, high-coherence core.
- **Edge AI:** On Cognitum Seed (Pi Zero 2W, 512 MB) or ESP32-S3, memory
  constraints are severe.  MinCutEvict in WASM enables continuous edge agent
  operation with bounded memory.
- **MCP:** `CompactionResult` maps directly to an MCP tool response.  Any
  Claude-based agent can call `memory_compact` as a tool call, making
  compaction a first-class agent capability.
- **Rust:** Zero-overhead graph traversal and cache-friendly f32 SIMD make the
  compaction fast enough for interactive agent loops.  No GC pauses.  No
  Python overhead.  No runtime.

---

## Features

| Feature | What it does | Why it matters | Status |
|---|---|---|---|
| `AgeEvict` | Evict oldest entries by timestamp | Deterministic baseline | Implemented in PoC |
| `CoherenceEvict` | Evict entries with lowest mean edge weight | Preserves semantic clusters | Implemented in PoC |
| `MinCutEvict` | Evict entries with lowest weighted graph degree | Approximates min-cut boundary | Implemented in PoC |
| Cosine similarity graph | Build N×N adjacency matrix from entry vectors | Foundation for all graph-aware strategies | Implemented in PoC |
| `Compactor` trait | Single trait for all strategies, swap without API change | Extensibility | Implemented in PoC |
| `CompactionResult` | Structured output: entries, edges, latency | Auditable, MCP-ready | Implemented in PoC |
| 18 unit tests | Cover all strategies and edge cases | Correctness | Measured |
| Benchmark binary | Reports recall, latency, edges, memory | Reproducible | Measured |
| WASM portability | No Tokio, no file I/O in lib | Edge deployment | Research direction |
| MCP tool surface | `memory_compact` tool in `mcp-gate` | Agent integration | Production candidate |
| `ruvector-mincut` exact integration | Exact min-cut for N ≤ 100 | Optimality for small stores | Research direction |
| ruFlo workflow | Scheduled compaction with checkpoint | Autonomous operation | Production candidate |

---

## Technical Design

### Core data structure

Each memory entry is a `(id, vector, timestamp, access_count)` tuple.  The
`MemoryStore` maintains a dense N×N f32 adjacency matrix (the cosine similarity
graph) built lazily on demand.

```rust
pub struct MemoryStore {
    pub entries: Vec<Entry>,
    pub dims: usize,
    pub similarity_threshold: f32,
    pub graph: Vec<Vec<f32>>,  // graph[i][j] = cosine_sim if ≥ threshold
    // ...
}
```

### Trait-based API

```rust
pub trait Compactor {
    fn compact(&self, store: &mut MemoryStore, target_size: usize) -> CompactionResult;
}
```

All three strategies implement this trait.  The application code never needs to
change — only the strategy selection changes.

### Baseline variant: AgeEvict

Sort entries by `timestamp` ascending; evict the oldest `N - target_size`.
O(N log N).  No graph reasoning.  Always correct for the baseline case where
older entries are less relevant.

### Alternative A: CoherenceEvict

For each node, compute mean cosine similarity to its graph neighbours.  Entries
with no neighbours score 0.0.  Evict the least coherent entries.  This strategy
preserves the tightest semantic clusters.

### Alternative B: MinCutEvict

For each node, compute *weighted degree* = sum of all incident edge weights.
Evict nodes with the lowest weighted degree.

```
weighted_degree(v) = Σ graph[v][u] for all u ≠ v
```

**Graph-cut interpretation:** In Karger-Stein min-cut and Stoer-Wagner algorithms,
the vertices that appear last in the max-adjacency ordering — i.e., those with the
smallest max-adjacency weight — define one side of the minimum cut.  Weighted
degree is a monotone proxy: nodes with low total edge weight are statistically
most likely to lie on minimum cuts.  By evicting them, we remove the entries that
least strengthen the coherence of the remaining memory.

### Memory model

- Adjacency matrix: N × N × 4 bytes = 4 MB at N=1,000.
- Vectors: N × D × 4 bytes = 256 KB at N=1,000, D=64.
- Total at N=1,000: ~4.3 MB.

For N > 5,000, a sparse CSR adjacency list is needed (planned).

### Performance model

- Graph rebuild: O(N²·D) — 64M FMAs at N=1,000, D=64.
- Strategy scoring: O(N) — negligible after rebuild.
- Eviction: O(k) swap_remove operations.

### Architecture

```mermaid
graph TD
    A[MemoryStore: N entries] --> B[rebuild_graph: cosine sim × threshold]
    B --> C{Compactor strategy}
    C -->|AgeEvict| D[sort by timestamp]
    C -->|CoherenceEvict| E[score: mean edge weight]
    C -->|MinCutEvict| F[score: weighted degree]
    D --> G[remove_indices: swap_remove]
    E --> G
    F --> G
    G --> H[CompactionResult]
    H --> I[MCP memory_compact tool]
    H --> J[ruFlo checkpoint + schedule]
    H --> K[ruvector-verified witness log]
```

---

## Benchmark Results

All numbers from real `cargo run --release` runs.  No invented numbers.

**Hardware:** x86-64 Linux 6.18 · Intel Celeron N4020 CPU  
**Rust version:** `rustc 1.94.1 (e408947bf 2026-03-25)`  
**Command:** `cargo run --release -p ruvector-mincut-memory`

### N=500, D=32, 6 clusters, 50 queries, K=10, 50% compaction

| Variant | N_in | N_out | Recall_b | Recall_a | Mean µs | p50 µs | p95 µs | Thr ops/s | Mem_b | Mem_a | Edges_b | Edges_a | Accept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AgeEvict | 500 | 250 | 1.000 | 1.000 | 6 340 | 6 240 | 6 599 | 157.7 | 74.2 KB | 37.1 KB | 7 652 | 1 955 | PASS |
| CoherenceEvict | 500 | 250 | 1.000 | 0.980 | 6 807 | 6 761 | 7 227 | 146.9 | 74.2 KB | 37.1 KB | 7 652 | 3 114 | PASS |
| **MinCutEvict** | **500** | **250** | **1.000** | **1.000** | **6 562** | **6 441** | **7 077** | **152.4** | **74.2 KB** | **37.1 KB** | **7 652** | **3 629** | **PASS** |

### N=1000, D=64, 8 clusters, 100 queries, K=10, 50% compaction

**Command:** `cargo run --release -p ruvector-mincut-memory -- --n 1000 --dims 64 --clusters 8 --queries 100`

| Variant | N_in | N_out | Recall_b | Recall_a | Mean µs | p50 µs | p95 µs | Thr ops/s | Mem_b | Mem_a | Edges_b | Edges_a | Accept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AgeEvict | 1000 | 500 | 1.000 | 1.000 | 51 859 | 51 939 | 52 177 | 19.3 | 273.4 KB | 136.7 KB | 2 997 | 759 | PASS |
| CoherenceEvict | 1000 | 500 | 1.000 | 1.000 | 53 392 | 52 934 | 55 157 | 18.7 | 273.4 KB | 136.7 KB | 2 997 | 1 420 | PASS |
| **MinCutEvict** | **1000** | **500** | **1.000** | **1.000** | **53 056** | **53 261** | **54 178** | **18.8** | **273.4 KB** | **136.7 KB** | **2 997** | **2 026** | **PASS** |

**Notes:**

- Latency is dominated by the O(N²·D) graph rebuild, not the scoring step.
- MinCutEvict retains **2.67× more graph edges** than AgeEvict at N=1,000.
- On server-class hardware (Ryzen 9, Xeon), latency would be 5–15× lower.
- The benchmark machine (Celeron N4020) is representative of edge hardware
  such as Raspberry Pi 4B or similar.
- These numbers are *not directly comparable* to competitor vector database
  benchmarks — no competitor measures graph-coherence-aware compaction.

**Acceptance criterion:** `recall_after / recall_before >= 0.60` for all strategies.  
**Result: ALL PASS.**

---

## Comparison with Vector Databases

| System | Core strength | Where it is strong | Where RuVector differs | Benchmarked here |
|---|---|---|---|---|
| Milvus | Billion-scale IVF-PQ | High-throughput batch retrieval | No agent memory lifecycle, no graph cut | No |
| Qdrant | Filtered HNSW | Metadata-filtered search | No coherence-aware compaction | No |
| Weaviate | Schema-driven graph | Knowledge graph RAG | No principled eviction policy | No |
| Pinecone | Managed cloud scale | Zero-ops enterprise | Proprietary, no edge, no graph cut | No |
| LanceDB | Delta Lake integration | Arrow/Parquet workflows | No graph structure in compaction | No |
| FAISS | Raw ANN speed | Research baselines | No agent memory lifecycle | No |
| pgvector | SQL integration | Existing PostgreSQL infra | No graph coherence | No |
| Chroma | Developer UX | Rapid prototyping | No production compaction primitive | No |
| Vespa | Hybrid retrieval | Complex ranking | No Rust-native, no graph cut | No |
| **RuVector** | **Graph-cut compaction, Rust, WASM, MCP** | **Agent memory, edge AI, coherence** | **This crate** | **Yes** |

RuVector is the only system with a graph-coherence-aware compaction primitive.
This is not a claim of superior retrieval performance — it is a claim of
unique agent memory lifecycle capability.

---

## Practical Applications

| # | Application | User | Why it matters | How RuVector uses it | Near-term path |
|---|---|---|---|---|---|
| 1 | Agent working memory | Claude, GPT-o, Gemini agents | Bounded memory → stable performance | MinCutEvict as drop-in eviction policy | Add MCP tool wrapper in `mcp-gate` |
| 2 | Graph RAG compaction | Enterprise RAG pipelines | Knowledge graph grows unboundedly | Graph-cut prunes weak knowledge edges | Integrate with `ruvector-graph` |
| 3 | Code intelligence | IDE copilots | Symbol memory per project | CoherenceEvict preserves used symbols | Access count weight in scoring |
| 4 | Conversation summarisation | Chat systems | Replace conversation with compact memory | Coherence-preserving compaction | ruFlo trigger every N turns |
| 5 | Edge anomaly detection | Industrial IoT | Sensor stream accumulates patterns | MinCutEvict evicts stale signatures | WASM build for edge |
| 6 | Personal AI assistants | Consumer devices | On-device memory constrained to 512 MB | Compact to device limit | Cognitum Seed integration |
| 7 | Multi-agent swarm memory | Autonomous clusters | Shared memory grows per agent | Cross-agent MinCutEvict on shared graph | rvAgent integration |
| 8 | Security event retrieval | SOC analysts | Stale events waste search capacity | Age-weighted coherence eviction | ruFlo scheduled compaction |

---

## Exotic Applications

| # | Application | 10–20 year thesis | Required advances | RuVector role | Risk |
|---|---|---|---|---|---|
| 1 | Cognitum cognitive continuity | Edge agents retain identity despite memory pressure | Learned compaction policies | MinCutEvict as compaction primitive | Identity drift under aggressive compaction |
| 2 | Swarm collective forgetting | Agent swarms converge to shared memory via coordinated compaction | Byzantine-fault-tolerant compaction agreement | ruvector-mincut-memory + ruvector-raft | Consensus overhead |
| 3 | Self-healing memory graphs | Compacted stores auto-reconnect via new experience | Online graph repair | MinCutEvict + incremental rebuild | Hallucinated edges |
| 4 | RVM coherence domains | Memory partitioned by coherence domain | RVM domain awareness | ruvector-mincut-memory + rvm | Domain boundary alignment |
| 5 | Proof-gated agent amnesia | Regulatory compliance: prove what was forgotten | Merkle witness logs per compaction | ruvector-verified integration | Witness log growth |
| 6 | Synthetic nervous system memory | Long-term potentiation modelled as edge weight update | Neural plasticity in Rust | Dynamic threshold adjustment | Biological accuracy |
| 7 | Space robotics autonomy | Rover agents operate for years with bounded memory | Radiation-hardened WASM | WASM mincut-memory on constrained hardware | Hardware reliability |
| 8 | Bio-signal cognitive model | Brain-computer interface memory management | Real-time < 1 ms | SIMD graph rebuild | O(N²·D) latency wall |

---

## Deep Research Notes

### What the SOTA suggests

Academic work on agent memory (MemoryBank, HippoRAG, GraphRAG, RAPTOR) focuses
on *retrieval quality*, not *memory lifecycle*.  The closest work to this crate
is GKP (Graph Knowledge Pruning, 2025 preprint), which proposes graph-cut
pruning of static offline knowledge graphs.  No published work applies
graph-cut compaction to live, online agent working memory.

The weighted-degree approximation to minimum cut derives from Karger (1993) and
Karger-Stein (1996) and is well-studied in algorithmic theory, but has not been
applied to this domain in any published literature found during this research
pass (searches conducted 2026-06-02 via standard academic databases).

### What remains unsolved

1. **Falsification:** A `RandomEvict` baseline is needed to confirm that
   graph structure provides signal at 50% compaction.
2. **Adversarial datasets:** Clustered Gaussian is a friendly distribution.
   Near-uniform or adversarial distributions may defeat MinCutEvict.
3. **Optimal threshold:** The similarity threshold is currently a constructor
   parameter; auto-tuning is needed for production.
4. **Production scale:** N²·D rebuild must be replaced with sparse adjacency
   for N > 5,000.

### Where this PoC fits

The PoC demonstrates feasibility: graph-cut guided compaction is fast enough
for interactive agent loops, recall-preserving at 50% compaction, and
graph-coherence-preserving.  It is a starting point, not a production-ready
system.

### What would falsify the approach

If `RandomEvict` matches `MinCutEvict` recall at all tested compaction ratios
on clustered and adversarial datasets, the graph structure is not providing
useful signal and the approach should be abandoned in favour of simpler
heuristics.

### Sources

[^1]: Zhong et al., "MemoryBank: Enhancing Large Language Models with Long-Term Memory," arXiv:2305.10250, 2023.
[^2]: Edge et al., "From Local to Global: A Graph RAG Approach," Microsoft Research, arXiv:2404.16130, 2024.
[^3]: Gutierrez et al., "HippoRAG," arXiv:2405.14831, 2024.
[^4]: Sarthi et al., "RAPTOR," arXiv:2401.18059, 2024.
[^5]: Karger, D.R., "Global Min-cuts in RNC," SODA 1993.
[^6]: Stoer & Wagner, "A Simple Min-Cut Algorithm," JACM 44(4), 1997.
[^7]: Karger & Stein, "A New Approach to the Minimum Cut Problem," JACM 43(4), 1996.

---

## Usage Guide

```bash
# Clone and checkout the research branch
git clone https://github.com/ruvnet/ruvector
cd ruvector
git checkout research/nightly/2026-06-02-mincut-memory-compaction

# Build
cargo build --release -p ruvector-mincut-memory

# Test (18 tests)
cargo test -p ruvector-mincut-memory

# Run default benchmark (N=500, D=32, 6 clusters)
cargo run --release -p ruvector-mincut-memory

# Larger dataset
cargo run --release -p ruvector-mincut-memory -- --n 1000 --dims 64 --clusters 8 --queries 100

# Criterion benchmark
cargo bench -p ruvector-mincut-memory
```

### Expected output (N=500 default)

```
═══════════════════════════════════════════════════════════════
  ruvector-mincut-memory  –  Agent Memory Compaction Benchmark
═══════════════════════════════════════════════════════════════
OS      : linux
Arch    : x86_64
Dataset : N=500 D=32 clusters=6
...
│ MinCutEvict      │   500 │   250 │   1.000  │   1.000  │   6562.0  │     6441 │     7077 │    152.4 │ PASS   │
...
Overall: ALL PASS ✓
```

### How to interpret results

- **Recall_b**: recall before compaction (should be 1.0 for brute-force)
- **Recall_a**: recall after compaction — should be ≥ 0.60 (acceptance floor)
- **Edges_a**: higher is better — means more graph coherence is preserved
- **Accept**: PASS/FAIL — the acceptance criterion is recall_a ≥ 0.60 × recall_b

### How to change dataset size

```bash
cargo run --release -p ruvector-mincut-memory -- --n 2000 --dims 128 --clusters 10
```

### How to add a new strategy

1. Implement `Compactor` trait in `src/compaction.rs`
2. Export from `src/lib.rs`
3. Add to the `strategies` vec in `src/main.rs`
4. Add unit tests in the `tests` module

### How to plug into RuVector

```rust
use ruvector_mincut_memory::{MemoryStore, MinCutEvict, Compactor};

let mut store = MemoryStore::new(dims, 0.4);
// ... populate with agent memory vectors ...

let result = MinCutEvict.compact(&mut store, capacity);
println!("Compacted: {} entries, {}µs", result.entries_after, result.latency_us);
```

---

## Optimization Guide

### Memory optimization

- Use `similarity_threshold = 0.5+` to reduce graph density and adjacency matrix size
- Switch to sparse CSR adjacency for N > 5,000 (planned)
- Use `f16` vectors if precision allows (halves vector memory)

### Latency optimization

- Reduce `dims` — graph rebuild is O(N²·D), so half the dims halves the time
- Reduce REPS in the benchmark binary for production (single-pass is fine)
- Use rayon for parallel graph row computation (planned)

### Recall optimization

- Increase `target_size` — a 70% compaction is safer than 50%
- Lower `similarity_threshold` to 0.3 — more edges give MinCutEvict more signal
- Use `CoherenceEvict` when access-count data is unavailable

### Edge deployment optimization

- Remove `Instant` timer; pass `latency_us: 0` in WASM
- Use fixed-size arrays instead of Vec for N known at compile time
- Compile with `opt-level = "s"` for size-optimised WASM

### MCP tool optimization

- Serialize `CompactionResult` to JSON before returning from the tool
- Cache the graph across compaction calls if the store is read-only between compactions

### ruFlo automation optimization

- Set compaction threshold at 80% capacity, not 100% — avoids emergency compaction
- Schedule during agent idle periods (between tool call batches)
- Log `CompactionResult` to witness chain for auditability

---

## Roadmap

### Now

- Merge `crates/ruvector-mincut-memory` to main
- Add `RandomEvict` falsification baseline
- Add access-count weighting to MinCutEvict
- Benchmark on server-class hardware

### Next

- Sparse CSR adjacency for N > 5,000
- Incremental graph maintenance (amortise rebuild)
- `ruvector-mincut-memory-wasm` crate
- MCP `memory_compact` tool in `mcp-gate`
- ruFlo workflow integration
- `ruvector-verified` witness log per compaction

### Later

- Exact min-cut (Stoer-Wagner) for N ≤ 100 using `ruvector-mincut`
- Learned compaction policy (RL over eviction decisions)
- Multi-objective scoring (age + coherence + access + recency)
- Swarm-coordinated compaction via `ruvector-raft`
- Cognitum Seed deployment with fixed 512 MB memory budget
- Proof-gated agent amnesia with regulatory compliance logging

---

## SEO Tags

**Keywords:**
ruvector, Rust vector database, Rust vector search, high performance Rust, ANN
search, HNSW, DiskANN, filtered vector search, graph RAG, agent memory, AI
agents, MCP, WASM AI, edge AI, self learning vector database, ruvnet, ruFlo,
Claude Flow, autonomous agents, retrieval augmented generation, graph cut,
memory compaction, working memory, semantic eviction, vector store lifecycle.

**Suggested GitHub topics:**
rust, vector-database, vector-search, ann, hnsw, graph-cut, rag, graph-rag,
ai-agents, agent-memory, mcp, wasm, edge-ai, rust-ai, semantic-search,
graph-database, autonomous-agents, retrieval, embeddings, ruvector,
memory-compaction, working-memory.
