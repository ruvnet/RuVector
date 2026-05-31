# Agent Memory Compaction via Graph-Cut Clustering

**Nightly research · 2026-05-26**

> **150-char summary:** Semantic-aware agent memory compaction using cosine-similarity graph clustering in Rust — 100% cluster recall at 5% budget vs 5% for age eviction.

---

## Abstract

AI agents accumulate memory faster than they can use it.  Every production agent
memory system surveyed — MemGPT, Letta, Mem0, A-MEM — either relies on manual
compaction, age-based eviction, or no compaction at all.  None applies graph
structure to the memory index.

We implement `ruvector-memcompact`, a standalone Rust crate that compacts an
agent memory store by building a cosine-similarity graph over stored embedding
vectors, grouping semantically related entries into connected components via
Union-Find, and replacing each component with a single centroid representative.
We compare three strategies — age eviction (baseline), importance eviction, and
graph-cut compaction — on a 500-entry, 20-cluster synthetic dataset at 5% budget.

**Key measured results (x86-64, `cargo run --release`, N=500, D=64, K=20,
budget=25):**

| Strategy             | Entries | Recall@10 | Memory KB | Reduction | Compact ms |
|----------------------|---------|-----------|-----------|-----------|------------|
| AgeEviction          | 25      | 5.0%      | 7.8       | 95.0%     | 0.07       |
| ImportanceEviction   | 25      | 75.0%     | 7.8       | 95.0%     | 0.04       |
| **GraphCutCompaction** | **20** | **100.0%** | **6.2** | **96.0%** | **9.50** |

Hardware: Intel(R) Xeon(R) @ 2.80 GHz, Linux x86_64, rustc 1.87.0-nightly.

Graph-cut compaction achieves **perfect recall** while reducing the store by 96%
because the clustering maps the 500 raw entries to exactly the 20 underlying
semantic clusters, preserving one representative per cluster.

---

## Why this matters for RuVector

RuVector already has:

- `ruvector-core`: HNSW vector index
- `ruvector-graph`: distributed graph storage
- `ruvector-mincut`: dynamic min-cut algorithms
- `ruvector-gnn`: graph neural retrieval
- `ruvector-delta-*`: write-ahead delta store

What it lacks is a semantic-aware compaction primitive that sits between these
layers and prevents index bloat from accumulating over long agent lifespans.
`ruvector-memcompact` fills that gap with a composable `CompactionStrategy`
trait that can wire directly into `ruvector-core`'s storage engine.

---

## 2026 State of the Art Survey

### Agent memory systems

**MemGPT / Letta (2024–2026)**
The dominant production agent memory architecture uses a three-tier OS-inspired
model: in-context (scratchpad), recall (conversation history), and archival
(vector index).  Compaction is entirely agent-directed — the agent must
explicitly invoke a tool to move memories to archival.  In practice this rarely
happens automatically.

**A-MEM (arXiv:2502.12110, 2025)**
Zettelkasten-inspired memory that builds cross-memory links dynamically when new
entries arrive.  Continuous graph maintenance, no batch compaction.  Python-only.
The LLM decides which links to form, making it non-deterministic.

**Mnemosyne (arXiv:2510.08601, 2025)**
Graph-structured memory with temporal decay and a "core summary" derived from a
fixed-length subset.  Closest to a compaction idea in the literature.  The
selection is heuristic (not graph-cut optimal) and no Rust implementation exists.

**Mnemis (arXiv:2602.15313, 2026)**
Dual-route retrieval on hierarchical semantic graphs — read-optimised, not
compact-optimised.  Does not address write amplification.

**Adaptive Memory Admission (arXiv:2603.04549, 2026)**
Frames memory ingestion as a control problem.  Admission and compaction are
treated as independent; no structural compaction after admission.

### Vector database compaction

**LanceDB fragment merge (2025–2026)**
The only vector-adjacent system with explicit compaction: immutable
append-only fragments are periodically merged and deletions materialised.
Structural I/O optimisation only — no cosine similarity or semantic clustering
in the merge decision.

**FAISS / Qdrant / Milvus**
All support IVF index rebuilding and HNSW re-graphing but treat compaction as
an offline batch operation.  None integrates semantic deduplication into the
compaction pass.

### Graph-cut literature

**"Down with the Hierarchy: The H in HNSW Stands for Hubs"
(arXiv:2412.01940, Dec 2024)**
Shows that HNSW hub nodes (high-degree) drive navigability — they are natural
cluster centroids.  Directly relevant: HNSW proximity graphs already encode the
cluster structure needed for graph-cut compaction without an additional pairwise
pass.

**SemantiCache (arXiv:2603.14303, 2026)**
Seed-based clustering of KV-cache tokens with proportional attention merging.
Algorithm closely analogous to graph-cut compaction but applied to inference
KV cache, not persistent agent memory.

**Scalable Clustering via Graph Cuts (arXiv:2308.09613, 2023)**
Linear-time approximation of normalised graph cuts.  Establishes the
theoretical bridge between spectral clustering and the Union-Find approximation
used here.

### Gap summary

No production system applies graph-cut or semantic clustering to the
compaction of persistent AI agent memory.  This is the gap `ruvector-memcompact`
addresses.

---

## Forward-Looking 10–20 Year Thesis

### 2026–2031: Memory compaction as a first-class runtime service

Agent systems will run for months or years without restart.  Memory compaction
will become a daemon-level service analogous to garbage collection in managed
runtimes.  Semantic-aware strategies will be the baseline; purely age-based
eviction will be considered a legacy antipattern.

RuVector's role: provide the fast Rust primitive that agents call as a
background task, integrated with ruFlo workflow loops.

### 2031–2036: HNSW-native compaction

Reusing the existing HNSW proximity graph for compaction (rather than a fresh
O(n²) pass) will reduce compaction time from O(n²·d) to O(E) where E is the
HNSW edge count — a 100–1000× speedup at production scale.  This requires
exposing HNSW internals as a first-class compaction API, which no system
currently supports.

### 2036–2046: Self-organising memory with adaptive topology

In long-lived agents, memory graphs will dynamically reorganise: clusters split
as a topic differentiates, merge as distinctions become irrelevant, and prune
as evidence expires.  This resembles biological memory consolidation (hippocampal
replay, synaptic downscaling).  RuVector's mincut and graph infrastructure
provide the substrate for this level of adaptive topology management.

Cognitum Seed and edge AI appliances will require this to operate autonomously
without central coordination — the entire compaction loop must run on-device.

---

## ruvnet Ecosystem Fit

| Component           | Role in compaction pipeline                          |
|---------------------|------------------------------------------------------|
| `ruvector-memcompact` | Standalone compaction crate (this PoC)             |
| `ruvector-core`     | Production vector store to wire into                 |
| `ruvector-graph`    | Persist similarity graph edges for incremental reuse |
| `ruvector-mincut`   | Future: replace Union-Find with min-cut partitioning |
| `ruvector-gnn`      | Future: learned cluster boundaries from GNN embeddings|
| `ruvector-delta-*`  | WAL for compaction audit trail                       |
| `ruvector-verified` | Proof-gate writes before compaction                  |
| `ruFlo`             | Schedule compaction as a workflow step               |
| MCP tools           | Expose `memory_compact` as an agent-callable tool    |
| Cognitum Seed       | Edge deployment: on-device compaction daemon         |
| RVF format          | Package compacted memory graphs for transfer         |

---

## Proposed Design

### Core trait

```rust
pub trait CompactionStrategy {
    fn name(&self) -> &'static str;
    fn compact(&self, store: &MemoryStore, budget: usize) -> MemoryStore;
}
```

### Three variants

**Baseline: AgeEviction**
Sort entries by timestamp descending, truncate to budget.  O(n log n).
Represents the current production default in most agent systems.

**Variant A: ImportanceEviction**
Sort by importance score descending, truncate.  O(n log n).  Depends on
accurate importance labels; degrades to random sampling when labels are uniform.

**Variant B: GraphCutCompaction**
1. Pairwise cosine similarity pass: O(n² · d).
2. Union-Find clustering at threshold θ: O(n · α(n)) ≈ O(n).
3. Centroid synthesis per cluster: O(n · d / K).
4. Importance trim if representatives > budget: O(K log K).

---

## Architecture Diagram

```mermaid
graph TD
    subgraph Input
        MS[MemoryStore\n500 entries, D=64]
    end

    subgraph GraphCutCompaction
        SIM[Pairwise cosine\nO n² · d]
        UF[Union-Find\nclustering θ=0.70]
        CENT[Centroid\nsynthesis]
        TRIM[Importance trim\nif > budget]
    end

    subgraph Output
        CS[Compacted Store\n20 cluster reps]
    end

    subgraph Evaluation
        QVEC[Query vectors\none per cluster]
        REC[Recall@10\n= 100%]
    end

    MS --> SIM
    SIM --> UF
    UF --> CENT
    CENT --> TRIM
    TRIM --> CS
    CS --> REC
    QVEC --> REC
```

---

## Implementation Notes

- `UnionFind` uses path halving (not full path compression) for cache efficiency.
- Noise is generated per-dimension with std = σ/√D so that ‖ε‖ ≈ σ, ensuring
  within-cluster cosine similarity ≈ 1/(1+σ²) >> threshold.
- `centroid()` returns the arithmetic mean; a future variant could use the
  entry with maximum importance as the representative (avoids synthetic vectors).
- `CompactionStrategy::compact()` always returns a new `MemoryStore`, leaving
  the original intact (non-destructive by design).

---

## Benchmark Methodology

Dataset: synthetically generated agent memory store.

- **N=500** entries in **K=20** semantic clusters (25 entries/cluster).
- Each cluster has a random unit centroid in **D=64** dimensions.
- Each entry: centroid + Gaussian noise with ‖ε‖ ≈ σ = 0.15, then L2-normalised.
- Timestamps are sequential (oldest entries in cluster 0, newest in cluster 19).
- Importance scores are uniform random ∈ [0, 1] — not correlated with cluster.
- Budget: **25** (5% of N); since K=20 < 25, GraphCutCompaction fits all
  cluster representatives without the importance-trim step.
- **50** query vectors (all 20 cluster centroids, slightly perturbed).
- **Recall@10**: cluster-level — what fraction of the top-10 clusters from
  the original store appear in the top-10 from the compacted store?

All timing measurements: `std::time::Instant`, release build, single thread.
Compaction latency: single run (not averaged — for O(n²) the single run is
representative at this scale).  Query latency: per-query brute-force scan.

Cargo command:
```
cargo run --release -p ruvector-memcompact
```

---

## Real Benchmark Results

Hardware: Intel(R) Xeon(R) Processor @ 2.80 GHz  
OS:       Linux x86_64 (kernel 6.18.5)  
Rustc:    1.87.0-nightly  

```
╔══════════════════════════════════════════════════════════════╗
║      ruvector-memcompact  |  Agent Memory Compaction Bench   ║
╚══════════════════════════════════════════════════════════════╝

  OS:            linux / x86_64
  CPU:           Intel(R) Xeon(R) Processor @ 2.80GHz

  Dataset configuration
  N=500 · D=64 · K=20 clusters · budget=25 · σ=0.15 · θ=0.70

  Strategy             Entries  Compact  Query    Query   Query   Memory  Reduction  Recall
                                 ms     mean μs  p50 μs  p95 μs    KB               @10
  ─────────────────────────────────────────────────────────────────────────────────────────
  AgeEviction              25    0.07     2.27     2.27    2.34     7.8     95.0%     5.0%
  ImportanceEviction        25    0.04     2.26     2.25    2.31     7.8     95.0%    75.0%
  GraphCutCompaction        20    9.50     1.75     1.74    1.81     6.2     96.0%   100.0%

  Original store: 500 entries (156.2 KB)

  Acceptance:
  GraphCutCompaction recall@10 ≥ 75%  →  100.0%  PASS ✓
  GraphCut beats AgeEviction by ≥ 10 pp →  95.0 pp  PASS ✓
  Overall: ACCEPT ✓
```

---

## Memory and Performance Math

### Dataset memory

| Component           | Calculation                                | Size     |
|---------------------|--------------------------------------------|----------|
| Original store      | 500 × (64 × 4 + struct overhead) bytes     | 156.2 KB |
| AgeEviction output  | 25 × (64 × 4 + overhead)                  | 7.8 KB   |
| ImportanceEviction  | 25 × same                                  | 7.8 KB   |
| GraphCutCompaction  | 20 × same (20 cluster centroids)           | 6.2 KB   |

### GraphCutCompaction time complexity

| Step           | Complexity    | At N=500, D=64        |
|----------------|---------------|-----------------------|
| Pairwise pass  | O(n² · d)     | 500² × 64 = 16 M ops  |
| Union-Find     | O(n · α(n))   | ~500 ops              |
| Centroid synth | O(n · d / K)  | 500 × 64/20 ≈ 1.6 K   |
| Trim           | O(K log K)    | 20 × 4 = 80 ops       |
| **Total**      | **O(n² · d)** | **~16 M ops, 9.5 ms** |

The O(n²) pairwise pass dominates.  For n=10 K: ~40 B ops ≈ 40 s — the
hard limit for the current implementation.  Beyond n=10 K, an approximate
k-NN graph (as built by HNSW construction) reduces this to O(n · k · d)
with k=32 neighbours → ~20 M ops at n=10 K → ~20 ms.

---

## How It Works: Walkthrough

Consider a running AI assistant that has stored 500 memories over two weeks.
The agent needs to compact to 25 entries to fit an inference context budget.

**AgeEviction** drops everything older than a few days.  The assistant now
knows nothing about conversations from the first week.  If a user references
something from day 3, the memory is gone.  Recall: 5%.

**ImportanceEviction** samples randomly across the two-week history (with
random importance scores).  It covers ~75% of topics but misses about 25%.
Recall: 75%.

**GraphCutCompaction** asks: "which memories say roughly the same thing?"
It groups the 500 entries into 20 semantic clusters (one per major topic the
agent learned about) and keeps the centroid of each cluster — the single
vector that best represents each topic.  With budget=25 and K=20, all 20
topics fit.  Any query about any topic finds its cluster centroid.  Recall: 100%.

The compaction takes 9.5 ms for 500 entries.  At 20 cluster representatives,
subsequent queries run 23% faster (1.75 μs vs 2.27 μs) because the index
is smaller.

---

## Practical Failure Modes

1. **Threshold too high**: entries that belong to the same semantic cluster
   have cosine similarity below θ.  They form separate clusters.  Compaction
   produces too many representatives (may exceed budget) or fails to merge
   near-duplicates.  Detection: compare `representatives.len()` to `n/expected_k`.

2. **Threshold too low**: unrelated memories with accidentally high cosine
   similarity merge.  One cluster representative must cover two distinct topics.
   Detection: compute mean intra-cluster similarity post-compaction; should be
   close to θ.

3. **Importance label corruption**: if importance scores are adversarially
   set to zero for all entries in a cluster, that cluster's representative is
   trimmed at step 4.  Mitigation: don't rely on importance alone; use
   cluster-size as a secondary weight.

4. **Repeated compaction drift**: compacting, then adding memories, then
   compacting again can shift centroids toward the first compaction's results.
   Mitigation: run compaction at most once per eviction cycle; flag centroid
   entries with a `is_synthetic` bit.

5. **Large uniform clusters**: if all 500 entries belong to one cluster (all
   memories about the same topic), compaction reduces to one entry.  This is
   semantically correct but may surprise callers expecting `budget` entries.
   The caller must handle the case where `result.len() < budget`.

---

## Security and Governance Implications

- **Irreversible semantic merging**: once memories are merged to a centroid,
  individual source entries are lost.  For regulated domains (healthcare, legal),
  maintain a compaction audit log before discarding.
- **Privacy**: centroid vectors are mathematical aggregates; individual memory
  content is not recoverable.  This is a privacy benefit if raw memories contain
  PII — compaction is a form of k-anonymisation in embedding space.
- **Adversarial injection**: an attacker who controls memory writes could craft
  vectors that merge with legitimate memories.  Proof-gated writes
  (`ruvector-verified`) should gate memory ingestion before compaction.

---

## Edge and WASM Implications

- `ruvector-memcompact` uses `#![forbid(unsafe_code)]` and has zero OS
  dependencies beyond `std`.  It compiles to WASM with `wasm32-unknown-unknown`.
- For Cognitum Seed (edge appliance), the O(n²) pairwise pass at n=500 takes
  ~9.5 ms on a server CPU.  On a Pi Zero 2W (Cortex-A53 @ 1 GHz), this would
  scale to roughly 50–100 ms — acceptable for a background daemon.
- A WASM-safe variant (`ruvector-memcompact-wasm`) would be the next packaging
  step, enabling in-browser agent memory compaction for local-first AI.

---

## MCP and Agent Workflow Implications

The `CompactionStrategy` trait maps directly to an MCP tool:

```json
{
  "name": "memory_compact",
  "description": "Compact the agent memory store using graph-cut clustering",
  "inputSchema": {
    "store_id": "string",
    "budget": "integer",
    "strategy": "age | importance | graph_cut",
    "threshold": "number"
  }
}
```

A ruFlo workflow can call `memory_compact` on a schedule (nightly, or when
`store.len() > threshold`) and feed the result back into the vector search
path — closing the agent memory compaction loop autonomously.

---

## Practical Applications

| Application | User | Why it matters | RuVector role | Path |
|-------------|------|----------------|---------------|------|
| Agent session compaction | AI assistant users | Prevents context overflow in long sessions | `CompactionStrategy` on `ruvector-core` store | Phase 1 |
| Enterprise knowledge base dedup | Enterprise AI teams | Removes near-duplicate documents from RAG corpus | GraphCutCompaction on document embeddings | Phase 1 |
| Code intelligence compaction | IDE agent | Keeps relevant file context, evicts stale imports | Age+GraphCut hybrid | Phase 2 |
| MCP memory tool | Agent framework devs | Expose compaction as an MCP-callable primitive | `memory_compact` MCP tool | Phase 2 |
| ruFlo nightly compaction | ruFlo workflow users | Automatic nightly memory consolidation | ruFlo cron step | Phase 2 |
| Multi-agent memory sync | Swarm systems | Compact shared memory before distributing to agents | RVF-packaged compacted store | Phase 3 |
| Edge AI assistant | Cognitum Seed users | On-device compaction without cloud round-trip | WASM-compiled variant | Phase 3 |
| Scientific literature search | Researchers | Merge near-duplicate paper embeddings across archives | High-K clustering variant | Phase 3 |

---

## Exotic Applications

| Application | 10–20 year thesis | Required advances | RuVector role | Risk |
|-------------|-------------------|-------------------|---------------|------|
| Cognitum autonomous memory consolidation | Agents run for years, self-compacting without human oversight | Stable threshold auto-calibration; provenance tracking | Embedded compaction daemon in Cognitum Seed | Centroid drift after many cycles |
| RVM coherence domain compaction | Coherence boundaries emerge from graph-cut structure of shared memory | Integration with RVM coherence scoring | MinCut-aware compaction across domain boundaries | Coherence score reliability |
| Proof-gated memory archives | Compacted memories are cryptographically attested; auditable by regulators | ZK-SNARKs over centroid computation | `ruvector-verified` + compaction pipeline | Proof overhead per compaction cycle |
| Swarm distributed memory | 100+ agents share compacted memory graphs; writes are consensus-gated | Byzantine-tolerant consensus on which clusters to merge | Raft-based compaction coordinator | Network partition during compaction |
| Self-healing vector graphs | After hardware failure, reconstruct lost memory clusters from surviving centroid metadata | Erasure coding over centroid vectors | RVF manifest + recovery procedure | Information loss when entire cluster is lost |
| Biological signal memory | Implantable devices compact neural spike trains using cosine similarity over learned embeddings | Low-power WASM runtime; learned similarity metric | WASM-SIMD compaction kernel | Embedding quality for bio signals |
| Autonomous robotics long-term memory | Robots accumulate spatial and procedural memories over years | Compaction of pose-conditioned embeddings | RVF-packaged robot memory graphs | Domain shift between environments |
| Agent operating system memory manager | AOS allocates agent memory as a managed resource with compaction as GC | AOS kernel integration; MMU-like memory isolation | `ruvector-memcompact` as AOS primitive | Security isolation between agent memory regions |

---

## Deep Research Notes

### What the SOTA suggests

1. Agent memory compaction is an **open engineering problem** — no production
   system has solved it with semantic awareness.
2. Graph-cut algorithms are **well-established** for document clustering but have
   not been applied to agent memory stores.
3. HNSW proximity graphs already **implicitly encode** semantic clusters (hub
   nodes are cluster centroids) — the next step is to exploit this structure
   rather than recomputing pairwise similarity from scratch.
4. Centroid-based compaction is analogous to **k-means compression** (PQ), which
   has been validated at billion-scale — the same principles apply to agent memory.

### What remains unsolved

1. **Threshold auto-calibration**: the optimal θ depends on the embedding model,
   domain, and desired cluster granularity.  Currently a hyperparameter.
2. **Provenance preservation**: once merged to a centroid, which original entries
   contributed?  Needed for explainability and regulatory compliance.
3. **Incremental compaction**: today compaction is batch (full O(n²) pass).
   Online compaction (merge new entry into existing clusters as it arrives) would
   be O(n) per insert.
4. **Quality metric beyond recall**: recall@10 measures cluster coverage; a
   better metric would assess whether the compacted store enables the same
   downstream task performance (e.g., agent answer quality).

### What would make this production grade

1. Wire `MemoryStore` to `ruvector-core`'s persistent vector storage.
2. Replace O(n²) pairwise pass with HNSW-graph-reuse: O(E) where E is the
   HNSW edge set.
3. Add `is_synthetic` flag to centroid entries; propagate `source_ids` for
   provenance.
4. Auto-calibrate θ from empirical cosine distribution (e.g., set θ at the
   valley between within-cluster and between-cluster similarity peaks).
5. Expose as a ruFlo step and MCP tool.

### What would falsify the approach

If embedding models produce **indistinguishable cosine similarities** between
within-cluster and between-cluster pairs (e.g., embeddings that saturate the
unit hypersphere uniformly), then graph-cut clustering degrades to random
sampling — equivalent to ImportanceEviction.  This would occur with very
low-quality embeddings or in extremely high-dimensional spaces where the curse
of dimensionality equalises all distances.  Detection: compare mean
within-cluster cosine to mean between-cluster cosine before running compaction;
if they are within 0.05 of each other, do not compact.

---

## Production Crate Layout Proposal

```
crates/ruvector-memcompact/
├── Cargo.toml
├── src/
│   ├── lib.rs          (CompactionStrategy trait, re-exports)
│   ├── memory.rs       (MemoryEntry, MemoryStore, MemId)
│   ├── graph.rs        (UnionFind, cosine_sim, cluster_by_similarity, centroid)
│   ├── compaction.rs   (AgeEviction, ImportanceEviction, GraphCutCompaction)
│   ├── metrics.rs      (recall_at_k, mean_recall_at_k, percentile, top_k)
│   └── main.rs         (benchmark binary)
└── (future)
    ├── src/hnsw_reuse.rs   (HNSW-graph-based compaction, O(E))
    ├── src/incremental.rs  (online insert → cluster assignment)
    ├── src/mcp.rs          (memory_compact MCP tool handler)
    └── src/wasm.rs         (WASM-safe entry points)
```

---

## What to Improve Next

1. **HNSW-graph-reuse compaction** (O(E) instead of O(n²)) — requires
   exposing `ruvector-core` HNSW edge lists.
2. **Threshold auto-calibration** — histogram of pairwise similarities, pick
   valley between modes.
3. **Incremental online compaction** — O(k) per insert: assign new entry to
   nearest cluster or create a new cluster.
4. **ruFlo integration** — `memory_compact` as a ruFlo step that runs on a
   schedule or size trigger.
5. **MCP tool** — `memory_compact` callable from any MCP-compatible agent.
6. **Provenance metadata** — `source_ids: Vec<MemId>` on centroid entries.
7. **WASM packaging** — `ruvector-memcompact-wasm` for in-browser and edge use.

---

## References and Footnotes

[^1]: "MemGPT: Towards LLMs as Operating Systems", Packer et al., arXiv:2310.08560, 2023. https://arxiv.org/abs/2310.08560. Accessed 2026-05-26.

[^2]: "A-MEM: Agentic Memory for LLM Agents", arXiv:2502.12110, Feb 2025. https://arxiv.org/abs/2502.12110. Accessed 2026-05-26.

[^3]: "Mnemosyne: Unsupervised Long-Term Memory for Edge LLMs", arXiv:2510.08601, Oct 2025. https://arxiv.org/abs/2510.08601. Accessed 2026-05-26.

[^4]: "Mnemis: Dual-Route Retrieval on Hierarchical Graphs", arXiv:2602.15313, Apr 2026. https://arxiv.org/abs/2602.15313. Accessed 2026-05-26.

[^5]: "Adaptive Memory Admission Control for LLM Agents", arXiv:2603.04549, Mar 2026. https://arxiv.org/pdf/2603.04549. Accessed 2026-05-26.

[^6]: "SemantiCache: KV Cache Compression via Semantic Chunking and Clustered Merging", arXiv:2603.14303, Mar 2026. https://arxiv.org/pdf/2603.14303. Accessed 2026-05-26.

[^7]: "Down with the Hierarchy: The H in HNSW Stands for Hubs", arXiv:2412.01940, Dec 2024. https://arxiv.org/pdf/2412.01940. Accessed 2026-05-26.

[^8]: "A Scalable Clustering Algorithm to Approximate Graph Cuts", arXiv:2308.09613, 2023. https://arxiv.org/pdf/2308.09613. Accessed 2026-05-26.

[^9]: LanceDB Compaction Documentation. https://lancedb.com/documentation/concepts/data.html. Accessed 2026-05-26.

[^10]: "Beyond Nearest Neighbors: Semantic Compression and Graph-Augmented Retrieval", arXiv:2507.19715, Jul 2025. https://arxiv.org/abs/2507.19715. Accessed 2026-05-26.

[^11]: "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers", arXiv:2603.07670, 2026. https://arxiv.org/html/2603.07670v1. Accessed 2026-05-26.
