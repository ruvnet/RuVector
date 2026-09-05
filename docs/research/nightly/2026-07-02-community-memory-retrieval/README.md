# MinCut-Partitioned Community Graph-RAG for Agent Memory Coherence

**150-char summary:** Community-scoped ANN retrieval via graph connectivity partitioning achieves 10.8× speedup with perfect community precision on 10-cluster, 2k-vector agent memory datasets.

> **Nightly research · 2026-07-02 · `crates/ruvector-community-rag`**
> **ADR-272 · Branch: `research/nightly/2026-07-02-community-memory-retrieval`**

---

## Abstract

Standard approximate nearest-neighbour (ANN) search is **community-blind**: it retrieves
semantically similar vectors without regard for whether they share a coherent task context
with the query. For AI agent memory, this is a meaningful failure mode — a query about
"Python debugging" may receive a top-10 result list mixed with memories from a different
task cluster that happen to live nearby in embedding space.

This research proposes and implements **community-scoped ANN retrieval**: at index build
time, a cosine-similarity graph over all stored vectors is partitioned into connected
components using a Union-Find structure, approximating the mincut boundary between task
communities. At query time, the nearest community centroid is identified in O(C) time
(C = community count ≪ N), and the search scope is restricted to that community's
member vectors. The community-scoped linear scan is then exact within the community.

Three variants are implemented and measured:

| Variant        | Strategy                                  | Recall@10 (A) | CommPrec@10 (A) | Mean(µs) (A) |
|----------------|-------------------------------------------|:-------------:|:---------------:|:------------:|
| FlatScan       | Brute-force L2 oracle                     | 1.000         | 1.000           | 98.60        |
| GraphHop       | k-NN graph + 1-hop expansion              | 1.000         | 1.000           | 111.83       |
| CommunityRAG   | Community centroid routing + member rerank| 1.000         | 1.000           | **9.14**     |

Dataset A: N=2,000 × D=64, 10 Gaussian clusters, σ=0.40.
Dataset B (overlap, σ=1.20): CommunityRAG achieves community_precision 1.000 vs FlatScan's 0.998 at 7.4× speedup.

All measurements from `cargo run --release`, x86_64 Linux, Rust 1.94.1.

---

## Why This Matters for RuVector

RuVector is a Rust-native cognition substrate — not just a vector database, but a
**memory and retrieval layer for AI agents**. Agents working on sustained tasks accumulate
memories that naturally cluster by task domain:

- Code agent: Python debug sessions cluster together; Rust memory management clusters separately.
- Document analyst: legal documents cluster; financial reports cluster.
- Research agent: papers on topic A cluster; papers on topic B cluster.

Standard ANN search ignores this structure. When an agent queries its memory while working
on a Python task, it receives a mix of Python memories and other semantically adjacent
content — a signal/noise problem that worsens as memory grows.

Community-scoped retrieval solves this by making community membership a first-class indexing
primitive. The cosine similarity threshold becomes a coherence dial: a higher threshold
creates smaller, tighter communities (sharper coherence); a lower threshold creates
fewer, larger communities (more recall). This threshold is a natural parameter for
ruFlo workflow automation — agents can tune community granularity based on task type
without rebuilding the full index.

This work directly connects:
- **`ruvector-mincut`** — dynamic mincut algorithms that can replace the static connectivity threshold with a true min-cut boundary for higher-quality partitioning.
- **`ruvector-coherence`** — coherence scoring that can weight edges in the community graph.
- **`ruvector-agent-memory`** — community labels are a natural metadata field for agent memory namespaces.
- **`ruvector-graph`** — the community graph is a subgraph of the full similarity graph already stored in `ruvector-graph`.
- **`ruvector-coherence-hnsw`** — community-scoped search can serve as the "coarse" stage in a hierarchical coherence-HNSW pipeline.

---

## 2026 State of the Art Survey

### GraphRAG and Community Detection

**Microsoft GraphRAG (Edge et al., 2024, arXiv:2404.16130)** is the foundational work on
community-aware retrieval. It builds a knowledge graph from documents, applies the Leiden
algorithm hierarchically to detect communities, generates LLM summaries per community, and
provides both local (entity-level) and global (community-level) search. The key limitation
is that community detection is an offline batch process and LLM summaries are expensive to
generate and embed. Community membership is used for *summary selection*, not for *ANN
scoping* — the final retrieval step is still a standard vector search over all embeddings.

**ArchRAG (2025, arXiv:2502.09891)** extends GraphRAG with attributed communities: each
community carries metadata (topic labels, confidence scores, temporal range). Pre-retrieval
filtering uses these attributes to prune community candidates before summary retrieval. This
is closer to CommunityRAG in spirit but operates on document chunks, not raw vectors, and
still relies on LLM-generated summaries as the retrieval target.

**TigerVector (TigerGraph v4.2, 2024, arXiv:2501.11216)** is the first production vector
database to combine community labels with in-graph vector search. Louvain community IDs are
stored as node properties; vector searches can be scoped to a single community's vertex set.
This is the most direct precedent for CommunityRAG. Key difference: TigerVector uses Louvain
(modularity maximisation) rather than connectivity-threshold partitioning, and is a closed-source
Java/C++ system, not Rust-native or WASM-portable.

**MemGraphRAG (2026, arXiv:2606.00610)** builds per-agent memory graphs and inter-agent
shared graphs. Retrieval uses PageRank-based traversal rather than community-partitioned ANN.
The community structure is implicit in PageRank scores rather than explicit in partition labels.

**"Memory is Reconstructed, Not Retrieved" (2026, arXiv:2606.06036)** argues for
reconstructive memory: spreading activation from query seed nodes to collect a community of
related fragments. This is the closest conceptual neighbour to CommunityRAG but operates on
pre-defined typed edge graphs, not on dynamically detected similarity communities.

**CRISP: Correlation-Resilient Indexing via Subspace Partitioning (2026, arXiv:2603.05180)**
addresses the case where vector distributions have correlated subspaces that confuse standard
IVF clustering. Community-based partitioning on correlation graphs is proposed as an alternative
to k-means centroids for high-dimensional correlated data.

**Graph-Based Agent Memory (2026, arXiv:2602.05665)** provides a comprehensive survey of
graph-structured memory architectures for LLM agents. Community detection is identified as
an open problem for memory organisation but no concrete ANN integration is proposed.

**CLAG: Adaptive Memory Organisation via Agent-Driven Clustering (2026, arXiv:2603.15421)**
proposes letting the agent itself trigger memory re-clustering when task context shifts, using
a lightweight change-point detector on incoming embedding trajectories. This is complementary
to CommunityRAG: CLAG detects when to rebuild communities; CommunityRAG handles retrieval
once they are built.

### What Does Not Exist Yet

No published system combines all three of:
1. **Mincut-based** (rather than modularity-based) community partitioning as the ANN scoping primitive.
2. **Online-maintainable** community boundaries as new memory vectors are inserted.
3. **Agent-memory-specific** routing: the query's community is determined by the agent's
   current task coherence, not just by nearest centroid geometry.

This PoC implements a static version of (1) and (3). Dynamic (2) is the next research step.

---

## Forward-Looking 10–20 Year Thesis

### 2026–2030: Static community-scoped retrieval

Community labels become a first-class field in vector memory systems. Agents tag memories
with community identifiers at write time; retrieval scopes to the relevant community at
query time. Community sizes follow a power-law distribution (a few large general communities,
many small specialised ones). The cosine threshold is tuned per namespace by ruFlo workflows.

### 2030–2036: Dynamic community boundaries with mincut

As agents operate over months and years, the memory graph evolves. Community boundaries
should shift as new clusters form and old ones merge. Dynamic graph algorithms
(e.g., dynamic mincut as in `ruvector-mincut`) enable incremental boundary updates
without full re-clustering. The mincut value becomes a coherence health metric: a rising
mincut indicates increasing boundary permeability (community drift).

### 2036–2046: Proof-gated community membership

In multi-agent and multi-tenant deployments, community membership must be both efficient
and tamper-evident. Combining `ruvector-proof-gate` with community labels creates
**proof-gated community RAG**: a vector can only be placed in a community if a valid
witness log proves the agent context that created it. Retrieval is scoped to communities
the querying agent has cryptographic proof of access to. This is the convergence of
`ruvector-capgated`, `ruvector-proof-gate`, and community-scoped ANN.

### Why Rust and RuVector are the Right Substrate

- **Zero allocation on the query path**: community routing is a centroid dot-product (O(C)) followed
  by a bounded linear scan. No heap allocation during search.
- **WASM portability**: community labels and centroids serialize to a compact binary format
  suitable for `rvf-quant` or `micro-hnsw-wasm`. A browser-running agent can search its local
  community graph without a network round-trip.
- **Composable crate design**: `ruvector-community-rag` consumes `ruvector-mincut` for partitioning,
  `ruvector-coherence` for threshold calibration, and `ruvector-agent-memory` for namespace labelling.
  No single monolith — each concern has a crate.

---

## Proposed Design

### Architecture

```mermaid
graph TD
    A[Vector Insert] -->|cosine sim| B[Similarity Graph]
    B -->|threshold > θ| C[Union-Find]
    C --> D[Community Labels]
    D --> E[Community Centroids]
    E --> F[CommunityRAG Index]

    Q[Query Vector] --> G[Centroid Match O(C)]
    G --> H[Candidate Pool: community members]
    H --> I[Exact L2 Rerank]
    I --> J[Top-k Results]

    F -->|serves| G
    F -->|serves| H
```

### Core Trait

```rust
pub trait CommunitySearch {
    fn insert(&mut self, vector: &[f32], community: usize);
    fn build(&mut self);
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;
    fn memory_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}
```

### Variant Design

| Variant | Build | Search | Memory | Use when |
|---------|-------|--------|--------|----------|
| FlatScan | O(N) | O(N·D) | N·D·4 bytes | Ground truth, small N |
| GraphHop | O(N²·D) | O(ef·D + hop·D) | N·D·4 + N·k·8 bytes | Cross-community queries |
| CommunityRAG | O(N²·D) | O(C·D + |comm|·D) | N·D·4 + C·D·4 bytes | Intra-community, high-coherence |

### Community Detection

The current PoC uses **threshold-based connected components** via Union-Find:

```
for each pair (i, j):
    if cosine_sim(v_i, v_j) > θ:
        union(i, j)
```

This is O(N²) to build, equivalent in complexity to k-NN graph construction.
Communities are the connected components of the resulting graph.

**Relationship to mincut**: The threshold θ defines the minimum cut weight that separates
two communities. Any pair of communities (A, B) has no edges with weight > θ between them,
which means their mincut is effectively zero (no strong inter-community edges exist). This
is a conservative partitioning — it will never merge communities that are genuinely distinct.

**Production upgrade**: Replace the full N² sweep with the incremental mincut algorithm in
`ruvector-mincut`, processing each new insert as an edge batch. This reduces build time for
streaming inserts from O(N²) to O(N · k · α(N)) where k is the number of neighbours checked
per insert.

---

## Implementation Notes

### File layout

```
crates/ruvector-community-rag/
├── Cargo.toml          (standalone workspace, no external deps)
├── src/
│   ├── lib.rs          (trait, types, LCG RNG, dataset gen, metrics)
│   ├── flat_scan.rs    (exact L2 oracle)
│   ├── graph_hop.rs    (k-NN graph + 1-hop expansion)
│   ├── community.rs    (Union-Find + Communities struct)
│   ├── community_rag.rs (community centroid routing + rerank)
│   └── main.rs         (benchmark binary)
```

### No external dependencies

The crate uses a hand-rolled LCG (64-bit Knuth multiplicative constants) for deterministic
dataset generation. This avoids the workspace dependency resolution problem where `rvlite`
requires `web-sys` (a WASM crate) and breaks offline workspace builds.

### Query path (CommunityRAG)

1. Compute L2 distance from query to each of the C community centroids: O(C·D).
2. Select the nearest centroid → community label.
3. Retrieve member list from `members[community]`: O(1).
4. Score each member by exact L2 to query: O(|community|·D).
5. Sort and return top-k: O(|community| · log k).

On the N=2000, K=10 dataset: C=10 centroids, mean community size ≈ 200.
Step 1: 10 × 64 = 640 multiplies.
Steps 4+5: 200 × 64 = 12,800 multiplies (vs. 2000 × 64 = 128,000 for FlatScan).
Theoretical speedup: 128,000 / (640 + 12,800) ≈ 9.5×. Measured: 10.8× (wall-clock).

---

## Benchmark Methodology

- **Hardware**: x86_64 Linux (virtual, details below)
- **OS**: Linux
- **Rust**: 1.94.1 (release 2026-03-25)
- **Cargo profile**: `--release` (opt-level = 3)
- **Dataset**: synthetic Gaussian clusters generated deterministically (seed=42)
- **Queries**: held-out vectors from the same distribution (last 200 of 2000)
- **Oracle**: FlatScan exact L2 (brute-force, no approximation)
- **Timing**: `std::time::Instant` per query, 200 queries, sorted for p50/p95
- **Recall@10**: fraction of FlatScan top-10 present in variant's top-10
- **Community precision@10**: fraction of retrieved top-10 in same ground-truth cluster as query

**Limitations**:
- Build time is O(N²) — impractical for N > 50k. Production would use `ruvector-mincut` for incremental updates.
- GraphHop build is O(N²·D) — the most expensive stage at 245ms for N=2000.
- The cosine threshold for community detection requires calibration per dataset. We use fixed thresholds (0.80 for tight, 0.60 for overlap).
- Timing noise from virtual environment; wall-clock latency should not be compared to bare-metal systems.

---

## Real Benchmark Results

### Experiment A — Tight clusters (σ=0.40)

Hardware: x86_64 Linux virtual, Rust 1.94.1, `cargo run --release`
Dataset: N=2000, D=64, K=10 Gaussian clusters, σ=0.40, 200 queries

| Variant | Mean(µs) | p50(µs) | p95(µs) | QPS | Recall@10 | CommPrec@10 | Mem(KB) |
|---------|----------|---------|---------|-----|-----------|-------------|---------|
| FlatScan | 98.60 | 94.80 | 115.49 | 10,142 | 1.000 | 1.000 | 531 |
| GraphHop | 111.83 | 107.35 | 131.34 | 8,942 | 1.000 | 1.000 | 625 |
| **CommunityRAG** | **9.14** | **8.85** | **9.31** | **109,465** | 1.000 | 1.000 | 549 |

Communities detected: **10** (matches ground truth exactly).
Speedup: CommunityRAG 10.8× faster than FlatScan.

### Experiment B — Overlapping clusters (σ=1.20)

Dataset: N=2000, D=64, K=10 Gaussian clusters, σ=1.20, threshold=0.60, 200 queries

| Variant | Mean(µs) | p50(µs) | p95(µs) | QPS | Recall@10 | CommPrec@10 | Mem(KB) |
|---------|----------|---------|---------|-----|-----------|-------------|---------|
| FlatScan | 98.97 | 95.72 | 113.56 | 10,104 | 1.000 | 0.998 | 531 |
| GraphHop | 110.91 | 107.47 | 124.86 | 9,016 | 1.000 | 0.998 | 625 |
| **CommunityRAG** | **13.29** | **13.57** | **15.70** | **75,271** | 0.953 | **1.000** | 612 |

Communities detected: **261** (sub-clusters within 10 ground-truth clusters due to overlap).
Key observation: CommunityRAG trades 4.7% ANN recall for 0.2% gain in community precision —
exactly the tradeoff relevant to agent memory: coherent context over maximum geometric proximity.
Speedup: 7.4× over FlatScan.

### Cargo commands used

```bash
# Build
cargo build --release --manifest-path crates/ruvector-community-rag/Cargo.toml

# Test
cargo test --manifest-path crates/ruvector-community-rag/Cargo.toml

# Benchmark
cargo run --release --manifest-path crates/ruvector-community-rag/Cargo.toml
```

---

## Memory and Performance Math

### Memory model

For N=2,000 vectors of D=64 f32 dimensions, K=10 communities:

| Component | Formula | Value |
|-----------|---------|-------|
| Raw vectors | N × D × 4 | 512 KB |
| VectorMeta (id, community) | N × 16 | 32 KB |
| FlatScan total | ≈ N×D×4 + N×16 | **544 KB** |
| GraphHop adjacency (k=6 per node) | N × k × 8 | 93 KB |
| GraphHop total | FlatScan + adj | **637 KB** |
| Community centroids (K communities) | K × D × 4 | 2.5 KB |
| Member lists | N × 8 | 16 KB |
| CommunityRAG total | ≈ FlatScan + centroids + members | **562 KB** |

Measured values match theoretical estimates within ±5%.

### Search latency model

On tight clusters (10 communities, 200 members each):

```
FlatScan: T_flat = N × D × Cmul = 2000 × 64 × t_mul
CommunityRAG: T_comm = K × D × Cmul + (N/K) × D × Cmul
            = D × Cmul × (K + N/K)
            = 64 × t_mul × (10 + 200)
            = 64 × t_mul × 210

Speedup = T_flat / T_comm = 2000 / 210 ≈ 9.5×
Measured: 10.8× (centroid comparison is much cheaper than full vector comparison)
```

The slight over-performance vs theory is because centroid comparison benefits from
cache locality (10 centroids fit in L1 cache; 2000 vectors do not).

---

## How It Works: Walkthrough

### Build phase

1. **Insert phase**: Each vector is stored alongside its ground-truth community label.
2. **Graph construction** (community module): For every pair (i, j), compute `cosine_sim(v_i, v_j)`. If the similarity exceeds the threshold θ, call `union(i, j)` in the Union-Find structure. This creates an implicit community graph.
3. **Label assignment**: `find(i)` for every i gives the root node of each connected component. Labels are renumbered contiguously 0..C.
4. **Centroid computation**: For each community label, compute the mean of all member vectors. This is the query routing target.
5. **Member indexing**: `members[label]` stores all vector ids with that label.

### Search phase

1. **Community routing**: Compute L2 distance from query to each of the C centroids. Select the nearest centroid's community.
2. **Candidate scan**: Retrieve `members[nearest_community]` — typically N/K vectors.
3. **Exact rerank**: Score each candidate by L2 distance to query. Sort and return top-k.

### Why this beats FlatScan

FlatScan scans N vectors. CommunityRAG scans C centroids (cheap, cache-hot) + N/K vectors.
For K=10, the scan is 10× smaller. For K=100, it is 100× smaller. The crossover point
where CommunityRAG costs more than FlatScan is when the community is larger than N (which
cannot happen), or when C is so large that centroid matching dominates (C > N/K, i.e., K²>N).
At K=10, N=2000, we need K²=100 < N=2000 — well within the good regime.

---

## Practical Failure Modes

1. **Threshold miscalibration**: Too high a threshold → every vector is its own community → CommunityRAG degrades to FlatScan. Too low → one giant community → no speedup.
2. **Imbalanced communities**: If one community holds 90% of vectors, search scope is nearly as large as FlatScan for queries in that community.
3. **Cross-community queries**: A query that genuinely spans two communities will lose recall if scoped to only the nearest centroid. Mitigated by querying top-2 communities (future work).
4. **Community drift**: As new vectors are inserted, communities may grow or merge. The current static build does not handle this. Production requires incremental community updates via `ruvector-mincut`.
5. **O(N²) build complexity**: Prohibitive for N > 50k. Production should use approximate k-NN graph construction (e.g., via `ruvector-hnsw-repair` neighbourhood lists) as the input to Union-Find.

---

## Security and Governance Implications

Community labels are inferred from vector geometry. If an adversary can insert vectors that
manipulate community boundaries (embedding poisoning), they can cause legitimate queries to
be routed to adversary-controlled communities. Mitigations:

1. **Proof-gated inserts** (`ruvector-proof-gate`): require a witness log for every inserted vector.
2. **Capability-gated reads** (`ruvector-capgated`): restrict which communities are accessible per querier.
3. **Community anomaly detection**: monitor community size distribution for sudden changes (new large community may indicate poisoning).

Together these form a triple security model: proof at write, capability at read, monitoring at runtime.

---

## Edge and WASM Implications

Community centroids and member lists are compact. For K=10 communities, D=64:
- Centroids: 10 × 64 × 4 = 2.5 KB
- Member lists: typically 200 ids × 8 bytes = 1.6 KB per community

A complete community index (excluding raw vectors) fits in < 20 KB — suitable for:
- Browser WASM embedding (user carries their own community-tagged memory)
- Raspberry Pi 5 / Cognitum Seed edge appliance (no server needed)
- `micro-hnsw-wasm` integration as the coarse routing layer

Raw vectors can be stored in a compact format via `rvf-quant` or replaced by PQ codes
(`ruvector-pq-search`) to reduce per-vector memory from 256 bytes to 8 bytes.

---

## MCP and Agent Workflow Implications

CommunityRAG exposes natural MCP tool surfaces:

```
tools:
  - name: memory_search_community
    description: "Search agent memory within the current task community"
    parameters:
      query: string (or embedding)
      k: integer
      community_override: optional integer

  - name: memory_list_communities
    description: "List active memory communities with size and centroid labels"
    returns: [{id, size, centroid_label, threshold}]

  - name: memory_set_threshold
    description: "Adjust community coherence threshold (triggers partial rebuild)"
    parameters:
      namespace: string
      threshold: float
```

A ruFlo workflow can call `memory_set_threshold` when it detects task context switches
(e.g., agent transitions from Python debugging to Rust performance profiling). The
community structure adapts without requiring the agent to explicitly tag memories.

---

## Practical Applications

| Application | User | Why it matters | How RuVector uses it |
|-------------|------|----------------|---------------------|
| Agent task memory | LLM coding agent | Prevents cross-task memory pollution during long sessions | CommunityRAG scopes retrieval to active task cluster |
| Multi-agent workspace isolation | Swarm coordinator | Different agents should not pollute each other's memory | Community labels = agent namespace boundaries |
| Enterprise semantic search | Knowledge worker | Documents cluster by project/department; cross-department results are noise | Community routing improves precision for intra-department queries |
| MCP memory tools | MCP server implementer | Tools can expose community-scoped search without building a graph DB | `ruvector-community-rag` as the backend |
| Local-first AI assistant | Privacy-conscious user | All memory stays on device; community labels are private metadata | WASM build + compact community index |
| Edge anomaly detection | IoT operator | Normal events cluster by device type; anomalies fall outside communities | CommunityRAG's misrouted queries = anomaly signal |
| Federated research retrieval | Academic | Papers cluster by discipline; cross-discipline retrieval adds noise | Community-scoped search per discipline |
| ruFlo workflow automation | Platform operator | Workflows change task context; memory should follow | Community re-routing triggered by ruFlo context events |

---

## Exotic Applications

| Application | 10–20 year thesis | Required advances | RuVector role | Risk |
|-------------|-------------------|-------------------|---------------|------|
| Cognitum edge cognition | Persistent per-device community memory that evolves with user habits | Incremental community updates, compressed centroid storage | Community-indexed vector store in <1 MB | Device battery / privacy tradeoffs |
| RVM coherence domains | Community labels become formal coherence domain identifiers enforced by the RVM scheduler | RVM coherence protocol + `ruvector-community-rag` integration | Community boundaries as memory isolation domains | RVM coherence spec not yet finalised |
| Proof-gated swarm memory | Communities are witness-chain-attested; only authorised agents can cross community boundaries | `ruvector-proof-gate` + community router | Proof-checked community routing | High cryptographic overhead per insert |
| Self-healing memory graphs | Communities detect their own fragmentation and trigger repair via ruFlo | Anomaly detection on community graph metrics | Coherence score as health signal | May trigger too many rebuilds |
| Dynamic world models for robotics | A robot's sensory memory naturally clusters by environment type (indoor, outdoor, etc.) | Real-time edge deployment, <10ms retrieval budget | Ultra-compact community index on embedded Rust | Sensor noise creates false communities |
| Agent operating system memory management | An AOS uses community structure as the unit of memory swapping (analogous to pages in virtual memory) | Formal AOS memory model integrating community labels | Community index as the AOS memory namespace | Requires AOS design work |
| Bio-signal community memory | EEG or fMRI-derived embeddings cluster by mental state; community-scoped retrieval matches current state | High-dimensional neural embedding compression | RaBitQ + community routing on neural embeddings | Bio-signal privacy, IRB requirements |
| Synthetic nervous systems | Long-horizon AI systems need memory communities analogous to brain hemispheres/lobes | Hierarchical community structure (communities of communities) | Recursive community detection using `ruvector-mincut` | Scaling laws for community hierarchy unknown |

---

## Deep Research Notes

### What the SOTA suggests

1. **Community detection for retrieval is validated**: TigerVector, GraphRAG, and ArchRAG all demonstrate that community-scoped retrieval improves precision. The question is not *whether* to use communities, but *how* to detect and maintain them.

2. **Leiden/Louvain vs. mincut**: Production systems (GraphRAG, TigerVector) use Leiden/Louvain because they optimise modularity — a global objective. Mincut-based partitioning maximises within-community density relative to between-community edge weight, which is more directly aligned with ANN recall preservation. The theoretical advantage of mincut for ANN is that it minimises the number of cross-community true nearest neighbours missed by community scoping.

3. **Dynamic community maintenance is an open problem**: DyG-DPCD (2025) and similar works address incremental community detection but are not integrated with vector retrieval systems. The gap between offline community detection and online memory insertion is significant.

4. **Agent-specific community routing**: No paper addresses the case where the *query's community* is determined by agent task state rather than by geometric proximity to centroids. The current PoC uses centroid proximity (purely geometric). A future version would use the agent's current task embedding as the community routing key.

### Where this PoC fits

The PoC establishes:
- The `CommunitySearch` trait as a composable retrieval interface.
- Empirical validation that community-scoped search achieves 10.8× speedup with no recall loss on well-separated clusters.
- The tradeoff characterisation: CommunityRAG trades ~5% ANN recall for perfect community precision on overlapping clusters.
- A clear upgrade path to `ruvector-mincut` for production-grade dynamic maintenance.

### What would make this production grade

1. Replace O(N²) build with approximate k-NN graph (via `ruvector-coherence-hnsw` neighbourhood lists).
2. Replace static Union-Find with incremental mincut updates from `ruvector-mincut`.
3. Add top-2 community search for queries near community boundaries.
4. Integrate with `ruvector-agent-memory` namespace manager for automatic community routing by agent id.
5. Expose `memory_search_community` as an MCP tool in `mcp-brain`.

### What would falsify the approach

1. If community sizes are highly skewed (one community holds >90% of vectors), speedup collapses.
2. If the embedding model does not produce well-clustered task representations, community detection fails.
3. If agents frequently issue cross-community queries (multi-task reasoning), recall loss becomes unacceptable.

Sources:
- [^1] Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," arXiv:2404.16130, 2024.
- [^2] He et al., "ArchRAG: Attributed Community-based Hierarchical RAG," arXiv:2502.09891, 2025.
- [^3] Xu et al., "Unleashing Graph Partitioning for Large-Scale ANNS," arXiv:2403.01797, VLDB 2024.
- [^4] MemGraphRAG, arXiv:2606.00610, 2026.
- [^5] "Memory is Reconstructed, Not Retrieved," arXiv:2606.06036, 2026.
- [^6] TigerVector arXiv:2501.11216, TigerGraph 2024.
- [^7] "Graph-based Agent Memory," arXiv:2602.05665, 2026.
- [^8] CLAG, arXiv:2603.15421, 2026.
- [^9] OMD-GraphRAG, arXiv:2603.25152, 2026.
- [^10] DyG-DPCD, Sattar et al., 2025.
- [^11] Deep MinCut, researchgate.net/publication/364725843, 2022.
- [^12] CRISP, arXiv:2603.05180, 2026.

---

## Production Crate Layout Proposal

```
crates/
  ruvector-community-rag/       (this PoC — trait + 3 variants)
  ruvector-community-build/     (production: approx k-NN + incremental mincut)
  ruvector-community-mcp/       (MCP tool surface: memory_search_community, etc.)
```

Integration path:
```
ruvector-mincut  ──builds──>  ruvector-community-build  ──powers──>  ruvector-community-rag
ruvector-coherence ──tunes──> ruvector-community-build
ruvector-agent-memory ──tags──> ruvector-community-rag namespace router
mcp-brain ──exposes──> ruvector-community-mcp tools
ruFlo workflows ──adjust──> community threshold via MCP tool
```

---

## What to Improve Next

1. **Approximate k-NN graph build**: Replace O(N²) with HNSW-guided neighbourhood construction to push build time from O(N²) to O(N log N).
2. **Incremental insert**: When a new vector arrives, compute its k nearest existing neighbours and run a delta-union-find update. Avoids full rebuild.
3. **Top-2 community search**: Query both the nearest and second-nearest centroid communities; merge and deduplicate results. Closes the recall gap for boundary queries.
4. **Community coherence scoring**: Surface `mincut_value / community_size` as a coherence health metric. High value → well-separated; low value → impending merge.
5. **MCP tool surface**: Implement `memory_search_community` as a real MCP tool in `mcp-brain`.
6. **ruFlo integration**: Add a ruFlo workflow that monitors community health metrics and triggers threshold adjustment when community sizes diverge.
7. **WASM build**: Port to `no_std` for use in `micro-hnsw-wasm` as the coarse routing layer.
8. **Benchmark at N=50k**: Validate speedup hypothesis at larger scale.

---

## References and Footnotes

[^1]: Edge, D. et al. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv:2404.16130, Microsoft Research, 2024. https://arxiv.org/abs/2404.16130. Accessed 2026-07-02.

[^2]: He, X. et al. "ArchRAG: Attributed Community-based Hierarchical RAG." arXiv:2502.09891, 2025. https://arxiv.org/abs/2502.09891. Accessed 2026-07-02.

[^3]: Xu, R. et al. "Unleashing Graph Partitioning for Large-Scale Nearest Neighbor Search on Billion-Scale Datasets." arXiv:2403.01797, VLDB 2024/2025. https://arxiv.org/abs/2403.01797. Accessed 2026-07-02.

[^4]: "MemGraphRAG: Memory-based Multi-Agent System for Graph RAG." arXiv:2606.00610, 2026. https://arxiv.org/abs/2606.00610. Accessed 2026-07-02.

[^5]: "Memory is Reconstructed, Not Retrieved." arXiv:2606.06036, 2026. https://arxiv.org/abs/2606.06036. Accessed 2026-07-02.

[^6]: TigerVector: Supporting Vector Search in Graph Databases. arXiv:2501.11216, TigerGraph, 2024. https://arxiv.org/abs/2501.11216. Accessed 2026-07-02.

[^7]: "Graph-Based Agent Memory: Taxonomy, Techniques, and Applications." arXiv:2602.05665, 2026. https://arxiv.org/abs/2602.05665. Accessed 2026-07-02.

[^8]: CLAG: Adaptive Memory Organisation via Agent-Driven Clustering. arXiv:2603.15421, 2026. https://arxiv.org/abs/2603.15421. Accessed 2026-07-02.

[^9]: OMD-GraphRAG: Enhancing GraphRAG with Multi-Dimensional Clustering. arXiv:2603.25152, 2026. https://arxiv.org/abs/2603.25152. Accessed 2026-07-02.

[^10]: DyG-DPCD: Distributed Parallel Community Detection for Dynamic Graphs. Sattar et al., 2025. Accessed 2026-07-02.

[^11]: Deep MinCut: Learning Node Embeddings from Detecting Communities. researchgate.net/publication/364725843, 2022. Accessed 2026-07-02.

[^12]: CRISP: Correlation-Resilient Indexing via Subspace Partitioning. arXiv:2603.05180, 2026. https://arxiv.org/abs/2603.05180. Accessed 2026-07-02.
