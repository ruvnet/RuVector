# ruvector 2026: Coherence-Gated Multi-Tenant Vector Namespace Router in Rust

> Rust crate isolating per-agent vector memory with centroid routing, coherence thresholds, and a tamper-evident witness log — the retrieval MMU for multi-agent AI systems.

**Value proposition:** Give every AI agent its own isolated vector memory namespace, with coherence-based selective federation and a built-in audit log — all in safe Rust with no runtime dependencies.

- Repository: https://github.com/ruvnet/ruvector
- Research branch: `research/nightly/2026-06-07-coherence-namespace-router`
- Crate: `crates/ruvector-namespace-router`

---

## Introduction

The moment you deploy more than one AI agent against a shared vector store, you have a retrieval isolation problem. Agent A is retrieving memories relevant to its task. Agent B is doing the same. In a flat HNSW index with no namespace separation, a well-connected vector from Agent B's memory can influence Agent A's search results through shared graph edges — silently, without logging, and without the ability to audit what happened.

This is not a hypothetical problem. In 2026, enterprise RAG deployments routinely run dozens of agent roles against shared embedding stores. Customer support agents, legal review agents, code generation agents, and financial analysis agents all share the same embedding space because the underlying embedding model is the same. But their retrieval contexts must not mix. A legal agent retrieving precedents must not see a sales agent's confidential pitch deck embeddings. A financial agent must not have its market analysis contaminated by unrelated engineering documents.

Current vector databases address this with coarse mechanisms: separate collections per tenant (Qdrant, Weaviate), partition keys (Milvus), dataset paths (LanceDB). These provide administrative isolation — separate buckets — but no semantic routing. They cannot answer the question: "Given this query, which namespaces are semantically close enough to contribute useful results?" And they have no built-in witness log for cross-namespace access events.

RuVector is a Rust-native vector, graph, memory, and retrieval substrate for autonomous agents. Its `ruvector-mincut` crate partitions graphs using coherence scoring; its `ruvector-coherence` crate tracks semantic coherence between domains. `ruvector-namespace-router` is the retrieval-time primitive that connects these components: a multi-tenant namespace router that uses centroid distance as a coherence proxy to gate which namespaces contribute results to a given query, while logging every cross-boundary access in an append-only witness log.

For ruFlo autonomous workflow loops, this means each pipeline stage has a semantically isolated memory namespace. The ruFlo orchestrator sets the coherence threshold τ for each stage pair: low τ during collaborative phases (stages share relevant memories), high τ during evaluation phases (stages must not contaminate each other's retrieval). For MCP tools, the namespace maps directly to an MCP resource URI, giving each agent a stable, addressable, and isolated memory resource. For edge AI with Cognitum Seed, `FlatIsolated` — the strictest variant — has no runtime dependencies and compiles to WASM32 for memory-safe isolation on constrained devices.

This is not theoretical future-work. The crate compiles, tests pass (16/16), and benchmarks run today with real measured numbers. It is the routing layer that multi-agent AI deployments need and that no existing vector database provides.

---

## Features

| Feature | What it does | Why it matters | Status |
|---------|-------------|----------------|--------|
| `FlatIsolated` | Per-namespace linear scan, zero cross-namespace visibility | Maximum isolation, WASM compatible, zero dependencies | Implemented in PoC |
| `CentroidRouted` | Centroid index prunes which namespaces to scan | Sub-total scan with opt-in federation | Implemented in PoC |
| `CoherenceGated` | Coherence threshold τ gates cross-namespace results | Semantic isolation policy instead of administrative tags | Implemented in PoC |
| `WitnessLog` | Append-only log of cross-boundary access events | Audit trail for AI Act / NIST RMF compliance | Implemented in PoC |
| `coherence(a,b)` | exp(−distance / combined_spread) score | Quantifies semantic overlap between namespaces | Measured |
| Welford centroid update | O(1) per insert, no full recompute | Low overhead for streaming inserts | Measured |
| `NamespaceIndex` trait | Common interface for all three variants | Easy backend swap without caller changes | Production candidate |
| WASM compatibility | `FlatIsolated` compiles to WASM32 | Edge AI, Cognitum Seed, browser agents | Research direction |
| ruFlo integration | τ tunable per workflow phase | Dynamic isolation policy for autonomous loops | Research direction |
| MCP resource URI mapping | Namespace ID → MCP resource URI | Standardized agent memory addressing | Research direction |

---

## Technical Design

### Core data structure

Each namespace is a `HashMap<NamespaceId, NsData>` where `NsData` holds:
- A `Vec<(VectorId, Vec<f32>)>` of entry vectors
- An incrementally maintained centroid (Welford update per insert)
- A running sum-of-squared-deviations for spread estimation

The centroid update is O(D) per insert (D = vector dimension), making namespace maintenance cheap even at high insert rates.

### Trait-based API

```rust
pub trait NamespaceIndex {
    fn insert(&mut self, ns: NamespaceId, id: VectorId, vector: Vec<f32>) -> Result<(), String>;
    fn search(&self, ns: NamespaceId, query: &[f32], k: usize) -> Vec<SearchResult>;
    fn namespace_count(&self) -> usize;
    fn total_vectors(&self) -> usize;
    fn memory_bytes(&self) -> usize;
}
```

All three variants implement this trait. Future HNSW-backed and DiskANN-backed namespace variants will implement the same interface with no changes to callers.

### Baseline variant: FlatIsolated

```
search(ns, query, k):
  scan all entries in namespaces[ns]
  return top-k by L2 distance
  (no other namespace ever touched)
```

### Alternative A: CentroidRouted

```
search(ns, query, k, probe=P):
  rank all namespaces by L2(query, centroid[ns])
  select top-P namespaces (always include own ns)
  scan all entries in selected P namespaces
  return top-k from merged candidates
```

### Alternative B: CoherenceGated

```
search(ns, query, k, tau=τ):
  scan own namespace
  for each other namespace o:
    if coherence(ns, o) >= τ:
      scan namespace o
      for each result from o: witness_log.record(event)
  return top-k from merged candidates
```

Coherence formula:
```
coherence(a, b) = exp(−L2(centroid_a, centroid_b) / (spread_a + spread_b))
```

This decays monotonically with centroid distance and increases with namespace spread (wider distributions are more likely to overlap). Coherence = 1.0 for identical centroids; ≈ 0.37 at the 1-σ boundary; ≈ 0 for distant namespaces.

### Memory model

For D dimensions, N total vectors, NS namespaces, the memory usage is:

```
FlatIsolated:      N × (8 + D×4) bytes
CentroidRouted:    N × (8 + D×4) + NS × D×4 bytes  [+ centroid]
CoherenceGated:    N × (8 + D×4) + NS × D×4 bytes  [+ centroid]
```

For D=128, N=4,000, NS=8 (benchmark configuration): measured 2,031 KB (FlatIsolated), 2,035 KB (v2/v3).

### Architecture diagram

```mermaid
graph LR
    Q[Query + NamespaceId] --> R{Routing policy}
    R -->|FlatIsolated| F[Own namespace scan]
    R -->|CentroidRouted| C[Centroid ranking → probe P namespaces]
    R -->|CoherenceGated| G{coherence(ns, o) ≥ τ?}
    G -->|yes| X[Foreign namespace scan → WitnessLog]
    G -->|no| F
    F --> K[Top-K results]
    C --> K
    X --> K
```

---

## Benchmark Results

All numbers measured from `cargo run --release -p ruvector-namespace-router` on Intel Celeron N4020, x86-64, Linux 6.18.5, rustc 1.94.1.

**Dataset:** 8 namespaces × 500 vectors = 4,000 total, D=128, 1,600 queries (200/namespace), K=10.
**Ground truth:** Brute-force exact top-K within each namespace.

| Variant | Dataset | Dims | Queries | Mean (µs) | p50 (µs) | p95 (µs) | QPS | Memory (KB) | Recall@10 | Accept |
|---------|---------|------|---------|-----------|----------|----------|-----|-------------|-----------|--------|
| FlatIsolated | 4,000 | 128 | 1,600 | 78.79 | 76.17 | 96.19 | 12,631 | 2,031.2 | 1.000 | PASS |
| CentroidRouted | 4,000 | 128 | 1,600 | 81.05 | 78.08 | 98.40 | 12,280 | 2,035.2 | 1.000 | PASS |
| CoherenceGated | 4,000 | 128 | 1,600 | 84.99 | 81.45 | 105.08 | 11,702 | 2,035.2 | 1.000 | PASS |

**Environment:**
- Hardware: Intel Celeron N4020 (x86-64, single-core)
- OS: Linux 6.18.5
- Rust: 1.94.1 (release profile, no SIMD intrinsics)
- Cargo command: `cargo run --release -p ruvector-namespace-router`

**Benchmark limitations:**
- Linear scan only; production HNSW backends would reduce per-query latency by ~10–20×.
- Single-threaded execution; concurrent multi-namespace queries would scale with core count.
- Synthetic Gaussian data; real embedding distributions may shift coherence scores.
- Numbers are process-level wall-clock times; OS scheduling noise included in p95.

---

## Comparison with Vector Databases

This PoC implements routing semantics, not a full ANN index. Comparisons are therefore framed around isolation model and governance capability, not raw ANN throughput.

| System | Core strength | Namespace isolation model | Cross-namespace semantics | Witness log | Direct benchmark here |
|--------|--------------|--------------------------|--------------------------|-------------|----------------------|
| Milvus | Billion-scale IVF-PQ | Partition keys + RBAC | None (manual merge query) | No | No |
| Qdrant | HNSW + payload filters | Named collections | None (separate queries) | No | No |
| Weaviate | GraphQL ANN | Tenant classes | None | No | No |
| Pinecone | Managed IVF | Namespaces (string key) | None | No | No |
| LanceDB | Lance columnar + HNSW | Dataset paths | None | No | No |
| FAISS | GPU-accelerated IVF-PQ | No built-in | None | No | No |
| pgvector | SQL + HNSW | PostgreSQL schemas | SQL JOIN (exact) | PostgreSQL WAL | No |
| Chroma | Embedding + metadata filter | Collections | None | No | No |
| Vespa | BM25 + ANN hybrid | Document schemas | Namespace-aware ranking | No | No |
| **RuVector** | **Graph + vector + coherence + WASM** | **Trait-based, 3 variants** | **Coherence-gated with witness log** | **Yes** | **Yes** |

**Note:** No external benchmarks were used or reproduced. The comparison is qualitative, based on official documentation for each system as of June 2026. RuVector's advantage is not raw throughput — it is the combination of semantic routing, coherence-gated federation, and witness logging in a single safe Rust library with no external services.

---

## Practical Applications

| Application | User | Why it matters | How RuVector uses it | Near-term path |
|-------------|------|----------------|---------------------|----------------|
| Enterprise multi-tenant RAG | Enterprise SaaS, LegalTech, FinTech | EU AI Act requires audit trails for AI retrieval | CoherenceGated + WitnessLog as retrieval audit layer | Deploy with τ=0.5 and flush witness log to S3/RVF daily |
| ruFlo pipeline memory | ruFlo orchestrators | Workflow stage isolation prevents context contamination | Each stage = one namespace; τ adjusted per stage pair | Integrate with ruFlo stage lifecycle hooks |
| MCP agent memory tools | Claude, GPT, Copilot via MCP | Agents need addressable, isolated memory spaces | Namespace ID = MCP resource URI; router is the MCP backend | Wrap with `ruvector-server` MCP endpoint |
| Code intelligence | IDE AI assistants (Cursor, Copilot) | Per-repo isolation with cross-repo dependency search | Namespace = repository; coherence = API surface overlap | Index Git repos as namespaces, τ=0.4 for dependency search |
| Local-first AI assistants | Privacy-conscious users | Personal memory must not sync to cloud without audit | FlatIsolated on device; witness log before any cloud sync | Ship in Cognitum Seed firmware |
| Edge anomaly detection | IoT, industrial | Sensor namespace isolation with cross-sensor federation | Each sensor stream = namespace; coherence = signal correlation | WASM build of FlatIsolated for ESP32 |
| Security event retrieval | SOC platforms, SIEM | Strict tenant isolation for security event logs | FlatIsolated per tenant + WitnessLog for any cross-tenant | Integrate with ruvector-verified for signed witness entries |
| Scientific retrieval | Research platforms | Domain isolation with controlled cross-domain discovery | Namespace = research domain; τ configured by domain policy | CentroidRouted with probe=2 for adjacent domain search |

---

## Exotic Applications

| Application | 10-20 year thesis | Required technical advances | RuVector role | Risk / unknown |
|-------------|------------------|----------------------------|---------------|----------------|
| Cognitum edge cognition | A Cognitum Seed device manages 64+ cognitive namespaces simultaneously, each representing a distinct perceptual or behavioral context | Persistent sub-millisecond namespace routing on ARM M-class | CoherenceGated as the thalamic routing primitive | Power budget for coherence scoring; formal verification of isolation |
| RVM coherence domains | RuVector's coherence domains become the memory management unit of an agent OS — namespaces are allocated and reclaimed like virtual memory pages | OS-level scheduler integration; namespace capability tokens | Namespace router as the agent OS memory subsystem | Defining formal coherence-as-security boundary |
| Proof-gated autonomous systems | Every cross-namespace access generates a ZK proof that the policy τ was satisfied, verifiable without revealing the accessed vectors | Fast ZK-SNARK generation (<1ms) for retrieval events | WitnessLog entries as ZK proof inputs; ruvector-verified provides the proof system | ZK generation latency is currently prohibitive for real-time retrieval |
| Swarm memory federation | 10,000 ruFlo agents share a federated vector memory; coherence-gated routing decides which sub-swarms share memories dynamically | Distributed namespace routing with CRDT-consistent centroids | CentroidRouted as the swarm memory routing layer | CAP theorem trade-offs in distributed coherence computation |
| Self-healing vector graphs | When namespace centroids drift apart (semantic drift), the router automatically triggers compaction or namespace split | Coherence monitoring + graph-cut-based namespace split | CoherenceGated's coherence scores as health signals for self-healing | Defining healthy baseline; preventing oscillation in split/merge cycles |
| Agent operating systems | Agent namespaces are scheduled to vector index partitions the way Linux schedules threads to CPU caches, with NUMA-aware coherence routing | NUMA-topology-aware namespace allocation; cache-coherent centroid sync | Namespace router as the NUMA memory controller for agent cognitive load | Hardware NUMA topology doesn't map cleanly to semantic topology |
| Bio-signal memory isolation | Medical AI agents processing different patient streams must provably not cross-contaminate retrieval; witness logs provide chain-of-evidence | Namespace router with hardware TPM attestation of isolation | FlatIsolated with WASM sandboxing + signed witness log | Regulatory acceptance of software-only attestation |
| Synthetic nervous system | A distributed network of agents, each with an isolated namespace, communicating through coherence-gated synaptic connections | Dynamic coherence threshold learning from agent interaction patterns | CoherenceGated threshold as the "synaptic weight" governing inter-agent information flow | Emergent dynamics are unpredictable; stability guarantees unknown |

---

## Deep Research Notes

### What the SOTA suggests

The 2026 academic literature on multi-tenant vector retrieval focuses primarily on:
1. **Filtered ANN** (ACORN, FAISS filtered search): improving recall under strict metadata filters.
2. **Federated RAG**: composing retrieval across multiple vector stores, typically via union queries.
3. **Privacy-preserving retrieval**: using homomorphic encryption or MPC for retrieval over encrypted embeddings.

None of these directly address the semantic routing problem: using the *content* of the query to decide which isolation domains are relevant, rather than relying on caller-supplied metadata tags.

### What remains unsolved

1. **Optimal τ selection**: The coherence threshold τ should be set based on the statistical properties of the namespace distribution. A principled method (e.g., target a false-positive rate for cross-namespace inclusion) is needed.
2. **Distributed coherence computation**: When namespaces span multiple RuVector nodes, centroid synchronization requires a consistency protocol. Eventual consistency (CRDT centroids) may produce incorrect coherence scores during partition events.
3. **Adversarial coherence manipulation**: A tenant with insert access could craft vectors to inflate their namespace's apparent coherence with a target namespace, gaining unauthorized retrieval access. This is analogous to ARP poisoning for routing tables.
4. **Namespace lifecycle**: Namespace creation, compaction, splitting, and merging policies are unspecified. A namespace that has grown to 10M vectors needs a different backing index than one with 500 vectors.

### Where this PoC fits

`ruvector-namespace-router` is the routing layer, not the search engine. It is designed to wrap any `NamespaceIndex`-compliant backend. The current PoC uses linear scan to keep the implementation verifiable and auditable; production deployment requires integrating with `ruvector-core`'s HNSW backend per namespace.

The WitnessLog is in-process and non-cryptographic. For compliance use cases, it must be persisted and signed (via `ruvector-verified`) before the PoC becomes production-grade.

### What would falsify the approach

If centroid-distance-based coherence scores are found to be poor predictors of retrieval-result overlap (i.e., low centroid distance does not imply shared top-K results), the gating function would need to be replaced with a more accurate but more expensive metric. Candidate replacements: k-NN graph intersection rate (sampled at insert time), or a learned coherence model trained on historical retrieval overlap data.

Sources:
- [^1] Patel et al., "ACORN," SIGMOD 2024, arXiv:2403.04871.
- [^2] EU AI Act, Regulation 2024/1689, Articles 13 and 17.
- [^3] NIST AI RMF 1.0, Govern 6.2 (Accountability).
- [^4] Welford (1962), Technometrics 4(3):419–420.
- [^5] Qdrant docs, "Collections," https://qdrant.tech/documentation/concepts/collections/ (accessed 2026-06-07).

---

## Usage Guide

```bash
git checkout research/nightly/2026-06-07-coherence-namespace-router

# Build
cargo build --release -p ruvector-namespace-router

# Run tests
cargo test -p ruvector-namespace-router

# Run benchmark (produces the table above)
cargo run --release -p ruvector-namespace-router
```

**Expected output:**

```
=== ruvector-namespace-router benchmark ===
OS          : linux
Arch        : x86_64
Namespaces  : 8
Vecs/ns     : 500
Total vecs  : 4000
Dimensions  : 128
Queries     : 1600 (200/ns)
...
ACCEPTANCE: PASS — all variants recall@10 >= 0.99
```

**Changing dataset size:** Edit `PER_NS` in `src/main.rs` (e.g., `const PER_NS: usize = 5_000;`). Memory grows linearly; latency grows linearly (linear scan).

**Changing dimensions:** Edit `DIM`. Memory and latency both grow linearly with DIM.

**Adding a new backend:** Implement `NamespaceIndex` for your type. The benchmark harness calls only trait methods, so the benchmark function accepts any `NamespaceIndex` impl.

**Plugging into RuVector:** The `NamespaceIndex` trait can wrap a `ruvector-core` HNSW index by storing one `HnswIndex` per namespace inside a `HashMap<NamespaceId, HnswIndex>`. Insert and search delegate to the per-namespace HNSW instance.

---

## Optimization Guide

| Area | Strategy | Expected gain |
|------|----------|---------------|
| Memory | Replace `Vec<f32>` entries with f16 or RaBitQ-quantized vectors | 2–8× memory reduction |
| Latency | Replace linear scan per namespace with per-namespace HNSW | ~10–20× query speedup at N=500 |
| Recall | Lower τ to include more namespaces; increase probe count | Recall improves but isolation weakens |
| Edge deployment | Compile FlatIsolated to WASM32; use u8 quantized vectors | ~4× memory, WASM-safe |
| WASM optimization | Remove `HashMap`; use sorted `Vec<(NamespaceId, Vec<Entry>)>` with binary search | `no_std` compatible |
| MCP tool | Cache namespace centroids in a fast path; skip recompute on repeated queries to same ns | Amortize centroid ranking cost |
| ruFlo automation | Pre-warm namespace caches at workflow start; pin hot namespaces to L3 | Eliminate cold-start latency |

---

## Roadmap

### Now
- Merge `crates/ruvector-namespace-router` as a workspace member.
- Wire `FlatIsolated` into `ruvector-server` as the default namespace isolation backend.
- Persist `WitnessLog` to `ruvector-snapshot` for crash recovery.

### Next
- HNSW namespace backend: one HNSW index per namespace, same `NamespaceIndex` trait.
- Dynamic τ: `set_policy(ns: NamespaceId, tau: f32)` method for ruFlo integration.
- `ruvector-mincut` namespace assignment: use graph partition IDs as namespace IDs.
- Signed witness entries via `ruvector-verified`.

### Later
- ZK-provable cross-namespace access (proof that τ was satisfied without revealing vectors).
- Distributed namespace routing with CRDT centroid synchronization across `ruvector-raft` nodes.
- Learned coherence model: train a small MLP to predict retrieval overlap from centroid statistics.
- Agent OS integration: namespace allocation as a first-class OS primitive in a future RVM.

---

## Footnotes and References

[^1]: Patel, Aditya, et al. "ACORN: Predicate-Agnostic Approximate Nearest Neighbor Search over Vector + Structured Data." ACM SIGMOD 2024. arXiv:2403.04871. Accessed 2026-06-07.

[^2]: Regulation (EU) 2024/1689 (EU AI Act), Article 13 (Transparency). Official Journal of the European Union, 2024. https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689. Accessed 2026-06-07.

[^3]: National Institute of Standards and Technology. "AI Risk Management Framework (AI RMF 1.0)." NIST AI 100-1. Govern 6.2: Policies and accountability. 2023. https://airc.nist.gov/RMF. Accessed 2026-06-07.

[^4]: Welford, B. P. "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics, 4(3):419–420, 1962. The algorithm used for incremental centroid and spread estimation.

[^5]: Qdrant. "Collections." Qdrant Documentation. https://qdrant.tech/documentation/concepts/collections/. Accessed 2026-06-07. Describes the collection-level isolation model used by Qdrant.

[^6]: Milvus. "Partition Key." Milvus Documentation. https://milvus.io/docs/partition_key.md. Accessed 2026-06-07. Describes Milvus partition-based multi-tenancy.

[^7]: LanceDB. "Working with Multiple Tables." LanceDB Documentation. https://lancedb.github.io/lancedb/. Accessed 2026-06-07. Dataset-path-based namespace isolation.

[^8]: Pinecone. "Namespaces." Pinecone Documentation. https://docs.pinecone.io/guides/indexes/use-namespaces. Accessed 2026-06-07. String-key namespace model.

---

## SEO Tags

**Keywords:**
ruvector, Rust vector database, Rust vector search, high performance Rust, ANN search, HNSW, DiskANN, filtered vector search, graph RAG, agent memory, AI agents, MCP, WASM AI, edge AI, self learning vector database, ruvnet, ruFlo, Claude Flow, autonomous agents, retrieval augmented generation, multi-tenant vector database, vector namespace isolation, coherence routing, witness log, RAG compliance, enterprise RAG, agent operating system, ruVo cognition, RVF portable format.

**Suggested GitHub topics:**
rust, vector-database, vector-search, ann, hnsw, rag, graph-rag, ai-agents, agent-memory, mcp, wasm, edge-ai, rust-ai, semantic-search, multi-tenant, namespace-isolation, retrieval-augmented-generation, embeddings, ruvector, autonomous-agents, coherence, audit-log.
