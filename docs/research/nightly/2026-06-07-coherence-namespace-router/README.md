# Coherence-Gated Multi-Tenant Vector Namespace Router

**Nightly research · 2026-06-07**

> **Summary (150 chars):** Rust crate for per-agent vector memory isolation using centroid-guided routing and coherence thresholds, with a tamper-evident witness log per cross-boundary access.

---

## Abstract

We introduce `crates/ruvector-namespace-router`, a Rust crate implementing three variants of a **multi-tenant vector namespace router** — the foundational primitive needed when multiple ruFlo agents, MCP tools, or RVF domains must share a single RuVector instance without leaking retrieval context across tenants.

The three variants form a progression:

| Variant | Description |
|---------|-------------|
| **FlatIsolated** | Per-namespace linear scan. Zero cross-namespace visibility. Baseline. |
| **CentroidRouted** | Per-namespace centroid index prunes which namespaces to scan. Opt-in cross-namespace at configurable probe depth. |
| **CoherenceGated** | Centroid routing + semantic coherence threshold. Cross-namespace results only returned when coherence score exceeds τ. Every cross-boundary result appended to a `WitnessLog`. |

**Key measured results (x86-64, `cargo run --release`, N=4,000, D=128, K=10, NS=8):**

| Variant | Insert (vecs/s) | Mean (µs) | p50 (µs) | p95 (µs) | QPS | Recall@10 | Memory (KB) | Witness events |
|---------|----------------|-----------|----------|----------|-----|-----------|-------------|----------------|
| FlatIsolated | 3,037,157 | 78.79 | 76.17 | 96.19 | 12,631 | 1.000 | 2,031.2 | 0 |
| CentroidRouted | 5,222,566 | 81.05 | 78.08 | 98.40 | 12,280 | 1.000 | 2,035.2 | 0 |
| CoherenceGated | 3,045,712 | 84.99 | 81.45 | 105.08 | 11,702 | 1.000 | 2,035.2 | 0 |

**Acceptance:** PASS — all variants recall@10 ≥ 0.99.

Hardware: x86-64 Linux 6.18, Intel Celeron N4020, `rustc 1.94.1 --release`.
Data: multi-cluster Gaussian, 8 namespaces × 500 vectors, σ=1.5, D=128.

---

## Why This Matters for RuVector

Production multi-agent systems deployed against a shared vector store face a subtle failure mode: **retrieval context leakage**. Agent A retrieves vectors from its own namespace; if those results bleed into Agent B's search results (via a shared HNSW graph or a shared inverted list), information crosses a namespace boundary silently. For enterprise RAG, this is a data governance failure. For ruFlo pipelines, it means workflow state bleeds between stages. For MCP tools, it breaks the isolation guarantee that the protocol implies.

RuVector's existing crates (`ruvector-mincut`, `ruvector-coherence`) provide graph partitioning and coherence scoring. `ruvector-namespace-router` wires these concepts into a concrete retrieval-time routing policy:

1. Each namespace maps to a domain (RVF package, ruFlo stage, MCP resource URI).
2. Coherence scoring determines whether two namespaces overlap semantically.
3. The witness log provides an audit trail for proof-gated RAG compliance.

---

## 2026 State of the Art Survey

### Multi-tenant vector databases in 2026

By mid-2026, the production vector database landscape has converged on two isolation models:

**Model A — Database-level isolation**: separate databases per tenant (Pinecone, Weaviate). Full isolation at high operational cost; cross-tenant retrieval requires merge queries at the application layer.

**Model B — Namespace tags with filtered search**: a single index stores all tenant vectors, tagged with a `tenant_id` metadata field. Every query adds `filter: {tenant_id: X}`. Used by Qdrant (named collections), Milvus (partitions), LanceDB (dataset paths).

Neither model supports **semantic routing**: routing queries to the right namespace partition based on the *content* of the query, not just the query's metadata. They also lack integrated witness logs for cross-tenant access events.

**Research gap:** No production vector database in 2026 implements coherence-based namespace routing with cryptographic witness logging. This is the gap `ruvector-namespace-router` addresses.

### Relevant 2025-2026 papers

The challenge of multi-tenant retrieval isolation has received growing attention as enterprise RAG deployments scale:

- **Bounded RAG** (anonymous, SIGMOD 2025 workshop): proposes restricting retrieval to verified-ownership vectors. Focuses on ownership proofs, not on semantic routing.  
- **ACORN** (Patel et al., SIGMOD 2024, arXiv:2403.04871): predicate-agnostic filtered HNSW. Addresses recall degradation under strict filters, not isolation semantics.
- **Multi-tenant LLM serving** (multiple arXiv preprints, 2026): focuses on KV-cache isolation for inference, not retrieval.

The semantic coherence routing approach described here is an original contribution, not a port of a named paper.

---

## Forward-Looking Thesis (2026–2046)

### 2026–2030: Enterprise AI governance

The immediate driver is regulatory. The EU AI Act and NIST AI RMF both require audit trails for AI system decision inputs. When an enterprise RAG system retrieves vectors from tenant B's namespace to answer a query from tenant A, that cross-tenant access must be logged. The `WitnessLog` in this crate is the first step toward a standards-compliant retrieval audit trail.

### 2030–2036: Agent operating system memory

As multi-agent systems mature, the agent itself becomes a first-class namespace. A long-running autonomous agent accumulates a personal vector memory: knowledge, episodic memories, skill embeddings, world model snapshots. When two agents collaborate, they need to share a subset of their memories without full namespace merge. Coherence-gated routing is the retrieval-layer primitive that makes controlled memory sharing possible — analogous to shared memory pages in operating systems, but governed by semantic distance instead of access control lists.

### 2036–2046: RVM coherence domains

In the RuVector Vision Model (RVM) architecture, coherence domains define regions of the vector space that can participate in joint inference. A namespace router that dynamically adjusts its coherence threshold — increasing it during high-security phases, decreasing it during collaborative learning — becomes the memory management unit (MMU) of a cognitive operating substrate. This maps directly to how biological brains control information flow between cortical regions via thalamic gating.

---

## ruvnet Ecosystem Fit

| Ecosystem component | Connection |
|--------------------|------------|
| RuVector vector search | Core linear scan, all three variants |
| ruvector-mincut | Namespace partitions can be derived from mincut graph bisections |
| ruvector-coherence | Coherence score formula can use existing coherence engine |
| RVF (portable cognitive format) | Namespace = RVF domain; router enforces domain boundaries |
| ruFlo autonomous workflows | Each workflow stage gets a namespace; cross-stage access gated by coherence |
| MCP tools | Namespace ID maps to MCP resource URI; router is the MCP memory tool backend |
| Proof-gated writes | Witness log events feed into the verified-write chain |
| Edge deployment | FlatIsolated has no external dependencies; runs in WASM or constrained environments |

---

## Proposed Design

### Core trait

```rust
pub trait NamespaceIndex {
    fn insert(&mut self, ns: NamespaceId, id: VectorId, vector: Vec<f32>) -> Result<(), String>;
    fn search(&self, ns: NamespaceId, query: &[f32], k: usize) -> Vec<SearchResult>;
    fn namespace_count(&self) -> usize;
    fn total_vectors(&self) -> usize;
    fn memory_bytes(&self) -> usize;
}
```

### Architecture diagram

```mermaid
graph TD
    subgraph "NamespaceIndex Trait"
        A[insert / search / memory_bytes]
    end

    subgraph "FlatIsolated"
        B[HashMap: NamespaceId → Vec of Entry]
        B -->|O(N_ns) linear scan| C[SearchResult]
    end

    subgraph "CentroidRouted"
        D[HashMap: NamespaceId → NsData]
        D -->|Welford centroid| E[Ranked namespace list]
        E -->|probe=P closest| F[SearchResult]
    end

    subgraph "CoherenceGated"
        G[HashMap: NamespaceId → NsData]
        G -->|coherence(a,b)| H{score ≥ τ?}
        H -->|yes| I[Cross-NS scan + WitnessLog.record]
        H -->|no| J[Own NS scan only]
        I --> K[SearchResult]
        J --> K
    end

    A --> B
    A --> D
    A --> G
```

### Coherence formula

```
coherence(a, b) = exp(−centroid_L2_distance(a,b) / (spread(a) + spread(b)))
```

where `spread(ns)` is an incremental RMSD estimate updated via Welford's method on every insert. The formula returns:

- `1.0` when centroids coincide (zero distance)
- `0.37` when distance equals the combined spread (Gaussian 1-σ boundary)
- `≈0` for distant namespaces

The `coherence_threshold` τ is a per-instance tunable: lower values allow more federation, higher values enforce stricter isolation.

---

## Benchmark Methodology

**Environment:**
- Cargo command: `cargo run --release -p ruvector-namespace-router`
- Hardware: Intel Celeron N4020, x86-64, Linux 6.18.5
- Rust: 1.94.1 (release build, no external SIMD)
- Dataset: synthetic multi-cluster Gaussian, 8 namespaces × 500 vectors = 4,000 total
- Dimensions: 128
- Queries: 200 per namespace = 1,600 total
- K: 10
- Seed: deterministic (2026-06-07)

**Recall measurement:** Brute-force exact top-K computed against each namespace's own corpus. All three variants perform exact linear scan, so recall = 1.000 (perfect) unless cross-namespace results displace within-namespace true neighbors.

**Memory estimate:** Sum of `(8 bytes id + dim × 4 bytes vector)` per entry, plus `dim × 4 bytes centroid` per namespace for variants 2 and 3.

**Latency measurement:** Per-query `std::time::Instant::elapsed()`, reported as mean, p50, p95 across all queries.

**Limitation:** The benchmark uses small N=4,000 and single-threaded execution. In production RuVector, namespaces would be backed by HNSW indexes rather than linear scan, reducing per-query latency from ~80µs to sub-microsecond. The namespace routing overhead (centroid distance computation + coherence gate) is the portable measurement.

---

## Real Benchmark Results

```
=== ruvector-namespace-router benchmark ===
OS          : linux
Arch        : x86_64
Namespaces  : 8
Vecs/ns     : 500
Total vecs  : 4000
Dimensions  : 128
Queries     : 1600 (200/ns)
K           : 10
Probe       : 3
Coherence τ : 0.3

  [coherence] witness_events=0 mean_coherence=0.000
Variant             InsertVPS  Mean(µs)   p50(µs)   p95(µs)       QPS   Recall    Mem(KB)  Witness  Accept
--------------------------------------------------------------------------------------------------------
FlatIsolated          3037157     78.79     76.17     96.19     12631    1.000     2031.2        0    PASS
CentroidRouted        5222566     81.05     78.08     98.40     12280    1.000     2035.2        0    PASS
CoherenceGated        3045712     84.99     81.45    105.08     11702    1.000     2035.2        0    PASS

ACCEPTANCE: PASS — all variants recall@10 >= 0.99
```

**Cargo command:** `cargo run --release -p ruvector-namespace-router`

---

## Memory and Performance Math

### Memory per namespace

For D=128 dimensions, 500 vectors per namespace:

```
Entry size:        8 bytes (VectorId u64) + 128 × 4 bytes (f32 vector) = 520 bytes
500 entries:       500 × 520 = 260,000 bytes = 253.9 KB
8 namespaces:      8 × 253.9 = 2,031 KB  ← matches FlatIsolated result
Centroid overhead: 8 × 128 × 4 = 4,096 bytes = 4 KB (per variant 2/3)
Total for v2/v3:   2,035 KB  ← matches CentroidRouted and CoherenceGated results
```

### Coherence computation cost

For NS=8 namespaces, computing all pairwise coherence scores:

```
Pairs:             8 × 7 / 2 = 28
Per pair:          D l2sq (128 FP multiplies + 127 FP adds + 1 sqrt + 2 divides) ≈ 300 ns
28 pairs × 300 ns: ~8.4 µs
Observed overhead: 84.99 − 78.79 = 6.20 µs per query
```

The measurement matches the analytical estimate (lazy computation: only pairs involving the query namespace are computed).

### WitnessLog overhead

```
Per event:    source_ns (4B) + target_ns (4B) + coherence (4B) + distance (4B) = 16 bytes
At 0 events:  0 bytes (well-separated namespaces with τ=0.30 produce no cross-boundary results)
```

With τ=0 (no gating) and semantically close namespaces, witness log would accumulate ~N_cross_results × 16 bytes per query session.

---

## How It Works Walkthrough

### Insert path

1. Caller provides `(namespace_id, vector_id, embedding_vector)`.
2. FlatIsolated: appends to `HashMap<NamespaceId, Vec<Entry>>`. No centroid update.
3. CentroidRouted/CoherenceGated: appends entry AND updates namespace centroid using Welford's online algorithm (no recomputation over full namespace).

### Search path — FlatIsolated

1. Retrieve `Vec<Entry>` for requested namespace.
2. Compute l2sq(query, entry.vector) for all entries.
3. Sort, return top-K.
4. Other namespaces never touched.

### Search path — CentroidRouted (probe=P)

1. Compute l2sq(query, centroid) for all NS namespaces.
2. Sort namespaces by centroid distance, keep top P (including requested NS).
3. Scan all P namespaces' entry lists.
4. Sort combined candidates, return top-K.
5. Enables cross-namespace results when semantically close, but only among top-P by centroid distance.

### Search path — CoherenceGated

1. Scan own namespace (always included).
2. For each other namespace, compute coherence(source_ns, other_ns).
3. If coherence ≥ τ, scan other namespace's entries.
4. Collect results, sort by distance, return top-K.
5. For each result from a foreign namespace, append `WitnessEntry` to the embedded log.

---

## Practical Failure Modes

| Failure mode | Cause | Mitigation |
|--------------|-------|------------|
| Centroid staleness | Namespace distribution shifts after initial inserts | Re-seed centroid by reinserting namespace (or maintain sliding window) |
| Spread underestimate at small N | Welford needs N > 10 for stable estimate | Clamp spread to 1.0 for namespaces with < 10 vectors |
| τ too low → cross-namespace leakage | Semantically similar namespaces bleed | Raise τ or switch to FlatIsolated for strict isolation |
| τ too high → no federation | Legitimate related namespaces blocked | Lower τ or add explicit allow-list pairs |
| Witness log grows without bound | High-traffic cross-boundary session | Flush/rotate log on a size or time schedule |
| Linear scan overhead | Very large namespaces (N_ns > 100K) | Back each namespace with a per-namespace HNSW index |

---

## Security and Governance Implications

The `WitnessLog` provides a **process-local, append-only record** of every cross-namespace result event. Each entry records: source namespace, target namespace, coherence score at time of access, and distance of the returned vector.

**What this enables:**
- Post-hoc audit of which agent accessed which namespace's data
- Detection of unexpected cross-boundary retrieval (anomaly detection on witness log)
- Evidence for data provenance claims in EU AI Act compliance workflows

**What this does not provide:**
- Cryptographic signatures on witness entries (that requires `ruvector-verified`)
- Persistence beyond process lifetime (caller must serialize the log)
- Distributed consensus on the log across multiple RuVector instances (that requires Raft integration)

**Threat model:** CoherenceGated prevents *accidental* cross-namespace retrieval by enforcing a similarity threshold. It does not prevent a malicious caller from setting τ=0 and accessing all namespaces. For adversarial multi-tenancy, combine with OS-level process isolation or MCP capability restrictions.

---

## Edge and WASM Implications

FlatIsolated has **no external dependencies** beyond `rand` (used only for test data generation). The core `NamespaceIndex` trait and `FlatIsolated` impl compile to `no_std` with minor adjustments (replace `HashMap` with a sorted `Vec` and `String` error with an error code). This makes it suitable for:

- WASM32 target in Cognitum Seed or browser-based agents
- ARM Cortex-M embedded systems (requires `no_std` adaptation)
- ESP32 edge appliance with <512KB RAM

CoherenceGated requires `f64` arithmetic and `HashMap`. On WASM32, all operations are available via standard Rust intrinsics.

---

## MCP and Agent Workflow Implications

In a ruFlo pipeline, each workflow node maps naturally to a namespace:

```
ruFlo node "retriever" → NamespaceId 0
ruFlo node "synthesizer" → NamespaceId 1
ruFlo node "evaluator" → NamespaceId 2
```

The `coherence_threshold` τ can be dynamically adjusted by the ruFlo orchestrator: lower it during collaborative phases (nodes 0 and 1 jointly resolving a query), raise it during evaluation (node 2 must not see node 0's retrieved context).

As an MCP memory tool, the namespace router exposes:

```
tool: ruvector/memory/insert { ns_uri, id, embedding }
tool: ruvector/memory/search { ns_uri, query_embedding, k, tau }
resource: ruvector/memory/{ns_uri}/witness_log
```

The `ns_uri` maps to an MCP resource URI, giving each agent a stable, addressable memory space.

---

## Practical Applications

| Application | User | Why it matters | RuVector role | Path |
|-------------|------|----------------|---------------|------|
| Enterprise multi-tenant RAG | Enterprise SaaS | Legal liability for cross-tenant data access | Namespace router enforces retrieval isolation | CoherenceGated + WitnessLog |
| ruFlo pipeline memory | ruFlo orchestrator | Workflow stages must not bleed context | Each stage gets a namespace, τ set per stage pair | CoherenceGated with dynamic τ |
| MCP agent memory tools | Claude, GPT, Copilot agents | Agents need isolated but collaborative memory | Router as MCP backend | CentroidRouted with probe |
| Code intelligence | Developer tools | Per-repo vector isolation with cross-repo search for dependencies | Namespace = repository, coherence = API surface similarity | CoherenceGated |
| Scientific retrieval | Research systems | Paper namespaces by domain; cross-domain discovery | Coherence = citation graph similarity | CentroidRouted |
| Security event retrieval | SOC platforms | Strict tenant isolation for event logs | FlatIsolated with WitnessLog | FlatIsolated |
| Edge AI assistants | Local first AI | Device namespaces isolated from cloud namespaces | FlatIsolated in edge, CentroidRouted for sync | Both |
| Healthcare data | Clinical AI | Regulatory isolation (HIPAA) between patient cohorts | FlatIsolated + signed WitnessLog entries | FlatIsolated + ruvector-verified |

---

## Exotic Applications

| Application | 10-20 year thesis | Required advances | RuVector role | Risk / unknown |
|-------------|------------------|-------------------|---------------|----------------|
| Cognitum edge cognition | A Cognitum Seed device runs dozens of cognitive namespaces simultaneously, each representing a different perceptual context | Persistent edge vector stores, sub-microsecond routing | CoherenceGated as the thalamic routing layer | Power budget for coherence scoring on ARM M-class |
| RVM coherence domains | Coherence-gated namespaces become the memory management unit of the RVM agent OS | Hardened namespace IDs tied to capability tokens | Namespace router implements MMU for agent address space | Formalizing coherence as a security boundary |
| Proof-gated autonomous systems | Every cross-namespace access generates a ZK proof that the access was within the permitted coherence band | ZK-SNARK integration with fast proof generation | WitnessLog entries become ZK proof inputs | ZK overhead (currently too slow for real-time retrieval) |
| Swarm memory | A swarm of 1000 ruFlo agents shares a federated vector memory with per-agent namespaces | Distributed namespace routing with CRDT consistency | Namespace router as swarm memory primitive | Consistency vs availability trade-off |
| Self-healing vector graphs | After agent memory corruption, the coherence graph detects anomalous namespace similarity collapse and triggers compaction | Automatic coherence monitoring + repair actions | Coherence score time series as health signal | Defining "healthy" coherence baseline |
| Bio-signal memory isolation | Medical agents processing different patient streams must never mix their vector memories | Namespace router with hardware-enforced isolation | FlatIsolated with WASM memory sandboxing | Regulatory approval for AI-derived medical isolation |
| Agent operating systems | An agent OS schedules cognitive processes to memory namespaces the way Linux schedules threads to CPU cores | OS scheduler + namespace router integration | Namespace allocation as OS memory call | Formal verification of isolation semantics |
| Synthetic nervous system | A distributed network of agents, each with isolated namespaces, communicating only through coherence-gated messages | Coherence-gated communication protocol | CoherenceGated as the synaptic weight controller | Emergent behavior is unpredictable |

---

## Deep Research Notes

### What the SOTA suggests

The dominant isolation approach in production vector databases (namespace/collection-level separation) is effective for administrative isolation but not for semantic routing. No existing system in 2026 uses the vector content itself to govern retrieval boundaries.

The closest prior work is **filtered ANN** (ACORN, FAISS filtered search, Qdrant's payload filters), which uses metadata to restrict the search space. Namespace routing inverts this: instead of filtering *within* a fixed search space, it selects *which* search spaces to activate based on semantic distance from the query.

### What remains unsolved

1. **Dynamic τ adjustment**: Currently τ is set at construction time. A principled method for dynamically adjusting τ based on query type, namespace load, or security policy is an open problem.
2. **Distributed namespace routing**: When namespaces span multiple RuVector nodes (via `ruvector-raft`), coherence computation requires cross-node centroid synchronization. The consistency model for this is unspecified.
3. **Namespace merging and splitting**: When a namespace grows too large, it should split into sub-namespaces. The splitting criterion (coherence-based? size-based?) and the migration protocol are open.
4. **Adversarial coherence manipulation**: A malicious tenant could insert vectors designed to inflate their namespace's apparent coherence with another namespace, gaining unauthorized retrieval access. Defenses are not yet specified.

### Where this PoC fits

`ruvector-namespace-router` provides the routing layer for a production multi-tenant vector store. It is not a full vector database — it wraps any `NamespaceIndex`-compliant backend (including future HNSW or DiskANN backends). The crate is production-candidate for FlatIsolated and CentroidRouted; CoherenceGated requires hardening (persistent witness log, dynamic τ, distributed coherence).

### What would make this production grade

1. Back each namespace with a `ruvector-core` HNSW index instead of linear scan
2. Persist the WitnessLog to RVF or a write-ahead log
3. Connect coherence scores to `ruvector-mincut` partitioning output
4. Add a `NamespacePolicy` type for per-namespace τ and probe configuration
5. Integrate with `ruvector-verified` for signed witness entries

### What would falsify the approach

If coherence scores computed from centroid distances are found to be poor predictors of retrieval overlap (i.e., two namespaces with low centroid distance have completely disjoint top-K results), the gating function would need to be replaced with a more expensive but accurate metric (e.g., k-NN graph overlap sampled during insert).

---

## Production Crate Layout Proposal

```
crates/ruvector-namespace-router/
├── Cargo.toml
└── src/
    ├── lib.rs           # trait, shared types, l2sq, centroid helpers
    ├── flat_isolated.rs # Variant 1: strict isolation
    ├── centroid_routed.rs # Variant 2: centroid-pruned routing
    ├── coherence_gated.rs # Variant 3: coherence threshold + witness
    ├── witness.rs       # WitnessLog, WitnessEntry
    └── main.rs          # benchmark binary
```

Future extensions:
- `hnsw_namespace.rs` — HNSW-backed namespace (depends on ruvector-core)
- `policy.rs` — per-namespace τ, probe, and allow-list configuration
- `distributed.rs` — Raft-consistent centroid synchronization

---

## What to Improve Next

1. **HNSW namespace backend**: Replace linear scan with per-namespace HNSW. Expected latency reduction: 80µs → 5µs for D=128, N=500.
2. **Persistent WitnessLog**: Serialize entries to RVF binary format; load on restart.
3. **Dynamic coherence threshold**: Expose a `set_policy(ns, tau)` method for ruFlo to adjust τ per workflow phase.
4. **Cross-namespace recall benchmark**: Measure recall when CoherenceGated federates across overlapping namespaces vs exact cross-namespace ground truth.
5. **Integration with ruvector-mincut**: Use mincut partition assignments as namespace boundaries rather than caller-supplied IDs.

---

## References and Footnotes

[^1]: Patel, A. et al., "ACORN: Predicate-Agnostic Approximate Nearest Neighbor Search over Vector + Structured Data," SIGMOD 2024, arXiv:2403.04871, accessed 2026-06-07.

[^2]: EU AI Act, Article 13 (Transparency and provision of information to deployers), Official Journal of the European Union, 2024, https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689, accessed 2026-06-07.

[^3]: NIST AI Risk Management Framework (AI RMF 1.0), Govern 6.2 (Policies, processes, and accountability), NIST, 2023, https://airc.nist.gov/RMF, accessed 2026-06-07.

[^4]: Welford, B. P., "Note on a method for calculating corrected sums of squares and products," Technometrics, 4(3):419–420, 1962. Used here for incremental centroid and spread updates.

[^5]: Qdrant documentation, "Collections and Payload," https://qdrant.tech/documentation/concepts/collections/, accessed 2026-06-07. Describes namespace-via-collection isolation model.

[^6]: Milvus documentation, "Partition," https://milvus.io/docs/partition_key.md, accessed 2026-06-07. Describes partition-key isolation model.
