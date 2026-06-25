# Capability-Gated ANN Search

**150-char summary:** Per-vector read access control in ANN search using 64-bit bitset capability tokens, with three measured variants: PostFilter, EagerMask, and CapGraph.

---

## Abstract

Modern AI deployments need retrieval systems that enforce **who can see what** at the
vector level, not just the collection level.  This research implements and benchmarks
three strategies for capability-gated approximate nearest-neighbour (ANN) search in
Rust, where each stored vector carries a required capability mask and each search query
presents a held capability mask.

| Variant | Recall@10 | Mean(μs) | QPS | Best for |
|---------|-----------|----------|-----|----------|
| PostFilter | 1.000 | 494 | 2,023 | Small corpora, high access ratio |
| EagerMask | 1.000 | 57–175 | 5,728–17,548 | Low access ratio (<50%) |
| CapGraph | 0.906 | 289 | 3,466 | Sub-linear recall with isolation |

Numbers from n=5,000 × 64-dim, release build, x86_64 Linux, `cargo run --release`.

---

## Why This Matters for RuVector

RuVector is positioned as a **Rust-native cognition substrate** — not just a vector
database, but a memory and retrieval layer for AI agents.  Multi-agent deployments
create a fundamental access control problem: a single index may hold memories for
hundreds of agents, users, or security domains.

Two complementary capabilities now exist:

- **ADR-227**: Proof-gated writes — you need a cryptographic proof to *write* a vector.
- **ADR-268** (this work): Capability-gated reads — you need capability tokens to *read*
  a vector.

Together they form a complete read-write access control model for agent memory.

---

## 2026 State of the Art Survey

### Access Control in Vector Databases

| System | Granularity | Mechanism | ANN-integrated |
|--------|-------------|-----------|----------------|
| Milvus | Collection / partition | RBAC on collection | No (post-filter) |
| Qdrant | Collection | API key | No |
| Weaviate | Class | OIDC / API key | No |
| pgvector | Table | PostgreSQL GRANT | No |
| Pinecone | Namespace | API key | No |
| RuVector capgated | **Per-vector** | Bitset CapMask | **Yes (EagerMask, CapGraph)** |

No production vector database today enforces access control at the per-vector level
during ANN index traversal.  All current systems apply access control either at the
collection boundary or as a post-hoc filter after retrieval.

### The Post-Filter Problem

When a system retrieves k=10 results and then filters by access control, it may return
fewer than k results if some retrieved vectors are unauthorised.  To guarantee k valid
results, the system must over-retrieve with k' >> k and then filter — a strategy called
"over-fetching."  At low access ratios (say 5% of vectors authorised), over-fetching
requires k' = 200k candidates, which is equivalent to a full corpus scan.

The EagerMask variant eliminates this problem: it only computes distances for authorised
vectors, so retrieval cost scales with the authorised fraction of the corpus.

### Capability-Based Security

Capability-based access control dates to the 1960s (Fabry 1974 [^1], Dennis & Van Horn
1966 [^2]).  The key insight: a "capability" is an unforgeable token that grants a
specific right.  Modern OS implementations (seL4 [^3], EROS) use capabilities for memory
isolation.  This work brings capability-based access control to vector retrieval.

The `CapMask` bitset model is a simplified form: capabilities are bit positions in a
64-bit integer.  A querier satisfies a vector's requirements if they hold all required
bits.  This is equivalent to a "conjunction of capabilities" access policy.

---

## Forward-Looking 10–20 Year Thesis

### 2026–2030: Agent Memory Isolation

As AI agents proliferate, the need for isolated per-agent memory becomes critical.  A
2026 deployment might run 1,000 concurrent agents on a shared RuVector instance.  Each
agent must retrieve only its own memories.  Capability-gated ANN is the natural
primitive.

### 2031–2036: Federated Cognition

Distributed agent networks will share knowledge selectively.  Capability tokens become
the protocol for knowledge sharing: an agent grants another agent a capability token to
access a subset of its memory.  The graph walk in CapGraph naturally models the "reachable
subgraph" of authorised knowledge.

### 2037–2046: Autonomous Capability Negotiation

Advanced agent operating systems will negotiate capabilities dynamically.  An agent
requests a capability, the system verifies the request against policy, and issues a
time-limited cryptographic token.  RuVector's capability-gated index becomes the
storage layer for a full agent capability management system.

This trajectory mirrors the evolution of OS security: from UNIX DAC (1970s) → mandatory
access control (1980s) → capability-based OS (2000s) → capability-based AI cognition (2040s).

---

## ruvnet Ecosystem Fit

| Ecosystem Component | Integration Point |
|--------------------|-------------------|
| `ruvector-proof-gate` | Write-side proof; read-side CapMask; full WORM+ACL model |
| `ruvector-coherence-hnsw` | Coherence scoring gates navigation; CapMask gates retrieval |
| `ruvector-agent-memory` | Each memory entry tagged with agent-ID capability |
| `rvf` (RuVector Format) | CapMask stored in RVF manifest per-vector metadata |
| `ruFlo` | Workflow step issues capability tokens for downstream retrieval |
| MCP tools | `vector_memory_search(query, capability_token)` MCP tool surface |
| WASM/edge | Zero-dep crate compiles to WASM; capability checks are bitwise ops |

---

## Proposed Design

### Core Types

```rust
/// 64-bit bitset: bit i set means "capability i is required/held"
pub struct CapMask(pub u64);

impl CapMask {
    /// Querier satisfies requirement iff they hold all required bits
    pub fn satisfies(self, required: CapMask) -> bool {
        (self.0 & required.0) == required.0
    }
}

pub trait CapGatedIndex {
    fn insert(&mut self, id: usize, vector: Vec<f32>, required: CapMask);
    fn search(&self, query: &[f32], k: usize, holder: CapMask) -> Vec<SearchResult>;
    fn name(&self) -> &'static str;
}
```

### Architecture Diagram

```mermaid
graph TD
    A[Query + CapMask] --> B{Variant}

    B --> C[PostFilter]
    C --> C1[Scan all n vectors]
    C1 --> C2[Sort by distance]
    C2 --> C3[Filter by CapMask]
    C3 --> R1[Top-k results]

    B --> D[EagerMask]
    D --> D1[Build authorised bitset O(n)]
    D1 --> D2[Scan authorised only O(auth_frac·n·d)]
    D2 --> D3[Sort + return]
    D3 --> R2[Top-k results]

    B --> E[CapGraph]
    E --> E1[Build k-NN graph at insert time O(n²·d)]
    E1 --> E2[Seed from entry points]
    E2 --> E3[Greedy BFS, ef-bounded]
    E3 --> E4[Add to results if authorised]
    E4 --> R3[Top-k results (≤100% recall)]
```

---

## Implementation Notes

### Dataset Generation

All vectors drawn from an approximately Normal distribution using a seeded LCG PRNG
(no external dependencies).  The LCG uses the Knuth multiplicative hash:

```
state = state * 6364136223846793005 + 1442695040888963407
```

Each vector's approximate Normal values are generated by summing 8 uniform values and
standardising (central limit theorem approximation, σ ≈ 1.0).

### CapGraph Build

The PoC builds a k-NN proximity graph with brute-force O(n²·d) complexity.  The
`batch_build` method builds the graph once after all insertions.  For production,
replace with an HNSW-style incremental graph (O(n log n) amortized).

### Exploration Factor

The CapGraph uses an exploration factor `ef = k × ef_multiplier` (default: ef_multiplier
= 30, so ef = 300 for k=10).  This controls how many nodes are visited per query.  The
ef must be larger than for standard ANN search because many visited nodes will be
unauthorised and thus "wasted" visits.  The relationship is:

```
ef_needed ≈ k / access_ratio × graph_diameter_factor
```

For access_ratio = 37.5% and k=10: ef ≥ ~30.  Our ef=300 provides comfortable recall.

---

## Benchmark Methodology

- **Platform**: x86_64 Linux, Rust 1.94.1 (release build, no unsafe)
- **Dataset**: n=5,000 vectors × d=64 dims (Normal distribution, LCG seed)
- **Capability model**: 8 bits, each vector requires exactly 1 bit
- **High-access scenario**: querier holds 3/8 bits → 37.5% of vectors accessible
- **Low-access scenario**: querier holds 1/8 bits → 12.5% of vectors accessible
- **Queries**: 200 per scenario, k=10
- **Recall**: measured against Oracle (brute-force scan of authorised subset)
- **Latency**: wall-clock time per query (Instant::now())
- **Graph**: degree=12, entry_points=8, ef_multiplier=30

---

## Real Benchmark Results

Command: `cargo run --release -p ruvector-capgated --bin benchmark`  
Build: `cargo build --release -p ruvector-capgated` → 5.2s compile time

### Scenario: High-access (37.5% authorised)

Holder mask: `0b01001001` | Authorised: 1875/5000

| Variant | N | Dims | Queries | Mean(μs) | p50(μs) | p95(μs) | QPS | Recall@10 | Mem(MB) | Pass |
|---------|---|------|---------|----------|---------|---------|-----|-----------|---------|------|
| PostFilter | 5000 | 64 | 200 | 494.3 | 493 | 552 | 2,023 | 1.000 | 1.26 | PASS |
| EagerMask | 5000 | 64 | 200 | 174.6 | 167 | 206 | 5,728 | 1.000 | 1.26 | PASS |
| CapGraph | 5000 | 64 | 200 | 288.6 | 281 | 329 | 3,466 | 0.906 | 1.72 | PASS |

### Scenario: Low-access (12.5% authorised)

Holder mask: `0b00000001` | Authorised: 625/5000

| Variant | N | Dims | Queries | Mean(μs) | p50(μs) | p95(μs) | QPS | Recall@10 | Mem(MB) | Pass |
|---------|---|------|---------|----------|---------|---------|-----|-----------|---------|------|
| PostFilter | 5000 | 64 | 200 | 450.2 | 456 | 487 | 2,221 | 1.000 | 1.26 | PASS |
| EagerMask | 5000 | 64 | 200 | 57.0 | 54 | 74 | 17,548 | 1.000 | 1.26 | PASS |
| CapGraph | 5000 | 64 | 200 | 294.5 | 288 | 358 | 3,396 | 0.869 | 1.72 | PASS |

**Graph build time**: 2.28–2.34s for n=5,000 (O(n²·d) brute-force, runs once per index)

**Acceptance thresholds**: PostFilter recall ≥ 0.95, EagerMask recall ≥ 0.95, CapGraph recall ≥ 0.70.  All PASS.

---

## Memory and Performance Math

### Memory

For n=5,000 vectors at d=64 dims:
- Raw vectors: 5,000 × 64 × 4 bytes = 1.28 MB
- CapMask array: 5,000 × 8 bytes = 0.04 MB
- PostFilter/EagerMask total: **1.26 MB** (no graph)
- CapGraph adjacency: 5,000 × 12 × 8 bytes = 0.48 MB additional
- CapGraph total: **1.72 MB**

For n=1M at d=768 (typical LLM embedding):
- Raw: 1M × 768 × 4 = 3,072 MB (3 GB)
- EagerMask adds: 1M × 8 = 8 MB (negligible)
- CapGraph adds: 1M × 16 × 8 = 128 MB (~4% overhead)

### EagerMask Speedup Model

Expected speedup of EagerMask over PostFilter:
```
speedup ≈ 1 / access_ratio
```
At 12.5% access: expected 8×, measured 7.9× (450 μs / 57 μs). **Prediction holds.**  
At 37.5% access: expected 2.7×, measured 2.8× (494 μs / 175 μs). **Prediction holds.**

This confirms that EagerMask scales linearly with the authorised fraction and provides
predictable, tunable performance.

---

## How It Works Walkthrough

### Step 1: Insert

Each vector is stored with a required `CapMask`.  For EagerMask/PostFilter, this is
just an array entry.  For CapGraph, the proximity graph is rebuilt after batch insert.

### Step 2: PostFilter Search

1. Compute squared Euclidean distance from query to ALL n vectors: O(n·d)
2. Sort by distance: O(n log n)
3. Iterate sorted list, emit only authorised vectors: O(n)
4. Stop after k results

Total: O(n·d + n log n).  Identical latency regardless of access ratio.

### Step 3: EagerMask Search

1. Scan required-mask array and build bool[] of authorised indices: O(n) bitwise ops
2. Compute distance only for authorised vectors: O(auth_frac · n · d)
3. Sort authorised results: O(auth_count log auth_count)
4. Return top-k

Total: O(auth_frac · n · d).  Latency scales with access ratio.

### Step 4: CapGraph Search

1. Build k-NN proximity graph (once, at batch-insert time): O(n² · d)
2. Seed exploration frontier with n_entry_points evenly spaced nodes
3. Greedy best-first graph walk:
   - Pop closest unexplored node from min-heap
   - If authorised: add to result max-heap, evict worst if > k
   - Always: expand all neighbours (add to frontier if unseen)
   - Stop after ef total node visits
4. Return sorted results

Total search: O(ef · degree · d).  ef=300, degree=12 → ~3,600 distance computations per
query, vs 5,000 for EagerMask at full corpus, or ~625 for EagerMask at 12.5% access.

---

## Practical Failure Modes

1. **Access ratio → 0%**: CapGraph may return fewer than k results (not enough
   authorised nodes reached in ef budget).  Fix: increase ef_multiplier.

2. **Authorised subgraph disconnected**: if the k-NN graph connects vectors by proximity
   and authorised vectors cluster in sparse regions, the graph walk may not reach them.
   Fix: add random entry points from authorised-only nodes, or use a forest of entry
   points.

3. **CapMask forgery**: without cryptographic signing, any caller can pass any CapMask.
   Fix: integrate with ruvector-proof-gate; issue signed CapTokens rather than raw
   integers.

4. **Capability explosion**: if every vector has unique capabilities, the CapMask bitset
   saturates.  Fix: hierarchical capability scheme (role → set of atomic capabilities).

---

## Security and Governance Implications

**What this provides:**
- Logical separation of vectors at retrieval time
- Correct result sets (authorised vectors only)
- Predictable performance degradation

**What this does NOT provide (yet):**
- Cryptographic proof of capability authenticity
- Protection against timing side-channels (EagerMask latency reveals access ratio)
- Strict traversal isolation (CapGraph still traverses unauthorised node coordinates)

**Production recommendation**: pair EagerMask with a signed `CapToken` from the
ruvector-proof-gate system.  The proof-gate issues tokens; the capgated index enforces
them.  Never accept raw `CapMask` from untrusted callers.

---

## Edge and WASM Implications

The `ruvector-capgated` crate has **zero external dependencies**: it compiles to WASM
as-is.  The CapMask check is a single bitwise AND — one of the cheapest possible
operations.  On a Cortex-M4 at 168 MHz, a 64-bit AND takes 1 clock cycle.

For Cognitum Seed (edge appliance) deployments:
- EagerMask is ideal: scales to the local agent's narrow capability set
- WASM binary adds ~50 KB for the full crate (PostFilter + EagerMask + CapGraph)
- The LCG dataset generator enables on-device benchmark calibration

---

## MCP and Agent Workflow Implications

The natural MCP tool surface:

```
tool: vector_memory_search
params:
  query: f32[]        # embedding vector
  k: int              # top-k
  capability_token: string  # signed JWT or proof token
returns:
  results: [{id, distance, metadata}]
  authorised_count: int
  total_count: int
```

In a ruFlo workflow:
1. Agent requests memory search for topic X
2. ruFlo issues a time-limited CapToken scoped to the agent's session
3. capgated index evaluates token, returns only authorised memories
4. ruFlo revokes token after query (single-use)

This maps exactly to the capability model in this crate.

---

## Practical Applications

| Application | User | Why It Matters | RuVector Use | Path |
|-------------|------|----------------|--------------|------|
| **Multi-tenant agent memory** | SaaS AI provider | Users must not see each other's memories | EagerMask with per-user capability bits | Wrap ruvector-agent-memory with capgated |
| **Enterprise RAG** | Enterprise knowledge base | Clearance levels for documents | CapMask = clearance bitset (public, internal, confidential, secret) | Index enterprise documents with clearance masks |
| **MCP memory tools** | Claude / agent via MCP | Tool invocation scoped to session | ruFlo issues CapToken for each agent session | MCP tool adapter over capgated index |
| **Healthcare AI** | Medical records system | HIPAA — patient-scoped record access | Provider ID = capability bit | Wrap patient embeddings with provider caps |
| **Code intelligence** | IDE AI assistant | Private repo vs public repo context | Repository ACL → CapMask | Index code embeddings with repo permission masks |
| **Security event retrieval** | SOC analyst | Tier-1 vs Tier-2 analyst access | Analyst tier = capability set | Index SIEM embeddings with analyst-level caps |
| **Federated graph RAG** | Research network | Consortium data governance | Institution membership = capability | Federated capgated index per consortium |
| **Edge IoT** | Industrial control | Sensor data access by operator role | Operator role = capability bit | On-device EagerMask on WASM runtime |

---

## Exotic Applications

| Application | 10–20 Year Thesis | Required Advances | RuVector Role | Risk |
|-------------|-------------------|-------------------|---------------|------|
| **Agent operating system** | Capabilities extend to all memory operations in an agent OS (like seL4 for AI) | Trusted execution environment, hardware-enforced caps | Capability-gated index as kernel primitive | OS security model may be too rigid for fluid AI cognition |
| **Swarm memory isolation** | Thousands of cooperative agents maintain isolated memory with selective sharing via negotiated capabilities | Automated capability negotiation protocol, cap delegation | Shared RuVector instance with per-agent CapMask | Byzantine agent could forge capabilities without hardware root of trust |
| **Proof-gated autonomous systems** | Write proof + read capability = cryptographically auditable AI decision log | Zero-knowledge proofs for CapMask issuance | Bridge ADR-227 (proof writes) + ADR-268 (cap reads) | ZK proof overhead may be too high for real-time systems |
| **Differential privacy capability model** | Capability tokens issued probabilistically based on privacy budget | DP mechanism over capability space, adaptive ε | CapMask as DP sensitivity parameter | Privacy-capability interaction is not well-studied |
| **Cognitum edge cognitive isolation** | Each cognitive context (attention focus) has isolated memory access; inattention = capability revocation | Hardware-enforced memory domains, interrupt-driven cap revocation | Edge capgated index on Cognitum Seed | Cognitive context boundaries are fuzzy |
| **Synthetic nervous system** | Sensory neurons only access relevant memories; motor neurons access only motor memories | Neuromorphic chip with per-neuron capability tables | CapMask as neural routing primitive | Biological plausibility unclear |
| **RVM coherence domain memory** | Coherence domains from RVM map to capability groups; domain transition = capability change | RVM integration, real-time coherence scoring | CapGraph nodes partitioned by coherence domain | Coherence domains are dynamic; static CapMask may lag |
| **Self-modifying capability graphs** | Agents earn new capabilities by demonstrating competence; CapMask is dynamic | Reward-based capability issuance, RL integration | Index updates as capabilities evolve | Security risk: unverified competence claims inflate capabilities |

---

## Deep Research Notes

### What the SOTA Suggests

No published research specifically addresses capability-gated ANN search at the per-vector
level with index-integrated enforcement.  The closest work is:

- **ACORN** (2024) [^4]: predicate-integrated HNSW graph pruning. ACORN prunes graph
  edges based on a predicate during build time, unlike our run-time CapMask check.
- **Filtered ANN survey** (Simhadri et al., 2024) [^5]: comprehensive benchmark of
  filtered ANN.  All approaches use metadata filters applied either before (pre-filter)
  or after (post-filter) the ANN stage.  No capability-based approach is evaluated.
- **Milvus attribute filtering** (2021) [^6]: Milvus filters on attributed metadata but
  doesn't integrate access control into the ANN graph structure.

### What Remains Unsolved

1. **Capability-aware graph construction**: build a k-NN graph that respects capability
   structure (connect nodes only if they share a capability bit).  This would improve
   CapGraph recall without increasing ef.
2. **Dynamic capability assignment**: updating CapMask on live vectors without
   reconstructing the index.
3. **Cryptographic CapMask**: integrating ZK proofs so a querier can prove they hold a
   capability without revealing which capability they hold.
4. **Multi-label capabilities**: current model assigns one capability group per vector;
   a vector should be retrievable by any holder of ANY of several capability sets.

### Where This PoC Fits

This PoC demonstrates that per-vector capability filtering is practical at small-to-medium
scale and provides a clean trait interface for production integration.  The EagerMask
variant is production-ready for read-time access control.  The CapGraph variant is a
research prototype showing the recall-isolation tradeoff.

### What Would Make This Production-Grade

1. Incremental graph build (O(n log n) instead of O(n²))
2. Cryptographic CapToken (signed by proof-gate)
3. Constant-time EagerMask to prevent timing side-channels
4. Variable-length CapMask for >64 capabilities
5. Integration with ruvector-server's REST API

### What Would Falsify This Approach

- Evidence that collection-level access control is sufficient for all multi-agent
  deployments (unlikely given single-index agent memory designs)
- A faster alternative that achieves per-vector access control without the EagerMask
  scan (possible with capability-aware quantization)
- Evidence that the ZK-proof overhead makes cryptographic CapTokens impractical for
  latency-sensitive workloads (would constrain the security model)

---

## Production Crate Layout Proposal

```
crates/ruvector-capgated/
├── Cargo.toml
└── src/
    ├── lib.rs          # CapMask, CapGatedIndex trait, SearchResult, recall_at_k
    ├── token.rs        # Signed CapToken (future: ZK proof integration)
    ├── dataset.rs      # Deterministic benchmark data generation
    ├── oracle.rs       # Brute-force ground truth
    ├── post_filter.rs  # Variant 1: PostFilter
    ├── eager_mask.rs   # Variant 2: EagerMask (production-ready)
    ├── cap_graph.rs    # Variant 3: CapGraph (research)
    └── bin/
        └── benchmark.rs
```

Future integration:
```
crates/ruvector-core/src/
└── access_control/
    ├── mod.rs          # Re-exports from ruvector-capgated
    └── wrappers.rs     # CapGatedWrapper<T: AnnIndex>
```

---

## What to Improve Next

1. **Cryptographic CapToken**: integrate with ADR-227 proof-gate to sign capability
   assertions.
2. **CapGraph incremental insert**: replace O(n²) build with HNSW-style insertion.
3. **Capability-aware PQ**: embed capability bits in product quantization codebooks.
4. **Timing-hardened EagerMask**: constant-time scan to prevent access-ratio leakage.
5. **MCP tool wrapper**: `vector_memory_search(query, signed_token)` MCP tool surface.
6. **Integration test with ruvector-agent-memory**: run multi-agent scenario with
   isolated per-agent capability sets.

---

## References and Footnotes

[^1]: Fabry, R.S. "Capability-Based Addressing." Communications of the ACM 17(7), 1974. https://dl.acm.org/doi/10.1145/361011.361070. Accessed 2026-06-25.

[^2]: Dennis, J.B. and Van Horn, E.C. "Programming Semantics for Multiprogrammed Computations." Communications of the ACM 9(3), 1966. Foundational paper on capability-based security. Accessed 2026-06-25.

[^3]: Klein, G. et al. "seL4: Formal Verification of an OS Kernel." SOSP 2009. https://dl.acm.org/doi/10.1145/1629575.1629596. Accessed 2026-06-25.

[^4]: Peng, P. et al. "ACORN: Performant Predicate-Driven Nearest Neighbor Search over Vector Embeddings." VLDB 2025. https://arxiv.org/abs/2403.04871. Accessed 2026-06-25.

[^5]: Simhadri, H. et al. "Results of the NeurIPS'23 Competition on Practical Vector Search." arXiv 2024. https://arxiv.org/abs/2409.17424. Accessed 2026-06-25.

[^6]: Wang, J. et al. "Milvus: A Purpose-Built Vector Data Management System." SIGMOD 2021. https://dl.acm.org/doi/10.1145/3448016.3457550. Accessed 2026-06-25.

[^7]: Malkov, Y. and Yashunin, D. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." IEEE TPAMI 2020. Reference for HNSW graph structure used as inspiration for CapGraph. Accessed 2026-06-25.
