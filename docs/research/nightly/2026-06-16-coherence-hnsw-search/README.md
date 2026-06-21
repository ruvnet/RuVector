# Coherence-Gated HNSW Search

**Nightly research · 2026-06-16 · crate: `ruvector-coherence-hnsw`**

> **150-character summary.** Traversal-direction coherence gates prune off-path HNSW beam expansion—7.5% fewer neighborhood expansions, 5% lower latency, ≤3% recall trade-off, measured in Rust.

---

## Abstract

Standard HNSW beam search expands every candidate's neighbor list unconditionally. When the search entry point is distant from the query — as it always is in HNSW's layer-0 phase after upper-layer greedy descent — some of those candidate expansions follow "off-path" branches: nodes that are close to the current candidate but directionally misaligned with the query. These expansions consume distance computations without improving recall.

This research introduces **traversal-direction coherence gating**: before expanding a candidate's neighbor list, we compute the cosine similarity between the direction from the entry point to the candidate and the direction from the entry point to the query. If the cosine (traversal coherence) falls below a threshold, the candidate's neighborhood is skipped. The candidate is still considered as a result; only the branch exploration is suppressed.

We implement three variants — Baseline, CoherenceGated (fixed threshold), and AdaptiveCoherence (dynamic threshold) — and measure them on a clustered navigable small-world flat graph in Rust. Key measured results:

| Variant | Expansions/q | Recall@10 | Mean µs |
|---------|-------------|-----------|---------|
| Baseline | 13.2 | 93.0% | 84.8 |
| CoherenceGated(t=0.50) | 12.2 (−7.5%) | 90.3% (−2.7%) | 77.0 (−9.2%) |
| AdaptiveCoherence | 13.2 (≈ BL) | 92.9% (≈ BL) | 81.9 |

All numbers are from a real cargo run on Linux, Rust 1.94.1 (release build).

---

## Why This Matters for RuVector

RuVector is not just a vector database: it is a Rust-native cognition substrate for agent memory, graph retrieval, and edge AI. Three direct connections:

1. **HNSW is the bottleneck.** `ruvector-core` uses HNSW (via `hnsw_rs`) as its primary ANN index. Any reduction in layer-0 beam work directly reduces RuVector query latency at scale.

2. **Agent memory is graph-based.** In ruFlo workflows and agent memory stores, memories are organized as proximity graphs. Off-path pruning during retrieval reduces noise: the agent receives fewer irrelevant memories, improving context coherence in LLM calls.

3. **Coherence bridges to ruvector-mincut.** The traversal coherence score is complementary to ruvector-mincut's spectral coherence: mincut identifies structurally important cuts in the graph; direction coherence identifies local traversal directions. Together, they provide a richer signal for selective graph exploration.

---

## 2026 State of the Art Survey

### ANN research landscape (mid-2026)

- **HNSW** (Malkov & Yashunin 2020)[^1] remains the dominant in-memory ANN algorithm. Its multi-layer structure achieves O(log N) navigation with ef-controllable recall-throughput tradeoff.
- **DiskANN/Vamana** (Jayaram Subramanya et al. 2019)[^2] extends ANN to SSD with a disk-optimized graph. RuVector already has `ruvector-diskann`.
- **ACORN** (Patel et al. 2024)[^3] improves filtered HNSW via denser graphs and predicate-agnostic traversal. RuVector implemented this in `ruvector-acorn` (2026-04-26 nightly).
- **RaBitQ** (Gao et al. 2024)[^4] achieves 1-bit quantization for ultra-fast ANNS. RuVector implemented this in `ruvector-rabitq` (2026-04-23 nightly).
- **Beam search pruning** is an active area. SpFresh (2022)[^5] maintains freshness in dynamic indexes. FINGER (2023)[^6] uses early termination based on first-dimensional distances. None use direction-coherence as a pruning signal.

### Gap identified

No published algorithm prunes HNSW beam expansion using **traversal direction coherence** — the alignment between the current movement direction and the query direction. This is a novel signal orthogonal to:
- Distance (how close is the candidate?)
- Predicate (does the candidate pass a metadata filter?)
- Freshness (was the candidate recently inserted?)

### Competitor posture

| System | Beam search style | Early termination | Direction pruning |
|--------|------------------|-------------------|------------------|
| FAISS | Pure ef-based beam | None (run full ef) | No |
| Qdrant | HNSW with ef tuning | Distance-based | No |
| Milvus | HNSW with ef_search | None documented | No |
| LanceDB | IVF + HNSW hybrid | None documented | No |
| pgvector | HNSW (local) | None documented | No |
| **RuVector** | **HNSW + coherence gate** | **Yes (this work)** | **Yes (this work)** |

---

## Forward-Looking 10–20 Year Thesis

### 2026: Foundation

Traversal-direction coherence is a heuristic pruning signal. It works best when:
- The embedding space has well-defined directional structure (clusters, hierarchies)
- The search path is long enough for coherence to diverge meaningfully
- The graph has long-range navigability (long-jump edges or multi-layer HNSW)

### 2030–2036: Dynamic coherence domains

As agent memory grows to billions of entries, traversal coherence will evolve from a per-query heuristic to a **persistent structural property**: each edge in the proximity graph will carry a stored coherence score relative to known "topic clusters". Search will gate edges at build time, not just query time, creating coherence-domain-indexed subgraphs.

This aligns with RuVector's existing `ruvector-coherence` crate and the `ruvector-mincut` spectral analysis tools.

### 2036–2046: Coherence-native agent operating systems

By 2040, agent operating systems (agent-OS) will manage millions of concurrent agent instances, each maintaining a personal memory graph. The CPU/GPU budget for vector retrieval will be a first-class resource constraint. Coherence gating, combined with learned-threshold tuning (using ruFlo feedback loops), will become a standard compiler optimization pass applied to all retrieval operations — similar to how branch prediction is handled in CPU pipelines today.

The RuVector coherence-gated graph will serve as the memory substrate for Cognitum Seed edge appliances, where power budgets (≤5W) make every avoided distance computation matter.

---

## ruvnet Ecosystem Fit

| Component | Connection |
|-----------|-----------|
| `ruvector-core` | Coherence gate applies to layer-0 HNSW traversal |
| `ruvector-coherence` | Spectral coherence complements traversal coherence |
| `ruvector-mincut` | Mincut identifies cluster boundaries; gate respects them |
| `ruvector-gnn` | GNN embeddings create richer direction signals |
| `ruvector-diskann` | DiskANN page-local graph benefits from gated traversal |
| `ruvf/rvf-manifest` | Coherence threshold can be packaged in RVF manifests |
| `ruFlo` | Adaptive threshold optimization becomes a ruFlo workflow |
| `Cognitum Seed` | Edge power budgets reward every avoided computation |
| `ruvector-verified` | Proof-gated writes can store coherence attestations |
| `mcp-gate` | MCP tool surface exposes coherence-controlled recall |

---

## Proposed Design

### Core trait

```rust
pub trait Searcher {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
    ) -> SearchResult;
}
```

### Traversal coherence formula

```
coherence(entry, candidate, query) =
    cos_sim(candidate − entry,  query − entry)
```

- **+1.0**: candidate is directly toward query
- **0.0**: candidate is perpendicular to query direction
- **−1.0**: candidate is directly away from query

### Three variants

```
Baseline            → expand all candidates (gate always passes)
CoherenceGated(t)   → expand only when coherence ≥ t (fixed threshold)
AdaptiveCoherence   → threshold rises as beam finds improvements, falls when stuck
```

### Architecture diagram

```mermaid
graph TD
    Q[Query vector q] --> EP[Entry point node 0]
    EP --> |"d₀ = l2_sq(q, entry)"| HEAP[Min-heap candidates]

    HEAP --> |"pop closest c"| GATE{Coherence gate}

    GATE --> |"coherence(entry, c, q) ≥ θ"| EXPAND[Expand c's neighbors]
    GATE --> |"coherence < θ"| SKIP[Skip expansion\nkeep c as result]

    EXPAND --> |"for nbr in neighbors[c]"| VISITED{Already visited?}
    VISITED --> |"no"| DIST[Compute l2_sq(q, nbr)]
    DIST --> |"if nbr better than worst beam"| HEAP
    VISITED --> |"yes"| NEXT[Next neighbor]

    SKIP --> RESULTS[Results heap top-k]
    EXPAND --> RESULTS
```

---

## Implementation Notes

### Graph construction

The `FlatGraph` builds via brute-force exact k-NN for local edges, plus random long-jump edges. Long-jump edges enable cross-cluster navigation from a fixed entry point, mimicking HNSW's upper-layer greedy descent.

Without long-jump edges, a fixed entry trapped in cluster 0 never reaches cluster 5, giving recall near 20%. With 6 long-jump edges per node, recall reaches 93%.

### Bounded beam

The beam maintains at most `ef` candidates. When a new candidate is pushed and the heap exceeds `ef`, the farthest candidate is evicted. This keeps search work bounded to O(ef × M × D) distance computations.

### Adaptive threshold

The adaptive variant tracks `best_dist` across pops. Each time a pop beats `best_dist`, the threshold rises by `adaptation_rate = 0.08`. Each time it doesn't, the threshold falls by `adaptation_rate × 0.5`. The threshold clamps to `[0.0, max_threshold = 0.65]`.

For the clustered dataset, the beam finds good results early (via long-jump), and the threshold rises, but it stabilizes below the level where pruning is meaningful. On larger graphs with longer navigation paths, the adaptive variant is expected to show more pruning.

---

## Benchmark Methodology

- **Dataset**: 8 Gaussian clusters × 250 vectors = 2,000 total, D=32, std=0.15, unit-normalized
- **Queries**: 200 cluster-aware queries (sampled near cluster centers, different seed)
- **Ground truth**: brute-force exact k-NN (O(N² · D))
- **Graph**: M=16 local + M_lj=6 long-jump = 22 connections/node
- **Search**: ef=80, k=10, fixed entry=0
- **Measurement**: wall-clock `Instant::now()` in Rust, 200 queries, percentiles computed
- **Recall**: fraction of true top-10 found in returned top-10

---

## Real Benchmark Results

**Environment:**
- OS: Linux
- Rust: 1.94.1 (e408947bf 2026-03-25)
- Build: `cargo run --release -p ruvector-coherence-hnsw --bin benchmark`

**Dataset:**
- N=2,000 vectors, D=32 dimensions, 8 clusters
- 200 queries, k=10, ef=80
- Graph build: 60 ms (brute-force O(N²·D))
- Memory: 432,000 bytes (421.9 KB) — vectors + adjacency list

| Variant | Mean (µs) | p50 (µs) | p95 (µs) | QPS | Pops/q | Expansions/q | Recall@10 |
|---------|-----------|----------|----------|-----|--------|-------------|-----------|
| Baseline | 84.8 | 81.8 | 123.7 | 11,794 | 14.2 | 13.2 | 93.0% |
| CoherenceGated(t=0.50) | 77.0 | 80.8 | 116.9 | 12,989 | 14.5 | 12.2 | 90.3% |
| AdaptiveCoherence | 81.9 | 79.1 | 116.1 | 12,209 | 14.2 | 13.2 | 92.9% |

**Acceptance results:**
- Baseline recall@10 ≥ 85%: **PASS** (93.0%)
- CoherenceGated recall@10 ≥ 82%: **PASS** (90.3%)
- AdaptiveCoherence recall within 5% of Baseline: **PASS** (92.9% vs 93.0%)
- CoherenceGated expansions ≤ 95% of Baseline: **PASS** (92.5% — saves 7.5%)

---

## Memory and Performance Math

**Graph memory:**
- Vectors: 2,000 × 32 × 4 bytes = 256,000 bytes
- Adjacency (22 neighbors/node): 2,000 × 22 × 4 bytes = 176,000 bytes
- **Total: 432,000 bytes ≈ 422 KB**

**Search work (per query):**
- Pops from heap: ~14 candidates
- Expansions (neighbor list iterations): 13.2 (baseline) vs 12.2 (gated)
- Distance computations: ~13.2 × 22 = 290 (baseline) vs ~12.2 × 22 = 268 (gated)
- **Savings: 22 distance computations per query (7.5% fewer)**

**Scaling projection (not measured, analytical):**
- At N=1M with M=32, ef=100: ~100 × 32 = 3,200 distance computations/query
- With 7.5% coherence saving: ~240 fewer distance computations/query
- At D=768 (BERT-like): 240 × 768 = 184,320 multiply-adds saved per query
- At 10,000 QPS: 1.84 billion multiply-adds/second saved — meaningful at this scale

*Note: The 7.5% savings is measured on a PoC dataset. Production HNSW graphs with longer navigation paths may show larger or smaller coherence savings depending on dataset structure.*

---

## How It Works: Walkthrough

1. **Entry**: All queries start at node 0 (in cluster 0). This simulates HNSW layer-0 after upper-layer greedy descent placed us near but not at the query.

2. **Initial heap**: Push (distance=l2_sq(query, node0), node=0) to the min-heap.

3. **Pop loop**: Pop the closest candidate C from the min-heap.

4. **Coherence check** (gated variants only):
   ```
   coherence = cosine(C.vec − entry.vec, query − entry.vec)
   ```
   If `coherence < threshold`: skip expanding C's neighbors, continue loop.

5. **Expansion** (passed gate or baseline):
   Iterate C's 22 neighbors, compute l2_sq to each, push to heap if closer than worst pending.

6. **Early stop**: When the closest candidate is already worse than the k-th best result found so far, the loop terminates.

7. **Result**: Top-k nodes sorted by distance.

The gate fires when C is "sideways" relative to the query direction. On the clustered dataset with long-jump navigation, the gate fires on ~7.5% of expansions — typically when the beam briefly explores nodes in the wrong cluster before correcting via a long-jump.

---

## Practical Failure Modes

| Scenario | Effect | Mitigation |
|----------|--------|------------|
| Threshold too high (>0.70) | Recall collapse — gate prunes on-path candidates | Adaptive variant; threshold calibration |
| Isotropic random dataset | Gate never fires (all candidates are roughly equicoherent) | Only valuable on structured/clustered data |
| Very short paths (warm start near query) | Gate irrelevant (search terminates in 3-5 pops) | Use fixed entry or global ef |
| Entry point in dense wrong cluster | Gate struggles — all local neighbors have similar coherence | Long-jump edges restore navigability |
| D=1 (1-dimensional space) | Coherence is binary (±1), may over-prune | Minimum D=4 recommended |

---

## Security and Governance Implications

- **Adversarial queries**: A carefully crafted query vector could manipulate the coherence gate to systematically avoid certain neighborhoods, potentially leaking which clusters exist. Mitigation: randomize the long-jump edge set per-session.
- **Proof-gated integration**: When used with `ruvector-verified`, the coherence threshold should be part of the verifiable computation record (witness log). A threshold of 0.50 means some true nearest neighbors may be returned with slightly lower confidence.
- **RAG safety**: In bounded RAG scenarios, coherence gating can complement access control: nodes with low coherence to the current context are also the most likely to be irrelevant, reducing the risk of context pollution.

---

## Edge and WASM Implications

The entire crate is `no_std`-compatible except for the benchmark binary (which uses `Instant`). The coherence computation is 3 dot products and 2 square-root calls — trivially WASM-safe.

On Cognitum Seed (Raspberry Pi Zero 2W, 512MB RAM, ~5W):
- 422 KB for 2,000-vector graph: trivially fits
- 7.5% fewer distance computations → 7.5% less CPU heat per query burst
- The adaptive variant's dynamic threshold overhead is ~3 comparisons/pop — negligible

For 100,000-vector edge deployments:
- Graph memory: ~21 MB (M=22, D=32)
- With coherence gating: estimated 5–15% reduction in active distance computation time (data-dependent)

---

## MCP and Agent Workflow Implications

An MCP-native vector search tool wrapping `ruvector-coherence-hnsw` would expose:

```json
{
  "tool": "vector_search",
  "params": {
    "query": "<embedding vector>",
    "k": 10,
    "ef": 80,
    "coherence_threshold": 0.50,
    "adaptive": true
  }
}
```

The coherence threshold becomes a per-call parameter, enabling agent orchestration layers (ruFlo, Claude Flow) to tune retrieval precision:
- High threshold → faster, lower recall → appropriate for rapid context building
- Low threshold (baseline) → full recall → appropriate for precise retrieval
- Adaptive → self-tuning → appropriate for unknown distribution queries

---

## Practical Applications

| Application | User | Why it matters | How RuVector uses it | Near-term path |
|-------------|------|----------------|----------------------|----------------|
| Agent memory retrieval | LLM agents in ruFlo | Fewer irrelevant memories = better context | Coherence-gated recall on agent memory graph | ruvector-core integration |
| Graph RAG | Enterprise search | Precision over recall in document retrieval | Gate prunes tangential document branches | ruvector-graph + coherence-hnsw |
| Semantic code search | Developers | Fast approximate search on code embeddings | Threshold-tuned recall control | ruvector-cli tool |
| Edge AI search | IoT/Cognitum Seed | Power budget: avoid unnecessary computations | WASM-compiled coherence kernel | micro-hnsw-wasm integration |
| Real-time anomaly detection | Security | Sub-millisecond latency on security event streams | Aggressive coherence gating for speed | ruvector-diskann + gate |
| MCP vector memory | Claude/agent tools | Agents query memories with tunable precision | mcp-gate wraps coherence search | mcp-brain integration |
| Scientific literature RAG | Researchers | Reduce noise in large-corpus vector search | High-threshold retrieval for precision | ruvector-server API param |
| Workflow automation | ruFlo pipelines | Self-optimizing retrieval parameters | Adaptive threshold as a ruFlo-controlled variable | ruFlo feedback loop |

---

## Exotic Applications

| Application | 10–20 year thesis | Required advances | RuVector role | Risk |
|-------------|------------------|-------------------|---------------|------|
| **Cognitum edge cognition** | Coherence-gated search as a neural-symbolic bridge: the gate models attention over memory | On-device embedding updates, coherence domain learning | Edge graph substrate + WASM kernel | Power constraints limit graph size |
| **RVM coherence domains** | Persistent coherence labels on edges enable "topic memory" within an agent session | RVM runtime support for domain-labeled graphs | ruvector-coherence + mincut partitioning | Label maintenance under concurrent writes |
| **Proof-gated AOS memory** | Agent operating systems with verifiable memory retrieval: every search has a coherence attestation | ruvector-verified + formal coherence proofs | Witness chain on coherence-gated results | Proof overhead may dominate search time |
| **Swarm memory coherence** | Thousands of agents share a coherence-annotated graph; each agent's search is gated by its current "focus coherence" | Distributed coherence graph with ruvector-raft | Replicated coherence graph, distributed gate | Coherence drift across replicas |
| **Synthetic nervous system memory** | Coherence gating mimics attention modulation in biological nervous systems: coherence with the "task direction" gates which memories are retrieved | Continuous embedding space aligned with neural representations | ruvector-nervous-system integration | Biological analogy may not hold quantitatively |
| **Self-healing memory graphs** | Detect and prune low-coherence edges: edges whose traversal coherence drops below a threshold are candidates for removal | Spectral coherence monitoring (ruvector-coherence spectral feature) | ruvector-mincut + coherence-hnsw | Graph repair may create disconnected components |
| **Bio-signal memory** | EEG/biosignal embeddings stored in coherence-gated graphs: retrieval is gated by physiological state coherence | Real-time biosignal embedding pipeline | ruvector-nervous-system + real-eeg integration | Biosignal noise creates false coherence signals |
| **Space autonomy memory** | Rover/satellite memory with coherence-gated retrieval for power-constrained environments (Mars: 300W budget) | Radiation-hardened WASM runtime | WASM kernel + deterministic coherence gate | Hardware WASM support uncertain |

---

## Deep Research Notes

### What the SOTA suggests

HNSW beam pruning is understudied. Most optimizations focus on:
- Index construction (DiskANN, NSG)
- Quantization (RaBitQ, PQ)
- Filtering (ACORN)

Traversal pruning during search is rarely studied. The closest work is FINGER[^6], which uses first-dimensional distance as an early rejection criterion for candidate distance computation — a related but orthogonal idea (FINGER prunes the *computation*, coherence gating prunes the *expansion*).

### What remains unsolved

1. **Threshold calibration**: The optimal threshold is dataset-dependent. For random unit vectors, 0.50 barely prunes anything. For highly clustered data, 0.70 might prune 30% of expansions. An automatic calibration procedure (perhaps using the first 100 queries to estimate the coherence distribution) is needed.

2. **Multi-layer HNSW integration**: This PoC uses a flat graph. In a true multi-layer HNSW, the entry for layer-0 comes from layer-1 greedy descent. The coherence should be computed from the layer-0 entry, not from the layer-1 starting point. The interface needs to thread the entry-point vector through the HNSW layers.

3. **Learned coherence**: Rather than using simple direction cosine, a learned coherence function (a small MLP mapping (entry_vec, candidate_vec, query_vec) → [0,1]) could be trained on retrieval trace data. This is the GNN-guided HNSW direction.

4. **Concurrent adaptation**: The adaptive threshold variant is per-query. A globally shared threshold (updated slowly with exponential moving average across all queries) might converge better and be amenable to ruFlo optimization.

### Where this PoC fits

This is a **Level 2 research result**: a working implementation demonstrating the mechanism, with honest measurements showing modest but real effects. It is not yet production-ready (missing: full HNSW multi-layer integration, automatic threshold calibration, concurrent access).

### What would make this production-grade

1. Integration into `ruvector-core`'s HNSW search path
2. Threshold calibration from query trace data
3. Concurrent-safe adaptive threshold (atomic float)
4. Benchmark on production-scale graphs (N=1M, D=768)
5. Comparison against FAISS IndexHNSW with the same graph
6. Recall-throughput Pareto curve across threshold values 0.0–0.9

### What would falsify the approach

- If coherence savings scale sub-linearly with N (the gate becomes irrelevant at scale)
- If learned coherence (GNN) outperforms direction cosine so much that direction cosine is not worth shipping
- If the 7.5% expansion savings disappears on production embeddings (non-isotropic, high-D)

---

## Production Crate Layout Proposal

```
crates/ruvector-coherence-hnsw/
├── Cargo.toml
├── src/
│   ├── lib.rs          — public API
│   ├── graph.rs        — flat navigable small-world graph
│   ├── coherence.rs    — direction coherence scoring
│   ├── search.rs       — Baseline, CoherenceGated, AdaptiveCoherence
│   ├── dataset.rs      — deterministic test data generation
│   ├── metrics.rs      — recall, latency stats
│   └── bin/
│       └── benchmark.rs — standalone benchmark binary
```

For production integration into `ruvector-core`:
```
crates/ruvector-core/src/
└── index/
    └── hnsw_coherence.rs  — CoherenceGate trait + impl, wired into HnswIndex::search_layer0
```

The threshold should be an optional parameter on `SearchParams`:
```rust
pub struct SearchParams {
    pub ef: usize,
    pub k: usize,
    pub coherence_threshold: Option<f32>,  // None = baseline
    pub adaptive_coherence: bool,
}
```

---

## What to Improve Next

1. **Integrate into `ruvector-core`**: Wire the coherence gate into the actual `hnsw_rs`-backed search in `ruvector-core`.
2. **Threshold sweep**: Run the benchmark across t ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8} and plot the recall-expansions Pareto frontier.
3. **Larger scale test**: N=50,000, D=128 (typical production embedding size) using approximate graph build (LSH or random projection tree) instead of brute-force.
4. **GNN coherence**: Replace the direction cosine with a 2-layer GNN readout that can capture non-linear coherence patterns in the graph.
5. **ruFlo integration**: A ruFlo workflow that periodically runs recall@10 probes and adjusts the coherence threshold via a feedback loop.
6. **RVF packaging**: Store the coherence threshold as a named field in the RVF index manifest so it travels with the index across deployment environments.

---

## References and Footnotes

[^1]: Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824–836. https://arxiv.org/abs/1603.09320. Accessed 2026-06-16.

[^2]: Jayaram Subramanya, S., Devvrit, F., Simhadri, H. V., Krishnawamy, R., & Kadekodi, R. (2019). DiskANN: Fast accurate billion-point nearest neighbor search on a single node. *NeurIPS 2019*. https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf. Accessed 2026-06-16.

[^3]: Patel, L., Kraft, P., Guestrin, C., & Zaharia, M. (2024). ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data. *SIGMOD 2024*. https://arxiv.org/abs/2403.04871. Accessed 2026-06-16.

[^4]: Gao, J., Long, C., Xu, J., & Yang, R. (2024). RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search. *SIGMOD 2024*. https://arxiv.org/abs/2405.12497. Accessed 2026-06-16.

[^5]: Zhang, Z., et al. (2022). SpFresh: Incremental In-Place Updating for Billion-Scale Vector Search. *SOSP 2023*. https://dl.acm.org/doi/10.1145/3600006.3613166. Accessed 2026-06-16.

[^6]: Jin, Y., et al. (2023). FINGER: Fast Inference for Graph-Based Approximate Nearest Neighbor Search. *WWW 2023*. https://dl.acm.org/doi/10.1145/3543507.3583318. Accessed 2026-06-16.
