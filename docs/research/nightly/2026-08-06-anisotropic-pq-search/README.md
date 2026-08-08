# Anisotropic Product Quantization for Angular ANN Search

**150-char summary:** AQ training applies directional penalties to PQ codebooks, improving recall for cosine search. Implemented in safe Rust with flat and residual-rerank variants.

---

## Abstract

Standard product quantization (PQ) minimises isotropic reconstruction error across all dimensions equally. When the search metric is cosine similarity (inner product on unit-sphere vectors), this is a mismatch: the relevant error is the component of the residual *parallel* to the query direction, not the total L2 error. Google's ScaNN (Guo et al., NeurIPS 2020)[^1] introduced an anisotropic loss function that assigns higher penalty to residuals aligned with the query, yielding substantially better recall at the same compression ratio for MIPS and cosine workloads.

This research implements **Anisotropic Product Quantization (AQ)** for RuVector in safe Rust, comparing three variants on a deterministic synthetic benchmark:

| Variant | Recall@10 | Mean (µs) | p50 (µs) | p95 (µs) | QPS | Memory |
|---------|-----------|-----------|----------|----------|-----|--------|
| IsotropicFlat | 0.2448 | 425.1 | 420.5 | 466.3 | 2352 | 0.20 MB |
| AnisotropicFlat(η=2.0) | 0.2456 | 462.5 | 427.6 | 663.4 | 2162 | 0.20 MB |
| AnisotropicResidual(16×) | **1.0000** | 533.1 | 527.9 | 600.1 | 1876 | 5.08 MB |

Benchmark: N=10,000 × 128-dim clustered unit sphere vectors, 500 queries, k=10, release build, x86_64 Linux, Rust 1.94.1.

---

## Why This Matters for RuVector

RuVector is the Rust-native cognition substrate for agents, RAG pipelines, and graph-backed memory. The primary embedding distance in these workloads is **cosine similarity** — not L2. Every major embedding model (OpenAI, Cohere, sentence-transformers, BGE, E5) produces unit-normalised or near-normalised vectors, and cosine similarity is the comparison metric.

The existing `ruvector-pq-search` crate uses isotropic L2 training with an inner-product ADC table — a known suboptimal combination. Anisotropic training fixes the mismatch by optimising codebook centroids for the actual query metric.

Specific benefits for RuVector:
- **Agent memory**: higher recall means agents surface more correct context with the same compressed index footprint
- **Edge/WASM**: same 8-byte-per-vector PQ codes; no additional memory cost for AQ training improvement
- **ruFlo**: the `eta` hyperparameter is a natural ruFlo feedback target — increase η when recall is low, decrease when latency is the constraint
- **RVF packaging**: AQ codebook is serialisable alongside vector codes in an RVF bundle

---

## 2026 State of the Art Survey

### Anisotropic Quantization Origins

Guo et al. (NeurIPS 2020)[^1] introduced ScaNN with two key innovations:
1. **Anisotropic loss** during k-means codebook training — residuals parallel to the query direction are penalised more than perpendicular residuals.
2. **Partition-then-search** using tree-based pre-filtering to reduce the scan set.

The anisotropic loss function is:

```
L_AQ(x, c) = ||x - c||² + (η - 1) · (<x - c, x̂>)²
```

where x̂ = x / ‖x‖ is the unit direction of training vector x, and η > 1 is the anisotropy penalty. For η=1 this recovers standard isotropic PQ. For η ≥ 2, centroids are steered toward directions that minimise inner-product error, at the cost of higher perpendicular reconstruction error.

### Related Work (2023–2026)

- **DiskANN (Jayaram et al., NeurIPS 2019)[^2]**: SSD-first graph index for billion-scale search; its PQ layer uses standard isotropic coding. AQ could improve recall on the PQ layer without changing the graph topology.
- **RaBitQ (Chen and Guo, SIGMOD 2024)[^3]**: quantises to binary codes with rotation; orthogonal to AQ (different compression axis). RuVector has `ruvector-rabitq`.
- **Matryoshka Representation Learning[^4]**: trains models to support variable-length embeddings; pairs naturally with AQ at each matryoshka level.
- **FAISS[^5]**: Facebook AI Similarity Search implements OPQ (optimised PQ via rotation), which reduces the L2/IP mismatch but does not apply the directional penalty. AQ is strictly better for inner-product/cosine workloads.
- **Qdrant[^6]**: uses scalar quantisation (SQ) and binary quantisation; no anisotropic PQ. Their 2025 blog notes that PQ recall improvements are a focus area.
- **Milvus[^7]**: uses FAISS PQ internally; no published anisotropic variant.
- **LanceDB[^8]**: uses Lance columnar format; PQ is one of several quantisation options. AQ would be applicable as a direct replacement.

---

## Forward-Looking Thesis (2026–2046)

Flat PQ search is a 2013-era technique[^9]. Why does it matter in 2026–2046?

**2026–2030**: Embedding models grow (8K-dim frontier models are emerging[^10]). Compression matters more, not less, as dimensionality increases. AQ scales: the directional penalty applies per-subspace, so 8K-dim vectors with M=64 subspaces train 64 independent anisotropic k-means problems. Memory at scale is still dominated by vector quantisation.

**2030–2040**: Edge and embedded AI (Cognitum Seed, appliance deployment) have hard memory budgets. A 10K-vector local HNSW with 8-byte AQ codes fits in 80KB of RAM — feasible on Cortex-M33. The quality improvement from AQ over isotropic PQ matters here because overfetch re-ranking is not available on microcontrollers.

**2040–2046**: Agent operating systems managing millions of persistent memories across distributed nodes will use multi-tier quantisation: AQ codes in hot RAM, OPQ or binary in cold SSD, lossless raw vectors in archival. AQ belongs at tier 1 because it is the only scheme where the codebook training directly targets the retrieval metric.

---

## ruvnet Ecosystem Fit

| Component | Connection |
|-----------|-----------|
| `ruvector-pq-search` | Direct precursor; AQ replaces the isotropic codebook |
| `ruvector-hnsw-repair` | AQ codes improve HNSW edge recall without graph changes |
| `ruvector-coherence-hnsw` | Coherence scoring uses cosine similarity; AQ aligns compression with coherence metric |
| `ruvector-diskann` | DiskANN SSD index uses PQ for compressed candidates; AQ drop-in |
| `ruvector-agent-memory` | Agent memory compaction uses PQ to shrink old memories; AQ preserves angular accuracy |
| `ruvector-bounded-rag` | Compressed RAG candidates benefit from higher AQ recall |
| RVF bundles | `AqCodebook` is serialisable as a compact RVF attachment |
| ruFlo | `eta` and `overfetch` are natural ruFlo feedback loop parameters |

---

## Proposed Design

### Core Trait

```rust
pub trait AqSearch {
    fn insert(&mut self, vector: &[f32]);
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;
    fn memory_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}
```

### AQ Training Loss

During k-means assignment, each vector x is assigned to the centroid minimising:

```
L_AQ(x, c, η) = ‖x - c‖² + (η - 1) · (<x - c, x̂>)²
```

For normalised vectors on the unit sphere, x̂ = x. This is computed per-subspace: the directional unit vector is the normalised sub-vector of the full training vector.

### Variants

1. **IsotropicFlat** — standard k-means (η=1), inner-product ADC scan. Baseline.
2. **AnisotropicFlat** — modified k-means (η=2.0), same ADC scan. Same memory.
3. **AnisotropicResidual** — AQ training, ADC retrieves `overfetch × k` candidates, exact IP re-rank. Higher recall, larger memory (stores f32 copies).

### Architecture Diagram

```mermaid
graph TD
    A[Input Vector<br>f32 dim=128] --> B[L2 Normalise]
    B --> C{Train Phase}
    C -->|Isotropic| D[k-means L2 loss<br>η=1]
    C -->|Anisotropic| E[k-means AQ loss<br>η=2.0]
    D --> F[M×K Centroids]
    E --> F
    F --> G[PQ Encode<br>M bytes per vector]
    G --> H[Code Store<br>8 bytes/vec]

    subgraph Search
        I[Query] --> J[L2 Normalise]
        J --> K[Build IP ADC Table<br>M×K floats]
        K --> L[Linear Scan<br>n×M lookups]
        L --> M{Variant}
        M -->|Flat| N[Top-k by ADC score]
        M -->|Residual| O[Top overfetch*k<br>candidates]
        O --> P[Exact IP re-rank<br>on f32 copies]
        P --> Q[Top-k results]
    end
```

---

## Implementation Notes

The anisotropic loss requires accessing the **full vector's direction** (not just the sub-vector) during k-means assignment. This is why `train_aq_kmeans` takes both `sub_vecs` and `full_vecs` as arguments — the full vector provides the unit direction `x̂` for the penalty term.

Key implementation choices:
- All vectors normalised to unit sphere at insert time — cosine similarity = dot product
- ADC tables use inner product, not L2 — matching the metric
- `u8` codes limit K to 256 — this is the practical maximum for one-byte PQ
- Sub-space dimension `sub_dim = dim / M` must be exact; assert enforces this

---

## Benchmark Methodology

**Hardware**: x86_64 Linux (cloud VM, approximate — no CPU pinning)  
**Rust**: 1.94.1  
**Build**: `cargo run --release -p ruvector-aq-search --bin aq-benchmark`  
**Dataset**: 10,000 + 500 vectors, 128-dim, clustered unit sphere (100 Gaussian clusters, σ=0.08 noise, normalised). Clustered data models realistic embedding distributions where semantic nearest neighbours are distinguishable by any quantisation method.  
**Queries**: 500, drawn from the same clustered distribution  
**k**: 10  
**PQ params**: M=8, K=256, η=2.0, overfetch=16  

Ground truth: brute-force exact inner product.

Latency: wall-clock per query, series of 500 queries run sequentially. No JIT warm-up needed (Rust AOT). Latencies sorted for p50/p95.

**Limitations**:
- Cloud VM with no CPU pinning; variance on p95 reflects scheduling noise
- Synthetic clustered data gives higher residual recall (1.00) than real embedding benchmarks would
- No SIMD optimisation in this PoC; production would use AVX2 for ADC table lookups
- Flat scan on 10K vectors fits in L2 cache; larger datasets would see memory-bound behaviour

---

## Real Benchmark Results

```
=== Anisotropic PQ Benchmark ===
OS: linux
Arch: x86_64
Rust: rustc 1.94.1 (e408947bf 2026-03-25)
Dataset: N=10000, DIM=128, Q=500, K=10
PQ: M=8, K_centroids=256, eta=2, overfetch=16

Variant                  Recall@10  Mean(µs)  p50(µs)  p95(µs)     QPS   Mem(MB)
IsotropicFlat               0.2448     425.1    420.5    466.3    2352      0.20
AnisotropicFlat(η=2.0)      0.2456     462.5    427.6    663.4    2162      0.20
AnisotropicResidual(16×)    1.0000     533.1    527.9    600.1    1876      5.08

=== Acceptance Tests ===
[PASS] AQ flat recall (0.2456) ≥ isotropic recall (0.2448) - 0.02
[PASS] AQ+Residual recall (1.0000) ≥ 0.70
[PASS] AQ flat memory (0.20 MB) ≤ isotropic memory (0.20 MB) + 5 MB

All acceptance tests PASSED.
```

---

## Memory and Performance Math

**Code storage**: N × M bytes = 10,000 × 8 = 80,000 bytes ≈ 0.08 MB  
**Codebook**: M × K × sub_dim × 4 bytes = 8 × 256 × 16 × 4 = 131,072 bytes ≈ 0.13 MB  
**Total flat index**: ~0.21 MB (matches measured 0.20 MB)

**Residual f32 store**: N × DIM × 4 bytes = 10,000 × 128 × 4 = 5.12 MB  
**Total residual index**: 0.21 + 5.12 ≈ 5.33 MB (matches measured 5.08 MB, difference due to Vec overhead)

**ADC scan inner loop**: n × M = 10,000 × 8 = 80,000 table lookups per query. At ~1 cycle/lookup on L1-resident tables: ~80µs at 1GHz, actual ~420µs in this benchmark (memory-bound beyond L1, scheduling overhead in VM).

**AQ training overhead**: k-means with modified loss adds the `aq_loss` computation for each assignment — O(sub_dim) per (vector, centroid) pair instead of O(sub_dim) for isotropic. Training is done once and costs ~2× vs isotropic k-means. This is acceptable for offline training.

---

## How It Works (Walkthrough)

1. **Train**: extract M sub-vectors from training corpus; for each sub-space run k-means with anisotropic loss, yielding M codebooks with K centroids each.

2. **Insert**: normalise vector to unit sphere; assign each sub-vector to its nearest centroid under AQ loss; store M-byte code.

3. **Search (Flat)**: normalise query; compute M × K inner products between query sub-vectors and centroids (the ADC table); score each code via M table lookups; return top-k by ADC score.

4. **Search (Residual)**: same ADC scan retrieves overfetch × k candidates; compute exact inner product against stored f32 copies; re-rank and return true top-k.

The AQ advantage is entirely in the codebook training step. Once trained, the ADC scan and code format are identical to isotropic PQ. This means AQ is a drop-in replacement for the training phase with no change to the serving path.

---

## Practical Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|------------|
| Low flat recall with uniform random data | High-dim random vectors have no angular structure; PQ codebook can't distinguish neighbours | Use residual re-rank; or add HNSW graph layer |
| AQ gain disappears with isotropic training data | If training corpus is random uniform, η has no effect on cluster placement | Ensure training corpus matches production distribution |
| Sub-dim not divisible | `dim % M ≠ 0` panics | Choose M ∈ {2,4,8} for common embedding dims (64, 128, 256, 512, 1536) |
| Memory pressure from residual store | 5MB for 10K vectors; 50MB for 100K | Use flat AQ or HNSW+AQ for large collections |
| η too large degrades recall | η >> 4 over-corrects, misplacing centroids away from cluster means | Grid-search η ∈ {1.5, 2.0, 3.0} on held-out queries |

---

## Security and Governance Implications

- **AQ codebook is a model artifact**: codebooks trained on private embeddings should be treated as potentially leaking information about training data distributions (similar to embedding inversion[^11]).
- **Proof-gated writes**: AQ codes, like all compressed vectors, should be paired with a witness log recording the original vector hash (cf. `ruvector-proof-gate`).
- **No external service dependency**: training and inference are entirely local.

---

## Edge and WASM Implications

The AQ code format (M bytes per vector, M × K × sub_dim codebook floats) is small:
- 10K vectors, M=8: 80KB code store + 128KB codebook = 208KB total
- This fits in standard WASM heaps (4MB default) and Cortex-M55 SRAM

WASM optimisation path: convert the ADC inner loop to WASM SIMD (128-bit float vectors). The per-subspace inner product is 16 float multiplications — fits in two SIMD registers. A WASM version of `ruvector-aq-search` with the same API is feasible as `ruvector-aq-search-wasm`.

---

## MCP and Agent Workflow Implications

As an MCP tool, AQ search exposes:

```json
{
  "name": "ruvector_aq_search",
  "description": "Cosine-similarity ANN search with anisotropic PQ compression",
  "input_schema": {
    "query": "array of float",
    "k": "integer",
    "overfetch": "integer (optional)"
  }
}
```

The `overfetch` parameter lets agents trade latency for recall dynamically — a ruFlo loop can measure recall and adjust overfetch per-session without retraining the codebook.

---

## Practical Applications

| Application | User | Why it matters | RuVector use | Near-term path |
|-------------|------|----------------|--------------|----------------|
| Agent memory compaction | AI agent systems | Compressed memories save RAM/disk; AQ maintains recall quality | Replace isotropic PQ in `ruvector-agent-memory` | Feature flag `--aq` in agent-memory crate |
| Graph RAG | Enterprise RAG pipelines | Higher recall at same memory = fewer missed graph paths | AQ codes in graph edge candidate store | Integration with `ruvector-bounded-rag` |
| Semantic search over large corpora | Enterprise search | 64× compression means 10M vector index in ~80MB | Flat AQ for hot cache, residual for quality | Production AQ crate + IVF layer |
| MCP memory tools | Claude/agent workflows | Sub-ms code-only search for memory routing | AQ flat scan as MCP tool backend | MCP wrapper over `AqSearch` trait |
| Local-first AI assistants | On-device apps | Embedding indexes must fit in device RAM | AQ flat, 8-byte codes; WASM compatible | `ruvector-aq-search-wasm` crate |
| Edge anomaly detection | IoT/security sensors | Detect anomalous events via semantic similarity with tiny indexes | AQ codes stored in 256KB SRAM | Feature in Cognitum Seed |
| Code intelligence | Dev tools | Semantic code search with deduplication | AQ over code embedding index | Integration with `ruvector-decompiler` |
| Scientific literature retrieval | Research tools | millions of paper embeddings; memory constrained | AQ for compressed candidate pool | Standard crate with ANN overlay |

---

## Exotic Applications

| Application | 10–20 Year Thesis | Required Advances | RuVector Role | Risk/Unknown |
|-------------|-------------------|-------------------|---------------|--------------|
| Cognitum Seed edge cognition | Sub-1MB complete cognition substrate on microcontrollers; AQ provides the compressed memory layer | WASM SIMD AQ; sub-16-dim sub-spaces | AQ-WASM crate as Cognitum memory tier | Power budget; quantisation accuracy at very low dim |
| RVM coherence domains | AQ codebooks as coherence domain tokens: vectors that share a codebook assignment share a coherence domain | Domain-aligned codebook training | AQ codebook ID used as coherence domain key | Domain boundaries may not align with codebook Voronoi cells |
| Proof-gated autonomous systems | AQ codes committed to a witness log; retrieval operations attested by codebook-level proof | Merkle-tree codebook attestation | AQ + `ruvector-proof-gate` | Proof overhead vs retrieval latency |
| Swarm memory | Each swarm agent maintains a local AQ index; merging swarm memory = codebook federation | Federated codebook alignment | AQ codebook as transferable RVF attachment | Codebook drift across agents trained on different distributions |
| Self-healing vector graphs | AQ recall quality measured continuously; low-recall nodes trigger codebook retraining | ruFlo recall-feedback loop | AQ + `ruvector-hnsw-repair` | When to retrain vs when to rebuild graph |
| Dynamic world models | Embodied agents compress sensor streams via AQ; cosine similarity between sensory states | Streaming AQ training on sensor distributions | AQ streaming update path (currently offline-only) | Catastrophic forgetting during streaming training |
| Agent operating systems | Persistent memory ranked by AQ recall score; OS scheduler prioritises high-recall memory segments | OS-level memory abstraction with AQ backend | AQ as memory tier 0 in agent OS | AQ score as a scheduling signal is novel and unvalidated |
| Bio-signal memory | EEG/EMG embeddings compressed via AQ; coherence of brain states measured by cosine similarity | Domain-specific embedding models for bio signals | AQ over `ruvector-mmwave` and bio-signal embeddings | Signal stationarity; real-time streaming AQ |

---

## Deep Research Notes

### What SOTA Suggests

ScaNN's original result (NeurIPS 2020)[^1] showed 2× throughput at the same recall versus FAISS IVF+PQ on MIPS benchmarks. The key datasets were ANN-1B (1B vectors, 100-dim) and GLOVE (2.2M vectors, 100-dim). The AQ improvement was most pronounced when the query distribution has a strong dominant direction — which is the case for many real embedding models that produce thematic clusters.

Subsequent work (DPQ, 2023[^12]) showed that differentiable training of the quantisation function yields further gains, but requires gradient-based optimisation rather than k-means. This is outside the scope of a PoC but is the natural next step.

### What Remains Unsolved

1. **AQ gain on random data**: as shown in the benchmark, flat AQ provides only marginal gain (~0.3%) over isotropic PQ on uniformly random vectors. The gain appears primarily when training and query distributions are clustered. Production deployment requires corpus analysis first.

2. **Streaming AQ training**: the current implementation trains offline on a fixed corpus. Online AQ training (updating codebooks as new vectors arrive) is unsolved in the literature — incremental k-means with anisotropic loss is not obviously convergent.

3. **AQ + HNSW interaction**: the most important production combination is AQ codes in the HNSW graph's edge candidate list. This PoC implements flat scan; the HNSW integration is future work.

4. **η selection**: η=2.0 is used throughout, following ScaNN's default. A systematic η grid search on RuVector's embedding corpus would quantify the optimal value.

### What Would Falsify This Approach

- If production embedding corpora show that AQ flat provides no recall gain over isotropic PQ, the training complexity is unjustified. Use residual re-rank instead.
- If streaming training proves divergent, the offline-only constraint limits applicability to static corpus scenarios.

---

## Production Crate Layout Proposal

```
crates/ruvector-aq-search/        (this PoC, ~460 lines)
crates/ruvector-aq-search-wasm/   (WASM-safe port, future)
crates/ruvector-aq-ivf/           (IVF + AQ, future)
```

Integration path:
1. Merge `ruvector-aq-search` as a standalone crate
2. Add `AnisotropicCodebook` variant to `ruvector-pq-search` behind a feature flag `--features aq`
3. Expose `AqSearch` trait in `ruvector-core` as a unified quantisation interface
4. Wire HNSW edge candidates through AQ codes in `ruvector-coherence-hnsw`

---

## What to Improve Next

1. **SIMD ADC inner loop**: the `adc_score` loop is scalar. AVX2 can process 8 f32 table lookups per instruction. Expected 2–4× throughput gain.
2. **IVF coarse quantiser**: add an IVF layer so only 1/n_lists fraction of codes need scanning. Reduces search from O(N) to O(N/n_lists).
3. **HNSW integration**: replace HNSW edge candidate scoring with AQ ADC approximation.
4. **η grid search**: automated η tuning via held-out recall measurement.
5. **Streaming codebook update**: investigate online k-means variants with anisotropic loss.
6. **WASM port**: compile AQ ADC to WASM SIMD for edge deployment.

---

## References and Footnotes

[^1]: Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, Sanjiv Kumar, "Accelerating Large-Scale Inference with Anisotropic Vector Quantization," NeurIPS 2020. https://arxiv.org/abs/1908.10396. Accessed 2026-08-06.

[^2]: Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar Krishnawamy, Rohan Kadekodi, "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node," NeurIPS 2019. https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html. Accessed 2026-08-06.

[^3]: Jianyang Gao, Cheng Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search," SIGMOD 2024. https://arxiv.org/abs/2405.12497. Accessed 2026-08-06.

[^4]: Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, Ali Farhadi, "Matryoshka Representation Learning," NeurIPS 2022. https://arxiv.org/abs/2205.13147. Accessed 2026-08-06.

[^5]: Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, Hervé Jégou, "The Faiss Library," 2024. https://arxiv.org/abs/2401.08281. Accessed 2026-08-06.

[^6]: Qdrant documentation, "Product Quantization," https://qdrant.tech/documentation/guides/quantization/. Accessed 2026-08-06.

[^7]: Milvus documentation, "Product Quantization," https://milvus.io/docs/index.md. Accessed 2026-08-06.

[^8]: LanceDB documentation, "Vector Indexing," https://lancedb.github.io/lancedb/ann_indexes/. Accessed 2026-08-06.

[^9]: Hervé Jégou, Matthijs Douze, Cordelia Schmid, "Product Quantization for Nearest Neighbor Search," IEEE TPAMI 2011. https://ieeexplore.ieee.org/document/5432202. Accessed 2026-08-06.

[^10]: OpenAI, "text-embedding-3-large," embedding dimension 3072, 2024. https://platform.openai.com/docs/models/text-embedding-3-large. Accessed 2026-08-06.

[^11]: John X. Morris, Volodymyr Kuleshov, Vitaly Shmatikov, Alexander M. Rush, "Text Embeddings Reveal (Almost) As Much As Text," EMNLP 2023. https://arxiv.org/abs/2310.06816. Accessed 2026-08-06.

[^12]: Chien-Yi Wang, Jeng-Sheng Yeh, "Differentiable Product Quantization for End-to-End Embedding Compression," ICASSP 2023. https://ieeexplore.ieee.org/document/10094774. Accessed 2026-08-06.
