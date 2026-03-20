# Algorithms & State of the Art

**Document ID**: spectral-sparsification/01-algorithms-sota
**Date**: 2026-03-19
**Status**: Research Complete
**Classification**: Algorithmic Survey — Dynamic Spectral Sparsification
**Series**: [00](./00-executive-summary.md) | **01** | [02](./02-rust-crate-design.md) | [03](./03-ruvector-integration.md) | [04](./04-companion-systems.md)

---

## 1. Static Spectral Sparsification

### 1.1 Foundations: The Laplacian Guarantee

A spectral sparsifier H of graph G is a sparse subgraph such that for all vectors x:

```
(1-ε) · xᵀ L_G x  ≤  xᵀ L_H x  ≤  (1+ε) · xᵀ L_G x
```

This is strictly stronger than cut sparsification (which only requires the guarantee for x ∈ {0,1}ⁿ). Spectral sparsifiers preserve cuts, effective resistances, mixing times, conductance, and solutions to Laplacian systems simultaneously.

### 1.2 Spielman-Teng (2004)

The near-linear time Laplacian solver revolution began with Spielman and Teng's Õ(m) solver for SDD systems. Their construction used hierarchical graph decomposition and low-stretch spanning trees to build spectral sparsifiers with O(n log^c n) edges for a large constant c.

**Key insight**: If you can solve Laplacian systems fast, you can build sparsifiers fast, and vice versa — the two problems are equivalent up to polylog factors.

### 1.3 Spielman-Srivastava (2008/2011)

The breakthrough simplification: sample edges proportional to their **effective resistance**.

**Theorem** (Spielman-Srivastava): For any weighted graph G = (V, E, w) and ε > 0, sample q = O(n log n / ε²) edges, where each edge e is chosen with probability proportional to w_e · R_eff(e). Reweight each sampled edge by the inverse of its sampling probability. The resulting graph H is a (1±ε) spectral sparsifier with high probability.

The key subroutine is computing approximate effective resistances in Õ(m) time using:
1. Solve L⁺ for random vectors via a fast Laplacian solver
2. Apply Johnson-Lindenstrauss projection to compress the pseudoinverse
3. Query approximate R_eff(u,v) = ‖Z(χ_u - χ_v)‖² in O(log n) time

**Sparsifier size**: O(n log n / ε²) edges — near-optimal for general graphs.

### 1.4 Batson-Spielman-Srivastava (2009)

The BSS barrier-method construction achieves optimal size:

**Theorem** (BSS): Every graph has a (1±ε) spectral sparsifier with O(n/ε²) edges. This is tight.

The construction is deterministic but runs in O(mn³/ε²) time — impractical for large graphs. The theoretical significance is that linear-size sparsifiers exist.

### 1.5 Practical Algorithms

| Algorithm | Year | Size | Construction Time | Practical? |
|-----------|------|------|-------------------|-----------|
| Spielman-Teng | 2004 | O(n log^c n / ε²) | Õ(m) | Partially |
| Spielman-Srivastava | 2008 | O(n log n / ε²) | Õ(m / ε²) | Yes |
| BSS | 2009 | O(n / ε²) | O(mn³/ε²) | No |
| Lee-Sun | 2017 | O(n / ε²) | Õ(m / ε²) | Partially |

---

## 2. Fully Dynamic Spectral Sparsification

### 2.1 ADKKP — The First Polylog Dynamic Algorithm (FOCS 2016)

**Paper**: Abraham, Durfee, Koutis, Krinninger, Peng. "On Fully Dynamic Graph Sparsifiers." FOCS 2016.

**Results**:
- (1±ε)-spectral sparsifier with **amortized** update time poly(log n, ε⁻¹)
- (1±ε)-cut sparsifier with **worst-case** update time poly(log n, ε⁻¹)
- Sparsifier size: n · poly(log n, ε⁻¹)

**Key technique**: t-bundle spanners.

The algorithm maintains a hierarchical decomposition:
1. **Backbone**: A sparse always-present structure (union of spanning forests) that guarantees global connectivity
2. **Layered sampling**: Edges are categorized into O(log n) levels based on their effective resistance estimate
3. **Per-level sparsification**: Each level maintains its own sparsifier independently

When an edge is inserted/deleted:
- Update the backbone (O(log n) via link-cut trees)
- Recompute resistance estimates for affected edges
- Adjust sampling at affected levels

The sparsifier has **arboricity** polylog n, meaning it can be decomposed into polylog n spanning forests. This structural property enables efficient subsequent algorithms.

### 2.2 Bernstein et al. — Adaptive Adversary (ICALP 2022)

**Paper**: "Fully-Dynamic Graph Sparsifiers Against an Adaptive Adversary." ICALP 2022.

Standard dynamic algorithms assume an oblivious adversary (updates don't depend on algorithm's random choices). This paper handles the **adaptive** adversary model:

- The adversary sees the algorithm's output and chooses future updates accordingly
- Achieves poly(log n, ε⁻¹) amortized update time
- Uses differential privacy techniques to hide internal state

**Significance for RuVector**: Real workloads are adaptive — query patterns depend on previous results. This result validates that dynamic sparsifiers work even under adversarial conditions.

### 2.3 Khanna-Li-Putterman — Dynamic Hypergraphs (STOC 2025)

**Paper**: "Near-optimal Linear Sketches and Fully-Dynamic Algorithms for Hypergraph Spectral Sparsification." STOC 2025.

**Results**:
- Fully dynamic (1±ε) spectral hypergraph sparsification
- Amortized update time: Õ(r · polylog(m) / ε²) where r is the maximum hyperedge rank
- Sparsifier size: Õ(n · polylog(m) / ε²) hyperedges

**Why it matters**: Hypergraph sparsification captures multi-way relationships (e.g., a document relevant to multiple queries simultaneously). For RuVector's multi-modal retrieval, this is directly relevant.

### 2.4 Zhao — Dynamic Directed Graphs (July 2025)

**Paper**: "Fully Dynamic Spectral and Cut Sparsifiers for Directed Graphs." arXiv:2507.19632.

**Results**:
- First polylog dynamic spectral sparsification for **directed** graphs
- Introduces **degree-balance preserving spectral approximation** (preserves in-degree minus out-degree)
- Amortized update time: O(ε⁻² · polylog(n))
- Sparsifier size: O(ε⁻² n · polylog(n))
- Also handles adaptive adversary for O(polylog(n))-partially symmetrized graphs

**Why it matters for RuVector**: Similarity graphs with asymmetric similarity (e.g., "A is similar to B" but not vice versa) are directed. This extends sparsification to asymmetric retrieval.

### 2.5 Forster-Goranci-Momeni — Directed Hypergraphs (STACS 2026)

**Paper**: "Fully Dynamic Spectral Sparsification for Directed Hypergraphs." STACS 2026. arXiv:2512.21671.

**Results**:
- First spectral sparsification for **directed hypergraphs**
- Amortized update time: O(r² log³ m)
- Sparsifier size: O(n² / ε² · log⁷ m)
- Also supports **parallel batch-dynamic**: k updates processed with O(kr² log³ m) work and O(log² m) depth

**Significance**: The batch-dynamic result enables efficient GPU parallelization of sparsifier maintenance.

---

## 3. Dynamic Kernel Graph Sparsification

### 3.1 Kernel Graphs and Embeddings

A kernel graph K(P, σ) for points P ⊂ ℝᵈ with kernel σ has edge weights σ(pᵢ, pⱼ). Common kernels:
- **Gaussian RBF**: σ(p,q) = exp(-‖p-q‖²/2h²)
- **Cosine**: σ(p,q) = p·q / (‖p‖‖q‖)
- **Polynomial**: σ(p,q) = (p·q + c)^k

RuVector's similarity graphs are kernel graphs — each vector embedding is a point, and similarity scores are kernel evaluations.

### 3.2 Recent Dynamic Results

Recent work shows that for geometric/kernel graphs:
- Spectral sparsifiers can be maintained under **point updates** (embedding moves) in n^{o(1)} time
- The same framework gives dynamic sketches for **Laplacian multiplication** and **Laplacian solving**
- An embedding update (vector reindexing) is literally a point move in the kernel graph

This is unusually relevant to RuVector because:
1. Vector updates = point moves in the kernel graph
2. The dynamic sparsifier maintains structural faithfulness across these moves
3. Laplacian operations on the sparsifier enable cheap graph-level analytics

### 3.3 Implications

The kernel graph result connects three things:
- **Dynamic similarity search** (RuVector's core operation)
- **Dynamic spectral sparsification** (the shadow graph)
- **Dynamic Laplacian algebra** (cheap analytics on the shadow)

Together, they mean that every time you update an embedding, the compressed world model updates cheaply and you can immediately run Laplacian-based analytics (spectral clustering, diffusion, coherence scoring) on the sparsifier.

---

## 4. Streaming Spectral Sparsification

### 4.1 Semi-Streaming Model

In the streaming model, edges arrive one at a time and the algorithm must maintain a sparsifier using O(n polylog n) space — sublinear in the number of edges.

**Kapralov et al.** showed single-pass spectral sparsification in O(n log² n / ε²) space using linear sketches of the incidence matrix.

### 4.2 Dynamic Streaming

In the **dynamic streaming** model (turnstile), edges can be both inserted and deleted:
- Kapralov: O(n log² n / ε²) space suffices
- Uses ℓ₂-heavy hitter sketches on random projections of the incidence matrix

**Relevance**: Streaming sparsification enables WASM deployment where memory is constrained. RuVector's ruvector-solver-wasm can leverage streaming sketches for memory-bounded environments.

---

## 5. Comprehensive Comparison

| Year | Authors | Scope | Update Time | Sparsifier Size | Adversary | Key Technique |
|------|---------|-------|-------------|-----------------|-----------|---------------|
| 2004 | Spielman-Teng | Static undirected | Õ(m) construction | O(n log^c n / ε²) | N/A | Low-stretch trees |
| 2008 | Spielman-Srivastava | Static undirected | Õ(m / ε²) construction | O(n log n / ε²) | N/A | Effective resistance sampling |
| 2009 | BSS | Static undirected | O(mn³/ε²) construction | O(n / ε²) optimal | N/A | Barrier method |
| 2016 | ADKKP | Dynamic undirected | poly(log n, ε⁻¹) amortized | n · poly(log n, ε⁻¹) | Oblivious | t-bundle spanners |
| 2020 | Kapralov et al. | Dynamic streaming | O(n polylog n / ε²) space | O(n log² n / ε²) | Oblivious | ℓ₂ sketches |
| 2022 | Bernstein et al. | Dynamic undirected | poly(log n, ε⁻¹) amortized | n · poly(log n, ε⁻¹) | **Adaptive** | Differential privacy |
| 2025 | Khanna-Li-Putterman | Dynamic hypergraphs | Õ(r · polylog(m) / ε²) | Õ(n · polylog(m) / ε²) | Oblivious | Linear sketches |
| 2025 | Zhao | Dynamic directed | O(ε⁻² · polylog(n)) | O(ε⁻²n · polylog(n)) | Partially adaptive | Degree-balance preservation |
| 2026 | Forster-Goranci-Momeni | Dynamic directed hyper | O(r² log³ m) | O(n²/ε² · log⁷ m) | Oblivious | Batch-dynamic parallel |

---

## 6. Open Problems

### 6.1 Worst-Case vs. Amortized

All fully dynamic spectral sparsifiers achieve polylog **amortized** update time. Worst-case polylog updates exist only for **cut** sparsifiers (ADKKP 2016). The question of worst-case polylog spectral updates remains open for general graphs.

### 6.2 Lower Bounds

No non-trivial lower bounds on dynamic spectral sparsification update time are known. The best lower bound is Ω(log n) from cell-probe complexity.

### 6.3 Practical Implementation Gap

Despite the mature theory:
- **No open-source implementation exists** in any language for fully dynamic spectral sparsification
- The constant factors in polylog update times are large (estimated 10-100x overhead)
- Practical implementations need to trade theoretical guarantees for engineering pragmatism

The approach taken in `ruvector-sparsifier` — theory-inspired but audit-driven — bridges this gap by using the theoretical framework for correctness verification while employing practical heuristics for speed.

---

## Sources

- [Spielman & Srivastava, "Graph Sparsification by Effective Resistances" (SIAM 2011)](https://arxiv.org/abs/0803.0929)
- [Batson, Spielman, Srivastava, "Spectral sparsification: theory and algorithms" (CACM 2013)](https://dl.acm.org/doi/10.1145/2492007.2492029)
- [Abraham, Durfee, Koutis, Krinninger, Peng, "On Fully Dynamic Graph Sparsifiers" (FOCS 2016)](https://arxiv.org/abs/1604.02094)
- [Bernstein et al., "Fully-Dynamic Graph Sparsifiers Against an Adaptive Adversary" (ICALP 2022)](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ICALP.2022.20)
- [Khanna, Li, Putterman, "Near-optimal Fully-Dynamic Algorithms for Hypergraph Spectral Sparsification" (STOC 2025)](https://arxiv.org/html/2502.03313)
- [Zhao, "Fully Dynamic Spectral and Cut Sparsifiers for Directed Graphs" (2025)](https://arxiv.org/abs/2507.19632)
- [Forster, Goranci, Momeni, "Fully Dynamic Spectral Sparsification for Directed Hypergraphs" (STACS 2026)](https://arxiv.org/abs/2512.21671)
- [Kapralov, "Single Pass Spectral Sparsification in Dynamic Streams"](https://theory.epfl.ch/kapralov/papers/dsparse.pdf)

---

## Document Navigation

- **Previous**: [00 - Executive Summary](./00-executive-summary.md)
- **Next**: [02 - Rust Crate Design](./02-rust-crate-design.md)
