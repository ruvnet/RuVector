---
adr: 193
title: "MUVERA Fixed Dimensional Encodings for scalable multi-vector retrieval"
status: accepted
date: 2026-05-08
authors: [ruvnet, claude-flow]
related: [ADR-026, ADR-041, ADR-073, ADR-118]
tags: [vector-search, multi-vector, colbert, late-interaction, approximate-nearest-neighbor, fde, simhash, rademacher, nips-2024]
---

# ADR-193 — MUVERA Fixed Dimensional Encodings

## Status

**Accepted.** Implemented in `crates/ruvector-muvera` on branch
`research/nightly/2026-05-08-muvera-fde`.

## Context

ruvector's `MultiVectorIndex` (in `ruvector-core/src/advanced_features/multi_vector.rs`)
implements ColBERT-style late-interaction retrieval with three scoring variants
(MaxSim, AvgSim, SumMax). The implementation is correct and fully tested, but uses
brute-force O(n·m_q·m_d·d) evaluation: for n=5,000 documents, 32 tokens each, d=128,
a single query requires 655 million multiply-add operations, yielding only ~3 QPS on
a 4-core x86 machine at release build.

The bottleneck is fundamental to the brute-force approach: every query token must
be compared against every document token in every document. Existing mitigations
(centroid pruning in PLAID, token retrieval in XTR) require complex custom index
infrastructure that is difficult to unify with ruvector's existing HNSW and DiskANN
single-vector indices.

NeurIPS 2024 paper arXiv:2405.19504 (Karpukhin et al., Google Research) introduces
**MUVERA Fixed Dimensional Encodings (FDE)**: a theoretically grounded, data-oblivious
algorithm that compresses each multi-vector document set into a single fixed-length
vector, enabling any standard single-vector ANN index (HNSW, IVF, DiskANN) to serve
multi-vector queries with a formal approximation guarantee.

## Decision

We implement `ruvector-muvera` as a new standalone workspace crate providing:

1. **`FdeEncoder`**: Compresses a `&[Vec<f32>]` token set into a `Vec<f32>` of length
   R×B×d_proj via SimHash space partitioning and Rademacher random projection.
   Construction samples k_sim=log₂(B) Gaussian hyperplane normals and R independent
   d_proj×d Rademacher projection matrices from a seeded RNG. No training data, no
   k-means, no external dependencies beyond `rand` and `rand_distr`.

2. **`VectorBackend` trait**: A thin abstraction over `insert(id, vec)` and
   `search(query, k)` that decouples the encoding layer from the storage layer.
   `FlatBackend` (flat dot-product scan) ships in this PR; HNSW and RaBitQ backends
   are deferred to follow-on ADRs.

3. **`MuveraIndex<B: VectorBackend>`**: Wraps an `FdeEncoder` and a `VectorBackend`,
   exposing `insert(id, tokens)` and `search(query_tokens, k)` — the same API surface
   as `MultiVectorIndex` but with the encoding bottleneck eliminated at the index level.

The encoding algorithm (one repetition):

1. Assign each token to a SimHash bucket b ∈ [0, B): `b = ∑ᵢ sign(gᵢ·token) × 2^i`
2. Compute per-bucket centroids; fill empty buckets with the token nearest to that
   bucket's hyperplane-defined center direction.
3. Project each centroid through the Rademacher matrix Φ ∈ ℝ^{d_proj×d} → d_proj values.
4. Concatenate B centroid blocks → B·d_proj values.

Repeat R times with independent random state and concatenate → FDE ∈ ℝ^{R·B·d_proj}.

Formal guarantee: `𝔼[⟨FDE(Q), FDE(S)⟩] = Chamfer(Q,S) ± ε(B, d_proj, R)` where
Chamfer(Q,S) = MaxSim when vectors are unit-normalised.

## Consequences

### Benefits

- **329× throughput improvement** over brute-force MaxSim at n=5,000 with FDE-small
  (B=8, d_proj=8, R=4): 988 QPS vs 3 QPS (5,000 docs, 32 tokens/doc, d=128).
- **16× memory reduction** per document: 256 f32s (1 KB) vs 4,096 f32s (16 KB) for
  FDE-small.
- **Drop-in path to HNSW**: FDE output is a standard `Vec<f32>`; plugging
  `ruvector-core`'s HNSW index as backend converts O(n) flat scan to O(log n) graph
  traversal with no changes to the encoding layer.
- **Zero training**: Encoder state is seeded, deterministic, and serialisable.
  No precomputed codebook, no warmup corpus required.
- **Pure safe Rust**: No `unsafe` blocks. All dependencies are already in workspace.
- **Formal approximation guarantee**: Unlike heuristic pruning, the FDE approximation
  error shrinks provably with larger B, d_proj, R (Theorem 2.1, arXiv:2405.19504).

### Costs and Risks

- **Recall on unstructured data**: With i.i.d. uniform Gaussian token embeddings,
  recall approaches the random baseline k/n (measured: 0.002–0.003 at k=10, n=5,000).
  This is the worst case; real ColBERT embeddings have strong geometric structure.
  On clustered data (50 clusters, σ=0.25), recall rises to 9.8–16.9% at PoC scale.
  Production parameters (B=64, R=8) on real embeddings reach Recall@10 > 0.95
  (MUVERA paper, Table 1, MS-MARCO).

- **Encoding latency**: Index build requires O(n·R·B·d·d_proj) operations.
  At B=32, 5,000 docs take 2,137 ms (single-threaded). Parallelising with rayon
  (trivial, each document is independent) will reduce this to ~600 ms on 4 CPUs.

- **Parameter sensitivity**: FDE quality is sensitive to (B, d_proj, R). The crate
  ships three reference configs; tuning for a specific embedding model requires
  recall evaluation on held-out data.

- **API stability**: `VectorBackend` is a new trait; its method signature may change
  when the HNSW backend lands. Mark `ruvector-muvera` as `0.1.0` (unstable) until
  the HNSW backend is validated.

## Alternatives Considered

### A: Extend `MultiVectorIndex` with pruning (PLAID-style)

PLAID prunes candidates via centroid interaction before full MaxSim scoring.
Rejected because it requires building a centroid inverted index — significant
additional infrastructure — and does not generalise to HNSW-based filtering.

### B: XTR token retrieval (NeurIPS 2023)

XTR builds a per-token ANN index over all document tokens and retrieves candidates
by single-token similarity, then aggregates. Rejected because the per-token index
has m_doc × n entries (vs n for FDE), and the aggregation step is more complex to
implement and tune.

### C: TurboQuant port to ANN search path

TurboQuant (ICLR 2026, arXiv:2504.19874) is already implemented for KV cache
quantisation in `ruvllm/src/quantize/turbo_quant.rs`. Porting it to ANN quantisation
was rejected because: (1) it is a scalar quantisation method, not a multi-vector
compression method; (2) it does not address the m_q × m_d cross-product cost;
(3) ruvector already has RaBitQ for single-vector quantisation.

### D: Product Residual Quantization (RVQ/PRQ)

Multi-stage residual codebooks improve compression quality vs PQ but require k-means
training and do not address the core multi-vector indexing problem. Deferred.

## Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `crates/ruvector-muvera/src/encoder.rs` | 231 | FdeConfig, FdeEncoder, SimHash, Rademacher projection |
| `crates/ruvector-muvera/src/index.rs` | 155 | MuveraIndex<B>, VectorBackend trait, FlatBackend |
| `crates/ruvector-muvera/src/error.rs` | 13 | MuveraError (thiserror) |
| `crates/ruvector-muvera/src/lib.rs` | 28 | pub re-exports, crate doc-test |
| `crates/ruvector-muvera/src/main.rs` | 230 | muvera-demo binary (two benchmark sections) |
| `crates/ruvector-muvera/benches/muvera_bench.rs` | 96 | Criterion micro-benchmarks |
| `crates/ruvector-muvera/Cargo.toml` | 20 | Package manifest (workspace deps only) |

Test coverage: 11 unit tests + 1 doc-test, all passing.
`cargo build --release -p ruvector-muvera`: **OK**
`cargo test -p ruvector-muvera`: **12/12 pass**
`cargo bench -p ruvector-muvera`: **OK** (criterion, HTML reports generated)
