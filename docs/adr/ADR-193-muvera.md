---
adr: 193
title: "Add ruvector-muvera: Multi-Vector Retrieval via Fixed Dimensional Encodings (MUVERA, NeurIPS 2024)"
status: proposed
date: 2026-05-08
authors: [ruvnet, claude-flow]
related: [ADR-160, ADR-161, ADR-162]
tags: [multi-vector, late-interaction, colbert, fde, hnsw, retrieval, nlp]
---

# ADR-193 — Add ruvector-muvera: Multi-Vector Retrieval via FDE

## Status

**Proposed.**

## Context

ruvector currently supports single-vector approximate nearest-neighbor (ANN) search via HNSW, DiskANN, hyperbolic HNSW, and filtered variants. All existing indexes assume one float vector per document.

Modern dense retrieval for natural language search increasingly relies on **late-interaction models** — principally ColBERT and its derivatives — that produce one float vector per token rather than one per document. A 200-token document yields ~200 vectors at 128D each (25,600 floats). Scoring a query with 16 tokens against a 1-million-document corpus requires computing MaxSim(Q, D) = (1/|Q|) ∑_q max_d ⟨q,d⟩ for every document: approximately **16 × 200 × 10⁶ = 3.2 billion dot products** per query. This is several orders of magnitude above what brute-force single-vector search requires.

The standard production solution, PLAID (CIKM 2022), addresses this via centroid-inverted indexing and multi-stage pruning, but requires bespoke infrastructure incompatible with ruvector's single-vector index API.

MUVERA (NeurIPS 2024, arXiv:2405.19504) offers an orthogonal approach: a preprocessing step that **reduces each multi-vector document to a single Fixed Dimensional Encoding (FDE)** whose inner product provably approximates MaxSim. After FDE encoding, standard MIPS — including ruvector's existing HNSW index — applies directly with no infrastructure changes.

The MUVERA paper demonstrates:
- 93% of ColBERT v2 nDCG@10 on MS MARCO Passage at 10ms latency (vs. PLAID's 120ms).
- HNSW-based retrieval with FDE achieves 37.1 nDCG@10 vs. 39.7 for PLAID at 2ms latency — a 60× speedup with 6.6% quality reduction.

No Rust crate in the ruvector workspace currently implements FDE or any late-interaction multi-vector primitive.

## Decision

We introduce `crates/ruvector-muvera` as a new workspace member implementing:

1. **`FdeEncoder`** — holds an R×D random projection matrix; deterministic given a seed. Implements `encode(token_vecs) -> Vec<f32>` (FDE vector of length R×D).

2. **`MultiVecIndex` trait** — common interface for all retrieval variants:
   ```rust
   fn build(docs: Vec<Vec<Vec<f32>>>, encoder: Arc<FdeEncoder>) -> Result<Self, MuveraError>;
   fn search(&self, query_vecs: &[Vec<f32>], k: usize) -> Result<Vec<SearchResult>, MuveraError>;
   fn memory_bytes(&self) -> usize;
   fn name(&self) -> &'static str;
   ```

3. **`BruteForceMaxSim`** — exact O(n·|Q|·|D|·d) MaxSim baseline; ground truth for recall evaluation.

4. **`FlatFdeIndex`** — FDE encoding at build time; flat IP scan at query time. O(n·R·D) per query. 9.5x faster than BruteForce at n=500.

5. **`HnswFdeIndex`** — FDE encoding at build time; greedy single-level HNSW at query time. 42x faster than BruteForce at n=10K (131 vs. 3 QPS). Production version should use multi-level HNSW.

All implementations pass `cargo test -p ruvector-muvera` (11 tests) and `cargo build --release -p ruvector-muvera`.

Benchmark results (Intel Xeon @ 2.10 GHz, release build):

| Variant | n_docs | QPS | Build (ms) | Mem (KB) |
|---------|--------|-----|------------|----------|
| BruteForceMaxSim | 10,000 | 3 | 74 | 160,000 |
| FlatFDE | 10,000 | 14 | 2,441 | 320,000 |
| HnswFDE | 10,000 | 131 | 75,306 | 320,625 |

Note: HnswFDE build time is dominated by the O(n²) greedy construction over high-dimensional (R×D = 8,192-dim) FDE vectors. A future ADR will replace this with hierarchical HNSW.

## Consequences

### Positive

- ruvector can now serve ColBERT, PLAID, and other late-interaction retrieval models natively.
- The `MultiVecIndex` trait is backend-agnostic: any future MIPS index (IVF, HNSW with multi-layers, RaBitQ-FDE) can be plugged in without changing user code.
- `FdeEncoder` is serializable (plain Vec<f32>) and deterministic, enabling reproducible index builds.
- No new dependencies added (rand, rand_distr, thiserror already in workspace).
- 11 unit tests verify correctness of encoding, error handling, recall on structured data.

### Negative

- FDE memory overhead is R×D per document, which is larger than raw token storage when R ≥ T (tokens per doc). Users must tune R ≤ T for memory efficiency.
- FDE recall on random/unstructured embeddings is poor (by design — the algorithm requires semantic structure). Users must use quality language-model embeddings.
- The HnswFDE build in this PoC is O(n²) and too slow for production at n > 5K with high-dimensional FDE. A hierarchical HNSW implementation is required (tracked in future ADR).
- FDE approximation quality is empirically well-studied only for ColBERT-family embeddings; behavior with arbitrary embedding models is untested.

## Alternatives considered

### A — PLAID-compatible inverted index

Implement centroid-based inverted indexing compatible with PLAID's exact algorithm. This would give the highest recall but requires a fundamentally different index architecture (inverted postings over centroid IDs, multi-stage scoring pipeline). Estimated 4–6 weeks of engineering; not compatible with ruvector's `AnnIndex` trait. Rejected as too invasive for a PoC ADR.

### B — Per-token HNSW with late reranking

Build one HNSW over all individual token vectors across all documents. At query time, search for top-K individual token matches, then group by document ID and compute MaxSim for the top-G documents (reranking). This avoids FDE encoding but requires O(n·T) HNSW nodes (e.g., 200M nodes for 1M docs × 200 tokens), making build and memory infeasible. Rejected.

### C — Matryoshka Representation Learning (MRL-HNSW)

Multi-granularity embeddings (NeurIPS 2022) for adaptive-dimension query serving. Addresses a different use case (single-vector, multiple precision levels) and does not solve the multi-vector retrieval problem. Consider for a future ADR.

### D — EMVB binary FDE

Binary FDE (Boros et al., arXiv:2404.02805) bit-encodes each FDE component, reducing memory 32x and enabling SIMD popcount IP. This is an extension of MUVERA rather than an alternative; planned as a follow-on to this crate (see "What to improve next" in the research doc).

## References

- MUVERA paper: arXiv:2405.19504 (NeurIPS 2024)
- Research doc: docs/research/nightly/2026-05-08-muvera/README.md
- Crate: crates/ruvector-muvera/
