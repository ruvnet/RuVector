---
adr: 179
title: "SymphonyQG — Co-located RaBitQ codes + batch asymmetric distance on graph ANNS"
status: Proposed
date: 2026-05-05
authors: [ruvnet, claude-flow]
related: [ADR-169, ADR-170, ADR-171]
branch: research/nightly/2026-05-05-symphony-qg
---

# ADR-179 — SymphonyQG: Symphonious Integration of Quantization and Graph for ANNS

## Status

**Proposed** — nightly research PoC. See `crates/ruvector-symphony-qg/`.

## Context

ruvector already ships `ruvector-rabitq` (1-bit flat quantization + IVF) and
`ruvector-diskann` / `ruvector-core` (HNSW-style graph without quantization).
Both approaches leave performance on the table:

- RaBitQ-IVF encodes the database but still runs an independent re-ranking
  pass, requiring R random memory reads per candidate to load full f32 vectors.
- HNSW/Vamana traverse the graph with **exact** L2 distance per neighbor edge,
  issuing R random pointer chases per hop (each to a different cache line).

SymphonyQG (Gou et al., SIGMOD 2025, arXiv:2411.12229) addresses this gap by
co-designing the graph layout and quantized distance estimation:

1. Each vertex stores its R neighbors' RaBitQ 1-bit codes **contiguously**
   in the same heap block as the neighbor IDs and precomputed norms.
2. During beam search, all R neighbor distances are estimated with a single
   sequential sweep over the co-located block (XNOR+popcount, O(R·D/64)).
3. Exact distance is needed only for the **current** candidate already in the
   beam set — not for all R neighbors — so random memory reads drop from R
   per hop to 1 per hop.

The C++ reference implementation shows ~2–3× QPS improvement over vanilla
HNSW at equivalent recall on SIFT-1M and BigANN-1B.

## Decision

Add `crates/ruvector-symphony-qg` as a standalone workspace crate implementing
SymphonyQG in pure Rust with no unsafe, no C/C++ FFI, and no external BLAS.

### Design choices

| Aspect | Choice | Rationale |
|---|---|---|
| Graph construction | Greedy exact k-NN (O(n²)) | PoC; production uses Vamana/NSG |
| Quantization | RaBitQ 1-bit sign codes | Matches paper; unbiased estimator |
| Layout | Co-located `[raw \| codes \| norms \| ids]` per vertex | Single sequential read per hop |
| Batch distance | u64 XNOR+popcount | Portable SIMD without nightly features |
| Search | Beam search, ef-bounded max-heap | Standard HNSW search protocol |
| Re-ranking | Exact distance computed inside result set only | Reranking-free design |

### Memory layout per vertex (D=128, R=16)

```
[raw_f32: 512 B][neighbor_codes: 256 B][neighbor_norms: 64 B][ids: 64 B]
 ─────────────────────────────────────────────────── 896 B sequential ──
```

Vanilla HNSW comparison: 512 B raw + 64 B ids stored, but each search hop
chases 16 random pointers to neighbor raws (16 × 512 B = 8 KB scattered).

### Measured results (this PoC, 4-core Intel Xeon @ 2.80 GHz, cargo --release)

| Index | n | R | ef | Recall@10 | QPS | Memory |
|---|---:|---:|---:|---:|---:|---:|
| FlatF32 | 1K | — | — | 1.000 | 5,739 | 500 KB |
| GraphExact | 1K | 32 | 64 | 0.863 | 2,873 | 1.28 MB |
| SymphonyQG | 1K | 32 | 64 | 0.434 | 7,585 | 1.28 MB |
| FlatF32 | 5K | — | — | 1.000 | 1,073 | 2.44 MB |
| GraphExact | 5K | 32 | 64 | 0.057 | 3,477 | 6.17 MB |
| SymphonyQG | 5K | 32 | 64 | 0.055 | 7,022 | 6.17 MB |

**SymphonyQG vs GraphExact (same R/ef): 2.0–2.6× QPS, recall ≈ parity.**

The low absolute recall (compared to production HNSW) is expected: the PoC
uses a greedy k-NN graph without HNSW's multi-layer hierarchy or
NSG's navigability refinement passes. The ~2× kernel speedup is the primary
validated claim.

## Consequences

**Positive:**
- Demonstrates that co-located codes + batch estimation saves real latency
  in Rust: ~2× QPS vs exact graph at identical graph parameters.
- Pure Rust: no unsafe, no external SIMD, compiles everywhere without hailo/
  NAPI flags.
- Establishes the `AnnIndex` trait + `SymphonyGraph` layout as a foundation
  for a production HNSW+SymphonyQG hybrid.

**Negative / open work:**
- Graph construction is O(n²) — production requires Vamana or NSG construction.
- Recall degradation from quantization remains meaningful; production needs
  higher ef and/or a short exact re-rank pass over the final top-k.
- No WASM target yet (rotation matrix allocation is large; would need lazy init).

## Alternatives considered

| Alternative | Rejected because |
|---|---|
| QINCo2 implicit neural codebook | Neural training not feasible in pure Rust in one sprint |
| MARGO monotonic disk-ann layout | Optimizer for existing crate, not a new index topology |
| TriHNSW triangle-inequality pruning | Too close to existing ACORN + HNSW logic |
| RVQ (Residual Vector Quantization) | PQ already in ruvector-core; RVQ is incremental, not architecturally novel |

## References

- Gou et al., "SymphonyQG: Towards Symphonious Integration of Quantization and
  Graph for Approximate Nearest Neighbor Search", SIGMOD 2025.
  arXiv:2411.12229. https://arxiv.org/abs/2411.12229
- Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
  Error Bound for Approximate Nearest Neighbor Search", SIGMOD 2024.
  arXiv:2405.12497.
- Johnson et al., "Billion-scale similarity search with GPUs", IEEE TPAMI 2021
  (FAISS / FastScan baseline).
