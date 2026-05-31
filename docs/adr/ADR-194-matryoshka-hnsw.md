# ADR-194: Matryoshka HNSW — Dimension-Adaptive Multi-Resolution Vector Search

**Status:** Draft  
**Date:** 2026-05-16  
**Authors:** ruvnet, claude-flow  
**Deciders:** RuVector core team  
**Related:** ADR-193 (RAIRS IVF), ADR-026 (model routing), crates/ruvector-matryoshka

---

## Context

Matryoshka Representation Learning (MRL, arXiv:2205.13147, NeurIPS 2022) has become
a de-facto training standard for production embedding models.  OpenAI text-embedding-3,
Nomic nomic-embed-text-v1.5, Google Gemini Embedding 2, Voyage AI, Jina, and BGE-M3
all ship Matryoshka-trained vectors.  Every agentic workflow that retrieves from these
APIs would benefit from Matryoshka-aware indexing.

RuVector currently offers:
- HNSW via `ruvector-acorn` and `ruvector-core`
- IVF via `ruvector-rairs`
- 1-bit quantization via `ruvector-rabitq`

There is no Matryoshka-aware search strategy: no cascade from coarse to full
dimensions, no multi-resolution index, and no trait that captures the concept of
"this index understands that early dimensions are more discriminative."

The cascade strategy — coarse-dimension linear scan → full-precision rerank of
top candidates — is the simplest correct approach.  It is already implemented in
production by Milvus (called "funnel search") and supported conceptually in Weaviate
and Qdrant through model-provider truncation.  RuVector has no Rust-native equivalent.

---

## Decision

Add `crates/ruvector-matryoshka` to the workspace, providing:

1. A `MatryoshkaIndex` trait for dimension-adaptive search.
2. Three concrete implementations: `FullScan` (baseline), `CoarseScan` (fast/lossy),
   `CascadeSearch` (Matryoshka-aware cascade).
3. A `MatryoshkaConfig` struct parameterising `full_dim`, `coarse_dim`, and
   `cascade_candidates`.
4. A synthetic dataset generator that produces Matryoshka-like cluster geometry,
   enabling deterministic benchmarks without external embedding dependencies.
5. A benchmark binary (`matryoshka-bench`) producing all key metrics.

This crate is initially a research PoC behind no feature flag.  The `MatryoshkaIndex`
trait is the API surface that should survive into production.

---

## Consequences

### Positive

- Enables correct retrieval from MRL-trained models (OpenAI, Nomic, etc.) without
  accepting the recall collapse of truncation-only search.
- Establishes a clean Rust trait (`MatryoshkaIndex`) that can be implemented by
  graph-based coarse stages (HNSW-lite) in future iterations.
- 2.28× throughput improvement over FullScan with identical recall@10 on Matryoshka-
  structured data (measured, `cargo run --release`).
- Coarse-only variant (`CoarseScan`) is trivially WASM-compatible (no rayon, no
  unsafe, no external deps); opens WASM-budget search for Cognitum Seed and Pi Zero.

### Negative

- Recall depends on `cascade_candidates` being large enough.  A misconfigured value
  silently degrades recall.  Users must validate on representative data.
- Flat coarse scan is O(N·D_c); for N > 1M a graph-based coarse stage is needed
  (HNSW on the coarse vectors).
- Dimension-split vector layout (separate coarse and residual arrays) would recover
  cache efficiency but is not yet implemented; measured speedup (2.28×) is below
  the theoretical op-count speedup (3.45×).

---

## Alternatives considered

### A. Truncation at query time without a cascade (status quo)

Truncate query and database vectors to `coarse_dim` before existing flat/HNSW search.
Simple but collapses recall.  On our test dataset, D=32 truncation gives 5.75%
recall@10 vs the full-precision ground truth — unusable for production.

### B. Multiple full-dim HNSW graphs at each granularity

Build one HNSW graph per dimension level (e.g., at D=32, D=64, D=128).  Higher
recall than cascade for the coarse-graph query.  Rejected for now: 3× memory
overhead, complex build coordination, not yet required for the PoC.

### C. Integrate directly into `ruvector-core`

Add CascadeSearch as a new index type in core.  Rejected for initial landing:
- Core has its own stability guarantees.
- A standalone crate allows faster iteration without risking core breakage.
- Migration path is clear: implement `MatryoshkaIndex` in core after the trait
  stabilises.

---

## Implementation plan

### Phase 1 — PoC (this ADR, done)

- [x] `MatryoshkaIndex` trait
- [x] `FullScan`, `CoarseScan`, `CascadeSearch` implementations
- [x] Synthetic dataset generator with shared cluster geometry
- [x] 8 unit tests, all passing
- [x] Benchmark binary with real latency, throughput, recall, memory
- [x] Acceptance test: CascadeSearch recall@10 ≥ 0.90

### Phase 2 — Graph coarse stage

- [ ] Implement `HnswCoarseStage` that builds an HNSW graph at `coarse_dim`
- [ ] Replace O(N·D_c) flat pass with O(log N) HNSW walk on coarse graph
- [ ] Expected: push throughput from 2.28× toward the 3.45× theoretical target

### Phase 3 — Production integration

- [ ] Dimension-split vector layout: separate `coarse` and `residual` storage arrays
- [ ] Feature flag `matryoshka` in `ruvector-core` exposing `MatryoshkaIndex` in search registry
- [ ] ruFlo plugin for online `cascade_candidates` tuning against recall SLA
- [ ] MCP tool surface: `mcp_search_cascade(query, coarse_dim, k)`

### Phase 4 — DiskANN integration

- [ ] Store coarse vectors in RAM, full vectors on SSD (bridge to `ruvector-diskann`)
- [ ] WASM build of `CoarseScan` for edge deployment

---

## Benchmark evidence

All numbers from `cargo run --release -p ruvector-matryoshka`, x86-64 Linux 6.18.5,
Intel Celeron N4020, rustc 1.87.0:

```
N=5 000 vectors, D=128, coarse_dim=32, cascade_candidates=200, K=10, 200 queries

Variant                  Mean(µs)  p50(µs)  p95(µs)   QPS  Recall@10  Mem(KB)
─────────────────────────────────────────────────────────────────────────────
FullScan (D=128)            860.7    840.5    990.4  1 162     1.0000    2 500
CoarseScan (D=32)           332.1    325.7    382.9  3 012     0.0575    2 500
CascadeSearch (D=32→128)    376.9    371.5    419.8  2 653     1.0000    2 500

Acceptance: CascadeSearch recall@10 = 1.0000 ≥ 0.90 → PASS ✓
```

---

## Failure modes

| Mode | Description | Detection | Mitigation |
|------|-------------|-----------|------------|
| Silent recall collapse | `cascade_candidates` too small; ground-truth neighbours not in coarse top-C | Monitor recall@k in production | Instrument recall; alert if < SLA |
| No embedding MRL property | Model not MRL-trained; coarse dims uninformative | Pre-check: coarse recall < 20% on validation set | Fall back to `FullScan` |
| Memory exhaustion | N × D × 4 bytes exceeds device RAM | OOM at build time | Use disk-backed variant or quantize |
| Latency regression on large N | Flat coarse scan O(N·D_c) too slow for N > 1M | Throughput drops below SLA | Graduate to HNSW coarse stage (Phase 2) |

---

## Security considerations

- No new network surface introduced.
- Coarse candidates could, in principle, leak information about which embeddings
  are "close in the low-dimensional projection" even if not close in full space.
  If embedding privacy is a concern, restrict coarse-pass candidate lists to
  authorised callers.
- For proof-gated RAG (ADR future), require a witness proof before the full rerank
  stage can access the full-precision vectors.

---

## Migration path

1. Existing callers using `FullScan` semantics continue to work unchanged.
2. Callers wishing to adopt cascade search: wrap existing `Vec<Vector>` in
   `CascadeSearch::new(config)` + `build()` + `search()` — same interface.
3. No existing crate APIs change.

---

## Open questions

1. **Optimal `cascade_candidates` scheduling.** Should it be a function of N, K,
   and estimated cluster density?  Current choice (200) is empirical.
2. **Dimension-split layout.** How to expose both coarse and residual arrays via a
   single `Vector` struct without breaking the existing API?
3. **HNSW coarse stage thread safety.** Phase 2 graph construction needs `Send +
   Sync`; current PoC is single-threaded.
4. **Query-aware dimension selection.** arXiv:2602.03306 shows per-query `coarse_dim`
   outperforms a global constant.  Should `search()` accept a per-query `coarse_dim`
   override?
5. **Integration with `ruvector-mincut`.** MinCut boundaries could prune candidates
   that are in a different coherence domain from the query after the coarse pass,
   further reducing the rerank set and improving precision.
