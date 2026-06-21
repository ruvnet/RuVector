# ADR-264: Matryoshka-Aware Coarse-to-Fine Vector Search

**Status:** Proposed  
**Date:** 2026-06-21  
**Author:** Nightly research agent  
**Crate:** `crates/ruvector-matryoshka`  
**Branch:** `research/nightly/2026-06-21-matryoshka-coarse-fine`

---

## Context

Every major embedding model deployed in 2026 is trained with Matryoshka Representation Learning (MRL): OpenAI text-embedding-3, Nomic nomic-embed-text-v2, Voyage 4, Cohere v4, Jina v5, Gemini Embedding 2. MRL ensures that any prefix of the full-dim vector is a meaningful lower-dimensional representation. This property is almost universally available in agent memory workloads today.

RuVector's current HNSW implementation in `ruvector-core` and `ruvector-coherence-hnsw` operates on full-dimension vectors. It does not use MRL prefix structure to reduce the cost of graph traversal. Every query pays the full-dim distance cost for every traversal hop, even when a coarse-dim estimate would suffice for candidate generation.

The academic literature (AdANNS NeurIPS 2023, Panorama arXiv:2510.00566, Milvus funnel search) demonstrates that a two-stage coarse-to-fine pipeline consistently achieves 1.5–90× speedup depending on the dimension reduction factor and the quality of the MRL structure.

This ADR proposes adding a `ruvector-matryoshka` crate that implements and benchmarks three ANN search variants: FullDimHNSW (baseline), TwoStage (coarse HNSW + full-dim rerank), and ThreeStage (coarse → mid → full funnel).

---

## Decision

**Adopt** the matryoshka coarse-to-fine search pattern as a first-class RuVector capability.

Specifically:
1. The `ruvector-matryoshka` crate enters the workspace as a standalone PoC.
2. The `Searcher` trait and `MatryoshkaConfig` struct define the public API contract.
3. The three variants (FullDimHNSW, TwoStage, ThreeStage) are implemented, tested, and benchmarked.
4. Production integration into `ruvector-core` is gated on a real-corpus evaluation (MTEB or MS MARCO).

---

## Consequences

**Positive:**
- TwoStage achieves **1.61× lower search latency** than FullDimHNSW at recall@10 = 0.903 (versus 1.000 for full-dim HNSW) on 3,000 × 128-dim MRL-structured synthetic data.
- ThreeStage achieves recall@10 = **0.947** at latency parity with FullDimHNSW.
- Build time drops **3×** (440 ms vs 1,285 ms) because the coarse HNSW is 4× smaller.
- Coarse index (32 dims, N=1000) is ~256 KB — fits in WASM and Cognitum Seed memory.

**Negative:**
- TwoStage loses −9.7 pp recall vs. full-dim HNSW on synthetic data. On real MRL corpora the gap is typically smaller (1–4 pp at 50% dims per published benchmarks).
- ThreeStage uses more memory (3,000 KB vs 1,875 KB for FullDimHNSW) because it stores coarse + mid + full arrays.
- The approach is only effective on MRL-trained embeddings. For legacy non-MRL embeddings, coarse-dim recall can be near-random and the funnel actively hurts.

---

## Alternatives Considered

**A. Single-stage full-dim HNSW (status quo)**  
Current approach. Simple, correct. No benefit from MRL prefix structure.  
*Rejected because:* leaves significant latency savings on the table for MRL workloads.

**B. Product Quantization (PQ) compression**  
Compress vectors to 8–16 bytes using codebooks; compute distances in compressed space.  
*Rejected for this ADR:* RaBitQ was already researched (nightly 2026-04-23). PQ adds a training step and a codebook dependency. Matryoshka funnels require zero index-time training.

**C. FINGER-style angle-based distance skipping**  
Skip HNSW distance computations using angle estimates from a low-rank basis.  
*Not rejected:* complementary to this ADR. FINGER could be applied within the coarse HNSW stage for additional speedup. Deferred to next iteration.

**D. Adaptive per-query dimension selection (arXiv:2602.03306)**  
Per-query classification of optimal truncation depth.  
*Not rejected:* orthogonal enhancement. Deferred to SONA integration once coarse-to-fine is production-stable.

---

## Implementation Plan

### Phase 1 (Complete): Standalone PoC

- [x] `crates/ruvector-matryoshka` added to workspace
- [x] `Searcher` trait with `build` and `search` methods
- [x] `FullDimIndex`, `TwoStageIndex`, `ThreeStageIndex` implementations
- [x] Deterministic synthetic dataset generator (`dataset.rs`)
- [x] Minimal HNSW with configurable working dimension (`hnsw.rs`)
- [x] 7 passing unit tests with numeric recall thresholds
- [x] Benchmark binary with real timing and recall output

### Phase 2: Production Integration

- [ ] Evaluate on real MRL corpora (Nomic embed v2 on MTEB-Retrieval subset)
- [ ] Add FINGER-style traversal pruning within coarse HNSW stage
- [ ] Integrate `MatryoshkaConfig` into `ruvector-core::search` module
- [ ] Add `matryoshka` feature flag to `ruvector-core/Cargo.toml`
- [ ] SONA integration: learn optimal `(coarse_dim, candidates)` per workload
- [ ] WASM build target for edge deployment

### Phase 3: Ecosystem Wiring

- [ ] `ruvector-agent-memory`: use TwoStage by default for MRL-trained embeddings
- [ ] `ruvector-proof-gate`: proof-gated stage transition (public coarse → private full-dim rerank)
- [ ] `rvf`: carry multiple-resolution matryoshka indexes in a single `.rvf` manifest
- [ ] `mcp-brain`: expose `matryoshka_search` as an MCP tool

---

## Benchmark Evidence

**Hardware:** x86_64, Linux 6.18.5, rustc 1.94.1  
**Dataset:** 3,000 × 128-dim synthetic MRL-structured vectors, 200 queries, k=10, ef=64  
**Command:** `cargo run --release -p ruvector-matryoshka --bin benchmark`

| Variant | Recall@10 | Mean (μs) | p50 (μs) | p95 (μs) | QPS | Mem (KB) |
|---------|-----------|-----------|---------|---------|-----|--------|
| FullDimHNSW | 1.000 | 168.4 | 164 | 216 | 5,939 | 1,875 |
| TwoStage | 0.903 | 104.8 | 98 | 142 | 9,541 | 2,250 |
| ThreeStage | 0.947 | 163.1 | 151 | 232 | 6,130 | 3,000 |

All three variants PASS acceptance tests.

---

## Failure Modes

| Mode | Detection | Response |
|------|-----------|----------|
| Non-MRL embedding corpus | `prefix_captures_cluster_structure` test fails (coarse recall < 0.4) | Disable funnel; fall back to FullDimHNSW |
| Candidate pool too small | Observed recall@k < 0.85 in production monitoring | Increase `two_stage_candidates`; `sona` can auto-tune |
| Model update changes MRL alignment | Recall drops on next eval | Rebuild coarse HNSW; ruFlo trigger on model version change |
| Memory pressure on small devices | ThreeStage: 3,000 KB vs 1,875 KB | Use TwoStage; drop mid_vecs |

---

## Security Considerations

- Coarse-dim vectors stored in the HNSW reveal less about the original embedding than full-dim vectors. A breach of the coarse index leaks less information.
- Proof-gated stage transition (Phase 3) enables namespace-scoped retrieval: coarse search over public namespace, full-dim rerank only with namespace capability token.
- The prefix projection (`&v[..coarse_dim]`) is non-invertible to the full vector, providing a limited form of embedding privacy by dimension reduction.

---

## Migration Path

This ADR introduces a new crate with no breaking changes to existing crates. Migration is opt-in:

1. Add `ruvector-matryoshka` as an optional dep to `ruvector-core`
2. Enable via `--features matryoshka`
3. Use `MatryoshkaConfig::default_128()` or tune to specific embedding model
4. Run `prefix_captures_cluster_structure` to verify MRL alignment quality
5. Compare recall@k in shadow mode before switching production traffic

---

## Open Questions

1. **What is the recall of TwoStage on real Nomic-embed v2 vectors on BEIR/MTEB?** AdANNS reports 0.5–1.5% higher accuracy vs rigid IVF, but this is for IVF not HNSW. We expect better than our synthetic results on real MRL corpora.

2. **What is the optimal coarse_dim for different embedding models?** text-embedding-3-large: 256 dims coarse from 3072? Nomic-embed-v2: 64 from 768? This needs empirical tuning per model.

3. **Can SONA learn the funnel schedule from agent query traces without an offline calibration dataset?**

4. **Should the coarse HNSW be built on L2-normalised prefix vectors (current) or on raw prefix vectors?** L2-normalisation makes distances comparable across prefix lengths; raw vectors preserve directional information. For cosine-similarity corpora, normalisation is correct.

5. **Does the proof-gated stage transition need a formal security proof or is capability-token gating sufficient for production?**
