# ADR-265: RuVector Comprehensive Benchmark Suite

**Status**: Accepted  
**Date**: 2026-06-21  
**Authors**: Claude Code MetaHarness Architect  
**Supersedes**: None  
**Related**: ADR-128 (SOTA Gap Implementations), ADR-266 (MetaHarness Darwin Mode), ADR-267 (SOTA Validation Protocol)

---

## Context

RuVector is a production vector database with 10+ optimization modules (HNSW, RaBitQ, Matryoshka, Product Quantization, Hybrid Search, LSM-ANN, HNSW Repair, DiskANN, ColBERT, KV-Cache Compression, MLA). Each module makes specific performance claims:

- **RaBitQ**: 512× compression, 0.75-0.92 recall@10
- **DiskANN**: billion-scale SSD-backed search, <5ms latency
- **Matryoshka**: 4-12× faster search, <2% recall loss
- **Hybrid (BM25+ANN)**: 20-49% retrieval improvement
- **LSM-ANN**: 150K insert/s streaming performance
- **ColBERT**: per-token late-interaction SOTA retrieval

**Current State**: Benchmarks are fragmented across Rust benches, Python scripts, and JSON results. No continuous validation against public leaderboards (ANN-Benchmarks, BEIR, VectorDBBench, MTEB).

**Problem Statement**: Without a unified, reproducible, audited benchmark suite:
1. Cannot claim SOTA status with scientific rigor
2. Performance regressions go undetected
3. Users cannot verify claims
4. Darwin Mode evolution has nowhere to score candidates

---

## Decision

Implement a **5-phase comprehensive benchmark suite** measuring RuVector against public leaderboards with:
- Unified measurement across 10+ modules
- Scoring function for Darwin Mode evolution
- Signed audit trails (ADR-267) for SOTA validation
- CI/CD integration with daily smoke tests

### Measurement Categories

| Category | Datasets | Metrics | Baseline | Target |
|----------|----------|---------|----------|--------|
| **ANN Recall/QPS** | SIFT1M, GIST1M, GloVe | recall@1/10/100, QPS, memory, p99 | Top-5 ANN-Benchmarks | Beat top-3 on 2+ metrics |
| **Compression** | SIFT1M, GloVe | recall@10 vs memory | ScaNN, FreshDiskANN | 512× with ≥0.9 recall |
| **Latency** | SIFT1M | p50/p99/p99.9 | Qdrant, Milvus | <2ms p99 |
| **Streaming** | Synthetic | insert rate | LanceDB, Fresh-DiskANN | 150K insert/s |
| **Embedding Quality** | BEIR (11) + MTEB (11) | NDCG@10, MRR, MAP | DPR, E5-large-v2 | ≥0.45 NDCG@10 on NQ |

### Scoring Function for Darwin Mode

```
score = 0.4 * recall@10_norm 
      + 0.3 * log(QPS/baseline_QPS)
      + 0.2 * (1 - min(1, memory/baseline_memory))
      + 0.1 * (1 - min(1, p99_ms/baseline_p99_ms))
```

Rationale:
- Recall weighted 0.4 (quality first)
- QPS log-scaled to reward improvement
- Memory & latency clamped [0,1] (no penalty for beating baseline)

---

## Success Criteria (All Phases)

- Phase 1: SIFT1M in <30s, benchmark <5min/config, ±1% accuracy vs Python baseline
- Phase 2: Grid sweep <2h, 10-15 non-dominated Pareto configs
- Phase 3: BEIR NDCG@10 ≥0.45 on NQ, VectorDBBench 5K QPS sustained
- Phase 4: Darwin evolves 3+ metric improvement, graceful degradation if missing
- Phase 5: MTEB <10h, all-MiniLM ≥0.45 NDCG@10 on NQ

---

## Implementation Plan (16 weeks, 8 agents)

See `docs/metaharness-implementation-plan.md` for full details.

Phase structure:
1. **Phase 1** (4w): ANN-Benchmarks loader + smoke test
2. **Phase 2** (3w): Grid sweep + Pareto frontier
3. **Phase 3** (4w): BEIR + VectorDBBench integration
4. **Phase 4** (3w): Darwin Mode evolution loop
5. **Phase 5** (2w): MTEB embedding quality

File structure: `scripts/benchmark/` (21 TypeScript files) + `crates/ruvector-bench/` (3 Rust files)

---

## Mutable vs Fixed

**Fixed** (not evolved):
- Dataset choice, metric definitions, baseline anchors, query set size

**Mutable** (evolved by Darwin):
- HNSW M/efConstruction, RaBitQ bits, Matryoshka search_dims, PQ bits, fusion strategy, cache eviction policy

---

## Rationale: Why Witness Signing Matters

SOTA claims need full provenance:
```json
{
  "timestamp": "2026-06-21T12:34:56Z",
  "ruvector_commit": "abc123...",
  "config": {"module": "hnsw", "M": 12, ...},
  "results": {"recall@10": 0.85, "qps": 45000, ...},
  "witness_signature": "ed25519_sig..."
}
```

Enables third-party verification and publication credibility.

---

## Uncertainty

- **High**: HDF5 loading, BEIR API stability
- **Medium**: Sweep explosion (mitigate: random sampling), Darwin stability
- **Low**: SOTA achievability, top-3 placement

**Rollback**: If Darwin unstable, fallback to Phase 2 grid + expert curation.

---

## References

- ANN-Benchmarks: https://github.com/erikbern/ann-benchmarks
- BEIR: https://github.com/beir-cellar/beir
- VectorDBBench: https://github.com/zilliztech/VectorDBBench
- MTEB: https://github.com/embeddings-benchmark/mteb
- ADR-128: SOTA Gap Implementations
- ADR-266: MetaHarness Darwin Mode Integration
- ADR-267: SOTA Validation Protocol
