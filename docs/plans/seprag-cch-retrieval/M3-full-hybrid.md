# M3 — Full Hybrid Pipeline & Multi-Hop Evaluation

**Status:** Planned (gated on M2) · **Est:** 2–3 weeks ·
**Depends on:** [M2](M2-customization-loop.md) · **Feeds:** [M4](M4-integration.md)
**ADRs:** [196](../../adr/ADR-196-seprag-cch-hierarchical-retrieval.md),
[199](../../adr/ADR-199-public-corpus-benchmark-harness.md)

## Purpose

Assemble the end-to-end pipeline and prove the **crossover**: SepRAG wins on
constrained / multi-hop queries; parity (no regression) on pure-semantic IR.

## Pipeline (ADR-196)

```
query ─► HNSW/DiskANN top-m (entry leaves)
      ─► SepRAG separator-tree branch & bound  (metric from M2, filters as subtree predicates)
      ─► ruvector-attn-mincut rerank           (cut-as-attention)
      ─► top-k + elimination-tree path (provenance)
```

## Tasks

1. **Entry-point bridge** — HNSW/DiskANN result → seed vertices in `G_nav`.
2. **Hierarchical filtering** — query constraints (tenant, recency, relation type,
   entity reachability) as `SepNode.may_satisfy(filter)` subtree predicates; combine
   with semantic LB pruning in one traversal.
3. **Rerank** — `ruvector-attn-mincut`, reusing separator cuts as attention masks.
4. **Provenance** — surface the elimination-tree path as a retrieval explanation.
5. **Upgrade corpus** — ingest Wikipedia + Wikidata + hyperlink graph (the deferred
   data engineering); use precomputed embeddings (Cohere `wikipedia-22-12` / BEIR).
6. **Evaluation harness** — two query shapes:
   - Pure-semantic IR: BEIR (NQ, FEVER, MS MARCO-style). Baseline: HNSW/DiskANN.
   - Multi-hop: HotpotQA, 2WikiMultiHopQA, MuSiQue. Ground truth: supporting facts.

## Metrics

| Query shape | Metric | Expectation |
|---|---|---|
| Semantic | Recall@k / nDCG vs qrels | parity with HNSW (no regression) |
| Multi-hop | Supporting-passage coverage, topic coherence | SepRAG wins |
| Constrained | Latency + recall vs HNSW + post-filter | SepRAG wins |

## Exit criteria

- [ ] No regression vs HNSW on pure-semantic IR.
- [ ] Measurable win on multi-hop supporting-passage coverage and/or constrained-query
      latency-at-recall.
- [ ] Provenance path is coherent (retrieved set stays on-topic / crosses gateways
      deliberately).

## Risks

- Wikipedia/Wikidata ingestion + entity alignment is the heaviest data engineering in
  the program — scope it as its own sub-task.
- Multi-hop ground-truth mapping (supporting facts → corpus passages) needs care.
