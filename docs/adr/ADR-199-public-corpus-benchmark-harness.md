---
adr: 199
title: "Public-Corpus Benchmark & Evaluation Harness for SepRAG"
status: proposed
date: 2026-06-04
authors: [ofershaal, claude-flow]
related: [ADR-194, ADR-195, ADR-196, ADR-197, ADR-198]
tags: [ruvector, benchmark, evaluation, wikipedia, wikidata, beir, hotpotqa, ogb, recall, graph-rag]
---

# ADR-199 — Public-Corpus Benchmark & Evaluation Harness for SepRAG

## Status

**Proposed.** This is the experimental backbone for [ADR-196]–[ADR-198]. It exists to
*answer empirically* the design questions the other ADRs leave open, using large public
datasets rather than synthetic graphs or a priori reasoning.

## Context

SepRAG ([ADR-196]) rests on one decisive, unprovable-by-argument assumption: that a
real hybrid memory graph has small enough separators that contraction does not blow up.
The only way to settle it is to load a large, *natively hybrid* public corpus (text +
explicit graph) into RuVector and measure. The benchmark also reveals the **crossover**
between flat ANN and SepRAG by testing two query shapes, so we learn *where* each wins
instead of cherry-picking.

This ADR also records the standing guidance that "huge" is a trap for the superlinear
Phase-1 ordering ([ADR-197]): start at 10⁵–10⁶ nodes, prove the win, then scale.

## Decision

### 1 — Corpus (natively hybrid backbone)

Primary: **Wikipedia + Wikidata + the Wikipedia hyperlink graph**, all aligned on the
same entities — text to embed, a real KG (`RelationType` edges), and a real link graph
in one corpus. Licensing is clean (Wikipedia CC-BY-SA; Wikidata CC0).

**Use precomputed embeddings** (e.g. Cohere `wikipedia-22-12` on HuggingFace, or BEIR's)
so embedding cost does not dominate and we are not bottlenecked on the embedder work in
[ADR-194]/[ADR-195]. Hash-embedding fallback is explicitly disallowed for scored runs
(per the silent-fallback lesson in [ADR-194]).

### 2 — Query workloads (test the crossover with two shapes)

- **Pure-semantic IR** — BEIR subsets (NQ, FEVER, MS MARCO-style passage retrieval).
  Ground truth: qrels. **Expectation: HNSW/DiskANN wins or ties.** This is the parity /
  no-regression guard for SepRAG.
- **Multi-hop / relational** — HotpotQA, 2WikiMultiHopQA, MuSiQue. Ground truth:
  supporting-fact passages. **Expectation: SepRAG wins** on supporting-passage coverage
  and topic coherence — the queries flat post-filtering handles poorly.

### 3 — Graph-structure datasets (separator quality / blowup, [ADR-197])

OGB graphs with node features as embeddings: **ogbn-arxiv (~170K, start here)**,
ogbn-products, and a capped ogbn-papers100M subset later. Real citation / co-purchase
structure with standard splits — ideal for measuring separator size and fill-in on
graphs of known character.

### 4 — Metrics (report all together; no metric in isolation)

| Metric | Why it matters |
|---|---|
| **Shortcut-blowup ratio `\|G+\|/\|G_nav\|`** | The gating viability metric ([ADR-197]). Decides go/no-go. |
| **Separator-size distribution** | Diagnoses "road-like vs expander-like" on real data. |
| Recall@k vs **hybrid-distance oracle** | Correctness of SepRAG's graph-distance k-NN. |
| Recall@k / nDCG vs **qrels** | End-task retrieval quality vs baselines. |
| Latency p50 / p95 | The headline performance claim. |
| Search-space size (vertices touched) | Mechanistic proof of region pruning. |
| **Customization time vs HNSW rebuild** | Quantifies the self-learning payoff ([ADR-198]). |
| **Adaptation lag** (queries until new feedback reflected) | Self-learning responsiveness. |
| Multi-hop supporting-passage coverage | Where SepRAG is expected to win. |

Baselines for every run: plain HNSW top-k, DiskANN beam search, and (for constrained
queries) HNSW + post-filtering.

### 5 — Milestones (incremental, start small)

```
M0  Toy validation     SBM + grid graphs; prove recall == brute force,
                        search space shrinks with separator size.        (~2 wks)
M1  SepRAG MVP          ogbn-arxiv + a Wikipedia category subgraph,
                        static metric. MEASURE BLOWUP RATIO (go/no-go).
M2  Customization loop  Wire GNN edge head → customize; time re-customize
                        vs HNSW rebuild; measure adaptation lag.          [ADR-198]
M3  Full hybrid         HNSW entry → SepRAG → filters as subtree preds →
                        attn-mincut rerank; multi-hop QA eval.
M4  Integration         ruvector-postgres fn seprag_knn(query,k,lens,filter);
                        ruvector-node bindings; topology in ruvector-snapshot.
```

### 6 — Harness location

Extend the existing `ruvector-bench` crate and `benches/` with a SepRAG suite. Every
run emits the blowup ratio and separator-size distribution alongside latency/recall, so
the expander risk is always visible.

## Consequences

**Positive.**
- The benchmark *answers* the open questions from [ADR-196] (query shape, sparsity,
  metric cadence, exact-vs-approximate recall) instead of requiring up-front answers.
- Two query shapes reveal the HNSW↔SepRAG crossover rather than a biased single number.
- M1's blowup measurement is an early, cheap go/no-go gate before heavy investment.

**Negative.**
- Wikipedia/Wikidata ingestion + KG alignment is non-trivial data engineering.
- Building hybrid-distance ground-truth oracles is expensive (brute-force graph
  distance) — budget for it, restrict to sampled query sets.
- Scaling the Phase-1 ordering beyond ~10⁶ nodes may need the GNN-learned ordering
  extension (ADR-196 E1) before ogbn-papers100M is feasible.

**Neutral.**
- All datasets are public and appropriately licensed; no secrets or PII involved.

## Alternatives considered

- **Synthetic graphs only.** Rejected as the *primary* corpus — they cannot settle the
  expander question for real embeddings; kept only for M0 sanity.
- **A single dataset.** Rejected — would hide the crossover; the two-shape design is
  the point.
- **Embed everything ourselves.** Rejected for v1 — precomputed embeddings de-risk the
  experiment and isolate retrieval performance from embedder throughput ([ADR-194]).
