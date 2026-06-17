# M1 — Blowup Measurement on Real Data (the decisive go/no-go)

**Status:** Planned · **Est:** 1–2 weeks · **Depends on:** [M0](M0-correctness-gate.md)
(reuses its implementation) · **Feeds:** [M2](M2-customization-loop.md)
**ADRs:** [196](../../adr/ADR-196-seprag-cch-hierarchical-retrieval.md),
[197](../../adr/ADR-197-navigation-graph-metric-independent-ordering.md),
[199](../../adr/ADR-199-public-corpus-benchmark-harness.md)

## Purpose

Answer the one question that can kill SepRAG: **do real embedding/graph data have small
enough separators that contraction does not blow up?** Measured, not argued. Because M1
runs the M0-validated code, a bad result means the *data* is expander-like — a clean,
trustworthy go/no-go signal.

## Dataset — ogbn-arxiv first (deliberately)

Start on **ogbn-arxiv** (~170K nodes, ~1.2M citation edges, 128-d node features):
- Ingest is near-free — ships as a downloadable graph with features and standard splits.
- Real citation structure of known character (good separator-quality probe).
- Avoids Wikipedia/Wikidata KG alignment, which is real data engineering deferred to M3.

Node features serve as embeddings; the citation graph is the relational backbone.

## Navigation graph `G_nav` (ADR-197)

Build `G_nav` as: citation edges ∪ a **degree-bounded / α-pruned kNN graph** over node
features (RNG-style; reuse `ruvector-diskann` Vamana pruning). **Do not** use the dense
kNN graph — pruning is what creates separators. Static metric = `1 − cos` on `G_nav`
edges (no GNN yet).

## Tasks

1. **Ingest** ogbn-arxiv → `ruvector-graph` (nodes + citation edges) + features into a
   vector store (`ruvector-diskann` / `ruvector-rabitq` quantized).
2. **Build `G_nav`** — citation ∪ α-pruned kNN; record degree distribution.
3. **Phase 1** — nested-dissection order + symbolic contraction (M0 code, real scale).
4. **Phase 2** — static customization (single `1 − cos` metric).
5. **Instrument blowup** — `|G+|`, `|G_nav|`, ratio; separator-size distribution by
   tree level; elimination-tree height.
6. **Query path** — separator-tree k-NN; build a **sampled** hybrid-distance oracle
   (brute-force is expensive at 170K — sample ~1–5K queries).
7. **Baselines** — plain HNSW top-k and DiskANN beam search on the same vectors.
8. **Report** — into `ruvector-bench`, emitting the full metric table below.

## Metrics (report together — ADR-199 §4)

| Metric | Decision role |
|---|---|
| **Shortcut-blowup ratio `\|G+\|/\|G_nav\|`** | **Primary gate.** |
| Separator-size distribution; elim-tree height | Diagnoses road-like vs expander-like. |
| Recall@k vs sampled hybrid-distance oracle | SepRAG correctness at scale. |
| Latency p50/p95 vs HNSW/DiskANN | Performance claim. |
| Search-space size (vertices touched) | Mechanistic proof of pruning. |

## Exit criteria / decision

**GO** if:
- [ ] Blowup ratio is "small" — target **≤ ~3–5×** `|G_nav|` (tune; >10× is a red flag).
- [ ] Separator sizes are clearly **sublinear** (not Θ(n)); elim-tree height manageable.
- [ ] Recall@k vs the sampled oracle is high (≥0.95) — confirms M0 correctness holds at
      scale on real (non-toy) structure.
- [ ] Search space is a small fraction of n (region pruning demonstrably works).

**NO-GO / fallback** (descend the ladder in [README](README.md)): hyperbolic backbone →
GNN-learned order → bounded-degree contraction → abandon. Record which rung is reached
and the blowup numbers that triggered it.

**Either outcome is a successful M1** — the point is a trustworthy signal, not a
predetermined yes.

## Risks

- **Confounding** — mitigated by M0 (validated code) and the sampled oracle recall check;
  if recall is low here, suspect a scale-dependent bug before blaming the data.
- **Oracle cost** — brute-force graph distance at 170K is heavy; restrict to a sampled,
  fixed query set.
- **Separator heuristic at scale** — `ruvector-mincut` balanced-cut quality directly
  drives blowup; budget time to tune it before declaring NO-GO.
