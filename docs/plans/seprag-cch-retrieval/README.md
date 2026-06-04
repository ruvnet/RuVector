# SepRAG — CCH-Inspired Retrieval: Milestone Plans

Implementation roadmap for the SepRAG retrieval layer described in
[ADR-196](../../adr/ADR-196-seprag-cch-hierarchical-retrieval.md) –
[ADR-199](../../adr/ADR-199-public-corpus-benchmark-harness.md).

SepRAG adapts Customizable Contraction Hierarchies (CCH) — nested dissection,
balanced separators, contraction shortcuts, elimination trees, and the
separator-tree k-NN algorithm — to hybrid vector + knowledge-graph memory. It
**complements** HNSW/DiskANN (entry-point search) rather than replacing it,
targeting constrained, relational, multi-hop, and re-weightable retrieval.

## The central bet

CCH's speedup requires **small balanced separators** (low treewidth). Road networks
have them; dense embedding kNN graphs (expander-like) do not. The entire program
lives or dies on one measured number — the **shortcut-blowup ratio** `|G+| / |G_nav|`
on real data. The milestones are sequenced to surface that number as cheaply and as
early as possible, on a *correctness-validated* implementation (so the signal is not
confounded by bugs).

## Outcome (2026-06-04): M0 ✅ · M1 ❌ NO-GO · M2–M4 not pursued

M0 and M1 ran; the decisive M1 gate returned **NO-GO** (full evidence in
[ADR-199 Empirical Outcome](../../adr/ADR-199-public-corpus-benchmark-harness.md#empirical-outcome-2026-06-04)).
The `ruvector-seprag` crate is a correct, tested reference implementation; CCH full
contraction does **not** fit embedding/citation retrieval graphs (near-linear treewidth).
M2–M4 are not pursued. Summary:

| Backbone (N=1500, recall 50/50) | blowup | elim height |
|---|---|---|
| roadNet-PA (control) | 7.6× | 136 (~3.5·√n) — CCH works |
| ogbn-arxiv citation | 23.8× | 941 (~0.6·n) |
| ogbn-arxiv feature kNN (k=10) | 42.4× | 837 (~0.56·n) |

The separator-tree **query** (pruning ~100% of scans, exact recall) is the salvageable
piece; the failure is preprocessing fill-in. Only plausible future niche: sparse
relational backbone + re-customizable metrics (ADR-196 scope), revisited only if it beats
HNSW.

## Milestone sequence (as originally planned)

| Plan | Goal | Retires which risk | Gate | Status |
|------|------|--------------------|------|--------|
| [M0](M0-correctness-gate.md) | Separator-tree k-NN correct on toy graphs | Implementation correctness | k-NN == brute-force oracle | ✅ done |
| [M1](M1-blowup-measurement.md) | Blowup ratio on ogbn-arxiv (static metric) | **Research viability (decisive)** | blowup small + separators sublinear → GO | ❌ NO-GO |
| [M2](M2-customization-loop.md) | GNN metric → customization; self-learning payoff | Re-weight cost vs rebuild | customize ≪ HNSW rebuild | ⏸ not pursued |
| [M3](M3-full-hybrid.md) | HNSW entry + filters + rerank; multi-hop QA | End-task quality / crossover | win on multi-hop, parity on semantic | ⏸ not pursued |
| [M4](M4-integration.md) | Postgres fn + node bindings + snapshot | Productionization | `seprag_knn()` callable end-to-end | ⏸ not pursued |

## Key sequencing principle

M0 is a **thin correctness gate (~2–3 days), not a destination**. Its only job is to
make M1's go/no-go number trustworthy: if blowup looks bad, we must know it is the
*data* (expander-like), not a bug in a fresh implementation. M1 reuses M0's exact
code, pointed at real data. Do **not** over-invest in toy benchmarks.

## Fallback ladder (if M1 blowup is catastrophic)

1. Hyperbolic backbone (`ruvector-hyperbolic-hnsw`) — tree-like geometry → small separators.
2. GNN-learned contraction order (ADR-196 extension E1) — learn an order minimizing fill-in.
3. Bounded-degree contraction (cap fill-in, trade exactness; recall-test).
4. Abandon SepRAG for this data class; keep flat ANN.

## Crate reuse (this is composition, not green-field)

`ruvector-mincut` (`jtree`, `expander`, `cluster`, `sparsify`, `linkcut`,
`algorithm`) · `ruvector-solver` (`bmssp`, `forward_push`, `backward_push`, `simd`) ·
`ruvector-sparsifier` · `ruvector-diskann` · `ruvector-hyperbolic-hnsw` ·
`ruvector-rabitq` · `ruvector-gnn` (`ewc`, `graphmae`, `query`) · `ruvector-attn-mincut` ·
`ruvector-graph::hybrid` · `ruvector-bench`.
