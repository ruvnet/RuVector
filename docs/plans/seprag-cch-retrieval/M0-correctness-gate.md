# M0 — Correctness Gate (toy graphs)

**Status:** Planned · **Est:** 2–3 days · **Depends on:** none ·
**Feeds:** [M1](M1-blowup-measurement.md) (reuses this exact implementation)
**ADRs:** [196](../../adr/ADR-196-seprag-cch-hierarchical-retrieval.md),
[197](../../adr/ADR-197-navigation-graph-metric-independent-ordering.md)

## Purpose

Prove the SepRAG algorithm is **correct** before pointing it at real data. M0 is a
*test harness*, not a milestone to polish. Its sole job: make M1's blowup go/no-go
signal trustworthy by eliminating "is it a bug or is it the data?" ambiguity.

**Critical caveat:** M0 success does **not** validate the thesis — synthetic SBM
graphs have small separators by construction, so M0 will pass almost regardless of
real-world viability. M0 validates *code*, M1 validates *the idea*.

## Scope

Build the core, metric-independent SepRAG pipeline + static query path on synthetic
graphs only. No embeddings, no real data, no GNN, no customization loop.

## Where it lives

New module `crates/ruvector-mincut/src/cch/` (reuses in-crate separator machinery), or
a thin new crate `ruvector-seprag` depending on `ruvector-mincut`. Decision recorded in
M0 task 1. Tests in-crate; toy benchmark wired into `ruvector-bench`.

## Tasks

1. **Module scaffold** — decide `ruvector-mincut::cch` vs new `ruvector-seprag` crate;
   define `CchTopology`, `SepTree`, `SepNode`, `CchMetric` structs (per ADR-197 §4).
2. **Nested-dissection order** — adapt `ruvector-mincut::jtree`/`expander`/`cluster` to
   emit a vertex order (separators ranked highest) + separator decomposition tree.
3. **Symbolic contraction** — build chordal `G+` + `elim_parent` (fill-in), metric-free.
4. **Cache-friendly layout** — relabel vertices to rank order; upward CSR with ascending
   rows; elimination tree in DFS post-order.
5. **Static customization** — single fixed metric (edge weight = graph edge cost); the
   bottom-up triangle sweep producing shortcut weights.
6. **Upward search + separator-tree k-NN** — the branch-and-bound with lower-bound
   pruning (ADR-196 Phase-3 algorithm).
7. **Brute-force oracle** — exhaustive graph-distance k-NN (Dijkstra/BFS per query) for
   exact comparison.
8. **Toy benchmark** — wire into `ruvector-bench`; emit search-space size + (for sanity)
   blowup ratio even on toy graphs.

## Test graphs

- **Stochastic Block Model** (clear communities → clean separators) — primary.
- **2-D / 3-D grid** (known √n separators) — separator-size sanity.
- **Path / cycle / clique** — degenerate edge cases (clique = worst-case fill-in).
- Sizes 100 → 10,000 vertices (small enough for exhaustive oracle).

## Exit criteria (all must hold)

- [ ] SepRAG k-NN output **exactly equals** the brute-force oracle for k ∈ {1,5,10,50}
      across all toy graphs and ≥100 random queries each (this is the gate).
- [ ] Search-space size (vertices touched) **shrinks** as separator size decreases —
      demonstrated by varying SBM inter-block density.
- [ ] Pruning is sound: no pruned subtree ever contained a true top-k result (assert in
      a debug "no-prune" oracle mode).
- [ ] Determinism: identical results across runs (fixed seeds; tie-break rule defined).
- [ ] `cargo test` + `cargo clippy` clean; the module builds in the workspace.

## Explicit non-goals

Real data · embeddings · GNN metric · dynamic updates · multi-metric quiver ·
performance tuning · WASM. All deferred to M1+.

## Risks

- **Over-investment** — the main risk. Cap at 2–3 days; if the implementation is
  fighting you, that itself is signal to simplify before M1.
- Separator heuristic quality in `ruvector-mincut` may need tuning even on toy graphs;
  if so, note it — it foreshadows M1 difficulty.
