# M2 — Customization Loop & Self-Learning Payoff

**Status:** Planned (gated on M1 = GO) · **Est:** 1–2 weeks ·
**Depends on:** [M1](M1-blowup-measurement.md) · **Feeds:** [M3](M3-full-hybrid.md)
**ADRs:** [198](../../adr/ADR-198-customizable-metric-layer-self-learning.md)

## Purpose

Demonstrate the core self-learning thesis: a relevance-metric change costs a
**customization pass (seconds)**, not an index rebuild. Quantify the gap vs HNSW.

## Tasks

1. **GNN edge head** — `w(u,v) = f_θ(h_u, h_v, edge_feats)`, constrained `≥ 0`
   (`ruvector-gnn`). Forward pass over `G+` edges → metric vector.
2. **Customization as solver specialization** — implement the bottom-up triangle sweep
   over elimination-tree levels as a `ruvector-solver` kernel (`bmssp`/push family),
   vectorized via `ruvector-solver::simd`.
3. **Re-customize-on-update** — drive `θ` updates (synthetic feedback first), re-run
   customization, time it. Topology untouched.
4. **Multi-metric quiver** — maintain `w_semantic`, `w_recency`, `w_trust`, `w_task`
   + on-the-fly blends `Σ λ_i w_i`; per-query lens selection (ADR-198 §3).
5. **EWC guard** — apply `ruvector-gnn::ewc` so metric drift doesn't collapse structure.

## Metrics

| Metric | Target |
|---|---|
| Customization time vs HNSW rebuild (same metric change) | orders-of-magnitude faster |
| Adaptation lag (queries until new feedback reflected in results) | low / bounded |
| Recall stability across re-weights | no collapse |
| Quiver memory cost `O(#metrics × \|G+\|)` | within configured lens budget |

## Exit criteria

- [ ] Re-customization is measurably ≪ HNSW rebuild for an equivalent metric change.
- [ ] A lens switch changes ranking correctly at near-zero marginal query cost.
- [ ] Recall does not degrade across a sequence of metric updates (EWC working).

## Risks

- Non-negativity / additivity constraint on the learned metric may limit expressiveness
  — mitigated by the cost-mapping forms in ADR-198 §1.
- Deep narrow elimination trees parallelize customization poorly — foreshadowed by M1
  separator quality.
