# M4 — Integration (Postgres, Node, Snapshot)

**Status:** Planned (gated on M3) · **Est:** 1–2 weeks ·
**Depends on:** [M3](M3-full-hybrid.md)
**ADRs:** [196](../../adr/ADR-196-seprag-cch-hierarchical-retrieval.md)

## Purpose

Expose SepRAG through RuVector's existing surfaces and make topology/customizations
durable, hot-swappable artifacts.

## Tasks

1. **Postgres extension** — `ruvector-postgres` function
   `seprag_knn(query vector, k int, lens text, filter jsonb) → setof (id, score, path)`.
2. **Node bindings** — `ruvector-node` API mirroring the Postgres signature.
3. **Persistence** — store `CchTopology` via `ruvector-snapshot`; treat each
   `CchMetric` (lens) as a hot-swappable artifact loadable without rebuild.
4. **Incremental updates** — wire `jtree`/`linkcut` incremental insert; periodic
   batch re-order job; weight-only updates trigger re-customization only.
5. **Operability** — metrics/telemetry (blowup ratio, customization time, query
   latency) surfaced for monitoring.

## Exit criteria

- [ ] `seprag_knn()` callable end-to-end from SQL and Node, returning ranked results
      with provenance.
- [ ] Topology survives restart via snapshot; lenses load without rebuild.
- [ ] Incremental insert + periodic re-order validated under a sustained write load.

## Risks

- Topology persistence format must version cleanly (order + shortcut set + sep tree).
- Re-order cadence vs query-time staleness is an operational tuning knob to document.
