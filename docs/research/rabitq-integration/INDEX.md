# RaBitQ Integration — Research Index

This directory surveys how `ruvector-rabitq` (the rotation-based 1-bit
quantizer published as crate `2.2.0`) is wired into the RuVector
workspace today, and where else it could plausibly land. Output is a
focused review, not a brain dump — read it in order.

- [`01-current-integration.md`](01-current-integration.md) — every
  call site that imports `ruvector_rabitq` today, with `crate:file:line`
  references. Establishes the baseline of three real consumers
  (`ruvector-rulake`, `ruvector-py`, the rabitq demo bin) and surfaces
  what is shipped vs. scaffolded inside the rabitq crate itself
  (notably `VectorKernel` exists; only `CpuKernel` implements it).

- [`02-integration-opportunities.md`](02-integration-opportunities.md) —
  candidate consumer crates, ranked by strategic value × engineering
  effort. For each: what they store, where similarity matters in the
  hot path, what 32× compression buys, the friction (typing,
  determinism, witness propagation), and an honest tier
  classification (now / mid-term / defer / kill).

- [`03-architectural-patterns.md`](03-architectural-patterns.md) — the
  three sane shapes for adding a new consumer: direct-embed, behind
  the `VectorKernel` trait, or through ruLake. Maps each candidate
  from §02 to its preferred pattern, and calls out the anti-patterns
  (re-implementing rotation, ad-hoc compression, witness fragmentation)
  that would silently break the existing ADRs.

- [`04-cross-cutting-concerns.md`](04-cross-cutting-concerns.md) —
  invariants every new integration must hold: determinism across
  architectures, witness format compatibility, memory ownership,
  API-version pinning, performance footprint on WASM/edge, cross-
  language story. The `originals_flat`-encapsulation lesson from the
  Python SDK PR is recorded as a load-bearing constraint.

- [`05-roadmap.md`](05-roadmap.md) — three phases, each with scope,
  files to touch, acceptance test, and LoC budget. Phase 1 picks the
  three top-of-bucket integrations from §02. Phase 2 makes the
  `VectorKernel` trait load-bearing for two consumers across two
  hardware targets. Phase 3 is the optional ADR-class question of
  whether RaBitQ should be the workspace's canonical compression.

- [`06-decision-record.md`](06-decision-record.md) — one page. The
  single sharpest insight from this research, the three integrations
  to start now, the one path we should refuse, and the open
  questions for stakeholders.

All references are to absolute paths under
`/home/ruvultra/projects/ruvector/`. Numbers cited (957 QPS, 32×,
1.02× tax, etc.) trace back to `crates/ruvector-rabitq/BENCHMARK.md`
and `crates/ruvector-rulake/BENCHMARK.md`.
