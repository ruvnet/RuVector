# ruvector-field example

Runnable reference implementation sketch of the RuVector field subsystem.
Full spec: [`docs/research/ruvector-field/SPEC.md`](../../docs/research/ruvector-field/SPEC.md).

This example deliberately uses only the Rust standard library so the shape of
the spec is visible end to end. It is **not** production code. Swap in a real
ANN index, HNSW, solver, and mincut backend when promoting this to a crate.

## What it demonstrates

1. Four logical shells: `Event`, `Pattern`, `Concept`, `Principle`
2. Geometric antipodes (cheap vector flip) kept separate from semantic antipodes (explicit contradiction link)
3. Multiplicative resonance scoring — one collapsed factor collapses the whole score
4. Coherence from an effective resistance proxy
5. Shell promotion rules driven by support and contradiction counts
6. Shell aware retrieval with a contradiction frontier and explanation trace
7. Four channel drift detection with agreement threshold
8. Routing hints based on role embeddings (advisory — privileged actions still need proof + witness)
9. Phi scaled compression budgets per shell

## Run

```bash
cargo run --manifest-path examples/ruvector-field/Cargo.toml
```

Expected output (abridged):

```
=== RuVector Field Subsystem Demo ===

Shell promotions:
  node   4: Event → Pattern
  ...

Retrieval:
  selected nodes: [...]
  contradiction frontier: [...]
  explanation trace:
    - node X has semantic antipode Y — flagged on contradiction frontier
    ...

Drift: semantic=... structural=... total=...

Routing hint: agent=Some(1001) gain=... reason="best role match: constraint"

Shell budgets (base = 1024):
  Event     → 1024.0
  Pattern   → 632.8
  Concept   → 391.1
  Principle → 241.7
```

## File layout

```
examples/ruvector-field/
├── Cargo.toml
├── README.md
└── src/
    ├── main.rs      # demo entry point
    ├── types.rs     # Shell / NodeKind / EdgeKind / AxisScores / FieldNode / ...
    └── engine.rs    # FieldEngine: ingest, promote, retrieve, drift, route
```

## Relationship to the rest of RuVector

| RuVector crate | Role in the field subsystem |
|---|---|
| `ruvector-sparsifier` | compressed field graph for coherence sampling and drift at scale |
| `ruvector-solver` | local coherence, effective resistance, route gain |
| `ruvector-mincut` | split / migration / fracture hints (outside the 50 µs epoch initially) |
| RuVix kernel | receives `PriorityHint`, `SplitHint`, `MergeHint`, `TierHint`, `RouteHint` — only after benchmarks show gain |

## Acceptance gate (from the spec)

The field engine only graduates from user space into RuVix kernel hints when
**all four** hold on a contradiction-heavy benchmark:

1. contradiction rate improves by ≥ 20 %
2. retrieval token cost improves by ≥ 20 %
3. long session coherence improves by ≥ 15 %
4. enabling hints does not violate the 50 µs coherence epoch budget or the
   sub-10 µs partition switch target
