# ruvector-graft-gate

Mincut-gated proximity-graph insertion: a write-time structural defense
against RAG corpus-poisoning insertions into a graph-based ANN index.

Baseline (`NoGate`) plus two gate variants (`CoherenceRatio`,
`MinCut`) are implemented and benchmarked against a synthetic,
deterministically-seeded single-target embedding-optimization attack
model. See `docs/research/nightly/2026-08-30-mincut-gated-insertion/README.md`
and `docs/adr/ADR-340-mincut-gated-insertion.md` in the repository root for
the hypothesis, attack model, benchmark methodology, and results.

```
cargo run --release -p ruvector-graft-gate --bin benchmark
cargo test --release -p ruvector-graft-gate
```
