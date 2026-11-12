# @ruvector/kge — Architecture Decision Records

Holographic (HolE-family) knowledge-graph embeddings for ruvector. All records
Proposed, 2026-09-21. Numbering is package-local (outside `scripts/adr-index.mjs`).

- **[ADR-001](ADR-001-architecture.md)** — new `ruvector-kge` crate (+ `-ffi`/`-wasm`), called by `ruvector-graph` Cypher; HNSW inner-product retrieval; v1/v2 boundaries.
- **[ADR-002](ADR-002-scorers-and-math.md)** — one `Scorer` trait; HolE circular correlation via FFT, HolE≡ComplEx as a unit test, RotatE opt-in; FFT-crate spike.
- **[ADR-003](ADR-003-training-and-evaluation.md)** — self-adversarial / 1-vs-all losses, N3, Adam/Adagrad; filtered MRR/Hits@k with CI-asserted RANDOM tie-breaking.
- **[ADR-004](ADR-004-self-optimization-loop.md)** — HPO-weighted bandit loops; EWC continual updates; typesafe ADR-004 promotion gate adopted by reference.
- **[ADR-005](ADR-005-security-model.md)** — threat model plus decoy-triple poisoning and membership-inference rows; no network/shell; new FFT-crate licence check.
- **[ADR-006](ADR-006-benchmarks-and-release-gates.md)** — dataset gates, ANN-recall and FFT-throughput as measured gates (not assumptions), provenance gate.
