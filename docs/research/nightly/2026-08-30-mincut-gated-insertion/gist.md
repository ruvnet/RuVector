# Mincut-Gated Insertion: A Structural Defense Against RAG Corpus Poisoning

## Problem

RAG corpus-poisoning attacks craft a handful of documents whose embeddings
are optimized to rank highly for a target query, so a retrieval-augmented
system ingests and later surfaces attacker-chosen content. PoisonedRAG
(USENIX Security 2025) showed ~0.04% corpus poisoning can drive a 98%
attack success rate. Documented defenses operate at retrieval time
(reranking, perplexity filtering) or generation time (activation-anomaly
detection). This experiment asks whether a graph-based ANN index can
refuse a poisoned vector at **insertion time**, using only the graph
structure the index already builds.

## Hypothesis

Two gate designs were compared against an ungated baseline, all sharing
one from-scratch NSW-style proximity-graph index over 5,000 bootstrapped
+ 1,000 additional legitimate 64-dim vectors (20 Gaussian clusters,
sigma=0.15) plus 200 synthetic poison insertions
(`normalize(0.7*target_query + 0.3*random_direction)`, formalizing an
optimization-based embedding attack):

- **`CoherenceRatio`**: reject if a candidate's similarity to its single
  closest existing neighbor is disproportionately higher than its mean
  similarity across its k=10 nearest neighbors (peakedness > 1.35).
- **`MinCut`**: build the induced subgraph over a candidate and its 10
  nearest existing neighbors, threshold edges at an adaptive local
  similarity cutoff, and reject if the max-flow/min-cut from the
  candidate to the neighborhood's best-connected member falls below 2.

Pre-registered: `MinCut` should out-catch `CoherenceRatio` by ≥10
percentage points, both should keep legitimate false-rejects ≤5%, and
`MinCut` should cut conditional attack success by ≥20pp vs. baseline —
all subject to <500µs mean gate overhead and ≤2pp recall@10 drop.

## Result

| Variant | Poison caught (/200) | Legit false-reject (/1,000) | Mean gate latency | Recall@10 |
|---|---|---|---|---|
| NoGate | 0 (0.0%) | 0 | 43 ns | 0.598 |
| CoherenceRatio | **122 (61.0%)** | 0 | 55 ns | **0.844** |
| MinCut | **0 (0.0%)** | 0 | 13,621 ns | 0.598 |

**REJECT.** The hypothesis's central claim — that graph min-cut would
outperform a cheap similarity-shape heuristic — was falsified in the
opposite direction. A follow-up instrumentation pass (temporary, reverted
before commit) showed why: at the tested cluster tightness, 97.25% of
1,200 evaluated candidates (poison and legitimate alike) saturated the
min-cut metric near its structural maximum (flow=9 or 10 of a possible
10), because a candidate's 10 nearest neighbors are almost always drawn
from one tight cluster whose members are already highly mutually similar
— the adaptive edge threshold never thinned that neighborhood enough to
expose a bridge/cut-vertex signature. `CoherenceRatio` succeeded because
it measures similarity *shape* (is one match disproportionately strong
relative to the rest), which survives that saturation; `MinCut`'s
binary-threshold graph construction discards that shape information.

`CoherenceRatio` was not the hypothesis under test, so this is reported
as a genuine side finding, not a rescued positive result: 61% catch rate
at 0% false-reject and negligible overhead (55ns), plus an unplanned
recall@10 improvement (0.598 → 0.844) from keeping poison out of the
graph's own link structure.

## Design

- Zero external dependencies; a from-scratch xorshift64* deterministic
  PRNG, a minimal single-layer NSW proximity graph (greedy best-first
  search, reciprocal-edge insertion pruned to `m` nearest neighbors,
  multi-entry-point bootstrap for reachability), and a bespoke ≤11-node
  Edmonds-Karp max-flow for the min-cut gate (unit-tested against
  hand-verified triangle/bridge/disconnected graphs).
- Deliberately does not depend on `ruvector-mincut` (a general-purpose
  dynamic min-cut engine) — the per-insertion subgraph is tiny and
  thrown away immediately, so a bespoke pass is simpler and avoids a
  heavy dependency on the insertion hot path.
- Same shuffled insertion order replayed against all three variants from
  cloned copies of one bootstrapped index, so no variant sees an easier
  sequence; gate-overhead timing brackets only the decision, after the
  `search()` call every variant already pays for.

## Evidence

`cargo run --release -p ruvector-graft-gate --bin benchmark`, raw output
(dim=64, 20 clusters, 5,000+1,000 legit, 200 poison, m=16,
ef_construction=64, gate_k=10):

```text
variant         gate_mean_ns   poison_catch   legit_fr_%   recall@10
NoGate                  43.3         0/200          0.0      0.5980
CoherenceRatio          55.3       122/200          0.0      0.8440
MinCut               13620.9         0/200          0.0      0.5980
```

`cargo test --release -p ruvector-graft-gate`: 18/18 passed.

## Limitations

Single synthetic attack model (no real embedding model or corpus); single
hardware configuration; single-layer NSW, not production HNSW; no
deletes, concurrency, or scale beyond ~6,200 vectors; `MinCut`'s specific
threshold calibration (median × 0.85, k=10) is shown to saturate at this
cluster tightness but was not recalibrated mid-experiment per the
pre-registration rule — that recalibration is deferred to a
separately-pre-registered follow-up, not retrofitted here. An adaptive
attacker targeting `CoherenceRatio`'s peakedness metric specifically was
not tested.

## Production Relevance

Do not ship `MinCut` as specified — measurable overhead (13.6µs, ~300× 
`CoherenceRatio`'s cost) for zero measured defensive value at this
calibration. `CoherenceRatio` is a plausible second-experiment candidate
(not a production recommendation yet): one attack model and one dataset
shape stand between this result and a deployment claim. The right next
step is a second, independently-designed attack model before any
adoption decision.

## References

- PoisonedRAG — summarized at [themenonlab.blog](https://themenonlab.blog/blog/poisonedrag-rag-knowledge-corruption-attack).
- [Practical Poisoning Attacks against RAG](https://arxiv.org/pdf/2504.03957) (arXiv:2504.03957).
- [Semantic Chameleon](https://arxiv.org/html/2603.18034v1) (arXiv:2603.18034).
- [GMTP: Gradient-based Masked Token Probability](https://arxiv.org/pdf/2507.18202) (arXiv:2507.18202).
- [RADAR: Defending RAG Dynamically against Retrieval Corruption](https://arxiv.org/pdf/2605.22041) (arXiv:2605.22041).
- [Anomaly Detection in Dynamic Graphs: A Comprehensive Survey](https://arxiv.org/html/2406.00134v1) (arXiv:2406.00134).
- In-repo: `ruvector-proof-gate` (ADR-227), `ruvector-retrieval-receipt` (ADR-304).
- Full write-up: `docs/research/nightly/2026-08-30-mincut-gated-insertion/README.md`,
  `docs/adr/ADR-340-mincut-gated-insertion.md`.
