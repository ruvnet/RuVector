# ADR-006: Benchmarks and release gates

## Status
Proposed

## Date
2026-09-21

## Context

This package makes measurable claims — competitive link prediction, honest
tie-breaking, ANN retrieval that approximates exhaustive scoring — and must keep
measuring them or the claims decay into marketing. Two research-brief gaps make
the gate design load-bearing rather than decorative: no Rust FFT-vs-direct
throughput number exists yet (Q7), and the HolE ANN-compatibility claim is
derived, not cited (Q5). Both are shipped as gates, not assumptions.

## Decision

### Datasets

| Dataset | Size | Source / licence | Used for | Gate |
|---|---|---|---|---|
| FB15k-237 | 310k triples, 14,541 ent, 237 rel | Freebase-derived; follows source, fetched at bench time, nothing redistributed | dense-graph link prediction | filtered MRR ≥ LibKGE ComplEx (0.348) − tolerance; RANDOM tie-break |
| WN18RR | 93k triples, ~41k ent, 11 rel | WordNet-derived; same stance | sparse/symmetric tie-break stress | filtered MRR ≥ LibKGE ComplEx (0.475) / RotatE (0.478) − tolerance; CI asserts non-TOP tie-break |
| CoDEx-M | 185k triples, 17,050 ent, 51 rel | Wikidata-derived, CC-BY (verify at fetch); ships hard negatives | mid-scale gate + adversarial | filtered MRR ≥ published − tolerance (**baseline read from arXiv:2009.07810 at harness build — no number in the brief**); hard-negative confidence reported separately |
| YAGO3-10 | ~1M triples, 123k ent, 37 rel | YAGO-derived; same stance | scale/throughput | LibKGE ComplEx MRR 0.551 informational; **latency/throughput gates set after the ADR-002 spike** |
| Query2Box / BetaE | 14 query types | Ren et al., ICLR 2020 / NeurIPS 2020 | compositional-query correctness | per-query-type MRR/AP, informational v1; gates once `COMPOSE` ships |

Tolerance is a stated per-number margin (default 3 pts, following typesafe
ADR-006's "− 3 pp" stance), applied to a **reproduced** figure whose source is
cited per row — never to an invented number.

### Protocol (mirrors typesafe ADR-006 mechanically)

1. **One frozen, content-hashed split per dataset**, committed and never
   regenerated between arms; `assert_train_eval_disjoint`
   (`crates/sona/src/darwin_guard.rs`) on every proposal. Test scored twice per
   campaign (baseline, champion).
2. **RANDOM tie-breaking, hard-coded and CI-asserted** (ADR-003 §5): a scorer
   returning identical scores for all candidates must yield mean rank
   ≈ `(|E|+1)/2` across seeds — not 1 (TOP), not `|E|` (BOTTOM).
3. **The HolE gate number is ComplEx's LibKGE number by the equivalence**
   (ADR-003 §7); the ADR-002 HolE≡ComplEx unit test is a precondition for
   trusting it.
4. **Transfer holdout:** gate on FB15k-237, transfer-check a CoDEx-M slice
   (ADR-004); no regression beyond the stated tolerance.
5. **Adversarial set:** CoDEx-M's shipped hard negatives plus a
   symmetry-pattern decoy-triple set (ADR-005, Bhardwaj 2021). Reported as
   accuracy **and** mean confidence on targeted facts; the gate is that
   confidence drops under attack, not that accuracy is perfect.
6. **ANN recall gate** (the derived Q5 claim, made measurable): recall@10 of the
   `DistanceMetric::DotProduct` HNSW plus exact-rerank pipeline versus exhaustive
   scoring on FB15k-237 test queries, measured **with and without** the
   augmented-vector MIPS→NN reduction. HolE entities are not unit-norm and HNSW
   over dot product is not a metric space, so this is confirmed empirically, not
   assumed; router-core's HNSW is single-layer/single-entry (typesafe ADR-001),
   adequate at these entity counts.
7. **Latency/throughput:** p50/p95 per query at concurrency 1 and 8, native and
   WASM, plus scored-triples/s. **All latency and throughput thresholds are set
   after the ADR-002 FFT-vs-direct spike**, not before.

### Release gates (all must pass to publish)

| Gate | Threshold |
|---|---|
| Link prediction (FB15k-237, WN18RR) | filtered MRR ≥ cited LibKGE number − 3 pts, RANDOM tie-break |
| Tie-breaking correctness | all-tied scorer yields mean rank ≈ (\|E\|+1)/2; non-TOP asserted in CI |
| HolE≡ComplEx | unit test: both forms agree on random params to fp tolerance |
| ANN recall | recall@10 of DotProduct-HNSW + rerank vs exhaustive ≥ a stated floor on FB15k-237 |
| Adversarial (symmetry decoy + CoDEx hard-neg) | confidence drops under attack; alarm exercised |
| Loop safety | no transfer-split regression beyond 1 pt; control-arm alarm exercised in CI (ADR-004) |
| Latency/throughput | thresholds set after the ADR-002 spike; then enforced |
| Security | ADR-005 CI assertions (no net symbols, no WASI fs/net, no shell) green |
| Provenance | platform packages resolvable on npm before the meta-package bump (`optional-deps-resolvable-on-npm`, regression-guard.yml:351) |

### Where it lives

`npm/packages/kge/bench/`: hash-verified dataset fetchers (nothing
redistributed), frozen fixtures, the harness (`kge bench` / `kge eval`), and one
signed receipt per run. CI runs a small slice (FB15k-237 / WN18RR subset) on
every PR; the full suite runs nightly and before any publish.

## Consequences

- Publishing is gated by measurements that take minutes, not by "tests pass".
- The two headline unmeasured claims (FFT throughput, HolE ANN recall) cannot be
  shipped as numbers until their spike/gate produces them — the ADR states this
  rather than asserting a figure.
- Numbers are RANDOM-tie-break and therefore *lower* than TOP-inflated papers;
  the README states this so the comparison is honest.

## Alternatives considered

- **Accuracy-only gates.** Would let a poisoned or ANN-degraded build pass; the
  adversarial and ANN-recall gates exist to fail those.
- **Assert FFT throughput from complexity theory.** The brief flags no Rust
  benchmark exists; asserting one would be the exact marketing this ADR set
  exists to prevent.
- **Gate CoDEx-M on a guessed baseline.** Rejected: the baseline is read from
  the cited source at harness build; guessing it is a fabricated number.

## Evidence

- Reproduced baselines: LibKGE README (ComplEx FB15k-237 0.348, WN18RR 0.475;
  RotatE WN18RR 0.478; YAGO3-10 0.551). Tie-breaking: Sun et al. 2020
  (arXiv:1911.03903). CoDEx: arXiv:2009.07810. Compositional: Query2Box
  arXiv:2002.05969, BetaE (NeurIPS 2020).
- ANN/MIPS: Budgeted MIPS arXiv:1610.03317; `DistanceMetric::DotProduct`
  (`crates/ruvector-router-core/src/{types.rs:14,distance.rs:18}`),
  single-layer HNSW (`index.rs:119-137`, PR #1005 / issue #430).
- Guards/paths: `.github/workflows/regression-guard.yml:351`;
  `crates/sona/src/darwin_guard.rs`.
- Unmeasured, listed here and gated: FFT-vs-direct throughput (set after ADR-002
  spike); HolE ANN recall (measured gate, not assumed); CoDEx-M baseline MRR
  (read at build).
