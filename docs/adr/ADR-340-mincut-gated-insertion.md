# ADR-340: Mincut-Gated Proximity-Graph Insertion

## Status

Proposed — experimental crate (`ruvector-graft-gate`), not wired into any
production insertion path. **The pre-registered hypothesis was
benchmarked and REJECTED** for its primary claim (`MinCut` gate
outperforming a `CoherenceRatio` gate); this ADR documents a negative
result with a mechanistically-explained cause, not an accepted design.
`CoherenceRatio`, an unplanned side finding, is flagged as a candidate for
a follow-up, separately pre-registered experiment — it is not itself
accepted for production by this ADR.

## Context

`ruvector-proof-gate` (ADR-227) gives RuVector a cryptographically
tamper-evident write path (does the write chain honestly to what was
admitted). `ruvector-retrieval-receipt` (ADR-304) extends tamper-evidence
to the read path. Neither answers a distinct, third question that matters
specifically for RAG corpus-poisoning defense: **is a candidate insertion
locally coherent with the neighborhood it would join, independent of
whether it is cryptographically honest?** A poisoned document can be
perfectly honestly attributed (a real write receipt, a real author) while
still being adversarially crafted to manipulate retrieval for a target
query.

RAG corpus-poisoning is an active 2025-2026 threat area. PoisonedRAG
(USENIX Security 2025) showed 5 documents sufficient to manipulate
targeted answers ~90% of the time; ~0.04% corpus poisoning driving 98.2%
attack success. Documented defenses (GMTP, RADAR, RevPRAG-class
activation-anomaly detection) operate at retrieval time or generation
time. This ADR's research pass did not find published work evaluating
**insertion-time rejection using the ANN index's own graph structure** —
the gap this experiment targets.

## Hypothesis

```text
Given a graph-based ANN index (NSW-style single-layer proximity graph,
M=16 max neighbors per node, ef_construction=64) bootstrapped with 5,000
deterministically-seeded 64-dim vectors drawn from 20 Gaussian clusters
(sigma=0.15, unit-normalized), and a pool of 50 deterministically-seeded
"target queries" representing topics an attacker wants to manipulate,

when 200 synthetic poison vectors (crafted as
normalize(0.7*target_query + 0.3*random_direction), 4 per target query)
are interleaved with 1,000 additional legitimate insertions and each
candidate is evaluated by CoherenceRatioGate (peakedness =
max_kNN_similarity / mean_kNN_similarity, reject if peakedness > 1.35)
and independently by MinCutGate (local k=10 threshold subgraph, edge if
cosine_sim >= neighborhood_median_sim * 0.85, reject if
min-cut(candidate -> neighborhood_anchor) < 2),

then (a) MinCutGate's poison-catch rate exceeds CoherenceRatioGate's by
>=10 percentage points, AND (b) both gates keep legitimate false-reject
<=5%, AND (c) MinCutGate's conditional attack-success rate is >=20
percentage points below the NoGate baseline,

subject to gating latency overhead <500 microseconds mean and recall@10
drop <=2 percentage points vs NoGate.
```

Full derivation and rationale for each constant: see the nightly
README's "Hypothesis" section.

## Decision

Land `crates/ruvector-graft-gate` as an experimental, unintegrated crate
implementing:

1. A from-scratch, zero-dependency NSW-style single-layer proximity
   graph (`GraphIndex`) — greedy best-first search, reciprocal-edge
   insertion pruned to `m` nearest neighbors, multi-entry-point bootstrap
   for reachability (see Implementation Notes below).
2. Three insertion gates behind `GateVariant`:
   - `NoGate` — baseline, always admits (matches current RuVector ANN
     insertion behavior).
   - `CoherenceRatio` — O(k) similarity-shape heuristic: reject if a
     candidate's similarity to its single closest existing neighbor is
     disproportionately higher than its mean similarity across its k
     nearest neighbors.
   - `MinCut` — builds the induced subgraph over a candidate and its k
     nearest existing neighbors, thresholds edges at an adaptive local
     similarity cutoff, and rejects if the max-flow/min-cut from the
     candidate to the neighborhood's best-connected member falls below a
     threshold. Implemented via a bespoke Edmonds-Karp max-flow
     (`gate.rs::max_flow`), not `ruvector-mincut` — see Alternatives.
3. A synthetic, explicitly-scoped poisoning attack model (`data.rs`) and
   a benchmark binary measuring gate latency, poison-catch rate,
   legitimate false-reject rate, attack success rate, and recall@10 for
   all three variants under one shared, fixed interleaved insertion order.

**The benchmark falsified the hypothesis's central claim.** `MinCut`
caught 0/200 poison insertions (0.0%); `CoherenceRatio` caught 122/200
(61.0%) — the opposite ranking from what was hypothesized. See Evidence.

## Implementation Notes on Graph Connectivity

An early implementation used a single fixed entry point. Two unit tests
failed: a self-query did not always return itself as top match, and a
densely-linked (m=32) 150-node graph returned zero brute-force/graph-search
overlap for an arbitrary query. Root cause (confirmed via a temporary,
since-reverted `eprintln` instrumentation pass): a plain, non-hierarchical
NSW graph's greedy search cannot cross between topic clusters unrelated
to a single fixed entry point's own region within a bounded exploration
budget — exactly the failure mode HNSW's multi-level hierarchy exists to
prevent. Fix: register every one of the first 64 inserted nodes as an
additional search entry point (covering every cluster given this crate's
round-robin data generation) plus one more every 137 insertions
thereafter (`EARLY_ENTRY_COUNT` / `ENTRY_POINT_INTERVAL` in
`graph_index.rs`). Documented here because an unnoticed connectivity bug
of this kind would have silently invalidated every recall/attack-success
number in this ADR.

## Evidence

Measured via `cargo run --release -p ruvector-graft-gate --bin
benchmark` (dim=64, 20 clusters, 5,000 bootstrap + 1,000 additional
legit, 200 poison, m=16, ef_construction=64, gate_k=10). Full raw output
in the nightly README; headline numbers:

| Variant | Poison caught (/200) | Legit false-reject (/1,000) | Mean gate latency | Recall@10 |
|---|---|---|---|---|
| NoGate | 0 (0.0%) | 0 | 43 ns | 0.598 |
| CoherenceRatio | 122 (61.0%) | 0 | 55 ns | 0.844 |
| MinCut | 0 (0.0%) | 0 | 13,621 ns | 0.598 |

`cargo test --release -p ruvector-graft-gate`: **18 passed, 0 failed**
(covers PRNG determinism, vector math, graph search/insert correctness
including the connectivity fix above, and the `max_flow` primitive
against three hand-verified graphs: triangle, single-bridge,
disconnected — confirming `MinCut`'s 0% catch rate is a calibration
finding, not a max-flow implementation bug).

**Root cause of `MinCut`'s failure** (see nightly README "Why MinCut
Caught Nothing" for the full instrumented measurement): 97.25% of all
1,200 evaluated candidates (poison and legitimate alike) saturated the
min-cut metric near its structural maximum. At `CLUSTER_SIGMA=0.15` in
64 dimensions, a candidate's 10 nearest existing neighbors are almost
always members of one tight cluster, already highly mutually similar to
each other; the adaptive edge threshold (`median * 0.85`) never thinned
that near-complete local subgraph enough to expose a bridge/cut-vertex
signature. `CoherenceRatio` succeeded because it measures similarity
*shape* (one disproportionately strong match vs. a flatter tail), which
survives that saturation; `MinCut`'s binary edge/no-edge threshold
discards that shape information entirely.

## Consequences

**Positive:**
- A rigorous, mechanistically-explained negative result for graph
  min-cut as an insertion-time poisoning defense at this calibration —
  saves future work from re-attempting the same design without first
  addressing the threshold-saturation failure mode identified here.
- `CoherenceRatio`'s unplanned 61% catch rate at 0% false-reject and 55ns
  overhead is a concrete, cheap, falsifiable candidate for a follow-up
  experiment.
- The multi-entry-point connectivity fix (Implementation Notes) is a
  reusable lesson for any future single-layer NSW work in this repo.

**Negative / costs:**
- `MinCut` as specified must not be presented as a poisoning defense —
  it provides measurable overhead (13.6µs/insertion, ~300x
  `CoherenceRatio`'s cost) for zero measured benefit at this calibration.
- `CoherenceRatio`'s 39% miss rate (78/200 poison admitted) means even
  the promising variant is a partial mitigation, not a solution; no
  production deployment should rely on it alone.
- Both gates share a bootstrap blind spot: the first `GATE_K` (10)
  insertions to any fresh index are admitted unconditionally.
- This experiment used one synthetic attack model with no real embedding
  model or corpus — see Limitations in the nightly README before drawing
  any broader conclusion.

## Alternatives Considered

- **Reuse `ruvector-mincut` directly instead of a bespoke max-flow.**
  Rejected: `ruvector-mincut` is a general-purpose *dynamic* min-cut
  engine (subpolynomial algorithms, j-tree decomposition, canonical/tiered
  coordinators, dependencies on `petgraph`/`rayon`/`crossbeam`/`dashmap`/
  `roaring`) built for graphs that persist and mutate over time. The
  per-insertion subgraph gated here has at most 11 nodes, is rebuilt from
  scratch on every candidate, and is discarded immediately after one
  query — a problem size where `ruvector-mincut`'s algorithmic advantages
  do not apply and its dependency weight would be pure cost on the
  insertion hot path. A bespoke O(V·E²) Edmonds-Karp pass, unit-tested
  against hand-verified graphs, is simpler to audit at this scale.
- **A fixed (non-adaptive) similarity threshold for `MinCut`'s edge
  test**, decided *before* benchmarking. Rejected in favor of an adaptive
  per-neighborhood threshold, judged more robust to varying cluster
  density across a real corpus. This benchmark's result is direct
  evidence the adaptive threshold was itself miscalibrated for the
  tested cluster tightness — but per the pre-registration discipline,
  this was not changed mid-experiment; recalibration is deferred to a
  new, separately pre-registered follow-up (see Rejection Criteria /
  nightly README "Next Research").
- **Gate on raw candidate-to-neighbor similarity alone, no graph or
  shape structure.** Rejected as too similar to existing retrieval-time
  similarity filtering to be a distinct insertion-time contribution.
  `CoherenceRatio` is the minimal structural compromise (similarity
  *shape*, not full graph topology) and it is the variant that worked.

## Implementation Plan

1. (This ADR) Land the experimental crate, benchmark, and tests —
   unintegrated, feature-isolated. `MinCut` ships as documented evidence
   of a rejected design, not as a recommended gate.
2. If pursued further: a second, separately pre-registered experiment
   recalibrating `MinCut`'s edge threshold (or abandoning it) and
   testing `CoherenceRatio` against a second, adversarially-adaptive
   attack model.
3. If `CoherenceRatio` survives a second experiment: integrate as an
   optional wrapper around `ruvector-agent-memory` insertion paths,
   feature-gated so default builds pay zero cost, paired with a
   quarantine-buffer (not hard-reject) policy per the nightly README's
   ruFlo note.
4. Real embedding-model validation: reproduce a published attack (e.g.
   PoisonedRAG's released method) against a real sentence-embedding
   corpus before any production recommendation.

## API Shape

```rust
let mut index = GraphIndex::new(dim, m);
let search_result = index.search(&candidate, ef_construction);
let decision = evaluate_gate(GateVariant::CoherenceRatio, &GateConfig::default(), &index, &search_result);
if decision.admit {
    index.insert_with_neighbors(candidate, &search_result);
}
```

## Feature Flags

None — the crate is opt-in by virtue of not being a dependency of any
other crate. No feature flag is proposed until a variant passes a second,
independent benchmark (see Implementation Plan).

## Benchmark Evidence

See `docs/research/nightly/2026-08-30-mincut-gated-insertion/README.md`
for full methodology and raw `cargo run --release` output.

## Security

- No `unsafe` code; zero external dependencies (`Cargo.toml` has an
  empty `[dependencies]` table).
- `max_flow` (the min-cut primitive) is unit-tested against three
  hand-computed graphs (triangle, single-bridge, disconnected),
  confirming the 0% `MinCut` catch rate is a gate-design calibration
  finding, not a max-flow correctness bug.
- Orthogonal to `ruvector-proof-gate`: this crate answers "is this
  insertion locally coherent," not "is this insertion honestly
  attributed." A write can pass one gate and fail the other; neither
  substitutes for the other, matching the Governance note ADR-304
  already established for retrieval receipts vs. capability gating.
- `MinCut`'s adaptive per-candidate threshold is itself an attack
  surface in principle (co-inserted supporting points could shift the
  local median) — not tested, noted as an open question.

## Governance

Neither gate variant may be presented as a poisoning *solution*.
`CoherenceRatio`'s 61% catch rate at 0% false-reject is a measured
partial mitigation on one synthetic attack model, not a certification of
poison-free ingestion. Any future production framing must state the
residual 39% miss rate and single-attack-model scope explicitly.

## Failure Modes

- `MinCut` fails closed to *ineffective*, not open to *dangerous*: it
  never raised false-rejects; it simply added latency without adding
  defense. The operational risk is an unfounded trust claim, not data
  loss.
- Bootstrap blind spot shared by both gates: the first `GATE_K`
  insertions to a fresh index are always admitted (no neighborhood yet
  exists to evaluate against).
- `CoherenceRatio`'s 39% miss rate means production use requires a
  second, independent defense layer, not reliance on this gate alone.

## Migration

N/A — new, unintegrated crate. No existing insertion path is modified.

## Rollback

Delete `crates/ruvector-graft-gate` and its workspace member entry; no
other crate depends on it.

## Rejection Criteria

This experiment's own pre-registered criteria, and their outcome on this
run:

- If `MinCut`'s catch rate is not ≥10pp above `CoherenceRatio`'s, clause
  (a) fails. **Triggered** — `MinCut` trailed `CoherenceRatio` by 61pp,
  the opposite direction.
- If either gate's legitimate false-reject rate exceeds 5%, clause (b)
  fails. **Not triggered** (both 0.00%).
- If `MinCut`'s conditional attack-success rate is not ≥20pp below
  `NoGate`'s, clause (c) fails. **Triggered** (0pp gap — `MinCut` never
  rejected any poison that reached the conditional denominator).

Overall pre-registered verdict: **REJECT**.

## Open Questions

- Does a recalibrated `MinCut` (fixed absolute threshold, or a much
  larger `mincut_edge_factor`) recover any discriminative power, or is
  local graph connectivity fundamentally the wrong signal for this
  attack class regardless of calibration? Not answered here — deferred
  to a separately pre-registered follow-up per the no-post-hoc-threshold-
  adjustment rule.
- Does `CoherenceRatio`'s 61% catch rate survive an attacker aware of
  and adapting to the specific peakedness metric?
- Does either gate's behavior change materially when composed on a real
  multi-level HNSW index instead of this crate's single-layer NSW
  approximation (recall@10 of 0.598 even ungated)?
- What is the right production policy for a partial (61%) insertion-time
  filter — hard reject, quarantine-and-review, or a scored signal fed
  into a separate downstream decision?
