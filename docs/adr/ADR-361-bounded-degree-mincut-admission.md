# ADR-361: Bounded-Degree K-Nearest-Centroid Min-Cut Admission

## Status

Proposed — REJECTED as a drop-in replacement, evidence retained.
Experimental crate (`ruvector-memory-admission`), not wired into
`ruvector-agent-memory` or any production write path. The crate's existing
production default (`MincutGatedAdmission`, ADR-344) is unchanged and
remains the recommended mechanism; this ADR documents a bounded Darwin
exploration of a cheaper variant that did not clear its own promotion
gate, plus a closed cost-ceiling finding that justifies keeping the
exploration result as part of this crate's lineage rather than discarding
it.

## Context

ADR-344 (`MincutGatedAdmission`, accepted) builds a Stoer-Wagner global
min-cut graph over *all* `C` existing cluster centroids plus the
candidate point before every write-time admission decision. Its own Next
Research item 1 named the obvious question explicitly: "Locate the O(C^3)
practical ceiling empirically... decide whether a bounded-degree graph
approximation (e.g., only the K nearest existing centroids, not all of
them) preserves the quality gain at lower cost." That question was asked
and left open; no prior nightly answered it. This nightly (2026-11-18)
answers it.

## Hypothesis

```text
Given the existing MincutGatedAdmission write-path (ADR-344), which builds
a global min-cut graph over ALL C existing cluster centroids plus the
incoming point before every admission decision,

when the graph is instead restricted to the incoming point's K nearest
centroids only (BoundedDegreeAdmission, K in {4, 8, 16, 32}), compared
against both the unbounded MincutGatedAdmission baseline and the original
fixed-threshold NearestCentroidThreshold baseline, all at the SAME
matched final cluster count,

then mean admission latency should close the O(C^3) cost ceiling (stay
near-flat in C for fixed K, in contrast to the full graph's cubic growth),

subject to: purity and recall@10 staying within 1 percentage point of the
unbounded MincutGatedAdmission result at the same matched cluster budget,
and the crate's existing test suite remaining green.
```

Fixed before running (not relaxed after seeing results): `MAX_QUALITY_DELTA_PP = 1.0`,
latency budget `MAX_MEAN_LATENCY_US = 500.0` (unchanged from ADR-344).

## Decision

**Do not promote `BoundedDegreeAdmission` as a replacement for
`MincutGatedAdmission`.** The cost-ceiling half of the hypothesis is
strongly confirmed; the quality-preservation half is not. Retain
`BoundedDegreeAdmission` in the crate as a documented, tested,
benchmarked-but-rejected variant (Darwin lineage), not as a production
default. Retain the matched-budget `tau` calibration methodology this run
added (`calibrate_tau_bounded`) for any future nightly that revisits this
question with a different mechanism (e.g. K-dependent tau, or hybrid
local+global cut criteria).

## Evidence

**Part 1 — scaling sweep** (exact `C`, via benchmark-only
`from_centroids` seeding, 80 held-out query decisions per point, DIMS=64,
release build, x86-64 Linux):

| C | Full graph mean (µs) | K=4 (µs) | K=8 (µs) | K=16 (µs) | K=32 (µs) |
|---|---|---|---|---|---|
| 8 | 2.3 | 1.0 | 3.8 | 3.6 | 2.9 |
| 32 | 83.0 | 1.2 | 4.5 | 18.2 | 84.6 |
| 64 | 425.5 | 2.0 | 5.2 | 19.3 | 84.9 |
| 96 | 1,120.9 | 2.5 | 5.9 | 19.6 | 84.5 |
| 128 | 2,309.0 | 4.9 | 8.2 | 26.1 | 110.2 |
| 256 | 14,709.5 | 6.3 | 10.2 | 24.5 | 111.3 |

Full-graph mean latency crosses the existing 500µs write-path budget at
**C=96** (1,120.9µs; a second run found the crossover at C=64, 500.4µs —
both runs agree the ceiling is real and lands in the same C=64–96 band;
see Reproducibility below). Every bounded-degree K stays under the budget
at that C. At C=256, K=4 is **~2,335x** faster than the full graph
(6.3µs vs 14,709.5µs) while doing 126x fewer similarity operations
(262 vs 32,896). **Scaling claim: PASS.**

**Part 2 — quality preservation, matched budget** (N_POINTS=6000,
K_TRUE=24, DIMS=64, N_QUERIES=400, TAU=0.005; target = full graph's 15
natural clusters):

| Variant | Clusters | Purity | Recall@10 | Mean (µs) |
|---|---|---|---|---|
| NearestCentroidThreshold (matched) | 15 | 0.3417 | 0.5218 | 0.01 |
| **MincutGatedAdmission (full graph)** | 15 | 0.3488 | 0.5435 | 11.86 |
| BoundedDegreeAdmission(K=4), tau calibrated | 18 | 0.4377 | 0.5240 | 1.04 |
| BoundedDegreeAdmission(K=8), tau calibrated | 23 | 0.4630 | 0.5697 | 3.80 |
| BoundedDegreeAdmission(K=16), tau calibrated | 15 | 0.3488 | 0.5435 | 11.91 |
| BoundedDegreeAdmission(K=32), tau calibrated | 15 | 0.3488 | 0.5435 | 12.25 |

A first, uncalibrated pass (same `tau` as the full graph, not shown
promoted here but printed by the benchmark as disclosed raw evidence)
found K=4/K=8 landing on 24 clusters against the full graph's 15 — a
confound (more clusters mechanically raises purity) that this ADR's
acceptance decision does **not** rely on. `calibrate_tau_bounded` (added
this run, mirroring ADR-344's own baseline-calibration precedent)
binary-searches `tau` per `K` to the same target; it could not hit 15
exactly for K=4/K=8 (18 and 23 — cluster count is not perfectly smooth in
`tau` for the bounded graph), so some residual budget mismatch remains
disclosed, not hidden.

At K=16 and K=32, the bounded graph's effective size
(`min(K, C) = min(16, 15) = 15` and `min(32, 15) = 15`) equals the full
graph exactly, so results are bit-identical to the full graph — this is
the predicted, unit-tested equivalence (see
`bounded_degree_matches_full_graph_when_k_covers_all_centroids`), and it
means *no cost benefit was observed at K=16/K=32 on this corpus*, because
this corpus's natural cluster count (15) never exceeds either K. At K=4
and K=8, where the approximation is actually exercised (C=18/23 > K),
quality diverges from the full graph by **8.9pp / 11.4pp purity** and
**2.0pp / 2.6pp recall** — in both cases *better*, not worse, than the
full graph on this corpus, but outside the pre-registered ±1pp
equivalence band either way, and confounded by the residual cluster-count
mismatch. **Quality-preservation claim: FAIL** for all four tested `K`.

**Reproducibility**: run twice, independently. Cluster counts, purity,
recall@10, and calibrated `tau` values were bit-identical between runs
(verified by diff, not asserted) — expected, since the dataset generator
and both policies use only a seeded LCG and plain `Vec` state, no
hash-order-dependent or concurrent structures. Wall-clock latency numbers
varied by up to ~20% between runs (expected measurement noise); the
*qualitative* findings (crossover band, relative ordering, equivalence at
K>=C) were stable across both runs.

**Overall acceptance: REJECT.** (`cargo run --release -p
ruvector-memory-admission --bin bounded_degree_benchmark` prints `REJECT`
and exits 1, by design — the binary's own exit code is this ADR's
acceptance signal, not a separately maintained claim.)

## Darwin Lineage

Bounded sweep: generation 1, 4 candidates (`K` in {4, 8, 16, 32}), budget
`generations<=3, candidates_per_generation<=4, max_promotions<=1` (used:
1 generation, 4 candidates, 0 promotions).

Fitness = `0.30*quality_norm + 0.25*latency_norm + 0.15*throughput_norm +
0.10*memory_norm + 0.10*correctness + 0.10*reproducibility`, with
`quality_norm = 1 - min(1, avg(|purity_delta_pp|, |recall_delta_pp|)/20)`,
`latency_norm = throughput_norm = max(0, (full_us - k_us)/full_us)`,
`memory_norm = correctness = reproducibility = 1.0` for every `K` (memory
footprint is determined by stored centroids, not by the cut graph's size;
all four pass the crate's full test suite; all four reproduced bit-exact
non-latency metrics above):

| K | quality_norm | latency_norm | fitness | Hard gate |
|---|---|---|---|---|
| 4 | 0.729 | 0.912 | **0.884** | FAIL (quality delta > 1pp) |
| 8 | 0.649 | 0.680 | 0.767 | FAIL (quality delta > 1pp) |
| 16 | 1.000 | -0.004→0 | 0.600 | FAIL (not faster; K>=C here) |
| 32 | 1.000 | -0.033→0 | 0.600 | FAIL (not faster; K>=C here) |

K=4 has the highest fitness score, driven almost entirely by its latency
weight (0.25+0.15=0.40 of the function) — and is exactly the candidate
Step 20's "never promote because fitness is high" rule exists to stop:
its fitness is high *because* it is fast, not because it is a correct
quality-preserving approximation. **No candidate is promoted.** Per the
harness's own rule ("if no Darwin candidate improves the parent, keep the
parent"), `MincutGatedAdmission` (the parent) remains this crate's
production-path answer, unchanged.

## Consequences

- `ruvector-memory-admission` gains a fourth, fully tested policy
  (`BoundedDegreeAdmission`) and a second benchmark binary
  (`bounded_degree_benchmark`), additive only — no change to
  `MincutGatedAdmission`, `AdaptiveMincutAdmission`,
  `NearestCentroidThreshold`, or the existing `benchmark` binary's
  acceptance gates.
- ADR-344's Next Research item 1 is answered: the O(C^3) ceiling is real
  (measured crossover band C=64–96 on this hardware/corpus) and a
  bounded-degree graph *does* close it computationally (up to ~2,335x at
  C=256) — but naively restricting to K nearest centroids without
  re-deriving the admission mechanism's quality behavior is not a safe
  drop-in fix. Both halves of that open question are now closed with
  measurement, not just closed by abandoning the idea.
- The confound found in the first, uncalibrated Part 2 pass, and the
  `calibrate_tau_bounded` fix for it, is retained in this crate's Flywheel
  record as a benchmark-methodology lesson for any future online-admission
  nightly comparing policies with different natural cluster counts.

## Alternatives

- **Promote `BoundedDegreeAdmission` anyway, citing the latency win** —
  rejected: this is exactly the reward-hacking-adjacent move Step 20
  exists to block (a fast candidate that fails the pre-registered
  correctness/quality gate is still a failed candidate, regardless of how
  attractive its fitness score is).
- **Relax `MAX_QUALITY_DELTA_PP` after seeing the 1pp band was missed** —
  rejected per the hard "never weaken an acceptance threshold to make an
  experiment pass" rule; the threshold was fixed before Part 2 ran.
- **Report only the matched-budget numbers, drop the confounded raw
  pass** — rejected: ADR-344 itself sets the precedent of disclosing an
  uncalibrated run as evidence rather than deleting it; this ADR follows
  that precedent.

## Implementation Plan / API Shape

`BoundedDegreeAdmission::new(tau: f32, max_clusters: usize, k_neighbors:
usize)` implements the existing `AdmissionPolicy` trait — no new trait,
no new public API surface beyond the struct itself and its benchmark-only
`from_centroids` constructor (mirrored on `MincutGatedAdmission` too, for
the scaling sweep). No feature flag: the module is always compiled
(the crate has no feature-gating convention to extend), but it is not
referenced from any other crate, so it has zero blast radius on anything
outside `ruvector-memory-admission`.

## Benchmark Evidence

`cargo run --release -p ruvector-memory-admission --bin
bounded_degree_benchmark` (also: `cargo run --release -p
ruvector-memory-admission --bin benchmark` for the pre-existing ADR-344
suite, unaffected by this change). Raw console output for both runs is
preserved in
`docs/research/nightly/2026-11-18-bounded-degree-mincut-admission/`.

## Security

No new attack surface: no new I/O, no new dependency (`Cargo.toml` adds
only a second `[[bin]]` entry), no concurrency change, no parsing of
untrusted input beyond the existing `&[f32]` vectors the trait already
accepted. `k_neighbors` is caller-controlled but bounded by an `assert!(>=
1)` and implicitly by `.min(self.centroids.len())` — cannot be used to
force unbounded work.

## Governance

Research-only; not wired into any production write path. No promotion
occurred, so no governance change to `ruvector-agent-memory` or any
consumer is implied by this ADR.

## Failure Modes

Same disclosed gaps as ADR-344 (concurrent writers, deletes — out of
scope for this PoC, unchanged). New to this ADR: `calibrate_tau_bounded`'s
binary search is not guaranteed to hit the target cluster count exactly
(cluster count is not perfectly monotonic/smooth in `tau` for the bounded
graph) — disclosed in Evidence above, not hidden.

## Migration / Rollback

None required — nothing is promoted. If `BoundedDegreeAdmission` is ever
revisited, this ADR's own fitness table and failed-gate reasons are the
starting lineage (do not re-run K in {4, 8, 16, 32} with the same
unmodified mechanism expecting a different quality result).

## Rejection Criteria (met)

Quality-preservation claim failing its pre-registered ±1pp band for every
tested `K`, as measured in Evidence above, is sufficient on its own to
reject promotion regardless of the latency result — which is exactly what
happened.

## Open Questions

1. Why does restricting to K nearest centroids change purity/recall in
   *either* direction at all, when the min-cut mechanism is otherwise
   identical? (The selected K-subgraph's min cut is a different graph
   from the full graph's — this is expected, not a bug — but this run
   does not explain the direction/magnitude of the effect beyond
   measuring it.)
2. Would a K-dependent `tau` that also conditions on *why* a point is
   locally ambiguous (not just a flat per-K calibrated constant) close the
   quality gap while keeping the latency win? (Candidate for a future
   nightly — see Next Research in the companion research document.)
3. `calibrate_tau_bounded` could not hit the exact target cluster count
   for K=4/K=8 — is there a cleaner calibration knob (e.g., jointly
   searching `tau` and a small `K` perturbation) that removes this
   residual confound entirely?
