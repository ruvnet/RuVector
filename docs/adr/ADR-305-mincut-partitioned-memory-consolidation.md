# ADR-305: Mincut-Partitioned Agent-Memory Consolidation

## Status

**Proposed, hypothesis REJECTED by measurement.** Experimental crate
(`ruvector-partition-memory`), not wired into any production compaction
path. Retained for its evidence, its two documented `ruvector-mincut`
defects, and its from-scratch correct min-cut implementation
(`mincut_exact.rs`), which is a candidate for reuse in future partitioning
work regardless of this ADR's own outcome.

## Context

Nightly 2026-06-14 (`crates/ruvector-agent-memory`) introduced
`CoherencePolicy`: a global top-score compaction rule scoring every stored
memory by `α·recency + β·frequency + γ·coherence(context)` and keeping the
top `target_size`. It measured 100% recall after 50% compaction on its
test corpus and remains the best-performing policy in the ecosystem.

`CoherencePolicy` is, by construction, a single global ranking scored
against one context window (in production, the agent's most recent working
context). That is also its structural risk: a memory topic unrelated to
the current context competes on the same scale as everything else, so a
minority topic can be evicted **in full** during a single consolidation
event, at a compaction ratio the aggregate recall number reports as
favorable. Nightly 2026-06-14 did not measure this — it reports mean
recall and LRU/LFU comparisons, not worst-topic behavior.

`ruvector-mincut` provides a subpolynomial dynamic minimum-cut engine
(Jin–Sun–Thorup) and graph-partitioning utilities
(`GraphPartitioner`, `RuVectorGraphAnalyzer`) that had not previously been
applied to agent memory. This ADR's premise: partitioning the memory
similarity graph before applying a retention budget, with a
per-partition floor, should protect a topic from being evicted in full
even when it loses on the global score — because that topic's competition
for its floor allocation is only the rest of its own partition, not the
whole corpus.

## Hypothesis

```text
Given a 4,000-memory corpus with 6 semantic clusters of unequal size
(1400/1000/720/600/200/80 — the two smallest are 5% and 2% of the corpus),
scored against a recency-biased context drawn from the largest cluster
(the realistic "what the agent was just working on" scenario),

when a partition-aware retention policy (floor + proportional budget per
graph partition) is used instead of CoherencePolicy's global top-score
ranking, at 50% compaction,

then the best candidate's worst-cluster recall@10 should exceed the
baseline's by >= 15 percentage points,

subject to: no candidate's overall recall@10 regressing more than 5pp
below baseline; partition+retention wall time staying under 30s per
candidate at this n; and the partition witness chain verifying.
```

Declared before the accepted run (see the research doc's Pass 2/3 and
calibration section); not modified afterward.

## Decision

Implement two partitioning strategies and compare both against the
`CoherencePolicy` baseline, using **the same scorer** in every retention
step so the only independent variable is budget allocation, not scoring:

- **Candidate A — `MincutFixedK`**: wraps the existing
  `ruvector_mincut::GraphPartitioner` (unweighted, edge-count recursive
  bisection, fixed `K`).
- **Candidate B — `MincutAdaptive`**: a new adaptive-depth recursive
  bisection that stops splitting a component once its cut is dense
  relative to its internal edge weight (no caller-chosen `K`).
- **Retention**: floor + largest-remainder proportional budget per
  partition (`retention.rs`), each partition ranked internally by
  `ruvector_agent_memory::CoherencePolicy` — reused as a library
  dependency, not re-implemented.

## Evidence

### A defect discovered before the hypothesis could be tested

`DynamicMinCut::partition()` (and the `GraphPartitioner` /
`RuVectorGraphAnalyzer` path built on it) was found, during this
candidate's own development, to return vertex splits **inconsistent with
its own `min_cut_value()`**, and nondeterministically so:

- 6-vertex repro (two triangles joined by one weak `0.05`-weight bridge;
  true min cut is uniquely `{0,1,2}` vs `{3,4,5}` at value `0.05`): of
  three runs, two returned the correct split, one returned a degenerate
  `{single vertex}` vs `{rest}` split — while `min_cut_value()` reported
  `0.05` correctly on **every** run.
- 100-vertex version (two 50-cliques, one `0.01`-weight bridge): every run
  returned the degenerate split; `min_cut_value()` still correctly
  reported `0.01`.
- `GraphPartitioner` was separately found to (a) drop vertices outright at
  n=100 (returned partitions covering only 50 of 100 vertices) and (b)
  fabricate vertex ids that were never in the input graph at all, when the
  id space is non-contiguous.
- `GraphPartitioner` was also measured to be severely slow: **8.4s at
  n=500**, and **did not finish in 5m42s at n=4000** (killed).

Full repro commands are in the research doc. This crate works around the
correctness defects with a from-scratch, tested Stoer–Wagner
implementation (`mincut_exact.rs`) used as the sole source of partition
vertex sets; `ruvector_mincut`'s `min_cut_value()` is still queried as an
independent cross-check (its *value* output, as opposed to its
*partition*, was never observed wrong). It works around the performance
defect by scale-gating candidate A (`fixed_k_max_n`, default 600) rather
than hanging the benchmark or silently omitting the comparison.

### The accepted hypothesis run (n=4000, `coherence_ratio=0.35`, `floor_min=3`)

```text
variant          overall_recall  worst_cluster_recall  coverage
GlobalTopScore   0.4193          0.1520                 1.000
MincutAdaptive   0.4873          0.1520                 1.000

per_cluster_recall GlobalTopScore = [0.996, 0.152, 0.216, 0.316, 0.396, 0.440]
per_cluster_recall MincutAdaptive = [0.792, 0.380, 0.504, 0.152, 0.556, 0.540]

worst_cluster_gain_pp = -0.00  (threshold: +15.00)
ACCEPTANCE_RESULT: REJECT
```

Overall recall improved (+6.8pp) and 4 of 6 clusters gained materially
(+15 to +23pp each), but the specific cluster that was *worst* under the
baseline (cluster 3, 600 members / 15% of the corpus) is **also** worst
under `MincutAdaptive`, at the identical value — because the partitioner
left cluster 3 merged with the 1400-member majority cluster (the 2000-size
partition in `sizes=[200, 2000, 1000, 720, 80]`), so its retention budget
was decided by the same global-style competition the hypothesis set out
to avoid. The `coherence_ratio=0.35` stopping rule, calibrated before this
run against the corpus's true global min cut (see the research doc), does
correctly find and isolate the genuinely weak seams — but cluster 0/3's
separation was not one of them at this threshold.

At n=500, both candidates were run (`fixed_k_max_n=600` admits n=500):
`MincutFixedK` reached `worst_cluster_gain_pp=8.00`, `MincutAdaptive`
reached a *worse* worst-cluster recall than baseline (`0.0` vs `0.10`,
because the true 10-member minority cluster is below `min_cluster_size`
(20) and can never be isolated on its own). Both REJECT.

A bounded, pre-declared-fitness sweep of `floor_min` over `{1,3,8,15}` at
n=4000, holding the same partition fixed, left `worst_cluster_recall`
essentially flat (`0.152`/`0.148` across all four values) — confirming
the bottleneck is the **partition step**, not the **retention-budget
step**: no floor value can protect a cluster the partitioner never
separated from the majority in the first place.

## Consequences

- **Do not promote** `MincutAdaptive`/`MincutFixedK` retention to
  production. The pre-registered hypothesis (worst-cluster recall
  protection) is rejected by direct measurement.
- `mincut_exact.rs`'s correct, tested Stoer–Wagner implementation is a
  reusable asset independent of this ADR's outcome — any future graph-cut
  work in this ecosystem needing a trustworthy partition should use it, or
  a fixed `ruvector-mincut`, in preference to `DynamicMinCut::partition()`
  as it stands today.
- The `ruvector-mincut` defects (partition/value inconsistency,
  nondeterminism, vertex loss/fabrication, severe `GraphPartitioner`
  latency) should be filed and fixed upstream in that crate; they affect
  every existing consumer of `DynamicMinCut::partition()` /
  `GraphPartitioner`, not just this experiment.
- A follow-up hypothesis worth testing (not implemented here): a
  **per-branch, not global**, stopping criterion — e.g. always attempt at
  least one more level of recursion on the largest remaining partition
  before accepting `coherence_ratio`'s verdict, or size-weight the
  threshold — might separate cluster 0/3 where the flat threshold did
  not. This is a new hypothesis, not a retroactive change to the one
  tested above.

## Alternatives

- **Ship `CoherencePolicy` unchanged.** Current state; the measured
  overall-recall improvement here (+6.8pp) does not offset a rejected
  primary hypothesis and a partitioner with two unresolved upstream
  correctness defects and a severe latency defect.
- **Global top-score with a per-cluster-label floor** (using a cheap
  clustering method like k-means on embeddings instead of graph min-cut)
  was considered but not implemented; it would sidestep `ruvector-mincut`
  entirely and is a reasonable next candidate.

## Implementation plan

Not applicable — hypothesis rejected; no production migration.

## API shape

`ruvector-partition-memory` (experimental, workspace member, not
re-exported by any production crate): `corpus`, `graph`, `mincut_exact`,
`partition`, `retention`, `metrics`, `witness`, `search` modules; see
`src/lib.rs` for the full surface.

## Feature flags

None; the crate is not on any production feature-gated path.

## Benchmark evidence

`docs/research/nightly/2026-08-17_mincut-partitioned-memory-consolidation/evidence/`
— raw, unedited command output: `bench_n4000.txt`, `bench_n500_with_fixedk.txt`,
`darwin_sweep.txt`, `calibration.txt`.

## Security

No new attack surface: the crate is a standalone research binary/library
operating on synthetic data, not wired into any request path. The witness
chain (`witness.rs`) is a correctness/audit mechanism, not an access
control mechanism, and makes no such claim.

## Governance

None of this crate's code should be treated as validated production
guidance for `ruvector-mincut` usage beyond the specific defects
documented above; those defects should be independently verified by
whoever owns that crate before any fix lands.

## Failure modes

- `DynamicMinCut::partition()` / `GraphPartitioner` defects: see Evidence.
- `AdaptiveConfig::min_cluster_size` (default 20) structurally prevents
  isolating any true topic smaller than that absolute count — observed
  directly at n=500 (10-member cluster, worst_cluster_recall=0.0).
- A coarse, single-threshold stopping rule can leave two clusters merged
  even when one is a minority worth protecting, if their graph-structural
  separation is weaker than the threshold demands elsewhere in the same
  corpus (observed at n=4000, clusters 0/3).

## Migration

None.

## Rollback

None — nothing shipped to a production path.

## Rejection criteria

Met: worst-cluster recall gain (0.00pp, both n=4000 and n=500) fell short
of the pre-declared 15pp threshold in every configuration tested,
including a bounded post-hoc sweep of the one parameter (`floor_min`)
that could plausibly have rescued it without changing the hypothesis
itself.

## Open questions

- Would a per-branch/size-weighted stopping criterion (see Consequences)
  cross the threshold? Untested — a genuinely new hypothesis for a future
  nightly, not this one.
- Do the two `ruvector-mincut` defects reproduce on that crate's own
  existing test suite, or does no existing test exercise
  `DynamicMinCut::partition()` / `GraphPartitioner::partition()`'s output
  against ground truth? Not investigated here; worth checking before
  filing upstream.
