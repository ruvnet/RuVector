# ADR-340: Structural-Time Memory Compaction

- **Status**: Proposed (experimental `CompactionPolicy`, not default-on)
- **Date**: 2026-08-29
- **Deciders**: RuVector Nightly Research Harness
- **Related**: ADR-251 (`emergent-time` — Structural Proper Time and the
  physics-formalism clocks this ADR reuses unmodified); the `CoherencePolicy`
  nightly result in `crates/ruvector-agent-memory/src/compaction.rs` (extends,
  does not replace); ADR-320/ADR-307 (the ledger/observation substrate that
  could source a real validation trace, see Open Questions)
- **Tags**: agent-memory, emergent-time, compaction, recency, nightly-research

## Status

Experimental. One nightly run, one synthetic benchmark, ACCEPT verdict on
pre-registered thresholds, reproducible across 8 seeds. Not validated
against real agent traces. See `docs/research/nightly/2026-08-29-structural-time-memory-compaction/README.md`
for full methodology, raw results, and disclosed limitations.

## Context

`ruvector-agent-memory::CoherencePolicy` (a prior nightly contribution)
scores each memory's retention importance as
`α·recency + β·frequency + γ·coherence`, where `recency` is normalized
`last_accessed_at` — a wall/step clock incremented by exactly one on every
insert or access, independent of what actually happened.

That normalization is `(t - min) / (max - min)` over the whole store. If an
agent's activity is bursty — dense writing sessions separated by long idle
periods (waiting on a human, a slow external call, a scheduled pause) — a
single idle gap that's orders of magnitude larger than the ticks spent on
real writes dominates that range. Every memory written *before* the gap
gets crushed toward `recency ≈ 0`, indistinguishable from genuinely stale
memories regardless of true relevance. `emergent-time` (a separate,
dependency-free RuVector crate; `ADR-251`) already implements a
clock that doesn't have this problem by construction: **Structural Proper
Time**, internal time as the arc length of a system's trajectory through
its own state space — an idle period with no state change contributes zero
internal time. That crate frames the idea generally (four physics
formalisms plus an "Agentic Time" diagnostic layer for health/alarm
classification) but nothing in the ecosystem had wired it into a
production-shaped retention *decision* before this run.

## Hypothesis

```text
Given a memory stream with a long idle gap between a dense "phase 1"
writing burst and a smaller "phase 2" burst right after the agent returns,

when compaction recency is computed from Structural Proper Time (embedding
arc length through the memory stream) instead of last_accessed_at,

then Recall@10 for held-out queries about the end of phase 1 should exceed
the wall-clock CoherencePolicy baseline by at least 3.0 percentage points,

subject to: on a steady (no-idle-gap) control workload of the same size,
the structural-time policy must not regress recall by more than 1.0
percentage point relative to the baseline.
```

Thresholds were fixed in the benchmark source before being evaluated
against the final dataset design (see the README's disclosed "Methodology
note" for the one dataset-discriminativeness iteration that preceded any
acceptance-number computation).

## Decision

Add two new `CompactionPolicy` implementations to
`crates/ruvector-agent-memory`, additive and independently selectable —
**not** a change to `CoherencePolicy`'s default behavior:

- `StructuralTimePolicy` (candidate A) — identical `CoherenceWeights`
  formula, recency from `emergent_time::structural_clock::StructuralProperTime`
  over the entries' embeddings (write order), with only the embedding
  channel weighted (`StructuralMetric::w_embedding = 1.0`, all other
  channels `0.0` — this crate has no honest per-entry entropy, graph, or
  prediction-error signal, so those weights are left at zero rather than
  fabricated).
- `GatedStructuralTimePolicy` (candidate B) — the same, with
  `StructuralMetric::gate` set above zero so embedding movement below the
  gate contributes no structural time (tests whether treating jitter as
  idle-equivalent noise helps).

Add `MemoryStore::advance_clock(ticks)` (a minimal, additive method) so
benchmarks and callers can represent an idle period explicitly.

Add `emergent-time` as a path dependency of `ruvector-agent-memory` — it is
itself dependency-free, so this adds zero new transitive dependencies.

## Evidence

Synthetic benchmark (`examples/temporal_compaction_bench.rs`): 2 000
memories / 20 clusters / 64-dim, 15 "phase 1" clusters written densely,
either a 500 000-tick idle gap (bursty-idle) or none (steady control), then
5 "phase 2" clusters. Held-out queries probe the last 3 phase-1 clusters;
the compaction context window is drawn only from phase-2 topics so
coherence cannot leak the evaluation answer. Compaction to 700/2 000 (35%).

```
[bursty-idle]                     Recall@10
CoherenceWeighted (baseline)          27.2%
StructuralTime (candidate A)          59.0%   (+31.8pp)
GatedStructuralTime (candidate B)     59.0%   (+31.8pp)

[steady control]                  Recall@10
CoherenceWeighted (baseline)          59.0%
StructuralTime (candidate A)          59.0%   (+0.0pp)
```

Acceptance: [A] +31.8pp ≥ 3.0pp margin → **PASS**. [B] same → **PASS**.
[C] steady-workload regression +0.0pp, tolerance −1.0pp → **PASS**.
**Verdict: ACCEPT.**

Direction (`StructuralTime > CoherenceWeighted` on the bursty-idle
workload) reproduces across 8 independently seeded dataset draws
(`structural_time_wins_across_multiple_seeds`,
`cargo test -p ruvector-agent-memory --example temporal_compaction_bench`).

Reproducibility seal: FNV-1a hash over `(seed, params, rounded recall
values)`, reusing `emergent_time::witness::fnv1a64` (no new hash
introduced) — `e0d3b9cf5b37176e`, reproduced byte-for-byte across the runs
used to write this report.

Disclosed honest counter-signal: plain `LruPolicy` (no coherence term)
also beats `CoherenceWeighted` on the bursty-idle workload (66.7% vs
27.2%) — not the claim being made (LRU isn't a like-for-like comparison),
but reported because it shows the coherence term, when its context is
realistically unrelated to what's later queried, actively degrades an
otherwise-cleaner recency cutoff under the idle gap, which `StructuralTime`
(same coherence formula) does not. See the research README for the full
"reading the numbers honestly" discussion, including that
`GatedStructuralTimePolicy` is not shown to add anything beyond
`StructuralTimePolicy` on this dataset (the gate never binds at the tested
noise level).

## Consequences

**Positive**:
- A `CompactionPolicy` variant demonstrably more robust to bursty-idle
  workloads than the wall-clock baseline, on the tested synthetic case, at
  ~10-15% compaction-latency overhead over `CoherenceWeighted`.
- Zero new dependencies; reuses `emergent-time`'s already-tested clock
  primitives and hash function rather than introducing new ones.
- Purely additive: existing callers of `CoherencePolicy` are unaffected.

**Negative / open risk**:
- Validated only on a synthetic, stylized worst-case dataset (one gap
  magnitude, one two-phase topic-disjoint shape). No real agent trace has
  been tested.
- `GatedStructuralTimePolicy` carries the maintenance cost of a second
  policy type without a demonstrated benefit distinct from candidate A.
- Structural time is `O(n)` extra work per compaction (one L2-distance pass
  over embeddings); acceptable at the tested scale (2 000 entries, ~6.5ms)
  but not characterized at larger scale.

## Security / Validation Gates

No new attack surface: this is a pure scoring-function change inside an
existing, already-gated compaction call (`compact()`); no new external
input, no new authority, no new serialization format. The acceptance
thresholds are constants in the benchmark source, evaluated by test code
that shares no state with the candidate policies (independent
evaluator/candidate, per repo research-hygiene convention). No reward-hack
surface identified: the benchmark's ground truth (brute-force top-10 over
the pre-compaction store) and dataset generator are fixed before any
policy result is computed, and neither `StructuralTimePolicy` nor
`GatedStructuralTimePolicy` has any path to influence the benchmark's own
acceptance thresholds or dataset.

## Affected Repos / Crates

`crates/ruvector-agent-memory` (new module + example + one new method on
`MemoryStore` + new dependency edge to `crates/emergent-time`). No other
crate is touched.

## Dependencies

`emergent-time` (path dependency, already an in-workspace crate, itself
dependency-free — no new transitive dependencies enter the workspace).

## Alternatives Considered

- **Entropy-based recency channel** (`emergent_time::structural_clock::EntropyClock`):
  rejected for this run because the crate has no honest per-entry entropy
  signal to feed it; fabricating one (rather than reusing a tested
  primitive with a real input) would have violated the "no fabricated
  benchmark data" constraint. Left as future work (see README "Next
  research").
- **`WindowedDeltaClock` from `emergent_time::agentic_time`**: uses a
  different, six-channel `AgentState`/`AgentClock` type this crate cannot
  honestly populate (belief/memory/retrieval/goal-graph/contradiction/plan
  are not distinct signals available per memory entry here); rejected in
  favor of the simpler, single-channel `structural_clock` API that matches
  what this crate actually has.
- **Do nothing / keep `CoherencePolicy` as the only coherence-aware
  policy**: rejected because the idle-gap failure mode is real, measurable,
  and directly addressed by an already-existing, already-tested RuVector
  primitive (`emergent-time`) that no other component was using for this
  purpose.

## Open Questions

1. Does the effect replicate on real agent-memory traces? (This crate's
   `ledger`/`observation` modules already capture timestamped, replayable
   history that could source such a validation corpus — not attempted in
   this run.)
2. What gap-size-to-activity-volume ratio is the threshold below which the
   effect disappears? Untested — this run used one gap magnitude (250× the
   phase-1 tick budget).
3. Does wiring an honest second structural channel (e.g. `ΔG` from
   `ruvector-mincut`) change candidate B's (currently inert) gate behavior?
4. Should promotion to default-on ever happen, or should this remain an
   opt-in policy selected per-deployment based on measured activity
   burstiness?

## Rejection Criteria

This ADR's premise is rejected if a real-trace validation (Open Question 1)
shows no material recall difference between `CoherencePolicy` and
`StructuralTimePolicy` — i.e., if the synthetic worst case constructed here
does not reflect any real agent activity pattern. Until that validation
exists, this remains Proposed/experimental, not a basis for default-path
changes.
