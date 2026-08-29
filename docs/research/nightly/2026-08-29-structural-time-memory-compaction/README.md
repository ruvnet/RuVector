# Structural-Time Memory Compaction

**Status**: PoC complete — **positive result**, ACCEPT.

## Summary

`ruvector-agent-memory`'s `CoherencePolicy` (a prior nightly result) scores
each stored memory's importance as `α·recency + β·frequency + γ·coherence`,
where `recency` is `last_accessed_at` — a wall/step clock — min-max
normalized over the whole store. This run tests whether swapping that
recency term for `emergent-time`'s **Structural Proper Time** (internal time
that accumulates with embedding movement, not clock ticks) improves memory
retention for agents whose activity is **bursty**: dense writing sessions
separated by long idle gaps (waiting on a human, a slow tool call, a
scheduled pause).

**Result**: on a synthetic bursty-idle workload, wall-clock `CoherencePolicy`
retains 27.2% of the queries that should have survived compaction;
structural-time recency retains 59.0% (+31.8pp) — while on a matched
steady-activity control workload (no idle gap) the two are statistically
identical (59.0% vs 59.0%, 0.0pp). The effect is directionally reproducible
across 8 independent random seeds. **ACCEPT.**

## Hypothesis

```text
Given a memory stream with a long idle gap between a dense "phase 1"
writing burst and a smaller "phase 2" burst right after the agent returns,

when compaction recency is computed from Structural Proper Time (embedding
arc length through the memory stream) instead of last_accessed_at,

then Recall@10 for held-out queries about the end of phase 1 (what the
agent was working on right before the gap) should exceed the wall-clock
CoherencePolicy baseline by at least 3.0 percentage points,

subject to: on a steady (no-idle-gap) control workload of the same size,
the structural-time policy must not regress recall by more than 1.0
percentage point relative to the baseline.
```

Both thresholds were fixed in the benchmark source before any acceptance
number was computed (see "Methodology note" below for the one dataset
iteration that *was* needed, and why it doesn't compromise this).

## Why this matters in 2026

Agent memory systems (RAG assistants, coding agents, long-running
autonomous workflows) are not busy uniformly. They burst — a flurry of
tool calls and writes, then minutes or hours of silence waiting on a human,
a scheduled job, or an external system. Every recency-based memory
management scheme shipped today (LRU, TTL, "last N messages") uses wall
time. This run shows a concrete, measurable failure mode of that design:
one long idle period can make an entire pre-gap working set statistically
indistinguishable from ancient history, regardless of how relevant it
actually is when the agent returns.

## Why it could matter in 2036 / 2046

As agentic systems move toward always-on, multi-day, multi-week
deployments — personal assistants, autonomous research loops, edge/robotic
agents that sleep between missions — the ratio of idle-to-active time only
grows. A memory substrate whose notion of "time" is decoupled from wall
clock and instead tracks *how much actually changed* is a primitive that
generalizes past this one compaction policy: it's the same idea underlying
proper time in physics, applied to computational history. `emergent-time`
already frames this generally (physics formalisms + "Agentic Time"); this
run is the first place in the RuVector ecosystem it is wired into a
production-shaped memory subsystem rather than used for diagnostics/alarms.

## RuVector ecosystem fit

- **`ruvector-agent-memory`** — the compaction subsystem being extended.
- **`emergent-time`** — supplies `structural_clock::{Clock, StateSnapshot,
  StructuralProperTime, StructuralMetric}` and `witness::fnv1a64`; both
  reused as-is, zero new hashing/clock primitives introduced.
- **Vector coherence scoring** (`ruvector-agent-memory::scoring`) — the
  `CoherenceWeights` formula and `coherence_score` function are reused
  unmodified; only the recency input changes.
- **Witness/provenance** — the benchmark's raw results are sealed with an
  FNV-1a hash over `(seed, params, rounded recall values)` using the exact
  hash function `emergent-time`'s own training-witness chain uses, so a
  re-run can be checked byte-for-byte against this report.

Four ecosystem pieces connect: agent memory, emergent time, vector
coherence scoring, and witness/provenance — without introducing a single
new algorithm; this is a new *composition* of existing, already-tested
RuVector primitives, per the novelty gate.

## MetaHarness / Flywheel / Darwin status (verified, not assumed)

This run checked, rather than assumed, tool availability:

- `npx metaharness --help` — **installed** (scaffolding CLI for generating
  new harness *projects*; `npx metaharness score/analyze/genome <repo>`
  exist but are for scoring a target repo from the outside, not an
  in-repo research orchestration daemon). Not used to drive this run — the
  work here is a single bounded hypothesis, not a multi-project harness
  build, so the scaffolding tool's surface area doesn't apply.
- `npx ruvector harness doctor/status/flywheel/darwin --json` — **not
  installed** (`npm error could not determine executable to run`; no
  `ruvector` CLI package resolves an in-repo "harness" subcommand). No
  Flywheel/Darwin/gate CLI exists to call in this repository today, so no
  such call is fabricated in this report. The roles those tools would
  play (evidence retention, bounded evolutionary search, promotion gating)
  were instead executed manually and directly, as described below.
- The evaluator (this benchmark) and the candidate (`StructuralTimePolicy`
  / `GatedStructuralTimePolicy`) are independent code paths reusing a
  pre-existing, unmodified scoring formula (`CoherenceWeights`) — the
  candidate cannot see or influence the acceptance thresholds, which are
  fixed constants in the benchmark source.

## Architecture

```mermaid
flowchart LR
    subgraph Write["Write path (unchanged)"]
        A[Agent writes memory] --> B[MemoryStore::insert]
    end

    subgraph Compact["Compaction (candidate swap)"]
        B --> C{CompactionPolicy}
        C -->|baseline| D["CoherencePolicy<br/>recency = last_accessed_at<br/>(wall/step clock)"]
        C -->|candidate A| E["StructuralTimePolicy<br/>recency = StructuralProperTime<br/>(embedding arc length)"]
        C -->|candidate B| F["GatedStructuralTimePolicy<br/>same, with a jitter gate"]
    end

    D --> G[importance = α·recency + β·freq + γ·coherence]
    E --> G
    F --> G
    G --> H[keep top target_size]
    H --> I[Recall@10 on held-out queries]
```

`StructuralTimePolicy` and `GatedStructuralTimePolicy` reuse
`CompactionPolicy`, `CoherenceWeights`, and `coherence_score` unchanged;
only `structural_recency()` (in `src/temporal_compaction.rs`) replaces the
`last_accessed_at`-based recency term with
`StructuralProperTime::cumulative()` over the entries' embeddings in write
order (only `w_embedding` is weighted — the crate has no honest per-entry
entropy/graph/prediction-error channel, so those `StructuralMetric` weights
are left at `0.0` rather than fabricated).

## Implementation

New/changed files, all in `crates/ruvector-agent-memory/`:

- `Cargo.toml` — adds `emergent-time = { path = "../emergent-time" }`
  (dependency-free crate; zero new transitive dependencies).
- `src/memory.rs` — adds `MemoryStore::advance_clock(ticks)`: advances the
  logical clock without touching any entry, modeling an idle period.
- `src/temporal_compaction.rs` (new, 246 lines) — `StructuralTimePolicy`
  (candidate A) and `GatedStructuralTimePolicy` (candidate B), plus 3 unit
  tests.
- `src/lib.rs` — registers the module and exports.
- `examples/temporal_compaction_bench.rs` (new) — the benchmark below, plus
  3 embedded acceptance/robustness tests run via `cargo test --example
  temporal_compaction_bench`.

## Benchmark methodology

- **Hardware/OS**: linux/x86_64 (container), Rust 1.94.1, cargo 1.94.1,
  `cargo build --release`.
- **Dataset** (deterministic, `StdRng` seeded): 20 topic clusters × 100
  entries = 2 000 memories, 64-dim unit vectors, cluster members drawn as
  `normalize(centroid + noise·N(0,1))` with noise 0.35 (identical generator
  to the crate's original `agent-memory-bench`).
  - **Phase 1** (15 clusters, 1 500 entries): the dense "before the break"
    burst.
  - **Idle gap**: `MemoryStore::advance_clock(500_000)` ticks — two orders
    of magnitude larger than the ~2 000 ticks spent on real writes — between
    phase 1 and phase 2. The steady control workload uses the identical
    generator with the gap set to 0; every other parameter, seed, and piece
    of randomness draw is unchanged.
  - **Phase 2** (5 clusters, 500 entries): the "current activity" burst
    right after the agent returns.
  - **Recall clusters**: the last 3 clusters written in phase 1 (indices
    12–14 of 0–14) — "what the agent was working on right before the
    break." 60 held-out queries (20/cluster) perturbed from each recall
    centroid; ground truth is the exact top-10 nearest neighbors in the
    *pre-compaction* store (brute-force cosine).
  - **Context window** (what `CoherencePolicy`'s coherence term sees): 20
    vectors drawn only from **phase-2** centroids. This deliberately never
    overlaps the recall clusters, so coherence cannot leak the evaluation
    answer — it actively favors phase 2, isolating recency as the only
    signal that can save phase-1 memories.
  - Access counts are left at 0 for every entry (no synthetic access
    simulation), so the `frequency` term contributes equally (0) to every
    policy, isolating the recency signal being tested.
- **Compaction**: target size 700/2 000 (35%). A 50% ratio was tried first
  and left *every* policy, including plain LRU, at a 100% recall ceiling —
  not discriminative. The ratio was tightened to 35% before any acceptance
  numbers were computed (see "Methodology note").
- **Policies compared**: `LruPolicy`, `CoherencePolicy` (baseline, wall
  clock), `StructuralTimePolicy` (candidate A), `GatedStructuralTimePolicy`
  (candidate B, jitter gate 0.05 L2 units).
- **Reproduce**:
  `cargo run --release -p ruvector-agent-memory --example temporal_compaction_bench`

### Methodology note (disclosed, not hidden)

The first benchmark design used a 50%-compaction ratio and a context window
drawn from the recall clusters themselves. Both choices made the benchmark
non-discriminative: every policy, including plain LRU (no coherence term at
all), retained 100% of the recall queries, because (a) 50% of 2 000 easily
accommodates the ~300 relevant entries regardless of policy, and (b) a
context window matching the recall topic let coherence alone carry every
policy to ceiling, and would have amounted to leaking the evaluation answer
into the compaction signal. Both were fixed — tighter compaction ratio,
context window restricted to the unrelated phase-2 topic — *before* the
acceptance thresholds were evaluated on the redesigned dataset; the 3.0pp /
1.0pp thresholds themselves were never touched. A second, unrelated bug was
also caught before any acceptance number was recorded: the benchmark
initially rebuilt each policy's store via `MemoryStore::insert`, which
silently re-timestamps entries from a fresh sequential clock — discarding
the idle-gap wall-clock information entirely before any policy ever saw it.
Fixed by copying entries verbatim (`MemoryStore::replace_entries`).

## Results (raw, from the run committed with this report)

```
Dataset
  Memories            : 2000 (20 clusters x 100)
  Dimensions          : 64
  Recall clusters     : last 3 clusters before the gap/tail
  Queries             : 60 (K=10)
  Target size         : 700 (35% compaction)
  Idle gap            : 500000 ticks (steady workload: 0)

[bursty-idle]
Policy                    Recall@10    Compaction (µs)
--------------------------------------------------------
LRU                           66.7%               196
CoherenceWeighted             27.2%              5965
StructuralTime                59.0%              6551
GatedStructuralTime           59.0%              6688

[steady (control)]
Policy                    Recall@10    Compaction (µs)
--------------------------------------------------------
LRU                           66.7%                56
CoherenceWeighted             59.0%              5851
StructuralTime                59.0%              6696
GatedStructuralTime           59.0%              6479

Acceptance test (pre-registered thresholds)
  [A] StructuralTime beats CoherenceWeighted by >= 3.0pp on bursty-idle: +31.8pp -> PASS
  [B] GatedStructuralTime beats CoherenceWeighted by >= 3.0pp on bursty-idle: +31.8pp -> PASS
  [C] StructuralTime does not regress > 1.0pp vs CoherenceWeighted on steady control: +0.0pp -> PASS

  Verdict: ACCEPT

Witness (FNV-1a over seed+params+rounded recall values): e0d3b9cf5b37176e
```

**Robustness**: a separate test (`structural_time_wins_across_multiple_seeds`)
re-runs the bursty-idle comparison for 8 independent seeds
(`1, 2, 3, 4, 5, 340, 7777, 99991`) and asserts `StructuralTime > CoherenceWeighted`
recall on every one. All 8 pass (`cargo test -p ruvector-agent-memory --example
temporal_compaction_bench`).

### Reading the numbers honestly

- **Candidate A vs baseline (the core claim)**: CoherenceWeighted collapses
  to 27.2% under the idle gap; StructuralTime holds 59.0% — identical to its
  own steady-workload number. Structural time is *unaffected* by the gap by
  construction (it doesn't see wall-clock ticks at all); wall-clock recency
  is severely degraded by it.
- **Candidate B adds nothing measurable here**: `GatedStructuralTimePolicy`
  scores identically to `StructuralTimePolicy` in both workloads (59.0% /
  59.0%). The 0.05 L2 jitter gate never binds, because this dataset's
  perturbation noise (0.35) puts nearly every consecutive embedding delta
  above the gate. This is reported as a genuine negative/neutral
  sub-result, not hidden: gating is not shown to help *or* hurt in this
  experiment, only to be inert.
- **An honest surprise — plain LRU also beats CoherenceWeighted here (66.7%
  vs 27.2%)**: this is *not* the paper's claim (LRU has no coherence term
  and is not a fair like-for-like comparison — it happens to keep the
  literal most-recent 700 entries, which incidentally covers 2 of the 3
  recall clusters completely). It's disclosed because it's informative: the
  coherence term, when its context window is (realistically) unrelated to
  what's being evaluated, doesn't just fail to help under the idle gap — it
  actively dilutes an otherwise-cleaner recency cutoff, by tying together
  1 500 already wall-clock-crushed candidates into an unstable-sort
  tiebreak. `StructuralTime`, using the *identical* coherence formula, does
  not suffer this because its recency term isn't crushed to begin with.
- **Latency**: `StructuralTime`/`GatedStructuralTime` cost ~6.5–6.7ms vs
  CoherenceWeighted's ~5.9ms per compaction of 2 000 entries (≈10–15%
  overhead from the extra O(n) arc-length pass) — both dwarfed by
  `coherence_score`'s O(n·|context|) cost, which dominates all three
  coherence-aware policies. `LruPolicy` (no scoring) is ~30× faster than
  any coherence-aware policy, as expected — an orthogonal latency/quality
  tradeoff, not something this run claims to change.

## Failure modes / limitations (explicitly not claimed)

- **Synthetic dataset only.** No real agent trace was used. The bursty-idle
  structure (one large gap, two topic-disjoint phases) is a stylized
  worst case chosen to isolate the mechanism cleanly, not a measured
  real-world access pattern.
- **Single dataset shape.** Robustness was checked across random seeds of
  *this* generator, not across different cluster counts, dimensionalities,
  gap sizes, or multi-gap traces. A gap much smaller than the real-activity
  tick budget would show a much smaller (or no) effect; this run does not
  characterize that curve.
- **No honest entropy/graph/prediction-error channel.** `StructuralMetric`
  supports five channels; only the embedding channel is populated here
  because this crate has no other real signal per memory entry. A future
  run wiring in `ruvector-mincut` (for `ΔG`) or per-entry surprise (for
  `ΔE`) would be a different, not-yet-tested variant.
- **`GatedStructuralTimePolicy` is not shown to add value** on this
  dataset — see above. It should not be read as validated; only as "did
  not hurt."
- **No production integration.** This is a library-level `CompactionPolicy`
  addition with a standalone benchmark; it is not wired into any live
  agent-memory deployment, MCP surface, or RVF/RVM path.

## RVF / RVM / ruFlo / MCP analysis

- **RVF**: `StructuralTimePolicy`'s per-entry recency values could be
  serialized alongside a memory snapshot as part of a portable RVF
  cognitive package, letting a restored agent recompute compaction
  decisions consistently across a suspend/resume or edge-transfer boundary
  without needing wall-clock alignment between source and destination.
  Not implemented here — this is a plausible follow-on, not a claim.
- **RVM**: no material fit identified. This is a scoring-function swap
  inside a single compaction call, not a capability boundary, isolation
  domain, or privileged-operation surface; forcing an RVM integration here
  would add complexity with no isolation benefit.
- **ruFlo**: a concrete, useful autonomous workflow — an idle-gap detector
  (already present in `emergent-time::adaptive`'s change-point machinery)
  that triggers `StructuralTimePolicy`-based compaction specifically when a
  long gap is detected, versus running `CoherencePolicy` during normal
  steady operation. Not implemented here.
- **MCP**: no new MCP surface is warranted. This is an internal library
  policy choice with no external side effects or authority implications.

## Practical applications

1. **Long-running coding agents** — an agent paused overnight or across a
   CI queue shouldn't lose context on what it was mid-way through when it
   resumes.
2. **Personal AI assistants** — memory of a conversation from three days
   ago (when the assistant was last actively used) should compete fairly
   with memory from five minutes ago, not lose by default.
3. **Support/ops agents** — an incident investigated, paused for
   escalation, then resumed hours later should retain its own
   pre-escalation findings preferentially over unrelated tickets opened in
   between.
4. **Scheduled/batch agents** (nightly research runs like this one) —
   memory of the prior night's unfinished thread should not be
   systematically deprioritized purely because a day passed.
5. **Edge/robotics agents that sleep between missions** — mission-relevant
   memory from the last active mission should survive a long power-down,
   which this mechanism can approximate without needing a synchronized
   wall clock at all.
6. **Multi-tenant agent platforms** — tenants with bursty usage patterns
   (very different idle/active ratios) get consistent compaction behavior
   without per-tenant recency-window tuning.
7. **RAG systems over episodic logs** — recall of "the last thing that
   happened before a system was quiescent" is a common operational
   question (log/alert triage) that this mechanism directly targets.
8. **Autonomous research/evolution harnesses** (this very repository's
   nightly process) — a Flywheel-style evidence store that itself gets
   compacted over many nightly runs benefits from not conflating "a week
   with no nightly run" with "this evidence is now irrelevant."

## Long-horizon applications

1. **Thesis**: agent memory infrastructure eventually needs an internal
   clock decoupled from wall time, the same way relativity needed proper
   time decoupled from a universal background clock. **Required advances**:
   validated multi-channel structural metrics (entropy, graph, prediction
   error, not just embedding), tested at scale. **RuVector role**: the
   substrate where this first becomes a production `CompactionPolicy`
   rather than a diagnostic tool. **Why this run matters**: it's the first
   time `emergent-time`'s clocks touch a retention *decision* instead of
   only alarms/health classification. **Primary uncertainty**: whether the
   effect holds on real (not synthetic) agent traces. **Falsification**:
   measure on a real multi-day agent trace; if recall parity holds
   (no gap pathology observed), the mechanism has no practical value beyond
   this synthetic worst case.
2. **Swarm memory** — multiple agents with independently bursty activity
   sharing a memory store need a *relative*, not wall-clock, notion of
   "how stale is this" that's comparable across agents with different
   activity rhythms. Required advances: a cross-agent structural clock
   normalization. Uncertainty: whether per-agent structural budgets are
   comparable at all without a shared reference activity rate.
3. **Self-healing memory indexes** — index repair/rebuild triggers could
   use structural time instead of wall-clock TTLs to decide when an index
   partition is "stale enough" to warrant repair, avoiding needless
   repair cycles during genuinely idle periods.
4. **World models with sleep/wake cycles** — a world model that is
   "paused" (no observations) shouldn't decay its belief state on wall
   time; structural time is the natural clock for belief staleness.
5. **Proof-gated autonomous infrastructure** — provenance chains
   (`ruvector-retrieval-receipt`, `ledger`/`ops` in this crate) that record
   *when* something was true could record structural time alongside wall
   time, making "this was true as of N real actions ago" a well-defined,
   replayable statement independent of how long the system happened to be
   idle.
6. **Robotics memory** — a robot idle between missions (days, unpowered)
   needs mission memory that survives the gap; this mechanism generalizes
   directly since it was designed around exactly that idle-gap shape.
7. **Scientific autonomous systems** — long-running experiment-tracking
   agents (this nightly harness included) that pause between funding
   cycles, hardware availability windows, or human review need retention
   that doesn't equate "administrative delay" with "no longer relevant."
8. **Agent operating systems** — if agent memory becomes an OS-level
   service shared across many processes/agents with heterogeneous duty
   cycles, wall-clock recency is the wrong primitive at the OS layer for
   the same reason it's wrong here, just at larger scale.

## Falsification criteria

This hypothesis would be falsified by: (a) `StructuralTime` failing to beat
`CoherenceWeighted` by the pre-registered 3.0pp margin on the bursty-idle
workload (it passed, +31.8pp), or (b) regressing steady-workload recall by
more than 1.0pp (it passed, +0.0pp exactly), or (c) the effect not
replicating across independent seeds (it replicated on all 8 tested).
None of these occurred; the hypothesis survives this run. It would still be
falsified by future work showing the effect vanishes on real (non-synthetic)
agent traces, or on datasets where phase-1/phase-2 topics are not as cleanly
separable as in this construction.

## Production path

Not proposed for default-on promotion from this single run. Recommended
next step: validate on a corpus of real agent-memory traces containing
genuine idle gaps (this crate's `ledger`/`observation` modules already
capture timestamped, replayable history that could source such a corpus)
before considering `StructuralTimePolicy` as an opt-in alternative to
`CoherencePolicy` in any real deployment. The code ships behind no feature
flag (it's a new, independently-selectable `CompactionPolicy`, not a
default-path change), so it can be adopted incrementally without any
migration or rollback machinery beyond "use a different policy value."

## Next research

1. Validate against a real (not synthetic) multi-day agent-memory trace.
2. Wire a second structural channel honestly (e.g. `ΔG` from
   `ruvector-mincut`'s cluster-stability signal) instead of leaving four of
   five `StructuralMetric` channels at zero.
3. Characterize the effect as a function of gap size relative to real
   activity volume (this run only tested one gap magnitude, 250× the
   phase-1 tick budget).
4. Test multi-gap traces (more than one idle period) rather than the
   single-gap construction used here.
5. If a real-trace validation succeeds, prototype the ruFlo idle-gap-aware
   compaction trigger described above.

## References

- Page & Wootters, "Evolution without evolution" (1983); DeWitt (1967);
  Connes & Rovelli, thermal time (1994) — via `emergent-time`'s physics
  formalisms, `crates/emergent-time/README.md`, `docs/adr/ADR-251-agentic-time.md`.
- This crate's own prior nightly result: `CoherencePolicy`
  (`docs/adr/` — see `crates/ruvector-agent-memory/src/compaction.rs`
  doc comment for its lineage and paper references).
