# Nightly Research: Structural-Time-Gated Memory Compaction Scheduling

**Date:** 2026-09-18
**Slug:** `structural-time-gated-memory-compaction`
**ADR:** [ADR-346](../../../adr/ADR-346-structural-time-gated-memory-compaction.md)
**Crate:** `ruvector-agent-memory` (`structural_gate` module, `structural-gate` feature) + `emergent-time` (reused as-is)
**Acceptance:** **REJECT** (for production use as calibrated) — real measured evidence retained; see [Acceptance Result](#acceptance-result)

## Summary

`ruvector-agent-memory` has three `CompactionPolicy` implementations
(`LruPolicy`, `LfuPolicy`, `CoherencePolicy`) that answer *what* survives a
compaction pass. Nothing in the crate has ever answered a genuinely
orthogonal question: *when* should a compaction pass run at all? Every
existing benchmark in the crate — including the 2026-09-05 mincut-gated-
forgetting nightly — invokes `compact()` exactly once, on a store that
never receives another write. A long-running agent doesn't get that
luxury.

This experiment introduces a `CompactionTrigger` trait and three
implementations: `FixedIntervalTrigger` (a wall-tick baseline),
`CapacityTrigger` (a capacity-ceiling baseline), and
`StructuralGateTrigger`, which reuses `emergent-time`'s
`StructuralProperTime` clock — an existing, independently-developed and
independently-tested workspace crate's "arc length through the system's
own state manifold" formalism — unmodified, as the scheduling signal. The
idea: a quiet run of near-duplicate writes should barely move the clock and
the trigger should stay silent, saving compaction cost; a burst of
semantically novel writes should move it a lot and the trigger should fire
promptly.

**The hypothesis was falsified, with a well-characterized root cause.** The
structural signal *does* discriminate regime — 70 of its 100 fires landed
on the 200 writes (14.3% of the stream) belonging to genuine "burst"
epochs, a roughly 14x higher fire-density than during the 1,200 "quiet"
writes — but the pre-registered calibration procedure (threshold = 20x the
mean per-write tick on an independent quiet baseline) was not conservative
enough: quiet-regime measurement noise alone crosses it about once every
40 writes. Total compaction calls (100) came in 4.2x *higher* than the
naive fixed-interval baseline (24) — the opposite of the primary
hypothesis — and wall-clock overhead was 3.6–3.9x higher than the 2x
ceiling. Two of four mandatory acceptance criteria failed → REJECT.

A genuine, previously-unreachable correctness bug was found and fixed
along the way: `MemoryStore::insert` assigned `id = entries.len()`, which
silently collides with a surviving entry's id once compaction and further
insertion interleave. Every prior benchmark only ever compacted once,
terminally, so the bug was unreachable until this experiment's
continuous-write-plus-trigger loop exercised it. Fixed to a monotonic
`next_id` counter; the full existing crate test suite (all feature
combinations) passes unchanged.

## Abstract

We ask whether an existing, unrelated workspace crate's clock formalism —
`emergent-time::structural_clock::StructuralProperTime`, "time = the
metric-weighted arc length a system traces through its own state manifold"
— can be reused as-is to gate *when* agent-memory compaction runs, as an
alternative to naive wall-tick or capacity-ceiling triggers. We implement
the three triggers behind a common `CompactionTrigger` trait, define a
falsifiable Given/When/Then hypothesis and four numeric acceptance
thresholds before running anything, and evaluate on a deterministic,
seeded 1,400-write synthetic stream alternating quiet single-cluster epochs
with brief novel-cluster bursts. The result is a clean rejection on two of
four criteria, with a diagnostic breakdown showing the underlying signal
is real (a 14x fire-density skew toward genuine bursts) but the specific
calibration is not conservative enough to win on absolute operational cost
— a well-characterized negative result plus one production-relevant
correctness fix.

## Why This Matters

### In 2026

Agent-memory systems (this crate, and the broader RAG/agent-memory
ecosystem) universally treat compaction/eviction as a *what-to-keep*
problem and leave *when-to-run-it* to an ad hoc schedule (a cron tick, a
request counter, a capacity ceiling) with no relationship to what the
agent has actually been writing. As agents run longer and write more
continuously (multi-day sessions, background workers, swarms), a
scheduling policy that adapts to actual information churn — running rarely
during quiet stretches, promptly during bursts of genuinely new material —
is a real operational-cost lever, if it can be made to work.

### In 2036

If agent memory becomes the durable substrate for long-running autonomous
systems (per this repository's own ADR trajectory — ADR-330 arbitration,
ADR-345 mincut-gated forgetting, ADR-134 witness chains), *maintenance
scheduling itself* — not just retrieval and eviction — becomes a
first-class governed operation. An internal clock endogenous to the
agent's own state trajectory, rather than wall time, is the natural
trigger for any maintenance operation (compaction, re-embedding, index
repair) whose cost should track actual semantic churn, not calendar time.

### In 2046

A system that measures its own "how much did reality move" and schedules
its own maintenance from that measurement — rather than from an operator's
wall-clock guess — is a small, concrete instance of a broader thesis this
repository already pursues in `emergent-time`: internal, structurally-
derived time as a first-class primitive for autonomous, edge-deployed
cognitive systems that cannot assume a reliable external clock or a human
operator tuning cron schedules.

## RuVector Ecosystem Fit

Connects (per the nightly process's "≥3, prefer ≥5" ecosystem-leverage
rule):

1. **`ruvector-agent-memory`** — the compaction/eviction layer this
   experiment adds a scheduling axis to.
2. **`emergent-time`** — the `StructuralProperTime`/`Clock` formalism,
   reused unmodified as the scheduling signal.
3. **`ruvector-coherence`** (conceptually) — the coherence channel driving
   the structural clock's dominant (and, per the root-cause analysis,
   dominant-noise-source) signal.
4. **ADR-134 witness chain machinery** (`witnessed_compaction`,
   already in this crate) — a natural next integration point: witness
   *why* a compaction fired (which trigger, what accumulated signal),
   not just what it evicted (see [MCP Implications](#mcp-implications)).
5. **RVF / RVM** — see below; both are relevant to a rejected-but-retained
   scheduling policy shipped as a portable, governed component.

### MetaHarness / Flywheel / Darwin — verified availability

Per this run's Step 0 tool-discovery pass: `npx metaharness --help`
resolves to an installed package (v0.4.16) that scaffolds *new* harness
projects (`score`, `analyze`, `genome`, `learn`, `avo`, `proxy`
subcommands) — it is not an in-repo research-orchestration layer this
session could invoke against `ruvector` itself. `npx ruvector harness
doctor --json` does not resolve to any executable in this environment
("could not determine executable to run"). Per the nightly process's own
rule ("do not assume a package exists solely because it appears in the
prompt; verify first"), this run did not fabricate MetaHarness/Darwin/
Flywheel tool output. The research → hypothesis → implementation →
measurement → critique → promotion-or-rejection loop described by those
systems was followed manually within this session instead, with the
adversarial "attack pass" performed as direct root-cause analysis on the
measured benchmark output (see [Root Cause](#root-cause-analysis)).

## Architecture

```mermaid
flowchart TD
    subgraph WriteStream["Agent write stream (1,400 writes)"]
        Q1[Quiet epoch: 300 writes,\nsame topic cluster] --> B1[Burst epoch: 50 writes,\nnew topic cluster]
        B1 --> Q2[Quiet epoch] --> B2[Burst epoch] --> Q3[...] --> B4[Burst epoch 4]
    end

    WriteStream -->|each write| Trigger{CompactionTrigger}

    subgraph Triggers["Three CompactionTrigger implementations (compared)"]
        FI[FixedIntervalTrigger\nfires every 50 writes]
        CAP[CapacityTrigger\nfires above 2x target_size]
        SG[StructuralGateTrigger\naccumulates StructuralProperTime\nover a 24-write sliding window]
    end

    Trigger -.-> FI
    Trigger -.-> CAP
    Trigger -.-> SG

    SG -->|centroid, coherence,\nentropy of window| ET["emergent_time::structural_clock\nStructuralProperTime (reused as-is)"]
    ET -->|tick / cumulative| SG

    FI -->|fires| Compact[compact\\(CoherencePolicy, target_size\\)]
    CAP -->|fires| Compact
    SG -->|fires| Compact

    Compact --> Metrics["n_compactions, excess-size integral,\nRecall@10, wall-clock, fire-location"]
```

## Implementation

`crates/ruvector-agent-memory/src/structural_gate.rs` (feature
`structural-gate`, off by default):

```rust
pub trait CompactionTrigger {
    fn name(&self) -> &str;
    fn on_write(&mut self, entries: &[MemoryEntry]) -> bool;
    fn on_compacted(&mut self, entries: &[MemoryEntry]);
}
```

- **`FixedIntervalTrigger`** — `O(1)` per write; fires every `interval`
  writes since the last compaction.
- **`CapacityTrigger`** — `O(1)` per write; fires once `entries.len() >
  max_size`.
- **`StructuralGateTrigger`** — `O(window * dims)` per write, deliberately
  *not* `O(store.len() * dims)`: maintains a `VecDeque<Vec<f32>>` of only
  the last `window` writes, from which it derives a
  `emergent_time::structural_clock::StateSnapshot { embedding: window
  centroid, coherence: mean cosine-sim to centroid, entropy: Shannon
  entropy of an 8-bin similarity histogram, graph: 0.0, pred_error: 0.0 }`
  every write, and accumulates `StructuralProperTime::tick(prev, cur)`
  since the last compaction. Fires when the accumulator crosses a
  threshold. `graph`/`pred_error` are left at `0.0` rather than fabricated
  — this trigger has no topology or predictive-error signal available.
  This bounded-window design is a direct, explicit response to the prior
  nightly run's (2026-09-05, mincut-gated-forgetting) rejection reason: an
  `O(n)`-or-worse structural signal that was 1,800–2,700x slower than the
  scalar baseline even at trivial corpus sizes.
- **`StructuralGateTrigger::calibrate_threshold`** — computes the mean
  per-write tick over a supplied baseline write slice, times a fixed
  multiplier, mirroring the `baseline_window` calibration pattern
  `emergent_time::structural_clock::alarm_step` already uses. Decided once
  from data the trigger will actually see, before the comparison run — not
  tuned after inspecting the full benchmark result.

6 unit tests cover all three triggers, including a near-duplicate-stream
silence test, an orthogonal-cluster-burst firing test, and a trait-object
safety check; all pass (`cargo test -p ruvector-agent-memory --features
structural-gate,mincut-forget,proof-gate` — 74 tests total across the
crate, 0 failures).

### Correctness fix: `MemoryStore` id collisions under interleaved compaction

`crates/ruvector-agent-memory/src/memory.rs`: `MemoryStore::insert`
previously assigned `id = entries.len() as u64`. Every existing call site
only ever compacted once, terminally, so `entries.len()` and a monotonic
counter always agreed. This experiment's continuous write-then-maybe-
compact loop is the first usage pattern in the crate that inserts *after*
a compaction has already shrunk the store — at which point a new insert's
`id` (the new, smaller `entries.len()`) can collide with an id a surviving
older entry already holds. This was caught by the benchmark's own
Recall@10 metric silently reading `0.0000` for every trigger (ground-truth
ids computed from an independently-built, never-compacted reference store
no longer corresponded to the same memories once ids collided). Fixed to a
`next_id: u64` monotonic counter, independent of `entries.len()`. Pure
correctness fix, applies identically regardless of which trigger is used,
verified by the full pre-existing test suite passing unchanged.

## Benchmark Methodology

- **Hardware/OS/toolchain:** this session's Linux container; `cargo
  build --release` (optimized; debug/logging overhead excluded from timed
  sections).
- **Dataset:** deterministic, seed `0x5EED_C0DE` (RNG stream) /
  `0x5EED_C0DE` xor'd for centroid/query generation. 32-dim unit vectors.
  5 topic clusters total (1 present from the start, 4 introduced one per
  burst epoch). 1,400 writes: 4x(300 quiet + 50 burst).
- **Compaction policy:** `CoherencePolicy::default()` (α=0.25 recency,
  β=0.35 frequency, γ=0.40 coherence-with-context) at every fire, for every
  trigger — the *what* axis is held constant so only the *when* axis
  varies. `target_size = 200`.
- **Ground truth for Recall@10:** exact top-10 neighbors (by cosine
  similarity) for 30 held-out queries drawn from the final (most recent)
  cluster, computed against an independently-built, never-compacted
  reference store built from the identical write schedule. Compared to
  each trigger's own final compacted-store search results via id
  (post-fix, ids are stable across compaction).
- **Trigger configuration — fixed a priori, not tuned on results:**
  `FixedInterval(50)`, `Capacity(400 = 2x target_size)`,
  `StructuralGate` with `window=24`, threshold calibrated once from an
  independent 50-write quiet-baseline slice at multiplier 20x.
- **Repetitions:** 3 full release-mode runs. Algorithmic outputs (call
  counts, excess-size integral, recall) are bit-identical across runs
  (fully deterministic, seeded); wall-clock timings varied within normal
  measurement noise (see table).
- **Exact command:**
  ```bash
  cargo run --release -p ruvector-agent-memory --features structural-gate \
    --example structural_gated_compaction_bench
  ```

## Benchmark Results

Raw output (run 1 of 3; algorithmic columns identical across all 3 runs,
wall_ms varied 6.2–6.3 / 2.4–2.5 / 22.2–23.5 across runs respectively):

```
=== Structural-Time-Gated Compaction Scheduling ===
writes=1400 clusters=5 target_size=200 dims=32
calibrated structural_threshold = 0.125561

trigger           compactions excess_size_integral      recall@10      wall_ms final_size
FixedInterval              24                30551         1.0000        6.161        200
Capacity                    5               120615         1.0000        2.415        200
StructuralGate            100                18913         1.0000       22.227        200

=== Diagnostic: fire location (1200 Quiet writes, 200 Burst writes) ===
trigger             fires@quiet    fires@burst
FixedInterval                20              4
Capacity                      4              1
StructuralGate               30             70

=== Acceptance (vs. FixedInterval, thresholds fixed pre-run) ===
compaction-call reduction:    -316.7%  (need >= 20%)
excess-size-integral reduction:    38.1%  (need >= 20%)
recall@10 gap:                 0.0000  (need <= 0.02)
wall-clock ratio (struct/fixed):    3.61x (need <= 2.0x)

ACCEPTANCE RESULT: REJECT
```

| Trigger | Compactions | Excess-size integral | Recall@10 | Wall (ms, 3-run range) | Fires@quiet / Fires@burst |
|---|---:|---:|---:|---:|---:|
| FixedInterval(50) | 24 | 30,551 | 1.0000 | 6.2–6.3 | 20 / 4 |
| Capacity(400) | 5 | 120,615 | 1.0000 | 2.4–2.5 | 4 / 1 |
| StructuralGate | 100 | 18,913 | 1.0000 | 22.2–23.5 | 30 / 70 |

## Memory Math

`StructuralGateTrigger`'s state is `window * dims * 4 bytes` (f32 vectors
in the `VecDeque`) plus a `StateSnapshot` (`dims * 8 bytes` for the f64
embedding copy + 4 f64 scalars). At `window=24, dims=32`: 24*32*4 = 3,072
bytes of window storage + 32*8 + 32 = 288 bytes snapshot ≈ 3.4KB total,
independent of `store.len()`. At a production embedding width (e.g. 1536)
this scales to ≈150KB — still bounded and independent of store size, the
property the prior nightly rejection (mincut-gated-forgetting) lacked.

## Performance Math

Per-write cost: `FixedIntervalTrigger`/`CapacityTrigger` are `O(1)`.
`StructuralGateTrigger` is `O(window * dims)` for the centroid recompute +
`O(window)` for the coherence/entropy pass ≈ `O(window * dims)` dominant
term. Measured 3.6–3.9x wall-clock ratio at `window=24, dims=32,
1400 writes` reflects both this per-write constant-factor cost *and* the
100-vs-24 compaction-call-count difference (each `compact()` call itself
costs `O(store.len() * context_window.len())` for `CoherencePolicy`
scoring) — the two effects are not separated in this run's timing (see
Limitations).

## Root Cause Analysis

`StructuralProperTime`'s coherence channel accumulates only on *loss*:
`(prev.coherence - cur.coherence).max(0.0)`. This is correct and
intentional for `emergent-time`'s original irreversible-drift semantics
(a system's coherence generally shouldn't regenerate on its own). A
24-write sliding-window coherence *estimate*, however, fluctuates from
sampling noise alone even under a perfectly stationary source (repeatedly
sampling near-identical vectors around one centroid) — half of those
fluctuations are downward, and loss-only accumulation means every downward
fluctuation adds internal time while no upward fluctuation ever cancels
it. Quiet-regime measurement noise is therefore read as monotone
structural drift by construction, not as a modeling error specific to this
trigger's summary statistics. A threshold calibrated purely from that same
quiet baseline's *mean* tick (20x mean) is still well within reach of the
baseline's own *variance*: the diagnostic shows the quiet regime alone
(1,200 writes) crossed it 30 times — about once every 40 writes, under
`FixedInterval`'s 50-write cadence, which is why total calls came in
higher rather than lower.

The diagnostic also shows the signal is **not** noise-only: 70 of 100
total fires landed on the 200 burst writes (14.3% of the stream), a ~14x
higher fire-density than the quiet regime's 30-in-1,200. The underlying
"does structure moving more mean more fires" relationship holds; the
tested calibration procedure simply wasn't conservative enough to also win
on *absolute* call count against a 50-write fixed baseline.

## Failure Modes

1. Loss-only coherence accumulation reads quiet-regime sampling noise as
   drift (primary root cause, above).
2. `O(window * dims)` per-write cost, uncompensated by fewer total calls
   in this run, made wall-clock a losing criterion too.
3. Not tested: behavior under deletes, concurrent writers, or an
   adversarially crafted write stream designed to starve or flood the
   trigger — out of scope for a scheduling-cadence experiment, but a real
   gap before any production consideration.

## Rejected Alternatives

- **Symmetric (not loss-only) coherence accumulation** — would likely fix
  the root cause directly, but forking `StructuralProperTime`'s metric
  semantics contradicts this experiment's explicit goal of reusing the
  shared clock *as-is*; left for a follow-up that explicitly forks rather
  than reuses.
- **EMA-smoothed window centroid** instead of a hard sliding window —
  would reduce jitter but adds a decay-rate hyperparameter and moves
  further from a direct `emergent-time` reuse; not tested.
- **Mincut-based structural signal** (2026-09-05 nightly) — already tried
  and rejected on performance grounds (1,800–2,700x slower than scalar
  baseline at 50–400 vertices); this experiment deliberately used a
  bounded-window design specifically to avoid repeating that failure mode,
  and succeeded on that axis (wall-clock ratio here is 3.6–3.9x, not
  1,800–2,700x) even though it failed on calibration.

## Security

No new attack surface. The trigger reads only `MemoryEntry` vectors
already resident in the store and compares a scalar accumulator to a
scalar threshold; no untrusted parsing, no new serialization format, no
witness or signature involvement. `structural-gate` is an additive,
opt-in Cargo feature (off by default) with no default-path exposure.

## Governance

Rejected with retained evidence per the nightly process's own rule: a
falsified hypothesis with good evidence is a successful run. No autonomous
promotion occurred. Re-testing with a materially different calibration
procedure (see Next Research) is a new, independently evaluated
hypothesis against the same acceptance thresholds — not a retroactive
adjustment of this one.

## MCP Implications

Not built this run (no MCP surface change). If revisited and eventually
promoted, the natural MCP surface would be **read-only**: a tool exposing
"structural time accumulated since last compaction" and "calibrated
threshold" as diagnostic state for an operator or a higher-level ruFlo
workflow to inspect — never a tool that lets a remote caller *set* the
threshold or force a compaction, which would reintroduce exactly the kind
of externally-triggered cost this experiment was trying to avoid.

## WASM / Edge Implications

`StructuralGateTrigger`'s `O(window * dims)` bounded-memory design (≈3.4KB
state at window=24/dims=32, independent of store size — see Memory Math)
is edge-friendly in principle: an edge agent with a small, fixed embedding
width could run this trigger with a small, fixed memory footprint
regardless of how large its memory store grows. Not measured for actual
WASM binary size or edge deployment in this run — no deployment claim is
made without evidence, per the nightly rules.

## RVF Implications

A compaction-trigger *policy* (its calibrated threshold, window size, and
choice of trigger type) is a small, portable piece of configuration state
— a natural fit for an RVF-portable "how this agent's memory maintenance
behaves" artifact, alongside the memory contents themselves. Not
implemented this run (the candidate was rejected before reaching a
promotion-worthy state); noted as the RVF-relevant shape a *future*,
recalibrated version of this trigger would take.

## RuFlo Implications

If a recalibrated trigger is eventually promoted, the natural ruFlo role
is **continuous benchmark-informed threshold tuning**: a workflow that
periodically re-runs this same benchmark harness (or a production
analogue) against live traffic statistics and adjusts the calibration
multiplier — exactly the kind of "continuous benchmark optimization"
ruFlo role this repository's other ADRs describe, applied here to a
concrete, currently-rejected starting point rather than a hypothetical
one.

## Practical Applications

1. **User:** operator of a long-running coding agent. **Problem:** fixed
   compaction cadence either wastes CPU during idle stretches or lets
   memory balloon during bursts of activity. **RuVector capability:**
   trigger machinery (once recalibrated). **Ecosystem integration:**
   `ruvector-agent-memory` + `emergent-time`. **Path:** recalibrate,
   re-benchmark, wire behind existing `compact()` call sites.
   **Value:** lower steady-state CPU cost. **Risk:** miscalibration (this
   run's own failure mode). **Horizon:** near-term, pending recalibration.
2. **User:** multi-tenant agent-memory service operator. **Problem:**
   uniform compaction schedules waste cost on quiet tenants.
   **Capability:** per-tenant structural thresholds. **Integration:**
   `ruvector-agent-memory`. **Path:** per-tenant `StructuralGateTrigger`
   instances. **Value:** cost proportional to actual tenant activity.
   **Risk:** cross-tenant calibration drift. **Horizon:** near-term.
3. **User:** edge/robotics agent with constrained compute. **Problem:**
   cannot afford frequent compaction scans. **Capability:** bounded-memory
   trigger (Memory Math). **Integration:** `emergent-time` + edge
   deployment. **Path:** validate window/dims scaling on-device. **Value:**
   predictable, bounded overhead. **Risk:** unvalidated WASM footprint.
   **Horizon:** mid-term.
4. **User:** Graph-RAG system maintainer. **Problem:** deciding when to
   re-run expensive graph maintenance (not just vector compaction).
   **Capability:** the same `Clock`/`StateSnapshot` pattern, with the
   `graph` channel populated (left at 0.0 here). **Integration:**
   `ruvector-mincut` + `emergent-time`. **Path:** extend `StateSnapshot`
   population to include a real graph-topology scalar. **Value:**
   maintenance scheduling proportional to graph churn, not wall time.
   **Risk:** repeats this run's calibration failure mode if not addressed
   first. **Horizon:** mid-term, blocked on root-cause fix.
5. **User:** security/anomaly-detection pipeline maintainer. **Problem:**
   deciding when to re-baseline a drift detector. **Capability:**
   `StructuralProperTime`'s existing early-warning use case in
   `emergent-time` itself (not this experiment, but the same underlying
   clock). **Integration:** `emergent-time` directly. **Path:** already
   demonstrated in that crate's own tests. **Value:** earlier warning than
   entropy-only clocks (`emergent-time`'s own measured result). **Risk:**
   none new from this experiment. **Horizon:** already available.
6. **User:** enterprise retrieval system with bursty ingestion (e.g.
   incident response, where a flood of new documents follows a quiet
   period). **Problem:** static reindex schedules over- or under-serve
   bursts. **Capability:** structural trigger, recalibrated. **Integration:**
   `ruvector-agent-memory` + ingestion pipeline. **Path:** same as (1),
   applied to reindexing rather than compaction. **Value:** faster
   response to genuine bursts. **Risk:** this run's exact failure mode.
   **Horizon:** mid-term.
7. **User:** scientific-search agent tracking a fast-moving literature
   area. **Problem:** knowing when accumulated new results justify
   re-clustering. **Capability:** structural trigger over document
   embeddings. **Integration:** `ruvector-agent-memory`. **Path:**
   analogous substitution of domain vectors for agent-memory vectors.
   **Value:** re-clustering timed to genuine topic shifts. **Risk:**
   same calibration risk. **Horizon:** mid-term.
8. **User:** local-first personal-assistant developer. **Problem:**
   minimizing background CPU/battery use on a laptop or phone while
   keeping memory fresh. **Capability:** bounded-cost, content-aware
   trigger. **Integration:** `ruvector-agent-memory` on-device.
   **Path:** same recalibration prerequisite as (1). **Value:** lower
   background resource use during idle periods. **Risk:** same
   calibration risk plus untested WASM footprint. **Horizon:** mid-term.

## Long Horizon Applications

1. **Self-healing graph memory** — a `graph`-channel-populated structural
   clock scheduling not just compaction but graph-repair passes.
   **Required advances:** fix the calibration/asymmetry root cause;
   populate the graph channel meaningfully. **RuVector role:**
   `ruvector-mincut` + `emergent-time` + `ruvector-agent-memory`. **Why
   this experiment matters:** first concrete attempt at reusing
   `emergent-time` as a scheduling signal for any maintenance operation.
   **Primary uncertainty:** whether the coherence-loss asymmetry
   generalizes as a problem to a graph-topology channel too. **Falsification
   path:** repeat this exact benchmark shape with a real graph-topology
   channel and see if the same over-firing pattern recurs.
2. **Agent operating systems** — internal clocks as the native scheduling
   primitive for *all* agent maintenance (not just memory), replacing
   wall-clock cron entirely. **Required advances:** a calibration
   methodology that doesn't fail as this one did. **RuVector role:**
   `emergent-time` as a shared OS-level clock service. **Why this
   experiment matters:** demonstrates both the appeal (14x fire-density
   discrimination) and the pitfall (noise-driven over-triggering) of the
   approach on real code. **Primary uncertainty:** generalization beyond
   memory compaction to arbitrary maintenance tasks. **Falsification
   path:** a second, independent maintenance-task experiment using the
   same clock.
3. **Swarm memory** — per-agent structural clocks whose relative rates
   inform swarm-level resource allocation (agents with more internal
   churn get more maintenance budget). **Required advances:** multi-agent
   calibration and fairness analysis. **RuVector role:**
   `ruvector-agent-memory` + swarm coordination layers. **Why this
   experiment matters:** single-agent baseline for the calibration
   problem swarm-scale would inherit and amplify. **Primary uncertainty:**
   whether per-agent noise characteristics differ enough to need per-agent
   calibration. **Falsification path:** multi-stream variant of this
   benchmark with heterogeneous noise levels.
4. **Dynamic world models** — an agent's internal sense of "how much has
   changed" as the trigger for world-model updates, not fixed frame rates.
   **Required advances:** extending the metric beyond embedding/coherence
   to whatever a world-model's own state representation is. **RuVector
   role:** `emergent-time` as the shared formalism. **Why this experiment
   matters:** concrete evidence of both the promise and the noise-
   sensitivity trap in a discrete, measurable setting. **Primary
   uncertainty:** whether continuous world-model states have the same
   loss-only-asymmetry noise problem. **Falsification path:** repeat on a
   continuous-state benchmark.
5. **Proof-gated autonomous infrastructure** — pairing a (fixed)
   structural trigger with `ruvector-proof-gate`/ADR-134 witness chains so
   *why* a maintenance operation fired is itself an auditable, signed
   fact. **Required advances:** the calibration fix, plus wiring into
   `witnessed_compaction`. **RuVector role:** `ruvector-agent-memory`'s
   existing witness machinery, unmodified. **Why this experiment matters:**
   establishes the un-witnessed baseline behavior first. **Primary
   uncertainty:** none new — this is an integration question once the
   trigger itself works. **Falsification path:** N/A, integration-only.
6. **RVM coherence domains** — isolated execution domains whose
   maintenance cadence is gated by their own internal coherence, not a
   shared external clock. **Required advances:** per-domain
   `StructuralProperTime` instances with domain-appropriate metrics.
   **RuVector role:** `rvm` + `emergent-time`. **Why this experiment
   matters:** memory compaction is the simplest instance of "domain
   maintenance"; this establishes feasibility and pitfalls. **Primary
   uncertainty:** whether domain isolation changes the noise
   characteristics. **Falsification path:** RVM-hosted variant of this
   benchmark.
7. **Robotics memory** — a robot's episodic memory compaction gated by
   genuine environmental/task-state change rather than fixed sensor-tick
   cadence. **Required advances:** populate the `graph`/`pred_error`
   channels with real sensor-fusion signals (this run left them at 0.0
   rather than fabricate them). **RuVector role:** `ruvector-agent-memory`
   + `agentic-robotics-*` crates. **Why this experiment matters:**
   establishes the pure-vector baseline before adding sensor channels.
   **Primary uncertainty:** whether sensor-derived channels have the same
   noise-asymmetry problem as the vector/coherence channels tested here.
   **Falsification path:** robotics-specific benchmark with real sensor
   traces.
8. **Autonomous edge cognition** — bounded-memory, bounded-CPU maintenance
   scheduling as a hard requirement (not just an optimization) on
   constrained hardware. **Required advances:** the calibration fix, plus
   actual WASM/edge measurement (not claimed here). **RuVector role:**
   `emergent-time` + edge/WASM crates. **Why this experiment matters:**
   demonstrates the memory-boundedness property (Memory Math) an edge
   deployment would require, independent of the calibration failure.
   **Primary uncertainty:** real hardware validation, entirely untested.
   **Falsification path:** on-device benchmark once calibration is fixed.

## Competitor Comparison

Not materially applicable to this specific experiment: no external vector
database (Milvus, Qdrant, Weaviate, Pinecone, LanceDB, FAISS, pgvector,
Chroma, Vespa) publishes a compaction-*scheduling* primitive comparable to
what's tested here (`documented_external_capability`: none found;
`directly_measured_capability`: N/A; `RuVector_architectural_difference`:
`ruvector-agent-memory`'s scalar `CoherencePolicy` for *what* to evict is
itself already a differentiator documented in the 2026-06-14 nightly, and
this experiment's addition is a *when*-axis, not a *what*-axis, so no
existing competitor comparison changes as a result of this run's REJECT).

## Evolution Results

No Darwin run: `npx ruvector harness darwin --help` was not reachable in
this environment (see [Ecosystem Fit](#ruvector-ecosystem-fit)), and the
single hand-picked calibration (20x mean quiet-baseline tick) already
failed two of four mandatory criteria clearly enough that a bounded
parameter search was not run this session, to avoid tuning the same
pre-registered hypothesis's threshold after seeing its own result (see
[Governance](#governance)). A follow-up run with an explicitly new,
independently-registered calibration hypothesis is the correct next step
(see Next Research), not a same-session retry.

## Promotion Decision

**Not promoted.** `beats_parent` is false (baseline `FixedIntervalTrigger`
outperforms `StructuralGateTrigger` on the primary call-count metric and
the wall-clock guard); per the nightly process's hard rule, a failed
mandatory gate means no promotion, and the rejected candidate (full
`structural_gate` module, tests, and this report) is retained in the
lineage rather than discarded, so a future run does not rediscover the
same loss-only-asymmetry pitfall blindly. The `MemoryStore::insert` id fix
is retained unconditionally — it is a correctness fix independent of the
rejected candidate.

## Witness Evidence

No signed witness chain was generated for this specific benchmark run (the
trigger has no witness integration — see MCP Implications). Evidence for
this report's claims is the benchmark's own deterministic, seeded,
re-runnable output (3 repeated runs, algorithmic columns bit-identical)
plus the crate's pre-existing (unrelated to this experiment) ADR-134
witness machinery, unaffected by this change and verified still passing
via the full test suite.

## Production Path

None at this time — rejected. A production path opens only after a
materially different, independently-registered calibration hypothesis
clears all four acceptance criteria on a repeat of this exact benchmark
shape (same hypothesis format, same thresholds, different calibration
procedure, per the nightly process's "don't move the goalposts" rule).

## Falsification Criteria

Stated before running (ADR-346 Hypothesis): compaction-call reduction ≥
20%, excess-size-integral reduction ≥ 20%, recall gap ≤ 2pp, wall-clock
ratio ≤ 2x, all four required. Two failed (call-count reduction:
−316.7%; wall-clock ratio: 3.6–3.9x) → falsified as pre-registered.

## Limitations

- Single synthetic dataset shape (4 quiet/burst epoch pairs, fixed sizes);
  not validated against a real agent-memory write trace.
- Wall-clock comparison conflates per-write trigger cost and
  per-compaction `compact()` cost (100 vs 24 calls); not decomposed in
  this run.
- No deletes, no concurrent writers, no adversarial write streams tested.
- `graph` and `pred_error` channels left at 0.0 (no fabricated signal);
  the structural clock's full five-channel design is not exercised.
- Single calibration multiplier (20x) tested; no sweep.

## Next Research

1. Repeat this exact benchmark (same hypothesis shape, same four
   thresholds) with a calibration procedure informed by both a quiet *and*
   a representative burst reference slice, not quiet alone — the most
   direct fix for the identified root cause.
2. Isolate whether the coherence channel's loss-only asymmetry or the
   entropy channel's histogram noise is the larger contributor, by running
   `StructuralMetric` variants with one channel zeroed at a time (a
   diagnostic ablation, not a re-tuning of the rejected hypothesis).
3. Decompose wall-clock cost into per-write trigger overhead vs.
   per-compaction `compact()` cost, so a future calibration fix's
   wall-clock impact can be attributed correctly.
4. If (1) changes the outcome, populate the `graph` channel with a real
   `ruvector-mincut` or `ruvector-namespace-merge` topology scalar and
   re-test the graph-aware variant explicitly flagged as future work
   above.

## References

- `crates/emergent-time/src/structural_clock.rs` (this repository,
  in-tree, existing and independently tested).
- `crates/ruvector-agent-memory/src/compaction.rs`,
  `src/graph_forget.rs` (this repository, in-tree).
- [ADR-345](../../../adr/ADR-345-mincut-gated-forgetting.md) and its
  [nightly report](../2026-09-05-mincut-gated-forgetting/README.md) —
  the prior structural-signal-for-compaction attempt this run's
  bounded-window design was built to avoid repeating the performance
  failure of.
- [ADR-134] (this repository's witness-chain schema, referenced but not
  modified this run).
