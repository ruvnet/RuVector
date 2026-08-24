# Structural-Time Memory Decay: Does `emergent-time`'s Structural Clock Beat Wall-Clock Recency for Agent Memory Compaction?

**150-char summary:** Swapped wall-clock recency for `emergent-time`'s embedding-arc-length clock in memory compaction — directionally correct, never worse, but a 10-seed average misses the pre-registered bar.

**Date:** 2026-08-24
**Crate:** `crates/ruvector-structural-memory`
**ADR:** [ADR-340](../../../adr/ADR-340-structural-time-memory-decay.md)
**Acceptance result: REJECT** (of the pre-registered ACCEPT threshold — see [Acceptance Result](#acceptance-result))

---

## Abstract

Agent memory compaction (`ruvector-agent-memory`, nightly 2026-06-14) scores a
stored memory's "recency" against wall-clock time: the count of turns/steps
since it was written. This nightly isolates that one variable and asks
whether `emergent-time`'s (ADR-251) `StructuralProperTime` — internal time
defined as accumulated *embedding-arc-length* rather than step count — is a
better clock for that recency term.

The mechanism under test: during a long, low-drift stretch of a session (an
agent heads-down on one topic), a structural clock accumulates almost no
internal time, so memories written early and late in that stretch end up at
nearly the same structural age even though many wall-clock steps separate
them. A wall clock cannot make that distinction. Three clocks — all literal
`emergent-time` types, no new clock math — were compared on a synthetic
agent-session benchmark (topic plateaus separated by sharp switches),
compacting to a fixed 25-memory budget and measuring recall@15 against an
oracle nearest-neighbour set, averaged over 10 deterministically-generated
seeds:

| Clock | Mechanism | Mean recall@15, plateau_len=150 (long) | Mean recall@15, plateau_len=20 (short) |
|---|---|---|---|
| `WallClock` (baseline) | age = step count | 0.1600 ± 0.0442 | 1.0000 ± 0.0000 |
| `StructuralEmbedding` (candidate) | age = accumulated `Δv` (`StructuralProperTime`, embedding channel only) | **0.1800 ± 0.0600** | 1.0000 ± 0.0000 |
| `StructuralFull` (exploratory) | age = `Δv` + real entropy signal | 0.1800 ± 0.0600 | 1.0000 ± 0.0000 |

**Key measured result:** the structural clock's mean long-plateau lead is
**+2.00 percentage points** (0.1800 vs 0.1600), below the **+5pp** threshold
fixed before this benchmark's final multi-seed form ran. Per-seed detail (10
seeds) shows the structural clock **never underperforms** WallClock — it
ties in 7/10 seeds and wins by exactly +6.67pp in 3/10 — but the win doesn't
happen reliably enough, or by enough margin per seed, for the pre-registered
mean-lead clause to pass. **Acceptance result: REJECT.** Compute overhead
(1.30x WallClock) and the no-regression clause at the short-plateau
configuration both passed. All numbers are from `cargo run --release -p
ruvector-structural-memory --bin benchmark` on the hardware below; raw
output is reproduced verbatim in [Benchmark Results](#benchmark-results).

**Hardware:** x86-64, 4 logical CPUs, Linux 6.18.44, `rustc 1.94.1`, release
build.

**A note on how this REJECT was reached, because it matters for trusting
it:** the very first run of this benchmark, on a single seed (`0xC0FFEE`),
showed a +6.67pp lead — comfortably over the 5pp bar. Three more seeds tried
while debugging an unrelated parameter (the context-noise scale; see
[Failure Modes](#failure-modes-and-things-that-almost-made-this-look-better-than-it-is))
showed two exact ties and one more +6.67pp win. Reporting the first seed
alone would have been exactly the "cherry picked seeds" pattern this
harness's rules explicitly forbid. The benchmark was rewritten to average
over 10 deterministically-generated seeds (`0xC0FFEE + i * 0x9E3779B9` for
`i in 0..10`, fixed before the final run, not chosen after seeing outcomes)
and gated on the mean. That honest aggregate is what REJECTs. This is the
nightly harness's "failed hypothesis with good evidence" case, not an
absence of a result.

---

## Hypothesis

```text
Given synthetic agent sessions of 4 topic plateaus separated by sharp
switches (plateau lengths 20, 60, or 150 steps; one memory written per
step, embedding dim 32),

when compaction retention score uses StructuralEmbeddingClock's
accumulated context drift (emergent_time::StructuralProperTime, embedding
channel only) instead of WallClock step count as the age signal,

then mean recall@15 of the oracle nearest-neighbour set — averaged over 10
independent seeds — after compacting to a fixed budget of 25 memories
improves by >= 5 percentage points in the long-plateau (150 steps/topic)
configuration, without regressing by more than 2 percentage points in the
short-plateau (20 steps/topic) configuration,

subject to compaction compute time staying within 5x WallClock's, and
causal order (monotone cumulative time) being preserved by every clock on
every seed.
```

**Result: REJECT.** Clause (a) — the long-plateau lead — measured 2.00pp
against a 5pp bar. Clauses (b), (c), (d) all passed. See
[Acceptance Result](#acceptance-result).

**What this does NOT claim:** that structural time is *worse* than wall
time — it never underperformed across 30 seed×plateau_len cells measured
(10 seeds × 3 plateau lengths). It claims only that the effect, as measured
here, is not reliably large enough to clear the bar set before benchmarking
began. See [Why the Effect Is Real But Small](#why-the-effect-is-real-but-small)
for the mechanism.

---

## Why This Matters for RuVector

RuVector positions itself as more than a vector database — a substrate for
agent memory, temporal reasoning, and long-running autonomous systems.
`emergent-time` (ADR-251) is a mature, 5,000-line, dependency-free crate
implementing several formalisms of internal/relational time (Wheeler-DeWitt,
Page-Wootters, entropic, thermal, structural proper time), but as of this
nightly it had never been wired into a concrete RuVector *use case* — its
existing benchmarks (early-warning lead, history compression) are generic
trajectory-monitoring demonstrations, not applied to a production surface.
This nightly is the first attempt to connect it to one.

Connects five RuVector ecosystem capabilities:

1. **Agent memory** (`ruvector-agent-memory`, 2026-06-14) — the production
   surface this experiment's clock-swap targets; not modified by this
   nightly, but directly comparable in methodology (same recall@k-after-
   compaction measurement).
2. **`emergent-time`** (ADR-251) — reused, not reimplemented: `Clock`,
   `StructuralProperTime`, `StructuralMetric`, `StateSnapshot`, `WallClock`,
   and `entropy::entropy_from_spectrum` are all called directly from
   `crates/emergent-time` via a path dependency.
3. **Vector search** — cosine similarity over synthetic memory embeddings is
   the retrieval mechanism the oracle and compaction both use.
4. **Witness/provenance** — `emergent_time::witness` already ships a
   hash-linked ledger for training-run provenance; a compaction event log
   sealed the same way is a natural extension (see
   [MCP Implications](#mcp-implications)), not implemented here.
5. **RVF** — a session's `Session { contexts, topics, memories, snapshots }`
   is a direct candidate for a portable, replayable RVF artifact (see
   [RVF Implications](#rvf-implications)).

---

## 2026 State of the Art

Agent-memory decay literature (Park et al. 2023 Generative Agents,
MemoryBank's Ebbinghaus curve, Mem0 2025, Xu 2026's five-stage lifecycle,
Karhade 2026's velocity/volatility decay — all surveyed in the 2026-06-14
nightly) uniformly measures "age" in wall-clock time or turn count. None of
the systems inspected for that prior nightly, nor any found for this one,
define memory age as a function of *how much the agent's own state has
changed* rather than how many turns have elapsed. `emergent-time`'s
structural-proper-time formalism (arc length of a system's worldline through
its own state manifold) is the direct mathematical tool for that
reparametrization, but its existing use in this repository was limited to
anomaly early-warning and trajectory compression (`structural_clock.rs`'s
own benchmark suite), not retention scoring. This nightly is a new
composition of an existing formalism against a use case it had not been
tried on — the novelty is the application, not the clock math itself, which
this crate deliberately does not modify.

---

## Architecture

```mermaid
flowchart LR
    subgraph Session["Synthetic session (ruvector-structural-memory::scenario)"]
        T[N topic centroids\nrandom unit vectors] --> C["Context trajectory\n(piecewise-constant + ramped switches)"]
        C --> M["One MemoryItem per step\n(context + independent noise)"]
        C --> E["StateSnapshot per step\nembedding = context\nentropy = H(softmax(cos-sim to topics))"]
    end

    subgraph Clocks["emergent-time (reused, not reimplemented)"]
        WC[WallClock\nage = step count]
        SE["StructuralProperTime\nw_embedding=1, rest=0\n(StructuralEmbedding)"]
        SF["StructuralProperTime\nw_embedding=1, w_entropy=1\n(StructuralFull)"]
    end

    E --> WC
    E --> SE
    E --> SF

    subgraph Compaction["ruvector-structural-memory::compaction"]
        SC["score(m) = w_coh*cos(m, final_context)\n+ w_recency*exp(-age_clock(m)/tau)"]
        K["keep top-25 by score"]
    end

    WC --> SC
    SE --> SC
    SF --> SC
    M --> SC
    SC --> K

    O["Oracle top-15\n(true cosine similarity\nto final context)"] --> R[recall@15]
    K --> R

    style Session fill:#1f6feb22,stroke:#1f6feb
    style Clocks fill:#8957e522,stroke:#8957e5
    style Compaction fill:#2ea04422,stroke:#2ea044
```

---

## Implementation

`crates/ruvector-structural-memory`:

- `src/clocks.rs` — fixes two `StructuralMetric` weight configurations on
  top of `emergent_time::StructuralProperTime`. No new clock math.
- `src/scenario.rs` — deterministic session generator (xorshift64* PRNG,
  same style as `emergent_time::structural_clock`'s own test generator): `N`
  topic centroids (random unit vectors), visited in sequence, each held for
  `plateau_len` steps with a `switch_width`-step linear ramp between them.
  One `MemoryItem` is written per step, embedded near that step's context
  plus independent noise. `build_snapshots` derives each step's
  `StateSnapshot`: `embedding` is the raw context; `entropy` is the Shannon
  entropy (via `emergent_time::entropy::entropy_from_spectrum`) of the
  softmax over cosine similarities from the context to every topic centroid
  — a genuine derived signal (peaks at a switch, near-zero mid-plateau; see
  `entropy_spikes_at_switch_and_settles_mid_plateau` test), not a fabricated
  curve. `coherence`/`graph`/`pred_error` are fixed at `0.0`: every clock
  instantiated here weights those channels at zero, so their value is inert
  — this crate has no honest signal source for them.
- `src/compaction.rs` — scores every memory once against the final context
  under a given clock's notion of age (`w_coherence * cos_sim + w_recency *
  exp(-age/tau)`, `tau` = a fixed fraction of that clock's *own* total
  elapsed time — see [Benchmark Hygiene](#benchmark-hygiene-and-methodology-notes)),
  keeps the top-`budget`, and separately computes the true top-k oracle set
  and recall.
- `src/main.rs` (`benchmark` binary) — the full sweep: 3 plateau lengths × 3
  clocks × 10 seeds, with 25 timing repetitions per cell, printing the
  aggregate table, acceptance clauses, per-seed detail for the deciding
  cell, and the exploratory `StructuralFull` comparison.
- `tests/` (inline `#[cfg(test)]` modules) — 7 unit/integration tests:
  topic-vector unit-length, one-memory-per-step invariant, the entropy
  discriminating property, budget-respecting compaction, oracle
  self-recall = 1.0, and an end-to-end structural-clock run.

No dependency beyond `emergent-time` (path dependency) and the Rust
standard library.

---

## Benchmark Hygiene and Methodology Notes

- **Release build**, `cargo run --release`.
- **25 timing repetitions** per (plateau_len, clock, seed) cell for the
  compute-overhead measurement; recall is deterministic given a seed (no
  repetition needed for it), but the *seed itself* is repeated 10x — see
  below.
- **10 seeds, generated deterministically before the final run**
  (`0xC0FFEE + i * 0x9E3779B9`), not chosen after looking at outcomes. This
  replaced an earlier ad hoc 1-seed, then 4-seed, exploration once it became
  clear the effect was seed-sensitive — see
  [Failure Modes](#failure-modes-and-things-that-almost-made-this-look-better-than-it-is).
- **Fixed absolute compaction budget (25)**, not a fraction of corpus size:
  a fractional budget (tried first, at 30%) made the experiment trivial —
  the budget comfortably contained the entire current-topic pool regardless
  of clock, so recall@15 was 1.0000 for every cell and the hypothesis was
  untestable. A budget fixed below a long plateau's own topic-pool size
  forces genuine within-topic competition, which is where the clock choice
  can matter.
- **Context noise fixed at 0.001** (small relative to a topic-switch jump,
  ≈√2 between two near-orthogonal unit centroids at dim=32): the first
  implementation used the same noise scale (0.05) later used for memory
  embeddings, which made per-step context movement during a "quiet" plateau
  comparable in magnitude to a topic switch — defeating the entire premise
  (a structural clock is only informative if quiet periods are actually
  quiet). See [Failure Modes](#failure-modes-and-things-that-almost-made-this-look-better-than-it-is).
- **Compute-overhead ratio** computed from mean wall-clock time (`Instant`)
  summed across all three plateau_len configurations for `WallClock` vs
  `StructuralEmbedding`.
- **Causal order** (`emergent_time::Clock::cumulative` is monotone
  non-decreasing) checked directly on every clock/config/seed combination,
  not merely assumed from the trait's non-negative-tick guarantee.

---

## Benchmark Results

Raw output from `cargo run --release -p ruvector-structural-memory --bin
benchmark`:

```text
ruvector-structural-memory benchmark
config: n_topics=4 dim=32 oracle_k=15 budget=25 timing_reps=25 n_seeds=10
hardware: x86_64-linux, rustc build=release
seeds: [12648430, 2667084199, 5321519968, 7975955737, 10630391506, 13284827275, 15939263044, 18593698813, 21248134582, 23902570351]

plateau_len  clock                total_steps  budget recall@15(mean±sd)   mean_time_ns    causal_ok
20           WallClock                     80      25    1.0000±0.0000           5975         true
20           StructuralEmbedding           80      25    1.0000±0.0000           7823         true
20           StructuralFull                80      25    1.0000±0.0000           7609         true
60           WallClock                    240      25    0.3867±0.1147          15180         true
60           StructuralEmbedding          240      25    0.4000±0.1075          19102         true
60           StructuralFull               240      25    0.4133±0.1024          19141         true
150          WallClock                    600      25    0.1600±0.0442          39362         true
150          StructuralEmbedding          600      25    0.1800±0.0600          51889         true
150          StructuralFull               600      25    0.1800±0.0600          51484         true

acceptance clauses (thresholds fixed before this run; means over 10 seeds):
  (a) mean long-plateau lead >= 5pp: measured 2.00pp -> FAIL
  (b) mean short-plateau regression <= 2pp: measured 0.00pp delta -> PASS
  (c) compute overhead ratio <= 5x: measured 1.30x -> PASS
  (d) causal order preserved for every clock/config/seed: -> PASS

ACCEPTANCE RESULT: REJECT

per-seed detail, plateau_len=150 (the deciding cell):
  seed              WallClock  StructuralEmbedding  StructuralFull
  0x0000000000c0ffee     0.2000               0.2667          0.2667
  0x000000009ef879a7     0.1333               0.1333          0.1333
  0x000000013d2ff360     0.2000               0.2000          0.2000
  0x00000001db676d19     0.2000               0.2000          0.2000
  0x00000002799ee6d2     0.1333               0.1333          0.1333
  0x0000000317d6608b     0.2000               0.2667          0.2667
  0x00000003b60dda44     0.1333               0.1333          0.1333
  0x00000004544553fd     0.1333               0.2000          0.2000
  0x00000004f27ccdb6     0.2000               0.2000          0.2000
  0x0000000590b4476f     0.0667               0.0667          0.0667

exploratory (not gating): StructuralFull vs StructuralEmbedding mean recall delta
  plateau_len=20: StructuralFull=1.0000 StructuralEmbedding=1.0000 delta=0.0000pp
  plateau_len=60: StructuralFull=0.4133 StructuralEmbedding=0.4000 delta=1.3333pp
  plateau_len=150: StructuralFull=0.1800 StructuralEmbedding=0.1800 delta=0.0000pp
```

Reproduce with: `cargo build --release -p ruvector-structural-memory && cargo
run --release -p ruvector-structural-memory --bin benchmark`.

---

## Acceptance Result

| Clause | Threshold | Measured | Result |
|---|---|---|---|
| (a) long-plateau mean lead | ≥ 5.00pp | 2.00pp | **FAIL** |
| (b) short-plateau mean regression | ≥ -2.00pp | 0.00pp | PASS |
| (c) compute overhead ratio | ≤ 5.00x | 1.30x | PASS |
| (d) causal order preserved | all cells | all cells | PASS |

One mandatory clause fails → **REJECT**, per the pre-registered "all clauses
must pass" rule. The thresholds were fixed alongside the budget/noise
parameters before the first benchmark run of this experiment and were not
loosened after seeing this result.

---

## Why the Effect Is Real But Small

The per-seed detail table shows the mechanism working exactly as designed
when it fires: 3 of 10 seeds show `StructuralEmbedding` beating `WallClock`
by exactly +6.67pp (one extra correct memory out of 15, at a 25-item
budget), and it never loses. The remaining 7 seeds tie exactly. This
discrete, seed-dependent pattern is consistent with a genuine but *boundary*
effect: within a stable plateau, `StructuralEmbeddingClock` assigns nearly
identical age to every memory in that plateau (since the plateau contributes
almost zero accumulated arc length), so ranking within the plateau falls
back almost entirely to the coherence term — closely tracking the oracle's
true-cosine ranking. `WallClock`, by contrast, imposes an artificial
recency bias across the plateau that the oracle ranking does not share. But
that bias only *changes which items land inside vs. outside the fixed
25-item budget* when the true-cosine ranking and the wall-clock-biased
ranking disagree specifically near the budget cutoff boundary — and with
only 4 near-orthogonal random topic centroids per session, whether that
boundary disagreement actually occurs is itself a matter of which
particular noise draw the session got. This is a plausible, structurally
motivated explanation for the observed distribution, not a rescued
retroactive justification for the REJECT threshold: the threshold was fixed
before any of this per-seed data was collected.

---

## Failure Modes (and Things That Almost Made This Look Better Than It Is)

1. **Fractional budget made the first version of this experiment trivial.**
   At 30% of corpus size, the budget always comfortably contained an entire
   plateau's memory pool, so recall@15 was 1.0000 for every clock — a
   methodology bug, not evidence of "no difference." Fixed by switching to
   a fixed absolute budget (25) smaller than a long plateau's own pool.
2. **Context noise too large relative to the switch-jump size initially
   made `StructuralEmbeddingClock` behave as a near-linear reparametrization
   of `WallClock`** (same recall numbers for both, seed after seed) — because
   independent per-step noise accumulates roughly linearly in step count,
   the same shape as wall-clock aging, just with a different constant.
   Reduced context noise ~50x (0.05 → 0.001) relative to the topic-switch
   jump size to make "quiet" plateaus genuinely quiet in embedding-arc terms.
3. **Single-seed cherry-picking risk.** The first fixed-budget,
   fixed-noise run (seed `0xC0FFEE`) happened to show a comfortable
   +6.67pp lead — a result that, reported alone, would have looked like an
   unambiguous ACCEPT. Three more manually-tried seeds while validating the
   noise fix showed the lead is not consistent per-seed. Per this harness's
   explicit prohibition on cherry-picked seeds, the benchmark was rewritten
   to average over 10 seeds fixed by a deterministic formula before the
   final run, and gated on that mean. That is the number reported as this
   nightly's result.
4. **The entropy channel (`StructuralFull`) added no measurable benefit**
   over the pure embedding-arc clock at the deciding (150) configuration,
   and only a marginal +1.33pp at the 60-step configuration — reported
   honestly as a null/marginal exploratory result, not suppressed.

---

## Security

No new attack surface: this crate reads a synthetic in-memory corpus and
performs no I/O, network access, or untrusted deserialization. It does not
touch `ruvector-agent-memory`'s ledger, proof-gate, or capability-token
paths. If a structural-time compaction policy were ever wired into a
production memory store, the risk to evaluate would be adversarial context
manipulation: an agent (or a prompt-injected tool result) that deliberately
keeps the *reported* context embedding static while the *actual* topic
drifts would make a structural clock under-forget stale memories — the
mirror image of the benefit demonstrated here. That risk is out of scope
for this synthetic benchmark and is listed as required future work before
any production integration (see [ADR-340](../../../adr/ADR-340-structural-time-memory-decay.md)).

---

## Governance

This nightly's result is a REJECT of a pre-registered ACCEPT threshold, not
a promoted capability. No production code path is modified. The new crate
is additive (`crates/ruvector-structural-memory`, added to the workspace
member list) and carries no feature flag because nothing consumes it yet.

---

## MCP Implications

Not applicable at REJECT: no capability is being exposed for external
invocation. If a future iteration of this direction reached ACCEPT, the
natural MCP surface would mirror `ruvector-agent-memory`'s existing
`memory_compact(context, target_pct)` tool, with an added `clock: "wall" |
"structural"` parameter — narrow, and read/write-scoped identically to the
existing tool, not a new authority class.

---

## WASM Implications

`emergent-time` is dependency-free and has a companion `emergent-time-wasm`
crate already in the workspace; `ruvector-structural-memory` adds only
`Vec<f64>` arithmetic and one HashSet, so a WASM build is architecturally
unblocked. Not attempted in this nightly — no deployment claim is made.

---

## RVF Implications

A `Session { contexts, topics, memories, snapshots }` is exactly the shape
of an RVF-portable trajectory artifact: replaying it deterministically
(same seed → same session, verified by this crate's own tests) is a
prerequisite RVF already expects for reproducible cognitive state. Not
implemented here — noted because the reusability was evident during
implementation, not retrofitted for this section.

---

## RVM Implications

None identified. This experiment has no privileged operation, isolation
boundary, or coherence-domain crossing to enforce.

---

## ruFlo Implications

If a future version of this direction reached ACCEPT, the natural ruFlo
workflow is a periodic memory-maintenance job: run compaction with the
structural clock on a live agent's memory store on a schedule, logging
recall-preservation estimates against a held-out query set. Not
implemented; the mechanism (a scheduled compaction pass) already exists
conceptually in `ruvector-agent-memory`'s design notes.

---

## Practical Applications

| # | User | Problem | RuVector capability | Time horizon |
|---|---|---|---|---|
| 1 | Long-running coding agent | Loses cheap access to early-session decisions during a long refactor even though they're still relevant | Structural-time-weighted compaction (if reworked to clear the bar) | Near |
| 2 | Customer-support agent | Wall-clock decay discards case history from a long, stable ticket thread | Same mechanism applied to `ruvector-agent-memory` | Near |
| 3 | Research assistant agent | Rapid topic-switching sessions accumulate stale memory from abandoned threads | Structural clock's fast aging right after a switch (the flip side of this benchmark) | Near |
| 4 | Multi-agent swarm coordinator | Shared memory pool needs per-topic, not per-turn, retention policy | `ruvector-structural-memory` + `ruvector-namespace-merge` (2026-08-08) combined | Mid |
| 5 | Edge/Cognitum agent | Constrained memory budget needs the most information-dense retention policy | Same, on `emergent-time-wasm` | Mid |
| 6 | Compliance/audit agent | Needs to justify *why* a memory was kept or dropped | Compaction event sealed via `emergent_time::witness` | Mid |
| 7 | Personal AI assistant (weeks-long context) | Wall-clock decay under stable-life-routine stretches discards useful habits/preferences | Same mechanism at much longer plateau lengths | Long |
| 8 | Autonomous research agent (months-long project) | Session length in *turns* is a poor proxy for "how much has actually changed" | Structural time as the native temporal unit for agent memory, not wall time | Long |

---

## Long Horizon Applications

| # | Thesis | Required advances | RuVector role | Primary uncertainty |
|---|---|---|---|---|
| 1 | Agents run for years, not sessions; memory needs a temporal unit that isn't turns | Reliable low-drift detection at scale, not just synthetic plateaus | Native structural-time index | Whether real context embeddings are ever this cleanly "quiet" (see Limitations) |
| 2 | Multi-agent swarms need a shared notion of "how much has the world changed" for coordinated forgetting | Distributed clock synchronization across agents | `emergent-time` + swarm memory | Consensus cost of a shared structural clock |
| 3 | Structural time as an anomaly-and-retention unifier: the same clock that flags drift also ages memory | Single production implementation serving both roles | `structural_clock.rs`'s existing early-warning code, reused | Whether one metric can honestly serve both purposes without conflicting incentives |
| 4 | Proof-gated forgetting: a witness chain proving *why* a memory was structurally aged out | Signed, verifiable compaction decisions | `emergent_time::witness` + `ruvector-proof-gate` | Whether "why" is auditable without leaking the memory content itself |
| 5 | Edge cognition with bounded memory needs the most information-dense retention rule physically realizable | WASM structural-time compaction at sub-millisecond budgets | `emergent-time-wasm` | Compute budget on real edge hardware, not simulated |
| 6 | World models that track "how much has my environment model changed" as their own internal clock | Structural time applied to model-state deltas, not just embeddings | `StructuralMetric`'s `ΔG`/`ΔE` channels, unused in this nightly | No honest signal source demonstrated yet (this nightly's own limitation) |
| 7 | Self-healing agent memory that ages faster near contradictions | `ΔC` (coherence loss) channel, unused here | Same crate, extended | Needs a real coherence signal (e.g. `ruvector-coherence`) wired in |
| 8 | A general theory of "agent time" as the native coordinate for all RuVector temporal reasoning, replacing wall-clock timestamps repo-wide | Much broader validation than one compaction benchmark | `emergent-time` as a foundational primitive | This nightly is one data point, not sufficient evidence for a repo-wide claim |

---

## Competitor Comparison

| System | Documented recency mechanism | Directly measured here? |
|---|---|---|
| MemGPT / Letta | Token-budget eviction | documented_external_capability |
| Mem0 (2025) | LLM-driven ADD/UPDATE/DELETE, no continuous decay | documented_external_capability |
| Zep | Temporal knowledge graph, wall-clock validity windows | documented_external_capability |
| LangChain memory | Wall-clock / turn-count windows | documented_external_capability |

No comparison system was run or benchmarked in this nightly; the table
above reflects public documentation only, per this harness's rule against
treating undocumented or unmeasured external claims as directly comparable.
`ruvector-agent-memory`'s own wall-clock baseline (this nightly's
`WallClock` variant) is the only directly measured comparison point.

---

## Limitations

- **Synthetic corpus only.** Topic centroids are random unit vectors in a
  32-dimensional space; real conversational embeddings are not uniformly
  near-orthogonal and carry residual within-topic drift that this
  experiment's near-zero context noise (0.001) may understate. This is
  flagged, not hidden: the noise scale was chosen to make the mechanism
  measurable at all, and a follow-up should sweep it to find the drift
  level at which the effect disappears.
- **Small effect size.** Even where the structural clock wins, it wins by
  exactly one memory out of 15 (+6.67pp) — a real but modest margin at this
  scale.
- **Single dataset shape.** Only one topic-visitation pattern (each topic
  visited exactly once, in sequence) was tested. Revisited topics, more
  than 4 topics, or higher/lower embedding dimensions are untested.
- **`StructuralFull`'s entropy channel showed no benefit** at the deciding
  configuration — the exploratory extension did not strengthen the case.

---

## Falsification Criteria

This hypothesis is falsified by exactly what was measured: a fixed,
pre-registered mean-lead threshold at a fixed configuration, evaluated
honestly over multiple seeds. It was falsified. A different result would
require either a different (larger) drift-vs-noise ratio, a different
compaction-pressure regime, or a genuinely different scenario shape — any of
which is a new experiment, not a reinterpretation of this one.

---

## Production Path

**Not recommended for promotion in its current form.** If this direction is
revisited:

1. Sweep context-noise/drift ratio to find where the effect crosses 5pp
   reliably, if it ever does, rather than fixing one value.
2. Test with a real embedding source (e.g. actual LLM-embedded conversation
   turns) instead of synthetic random-unit-vector topics.
3. Wire a real `ΔC`/`ΔG` signal (e.g. from `ruvector-coherence` or
   `ruvector-mincut`) into `StructuralFull` rather than leaving those
   channels at zero weight.
4. Re-run the full multi-seed protocol against the new configuration before
   any acceptance claim.

---

## Next Research

- Sweep drift-vs-noise ratio (item 1 above) as a standalone follow-up —
  answers whether this REJECT is a parameter-regime artifact or a durable
  ceiling on the mechanism's effect size.
- Wire `ruvector-coherence`'s cluster-coherence score into `StructuralFull`'s
  `ΔC` channel as a genuine (not zero-weighted) signal.
- Test structural-time compaction on a real (not synthetic) embedded
  conversation corpus if one becomes available in-repo.

---

## References

- ADR-251 — `emergent-time`: calculus of emergent/relational time.
- 2026-06-14 nightly — Coherence-Weighted Agent Memory Compaction
  (`ruvector-agent-memory`), the production baseline this experiment's
  methodology mirrors.
- Park et al. 2023, *Generative Agents* (arXiv:2304.03442).
- Zhong et al. 2023, *MemoryBank* (arXiv:2305.10250, AAAI 2024).
- Mem0, 2025 production paper (arXiv:2504.19413).
- `crates/emergent-time/src/structural_clock.rs` — source of
  `StructuralProperTime`, `Clock`, `StateSnapshot`, `WallClock`, and this
  crate's own reused synthetic-scenario generation style.
