# ADR-305: Spectral Drift Detection for Agent Memory Graphs — Targeted Repair Hypothesis Rejected

## Status

**Rejected for the repair mechanism as designed. Accepted (validated) for the
detection mechanism.** Experimental crate (`ruvector-memory-self-repair`),
not wired into any production agent-memory path. This ADR documents a
negative result with a clear, reproduced mechanism, per the nightly research
process's rule that a falsified hypothesis with good evidence is a valid
outcome — this is not a rejected-and-forgotten idea, it is a specific
finding about where a plausible design breaks.

## Context

Two RuVector capabilities existed independently before this work:

- `ruvector-coherence`'s `spectral` feature (added 2026-07-28, ADR context
  in that crate) computes a composite Spectral Coherence Score — Fiedler
  value, spectral gap, effective resistance, degree regularity — over a
  graph Laplacian, via `HnswHealthMonitor`. It was built for HNSW *index*
  health monitoring and was never applied to an agent-memory association
  graph.
- The 2026-06-14 agent-memory-compaction nightly (`ruvector-agent-memory`)
  established coherence-weighted eviction (recency + frequency + local
  semantic coherence) as a reasonable compaction baseline, and its own
  README explicitly flagged "spectral gate future" as unimplemented
  follow-on work.

Neither prior nightly closed that loop: nothing in the repository used
spectral graph-health signals to *detect* structural drift in an evolving
agent-memory graph, and nothing attempted a *repair* action gated on that
signal. The Step-2/Step-6 novelty gate for this run confirmed this via an
Explore survey of all 27 prior nightly topics (rabitq through
retrieval-receipts / entropy-adaptive-ann): mincut is used for RAG context
bounding and namespace routing, coherence scoring drives retrieval ranking
and compaction, and both write- and read-provenance receipts exist — but no
prior work combines spectral graph monitoring with agent-memory structural
repair.

## Hypothesis

```
Given an agent memory graph (nodes = memories, edges = semantic top-k +
temporal-chain associations) evolving over 3000 arrivals with periodic
coherence-weighted compaction (evicting 3 lowest-retention-score memories
every 25 steps) and injected structural drift (cross-topic bridge edges
decayed to 15% of their weight, or removed below a floor, in waves every
60 steps, cycling through all topic pairs),

when a bounded, spectrally-triggered repair reconnects the alive nodes
with the smallest-magnitude Fiedler-eigenvector components (i.e. those
sitting closest to a weak graph cut) to their current top-k semantic
neighbors, capped at 12 nodes / 3 edges each per repair event,

then recall@10 of each memory's *originally-formed* associative neighbors
(the ones it was wired to at creation time, still alive) should improve by
at least 10 percentage points relative to a no-repair baseline,

subject to: repair touching no more than 8% of the final graph's edges
(so it counts as targeted, not a rebuild-in-disguise), a hash-chained
witness log of every repair verifying with 100% integrity, and the
underlying `HnswHealthMonitor`/Fiedler-vector reuse from
`ruvector-coherence` remaining unmodified.
```

## Decision

**Do not promote the repair mechanism.** The spectral trigger (reused,
unmodified `ruvector-coherence::spectral`) reliably *detects* injected
drift — 100% detection recall across three seeds, within a 20-step window
of every injection — but the targeted-reconnection repair action **did not
improve, and mildly hurt, recall of a memory's original associations**,
consistently across three independent seeds:

| seed | baseline recall@10 | spectral-repair recall@10 | uplift |
|-----:|--------------------:|---------------------------:|-------:|
| 42   | 0.7603              | 0.7487                     | −1.16 pp |
| 7    | 0.7447              | 0.7385                     | −0.62 pp |
| 123  | 0.7538              | 0.7446                     | −0.92 pp |

The +10pp acceptance threshold was not met on any seed; the sign is
consistently negative, not merely noisy. **Acceptance result: REJECT** (see
Evidence below for full methodology and the honest mid-run correction that
was required to get a benchmark capable of measuring this at all).

## Evidence

Ran via `cargo run --release -p ruvector-memory-self-repair --bin benchmark`
(seed overridable with `SIM_SEED`), on the container's x86_64 Linux host,
rustc 1.94.1, release profile, single-threaded, three variants over an
identical arrival/compaction/drift script per seed:

- `NoRepair` — baseline, no monitoring, matches wall time budget of the
  underlying system with zero overhead.
- `MonitorOnly` — runs the exact same `HnswHealthMonitor` checks as the
  repair variant every 10 steps, but never mutates the graph. Its graph is
  byte-for-byte identical to `NoRepair`'s (edge count and alive-node count
  match exactly on every seed) — this is a built-in sanity check that
  monitoring itself is causally inert, isolating the repair action as the
  only source of any recall difference.
- `SpectralRepair` — same monitoring, plus the bounded repair action
  described above when `check_health()` returns ≥2 simultaneous alerts
  (the same threshold at which `HnswHealthMonitor` itself already emits
  `RebuildRecommended`).

Config (fixed before any run, unchanged after seeing results):
`dim=24, n_topics=8, total_steps=3000, semantic_k=5, embedding_noise=0.12,
compaction_interval=25, compaction_evict_count=3, drift_interval=60,
drift_decay_factor=0.15, drift_max_edges_per_wave=40`. Full raw JSON
evidence per seed is written by the benchmark binary itself (not hand
transcribed) and referenced in the research README.

**Detection quality** (from the `MonitorOnly` run, seed 42 shown, all three
seeds within noise of each other): 49 drift waves injected, 49/49 detected
within a 20-step window (100% recall). Alert precision (fraction of the
297 raised alerts falling within that window of *some* injected drift) was
~49% — the other alerts are plausibly real: compaction-driven eviction is
itself a structural-drift source this design deliberately did not suppress,
so an honest reading is that the monitor is also catching compaction-driven
fragmentation, not raising false alarms in the classic sense. This was not
further disambiguated and is flagged as an open question below rather than
asserted.

**Repair stayed targeted, not a rebuild**: cumulative edges added by repair
were 3.0–3.1% of the final graph's edge count across all three seeds — well
under the 8% bound — and the witness chain (SHA-256, hash-chained receipts
per repair event, `WitnessLog::verify()`) validated with zero failures on
every run, including the four-test acceptance suite (`tests/acceptance.rs`)
that exercises this end-to-end rather than only the benchmark binary.

**Monitor overhead is real and non-trivial**: baseline wall time ~212ms for
the full 3000-step simulation; `MonitorOnly` ~3.0–3.1s (≈14×); with repair,
~7.0–7.4s (≈34×). This is 299 full spectral recomputes (conjugate-gradient
Fiedler estimation via `estimate_fiedler`) over a graph that grows to ~2600
alive nodes / ~15k edges. This cost was not gated in the acceptance
criteria (only recall uplift, repair-edge fraction, and witness integrity
were), but it is reported honestly because it would matter for any future
attempt at this direction: periodic full recompute at this frequency is
not free, and `SpectralTracker::update_edge`'s incremental path (unused
here — see Limitations) exists precisely to avoid this cost.

### A methodology correction worth recording

The first benchmark run used ground truth = "any other alive same-topic
memory" and a generous 200-node search budget. Every variant, including the
undrifted, unmonitored baseline, scored **recall@10 = 1.0000**. This was not
a positive result — it meant the metric could not discriminate anything: with
several hundred same-topic memories typically alive and a top-10 metric, a
best-first walk trivially fills its top 10 from whatever is locally
abundant, without ever needing to cross a decayed bridge. Ground truth was
redefined to each memory's own originally-formed neighbor set (frozen at
creation time, filtered to still-alive), and the search budget was tightened
to 20 expansions — both changes made *before* drawing any conclusion from a
discriminating result, not after seeing which variant they favored (per
Step 32/40 of the nightly process: do not change the hypothesis after
benchmarking begins; this was a repair of a degenerate instrument, not a
hypothesis change — the corrected version is what the Given/When/Then above
describes, and it is the only version whose numbers are reported).

## Why the repair likely hurts, not just fails to help

Not independently confirmed by a further ablation (flagged as next-step
work, not asserted as proven), but consistent with the measured data: the
repair action reconnects a target node to its **current** top-k semantic
neighbors, not necessarily the **specific** neighbor recall@10 is scored
against. Because new memories keep arriving throughout the 3000-step run, a
node's true top-5 semantic match at creation time can be displaced from its
*current* top-5 by later, closer arrivals — so a repair can legitimately
improve the node's general connectivity while doing nothing for, or even
crowding out attention from, the specific original edge being measured.
Compounding this, `scoring::local_coherence` (mean adjacent edge weight)
feeds directly into the compaction retention score, so a repaired node's
higher connectivity makes it comparatively *more* likely to survive future
compaction rounds — which, since compaction always evicts a fixed count
per interval, makes some other node *more* likely to be evicted instead.
That second-order effect can remove precisely the neighbors the metric
credits, in variants where repair ran and baseline did not.

## Consequences

- `ruvector-coherence::spectral` gains a second validated consumer
  (`HnswHealthMonitor`/`estimate_fiedler`, used unmodified) beyond its
  original HNSW-index-health use case, with evidence it generalizes
  cleanly to an agent-memory association graph without any change to the
  reused crate.
- The specific repair heuristic (Fiedler-vector-magnitude node selection →
  reconnect to current top-k semantic neighbors) is falsified for the
  "restore original associations" goal and should not be reused as-is by a
  future attempt at this direction.
- The witness-chain pattern (hash-chained repair receipts, following the
  2026-08-13 retrieval-receipts technique) worked exactly as designed —
  100% chain integrity across every run — and is a reusable building block
  independent of whether the repair itself is useful.
- No production code path is affected; this stays an experimental,
  non-workspace-default crate pending a redesigned repair mechanism or a
  decision to abandon the repair half of this direction while keeping the
  detection half.

## Alternatives (for a follow-on attempt, not implemented here)

1. **Snapshot-restore repair**: instead of reconnecting to *current*
   top-k, repair could re-add the *specific* original edge if the
   neighbor is still alive (a much narrower, more mechanically obvious
   fix — but see Rejection Criteria below on why this wasn't the first
   thing tried: it doesn't test whether the spectral signal is useful for
   anything beyond "put back what was removed", which is a weaker claim).
2. **Decouple repair from compaction scoring**: exclude repaired edges
   from `local_coherence` (or discount them) so repair cannot change
   *who* gets evicted, isolating whether the negative result is really the
   second-order compaction-interaction effect described above.
3. **Reduce monitor cost** via `SpectralTracker::update_edge`'s
   incremental path (built into `ruvector-coherence` already, unused by
   this experiment, which always called `full_recompute` for simplicity
   and measurement clarity) before any latency-sensitive production
   consideration.

## Implementation Plan

Not applicable — rejected for promotion. If a follow-on nightly pursues
Alternative 1 or 2 above, it should be a new dated nightly topic, not a
silent edit to this one, per the "never silently change the hypothesis
after seeing results" rule; this ADR's numbers stand as the record of what
was actually tried and measured here.

## API Shape

`ruvector-memory-self-repair` exposes: `MemoryGraph`, `SimConfig` +
`build_arrivals`/`build_drift_schedule`/`apply_arrival`/
`apply_compaction`/`apply_drift_wave`, `Runner`/`RepairConfig`/`Variant`
(`NoRepair` | `MonitorOnly` | `SpectralRepair`), `WitnessLog`, and
`mean_recall_at_k`. None of this is intended as a stable public API; it is
research-tier scaffolding for reproducing or extending this experiment.

## Feature Flags

None. The crate is entirely experimental and workspace-buildable but not
depended on by any production crate.

## Benchmark Evidence

See the Evidence section above and
`docs/research/nightly/2026-08-16-spectral-memory-self-repair/README.md`
for full detail, raw JSON evidence, and the reproduction command.

## Security

- Repair actions are bounded (≤12 nodes, ≤3 reconnects each per event) and
  can only *add* edges derived from the node's own already-stored,
  never-decayed embedding — it cannot fabricate associations to memories
  it has no legitimate semantic basis for connecting to.
- Every repair produces a SHA-256 hash-chained witness receipt
  (`WitnessLog`); `verify()` detects any single mutated field in any
  receipt in the chain (unit-tested: `witness::tests::tampering_is_detected`).
- This experiment never grants the monitored/repaired system authority to
  evict or delete anything beyond the existing compaction policy it did
  not change; repair is strictly edge-additive.

## Governance

No autonomous production authority is proposed or implied. This ADR
documents a rejected design; nothing here should be wired into a live
agent-memory path without a follow-on nightly that either fixes the
mechanism (see Alternatives) or independently re-validates a different one.

## Failure Modes

- **Monitor cost dominates at high check frequency**: at `MONITOR_INTERVAL
  = 10` steps, monitoring alone was ~14× baseline wall time on a ~2600-node
  graph using full Laplacian recompute every check. Not tuned or optimized
  in this experiment; see Alternatives #3.
- **Repair can compound compaction bias**: see mechanism discussion above;
  not confirmed by ablation, flagged as the leading hypothesis.
- **Alert precision (~49%) was not disambiguated** between "false alarm"
  and "correctly detecting compaction-driven drift the injection schedule
  didn't cause" — an open question, not resolved here.

## Migration

None — nothing in production depends on this crate.

## Rollback

Trivial: the crate is additive, workspace-member-only, and not depended on
by any other crate. Removing it from `Cargo.toml`'s `members` list and
deleting `crates/ruvector-memory-self-repair` fully reverts this change.

## Rejection Criteria

This specific repair mechanism (Fiedler-magnitude node selection → current
top-k semantic reconnection) should remain rejected unless a follow-on
experiment demonstrates, with the same or stricter methodology (frozen
ground truth, tightened search budget, multi-seed reporting, unmodified
reuse of `ruvector-coherence::spectral`):

- A statistically consistent **positive** recall uplift across ≥3 seeds
  (not just a single favorable run), or
- A clean ablation showing the negative result was entirely the
  compaction-interaction second-order effect (Alternative #2), in which
  case a decoupled variant should be re-benchmarked before any promotion
  decision.

## Open Questions

- Is the ~49% alert precision actually a false-positive problem, or is the
  monitor correctly also catching compaction-driven (non-injected)
  structural drift? Answering this needs an ablation with compaction
  disabled, isolating drift-only alerts.
- Would Alternative 1 (snapshot-restore repair) meet the acceptance
  threshold, and if so, does that mean the spectral trigger is doing
  useful work, or would a much simpler "periodically re-check original
  edges still exist" policy achieve the same result without any spectral
  computation at all? This is the sharpest open question for whether
  spectral monitoring earns its cost in this application, as opposed to
  the HNSW-index-health application it was originally built for.
- Does `SpectralTracker::update_edge`'s incremental-update path (unused
  here) change the overhead picture enough to make higher-frequency
  monitoring viable, independent of whether repair itself is fixed?
