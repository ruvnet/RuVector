# Reusing an HNSW Health Monitor to Detect (and Fail to Fix) Agent Memory Drift

## Problem

Agent memory graphs — nodes are memories, edges are semantic/temporal
associations — drift structurally as they run: compaction evicts memories, and
associations that once connected two topics weaken as an agent's context moves on.
Most systems either rebuild the whole structure on a schedule (coarse, expensive)
or never check at all (silent quality decay). Is there a cheap, reused signal that
can trigger a *targeted* fix instead?

## Hypothesis

`ruvector-coherence` already ships a Spectral Coherence Score — Fiedler value,
spectral gap, effective resistance, degree regularity — built for monitoring HNSW
*index* health. Nothing had applied it to an agent-memory *association* graph, or
paired it with a repair action. The hypothesis: reuse the monitor unmodified as a
drift trigger, add a bounded repair (reconnect the nodes nearest a weak graph cut,
per the Fiedler eigenvector, to their current semantic neighbors), and see whether
recall of each memory's original associations recovers after injected drift.

## Technical Design

A deterministic simulation (seeded, reproducible) runs 3000 memory arrivals against
8 topic clusters, with coherence-weighted compaction every 25 steps (evict the 3
lowest-scoring memories: recency + frequency + local edge-weight coherence) and
injected drift every 60 steps (decay cross-topic bridge edges to 15% weight, or
remove them below a floor, cycling through all topic pairs). Three variants share
the exact same arrival/compaction/drift script:

- **NoRepair** — no monitoring, the baseline.
- **MonitorOnly** — runs `ruvector_coherence::spectral::HnswHealthMonitor` every 10
  steps, records alerts, never mutates the graph.
- **SpectralRepair** — same monitoring; on ≥2 simultaneous alerts, reconnects the
  12 alive nodes with smallest `|fiedler_vector[i]|` (closest to a weak cut) to up
  to 3 current top-k semantic neighbors each, and logs a SHA-256 hash-chained
  witness receipt for every repair.

Ground truth for scoring: each memory's neighbor set *at the moment it was
created*, frozen and filtered to still-alive nodes at measurement time. Retrieval
is a tightly-budgeted (20-expansion) best-first graph walk, not a raw
nearest-neighbor lookup — it's testing whether the *graph structure* still
supports finding what it once could.

## Implementation

Real Rust, `crates/ruvector-memory-self-repair`, 8 source modules plus a benchmark
binary and an integration-test acceptance suite, workspace member, depends on
`ruvector-coherence` (path dependency, `spectral` feature) — not a reimplementation
of the spectral math, an actual reuse of the existing crate.

## Actual Benchmark Evidence

Three seeds (42, 7, 123), release build, x86_64 Linux, rustc 1.94.1:

| seed | baseline recall@10 | repair recall@10 | uplift |
|---:|---:|---:|---:|
| 42 | 0.7603 | 0.7487 | −1.16 pp |
| 7 | 0.7447 | 0.7385 | −0.62 pp |
| 123 | 0.7538 | 0.7446 | −0.92 pp |

Pre-registered acceptance threshold: **+10pp**. Result: **consistently negative**
across all three seeds — not noise around zero.

What *did* work: drift detection. 49/49 injected drift waves were caught within a
20-step window on every seed (100% recall), using the health monitor's existing,
unmodified public API. `MonitorOnly` produced a graph byte-identical to `NoRepair`
on every seed (edge count, alive-node count match exactly) — confirming monitoring
itself is causally inert, which isolates the repair action as the sole source of
the (negative) recall difference. Repair stayed correctly bounded (3.0–3.1% of
final edges, under an 8% cap) and the witness chain verified with zero failures
across 279–283 receipts per seed.

### A methodology bug worth naming

The first version of this benchmark scored ground truth as "any other alive
same-topic memory" with a 200-expansion search budget. Every variant — including
the undrifted baseline — hit **recall@10 = 1.0000**. That's not a result, it's a
saturated instrument: with hundreds of same-topic memories typically alive, a
top-10 metric is satisfied by whatever's locally abundant, never forcing the walk
to cross a decayed bridge. Fixed by narrowing ground truth to each memory's frozen
original neighbors and tightening the budget to 20 — before looking at which
variant that favored, not after.

## Why Repair Likely Hurt (Hypothesis, Not Confirmed)

Repair reconnects to a node's *current* top-k semantic neighbors, not necessarily
the *specific* neighbor the metric is scored against — by step 3000, newer, closer
arrivals can have displaced a node's original best match from its current top-5.
Worse, the repaired edges feed into `local_coherence`, which feeds into compaction's
retention score — so a repaired node becomes comparatively less likely to be
evicted, which (since compaction always evicts a fixed count) makes some other node
more likely to be evicted instead, possibly removing exactly the neighbor the
metric credits. Not confirmed by an ablation in this PoC; that ablation is the
top item in Next Research.

## Limitations

Synthetic data, not real agent-memory content. `ruvector-agent-memory`'s exact
compaction formula was reimplemented locally (it's outside the cargo workspace)
rather than depended on directly. Single-threaded, no concurrency stress test. No
WASM build measured, though the underlying spectral code has no external solver
dependency. The compaction-interaction mechanism is a hypothesis, not a proven
cause.

## Production Relevance

None yet, and that's the honest headline: this specific repair heuristic is
rejected. What's production-relevant is narrower and still useful — the spectral
health monitor built for HNSW indices generalizes, unmodified, to a completely
different graph domain, and the witness-chain pattern for auditing autonomous
structural changes worked exactly as designed. Both are reusable building blocks
for whichever repair heuristic eventually clears the bar this one didn't.

## RuVector Ecosystem Implications

Connects `ruvector-coherence` (spectral monitor, reused unmodified), agent-memory
compaction (2026-06-14 nightly), and the witness/provenance pattern (2026-08-13
retrieval-receipts nightly) — closing a "spectral gate future" placeholder left
open in the 2026-06-13 temporal-coherence-agent-memory nightly. A follow-on that
fixes the repair mechanism is a natural ruFlo scheduled workflow and, longer term,
an RVM-gated capability boundary with RVF-portable health receipts — neither
implemented here, both flagged as materially relevant.

## Future Direction

1. Ablation isolating the compaction-interaction hypothesis (decouple repaired
   edges from retention scoring, re-run, check if the sign flips).
2. A narrower "snapshot-restore" repair (re-add the specific decayed original
   edge instead of reconnecting to current top-k) as a cleaner test of whether
   *any* repair can clear the bar.
3. Real dependency on `ruvector-agent-memory` instead of a local reimplementation.
4. Measure the incremental (`SpectralTracker::update_edge`) monitoring path against
   the `full_recompute` used throughout this PoC — overhead was 14–34× baseline
   wall time here, all of it from full recomputation every check.

## References

- `crates/ruvector-coherence/src/spectral.rs` (reused, unmodified)
- `docs/research/nightly/2026-06-14-agent-memory-compaction/`
- `docs/research/nightly/2026-06-13-temporal-coherence-agent-memory/`
- `docs/research/nightly/2026-08-13-retrieval-receipts/`
- Full methodology, mermaid architecture diagram, and per-seed raw evidence:
  `docs/research/nightly/2026-08-16-spectral-memory-self-repair/README.md`
- `docs/adr/ADR-305-spectral-memory-self-repair.md`
