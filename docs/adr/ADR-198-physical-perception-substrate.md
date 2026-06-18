---
adr: 198
title: "Physical Perception Substrate — delta → boundary → coherence → proof → action"
status: accepted
date: 2026-06-08
authors: [ruvnet, claude]
related: [ADR-196, ADR-197]
tags: [perception, sensing, coherence, min-cut, proof-gate, edge-ai, csi, ruview]
---

# ADR-198 — Physical Perception Substrate

## Status

**Accepted (initial vertical slice implemented).** Crate
`crates/ruvector-perception`.

## Context

WiFi/edge sensing SOTA is converging on better **classifiers**: CSI foundation
models, self-supervised CSI representations (CSI-JEPA-style), adaptive near-sensor
fusion (FusionSense-style), and dynamic-graph anomaly detection (which still
flags interpretability + scalability as open). All answer *"what is this?"* and
emit *confidence → alert*.

The wedge is not a better classifier. It is the **layer underneath** one: a
trusted-physical-memory engine that answers *"what changed, where did the
boundary move, and is the change coherent enough to act on?"* and requires
**evidence, not confidence**, before exercising any authority. This reframes the
pipeline:

```
classification → confidence → alert      (today)
delta → boundary → coherence → proof → action   (this ADR)
```

It also removes the dependence on a fixed task label (fall / gesture / occupancy
/ leak / bearing-failure): it models **state transition itself**.

## Decision

Implement the pipeline as a standalone crate built on the dynamic min-cut engine.

1. **Delta** (`state`, `engine`) — every reading becomes a delta against a
   rolling per-(zone, modality) baseline (EWMA), plus a learned *responsiveness*
   (how often that channel reacts in that zone).
2. **Boundary** (`coherence`) — zones are nodes in a coherence graph (edge weight
   = delta-pattern agreement). Dynamic min-cut (`ruvector-mincut`) isolates the
   side that broke away — the moved boundary, not a class.
3. **Contradiction as information** — a modality that *usually* reacts in a zone
   but stayed silent is a first-class contradiction, weighted by the modality's
   physical **spoof-resistance** (modalities are physically typed: RF ≠ thermal).
   This is what flags an inert object-move (RF/vibration/acoustic respond,
   thermal — which would respond to an animate source — does not).
4. **Proof** (`witness`) — a proof gate maps (novelty, coherence, contradiction)
   to **bounded authority** `Ignore → Observe → Alert → Mutate`, and emits an
   auditable SHA-256 evidence chain (raw hash, feature hash, scores, boundary,
   policy, prior-witness hash). Contradicted evidence is **capped at Observe** —
   it never escalates on confidence alone.
5. **Absence** (`absence`) — a *missing* expected continuation (e.g.
   `bed_exit → bathroom_path → return_path` where the return never arrives) is
   detected as structural incompleteness, a safety signal, not a threshold.

The headline output is a `DeltaWitness` (changed_boundary, supporting /
contradicting modalities, novelty, coherence, contradiction, action,
evidence_hash, prev_hash) — a structured delta, not a label.

## Consequences

**Positive**
- Task-label-free: detects unknown physical changes without retraining.
- Auditable: every action is backed by a replayable evidence chain (matters for
  elder care / industrial / civic / medical governance).
- Interpretable localisation: min-cut says *where* coherence broke and *why*
  (which modalities support vs contradict) — addressing the open
  interpretability gap in dynamic-graph anomaly work.
- Reuses existing min-cut machinery; small, dependency-light, `#![forbid(unsafe_code)]`.

**Negative / honest scope**
- This is the **mechanism**, demonstrated on **synthetic** multi-modal deltas —
  not validated on real CSI/hardware, and not benchmarked against CSI-JEPA /
  FusionSense (different layer). No accuracy claims.
- Novelty (nearest-prior distance), contradiction (responsive-but-silent), and
  coherence (cut cleanliness) are principled **heuristics**, not learned.
- Single-window; no temporal model of the delta beyond EWMA baselines and the
  absence-sequence monitor. Boundary detection is O(zones²) edges + exact min cut
  (fine for rooms/facilities, not yet city-scale).

## Capability modules (built on the substrate)

Five further beyond-classification capabilities from the brief are implemented as
self-contained modules (each emits structure, not a label):

- **`captcha`** — Physical CAPTCHA: a learned per-stimulus multi-modal
  challenge-response profile; a fresh response is verified within delay/magnitude
  tolerance, weighted by spoof-resistance, yielding a `RealityProof`. Detects
  replay/spoof (proof-of-real-physical-field).
- **`predict`** — Boundary-first world model: forecasts *where coherence breaks
  next* (`instability = coherence·(1+contradiction)`, level + least-squares
  trend) rather than full future states.
- **`identity`** — Resonant identity / continuity: per-object EWMA signature;
  cosine-distance drift detection answers "is this still the same physical
  thing?" (panel loosened, bearing worn, casing tampered).
- **`hypothesis`** — Multi-modal disagreement engine: contradictions produce
  *ranked hypotheses* (RealEvent / SensorDrift / SensorRelocation /
  AdversarialReplay / EnvironmentalArtifact), not forced agreement.
- **`topology`** — Self-healing sensor topology: an EWMA agreement graph
  classifies each sensor Critical / Redundant / Noisy / Normal; Critical =
  articulation point (removal fragments the graph — the extreme single-edge cut).
- **`swarm`** — Facility/swarm-scale fragility: rooms/machines/routers as a
  coupling graph; global min-cut answers "where is the system structurally
  closest to breaking?" Bottlenecks are derived from the weakest link (edge
  weights), because the engine's min-cut *value* is reliable but its *partition*
  is not.
- **`custody`** — Sensor chain of custody: a tamper-evident, replayable ledger
  of witnesses (chain-linkage verification over the SHA-256 evidence hashes;
  honest scope — link integrity, not raw-signal re-hash).
- **`reality`** — Reality-graph agent grounding: an agent *queries reality*
  (presence / changed-since / which-untrusted / action-allowed) and gets answers
  **backed by witness evidence hashes**, not prompt inference.
- **`node`** — `NervousSystemNode`: the appliance facade wiring engine + reality
  graph + custody ledger + boundary forecaster. Ingests readings, emits
  deltas/boundaries/coherence/witnesses/forecasts (never raw signal), and answers
  grounded queries.

## Future work (from the brief, not yet built)

The remaining items are out of pure-software scope: the physical "ambient
nervous system" **hardware** node, and replacing the heuristic scorers
(novelty / contradiction / coherence) with **learned** models validated on real
CSI. Everything above is a mechanism demonstration on synthetic signals.

Known limitation surfaced during testing: coherence boundary detection is
ambiguous with exactly **two** zones (a single-edge min cut splits symmetrically;
the minority side is arbitrary). Use ≥3 zones for a well-defined changed
boundary — documented and reflected in the tests.

## Validation

59 tests (54 unit + 2 integration + 3 doctest), deterministic across repeated
runs. Highlights: the brief's exact flagship scenario (inert object move →
RF/vibration/acoustic support, thermal contradicts, novelty high, action =
observe); the missing-routine-return absence signal; physical-CAPTCHA replay
rejection; boundary forecast of a destabilising zone; identity drift on a
tampered signature; ranked hypotheses (RealEvent / SensorDrift / AdversarialReplay
first under the right evidence); topology roles (bridge → Critical, near-duplicate
→ Redundant, lone-disagreer → Noisy); facility fragility (weakest link found);
custody chain verify + tamper detection; reality-graph grounded queries; and the
end-to-end `NervousSystemNode` (witness chain + grounded query). Built across two
parallel agent swarms, then integrated and validated. clippy clean; all source
files < 500 lines.
