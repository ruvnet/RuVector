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

## Future work (from the brief, not yet built)

Resonant identity / continuity recognition, physical CAPTCHA (challenge-response
proof-of-reality), boundary-first prediction, self-healing sensor topology,
swarm-scale min-cut, and the "ambient nervous system" hardware node. This ADR is
the substrate those build on.

## Validation

11 tests (9 unit + 2 integration), including the brief's exact flagship scenario
(inert object move → RF/vibration/acoustic support, thermal contradicts, novelty
high, action = observe) and the missing-routine-return absence signal. clippy
clean; all source files < 500 lines.
