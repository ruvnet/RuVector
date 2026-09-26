# Architecture Decision Records — Earth Pulse Observatory

This directory records the architectural decisions for the **Earth Pulse
Observatory** harness, a MetaHarness Darwin-Mode bundle investigating
Earth's stable ~26-second microseism pulse (Gulf of Guinea source) and
the associated gliding tremors documented by Bruland & Hadziioannou
(2023). The model and the physics are **frozen**; the *harness* —
feature extraction, retrieval, scoring, validation, reporting — is what
evolves. ruVector provides the planetary memory/embedding layer.

All ADRs use the standard format: Title, Status, Context, Decision,
Consequences (positive/negative), Alternatives considered.

| ADR | Title | Status | Summary |
|-----|-------|--------|---------|
| [ADR-001](./ADR-001-freeze-physics-evolve-harness.md) | Freeze the Physics, Evolve the Harness | Accepted | Use Darwin Mode to evolve the investigation workflow across 7 surfaces; never the model, observations, or scientific truth (4 forbidden mutations). |
| [ADR-002](./ADR-002-ruvector-planetary-memory.md) | ruVector Planetary Memory with Separated Embeddings | Accepted | Store each pulse as a context-embedded event with separate waveform/environment/source/literature embeddings to avoid false similarity; contrastive pairing for causality. |
| [ADR-003](./ADR-003-hypothesis-scoring-and-promotion-gate.md) | Hypothesis Scoring and Promotion Gate | Accepted | Weighted six-component discovery score; promote a child only on out-of-sample F1 +3%, no FPR increase, held-out error -5%, mapped citations, and zero leakage. |
| [ADR-004](./ADR-004-validation-leakage-contradiction-logging.md) | Validation, Leakage Control, Contradiction Logging | Accepted | Hold out storm and calm-sea weeks, beat seasonal/swell/tide baselines out-of-sample, log contradictions; six-level discovery ladder with acceptance tests. |
| [ADR-005](./ADR-005-darwin-mutation-surfaces-and-safety.md) | Darwin Mutation Surfaces and Safety Policy | Accepted | Declare the 7 mutable surfaces; deterministic air-gapped mutator default; every mutation passes `validateGeneratedCode` (no new imports/network/fs/shell/env/deps), sandboxed, promote only on measured gain. |
| [ADR-006](./ADR-006-mincut-signal-class-partition.md) | Signal-Class Partition via ruVector Dynamic MinCut | Accepted | Partition the event-similarity graph into signal classes with `@ruvector/mincut-wasm` (dynamic min-cut / φ-expander hierarchy); `classes` = Level-2 cluster count; dynamic stream detects a new mechanism entering the record; graceful connected-components fallback. |

## Pipeline reference

- `src/detect-26s.ts` — spectral detection + source localization
- `src/extract-features.ts` — feature schema extraction
- `src/embed-events.ts` — separated ruVector embeddings (ADR-002)
- `src/score-hypotheses.ts` — weighted discovery score (ADR-003)
- `src/validate.ts` — holdout / leakage / contradiction validation (ADR-004)
- `.metaharness/objective.json` — scoring weights + gate thresholds
- `.metaharness/safety-policy.json` — mutation safety gate (ADR-005)
- `.metaharness/genome.json` — declared mutable surfaces (ADR-001, ADR-005)

## References

- Bruland, A. & Hadziioannou, C. (2023). Gliding tremors associated with
  the 26 s microseism.
- Microseism theory: ocean-wave / seafloor coupling (primary and
  secondary/double-frequency microseisms).
- Sibling harness: `harnesses/timesfm-harness` (same Darwin kernel).
