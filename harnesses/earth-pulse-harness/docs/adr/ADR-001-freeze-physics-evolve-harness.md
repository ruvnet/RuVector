# ADR-001: Freeze the Physics, Evolve the Harness (MetaHarness Darwin Mode)

## Status

Accepted

## Context

The Earth Pulse Observatory investigates a genuinely puzzling natural
phenomenon: a remarkably stable, persistent microseismic signal with a
dominant period near 26 seconds, long associated with a source in the
Gulf of Guinea, and the *gliding tremors* documented by Bruland &
Hadziioannou (2023, "Characterizing the seismic gliding tremors
associated with the 26 s microseism"). The broader physical framing is
the established microseism theory of ocean-wave / seafloor coupling
(primary and secondary/double-frequency microseisms), in which ocean
swell loads the seafloor and radiates Rayleigh-wave energy detectable
on stations worldwide.

We are building a *research harness*, not a model. The harness ingests
seismic windows plus oceanographic context, detects pulse events,
extracts features, embeds them into ruVector's planetary memory, scores
competing scientific hypotheses, and validates conclusions. Two things
are explicitly **frozen**:

1. **The model** — the inference/embedding model is a FROZEN, pinned
   artifact. We do not fine-tune it, swap it, or let any evolutionary
   process touch its weights. This guarantees that a result is
   attributable to the *workflow*, not to silent model drift.
2. **The physics / observations** — the seismic record and the
   established physical theory are ground truth. The harness may
   reinterpret, re-weight, and re-rank, but it may never invent or
   alter an observation.

What we *do* want to improve over time is the **investigation
workflow** itself: how we window the spectrum, how we check
source localization, how we shape embeddings, how we score hypotheses,
how we detect anomalies, how we generate reports, and how we detect our
own failures. MetaHarness Darwin Mode gives us a disciplined,
auditable way to evolve exactly those surfaces while leaving truth
untouched.

This mirrors the sibling `harnesses/timesfm-harness`, which uses the
same `@metaharness/kernel` Darwin pattern (frozen TimesFM model,
evolvable engineering harness). The Earth Pulse harness applies the
same discipline to a scientific-discovery domain instead of a coding
domain.

## Decision

We adopt **MetaHarness Darwin Mode** with a strict separation between
the FROZEN substrate and the EVOLVABLE harness.

### The 7 evolvable surfaces

Darwin Mode (the `evolve` skill) is permitted to propose mutations to,
and only to, these seven surfaces:

1. **Feature extraction** — spectral feature design, amplitude-envelope
   shaping, glide-slope estimation in `src/extract-features.ts`.
2. **Source-localization checks** — beamforming parameters, array
   geometry handling, and the source-stability tests that confirm the
   Gulf of Guinea origin in `src/detect-26s.ts`.
3. **Embedding schemas** — the separated sub-embedding layout (waveform,
   environment, source, literature) and normalization strategy in
   `src/embed-events.ts` (see ADR-002).
4. **Hypothesis scoring** — the weighted discovery-score function and
   evidence aggregation in `src/score-hypotheses.ts` (see ADR-003).
5. **Anomaly / contradiction tests** — the statistical tests that flag
   surprising windows and the contradiction logger in `src/validate.ts`
   (see ADR-004).
6. **Report generation** — how findings, contradictions, and citations
   are assembled into an auditable report.
7. **Failure detection** — the self-checks that decide a harness run is
   untrustworthy (leakage detected, baseline not beaten, citation
   unmapped) before any promotion (see ADR-003, ADR-004).

These surfaces are declared in `.metaharness/genome.json`; the
objective they optimize is declared in `.metaharness/objective.json`.

### The FORBIDDEN mutations

The following are hard-forbidden and enforced by the safety policy
(`.metaharness/safety-policy.json`, see ADR-005). A child harness that
performs any of these is rejected regardless of measured score:

- **No fabricated seismic observations.** A mutation may not synthesize,
  alter, interpolate-over, or delete entries in the observed record.
  Observations are read-only inputs.
- **No invented citations.** Every cited claim must map to a real source
  document already present in the literature corpus. Generating a
  plausible-looking reference that does not exist is an automatic
  rejection (citation grounding is also a scored component, ADR-003).
- **No test-label leakage.** A mutation may not let information from
  held-out test windows (storm weeks, calm-sea weeks) influence
  training/feature-fitting windows. Leakage is checked structurally by
  `src/validate.ts`, not merely scored.
- **No promoting a hypothesis without beating a baseline.** A hypothesis
  or a child harness may only be promoted if it beats the relevant
  baselines (seasonal average, swell-only, tide-only) *out-of-sample*.
  Promotion on in-sample fit alone is forbidden.

### Why the model is frozen

A frozen model is the control variable. If both the model and the
harness could change, an improvement in the objective could not be
attributed to better *methodology* versus a luckier *model*. By pinning
the model artifact:

- Results are reproducible: re-running a promoted harness on the same
  data yields the same conclusions.
- Darwin's selection pressure is applied purely to the workflow, which
  is the thing we actually want to learn about.
- Scientific claims remain falsifiable: a contradiction logged today
  remains a contradiction tomorrow unless the *data* changes.

## Consequences

### Positive

- Clean attribution: every gain is a workflow gain, not model drift.
- Scientific integrity is structurally enforced, not merely encouraged —
  the forbidden mutations are validator-checked, not vibes-checked.
- Reproducibility: a promoted harness + frozen model + dataset hash is a
  fully replayable experiment.
- The same Darwin machinery already proven in `timesfm-harness` is
  reused, lowering operational risk.

### Negative

- A frozen model caps absolute performance: if the model itself is the
  bottleneck, the harness cannot route around that limit. This is an
  accepted trade for trustworthiness.
- Maintaining the FORBIDDEN-mutation gates adds validator complexity and
  some false-rejection risk for legitimate-but-unusual mutations.
- Evolution is slow by construction: each promotion requires
  out-of-sample, leakage-free evidence.

## Alternatives considered

1. **Evolve the model too (full AutoML).** Rejected: destroys
   attribution and reproducibility, and risks the model "learning" the
   answer rather than the harness discovering it.
2. **Hand-tune the harness, no Darwin Mode.** Rejected: loses the
   systematic, logged, gated search over the seven surfaces and makes
   improvement ad hoc and hard to audit.
3. **Soft guidelines instead of hard forbidden-mutation gates.**
   Rejected: in a discovery setting, the cost of a single fabricated
   observation or invented citation is catastrophic to credibility, so
   these must be hard structural rejections.

## References

- Bruland, A. & Hadziioannou, C. (2023). Gliding tremors associated with
  the 26 s microseism.
- Microseism theory: ocean-wave / seafloor coupling (primary and
  secondary/double-frequency microseisms).
- `.metaharness/genome.json`, `.metaharness/objective.json`,
  `.metaharness/safety-policy.json`
- Sibling: `harnesses/timesfm-harness` (same Darwin kernel pattern)
- ADR-002, ADR-003, ADR-004, ADR-005
