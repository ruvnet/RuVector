# ADR-005: Darwin Mutation Surfaces and Safety Policy

## Status

Accepted

## Context

ADR-001 establishes the principle: freeze the physics and the model,
evolve the harness. This ADR makes that operational by specifying
**exactly what Darwin Mode may mutate** and **what safety gate every
mutation must pass** before it can run, let alone be promoted. Darwin
Mode is invoked through the `evolve` skill (the same skill present in the
sibling `harnesses/timesfm-harness`, which uses an identical
deterministic-mutator-by-default setup under `@metaharness/kernel`).

Letting an automated process rewrite parts of a scientific pipeline is
inherently risky: a mutation could quietly add a network call that
phones home with data, read a file outside the sandbox, shell out, pull
in an unpinned dependency, or — worst of all in this domain — fabricate
observations or citations. The safety policy exists to make all of those
structurally impossible, not merely discouraged.

The mutable surfaces are declared in `.metaharness/genome.json`; the
safety gate is declared in `.metaharness/safety-policy.json`.

## Decision

### What Darwin Mode is allowed to mutate

Darwin Mode may propose mutations to these seven surfaces (the concrete
expansion of the ADR-001 evolvable surfaces), and nothing else:

1. **Spectral windows** — FFT/STFT window length, overlap, taper, and
   the analyzed period band (the ~25-28 s region of interest) in
   `src/detect-26s.ts`.
2. **Beamforming params** — array steering, azimuth resolution, and
   coherence thresholds used for source localization in
   `src/detect-26s.ts`.
3. **Feature schema** — which spectral/envelope/glide features are
   computed and how they are normalized in `src/extract-features.ts`.
4. **ruVector retrieval strategy** — how the separated sub-embeddings
   (ADR-002) are queried, weighted, and combined for nearest-neighbor
   and contrastive retrieval in `src/embed-events.ts`.
5. **Scoring weights** — the discovery-score weights and evidence
   aggregation in `src/score-hypotheses.ts` (defaults in
   `.metaharness/objective.json`, ADR-003).
6. **Holdout strategy** — how storm/calm partitions are constructed in
   `src/validate.ts`, subject to the invariant that test and train
   windows stay disjoint (ADR-004).
7. **Contradiction detector** — the statistical tests that surface and
   log contradictions in `src/validate.ts` (ADR-004).

Anything outside this list is immutable: the frozen model, the observed
seismic record, the literature corpus, the FORBIDDEN-mutation rules
(ADR-001), and the structural leakage/citation checks themselves.

### Safety policy

The deterministic mutator is the **default**, and the policy is
air-gapped by design:

- **Air-gapped / no key by default.** The default mutator is
  deterministic and requires no API key and no network. Evolution can
  run fully offline; an LLM-backed mutator is opt-in and still subject
  to every gate below.
- **`validateGeneratedCode` gate on every mutation.** Before a mutated
  child is executed, its generated code must pass a static gate. A
  mutation is **rejected** if it introduces any of:
  - **new imports** (no new modules beyond the existing allowlist),
  - **network access** (no `fetch`, sockets, HTTP clients),
  - **filesystem access** beyond the declared sandbox paths,
  - **shell / process execution** (no `exec`, `spawn`, child processes),
  - **environment access** (no reading `process.env` / secrets),
  - **new dependencies** (no changes to `package.json` / lockfile).
- **Sandboxed execution.** Every child runs in a sandbox with read-only
  access to the dataset and write access only to its own scratch and
  report outputs.
- **Promote only on measured improvement.** A child that passes the
  static gate still does nothing until it clears the ADR-003 promotion
  gate (out-of-sample F1 / FPR / held-out error, citation mapping, no
  leakage). Passing safety is necessary, not sufficient.

These rules are encoded in `.metaharness/safety-policy.json` and
enforced by the `evolve` skill's kernel, so they cannot be bypassed by a
mutation editing its own policy (the policy file is outside the mutable
surface set).

### Relationship to the sibling harness

This mirrors `harnesses/timesfm-harness`: same `@metaharness/kernel`,
same `evolve` skill, same deterministic-default + `validateGeneratedCode`
+ sandbox + promote-on-measured-improvement discipline. The only
difference is the domain — timesfm evolves a coding/forecasting harness
around a frozen TimesFM model, while this harness evolves a
scientific-investigation workflow around a frozen embedding model. Reusing
the proven Darwin setup keeps operational and security behavior
identical across the harness family.

## Consequences

### Positive

- A precise, declared mutation surface means evolution can never wander
  into the frozen substrate or the integrity checks.
- The static `validateGeneratedCode` gate makes the dangerous classes
  (network, fs, shell, env, deps, new imports) structurally
  unreachable, independent of what a mutator "intended".
- Air-gapped default means evolution is reproducible and leaks nothing.
- Consistency with `timesfm-harness` reduces cognitive and security
  surface across the family.

### Negative

- The static gate will sometimes reject a legitimate mutation that
  happens to need a new utility import; such cases require a deliberate,
  human-reviewed allowlist change rather than an automatic one.
- The deterministic mutator explores a narrower space than an
  LLM-backed one; richer search requires opting into the gated LLM path.
- Sandbox + double gate (safety then promotion) adds latency per
  generation.

## Alternatives considered

1. **Open-ended code mutation, review after the fact.** Rejected:
   review-after-execution is too late once a mutation can touch the
   network, filesystem, or the observed record.
2. **LLM mutator by default.** Rejected as default: introduces a key
   requirement, network dependence, and non-determinism; kept as a
   gated opt-in instead.
3. **Trust the mutator, skip the static gate.** Rejected: the entire
   point is that no mutation, however generated, can introduce the
   forbidden capabilities.

## References

- `.metaharness/safety-policy.json`, `.metaharness/genome.json`,
  `.metaharness/objective.json`
- `src/detect-26s.ts`, `src/extract-features.ts`, `src/embed-events.ts`,
  `src/score-hypotheses.ts`, `src/validate.ts`
- `evolve` skill (`@metaharness/kernel`)
- Sibling: `harnesses/timesfm-harness` (identical Darwin/safety setup)
- ADR-001 (frozen substrate + forbidden mutations), ADR-003 (promotion
  gate), ADR-004 (holdout/leakage invariants)
