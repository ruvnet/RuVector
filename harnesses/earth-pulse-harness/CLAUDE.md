# earth-pulse-harness

Earth Pulse Observatory — a MetaHarness Darwin-Mode research pod for the 26-second microseism (Gulf of Guinea). **Freeze the physics; evolve the harness.**

> Causal-discovery research harness · domain: `geophysics/seismology`. Built in the metaharness bundle style of the sibling `harnesses/timesfm-harness`.

## Behavioral rules

- Use the harness's MCP tools (`mcp__earth-pulse-harness__*`) for orchestration.
- Memory and routing are handled by the kernel — you don't need to learn them.
- **Never fabricate observations or citations.** Real seismic/ocean/tide/bathymetry data goes under `data/`; every promoted claim must map to a document in `data/papers/`.
- Defer destructive operations to the user. `data/` is write-denied by default (see `.claude/settings.json`).

## Agents (research pod)

| Agent | Tier | Role |
|---|---|---|
| `investigator` | opus | Designs the falsifiable experiment before any analysis. |
| `feature-engineer` | sonnet | Builds the detector, features, and ruVector embeddings. |
| `hypothesis-scorer` | opus | Scores and ranks mechanisms against evidence. |
| `validator` | opus | Enforces the leakage / contradiction / promotion gate. |

## Skills

- `/plan-change` — Turn a research question into a minimal, falsifiable investigation plan.
- `/hypothesis-sweep` — Score and rank the candidate mechanisms with contradictions and a next test.
- `/evolve` — Darwin Mode self-improvement: frozen model + frozen physics, evolving workflow.

## Commands

- `doctor` — Health-check the harness (kernel, MCP, pipeline, host adapter).
- `review-diff` — Review the working diff for correctness, leakage, and safety-policy compliance.

## Pipeline

```
detect-26s.ts  ->  extract-features.ts  ->  embed-events.ts  ->  score-hypotheses.ts  ->  validate.ts
   (24-28s          (spectral / glide /        (separate            (weighted               (promotion
    spectral peak)   envelope / geometry)       waveform/env/        discovery score)        gate + leakage)
                                                 source vectors)
```

Run it over the bundled fixtures: `npm run build && npm run pipeline`.

## Darwin Mode

The model AND the physics are frozen. Darwin evolves only the workflow surfaces declared in
`.metaharness/safety-policy.json`, promoting a child only when it measurably beats the gate in
`.metaharness/objective.json`. Default mutator is deterministic and air-gapped. See `/evolve`.

## Architecture

This harness uses [@metaharness/kernel](https://www.npmjs.com/package/@metaharness/kernel) — a Rust-compiled WASM module with a NAPI-RS native fallback — so the same code runs identically on every platform. ruVector provides the planetary memory / embedding layer (see ADR-002).

## Docs

- `docs/adr/` — ADR-001…005 (freeze-physics, ruVector memory, scoring/gate, validation, Darwin safety).
- `docs/research/` — literature review, benchmark design, hypothesis catalog.
