---
name: evolve
description: "Evolve this research harness with Darwin Mode — frozen model and frozen physics, evolving workflow (real, sandboxed, safety-gated)."
---

# evolve — Darwin Mode self-improvement

`earth-pulse-harness` ships with **Darwin Mode** (`@metaharness/darwin`): the model is
frozen AND the physics is frozen — only the *investigation workflow* evolves. Each
generation mutates ONE of the 6 surface files (detector, feature extractor, embedding
schema, retrieval, hypothesis scorer, validator), sandboxes each child, scores it
against the objective, and keeps only variants that *measurably* improve — building an
archive of successful descendants.

## Run it

```bash
npm run evolve        # real substrate: runs the test suite per variant (deterministic mutator — no API key, no network)
npm run evolve:dry    # mock substrate: fast, fully offline, no test execution
```

Or directly (flags pass through to metaharness-darwin):

```bash
npx metaharness-darwin evolve . --sandbox real --generations 20 --children 8 --concurrency 4 --seed 26
npx metaharness-darwin evolve . --generations 3 --children 4 --dry-run   # dry run first
```

## What Darwin may mutate (and may NOT)

Evolvable surfaces and forbidden mutations are declared in
`.metaharness/safety-policy.json`. In short:

| Surface | Example mutation | Why it matters |
|---|---|---|
| Feature extractor | Shift spectral windows 24→28s | Tests frequency stability |
| Context builder | Add tide phase, swell direction, pressure | Tests ocean coupling |
| ruVector schema | Separate waveform / environment / paper embeddings | Reduces mixed-signal noise |
| Retriever | Top-k by source region + spectral similarity | Finds causal neighbors |
| Hypothesis scorer | Adjust amplitude / seasonality / source-fit weights | Improves explanation ranking |
| Validator | Hold out storm weeks or calm-sea weeks | Prevents fake correlation |

**Forbidden:** fabricating observations, inventing citations, leaking test windows into
training, promoting a hypothesis without beating its baseline, or adding any new
import / network / filesystem / shell / env access.

## Promotion gate

A child is promoted ONLY if (see `.metaharness/objective.json`, `src/validate.ts`):

```
pulse_detection_f1 improves by >= 3%
AND false_positive_rate does not increase
AND held_out_prediction_error improves by >= 5%
AND every cited claim maps to a source document
AND no leakage from test windows into training windows
```

## Safety (secure by default)

- **Deterministic mutator** is the default — **no network, no API key, air-gapped**.
- Every mutation passes the `validateGeneratedCode` gate (no new imports/network/fs/shell/env/deps).
- Mutations run in a **sandbox**; only variants that pass the tests AND the promotion gate are archived.
- Nothing is promoted without measured improvement (guard against Goodharting the score).

The optional real-LLM mutator (`OpenRouterMutator`) is library-only; wire it via
`scripts/evolve-openrouter.{sh,mjs}` (key sourced from a secret manager at runtime,
never stored in the repo).
