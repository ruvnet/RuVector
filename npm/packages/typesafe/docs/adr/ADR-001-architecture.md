# ADR-001: @ruvector/typesafe — architecture

## Status
Proposed

## Date
2026-09-21

## Context

Jev (typesafe.ai) is a hosted "System One" typed-decision API: a text `state` plus
a batch of questions — `choice` (pick one of up to 255 options, each with a
description or `{what, not_for, examples}`), `score` (ordinal legend buckets), and
`noul` (a 0–1 predicate) — returns per-question `{choice, probabilities,
confidence}` / `{score, legend, probabilities}` / `{noul}` and token `usage`.

Measured against it on 2026-09-21 (500 synthetic support tickets, 8 departments,
frozen 150-item test split): p50 178 ms and p95 216 ms over the network, ~22
items/s at concurrency 4, $42 per 1B input tokens, department accuracy 85.3 →
90.0 % after criteria optimization, `confidence` saturated near 1.0 (ECE 0.07),
and `noul` urgency detection at 60 % — *below* the 70 % a constant "not urgent"
would score.

We want the same contract, locally: no per-token cost, no network dependency,
sub-10 ms decisions, confidence that means something, and a loop that improves
itself under measurement rather than by fiat. ruvector already ships most of the
substrate (a napi-rs + wasm-bindgen HNSW router, ONNX embedders, quantization,
lineage storage), split across several unmerged tracks (see ADR-002).

## Decision

`@ruvector/typesafe` is an npm package with a Rust core exposed two ways:

```
                 ┌──────────────────────────────────────────────┐
  TypeScript API │ typesafe(): decide({state, questions})         │  Jev-compatible shape,
  (index.ts)     │ choice<K> / score<L> / noul  ·  batch  ·  CLI  │  criteria-keyed generics
                 └───────────────┬──────────────────────┬────────┘
                                 │ native (napi-rs)     │ wasm (wasm-bindgen)
                 ┌───────────────▼──────────┐  ┌────────▼──────────────────┐
  Rust core      │ typesafe-core             │  │ same crate, wasm32 target │
                 │  embed  → ruvector-embed  │  │  embed → tract (WASM)     │
                 │  index  → router-core     │  │  index → router-wasm      │
                 │  heads  → centroid/probe  │  │  heads → identical Rust   │
                 │  calib  → temperature     │  │                           │
                 └──────────────────────────┘  └───────────────────────────┘
```

1. **One decision engine, two targets.** `typesafe-core` (Rust) holds everything
   that decides: option encoding, the decision heads, calibration, and the
   example bank. It is compiled to a napi-rs `cdylib` (five prebuilt platform
   packages under `optionalDependencies`, exactly as `@ruvector/router` ships) and
   to `wasm32-unknown-unknown` via wasm-bindgen for browsers and for Node when
   no native binary matches. Both targets run the *same* head and calibration
   code; only the embedding runtime differs (ADR-002).

2. **Embeddings are a pluggable stage, never the decision.** A `choice` is
   decided by a head over embeddings (ADR-003), not by raw cosine similarity —
   the evidence is that zero-shot cosine tops out around 75–87 % on the standard
   few-shot intent benchmarks, below Jev's measured 85–90 %.

3. **Jev's wire shape is the public contract.** `decide()` accepts and returns
   Jev's request/response JSON so existing Jev callers can point at this package
   unchanged; the TypeScript layer adds criteria-keyed generics on top
   (`ChoiceAnswer<keyof C>`), a CLI, and streaming/batching. `usage` reports
   local compute (embedding tokens, wall time) in the same field names.

4. **Self-optimization is a governed loop, not a background process** (ADR-004):
   every change to the example bank, the model arm, or the criteria text is a
   proposal that must beat a frozen validation split under a sequential test,
   with a permanent control arm, and is recorded as a receipt.

5. **No network by default** (ADR-005): models are bundled or loaded from a
   pinned, hash-verified local manifest; the WASM build links no filesystem or
   network capability.

6. **Every claim in this package is a benchmark first** (ADR-006): accuracy,
   calibration, latency, and cost against Jev on the same items, with release
   gates, live in the repo and run in CI.

### Package layout

```
npm/packages/typesafe/
  package.json           @ruvector/typesafe, optionalDependencies → 5 platform packages
  src/index.ts           public API, generics, Jev-shape adapters, CLI entry
  src/cli.ts             typesafe ask | train | eval | bench | serve
  wasm/                  wasm-bindgen output (bundler + nodejs targets)
  docs/adr/              this record set
  bench/                 datasets, harness, receipts (ADR-006)
crates/typesafe-core/    Rust: engine, heads, calibration, bank, receipts
crates/typesafe-ffi/     napi-rs bindings (mirrors ruvector-router-ffi)
crates/typesafe-wasm/    wasm-bindgen bindings (mirrors ruvector-router-wasm)
```

## Consequences

- Drop-in for Jev callers, with zero per-decision cost and no egress. The cost
  moves to CPU time and a one-time model download or bundle (~25–35 MB INT8).
- Two inference runtimes (native `ort`, WASM `tract`) must stay numerically
  consistent; ADR-002 makes cosine-equivalence between them a CI gate, as
  ADR-194 already does for the worker pool.
- The head/calibration layer needs *some* labeled examples per deployment to
  reach Jev-level accuracy and to calibrate. A zero-labeled deployment falls back
  to centroid + hard-negative margin and must report that its confidence is
  uncalibrated (ADR-003) rather than pretend.
- Reusing `ruvector-router-core` means inheriting its single-layer, single-entry
  HNSW. That is adequate for example banks in the thousands-to-low-millions; a
  multi-layer graph is a v2 concern only if a real bank outgrows it.

## Alternatives considered

- **Call an LLM locally (llama.cpp / ruvllm) with structured output.** Matches
  Jev's flexibility on arbitrary questions, but 100× the latency and cost of an
  embedding head, and its confidence is no better calibrated. Rejected for v1;
  the `agent` sandbox in metaharness can judge, not decide.
- **Pure vector search (`@ruvector/router` SemanticRouter as-is).** Zero
  training, but the evidence says it cannot reach Jev's accuracy, and its
  `threshold` is a similarity cut-off, not a probability. Kept as the
  zero-labeled fallback inside the engine.
- **TypeScript-only implementation on transformers.js.** Fastest to write and
  a fine browser fallback, but it forks the embedding runtime away from the
  Rust stack whose cosine-equivalence discipline the rest of ruvector depends on.

## Evidence

- Jev measurements: `bench/jev-baseline-2026-09-21.json` (from the scorecard
  run; 2,850 requests, 0 errors).
- Few-shot intent baselines and head ranking: SetFit (arXiv:2209.11055),
  probing study cited in ADR-003.
- Reuse map and runtime evidence: ADR-002; ruvector ADR-194.
- Loop governance evidence: ruvector ADR-276, ADR-271, ADR-282, ADR-288.
