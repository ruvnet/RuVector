---
adr: 195
title: "RuVector ONNX Embedder Unification — Plan (issue #523 deeper fix)"
status: proposed
date: 2026-05-30
authors: [ruvnet, claude-flow]
related: [ADR-194]
tags: [ruvector, onnx, embeddings, refactor, plan, issue-523]
---

# ADR-195 — ONNX Embedder Unification Plan

## Status

**Proposed (plan only, not executed).** Follow-up to ADR-194. Awaiting approval
before any code change. Target branch: `fix/onnx-embedder-api-523` (or a fresh
branch off it).

## Context

The `ruvector` package currently has **two independent ONNX embedder
implementations** that do not share state — the root cause of issue #523's bugs
#2 and #3:

| Implementation | File | Shape | Model load |
|---|---|---|---|
| Module singleton | `onnx-embedder.ts` | free functions (`initOnnxEmbedder`, `embed`, `embedBatch`, `isReady`) + `OnnxEmbedder` class wrapper | loads FP32 via `ModelLoader` |
| Class | `onnx-optimized.ts` | `OptimizedOnnxEmbedder` + `getOptimizedOnnxEmbedder()` singleton | loads its own model via a **separate** `doInit()` |
| Adapter | `adaptive-embedder.ts` | `AdaptiveEmbedder` wraps the module singleton (`embed`/`embedBatch` from `onnx-embedder`) | delegates to module singleton |

Because the module path and `OptimizedOnnxEmbedder` each construct their own
`WasmEmbedder` from their own model load, the issue's probe saw: `initOnnxEmbedder()`
ready but `getOptimizedOnnxEmbedder().isReady()` false (#2), and two model loads (#3).
ADR-194 patched the symptoms (readiness gates, in-memory + on-disk model memo so the
second load is cheap, removed the misleading FP16 log). This ADR proposes removing the
**duplication itself** so there is one model, one WASM embedder, one readiness state.

## Decision (proposed)

Make the `onnx-embedder.ts` module singleton the **single source of truth** for model
loading + the underlying `WasmEmbedder`, and reduce `OptimizedOnnxEmbedder` to a thin
**caching/ergonomics layer** over it rather than a parallel implementation.

Rationale for that direction (not the reverse):
- The module singleton already owns the canonical load path, the worker pool, the
  model memo, and the `isReady`/`isOnnxInitialized` contract from ADR-194.
- `AdaptiveEmbedder` already delegates to it.
- `OptimizedOnnxEmbedder`'s genuine value-add is its LRU embedding cache + tokenizer
  cache + `Float32Array` ergonomics — orthogonal to model ownership, easy to keep as a
  wrapper.

## Plan

### Phase 0 — safety net (no behavior change)
1. Confirm the green baseline: `npm run build`, `npm test` (69+2), `node --test tests/`
   (8), `scripts/bench/pool-test.mjs` (cosine = 1.0). Already green as of ADR-194.
2. Add a characterization test asserting `getOptimizedOnnxEmbedder().embed(x)` is
   cosine-equivalent to the module `embed(x)` — locks behavior before refactor.

### Phase 1 — share the loaded model
3. Export an internal accessor from `onnx-embedder.ts` exposing the loaded
   `WasmEmbedder` (or its `embedOne`/`embedBatch`/`dimension`) plus model bytes/tokenizer
   (already retained as `loadedModelBytes`/`loadedTokenizerJson` from ADR-194).
4. Change `OptimizedOnnxEmbedder.doInit()` to call `initOnnxEmbedder()` and reuse that
   embedder instead of constructing its own `WasmEmbedder`. Its `initialized`/`isReady()`
   then reflect the shared state → fixes #2 at the root, eliminates the second load (#3)
   structurally rather than via memo.
5. Keep the LRU embedding/tokenizer caches and `Float32Array` return shape unchanged.

### Phase 2 — collapse config surface
6. Reconcile the two config objects (`OnnxEmbedderConfig` vs `OptimizedOnnxConfig`):
   `maxLength`, `normalize`, `modelId` move to the shared loader; `cacheSize`,
   `tokenizerCacheSize`, `batchSize`, `batchThreshold` stay on the wrapper.
7. Remove the now-dead `QUANTIZED_MODELS` URL table (or relocate it behind a documented
   "not yet wired — see ADR-194 quant track" comment), since the loader resolves by
   `modelId` and quantization is backend-blocked.

### Phase 3 — converge the parallel story
8. `OptimizedOnnxEmbedder.embedBatch()` and the module `embedBatch()` both route large
   batches through the single bundled worker pool (ADR-194), so there is one parallel
   path, not two.

### Phase 4 — verify + document
9. Re-run the full gate (build, 69+2, 8 CI tests, pool cosine = 1.0, the Phase-0
   characterization test). Update ADR-194's "two classes coexist" consequence.

## Call-site impact

Public exports that MUST keep working (barrel re-exports from `src/core/index.ts`):
`isOnnxAvailable`, `isOnnxInitialized`, `isReady`, `initOnnxEmbedder`, `embed`,
`embedBatch`, `similarity`, `cosineSimilarity`, `getStats`, `shutdown`, `OnnxEmbedder`,
`OptimizedOnnxEmbedder`, `getOptimizedOnnxEmbedder`, `initOptimizedOnnx`,
`AdaptiveEmbedder`, `getAdaptiveEmbedder`, `initAdaptiveEmbedder`, plus the ADR-194
pool functions. The refactor is **internal**; no public signature changes — only the
guarantee that `OptimizedOnnxEmbedder` shares the module's model. Internal importers to
re-check: `diff-embeddings.ts`, `intelligence-engine.ts`, `adaptive-embedder.ts`,
`workers/benchmark.ts`, `workers/native-worker.ts`, `bin/cli.js`.

## Risk / Effort

- **Effort:** ~M (touches 2–3 files + 1 new test; no API change).
- **Risk:** Medium — `OptimizedOnnxEmbedder` consumers currently get an independently
  configured embedder (e.g. different `maxLength`); sharing the module embedder means the
  first initializer wins. Mitigation: if configs differ, log once and document
  "first-init-wins," or keep a separate instance only when an explicitly different
  `modelId`/`maxLength` is passed.
- **Backout:** revert the branch; the ADR-194 memo already makes the un-unified path
  correct (just slightly wasteful), so unification is an optimization+clarity win, not a
  correctness prerequisite.

## Decision needed

Approve this plan to execute Phases 0–4 on the branch, or adjust scope (e.g. Phase 1
only — share the model and fix #2/#3 structurally — and defer config/parallel
convergence).

## References

- ADR-194 (symptom fixes + worker pool + quant-blocked finding)
- Issue: https://github.com/ruvnet/RuVector/issues/523
- Files: `src/core/onnx-embedder.ts`, `src/core/onnx-optimized.ts`,
  `src/core/adaptive-embedder.ts`
