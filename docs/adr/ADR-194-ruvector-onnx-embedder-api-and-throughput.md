---
adr: 194
title: "RuVector Bundled ONNX Embedder — API Contract Hardening & Cosine-Safe Throughput"
status: accepted
date: 2026-05-30
authors: [ruvnet, claude-flow]
related: [ADR-193]
tags: [ruvector, onnx, embeddings, npm, wasm, performance, worker-threads, tract, issue-523]
---

# ADR-194 — RuVector Bundled ONNX Embedder: API Contract & Throughput

## Status

**Accepted.** Implemented on branch `fix/onnx-embedder-api-523`
(npm package `ruvector`, `npm/packages/ruvector/`). Not yet merged or published.

## Context

Issue [#523](https://github.com/ruvnet/RuVector/issues/523) reported four API-contract
defects in the `ruvector` npm package's bundled ONNX embedder, found while building a
BEIR benchmark harness for ruflo. The defects made it impossible to write a confident
"should I use this embedder?" probe, so downstream callers silently fell back to hash
embeddings for an entire benchmark run without noticing.

Reported defects:

1. `isOnnxAvailable()` returned `true` before init — a capability check being (mis)used as
   a readiness gate by callers.
2. `getOptimizedOnnxEmbedder().isReady()` stayed `false` even after a successful `embed()`.
3. A double model load, and a log line claiming "Using FP16 quantized model" that was never
   actually applied (the loader ignored the computed quantized URL).
4. `AdaptiveEmbedder.isReady()` returned `undefined`, violating its typed contract.

Investigation surfaced a fifth, latent packaging defect: the `build` script copied only
`src/core/onnx/pkg/` into `dist/`, dropping `dist/core/onnx/loader.js`. A clean build
shipped an embedder that throws at runtime — the same silent-fallback failure mode the
issue describes.

Two deeper questions were raised: (#8) break the single-thread throughput ceiling via
INT8/FP16 quantization, and (#9) unify the two parallel embedder implementations
(`onnx-embedder` module singleton vs `OptimizedOnnxEmbedder` class).

Measured baseline (32-core workstation, Node 22, all-MiniLM-L6-v2 FP32): single embed
p50 ≈ 192 ms; batch throughput ≈ 5.2 embeds/sec — `embedBatch` gave no speedup because the
bundled WASM runs items sequentially.

## Decision

**1 — Fix the API contract, preserving the pre-init use case.** Keep `isOnnxAvailable()` as
the capability check ("files bundled, can init"); add a distinct `isOnnxInitialized()`
post-init gate (named to avoid colliding with the WASM-core `isInitialized` on the package
barrel). Add `AdaptiveEmbedder.isReady()` returning a real boolean. Remove the misleading
FP16 log and the dead `modelUrl` computation in `onnx-optimized.ts`.

**2 — In-memory model memo** in `ModelLoader.loadModel()` (Node has no Cache API), keyed by
model name. Probe downloads dropped 4 → 2.

**3 — Fix packaging:** build copies the entire `src/core/onnx/` directory; `verify-dist`
passes on a clean build.

**4 — Cosine-safe parallelism instead of per-call latency.** A self-contained
`worker_threads` pool over the bundled WASM (no external dependency) shards batches across
cores, sharing the loaded model bytes via `SharedArrayBuffer` (no per-worker download).
Output vectors are bit-identical to the single-thread path. Exposed as
`initParallelEmbedder()`, `embedBatchParallel()`, `getParallelWorkerCount()`,
`shutdownParallelEmbedder()`.

**5 — Quantization (#8) is backend-BLOCKED.** A feasibility experiment fed FP16 and INT8
model bytes to the bundled WASM. Both failed:
`Failed to optimize: Failed analyse for node "/Unsqueeze" AddDims`. The crate
(`examples/onnx-embeddings-wasm`) pins `tract-onnx 0.21`, which cannot optimize these
quantized graphs and has no quantization handling. Quantization requires upstream Rust work
(tract upgrade with uncertain operator coverage, or a backend swap) — recorded as a
follow-up, not attempted in JS.

**6 — Embedder-class unification (#9) is deferred** pending the quantization decision, since
the backend outcome could change which class should be canonical.

## Architecture

| Component | Role |
|-----------|------|
| `src/core/onnx-embedder.ts` | Module singleton; adds `isOnnxInitialized()`, retains loaded model bytes/tokenizer, owns worker-pool lifecycle |
| `src/core/onnx/bundled-parallel.mjs` | `ParallelEmbedder` — worker pool, batch sharding, `SharedArrayBuffer` model distribution, promise routing |
| `src/core/onnx/embed-worker.mjs` | Worker entry; builds a WASM embedder from shared bytes with identical config (mean pooling, normalize) → cosine-equivalence by construction |
| `src/core/onnx/loader.js` | In-memory model memo |
| `src/core/onnx-optimized.ts` | Removed misleading quantization log / dead URL |
| `src/core/adaptive-embedder.ts` | Adds `isReady(): boolean` |
| `scripts/bench/` | `onnx-bench.mjs`, `pool-test.mjs`, recorded `onnx-bench-results.json` |
| `tests/onnx-api-contract.test.mjs` | Six #523 regression tests |

Results: worker pool reaches **72.8 embeds/sec at 30 workers (14×)** with **min cosine =
1.000000** vs single-thread. Throughput is flat across batch sizes 120/240/480 — a
memory-bandwidth ceiling, not a core or overhead limit. The 80 eps target was not met; the
gap requires quantization, which is backend-blocked.

## Consequences

**Positive:** Honest three-state contract (available / initialized / ready), no silent hash
fallback. 14× batch throughput with provably zero quality drift, no new runtime dependency.
Packaging defect that would ship a broken embedder is fixed and gated by `verify-dist`. Full
suite green: 69 + 2 signal tests, plus 6 new contract tests.

**Negative / accepted:** Single-embed latency stays ~192 ms (WASM FP32 floor) — the approved
latency lever (quantization) is backend-blocked. Two embedder classes still coexist (#9
deferred). The worker pool needs hardening before production reliance.

**Follow-ups:** (a) worker-pool error/timeout handling (reject in-flight on worker crash;
per-request timeout); (b) promote the cosine-equivalence check into the CI suite; (c) on-disk
model cache under `~/.ruvector/models` to kill repeat downloads across processes; (d) route
the existing `enableParallel` option to the bundled pool and drop the phantom
`ruvector-onnx-embeddings-wasm/parallel` reference; (e) quantization track in the Rust crate
(tract upgrade / backend evaluation); (f) embedder-class unification; (g) remove stale tracked
`src/core/*.js` / `*.d.ts` artifacts.

## Alternatives Considered

- **Literal #523 fix (`isOnnxAvailable()` false until init):** rejected — breaks the
  "should I init?" use case; the gate would never be true before the call that flips it.
- **Quantize for latency now:** rejected — empirically blocked at `tract-onnx 0.21`
  (`AddDims` optimize failure on FP16 and INT8 graphs).
- **External parallel package (`ruvector-onnx-embeddings-wasm/parallel`):** rejected — not
  installed, not a declared dependency; a ~150-line zero-dep bundled pool replaces it.
- **Bigger batches for throughput:** tested — no gain past the memory-bandwidth ceiling.

## References

- Issue: https://github.com/ruvnet/RuVector/issues/523
- Crate: `examples/onnx-embeddings-wasm` (`tract-onnx 0.21`)
- Benchmarks: `npm/packages/ruvector/scripts/bench/onnx-bench-results.json`
- Branch: `fix/onnx-embedder-api-523`
