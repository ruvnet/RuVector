# ADR-002: Inference backend — `ort` native, `tract` WASM, one engine crate

## Status
Proposed

## Date
2026-09-21

## Context

The decision engine needs sentence embeddings (a small BERT-class encoder) on
CPU, in two places: a native Node binary and a WASM build. ruvector already
contains three partial answers that were never joined:

| Component | Where | State |
|---|---|---|
| HNSW router, napi-rs + wasm-bindgen | `crates/ruvector-router-{core,ffi,wasm}`, `npm/packages/router` | Production-shaped. Published 0.1.30 lacked the #430 fix; PR #1005 ships 0.1.31 with the fix plus the neighbour-selection heuristic (recall@10 1.000 on 5,000 real embeddings vs 0.036 published). |
| Native ONNX embedder (`ort` 2.0.0-rc.9 + `tokenizers` 0.20) | `examples/onnx-embeddings` | Workspace-excluded; not built or tested by CI. |
| WASM ONNX embedder (`tract-onnx`) | `examples/onnx-embeddings-wasm` on 0.23; the *shipped* `npm/packages/ruvector` embedder pins 0.21 | On 0.21, INT8/FP16 models fail to load (`/Unsqueeze AddDims`, ADR-194 §5); FP32 works at ~192 ms/embed single-thread, 72.8 embeds/s across a 30-worker pool with cosine-exact parity. |
| Vector quantization (RaBitQ) | `crates/ruvector-rabitq{,-wasm}` | Reusable as-is for the index; unrelated to model-weight quantization. |

## Decision

1. **`typesafe-core` is one Rust crate compiled for both targets.** Everything
   downstream of the embedding — option encoding, heads, calibration, example
   bank, receipts — is target-independent Rust with no `cfg` forks. The
   embedding stage is a trait:

   ```rust
   pub trait Embedder { fn embed(&self, texts: &[&str]) -> Result<Array2<f32>>; fn dims(&self) -> usize; fn id(&self) -> &str; }
   ```

   with two implementations: `OrtEmbedder` (native, feature `native`) and
   `TractEmbedder` (feature `wasm`). The trait's `id()` is the model manifest
   hash, so a receipt records exactly which weights decided.

2. **Promote `examples/onnx-embeddings` into the workspace as `ruvector-embed-core`**
   and make it the native implementation. It leaves `[workspace]` exclusion,
   gains CI, and is pinned to a released `ort` (2.0.0-rc.12 or the first stable).

3. **Bump the WASM path to `tract-onnx` 0.23** and re-run ADR-194's exact
   INT8 repro. This is a spike with a pass/fail outcome: if INT8 loads, WASM
   ships INT8 (3.2× CPU speedup at <1 % MTEB loss is the published expectation);
   if not, WASM ships FP32 and INT8 is native-only, stated in the README.

4. **Bindings mirror the router crates exactly.** `typesafe-ffi` is a napi-rs
   `cdylib` (`lto = true`, `codegen-units = 1`, `strip = true`), using
   `AsyncTask` for decisions (no callbacks into JS are needed) and returning
   probabilities as zero-copy `Float32Array`. `typesafe-wasm` uses wasm-bindgen
   with `opt-level = "z"`, `panic = "abort"`, and is packaged for both
   `bundler` and `nodejs` targets.

5. **Packaging copies `@ruvector/router`:** five platform packages
   (`linux-x64-gnu`, `linux-arm64-gnu`, `darwin-x64`, `darwin-arm64`,
   `win32-x64-msvc`) as `optionalDependencies`, built by a `build-typesafe.yml`
   cloned from `build-router.yml`, guarded by the `optional-deps-resolvable-on-npm`
   job (issue #411) copied verbatim. Version bumps land only *after* the
   platform packages exist on npm — the sequence the router's own 0.1.31 revert
   documented.

6. **Default model: `bge-small-en-v1.5`** (384-d, MIT, ONNX, strong MTEB
   classification among small models), with `all-MiniLM-L6-v2` as the smallest
   fallback and `nomic-embed-text-v1.5` (Matryoshka-truncatable) selectable for
   long `state` text. The model arm is chosen per deployment by the loop in
   ADR-004, never hardcoded in the head.

7. **Numerical parity is a CI gate.** For a fixed probe set, native and WASM
   embeddings must agree to cosine ≥ 0.9999 and decisions must be identical.
   This is the discipline ADR-194 already enforces between the single-thread and
   pooled paths; a backend that cannot pass it does not ship.

## Consequences

- Two runtimes to keep in step, mitigated by the parity gate and by keeping
  every non-embedding line of code shared.
- The native binary carries `libonnxruntime` (tens of MB per platform package);
  the WASM build carries the model bytes. Install size is a stated trade-off,
  not a surprise.
- `ort` is not wasm32-capable, so `candle` (which runs both targets from one
  crate with Bert support) is the v2 consolidation path if maintaining two
  engines proves costly; the `Embedder` trait is the seam that makes that swap
  local.

## Alternatives considered

- **`candle` for both targets now.** Structurally cleanest, but a rewrite of a
  path that already works natively, and its WASM SIMD performance for BERT-class
  models is less proven than tract's in this repo. Deferred to v2.
- **transformers.js (onnxruntime-web) as the embedder everywhere.** 8–12 ms warm
  on M2 and very convenient, but a JS runtime with its own numerics; it would
  break the cosine-equivalence gate that ties this package to the rest of
  ruvector. Allowed only as a last-resort browser fallback behind the same
  parity test.
- **Full `ruvector-node` VectorDb instead of router-core.** Heavier surface
  (collections, filters, metrics) than a decision engine needs; the async NAPI
  bridge plus full-vector payloads measured ~6 ms p95 vs 0.3–0.6 ms for the sync
  router path on the same data.

## Evidence

- Router recall/latency measurements: PR #1005 description and
  `bench/ruvector-router-2026-09-21.json`.
- ADR-194 (ruvector): worker-pool throughput, cosine-exact parity, INT8 failure
  on tract 0.21.
- INT8 ONNX speedup 3.2× at <1 % MTEB loss: ONNX Runtime quantization docs;
  Nixiesearch/Zilliz small-model benchmarks (2024–2025).
- ort/tract/candle target support: crate docs (`ort` 2.0 rc, `tract` 0.23,
  `candle` wasm examples).
