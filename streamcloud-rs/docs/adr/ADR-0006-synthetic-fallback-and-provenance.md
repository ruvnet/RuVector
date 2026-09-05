# ADR-0006: Synthetic-Fallback Strategy & Checkpoint Provenance

- **Status**: Accepted
- **Date**: 2026-06-27

## Context

The upstream LingBot-Map checkpoint is ~4.63 GB. It cannot be downloaded or run
to completion in the CI/sandbox environment, and full-fidelity reproduction of a
SOTA streaming 3D model is a multi-week research-engineering effort. Yet the
deliverable requires a *runnable, testable* end-to-end system: streaming
inference → 3D point cloud → PNG/MP4 → native + WebGPU demos.

## Decision

Provide two interchangeable backends behind one `streamcloud-model::Reconstructor`
interface:

1. **`SyntheticReconstructor`** (default, pure-Rust, `wasm32`-safe) — derives a
   plausible depth field from monocular cues (shading + ground-plane vertical
   gradient), projects through a pinhole camera to a colored point cloud, and
   folds retrieved anchor context in as a bounded drift-correction term. It
   exercises the *entire* system (memory retrieval, projection, rendering,
   encoding) deterministically and fast.
2. **`transformer::GeometricContextTransformer`** (feature `candle`) — the real
   architecture that loads the upstream safetensors weights.

The synthetic path is clearly labeled — in code, docs, and demo UI — as **not
the trained model**. It is a harness, not a reconstruction claim.

### Checkpoint provenance

- Weights are **borrowed at runtime** (local path or HF download), never
  committed (`.gitignore` excludes `*.safetensors`).
- The upstream model is Apache-2.0; this port is an independent reimplementation
  and carries the same license. Model weights remain the original authors'
  property under their license.
- `streamcloud-tensor::WeightIndex` validates a checkpoint's tensor inventory
  (names/shapes/dtype) without loading the payload, so provenance/compatibility
  can be checked on constrained machines.

## Consequences

- Demos and tests are hermetic and deterministic without network or GPU.
- Visual output from the synthetic path is illustrative geometry, not a faithful
  3D reconstruction; this is stated wherever output is presented.
- When real weights are supplied and the candle backend is validated against
  them, the same pipeline/demo code produces real reconstructions with no
  structural change — only the backend swap.
