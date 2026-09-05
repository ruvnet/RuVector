# ADR-0003: candle Tensor Backend & safetensors Weight Loading

- **Status**: Accepted
- **Date**: 2026-06-27

## Context

The port needs a Rust tensor/NN framework to express the Geometric Context
Transformer, plus a way to load the upstream checkpoint (~4.63 GB safetensors).

## Decision

Use **candle 0.9** (`candle-core`, `candle-nn`) — already a vendored dependency
elsewhere in the monorepo (`ruvllm`, `timesfm`), so versions are consistent and
the crates are cached.

Weight handling is split in two:

1. **Header-only indexing** (`streamcloud-tensor::WeightIndex`) — parses the
   safetensors header to map tensor name → shape/dtype **without** reading the
   multi-GB payload. This validates a checkpoint and lets us plan loading on
   machines that cannot hold the weights in RAM.
2. **Materialization** (`streamcloud-tensor::load`, behind the `candle` feature) —
   loads named tensors into `candle_core::Tensor` on a chosen `Device`.

`ModelConfig` is `serde`-loaded from the upstream `config.json` with sane
defaults (1024 hidden, 24 layers, 16 heads, 518×378 input, patch 14).

## Rationale

- Pure-Rust (no libtorch) keeps the port portable and WASM-friendly.
- Header-only indexing decouples "do we have a valid checkpoint" from "can we
  fit it in memory" — essential in constrained/sandbox environments.
- Feature-gating candle keeps `streamcloud-tensor`'s default build fast for CI and
  for crates (like `streamcloud-memory`) that don't need tensors.

## Consequences

- candle 0.9 op coverage bounds what we can express; any missing op is
  implemented manually on `Tensor` data.
- bf16/f16 checkpoints are supported via `half`; dtype is surfaced by
  `TensorInfo.dtype`.
- The real checkpoint is *borrowed* from the upstream project at runtime (path
  or HF download), never committed. See ADR-0006 for the synthetic fallback used
  when weights are absent.
