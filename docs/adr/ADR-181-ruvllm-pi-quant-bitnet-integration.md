---
adr: 181
title: "Replace candle Q4_K_M with in-tree pi_quant + BitNet b1.58 ternary weights"
status: proposed
date: 2026-05-05
authors: [ruvnet, claude-flow]
related: [ADR-179, ADR-180, ADR-024, ADR-090]
branch: feature/ruvllm-pi-quant-native
---

# ADR-181 — In-tree pi_quant + BitNet b1.58 on Pi 5

## Status

**Proposed.** Stacks on ADR-180. ADR-180 raises throughput by
amortizing forward passes across in-flight requests; ADR-181 raises
per-token throughput by replacing the matmul kernel itself with the
hand-tuned 2-3 bit and 1.58-bit kernels already in
`crates/ruvllm/src/quantize/` and `crates/ruvllm/src/bitnet/`.

## Context

ADR-179 iter 27 found that **smaller GGUFs (Q3_K_S, Q2_K) ran
*slower* than Q4_K_M** — candle's Q4_K kernel is heavily tuned;
the Q3/Q2 codepaths fall to less-optimized dequant loops. The
memory-bandwidth saving is wiped out by the per-token compute
overhead.

But the in-tree quantization stack was never wired in. We have:

| Asset | Bits | Module | Status |
|---|---:|---|---|
| `pi_quant` (ADR-090) | 2-3 | `crates/ruvllm/src/quantize/pi_quant.rs` + `pi_quant_simd.rs` | Pi-targeted SIMD, never connected to a backend |
| `turbo_quant` | 2-3 (composable) | `crates/ruvllm/src/quantize/turbo_quant.rs` | Profiles tile-wise scales |
| `quip` | rotation precondition | `crates/ruvllm/src/quantize/quip.rs` + `hadamard.rs` | Improves quant error pre-codebook |
| `bitnet/` (ADR-024) | 1.58 (ternary) | `crates/ruvllm/src/bitnet/{quantizer,dequantize,ternary_tensor,backend}.rs` | Full pipeline + eval; not exposed via LlmBackend trait |
| `RaBitQ` (ADR-154) | 1 | `crates/ruvector-rabitq/` | KV-cache target (separate ADR) |

The pieces are there. Wiring them into the candle backend's
forward pass — or replacing candle entirely with a ruvllm-native
backend — is the remaining work.

## Decision

Two-phase implementation.

### Phase A — pi_quant in-tree as a candle override

For each `Linear` layer in candle's loaded Llama, intercept the
matmul with a pi_quant 3-bit dequant + GEMV. Same `LlmBackend`
trait, same model loading flow, same `ServingEngine` integration —
just a different inner matmul kernel.

Implementation surface:

```rust
// New backend in crates/ruvllm/src/backends/pi_quant_backend.rs
pub struct PiQuantBackend {
    weights: HashMap<String, PiQuantTensor>, // 3-bit packed + Hadamard
    candle_model: LlamaModel,                // structure only — weights
                                             // get rerouted to PiQuantTensor
    cache: KvCachePool,
}

impl LlmBackend for PiQuantBackend { ... }
```

Loaded by adding `Quantization::PiQuant3` variant. Cross-build
keeps the rustls-tls + Cortex-A76 settings from ADR-179 iter 2.

**Projected delta vs ADR-180 baseline:**
- per-Pi tok/s: 9.0 (Q4_K_M) → ~16-20 (pi_quant 3-bit)
- aggregate 4-Pi: 20.5 → ~50-70 with continuous batching, ~30-40 without

### Phase B — BitNet b1.58 ternary weight conversion

Convert TinyLlama-1.1B's safetensors → ternary weight blob via the
existing `crates/ruvllm/src/bitnet/quantizer.rs` + `gguf_export.rs`.
Output: `tinyllama-1.1b-bitnet158.qm` (~270 MB on disk).

`BitNetBackend` exists at `crates/ruvllm/src/bitnet/backend.rs` —
needs to implement the `LlmBackend` trait (currently it's a
standalone struct). Wire that in.

**Projected delta vs Phase A:**
- per-Pi tok/s: 16-20 → 25-40 (1.58-bit weights, 5× memory bw saved)
- aggregate 4-Pi: 30-40 → ~80-100 with continuous batching

Quality: BitNet b1.58 paper claims perplexity within 1% of fp16
when the model was *trained* in BitNet; we're applying it
post-hoc to TinyLlama, so perplexity will degrade more — measure
before committing.

## Acceptance

ADR-181 Phase A:

- [ ] `PiQuantBackend` impl of `LlmBackend` trait
- [ ] `Quantization::PiQuant3` and `PiQuant2` variants in
      `crates/ruvllm/src/backends/mod.rs`
- [ ] Worker loads pi_quant weights from a `.qm` blob
- [ ] **≥1.5× per-Pi tok/s** vs ADR-180 baseline
- [ ] Perplexity within 2% of fp16 reference (5 prompts)

ADR-181 Phase B:

- [ ] Conversion script: safetensors → BitNet 1.58 `.qm`
- [ ] `BitNetBackend` implements `LlmBackend`
- [ ] Worker loads BitNet weights
- [ ] **≥1.5× per-Pi tok/s** vs Phase A
- [ ] Perplexity within 5% of fp16 reference (post-hoc tolerance)

## Risks

| Risk | Mitigation |
|---|---|
| pi_quant_simd targets generic NEON; need to verify Pi 5 Cortex-A76 dispatches the right kernel | Check `pi_quant_simd::dispatch()` selects via runtime CPU feature detection |
| Pi 5 doesn't have AVX2 — pi_quant has both NEON + AVX2 paths; ensure NEON one ships | `cfg(target_arch = "aarch64")` already in tree |
| Re-quantizing fp16 → pi_quant introduces error per layer; cumulative error may exceed quality budget | Apply Hadamard rotation pre-conditioning (`quip.rs`) + per-layer calibration on a small dataset |
| BitNet post-hoc on a non-BitNet-trained model may break grammar | Measure perplexity early; gate the rollout |
| Cross-build complexity — pi_quant_simd has `target_feature` attrs that may collide with our +lse +rcpc +fp16 +crc rustflags | Test build before deploying |

## Implementation plan

| Iter | Step |
|---|---|
| 1 | Audit `LlmBackend` trait — list all methods PiQuantBackend must impl |
| 2 | Stub `PiQuantBackend` (load_model accepts `.qm` path, generate panics) |
| 3 | Wire `pi_quant` as the Linear-layer matmul replacement |
| 4 | Quantize TinyLlama-1.1B weights via existing pi_quant CLI tool |
| 5 | Smoke single-request on host then Pi |
| 6 | Cluster bench, measure delta vs ADR-180 |
| 7 | Phase B: BitNet conversion + bench |
| 8 | Convergence email |

## Tracking

- Branch: `feature/ruvllm-pi-quant-native`
- Quantization artifacts go in `~/.cache/ruvllm/quant/<model>/<scheme>.qm`
  on ruvultra; rsync to Pi
