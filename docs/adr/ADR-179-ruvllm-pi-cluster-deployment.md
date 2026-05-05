---
adr: 179
title: "EPIC — Deploy ruvllm LLM inference on the 4-node Pi 5 + AI HAT+ cluster with pi_quant / turbo_quant / RaBitQ"
status: proposed
date: 2026-05-04
authors: [ruvnet, claude-flow]
related: [ADR-167, ADR-172, ADR-173, ADR-176, ADR-177, ADR-178]
branch: feature/ruvllm-pi-cluster
---

# ADR-179 — EPIC: ruvllm LLM inference on Pi 5 cluster

## Status

**Proposed.** Implementation iterating under `/loop 5m` cron `0dd7f865`
on branch `feature/ruvllm-pi-cluster`. Convergence target: SOTA per-node
tok/s for Llama-3.2-1B / Qwen2.5-1.5B / phi-3-mini at ≤1% perplexity gap
vs fp16 reference.

Canonical iteration log:
`crates/ruvector-hailo-cluster/RUVLLM_CLUSTER_PLAN.md`.

## Context

The Pi 5 + AI HAT+ cluster brought online via ADR-176 (HEF NPU
embeddings) and the rebirth-clone tooling shipped in iter 244 has 4
nodes serving the embedding worker on `:50051`. ADR-178 closed the
last bridge install gap. The cluster is currently 100% embedding —
generative inference still routes to the Python `ruos-llm-serve`
(Qwen2.5-3B-Instruct) on ruvultra's `127.0.0.1:8080`, which:

1. Single-host (no HA, no horizontal scale)
2. Python + transformers, no quantization frontier
3. Uses ruvultra's CUDA — orthogonal to the Pi fleet
4. Doesn't exercise any of the in-tree work in `crates/ruvllm/`

Meanwhile `crates/ruvllm/` already ships:

- A complete inference engine (`serving::engine`, `paged_attention`,
  `kv_cache`, `lora`, `moe`, `kernels`)
- Pi-targeted quantization paths: `pi_quant`, `pi_quant_simd`
- `turbo_quant` + `turboquant_profile`
- `quip` + `hadamard` + `incoherence` rotations
- `ruvltra_quant` (named for ruvultra hardware)
- A `ruvllm` CLI binary at `crates/ruvllm-cli/` with
  `serve | quantize | benchmark | download | chat | info | list`

`crates/ruvector-rabitq/` is a sibling crate already implementing
RaBitQ for vector quantization. These pieces have never been
integrated into a per-node cluster deploy.

## Decision

Deploy the in-tree `ruvllm` engine on each of the 4 cognitum Pi 5
nodes as a **second worker per node**, listening on a separate port
(`:50053`, alongside the embedding worker on `:50051`), and add a
cluster-side dispatcher binary that mirrors the existing P2C+EWMA
pattern from `ruvector-hailo-cluster::dispatch` for completion RPCs.
Iterate quantization (pi_quant → turbo_quant → +RaBitQ KV) until
throughput and quality converge.

### What ships

| Component | Location | New / existing |
|---|---|---|
| aarch64 `ruvllm` binary | `target/aarch64-unknown-linux-gnu/release/ruvllm` | existing crate, new build target |
| `ruvllm-pi-worker.service` | `crates/ruvector-hailo-cluster/deploy/` | new |
| `ruvllm-pi-worker.env.example` | `crates/ruvector-hailo-cluster/deploy/` | new |
| `install-ruvllm-pi-worker.sh` | `crates/ruvector-hailo-cluster/deploy/` | new (mirror of `install.sh`) |
| `ruvllm-cluster-bench` bin | `crates/ruvector-hailo-cluster/src/bin/` | new |
| `ruvllm-cluster-embed` (cluster client for completions) | `crates/ruvector-hailo-cluster/src/bin/` | new |
| Iteration log | `crates/ruvector-hailo-cluster/RUVLLM_CLUSTER_PLAN.md` | new |
| Dispatcher hook for completion RPCs | `crates/ruvector-hailo-cluster/src/dispatch.rs` | extension |

### What stays out of scope (for v1 of this ADR)

- **Hailo-8 NPU acceleration of LLM ops**: the chip is optimized for
  fixed-shape vision encoders. Compiling Llama-class HEFs is a
  research project, not a deploy step. Tracked as ADR-180 (future).
- **Pipeline / tensor parallelism across Pis**: Tailscale RTT (5–15
  ms) makes per-token cross-node decode untenable. Each Pi runs the
  full model, replicated, request-level load-balanced.
- **Replacing `ruos-llm-serve` on ruvultra**: orthogonal. The Python
  service stays; the Pi cluster adds horizontal capacity for small
  quantized models.

## Architecture

```
                              ┌──────────────────────┐
                              │  ruvllm bin          │   one per Pi 5
                              │  (serve mode)        │   :50053
                              │  pi_quant + turbo Q  │
                              │  pool=N requests     │
                              └──────────▲───────────┘
                                         │ completion RPC (gRPC or
                                         │ OpenAI-compat HTTP)
                          ┌──────────────┼──────────────┐
                          │              │              │
                  cognitum-v0       cluster-1       cluster-2/3
                  :50051 embed      :50051 embed    :50051 embed
                  :50053 llm        :50053 llm      :50053 llm
                                         │
                          ┌──────────────┴──────────────┐
                          │ ruvllm-cluster-bench        │   ruvultra
                          │ P2C+EWMA across 4 :50053    │
                          └─────────────────────────────┘
```

## Models in scope

| Model | Params | fp16 size | Q4 size | Per-Pi feasible |
|---|---:|---:|---:|---|
| Llama-3.2-1B | 1.0 B | ~2.0 GB | ~600 MB | Yes — 8 GB Pi has wide margin |
| Qwen2.5-1.5B | 1.5 B | ~3.0 GB | ~900 MB | Yes |
| phi-3-mini | 3.8 B | ~7.6 GB | ~2.3 GB | Yes (tight; ≤2 in-flight) |

## Quantization sequencing

1. **pi_quant** (ADR-090): 3-bit / 2-bit per-block uniform with
   Hadamard rotation precondition. Targets >1 GB/s quantize, >10
   GB/s dequant on NEON. Already benched at `pi_quant_bench`.
2. **turbo_quant** stacked on top: tile-wise scale interpolation
   (per `turboquant_profile.rs`).
3. **QuIP / incoherence** (`quip.rs`): rotation pre-conditioning to
   improve quantization error before the codebook step.
4. **RaBitQ on KV-cache** via `crates/ruvector-rabitq` (the
   `sparse_inference::ruvllm` integration already exists per
   `crates/ruvector-sparse-inference/src/integration/ruvllm.rs`)
   — KV memory shrink without re-quantizing weights.

## Convergence rule

Per the operator `/loop` directive: stop iterating when **both**
tok/s and p50 stop improving for 2 consecutive iterations across all
3 target models, AND perplexity stays within 1% of fp16 reference
(measured against the existing `crates/ruvllm/src/evaluation/`
harness). Then declare "fully optimized and benchmarked", `CronDelete
0dd7f865`, render a benchmark report, email to ruv@ruv.net via the
Resend `cluster@cognitum.one` channel established in the embed
optimization run.

## Risks

| Risk | Mitigation |
|---|---|
| `crates/ruvllm` doesn't cross-build for aarch64 cleanly (CUDA/Metal/ANE backends pulled in by default features) | Feature-gate audit; build with `--no-default-features --features cpu,quantize,serve-grpc` (audit on iter 1) |
| Pi 5 RAM saturation with phi-3-mini Q4 + KV under load | Cap concurrent requests via `RUVLLM_MAX_INFLIGHT`; scheduler drops to queue when RAM pressure detected |
| Quality regression from aggressive quantization | Quality gate in convergence rule; fallback to less aggressive setting on regression |
| Quantization compile times extend iter beyond 5 min | Quantize once per (model × scheme), persist `.qm` artifact in `/var/lib/ruvllm/models/` and rsync to nodes |

## Acceptance

This ADR is **accepted** when all checked:

- [ ] `ruvllm-pi-worker.service` running on all 4 nodes
- [ ] All 3 target models loaded, quantized, serving completions
- [ ] `ruvllm-cluster-bench` reports per-node + 4-node aggregate tok/s
- [ ] Perplexity ≤1% of fp16 reference
- [ ] Convergence (2 consecutive non-improving iterations) declared
- [ ] Benchmark report emailed
- [ ] Branch merged to main

## Tracking

- Branch: `feature/ruvllm-pi-cluster`
- Cron: `0dd7f865` (every 5 min)
- Iter log: `crates/ruvector-hailo-cluster/RUVLLM_CLUSTER_PLAN.md`
- Backup: SD images at `/home/ruvultra/backups/cognitum-v0/`
