# RuvLLM — Project Overview & Product Definition Record

> Self-learning LLM orchestration over a frozen base model, with sub-millisecond
> routing, adaptive vector memory, and three temporally separated learning loops.

## Vision

RuvLLM is an orchestration layer that turns a static, pre-trained base model
(LFM2) into a continuously improving system **without ever fine-tuning the base
weights**. Adaptation happens entirely in side-car components — vector memory,
gated routing, lightweight LoRA adapters, and Elastic Weight Consolidation —
which together let the system learn from every interaction while preserving the
foundation model's general competence.

The design target is two-fold:

1. **Sub-millisecond orchestration latency** so RuvLLM can sit in front of any
   inference endpoint without becoming the bottleneck. Measured P50 ~0.06 ms,
   P95 ~0.08 ms (see `benches/pipeline.rs`).
2. **Edge-to-cloud portability** — the same crate runs as an Axum server on a
   workstation and, via the `esp32/` sub-crate, as quantized firmware on
   ESP32-class microcontrollers with 320–512 KB of SRAM.

## Problem Domain

Production LLM stacks face three recurring tensions:

| Tension | Symptom | RuvLLM's response |
|---|---|---|
| Adaptation vs. catastrophic forgetting | Fine-tuning erodes general skills | Frozen base + LoRA adapters + EWC++ Fisher penalties |
| Latency vs. richness of context | Long context windows = slow inference | HNSW-backed vector memory + gated routing decides what to inject |
| Centralized inference vs. edge cost | Cloud round-trips dominate | INT8/INT4/Binary quantization, no_std ESP32 target |

RuvLLM treats these as a single architectural problem: **what learns, where,
and on what time scale**. The answer is the three-loop hierarchy described
below.

## Key Innovations

### 1. Three Temporal Learning Loops

Adaptation is decomposed across three time scales so each loop can use the
right algorithm without blocking the request path. The full architecture is
documented in [SONA Overview](SONA/00-OVERVIEW.md) — this section is a summary.

| Loop | Cadence | What learns | Mechanism |
|---|---|---|---|
| Instant | <100 µs / request | Per-request adapters | MicroLoRA rank 1–2, in-place |
| Background | hourly | Pattern extraction | K-means++ over reasoning trajectories |
| Consolidation | weekly | Stable knowledge | EWC++ online Fisher into BaseLoRA rank 4–16 |

The instant loop runs **inline** with the request and is bounded by the
sub-millisecond latency budget. The background loop runs as a tokio task
operating on a replay buffer. The weekly loop runs the EWC++ pass that decides
which MicroLoRA deltas graduate into the BaseLoRA.

### 2. Sub-Millisecond Orchestration

The full orchestrator path — embedding lookup → HNSW memory search →
FastGRNN routing → multi-head graph attention → inference dispatch — completes
in microseconds because every hot-path component is cache-friendly and SIMD-
accelerated:

- `simsimd` 5.9 for distance kernels (AVX2, SSE4.1, NEON detected at runtime).
- `dashmap` 6.1 for concurrent embedding cache without global locks.
- `parking_lot` 0.12 for the few read-mostly mutexes on the hot path.
- `ndarray` 0.16 with the `rayon` feature for GEMM/GEMV when `parallel` is on.

Mock inference (`inference.rs`) and SIMD inference (`simd_inference.rs`) provide
two backends for benchmarking the orchestrator independently of model load.
Real inference flows through `inference_real.rs` using the Candle stack
(`candle-*` 0.8) when the `real-inference` feature is enabled.

### 3. Edge Deployment via ESP32

The `esp32/` sub-crate is a separate `no_std` library sized for the ESP32
family of microcontrollers. It strips out tokio, ndarray, and HNSW and replaces
them with `heapless` 0.8 collections, `libm` for math, and `fixed` for
deterministic arithmetic. Quantization is pluggable via Cargo features:

- `q8` — INT8 weights, default for ESP32-S3 with PSRAM.
- `q4` — INT4 packed, halves memory at small accuracy cost.
- `binary` — 1-bit XNOR layers for ultra-tight memories.
- `esp32s3-simd` — uses the S3 vector instructions when available.
- `federation` — turns on the federated-aggregation primitives so a fleet of
  ESP32 boards can share weights without a central coordinator.

The companion `esp32-flash/` crate is the flashable firmware: it depends on the
`esp32` library, adds `main.rs`, a `Makefile`, a `Dockerfile`, an
`install.sh`, and a `cluster-flash.sh` script for flashing many chips at once.
It targets `xtensa-esp32-espidf` and is published as `publish=false`.

## Target Users

| Audience | Why RuvLLM fits |
|---|---|
| LLM-platform researchers | Frozen-base + LoRA + EWC is a clean substrate for studying continual learning without retraining the base. |
| Latency-bound application teams | Sub-ms orchestration lets RuvLLM sit in front of an existing endpoint without budget impact. |
| Edge-AI / IoT deployments | ESP32 sub-crate gives a coherent path from server to microcontroller with the same memory and routing logic. |
| Self-learning agent builders | The reasoning bank + trajectory store + replay buffer are first-class, not bolt-ons. |

## Success Metrics

The benchmark suite in `benches/` quantifies whether each architectural claim
holds. Run `cargo bench` to reproduce; HTML reports land in
`target/criterion/report/index.html`.

| Metric | Target | Source |
|---|---|---|
| End-to-end query P50 | <0.10 ms | `benches/pipeline.rs` |
| End-to-end query P95 | <0.15 ms | `benches/pipeline.rs` |
| FastGRNN forward (dim 128) | µs-class | `benches/router.rs` |
| HNSW search, 768D, 500-batch | sub-ms | `benches/memory.rs` |
| MicroLoRA forward | <100 µs | `benches/sona_bench.rs` |
| Trajectory append | <1 µs / step | `benches/sona_bench.rs` |
| InstantLoop full pass | <1 ms | `benches/sona_bench.rs` |

These numbers are the contract. Regressions on any of them are treated as
release-blocking. See [Testing Guide](testing-guide.md) for how to run the
suite and where the per-bench reports live.

## Scope Boundaries

**In scope.** Orchestration of a frozen base model, vector-memory recall,
adaptive routing, three-loop learning, edge quantization, an HTTP server, a
Node.js binding (`napi` feature), and a HuggingFace export pipeline
(`hf-export` feature).

**Out of scope.** Pre-training the base model itself, distributed training of
the base, multi-GPU scheduling beyond what Candle provides, and any form of
prompt-engineering DSL — RuvLLM is the substrate, not the agent layer.

## Crate Shape

`ruvllm` is a single mixed `cdylib + rlib` crate. It is **not** a workspace.
Six binary targets live alongside the library:

| Binary | Purpose |
|---|---|
| `ruvllm-demo` | Interactive REPL with mock inference |
| `ruvllm-server` | Axum HTTP server (requires `server` feature) |
| `ruvllm-bench` | Quick latency check |
| `ruvllm-benchmark-suite` | Comprehensive Criterion suite |
| `ruvllm-simd-demo` | Runtime SIMD detection demo |
| `ruvllm-pretrain` | Training pipeline driver |
| `ruvllm-export` | HuggingFace export (requires `hf-export` feature) |

The full directory and module layout is documented in
[Codebase Summary](codebase-summary.md), and the per-component design is in
[System Architecture](system-architecture.md).

## Documentation Map

This file is the entry point. The rest of the documentation set:

- [Codebase Summary](codebase-summary.md) — directory tree, modules, deps.
- [System Architecture](system-architecture.md) — diagrams + module narrative.
- [API Reference](api-reference.md) — HTTP endpoints + library API.
- [Configuration Guide](configuration-guide.md) — every TOML key, with tuning patterns.
- [Deployment Guide](deployment-guide.md) — server, Docker, ESP32 flashing.
- [Testing Guide](testing-guide.md) — unit, integration, Criterion benches.
- [Code Standards](code-standards.md) — Rust conventions used here.
- [SONA Overview](SONA/00-OVERVIEW.md) — the learning architecture deep dive.
- [SPARC Specification](sparc/01-specification.md) — methodology spec.
- [docs/index.md](index.md) — the canonical navigation index.

## See also

- [SONA Overview](SONA/00-OVERVIEW.md)
- [System Architecture](system-architecture.md)
- [Codebase Summary](codebase-summary.md)
- [Deployment Guide](deployment-guide.md)
