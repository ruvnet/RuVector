# Codebase Summary

A map of the `ruvllm` crate: directory layout, source modules, dependencies,
and binary targets.

## Directory Tree (top three levels)

```
ruvLLM/
├── Cargo.toml                  # crate manifest, features, bin targets
├── README.md                   # short user-facing intro (do not modify)
├── config/
│   └── example.toml            # canonical configuration template (8 sections)
├── src/                        # library + binary sources
│   ├── lib.rs                  # crate root
│   ├── orchestrator.rs         # request pipeline
│   ├── types.rs                # shared data types
│   ├── config.rs               # TOML config loader
│   ├── error.rs                # thiserror-based error enum
│   ├── embedding.rs            # LRU + tokenization
│   ├── memory.rs               # HNSW vector store
│   ├── router.rs               # FastGRNN gated routing
│   ├── attention.rs            # multi-head graph attention
│   ├── inference.rs            # mock + SIMD pool dispatch
│   ├── inference_real.rs       # Candle backend (real-inference)
│   ├── simd_inference.rs       # AVX2/SSE4.1/NEON kernels
│   ├── learning.rs             # replay buffer + EWC + async writeback
│   ├── compression.rs          # quantization helpers
│   ├── training.rs             # pretrain driver
│   ├── napi.rs                 # Node.js bindings (napi feature)
│   ├── bin/                    # binary entry points
│   └── sona/                   # learning subsystem
│       ├── engine.rs           # SONA orchestrator
│       ├── lora.rs             # MicroLoRA + BaseLoRA
│       ├── ewc.rs              # online Fisher Information
│       ├── reasoning_bank.rs   # K-means++ pattern store
│       ├── trajectory.rs       # per-request reasoning trace
│       └── loops/
│           ├── instant.rs      # <100 µs path
│           ├── background.rs   # hourly extraction
│           └── coordinator.rs  # weekly EWC++ pass
├── tests/                      # integration tests
│   ├── integration.rs          # async pipeline tests
│   └── sona_integration.rs     # learning-loop tests
├── benches/                    # Criterion benches
│   ├── pipeline.rs
│   ├── router.rs
│   ├── memory.rs
│   ├── attention.rs
│   └── sona_bench.rs
├── docs/                       # this documentation set
│   ├── index.md                # canonical nav (authoritative)
│   ├── SONA/                   # learning deep dives (authoritative)
│   ├── sparc/                  # SPARC methodology specs (authoritative)
│   └── *.md                    # generated guides
├── esp32/                      # ESP32 library sub-crate (no_std)
└── esp32-flash/                # ESP32 firmware (publish=false)
```

## Source Module Table

Every top-level `.rs` file in `src/` and its responsibility.

| Module | Purpose | Hot path? |
|---|---|---|
| `lib.rs` | Crate root, re-exports public API | n/a |
| `orchestrator.rs` | Chains embedding → memory → routing → attention → inference → learning | yes |
| `types.rs` | Shared structs (`Query`, `Response`, etc.) | yes |
| `config.rs` | Loads `config/example.toml` style files | startup |
| `error.rs` | `thiserror`-derived error enum | n/a |
| `embedding.rs` | LRU cache + tokenizer wrapper | yes |
| `memory.rs` | HNSW index over 768-D vectors | yes |
| `router.rs` | FastGRNN adaptive routing, sparse forward | yes |
| `attention.rs` | Multi-head graph attention over retrieved nodes | yes |
| `inference.rs` | Mock backend + SIMD-pool dispatcher | yes |
| `inference_real.rs` | Candle CPU/GPU/Metal real inference | yes (gated) |
| `simd_inference.rs` | AVX2 / SSE4.1 / NEON kernels with runtime detection | yes |
| `learning.rs` | Replay buffer + EWC consolidation + async writeback | background |
| `compression.rs` | INT8 / INT4 / binary quantization helpers | offline |
| `training.rs` | Pre-training driver used by `ruvllm-pretrain` | offline |
| `napi.rs` | Node.js bindings emitted under the `napi` feature | n/a |

The `sona/` submodule is a sub-system, not a single module. Each file there is
described in [System Architecture](system-architecture.md) and in greater
depth in [SONA Overview](SONA/00-OVERVIEW.md).

## Binary Targets

All binaries live in `src/bin/` and are declared in `Cargo.toml`. They share
the library code; features control which ones are buildable.

| Binary | Default? | Required feature | Description |
|---|---|---|---|
| `ruvllm-demo` | yes | — | Interactive REPL using mock inference, useful for smoke-testing the orchestrator end-to-end without loading a real model. |
| `ruvllm-server` | no | `server` | Axum HTTP server exposing `/health`, `/query`, `/stats`, `/feedback`, `/session`. See [API Reference](api-reference.md). |
| `ruvllm-bench` | yes | — | Quick latency probe; useful as a CI smoke test. |
| `ruvllm-benchmark-suite` | yes | — | Wraps the full Criterion suite for one-shot reproducible numbers. |
| `ruvllm-simd-demo` | yes | — | Prints which SIMD instruction set was selected at runtime. |
| `ruvllm-pretrain` | yes | — | Drives the pre-training pipeline implemented in `training.rs`. |
| `ruvllm-export` | no | `hf-export` | Exports trained adapters/weights to HuggingFace Hub format. |

## Key Dependencies

The top dependencies that shape the runtime, organized by role.

| Crate | Version | Role | Phase |
|---|---|---|---|
| `ruvllm-lib` | path `../../crates/ruvllm` | Flash Attention 2 + NEON/Metal kernels | runtime |
| `ruvector-core` | path `../../crates/ruvector-core` | Embedding + HNSW primitives | runtime |
| `tokio` | 1.41 | Async runtime (multi-thread + sync + macros) | runtime |
| `ndarray` | 0.16 | Tensor math, with `serde` + `rayon` features | runtime |
| `serde` | 1.0 | Serialization, used pervasively | runtime |
| `serde_json` | 1.0 | JSON for HTTP and config | runtime |
| `simsimd` | 5.9 | SIMD distance metrics on the hot path | runtime |
| `dashmap` | 6.1 | Concurrent hashmap for caches | runtime |
| `parking_lot` | 0.12 | Faster `Mutex` / `RwLock` than std | runtime |
| `candle-*` | 0.8 | Real inference backend (optional) | runtime (gated) |
| `hf-hub` | 0.3 | HuggingFace download (optional) | runtime (gated) |
| `thiserror` | — | Error derives, see [Code Standards](code-standards.md) | runtime |

Dev-only dependencies of note: `criterion` 0.5 with `async_tokio` and
`html_reports` for the benches.

## Feature Flags

The Cargo features map to optional functionality. Features compose: enable
several at once or use `full`.

| Feature | Default | Effect |
|---|---|---|
| `storage` | yes | Persistent vector store + HNSW index |
| `metrics` | yes | Prometheus metric export |
| `server` | no | Axum + Tower HTTP stack for `ruvllm-server` |
| `real-inference` | no | Candle CPU SIMD + HF Hub model loading |
| `hf-export` | no | HuggingFace export via `ruvector-sona` |
| `parallel` | no | Rayon-parallel GEMM / GEMV (4–6× speedup) |
| `candle` | no | Candle backend without HF Hub |
| `metal` | no | Metal GPU backend |
| `inference-metal` | no | Metal-specialized inference path |
| `napi` | no | Node.js native module |
| `full` | no | Enables every above feature |

See [Configuration Guide](configuration-guide.md) for which features pair with
which TOML sections, and [Deployment Guide](deployment-guide.md) for the
recommended feature combinations per target.

## Tests

| File | Style | Coverage |
|---|---|---|
| `tests/integration.rs` | `#[tokio::test]` async | Full pipeline: query, context, confidence threshold, latency budget |
| `tests/sona_integration.rs` | `#[tokio::test]` async | Trajectory → ReasoningBank → LoRA flow, concurrent safety, instant-loop latency under load |

Run with `cargo test`. See [Testing Guide](testing-guide.md) for details.

## Benchmarks

All benches use Criterion 0.5 with `async_tokio` and HTML reports.

| Bench | Measures |
|---|---|
| `pipeline.rs` | End-to-end query latency vs. input length |
| `router.rs` | FastGRNN forward and training, dim 64–512 |
| `memory.rs` | HNSW insert and search, 768-D, batches 10–500 |
| `attention.rs` | Multi-head attention on variable subgraphs (768-D) |
| `sona_bench.rs` | MicroLoRA <100 µs, trajectory <1 µs/step, ReasoningBank, InstantLoop <1 ms, EWC++ |

Reports land in `target/criterion/report/index.html`. See
[Testing Guide](testing-guide.md) for invocation patterns.

## ESP32 Sub-Crates

Two separate crates, both outside the main `src/` tree.

| Crate | `publish` | Role |
|---|---|---|
| `esp32/` | yes | Library: INT8/INT4/Binary quantization, no_std, ESP32 family (320–512 KB SRAM). Features: `esp32-std`, `no_std`, `federation`, `q8`, `q4`, `binary`, `esp32s3-simd`. Deps: `heapless` 0.8, `libm`, `fixed`, `postcard`. |
| `esp32-flash/` | no | Firmware: depends on `esp32` lib, adds `main.rs`, `Makefile`, `Dockerfile`, `install.sh`, `cluster-flash.sh`. Target `xtensa-esp32-espidf`. |

See [Deployment Guide](deployment-guide.md) for flashing instructions.

## Configuration

Canonical TOML lives in `config/example.toml` and is split into eight
sections: `[system]`, `[embedding]`, `[memory]`, `[router]`, `[inference]`,
`[learning]`, plus the runtime-specific sections covered in
[Configuration Guide](configuration-guide.md).

## See also

- [Project Overview & PDR](project-overview-pdr.md)
- [System Architecture](system-architecture.md)
- [Configuration Guide](configuration-guide.md)
- [Testing Guide](testing-guide.md)
