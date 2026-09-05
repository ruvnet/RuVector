# StreamCloud (`streamcloud-rs`)

**StreamCloud** is a standalone **Rust port of [LingBot-Map](https://huggingface.co/robbyant/lingbot-map)** —
a feed-forward, streaming 3D reconstruction foundation model (arXiv:2604.14141,
Apache-2.0). This port replaces the Python/PyTorch + FlashInfer stack with a
pure-Rust [candle](https://github.com/huggingface/candle) tensor backend and an
HNSW trajectory-memory layer backed by
[`ruvector-core`](../crates/ruvector-core).

> **Status:** complete (see [`PROGRESS.md`](./PROGRESS.md)). All crates, ADRs,
> the native CLI, and the WebGPU/WASM demo are implemented and tested (native +
> candle + wasm32). The synthetic backend (ADR-0006) lets the whole pipeline run
> end-to-end without the 4.63 GB checkpoint; the real-weights `candle` backend
> and safetensors loading are wired and compile.

## Why a Rust port?

| Concern | LingBot-Map (Python) | streamcloud-rs |
|---|---|---|
| Tensor backend | PyTorch 2.9 | candle 0.9 (pure Rust, WASM-capable) |
| Long-range memory | FlashInfer **paged KV cache** in VRAM, forgets past a sliding window | **ruvector-core HNSW**, `O(log N)` retrieval, no VRAM ceiling |
| Deployment | Python + CUDA | native (wgpu) **and** WebGPU/WASM in the browser |

The headline change is the memory layer: instead of a linearly-growing KV cache
bounded by GPU VRAM, keyframe features stream into a lock-free HNSW index and the
transformer retrieves the top-K structurally similar past frames each step —
solving long-range drift without a forgetful sliding window. See
[ADR-0002](./docs/adr/ADR-0002-ruvector-streaming-memory.md).

## Crates

| Crate | Purpose |
|---|---|
| [`streamcloud-memory`](./crates/streamcloud-memory) | Streaming trajectory memory over `ruvector-core` HNSW (KV-cache replacement) |
| [`streamcloud-tensor`](./crates/streamcloud-tensor) | `ModelConfig` + safetensors header indexing + candle weight loading |
| [`streamcloud-model`](./crates/streamcloud-model) | `SyntheticReconstructor` (pure-Rust) + candle `GeometricContextTransformer` |
| [`streamcloud-io`](./crates/streamcloud-io) | `FrameSink`, PNG export, streaming H.264/MP4 (`openh264` + `mp4`) |
| [`streamcloud-pipeline`](./crates/streamcloud-pipeline) | Streaming loop + CPU orbit renderer + synthetic scene |
| [`streamcloud-cli`](./crates/streamcloud-cli) | Native demo (`streamcloud render` → PNG + MP4, `streamcloud inspect`) |
| [`streamcloud-wasm`](./crates/streamcloud-wasm) | WebGPU/WASM browser demo bindings |

## Run the native demo

```bash
cargo run -p streamcloud-cli --release -- render \
  --frames 60 --width 640 --height 480 --fps 20 --top-k 16 \
  --out out.mp4 --png-dir frames
# validate a checkpoint header (no multi-GB load):
cargo run -p streamcloud-cli -- inspect --weights model.safetensors
```

Produces a streaming H.264 **MP4** (orbiting camera over the reconstructed point
cloud), a **PNG** sequence, and a final still.

## Run the web demo

See [`demo/README.md`](./demo/README.md). Built and deployed to GitHub Pages by
`.github/workflows/streamcloud-pages.yml`.

## Build & test

```bash
cd streamcloud-rs
cargo build
cargo test
```

The build uses the crates.io **sparse** protocol (configured in
`.cargo/config.toml`) for sandboxed/offline-friendly environments.

## The model checkpoint

The ~4.63 GB checkpoint is **borrowed from the upstream project at runtime**
(local path or HF download) and never committed. When weights are absent, a
deterministic synthetic fallback drives the full pipeline so the demo and
video/image export run end-to-end. See
[ADR-0003](./docs/adr/ADR-0003-candle-tensor-backend-and-weights.md).

## Architecture Decision Records

See [`docs/adr/`](./docs/adr). ADR-0001 (topology) · 0002 (memory) · 0003
(tensors/weights) · 0004 (rendering/deploy) · 0005 (MP4/PNG output).

## License

Apache-2.0, matching the upstream LingBot-Map model. See [`LICENSE`](./LICENSE).
This is an independent reimplementation; model weights remain property of the
original authors under their license.
