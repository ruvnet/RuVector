# ADR-0001: Standalone Workspace & Crate Topology

- **Status**: Accepted
- **Date**: 2026-06-27
- **Context**: StreamCloud

## Context

We are porting [LingBot-Map](https://huggingface.co/robbyant/lingbot-map) — a
feed-forward streaming 3D reconstruction foundation model (PyTorch + FlashInfer)
— to Rust. The port must be a *standalone* deliverable: its own workspace,
buildable and splittable into its own repository, while currently living inside
the `ruvnet/ruvector` monorepo (the only repository this effort has write access
to).

## Decision

Create a self-contained Cargo workspace at `streamcloud-rs/` with a focused set
of single-responsibility crates:

| Crate              | Responsibility |
|--------------------|----------------|
| `streamcloud-memory`   | Streaming trajectory memory over `ruvector-core` HNSW (KV-cache replacement). Candle-free; operates on `&[f32]`. |
| `streamcloud-tensor`   | `ModelConfig`, safetensors header indexing, candle weight materialization (feature-gated). |
| `streamcloud-model`    | Geometric Context Transformer (candle): patch embedding, anchor cross-attention, drift-correction head. |
| `streamcloud-io`       | Frame sources, PNG export, streaming MP4 muxing. |
| `streamcloud-pipeline` | Streaming inference orchestration; candle ⇄ memory bridge. |
| `streamcloud-cli`      | Native demo: wgpu point-cloud viewer + headless render. |
| `streamcloud-wasm`     | WebGPU/WASM bindings for the browser demo. |

The workspace is added to the monorepo root `Cargo.toml` `exclude` list so it
never participates in (or breaks) the parent build, and so it can be lifted out
verbatim into its own repo.

## Rationale

- **Bounded contexts**: memory, tensors, model, IO, and presentation are
  independent concerns with different dependency weights. Keeping
  `streamcloud-memory` candle-free means it compiles in seconds and is unit-testable
  without a tensor backend.
- **Fast CI**: heavy dependencies (candle, wgpu) are isolated to the crates that
  need them and gated behind features where possible.
- **Splittability**: path dependencies + `exclude` keep the port a clean island.

## Consequences

- `ruvector-core` is referenced by relative path (`../crates/ruvector-core`);
  splitting the repo later means switching to a git/version dependency.
- The crate list in `[workspace].members` grows as crates land; crates are added
  to `members` only once they exist so `cargo build` is always green.

## Build note

The sandbox build environment blocks the crates.io **git** index protocol.
`streamcloud-rs/.cargo/config.toml` forces the **sparse** protocol
(`index.crates.io`, allow-listed). See PROGRESS.md.
