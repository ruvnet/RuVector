# ruvector-hailo

Embedding backend for the Hailo-8 NPU on Raspberry Pi 5 + AI HAT+.
Implements `ruvector_core::embeddings::EmbeddingProvider` so any
caller holding `Arc<dyn EmbeddingProvider>` can swap from CPU to NPU
inference with zero code changes.

> **Status:** library, 3 build paths (default / cpu-fallback / hailo
> + cpu-fallback), 11 unit tests passing. Per-Pi throughput:
> ~70 RPS (single-pipeline) / ~70 RPS at p50=43.5ms (pool=2 default
> at concurrency=4). Hard NPU+PCIe ceiling per single-batch HEF —
> see iter-236 commit on the parent cluster crate for the
> measurement that ruled out multi-pipeline overlap.

## Three feature-gated build paths

| `--features` arg | Build does | Runtime path |
|------------------|------------|--------------|
| (none, default) | `cargo check` works on x86 dev hosts | every API returns `Err(HailoError::FeatureDisabled)` |
| `cpu-fallback` | links candle-transformers + tokenizers + safetensors | host-CPU BERT-6 inference (~7 RPS Pi 5) |
| `hailo,cpu-fallback` | also links libhailort via `hailort-sys` | NPU inference via the iter-156b HEF (~70 RPS Pi 5) |

The `cpu-fallback` feature is the floor: even in NPU builds, if the
HEF artifact is missing on disk, the worker falls back to candle
without needing a redeploy. Same model fingerprint is reported in
both modes so cluster integrity gates still work.

## Architecture

```
┌──────────────────────── ruvector-hailo ────────────────────────┐
│                                                                  │
│  ┌─ HailoEmbedder (lib.rs) ──────────────────────────────────┐  │
│  │   open(model_dir) →  hef_path | safetensors → backend     │  │
│  │   embed(text)     →  HefBackend::embed                    │  │
│  │                                                            │  │
│  │   ┌─ HefBackend (iter 235) ────────────────────────────┐  │  │
│  │   │   Single(HefEmbedder)         pool=1 (low load)    │  │  │
│  │   │   Pool(HefEmbedderPool)       pool>=2 (concurrent) │  │  │
│  │   └────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │   ┌─ CpuEmbedder (cpu_embedder.rs, iter 133) ──────────┐  │  │
│  │   │   candle BertModel × pool size                     │  │  │
│  │   └────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ HefPipeline (hef_pipeline.rs) ────────────────────────────┐  │
│  │   hailo_configure_vdevice → input_vstream + output_vstream │  │
│  │   forward_into(embeds, hidden_state)                       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ HostEmbeddings (host_embeddings.rs) ──────────────────────┐  │
│  │   safetensors mmap → BERT embedding-table lookup           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ Tokenizer (tokenizer.rs / tokenizers crate) ──────────────┐  │
│  │   tokenizer.json → input_ids + attention_mask              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

End-to-end pipeline: `text → tokenize → host embed-lookup →
NPU forward → mean-pool → l2-normalize → Vec<f32>` (384 dims for
all-MiniLM-L6-v2).

## Benchmarks

Single-Pi cognitum-v0 (Pi 5 + AI HAT+, all-MiniLM-L6-v2 HEF
sha256 cdbc89...), measured via `ruvector-hailo-cluster-bench`:

| pool size | concurrency | throughput | p50    | p99    |
|-----------|-------------|------------|--------|--------|
| 1         | 1           | 70.6 RPS   | 14.1ms | 15.8ms |
| 1         | 4           | 70.7 RPS   | 56.7ms | 74.7ms |
| 1         | 8           | 70.7 RPS   | 112.7ms| 170.7ms|
| 2         | 4           | 70.7 RPS   | 43.3ms | 84.7ms |
| 4         | 4           | 70.7 RPS   | 43.5ms | 84.9ms |

Throughput plateau confirms NPU+PCIe-bound 70 RPS ceiling per
single-batch HEF; pool=2 cuts p50 by 23% at multi-concurrent load
because each request gets its own host-side queue slot. RSS cost
of the pool: pool=2 = 142 MB, pool=4 = 251 MB (vs 87 MB for pool=1).

CPU fallback bench (cpu-fallback feature, host-CPU candle):
~7 RPS Pi 5, ~3-4 RPS Pi 4 (estimated; see ADR-177).

## Security posture

ADR-172 hardening landed before this branch:
- HEF magic-byte check + optional sha256 pin via `RUVECTOR_HEF_SHA256`
  (iter 173/174/198, defends against substituted HEF).
- 1 MB cap on PEM / manifest / signature reads (iter 210/211/212).
- 16 MB cap on tokenizer.json (iter 213).
- Worker-side `RUVECTOR_LOG_TEXT_CONTENT=full` capped at 200 chars
  per entry (iter 247) to bound journald volume.
- All operator-controlled file reads stat-then-read so a misconfig
  pointing at /var/log/* errors instead of OOMing the worker.

## API surface

```rust
use ruvector_hailo::{HailoEmbedder, HailoError};
use std::path::Path;

let embedder = HailoEmbedder::open(Path::new("/var/lib/ruvector-hailo/models/all-minilm-l6-v2"))?;
println!("dim = {}", embedder.dimensions());
println!("device = {}", embedder.device_id());
let v: Vec<f32> = embedder.embed("hello world")?;
assert_eq!(v.len(), 384);
```

`HailoEmbedder` implements `ruvector_core::embeddings::EmbeddingProvider`
since iter-218, so it slots into the trait-object dispatch:

```rust
let provider: Arc<dyn ruvector_core::embeddings::EmbeddingProvider> =
    Arc::new(embedder);
```

## Environment variables

| var | meaning |
|-----|---------|
| `RUVECTOR_HEF_SHA256` | iter-174 — pin HEF sha256, refuse to start on mismatch |
| `RUVECTOR_NPU_POOL_SIZE` | iter-235 — pipeline pool size, default 1 (single-mutex), >=2 enables `HefEmbedderPool` |
| `RUVECTOR_CPU_FALLBACK_POOL_SIZE` | iter-147 — parallel candle BERT pool slots when on CPU fallback |

## See also

- `crates/ruvector-hailo-cluster/README.md` — multi-Pi coordinator
  layered on top of this crate. The systemd-deployed worker, the
  three sensor bridges, and the `ruvector-hailo-cluster-bench` /
  `-embed` / `-stats` CLIs all live there.
- `docs/adr/ADR-167-*.md` — design rationale for the NPU backend.
- `docs/adr/ADR-176-*.md` — iter-156b HEF compile + iter-163 NPU
  default switch.
- `docs/adr/ADR-178-*.md` — workspace integration gap analysis.
