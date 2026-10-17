# ADR-0002: ruvector-core Streaming Memory Replaces the Paged KV Cache

- **Status**: Accepted
- **Date**: 2026-06-27

## Context

LingBot-Map streams 3D reconstruction from video. The PyTorch original manages
temporal frame representations in a **paged KV cache** (FlashInfer) resident in
GPU VRAM. This cache grows linearly with frame count: past a few thousand frames
the model exhausts VRAM and must switch to a **sliding window** (`--mode
windowed` with keyframe overlap), which permanently forgets older geometry and
reintroduces long-range trajectory drift.

## Decision

Replace the paged KV cache with a lock-free **HNSW** index backed by
[`ruvector-core`], exposed through `streamcloud-memory::StreamingMemory`:

- `insert_keyframe(frame_id: u64, features: &[f32])` — ingest a keyframe's
  flattened geometric feature vector. Lock-free, so it can run on a background
  ingestion thread while inference proceeds.
- `retrieve_context(query: &[f32], top_k)` — fetch the top-K structurally most
  similar past keyframes (any distance back in time) for drift correction and
  pose-reference windowing.

Frame ids are stored as `VectorEntry` ids and parsed back on retrieval, so the
database speaks the streaming timeline directly.

### Complexity & capacity

| | Paged KV (linear scan / window) | ruvector HNSW |
|---|---|---|
| Lookup cost | `O(N·d)` (or window-bounded, forgetful) | `O(log N · d)` |
| Capacity | GPU VRAM | system RAM (mmap-backed when persisted) |
| Long-range recall | lost beyond window | preserved for all N |

## API reality

Earlier sketches referenced a `VectorDb` / `IndexConfig` / `MetricType` surface.
The **real** `ruvector-core` API is `VectorDB` driven by `DbOptions`,
`VectorEntry`, `SearchQuery`, and `SearchResult`. `StreamingMemory` is the thin,
intent-revealing adapter over that real API. We build it with
`default-features = false, features = ["hnsw", "parallel"]` to get an in-memory,
lock-free index without pulling redb persistence or AVX-512 (nightly).

## Consequences

- The windowed-mode code path of the original is eliminated; memory is global,
  persistent (optionally), and dynamically retrieved.
- Retrieval returns *candidate anchors* — the model still loads/attends to only
  those K frames, bounding per-step attention cost regardless of total N.
- HNSW is approximate; `ef_search` is tunable (`StreamingMemoryConfig`) to trade
  recall vs latency. Verified: exact-feature query returns its own frame as the
  top hit, including at a 5000-frame range (see crate tests).
