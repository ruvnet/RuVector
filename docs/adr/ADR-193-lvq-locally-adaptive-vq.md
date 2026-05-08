---
adr: 193
title: "Locally-Adaptive Vector Quantization (LVQ) crate for sub-fp32 memory ANN"
status: proposed
date: 2026-05-08
authors: [ruvector-nightly, claude-code]
related: [ADR-143, ADR-187, ADR-188, ADR-189, ADR-190, ADR-191, ADR-192]
tags: [vector-search, quantization, lvq, hnsw, diskann, memory, recall, ann]
---

# ADR-193 — Locally-Adaptive Vector Quantization (LVQ) crate

## Status

**Proposed.** A working PoC ships on branch
`research/nightly/2026-05-08-lvq-locally-adaptive-vq` as the new crate
`crates/ruvector-lvq` (added to the workspace `members` list). All ten
acceptance tests pass under `cargo test -p ruvector-lvq --release`.
Real benchmark numbers from a 200 000 × 128 dataset on Apple M4 Max are
captured in
[`docs/research/nightly/2026-05-08-lvq-locally-adaptive-vq/README.md`](../research/nightly/2026-05-08-lvq-locally-adaptive-vq/README.md).

## Context

ruvector already exposes two ends of the vector-compression spectrum:

| Crate                        | Bits/dim | Recall  | Memory  | Niche                          |
|------------------------------|----------|---------|---------|--------------------------------|
| `ruvector-rabitq`            | 1        | medium  | ~3.1%   | extreme compression            |
| `ruvector-core` (fp32 HNSW)  | 32       | perfect | 100%    | uncompressed baseline          |

Customers running cosine-similarity workloads on dense LLM embeddings
(e.g. OpenAI `text-embedding-3-large`, 3 072-dim; mistral-embed,
1 024-dim) sit in a different operating point: they want **memory
reduction without measurable recall loss**. Binary quantization gives
up too much for them; uncompressed fp32 burns RAM that could fund
larger graph fan-out.

Intel's *Scalable Vector Search* (SVS, VLDB 2024) introduced
**Locally-Adaptive Vector Quantization (LVQ)** to fill exactly this
gap: per-vector 8-bit codes with a per-vector `(mean, bias, scale)`
triple, optionally followed by a residual second level. Empirically it
matches fp32 recall at ~50% of the memory while paying ~10–30% extra
latency on a flat brute-force scan and *less* (cache effects flip the
sign) on graph indexes at billion-vector scale.

There is no LVQ implementation in the open Rust ANN ecosystem today —
all SOTA references are C++ (SVS, FAISS) or Python (Pinecone, Weaviate
internal). Shipping one in ruvector lets the project occupy this
operating point and lays the foundation for LeanVec
(orthogonal-projection extension) and asymmetric int8 SIMD kernels.

## Decision

Introduce a **standalone Rust crate `ruvector-lvq`** with the following
public surface:

* `Lvq8`, `Lvq8Stats`, `Lvq8Code` — single-level encoder, decoder, and
  storage container.
* `Lvq8x8` — two-level encoder using the residual.
* `lvq8_l2sq`, `lvq8_dot`, `lvq8x8_l2sq`, `lvq8x8_l2sq_primary` —
  asymmetric distance kernels (fp32 query, int8 + per-vector scalars
  database).
* `FlatF32`, `FlatLvqIndex`, `IndexKind`, `SearchHit` — brute-force
  baseline and reranking-friendly index used both for ground-truth
  comparisons and as the integration target for higher-level graphs.
* `LvqError` — typed error enum (`DimMismatch`, `NonFinite`,
  `KTooLarge`, `Empty`, `AlreadyBuilt`).

**Key constraints honoured:**

* `#![forbid(unsafe_code)]` at the crate root.
* Pure-Rust, deterministic across architectures (no platform-specific
  intrinsics; the compiler auto-vectorises the inner loops).
* All files < 500 lines (largest is `index.rs` at 297 LOC).
* No mocked benchmarks — every number in the research doc comes from a
  real `cargo run -p ruvector-lvq --release --bin ruvector-lvq-bench`.
* Workspace-friendly — added to `members`, not `exclude`; default
  build under `cargo build --workspace` is unaffected.

The crate is *not* yet wired into `ruvector-core`'s HNSW or
`ruvector-diskann`. That integration is deliberately out-of-scope for
this ADR; it is enumerated as the immediate next step in the research
doc's "What to improve next" section.

## Consequences

**Positive**

* New (memory ÷ recall) tradeoff point available to ruvector users:
  **27% of fp32 memory at recall@10 ≥ 0.94** (LVQ-8 alone), or
  **55% at recall@10 = 1.000** (LVQ-8x8 with 10× rerank).
* Reranking API matches the standard "coarse → fine" pattern, so the
  crate plugs into any graph index with a single distance-callback
  swap.
* Establishes the design vocabulary (per-vector stats, residual level,
  asymmetric distance) that LeanVec, asymmetric int8 SIMD, and on-disk
  block formats will reuse.
* No `unsafe` and no platform intrinsics → identical results across
  x86_64, ARM64, and WASM (when a `-wasm` sister crate lands).

**Negative / costs**

* **Brute-force scan latency does not improve at small scale.** The
  benches show LVQ-8 is ~22% *slower* than the fp32 baseline at
  `n=200K, d=128` on Apple M4 Max because the f32 baseline is already
  SIMD-bound and the LVQ kernel reconstructs floats from byte codes.
  The expected QPS win materialises only above L2 cache pressure
  (≥1 M vectors at high-d) and inside graph indexes; this needs to be
  communicated clearly so users do not expect a speedup at 50 K
  vectors.
* +1 crate in the workspace, +12 bytes of per-vector overhead for the
  stats triple.
* Build time: cold `cargo build -p ruvector-lvq --release` adds ~3 s
  on M4 Max. Negligible at workspace scale.
* Persistence (rkyv on-disk format) and the Node/WASM bindings are
  follow-on work; this ADR does not block them but does not deliver
  them.

## Alternatives considered

1. **Add LVQ as a feature flag on `ruvector-core`.** Rejected: the
   distance-kernel surface is large enough to deserve its own crate,
   and a standalone crate is easier to depend on from `diskann`,
   `rabitq` reranking pipelines, and the future `ruvector-lvq-wasm`.
2. **Use scalar `SQ8` (global scale + global bias).** Rejected: a
   global scale forces precision loss on small-magnitude vectors when
   the dataset has any high-magnitude outliers, which is the common
   case for LLM embeddings. SOTA papers consistently show LVQ
   dominates SQ8 at the same bit budget.
3. **Use Product Quantization (PQ).** Already represented in the
   ecosystem (Milvus, FAISS). PQ excels at extreme compression but its
   training step (k-means per subspace) is non-trivial and its
   reranking story is worse — LVQ's per-vector approach has *no*
   training step and gives perfectly reproducible codes from
   construction time forward. Both are useful; this ADR adds the
   missing one.
4. **Wait until SVS publishes a Rust port.** Rejected: SVS is C++ and
   the upstream team has not signalled Rust support. A clean-room
   Rust implementation (this PoC) is more aligned with ruvector's
   `forbid(unsafe_code)` posture and unblocks downstream WASM/embedded
   use immediately.

## Verification

* `cargo build -p ruvector-lvq --release` — succeeds.
* `cargo test -p ruvector-lvq --release` — **10/10 tests pass** (3
  unit + 4 module + 3 integration).
* `cargo run -p ruvector-lvq --release --bin ruvector-lvq-bench` —
  prints memory + latency + recall numbers reproduced verbatim in the
  research document.
* Recall acceptance bars baked into tests:
  * LVQ-8: `recall@10 > 0.90`
  * LVQ-8x8 reranked (10×): `recall@10 > 0.98`
  * Two-level residual L2 error < 25% of single-level
  * LVQ-8 byte footprint < 30% of fp32

## Follow-ups

* Wire `lvq8_l2sq` into `ruvector-core::hnsw` as a selectable distance
  backend (separate ADR; expected 2026-05).
* Wire LVQ codes into `ruvector-diskann` block format.
* Add `ruvector-lvq-wasm` and `ruvector-lvq-node` mirror crates.
* Asymmetric int8 SIMD kernels via `simsimd`.
* LeanVec orthogonal-projection front-end on top of `Lvq8`.
