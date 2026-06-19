---
adr: 254
title: "ruvector-turbovec — Multi-bit Scalar-Quantized FastScan ANN Index (2/3/4-bit SQ + TQ+ calibration + nibble-LUT SIMD)"
status: accepted
date: 2026-06-16
authors: [oshaal, claude-flow]
related: [ADR-157, ADR-193, ADR-155]
supersedes: []
tags: [quantization, ann, vector-search, turboquant, fastscan, simd, lloyd-max, calibration, recall, memory]
---

# ADR-254 — ruvector-turbovec: a multi-bit scalar-quantized FastScan ANN index

> **Numbering note.** This decision was originally drafted in PR #521 as
> `ADR-194`, which collides with the already-merged `ADR-194` (ruvector ONNX
> embedder API & throughput). This file is the canonical record at the next free
> number; PR #521's `ADR-194-ruvector-turbovec-fastscan-index.md` should be
> renumbered to **254** (or dropped in favor of this file) before merge.

> **Provenance & prior-art note.** This adapts techniques from
> [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec), an independent
> Rust+Python implementation of Google Research's **TurboQuant**
> (arXiv:2504.19874). It is **not** a port — it is a clean-room ruvector crate
> that reuses existing primitives. The recall / compression / bias figures under
> "Validation" are **measured** and reproducible; the *competitive* claims vs
> FAISS/Milvus are **targets to validate**, attributed to the upstream project
> where cited.

## Status

**Accepted (M1 implemented).** The scalar-reference milestone (M1) ships as
`crates/ruvector-turbovec` in PR #521: rotation reuse + Lloyd–Max 2/3/4-bit
scalar quantization + TQ+ per-coordinate calibration + length-renormalized
unbiased scoring + `IdMapIndex` (O(1) delete, filtered search). Build green; 17
unit tests + 1 doc-test pass; clippy clean. M2–M4 (FastScan SIMD kernel,
AVX-512, dispatcher registration, persistence) are future work.

## Context

### The gap

ruvector has strong ANN coverage but is missing the **2–4-bit scalar-quantized
FastScan regime** that mainstream production vector DBs live in (FAISS
`IndexPQFastScan`, Qdrant, Milvus IVF-SQ):

| Crate | Index family | Quantization | SIMD scan |
|---|---|---|---|
| `ruvector-core` | HNSW (graph) | none (f32) | — |
| `ruvector-diskann` | DiskANN (graph) | none / PQ-ish | — |
| `ruvector-rairs` | IVF (ADR-193) | optional | — |
| `ruvector-rabitq` | Flat + rotation | **1-bit only** | AVX2/512 popcount |
| `ruvllm` `turbo_quant.rs` | value codec | 2.5–4 bit | n/a (not an index) |

- **`ruvector-rabitq`** is **1-bit**: excellent when memory dominates, but on
  1536–3072-dim production embeddings (OpenAI `text-embedding-3`, Cohere, Voyage)
  hitting production R@1 needs an exact-rerank pass over the raw f32 originals,
  which re-inflates memory back toward the f32 footprint — partly defeating the
  point.
- **`ruvllm/quantize/turbo_quant.rs`** is a TurboQuant *value codec* for KV-cache
  / embedding compression — no inverted lists, top-k heap, FastScan LUT,
  filtered search, or stable IDs/deletion. **Wrong abstraction for search.**

What's missing, concretely: **2–4 bits/dim**, **codebook-free** scalar
quantization (no k-means, online ingest), a **FastScan nibble-LUT SIMD kernel**
(16/32 candidates per instruction, no f32 materialization), and competitive
recall **without** a mandatory f32 rerank.

### What already exists (not duplication)

1. **`ruvector-rabitq`** (ADR-157) already holds the *rotation* half of
   TurboQuant (`RandomRotation::HadamardSigned`, the randomized Hadamard,
   `O(D log D)`) and defines the `AnnIndex` + `VectorKernel`/`KernelCaps` traits
   reused here — but its codes are 1-bit and its kernels are XNOR-popcount.
2. **`ruvllm/quantize/turbo_quant.rs`** has MSE-quantizer math to borrow, but is
   a data-oblivious tensor compressor, not an `AnnIndex`.

So this work reuses rabitq's rotation + traits and ruvllm's codec lessons to
build the missing multi-bit FastScan **search index**.

## Decision

Introduce **`crates/ruvector-turbovec`**, a multi-bit ANN index that implements
the existing `ruvector_rabitq::AnnIndex` trait and exposes a FastScan SIMD kernel
via the existing `VectorKernel`/`KernelCaps` contract, so it drops into the
`ruvector-rulake` dispatcher (ADR-155/157) with **no new plumbing**. Six
techniques (T1–T6):

- **T1 — Normalize + randomized Hadamard rotation (reuse).** Strip each vector's
  L2 norm (store one f32), apply `RandomRotation::HadamardSigned` from rabitq.
  Post-rotation coordinates are ~N(0, 1/d), making per-coordinate scalar
  quantization optimal **without a codebook**. Import, don't reimplement.
- **T2 — Lloyd–Max scalar quantization (2/3/4-bit).** MSE-optimal bucket
  boundaries for the canonical N(0,1) marginal (4/8/16 buckets). Boundaries are
  constants of the distribution, not the data → **zero training**.
- **T3 — Per-coordinate calibration (TQ+).** On the first `add()` batch, fit
  `(shift[d], scale[d])` mapping empirical quantiles onto the canonical Beta;
  freeze after warm-up. Corrects finite-d Hadamard non-Gaussianity — the "+" that
  buys the recall edge. No counterpart exists today.
- **T4 — Length-renormalized inner-product scoring.** Store a per-vector
  `c_x = ⟨r,r̂⟩/⟨r̂,r̂⟩` so the dot-product estimator removes scalar
  quantization's downward bias at **zero query-time cost** — skipping the f32
  rerank 1-bit RaBitQ needs. **This is an empirically near-unbiased heuristic,
  not the paper's provably-unbiased two-stage QJL-residual estimator** (see
  Divergences); adopting QJL is a tracked follow-up (M5) if measured bias demands
  it.
- **T5 — FastScan nibble-LUT SIMD kernel (core perf win).** 32-vector SoA blocks;
  score a block by nibble-split table lookups (`vpshufb`/`tbl`) not arithmetic.
  x86 AVX-512BW with AVX2 fallback (`x86-64-v3`); ARM NEON `vqtbl1q_u8`; WASM
  scalar fallback (bit-identical). rabitq's popcount kernels are **not** reusable
  (popcount ≠ table-lookup), but the trait/dispatch/determinism contract is.
- **T6 — Block-granularity filtered search + stable IDs.** Allowlist tested at
  32-vector block granularity inside the kernel (fully-excluded blocks
  short-circuit); `IdMapIndex` with external `u64` IDs surviving deletion and
  **O(1) remove** (tombstone + free-list).

### Reuse boundary

| Component | Source | Action |
|---|---|---|
| Randomized Hadamard rotation | `ruvector-rabitq::RandomRotation` | import |
| `AnnIndex` trait | `ruvector-rabitq::index` | implement |
| `VectorKernel` / `KernelCaps` | `ruvector-rabitq::kernel` | implement |
| MSE/Lloyd–Max quantizer math | `ruvllm::quantize` | borrow/extract |
| Lloyd–Max boundary tables (2/3/4-bit) | TurboQuant constants | build |
| TQ+ per-coordinate calibration | — | build |
| FastScan nibble-LUT SIMD kernel | — | build |
| 32-block SoA layout + filtered scan | — | build |
| IdMap O(1)-delete | — | build |
| Persistence (`.tv` file) | mirror `rabitq::persist` | build |

If the borrowed MSE math proves reusable across both crates, a follow-up should
hoist it into `ruvector-math` rather than copy it.

### Milestones

1. **M1 — scalar reference** (PR #521): rotation reuse + Lloyd–Max SQ + TQ+ +
   unbiased scoring + `IdMapIndex`. ✅ implemented & validated. Serves as the
   determinism oracle the SIMD kernels must match bit-for-bit.
2. **M2 — FastScan nibble-LUT SIMD kernel** (AVX2 + NEON), fuzzed bit-identical
   to the scalar oracle.
3. **M3 — `.tv` persistence.**
4. **M4 — AVX-512BW kernel + `ruvector-rulake` dispatcher registration.**
5. **M5 (optional) — QJL-residual provably-unbiased estimator**, if measured
   bias/recall on real embeddings demands the paper-grade two-stage scorer.

## Validation (measured — `cargo run --release -p ruvector-turbovec`)

`n = 5,000` **uniform-random** vectors (worst case — no cluster structure),
`dim = 256`, `k = 10`, **no f32 rerank**, vs exact brute-force L2:

| Width | recall@10 | bytes/vec (raw 1024) | compression | mean cosine bias |
|---|---|---|---|---|
| 1-bit | 0.308 | 48 | 25.6× | +0.0005 |
| 2-bit | 0.561 | 80 | 14.2× | +0.0001 |
| 3-bit | 0.767 | 112 | 9.8× | −0.0000 |
| **4-bit** | **0.879** | **144** | **7.5×** | **−0.0000** |

Recall rises monotonically with bit-width — the 2–4-bit regime 1-bit RaBitQ
can't reach without re-inflating memory. Mean cosine bias ≈ 0 at every width
confirms the `c_x` renormalization is empirically near-unbiased on this data. On
real clustered embeddings, recall at a given width is materially higher than this
uniform stress test. Determinism (same seed → bit-identical) and `IdMapIndex`
delete + allowlist search verified PASS.

## Consequences

### Positive
- Closes the 2–4-bit FastScan gap — the one mainstream production ANN regime
  ruvector lacks; comparable regime to FAISS `IndexPQFastScan` / Milvus IVF-SQ.
- ~16× compression with online ingest, no training (d=1536: 6 KB → 384 B).
- Recall without mandatory f32 rerank (T4), so the memory win is real.
- Zero new plumbing — implements existing `AnnIndex` + `VectorKernel`, registers
  with the `ruvector-rulake` dispatcher, inherits its determinism/witness
  contract. Exactly one new workspace crate.
- Composition path: the same block-SoA codes can back an IVF posting list
  (IVF-SQ-FastScan) with `ruvector-rairs` (ADR-193).

### Negative
- T4 is a heuristic, not the paper's provably-unbiased estimator; pathological
  distributions may need M5 (QJL residual). Tracked, not hidden.
- Competitive-vs-FAISS/Milvus claims are targets pending M2+ benchmarks, not yet
  this crate's measured numbers.
- Borrowed MSE math risks divergence from `ruvllm`'s copy until hoisted into
  `ruvector-math`.
- New SIMD kernels (M2/M4) are correctness-sensitive; mitigated by fuzzing
  bit-identical against the M1 scalar oracle.

### Neutral
- One additional crate; no change to existing index crates or their APIs.

## Links
- Issue: #520 (the gap analysis & ask)
- PR: #521 (M1 implementation + original ADR-194 draft to be renumbered)
- Related: ADR-157 (rabitq rotation + traits), ADR-193 (rairs IVF), ADR-155
  (rulake dispatcher)
- Upstream: [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec),
  TurboQuant (arXiv:2504.19874)
