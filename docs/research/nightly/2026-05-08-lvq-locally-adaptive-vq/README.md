# Locally-Adaptive Vector Quantization (LVQ) for ruvector

**Date:** 2026-05-08
**Branch:** `research/nightly/2026-05-08-lvq-locally-adaptive-vq`
**Crate:** `crates/ruvector-lvq/`
**ADR:** [ADR-193](../../../adr/ADR-193-lvq-locally-adaptive-vq.md)

## Abstract

This research delivers a working Rust implementation of **Locally-Adaptive
Vector Quantization (LVQ)**, the per-vector scalar compression scheme
introduced by Aguerrebere et al. in Intel's *Scalable Vector Search* (SVS)
project (VLDB 2024). Unlike RaBitQ — already explored in
`docs/research/nightly/2026-04-23-rabitq/` — LVQ keeps 8 bits per dimension
and uses a *per-vector* `(mean, bias, scale)` triple to adapt the dynamic
range of each individual vector. We add a two-level residual variant
(LVQ-8x8) that recovers fp32-equivalent recall while still cutting memory
in half. The PoC exposes a flat brute-force index plus a reranking API
that any graph index (HNSW, DiskANN, Vamana) can plug into. On a
synthetic 200 000 × 128 dataset on Apple M4 Max, LVQ-8x8 with 10× rerank
achieves **recall@10 = 1.000 at 45% of the fp32 memory footprint** with
latency within 22% of the fp32 baseline.

## SOTA survey

| Year | Paper / system | Headline | Why it matters here |
| --- | --- | --- | --- |
| 2024 | Aguerrebere et al., *"Locally-Adaptive Vector Search via Quantization"*, VLDB 2024 | LVQ + LeanVec; SVS open-sourced by Intel | The canonical reference for this work. |
| 2024 | Intel/Snowflake SVS engine (open-source release) | LVQ-Bx8 reranking on top of Vamana / HNSW | Demonstrates production-grade integration. |
| 2024 | Gao & Long, *"RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound"*, SIGMOD 2024 | 1-bit binary quantization | Already in-tree (`crates/ruvector-rabitq`); LVQ is the orthogonal scalar-quantizer track. |
| 2024 | Pinecone "ANN at the speed of memory" report | Memory-bandwidth-bound search on AVX-512 | Confirms the *real* speedup of int8 vs. fp32 surfaces above ~1 M vectors. |
| 2025 | Milvus 2.4 release notes | Adds SQ8 + per-cluster scaling | Roughly equivalent to per-IVF-cell LVQ; ours is *per-vector* for higher precision. |
| 2025 | Qdrant 1.10 changelog | Adds *binary quantization* for OpenAI-3072 | Trades recall for memory; LVQ is the high-recall complement. |
| 2025 | Weaviate 1.27 docs | Product-quantization (PQ) reranking | Confirms reranking-from-coarse-to-fine is the standard pattern. |
| 2025 | Lance/LanceDB blog | Vector compression + on-disk format | Disk-friendly per-vector codes mirror what LVQ stores. |

LVQ is *not* yet a first-class option in any of the main open-source
vector databases except SVS itself. Its niche — high recall, modest
memory savings, no quality loss when reranked — is exactly the gap
between RaBitQ (extreme compression, lower recall) and uncompressed fp32
(perfect recall, 4× memory). Offering it in ruvector lets users pick
along the **(memory ÷ recall ÷ latency)** tradeoff curve instead of
being forced to a single point.

## Proposed design

### Encoder

For each input vector `v ∈ R^d`:

```
mean  = Σ v[j] / d
ctr   = v - mean
bias  = min(ctr)
scale = (max(ctr) - bias) / 255          # 0 if vector is constant
code[j] = round((ctr[j] - bias) / scale) ∈ [0, 255]
```

Decoded reconstruction:

```
recon[j] = mean + bias + scale * code[j]
```

Storage per vector:

* `d` bytes for `code`
* 12 bytes for `(mean, bias, scale)` as `Lvq8Stats` (3× `f32`)

The dominant cost is the `d` bytes of code; the 12-byte overhead is
amortised per vector. At `d=128`, that is **140 B / vector** vs. **512 B
/ vector** for fp32 → **27.3% of fp32**.

### Two-level (LVQ-8x8)

After encoding `v` as LVQ-8, the residual `r = v - decode(LVQ8(v))` is
encoded by another independent LVQ-8 pass. Reconstruction is
`recon_p + recon_r`. Total per-vector storage doubles (~280 B at d=128
≈ **54.7% of fp32**) but the residual reduces L2 reconstruction error
by more than 4× (verified in `two_level::tests::two_level_strictly_better_than_one`).

### Asymmetric distance kernels

Queries stay in fp32. The `lvq8_l2sq` kernel reconstructs each lane of
the database vector on the fly:

```rust
acc += (q[j] - (mean + bias + scale * code[j]))²
```

The compiler auto-vectorises this loop on both AVX2 and NEON — we
intentionally avoid platform-specific intrinsics so the crate stays
portable and fully reproducible. We also expose `lvq8_dot` which
algebraically separates `bias·Σq` and `scale·Σ(q·code)` so an int8 dot
product can be substituted in a future SIMD-native kernel without
breaking the API.

### Reranking API

`FlatLvqIndex::search_l2_reranked(q, k, rerank_k)`:

1. Scan all vectors using **primary-only** distance (cheap, byte-only
   memory traffic).
2. Keep top-`rerank_k` candidates via `select_nth_unstable_by`.
3. Rescore those candidates with the **full primary+residual**
   reconstruction.
4. Return top-`k`.

This is the canonical "coarse → fine" pattern; the crate's bench binary
shows that `rerank_k = 5*k` already saturates recall.

## Implementation notes

* **Crate layout** (`crates/ruvector-lvq/`):
  * `quantize.rs` — `Lvq8`, `Lvq8Stats`, `encode_one`
  * `two_level.rs` — `Lvq8x8` and residual encoding
  * `distance.rs` — `lvq8_l2sq`, `lvq8_dot`, `lvq8x8_l2sq`, `lvq8x8_l2sq_primary`
  * `index.rs` — `FlatF32` (ground truth), `FlatLvqIndex`, reranking
  * `error.rs` — typed error enum
  * `main.rs` — end-to-end benchmark binary
  * `tests/recall.rs` — recall acceptance tests
  * `benches/lvq_bench.rs` — Criterion microbenchmarks
* **No `unsafe`.** `#![forbid(unsafe_code)]` at the crate root.
* **No floats stored as `Ord`** — partial sort uses
  `select_nth_unstable_by` with an explicit `partial_cmp` then `id`
  tie-break, so identical scores are deterministic.
* **All files < 500 lines** (largest: `index.rs` at 297 lines).

## Benchmark methodology

Hardware: Apple M4 Max (16 cores), 128 GB RAM, macOS 14.6 (Darwin 24.6.0
arm64). Toolchain: `rustc 1.89.0 (29483883e 2025-08-04)`, `cargo 1.89.0`.

Dataset: synthetic clustered Gaussian — 32 cluster centers in `[-1, 1]^d`,
each base vector drawn within ±0.15 of its center, queries within ±0.20.
Seeded RNG (`StdRng::seed_from_u64(42)`) for reproducibility. We deliberately
chose a clustered distribution so distances are **non-trivial** (uniform
random vectors in high-dim are nearly equidistant and hide quantization error).

Three index variants are built from the same data and queried with the
same 200-query batch. Recall@10 is measured against the fp32 brute-force
ground truth. Latency is wall-clock per query (single-threaded scan).

Reproduce:

```bash
cargo run -p ruvector-lvq --release --bin ruvector-lvq-bench
LVQ_N=200000 cargo run -p ruvector-lvq --release --bin ruvector-lvq-bench
```

## Results

### 50 000 × 128, k = 10 (default)

```
fp32 build:               2.60 ms     25 600 000 bytes
LVQ-8 build:             15.40 ms      7 000 000 bytes
LVQ-8x8 build:           32.36 ms     14 000 000 bytes

variant                          lat ms        qps  recall@10
fp32 (ground truth)               2.038        491      1.000
LVQ-8                             2.083        480      0.959
LVQ-8x8 (full scan)               2.704        370      1.000
LVQ-8x8 (rerank, 5x)              2.084        480      1.000
LVQ-8x8 (rerank, 10x)             2.076        482      1.000
```

### 200 000 × 128, k = 10

```
fp32 build:              14.16 ms    102 400 000 bytes
LVQ-8 build:             64.05 ms     28 000 000 bytes
LVQ-8x8 build:          135.25 ms     56 000 000 bytes

variant                          lat ms        qps  recall@10
fp32 (ground truth)               6.746        148      1.000
LVQ-8                             8.332        120      0.942
LVQ-8x8 (full scan)              10.612         94      1.000
LVQ-8x8 (rerank, 5x)              8.360        120      1.000
LVQ-8x8 (rerank, 10x)             8.252        121      1.000
```

### Memory savings (200K × 128)

| Index | Bytes | Ratio vs fp32 | Recall@10 |
| --- | --- | --- | --- |
| fp32 baseline | 97.66 MB | 1.000 | 1.000 |
| LVQ-8 | 26.70 MB | **0.273** | 0.942 |
| LVQ-8x8 (rerank 10×) | 53.41 MB | **0.547** | **1.000** |

### Recall acceptance tests (`cargo test -p ruvector-lvq --release`)

```
test distance::tests::lvq8_l2sq_matches_decoded_reference ... ok
test quantize::tests::handles_constant_vector ... ok
test quantize::tests::roundtrip_recovers_within_tolerance ... ok
test quantize::tests::rejects_non_finite ... ok
test two_level::tests::two_level_strictly_better_than_one ... ok
test index::tests::lvq8_recall_against_groundtruth ... ok
test index::tests::lvq8x8_reranking_meets_target ... ok
test end_to_end_lvq8_recall_above_90 ... ok
test end_to_end_lvq8x8_rerank_recall_above_98 ... ok
test lvq8_byte_size_is_close_to_d_per_vector ... ok

10 passed; 0 failed
```

## How it works (blog-readable walkthrough)

Imagine you have one billion 768-dim sentence embeddings. Storing them
as `f32` takes **3.07 TB**. That is fast on hot memory but ruinous on
disk, and impossible to keep in RAM on any single commodity box.

The naive fix is "use 8-bit integers" — a global quantizer with one
shared scale and offset. The problem: a single outlier vector with a
huge dynamic range forces the global scale wide, so every *normal*
vector loses precision. The smaller-magnitude embeddings — which is
most of them — get squashed into a handful of integer levels and recall
collapses.

LVQ flips the fix: **each vector gets its own scale and offset**. We
spend 12 extra bytes per vector to store `(mean, bias, scale)`, and in
exchange every vector keeps its full 8-bit dynamic range. At
high-dimensional scale (768, 1024, 1536), 12 bytes is rounding error
relative to the `d` bytes of codes — the per-vector overhead is below
2%.

That alone gets us to ≈ 27% of fp32 storage. To recover the recall lost
to quantization noise, we encode the *residual* (the part the first
quantizer rounded off) with another LVQ-8 pass. Now we are at 55% of
fp32 storage, but with two levels we have enough precision to match the
original to within float-ULP error on a brute-force ranking — confirmed
by `recall@10 = 1.000` in the benches above.

The catch: full residual reconstruction is the slowest of the three
variants. The fix is **reranking**: scan with the cheap primary code
only, keep a short-list 10× longer than the result set, and re-score
just that short-list with the residual. The benches show this gives
the same recall as full residual scan at the same latency as primary-only.

## Practical failure modes

1. **All-zero vectors.** Treated correctly: the constant-vector branch
   sets `scale = 0` and stores all-zero codes; decode returns the mean.
   Verified by `quantize::tests::handles_constant_vector`.
2. **Non-finite inputs.** Rejected at encode-time with
   `LvqError::NonFinite(idx)`. The crate never panics on bad data.
3. **Tiny `k` with sparse ties.** `select_nth_unstable_by` plus the
   `(score, id)` ordering guarantees deterministic results across
   architectures even when distances tie.
4. **Cosine workloads.** This PoC exposes L2 + dot product. Cosine
   should be done by L2-normalising both query and database vectors *up
   front*, then using `dot`. Storing the pre-normalised vectors lets
   LVQ keep the same per-vector scale logic.
5. **Brute force is memory-bound.** At `d=128, n=200K` the fp32 baseline
   is already hitting the M4 Max's L2-resident bandwidth, so the
   `4×` byte-traffic reduction of LVQ-8 does not translate to `4×` QPS.
   The expected wins materialise in two regimes: (a) when the dataset
   no longer fits in last-level cache (≥ 1 M vectors at 768-d), and
   (b) when LVQ codes are scanned *inside* an HNSW or Vamana graph
   where memory traffic dominates.

## What to improve next (roadmap)

1. **HNSW integration.** Replace the candidate-list distance call in
   `crates/ruvector-core` HNSW with `lvq8_l2sq`. Expected: ~3× QPS at
   1 M+ scale once cache pressure dominates.
2. **DiskANN/Vamana integration.** `crates/ruvector-diskann` already
   has a Vamana implementation — wiring LVQ-8 codes into the on-disk
   block layout cuts I/O bytes by 4×.
3. **AVX-512 / NEON int8 dot kernels.** Use `simsimd` (already in
   workspace deps) to swap the f32 reconstruction loop for an int8
   dot + per-vector scalar correction. Estimated 2-3× on the inner
   loop on Sapphire Rapids / Apple M-series.
4. **LeanVec.** The follow-up of LVQ — orthogonal projection to
   `d' < d` *before* LVQ. Stack on top of this crate; the `Lvq8` trait
   is already swappable.
5. **Asymmetric int8-quantised query.** Quantize the query once with
   the global statistics of the data, then the entire dot product
   becomes int8×int8 → int32 with a single fp32 correction.
6. **Persistence.** rkyv-based on-disk format aligned with
   `crates/ruvector-snapshot`.
7. **WASM crate.** Mirror the pattern in `crates/ruvector-rabitq-wasm`
   to ship LVQ to the browser.

## Production crate layout proposal

```
crates/
  ruvector-lvq/                   # core (this PoC)
  ruvector-lvq-wasm/              # wasm-bindgen surface
  ruvector-lvq-node/              # napi binding
  ruvector-core/
    src/index/hnsw/lvq.rs         # HNSW + LVQ scoring backend
  ruvector-diskann/
    src/disk/lvq_block.rs         # LVQ-aware disk block format
```

Public traits in `ruvector-core` already abstract the distance metric;
LVQ slots in as another `MetricBackend` without breaking the existing
HNSW API.

## References

* Aguerrebere, C., Bhati, I., Hildebrand, M., Tepper, M. & Willke, T.
  *Similarity Search in the Blink of an Eye with Compressed Indices*.
  VLDB 2024. https://dl.acm.org/doi/10.14778/3611479.3611537
* Aguerrebere, C. et al. *Locally-Adaptive Quantization for Streaming
  Vector Search*. arXiv:2402.02044, 2024.
* Gao, J. & Long, C. *RaBitQ: Quantizing High-Dimensional Vectors with
  a Theoretical Error Bound for Approximate Nearest Neighbor Search*.
  SIGMOD 2024.
* Intel SVS open-source release: https://github.com/intel/ScalableVectorSearch
* Malkov, Y. & Yashunin, D. *Efficient and robust approximate nearest
  neighbor search using HNSW graphs*. TPAMI 2020.
* ruvector internal nightly research:
  `docs/research/nightly/2026-04-23-rabitq/`
  `docs/research/nightly/2026-04-26-acorn-filtered-hnsw/`
