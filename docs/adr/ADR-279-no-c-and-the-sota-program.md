---
adr: 279
title: "No C in the Core; and the 2026 SOTA Program for Vector Search"
status: accepted
date: 2026-08-02
authors: [Reuven Cohen]
project: "RuVector Core"
related: [ADR-264, ADR-267, ADR-268, ADR-272]
supersedes_parts_of: [ADR-267]
tags: [ruvector, performance, simd, ffi, wasm, ann, quantization, benchmarks, sota]
---

# ADR-279 — No C in the Core; and the 2026 SOTA Program

## Status

**Accepted.** Owner: Reuven Cohen. Date: 2026-08-02.

Answers two questions asked together: *would adding C improve RuVector's
performance and capabilities?* and *what would actually make it state of the
art?* The answers turn out to be independent — and the second is far more
consequential than the first.

## 1. Decisions

1. **No C/C++ in the default build graph.** The premise does not survive 2026
   evidence, and the costs land precisely where RuVector is most exposed.
2. **Retarget the SOTA harness from ann-benchmarks to VIBE, and add
   `1/Ratio@k`.** This is the highest-priority item in this ADR and blocks
   every performance claim.
3. **Adopt a ranked SOTA program** (§5), led by 8-bit rotational quantization
   and a SymphonyQG-class packed quantized graph.
4. **Bind, don't rewrite, for GPU.** `cuvs-sys` for CAGRA build; no pure-Rust
   CAGRA.
5. **Differentiate on streaming stability**, which is the least crowded
   frontier and happens to be our actual workload.

## 2. The C question: no

### 2.1 The premise fails

Every capability commonly cited as C-only has a production Rust path in 2026:

| Claimed C-only | 2026 reality |
|---|---|
| io_uring | `io-uring` crate is pure Rust; `bindgen` optional, bindings checked in |
| CUDA | `cudarc` defaults to dynamic loading — no toolkit at build time |
| BLAS/LAPACK | `faer` matches or surpasses OpenBLAS/LAPACK/Eigen |
| AVX-512 | Stable in `std::arch` since Rust **1.89**; **FP16 since 1.94** (Mar 2026) |
| NUMA / hugepages | Syscalls via `libc` — FFI *declarations*, not compiled C |
| RocksDB | Qdrant **removed** it |
| FAISS / hnswlib | Binding it forecloses filtered-search-inside-traversal |
| simdjson | `simd-json` supports AVX2/SSE4.2/NEON **and wasm simd128** natively |

Two results are decisive rather than merely suggestive:

- **`zlib-rs` is faster than zlib-ng in C on native, and is the fastest WASM
  zlib in existence** (2× miniz-oxide). A direct counterexample to "C is
  faster", on exactly the axis proposed.
- **Qdrant — the closest analogue to RuVector — spent two minor versions
  removing its one C++ dependency**, naming *"interoperating with C++ slowed us
  down"*, plus compaction latency spikes and tuning burden. The comparable
  project went the opposite direction from the proposal.

### 2.2 The cost lands on WASM, which is our largest commitment

37 of 166 crates are WASM. The rustc platform-support book states plainly that
`wasm32-unknown-unknown` **has no C/C++ toolchain** — not "awkward", absent by
design. Consequences:

- Any unconditional `cc::Build` in the graph breaks the browser build.
- Escape hatches don't help: emscripten produces an app, not a
  `wasm-bindgen` library; `wasm32-wasip1` gives edge/server but not browser.
- The `extern "C"` ABI on wasm is *mid-migration* (future-incompat warning
  since Rust 1.87), so any FFI boundary there stands on changing ground.
  Rust-to-Rust is unaffected.
- Every C feature must be reimplemented in Rust for the browser anyway. **You
  pay for each feature twice and get behavioral divergence for free.**

### 2.3 Costs beyond WASM

- **Miri cannot execute across an FFI boundary.** For a database with
  concurrent index structures, losing Miri on the hot paths is a real
  regression in verification capability.
- A 320-bug study of bindgen/cbindgen/CXX (ACM TOSEM, Feb 2026) found the
  dominant failure mode is **not** crashes but *silently generating code
  unfaithful to intent*, rooted in data-layout mismodeling.
- `cargo-audit`/`cargo-vet` give **no meaningful coverage of vendored C**
  (cf. RUSTSEC-2023-0061, libwebp).
- The FFI boundary is a *pessimization* on fine-grained work: call overhead and
  lost cross-module inlining make a C distance kernel slower than the
  equivalent `std::arch` kernel. Distance kernels are the definition of
  fine-grained.

### 2.4 The headline "C is faster" number is a datatype artifact

SimSIMD/NumKong is the most-cited evidence for C kernels, claiming **20–118×**
over autovectorized code. Decomposed, that table compares **NumKong f16 against
GCC's f32** — there is no f32-vs-f32 row, and the README states outright that
GCC "struggles with `_Float16`". Against the compiler's competent f32 output the
real effect is **1.15×–2.1×**. The 3–200× figures elsewhere are against
SciPy/NumPy Python, not compiled code.

It is also independently contradicted. **PDX (SIGMOD '25) beat SimSIMD's and
FAISS's hand-written kernels by 2.0× on average using plain scalar C++ with no
intrinsics at all**, purely by changing data layout — 3–10× at low dimension.
And an independent harness found NumKong *loses* bulk scoring by **1.85–3.04×**,
because it has no bulk API and cannot hide memory latency.

**The lever is layout and API shape, not the language the intrinsics are typed
in.** Two expert C teams differ from each other by more (0.72–1.20×) than Rust
differs from C.

### 2.5 Qdrant ships zero C

A code search for `simsimd` in `qdrant/qdrant` returns **0 hits**. Qdrant has
AVX-512 `vpopcntq`, `pshufb`/`maddubs`/`VPDPBUSD`, and NEON `SDOT` quantization
kernels — all in Rust `std::arch` — and shipped an ICLR-2026 quantizer
(TurboQuant, v1.18, May 2026) before most C++ engines had it. FAISS, in C++,
still lacks AVX-512 FastScan.

Separately: plain **23-line Rust with no intrinsics beat the `simsimd` C crate
on Hamming distance** across three machines — autovectorization winning over
manual SIMD on precisely the binary-quantization inner loop.

### 2.6 The one real gap closes in 18 days

The largest historically-measured Rust-vs-C gap in this workload was **f32
reduction reassociation: 8.4×** (84 µs vs 10 µs with Clang fast-math).
`float_algebraic` (`f32::algebraic_add`/`algebraic_mul`) stabilized in PR
#157029, **shipping in Rust 1.98 on 2026-08-20**. Adopt it the day it lands.

What remains genuinely blocked on stable Rust is narrower than assumed:
**f16/bf16 arithmetic** (tracking #116909) and **ARM SVE/SME** (#145052).
Note that even LanceDB — the one major Rust vector DB linking C — uses it for
three files, and **two of them are pragma-autovectorized C, not hand-written
intrinsics**. What C bought Lance was `_Float16` and fast-math, and one of those
two arrives on 2026-08-20.

For f16 the cheaper answer is to store f16 and upconvert with stable
`_mm512_cvtph_ps`/`vcvt_f32_f16`, accumulating in f32 at ~1.15–2.1× — or skip
f16 entirely for int8/binary quantization, which is faster *and* smaller.

### 2.7 The existence proof is already in this repo

`ruvector-rabitq/src/scan.rs` performs RaBitQ scanning with
`_mm512_popcnt_epi64` and `_mm512_xor_si512`, runtime-detected via
`avx512vpopcntdq`, with an AVX2 fallback — hand-written Rust `std::arch`. That
is precisely the work C would have been imported to do, already done, and it
compiles for WASM.

Note a correction to an earlier survey of this repo: an initial pass concluded
there were "no binary/hamming popcount kernels". That was wrong — it
generalized from one file (`ruvector-core/simd_intrinsics.rs`) across a
166-crate workspace. The kernel exists.

### 2.8 The narrow exceptions

C is acceptable **only** when all hold: behind a non-default feature; in a
separate crate the core does not depend on; with a correctness-equivalent
pure-Rust fallback under differential test; genuinely unreachable in Rust; and
with WASM CI proving the core still builds `--no-default-features`.

Realistically that means **FIPS-validated crypto** (a compliance case, not a
performance one) and **vendor accelerator SDKs**. `hailort-sys` is already
exactly this, correctly isolated. `cuvs-sys` (§5) qualifies under the same rule.

## 3. The finding that matters more than C


**`ruvector-sota-bench` measures against `ann-benchmarks.com`, which is
deprecated.** Its README now reads: *"no longer actively maintained… consider
submitting your work to different benchmarks, such as VIBE."* Our dataset list
is SIFT-128, GloVe-25/100, Deep-image-96 — precisely the sets VIBE was built
because they are *"no longer representative of the current applications of ANN
search."*

ADR-267 (SOTA Validation Protocol) does not mention VIBE.

So: **every SOTA claim RuVector could make today rests on an unmaintained
artifact and non-representative data.** The instrument is pointed at the wrong
thing, and no amount of kernel work fixes that. This is the same failure mode as
withdrawing the SWE-bench-Verified gate in ADR-277 §5.1 — a benchmark can retire
underneath you, and continuing to cite it is how retractions happen.

### 3.1 Recall@k is itself under credible attack

Two 2026 papers argue the field has optimized the wrong objective. *ANN Search:
Recall What Matters* (arXiv:2606.04522) proposes **`1/Ratio@k`** and measures
**1.86×–9.36× fewer distance computations** to reach equal downstream quality
versus optimizing Recall@0.95. Downstream validation: image-classification
label precision held at 0.943–0.978 while Recall fell 1.0 → 0.4, and RAG answer
quality varied ≤5% across the same Recall range — `1/Ratio@k` tracked with MAD
0.5–2.6% against Recall's ~29%.

If that reproduces, a large fraction of tuning effort industry-wide is spent
recovering near-equidistant, semantically irrelevant neighbors.

## 4. What a defensible claim requires

Adopted as an amendment to ADR-267:

- **VIBE datasets, including the out-of-distribution splits.** OOD is a
  first-class axis; HNSW's OOD gap is one of its known weaknesses.
- **Both `Recall@k` and `1/Ratio@k`.**
- **QPS at fixed recall on the Pareto frontier**, never single points.
- **Index build time and peak RSS**, reported alongside.
- **Full hardware disclosure** — exact CPU, cores, ISA level. AVX-512 vs AVX2
  vs NEON *changes rankings*.
- **Single-thread and multi-thread separately.**
- **Hyperparameter search budget disclosed for all baselines**, not just ours.
- **Self-published gists and unreproduced blog numbers are not evidence** —
  including our own. The research sweep explicitly excluded ruvnet-authored
  gists on SymphonyQG/MUVERA/RVQ for this reason, and that discipline stands.
- **Do not headline SIFT1M/GIST1M.** Leading there proves nothing in 2026.

## 5. The SOTA program, ranked

| # | Item | Measured effect | Effort | Needs C/CUDA |
|---|---|---|---|---|
| 1 | **Retarget to VIBE + `1/Ratio@k`** | Prerequisite for every claim; 1.86–9.36× wasted compute at stake | 2–3 wks | No |
| 2 | **8-bit rotational quantization** as default codec | >99% recall10@10, 4× compression, ~2.3× faster, **zero training** | **Days** | No |
| 3 | **SymphonyQG-class packed quantized graph** | 1.5–4.5× QPS vs best baselines, 3.5–17× vs hnswlib @95%; best hard-query robustness in VIBE | 2–3 mo | No — `std::arch` VPSHUFB |
| 4 | **Streaming-stable quantizer + dual hot/stable index** | Constant recall under drift vs progressive decay | 6–10 wks | No |
| 5 | **Adaptive filtered-query router** | Specialized-vs-general gap up to 10×, and it *inverts* with scale | 4–6 wks | No |
| 6 | **MUVERA FDE for multi-vector** | ~10% higher recall at 90% lower latency vs PLAID; 32× FDE compression | 3–4 wks | No |
| 7 | **CAGRA build → HNSW serve** via `cuvs-sys` | 12.3× build speedup; no GPU at serve time | 2 wks | **Bind cuVS** |

**Item 2 has by far the best ratio in this document** — a rotation plus scalar
quantization, no clustering, no training, days of work. Do it while item 3 is
in flight.

**Item 4 is where we can lead rather than catch up.** Every incumbent is weak:
SPFresh cannot update stably under contention, FreshDiskANN *"fails to maintain
a graph of great quality because streaming data destroys the navigability of the
original index"*, and **all** data-dependent quantizers — PQ, LVQ, RaBitQ,
ScaNN — are trained on a snapshot and go stale. For an agent memory database
this is not a nice-to-have, it is the workload. The relevant papers are from
Dec 2025 – Jun 2026 and none are reproduced.

### 5.1 Kernel build order (where the SIMD evidence lands)

Two independent research threads converged on the same conclusion: **layout is
the lever, not intrinsics.** PDX got 2.0× from layout alone with no intrinsics;
SymphonyQG's win is also fundamentally a layout change (RaBitQ codes
co-located with neighbor IDs, FastScan-packed). Order accordingly:

1. **PDX-style vertical/blocked layout** — 2.0× average, 3–10× at low dimension
2. **Bulk kernel APIs** (1 query × N vectors) that amortize dispatch and hide
   memory latency — worth 1.85–3.04× on its own, and the specific reason
   NumKong loses bulk scoring despite better single-pair kernels
3. **Binary/int8 quantized kernels** in Rust `std::arch` (`vpopcntq`, `pshufb`,
   `VPDPBUSD`, `SDOT`) — all stable since 1.89
4. **`algebraic_*` f32 paths** — adopt on 2026-08-20
5. **SVE** only if and when Rust stabilizes it

Adding C appears nowhere in that list.

### 5.2 Explicitly deprioritized

- **TurboQuant** — deprioritized *as a RaBitQ replacement*: theoretically
  dominated (bits scale `log log(1/δ)` vs `log(1/δ)`), with reported
  quantization times ~2 orders of magnitude optimistic.
  **Caveat, stated because the two research threads are in apparent tension:**
  a separate thread reports Qdrant shipped TurboQuant in v1.18 (May 2026, ICLR
  2026) beating *binary quantization* by 9–24 pp recall at 16× compression.
  These are different comparisons — TurboQuant-vs-RaBitQ on theory, and
  TurboQuant-vs-BQ on measured recall — and conflating them would be an error.
  The RaBitQ comparison is what governs this decision; the BQ result does not
  contradict it.
- **LVQ / LeanVec** — closed-source, Intel-hardware-only, and measured as the
  *worst* modern method on quantization error.
- **Pure-Rust CAGRA rewrite** — ~zero gain over binding `cuvs-sys`.
- **SAQ** — the measured frontier (1.8–5.4× lower error than Extended RaBitQ)
  but unreproduced and same-community-authored. Revisit after item 1 makes it
  measurable.

## 6. Consequences

**Positive.** Closes a live risk: we were positioned to publish claims against
a dead benchmark. The program is entirely Rust-native, so it composes with the
WASM story rather than fighting it. Items 2 and 7 are cheap and land early.

**Negative.** Item 1 delays visible performance work by 2–3 weeks. That
sequencing is deliberate and matches the harness lesson in ADR-273 §6: a
measurement you cannot trust makes every subsequent optimization unfalsifiable.

**Risk.** Several headline numbers (SAQ, CoDEQ, OctopusANN) are single-source
and unreproduced. They are treated as hypotheses to re-measure under item 1,
not as constants — the same evidence discipline applied in ADR-273.

## 6.1 Measured: PDX layout did NOT reproduce (2026-08-02)

A first implementation of the PDX vertical layout (`ruvector-core::pdx`) was
built and benchmarked against the existing row-major batch path on this host
(AVX-512, 4 cores). **The paper's ~2.0× did not reproduce.** Measured, both
paths compiled with `-C target-cpu=native`, 4096 vectors:

| n × dim | working set | row-major | PDX vertical | ratio |
|---|---|---|---|---|
| 256 × 768 | 0.75 MB | 13.73 µs | 19.52 µs | **0.70×** |
| 512 × 768 | 1.5 MB | 29.78 µs | 45.59 µs | **0.65×** |
| 1024 × 768 | 3 MB | 123.5 µs | 140.4 µs | 0.88× |
| 4096 × 768 | 12 MB | 650.2 µs | 505.4 µs | 1.29× |
| 4096 × 1536 | 25 MB | 1158 µs | 1065 µs | 1.09× |

PDX **loses** on cache-resident working sets and wins only when streaming.

Three caveats, stated because they bound what this measurement proves:

1. **The first run was invalid.** It showed PDX 14–18× slower, because the
   row-major path runtime-dispatches to AVX-512 via `is_x86_feature_detected!`
   while the new code compiled for baseline x86-64 (SSE2). That was an ISA
   comparison wearing a layout comparison's clothes. Fixed by building both
   with `target-cpu=native`.
2. **The remaining comparison is still confounded.** The row-major path takes
   `Vec<&[f32]>` derived from `Vec<Vec<f32>>` — 4096 separate heap allocations,
   so it pointer-chases — while `PdxIndex` is a single contiguous buffer. The
   streaming win may be *allocation contiguity*, not vertical layout. A clean
   experiment needs a contiguous row-major baseline.
3. **The PDX kernel is autovectorized generic Rust**, competing against
   hand-written AVX-512 intrinsics. That the naive version reaches parity at
   all is notable, but it is not a like-for-like layout test.

**Conclusion for now: do not adopt PDX layout on this evidence.** The honest
reading is that at f32 precision these workloads are bandwidth-bound, so layout
cannot help much — which *strengthens* the case for §5 item 2 (8-bit rotational
quantization), since 4× less data to stream attacks the actual bottleneck.
Revisit PDX for quantized codes, where the working set shrinks enough to become
compute-bound.

This is recorded rather than discarded because a negative result that cost a
day is worth more written down than repeated.

## 7. Open

- Third research thread (SIMD kernel evidence across FAISS/SimSIMD/usearch)
  outstanding; it refines kernel specifics — VNNI int8, f16/bf16 now that
  AVX-512 FP16 is stable — not the decisions above.
- Pull VIBE's `results/summary.parquet` to calibrate against the field before
  writing index code.
- Audit whether `ruvector-rabitq` implements 1-bit only or extended/multi-bit
  RaBitQ, and whether `rotation.rs` uses a fast Hadamard/Kac's walk or an
  O(d²) dense rotation.
