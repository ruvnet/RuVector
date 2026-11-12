# ADR-296: Turbo4 — 4-bit Lloyd-Max Quantized Vector Datatype with Direct Packed HNSW Scoring

- **Status**: Accepted
- **Date**: 2026-08-06
- **Related**: ADR-254 (ruvector-turbovec multi-bit TurboQuant scalar reference), issue #563 (`DbOptions.quantization` persisted but not applied)
- **References**: Qdrant Turbo4 primary-datatype quantization; TurboQuant (arXiv:2504.19874); Max, *Quantizing for Minimum Distortion*, IRE 1960; RaBitQ (SIGMOD 2024)

## Context

RuVector has three partial quantization stories, none of which delivers a Qdrant-style
Turbo4 (4-bit primary vector datatype):

1. `ruvector-core::quantization::Int4Quantized` — a per-vector min/max 4-bit codec
   that is **never applied** to storage or indexes. `DbOptions.quantization` is
   persisted and restored but the HNSW index retains `Vec<f32>` everywhere
   (issue #563; `VectorDB::new` warns loudly about this).
2. `ruvector-turbovec` (ADR-254) — a correct Lloyd-Max 2/3/4-bit scalar
   *reference* index over the RaBitQ Hadamard rotation, but it is a standalone
   flat-scan crate: no HNSW, no SIMD kernel (explicitly deferred), not reachable
   from `VectorDB`.
3. `ruvllm`'s `TurboQuantCompressor` — a KV-cache codec that stores 4 scalar
   bits + 1 residual bit + block metadata (~704 bytes at 1024-D, ≈5.8×), and
   whose consumers decompress to f32 before use. It is **not** a faithful
   Turbo4 storage implementation and must not be wired into HNSW.

True Turbo4 semantics (per Qdrant / TurboQuant):

- Apply a randomized rotation so coordinates become approximately i.i.d. Gaussian.
- Map every coordinate onto one of 16 precomputed Lloyd-Max levels.
- Store **only** the packed 4-bit codes (plus O(1) per-vector constants) — the
  original float vector is never retained by the index.
- Score directly on packed codes (asymmetric for queries, symmetric between
  stored codes), never reconstructing f32 vectors during graph traversal.

At 1M × 1536-D: FP32 payload 6.144 GB → Turbo4 payload 0.768 GB (+8 B/vector
constants), a ~7.9× payload reduction.

## Decision

### 1. New crate `crates/ruvector-turboquant` (the Turbo4 codec)

A dependency-free (serde/thiserror only), WASM-safe crate providing:

- **Deterministic randomized rotation** — rounds of {±1 sign diagonal →
  permutation → block Fast Walsh-Hadamard Transform}, seeded by a `SplitMix64`
  PRNG implemented in-crate. No `rand` dependency: encoded bytes are persisted,
  so the rotation must be bit-stable across platforms, architectures, and
  dependency upgrades. Arbitrary dimensions are handled by decomposing `D` into
  power-of-two FWHT blocks (binary decomposition) with cross-block permutations
  between rounds — the vector is **not** zero-padded, so code size is exactly
  `ceil(D/2)` bytes.
- **Precomputed Lloyd-Max tables** — the 16 optimal N(0,1) reconstruction
  levels (Max 1960, same constants as ADR-254). Coordinates are standardized by
  the per-vector factor `α = ‖v‖₂/√D` (exact under orthogonal rotation), so a
  single analytic table serves all dimensions; no training pass, online ingest.
- **Packed nibble codes** — layout `[ceil(D/2) packed nibbles | α: f32 LE |
  S = Σ level(cᵢ)²: f32 LE]`. Even dimension in the low nibble.
- **Scoring without reconstruction** — every supported metric decomposes over
  the level dot product:
  - `dot(a,b) ≈ αₐ·α_b · Σ L[aᵢ]·L[bᵢ]`
  - `L2²(a,b) ≈ αₐ²Sₐ + α_b²S_b − 2·dot(a,b)`
  - `cos(a,b) ≈ Σ L[aᵢ]·L[bᵢ] / D` (the α factors cancel against the stored
    exact norms — no pre-normalization required)
  The integer kernel maps nibbles to an int8 level grid (`L_i8[j] =
  round(L[j]·127/L_max)`, a 16-entry table that fits a single `pshufb`/`tbl`
  shuffle register) and accumulates `i8×i8` products:
  - **Symmetric** (code × code): used for HNSW graph construction and
    neighbor-to-neighbor evaluation.
  - **Asymmetric** (query × code): the query is rotated once, quantized to a
    per-query int8 grid (8-bit query × 4-bit code, Qdrant-style), giving higher
    traversal fidelity than symmetric 4×4 scoring.
  - **Exact rescore** (f32 query × code): a per-query `D×16` f32 LUT scores
    top candidates with no integer error, used for final re-ranking.
- **SIMD kernels** — runtime-dispatched AVX2 (nibble unpack → `pshufb` level
  lookup → widening multiply-add) with a scalar fallback that is the
  correctness oracle. AVX-512 (VNNI), NEON (`vqtbl1q_s8` + `vdotq_s32`), and
  WASM SIMD128 kernels follow the same dispatch seam (phased below).

### 2. Turbo4 as an applied `QuantizationConfig` variant in `ruvector-core`

```rust
QuantizationConfig::Turbo4 {
    rotation_seed: u64,          // default 42
    rescore_multiplier: usize,   // default 4: rescore k·mult candidates exactly
}
```

Unlike the legacy variants, **this variant is actually applied**: when set with
an HNSW index, `VectorDB::new` builds a `Turbo4HnswIndex` instead of the f32
`HnswIndex`, and the "quantization not applied" warning is scoped to the legacy
variants only.

### 3. `Turbo4HnswIndex`: packed codes as the HNSW element type

`hnsw_rs` is generic over the element type; the index instantiates
`Hnsw<'static, u8, Turbo4Distance>` where each point's data **is the packed
code blob** — no `Vec<f32>` is retained anywhere in the index. Because
`Distance::eval(&[u8], &[u8])` receives raw slices, the two roles are
distinguished structurally by blob length (query blobs are `D+8` bytes, code
blobs `ceil(D/2)+8`; disjoint for all D ≥ 2):

- insert / graph maintenance → symmetric code×code kernel;
- search traversal → asymmetric int8-query×code kernel;
- after traversal returns `k · rescore_multiplier` candidates, they are
  re-scored with the exact f32 LUT and truncated to `k`.

Supported metrics: Euclidean, Cosine, DotProduct (same distance conventions as
the f32 `DistanceFn`). Manhattan does not decompose over the dot product and is
rejected at construction with a clear error.

### 4. Separation of source datatype and search quantization (North Star)

The end-state API separates what is *stored* from what *accelerates search*:

```rust
vector_datatype: VectorDataType::Turbo4 { rotation_seed: 42 },
search_quantization: SearchQuantization::RaBitQ1 { rescore_multiplier: 8 },
```

i.e. RaBitQ 1-bit codes (32×) for cheap candidate generation over the graph,
rescored against the Turbo4 codes (8×) — stronger than either standalone path.
This ADR implements the Turbo4 datatype and its direct-scored HNSW path;
the `VectorDataType`/`SearchQuantization` split and the RaBitQ cascade build on
it (phases below) without changing the storage format defined here.

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | `ruvector-turboquant` codec: rotation, tables, packed codes, symmetric/asymmetric/exact scoring, scalar + AVX2 kernels, property tests | this ADR |
| 2 | `Turbo4HnswIndex` + `QuantizationConfig::Turbo4` applied in `VectorDB` | this ADR |
| 3 | NEON + AVX-512 VNNI + WASM SIMD128 kernels; kernel microbenches vs scalar oracle | follow-up |
| 4 | Compressed at-rest storage (redb stores Turbo4 codes; f32 never persisted when datatype is Turbo4); index serialization | follow-up |
| 5 | `VectorDataType`/`SearchQuantization` split; RaBitQ1 candidate generation cascade with Turbo4 rescoring | follow-up |
| 6 | SIFT1M / GIST1M acceptance runs in `ruvector-sota-bench` | follow-up |

## Acceptance criteria

On SIFT1M, GIST1M, and one real RuVector embedding corpus:

- ≥ 7.5× vector payload compression (achieved by construction: `ceil(D/2)+8`
  bytes vs `4·D`; e.g. 776 B vs 6144 B at 1536-D ⇒ 7.92×).
- Recall@10 loss < 0.5 pp vs the f32 HNSW baseline at default
  `rescore_multiplier = 4` (tunable upward).
- P95 latency no worse than FP32 at equal recall.

## Consequences

- Index RSS drops ~3–6× end-to-end (graph links, ids, and metadata remain f32-era).
- Insert cost gains an O(D log D) rotation + O(D) quantization per vector — noise
  next to HNSW graph construction.
- Recall depends on the rescoring pass; `rescore_multiplier` is the recall/latency
  dial, mirroring Qdrant's `oversampling`.
- The rotation seed becomes part of the persistent index contract: codes encoded
  with one seed are meaningless under another. The seed is stored in `DbOptions`
  (already persisted) and validated on load.
- `hnsw_rs` currently keeps one copy of each code blob inside the graph and the
  index keeps one in its id→code map for rescoring/serialization (2× code bytes
  ≈ still ~4× smaller than one f32 copy). Deduplicating via the hnsw_rs datamap
  is a phase-3 optimization.

## Refinements from verified research (2026-08-06)

A deep-research pass over primary sources (TurboQuant arXiv:2504.19874;
Qdrant 1.18 release + quantization docs; RaBitQ SIGMOD 2024 / extended
RaBitQ SIGMOD 2025 arXiv:2409.09913; RaBitQ-team rebuttal arXiv:2604.19528)
confirmed the design and produced three refinements:

1. **Length renormalization (adopted)** — Lloyd-Max reconstructions are
   systematically short (`‖r‖ = α√S < α√D`), biasing inner-product estimates;
   this is the bias the TurboQuant paper counters with its QJL residual stage
   and Qdrant counters with RaBitQ-style renormalization. All three scoring
   tiers now scale the level dot by `√(D/S)` per encoded side and use exact
   norms `α²D` in the metric decomposition — zero storage cost, one multiply
   per candidate, since `α`, `S`, `D` are already in the blob.
2. **4-bit needs no raw-vector rescoring** (validated) — Qdrant serves 4-bit
   results directly from quantized scores (oversampled rescoring is default
   only at 1/1.5/2 bits). Our exact-LUT rescore over the *codes* plus
   ADR-297 adaptive escalation matches this: original floats are never
   needed on the search path.
3. **Kernel roadmap** (phase 3) — production kernels converge on
   pshufb-LUT + `maddubs` (fusing to `VPDPBUSD` on VNNI): bias the level
   table to u8 and correct with the per-query constant `128·Σq`, replacing
   the widen-to-i16 `madd` pair. To be adopted with criterion benchmarks, not
   assumed. Extended RaBitQ's results (>95 % recall at ~5 bits/dim with no
   rescoring, provably optimal space-error tradeoff, stronger tail bounds
   than TurboQuant) reinforce the ADR-297 phase C RaBitQ cascade and make an
   extended-RaBitQ multi-bit codec a candidate for the codec plane.

## Alternatives considered

- **Wire `ruvllm`'s `TurboQuantCompressor` into HNSW** — rejected: it
  decompresses candidates to f32 (defeating bandwidth savings), and its
  4+1-bit + block-metadata format is 5.8×, not 8×.
- **Reuse `ruvector-turbovec` directly as the core index** — rejected: it is a
  flat-scan reference without HNSW or SIMD; its Lloyd-Max constants and design
  learnings (padding pitfalls, calibration) are reused, and it remains the
  scalar determinism oracle for cross-crate tests.
- **Reuse `ruvector_rabitq::RandomRotation`** — rejected for the storage codec:
  it draws from `rand`'s `StdRng`, whose stream is only stable per `rand` major
  version — unacceptable for a persisted storage format. The in-crate SplitMix64
  construction is bit-stable by fiat and keeps the codec dependency-free.
- **Zero-padding to the next power of two** (turbovec's choice) — rejected:
  at 1536-D padding inflates codes from 768 B to 1024 B, silently degrading the
  headline compression to 6×. Block-FWHT with cross-block permutations keeps
  exact `ceil(D/2)` while preserving orthogonality exactly and Gaussianization
  approximately.
- **Per-vector min/max scaling** (existing `Int4Quantized`) — rejected: uniform
  levels on a per-vector range are strictly worse than Lloyd-Max on the rotated
  Gaussian marginal, and per-vector ranges break shared-table symmetric scoring.
