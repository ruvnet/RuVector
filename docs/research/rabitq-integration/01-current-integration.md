# 01 — Current RaBitQ Integration in RuVector

## What `ruvector-rabitq` ships (the supplier)

Crate `ruvector-rabitq` 2.2.0 (workspace version, `Cargo.toml:215`) lives
at `crates/ruvector-rabitq/` and exports four pieces from
`crates/ruvector-rabitq/src/lib.rs:45-59`:

| Item | Source | Status |
|------|--------|--------|
| `FlatF32Index`, `RabitqIndex`, `RabitqAsymIndex`, `RabitqPlusIndex` | `src/index.rs` | shipped, all four behind `AnnIndex` |
| `BinaryCode`, `pack_bits`, `unpack_bits` | `src/quantize.rs` | shipped |
| `RandomRotation`, `RandomRotationKind` | `src/rotation.rs` | shipped (Haar + Hadamard-signed) |
| `persist::save_index` / `load_index` (`.rbpx` v1) | `src/persist.rs:118,187` | shipped, deterministic seed-based |
| `VectorKernel`, `KernelCaps`, `ScanRequest`, `ScanResponse`, `CpuKernel` | `src/kernel.rs:78-126` | **trait shipped, only `CpuKernel` implements it** |

The "shipped vs. scaffolded" map for the kernel surface is critical:
the trait is ready and a default kernel exists, but the dispatch lives
in **no caller** today (see ruLake gap below).

## Real consumers in the workspace

Three call sites import `ruvector_rabitq`. They are the universe of
integration as of HEAD.

### 1. `ruvector-rulake` — the showpiece

`crates/ruvector-rulake/Cargo.toml:16` pins
`ruvector-rabitq = { path = "../ruvector-rabitq", version = "2.2" }`.
The crate is the only one in the tree that already exercises every
public surface of rabitq:

| Surface | Used in | Lines |
|---------|---------|-------|
| `RabitqPlusIndex::from_vectors_parallel` (build) | `crates/ruvector-rulake/src/cache.rs:402` | rayon-parallel rotate+pack on cache prime |
| `RabitqPlusIndex::new` + `add` (incremental) | `crates/ruvector-rulake/src/cache.rs:409` | small-batch path |
| `Arc<RabitqPlusIndex>` cache slot | `crates/ruvector-rulake/src/cache.rs:213,488,499,667` | concurrency story (see ADR-155 §"Arc-concurrency 12×") |
| `AnnIndex::search` / `search_with_rerank` | `crates/ruvector-rulake/src/cache.rs:708,833` | hot path |
| `persist::save_index` / `load_index` (`.rbpx`) | `crates/ruvector-rulake/src/lake.rs:304,399` | bundle warm/freeze |
| `RabitqError` `From` conversion | `crates/ruvector-rulake/src/error.rs:17-18` | error propagation |
| `RandomRotationKind::HadamardSigned` | `crates/ruvector-rulake/benches/*` (per BENCHMARK.md) | rotation-flavor toggle |

Total: **15 references** across `cache.rs`, `lake.rs`, `error.rs`, the
demo bin, and the federation smoke test (count from
`grep -n rabitq crates/ruvector-rulake/src/{lib,cache,lake}.rs`).

ruLake exposes ruvector-rabitq's contract under three witness modes
(`Consistency::{Fresh, Eventual, Frozen}` — `lake.rs`, ADR-155). The
measured intermediary tax on a cache hit is **1.02× direct
`RabitqPlusIndex::search`** (`crates/ruvector-rulake/BENCHMARK.md` and
ADR-157 §Context). This is the cost ceiling against which every other
integration should be measured.

**Gap: `VectorKernel` is referenced but not wired.** `lake.rs:595` is
literally a doc comment "this is also the plug-point for the future
`VectorKernel` trait (ADR-157)". `register_kernel` does not exist as a
method in `crates/ruvector-rulake/src/lake.rs`. The README confirms
under "M2+ on the roadmap":
`crates/ruvector-rulake/README.md:507` — `VectorKernel` trait
scaffolding (M1, done) → `crates/ruvector-rulake/README.md:515` — GPU
kernels in separate crates (M2+, deferred). The dispatch policy from
ADR-157 has no caller.

### 2. `ruvector-py` — the third major consumer (PR #381 / commit `e7f5a391f`)

`crates/ruvector-py/Cargo.toml:25` pins `ruvector-rabitq = { path =
"../ruvector-rabitq" }` and exposes a single `RabitqIndex` PyO3 class
backed by `RabitqPlusIndex`. Surface used:

| Surface | Used in | Lines |
|---------|---------|-------|
| `RabitqPlusIndex::from_vectors_parallel` (with GIL release) | `crates/ruvector-py/src/rabitq.rs:118` | `py.allow_threads` wraps the rotate+pack |
| `AnnIndex::search_with_rerank` | `crates/ruvector-py/src/rabitq.rs:154` | per-call rerank override |
| `RabitqPlusIndex::export_items` | `crates/ruvector-py/src/rabitq.rs` (in `save`) | replay-source recovery |
| `persist::save_index` / `load_index` | `crates/ruvector-py/src/rabitq.rs:198` | NumPy-friendly disk roundtrip |
| `RabitqError → PyErr` | `crates/ruvector-py/src/error.rs:25` | typed Python error |

This consumer's lesson, recorded directly in the source comment at
`src/rabitq.rs:35-43`: *RaBitQ does not expose `originals_flat`
directly; the wrapper must call `export_items()` to re-materialise the
items vector for `save_index`.* This drives the §04 design rule.

### 3. The rabitq demo binary

`crates/ruvector-rabitq/src/main.rs:28-29` imports every index variant
(`FlatF32Index`, `RabitqAsymIndex`, `RabitqIndex`, `RabitqPlusIndex`)
and benches them on clustered Gaussian data. This is internal
benchmarking, not an integration in the workspace sense, but it's the
canonical place to read all four indexes used together.

## The integration map at HEAD

```
            consumers                   supplier
            ─────────                   ────────
  ruvector-rulake   ────────►  ┌────────────────────────────┐
    (cache, lake,              │  ruvector-rabitq 2.2.0     │
     bundle, witness)          │                            │
                               │  - RabitqPlusIndex (build, │
  ruvector-py       ────────►  │    add, search, persist)   │
    (Python wheel,             │  - VectorKernel trait      │
     M1)                       │  - CpuKernel only          │
                               │                            │
  rabitq-demo       ────────►  │  - rotation, pack/unpack   │
    (internal bench)           └────────────────────────────┘
```

Every other crate in the workspace **does not** depend on
`ruvector-rabitq`. The 126 other crates listed under `crates/` are
empty space from rabitq's perspective. That gap is what §02 surveys.

## Three properties every existing consumer relies on

These show up in the source comments of all three call sites and they
are the load-bearing API contract:

1. **Determinism across processes.** `(dim, seed, items) →
   bit-identical index` (`crates/ruvector-rabitq/src/persist.rs:14-17`,
   re-cited in `ruvector-rulake::cache::CacheEntry` and the
   roundtrip-preserves-search-results test at
   `persist.rs:258-318`). ruLake's witness chain (ADR-155) and
   cross-backend cache sharing depend on this.
2. **Encapsulation: no exposed `originals_flat`.** Consumers that need
   raw vectors call `export_items()` (`src/index.rs:589`) — the field
   itself is private (`src/index.rs:546`). Both rulake and the Python
   SDK live with this; new consumers must too.
3. **`AnnIndex` is the only stable trait.** `RabitqPlusIndex::search`,
   `search_with_rerank`, `len`, `dim`, `external_ids`, `ids_u64` —
   these are the public hot-path surface. Internals (`originals_flat`,
   `last_word_mask`, `cos_lut`) are private and the persist format
   exists precisely to avoid widening that encapsulation
   (`crates/ruvector-rabitq/src/persist.rs:1-18`).

These three are what §04 elaborates as "must not break".
