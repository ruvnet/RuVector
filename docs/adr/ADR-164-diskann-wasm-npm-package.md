# ADR-164: Add `ruvector-diskann-wasm` and publish as `@ruvector/diskann-wasm` on npm

**Status**: Proposed
**Date**: 2026-04-28
**Driver**: User-flagged gap — `ruvector-diskann` has Node bindings (`@ruvector/diskann@0.1.0` via `crates/ruvector-diskann-node`) but **no WASM crate and no WASM npm package**. Sister ANN backends already have both: RaBitQ (`@ruvector/rabitq-wasm` via ADR-161) and ACORN (`@ruvector/acorn-wasm` via ADR-162). DiskANN/Vamana is the standard graph-based ANN baseline; not having a browser/edge build is a hole in the lineup.

## Context

`crates/ruvector-diskann` (commit `8fbe76862`, "DiskANN/Vamana — SSD-friendly approximate nearest neighbor search with product quantization") implements:

- Vamana graph construction with α-robust pruning (R, L_build, α)
- Product Quantization for compressed candidate distances (M subspaces, k-means trained)
- mmap-backed graph + PQ codes for SSD-resident operation
- Disk-backed rerank that lazily loads exact vectors from disk during search
- `parking_lot` locking and `rayon` parallelism for build

The crate ships two consumer surfaces today:

- **`crates/ruvector-diskann-node`** → `@ruvector/diskann@0.1.0` (NAPI-RS / Node addon)
- *(none)* → no `crates/ruvector-diskann-wasm`, no `@ruvector/diskann-wasm` on npm

Sister WASM crates that already exist:

- `crates/ruvector-rabitq-wasm` → `@ruvector/rabitq-wasm@0.1.0` (ADR-161)
- `crates/ruvector-acorn-wasm` → `@ruvector/acorn-wasm@0.1.0` (ADR-162)
- `crates/ruvector-graph-wasm` → `@ruvector/graph-wasm@2.x` (the original pattern)

A DiskANN WASM build closes the matrix: every browser-shaped ANN backend in the repo gets an npm-distributable artifact for browsers / Cloudflare Workers / Deno / Bun.

## What does NOT translate cleanly to wasm32

DiskANN's name comes from features that aren't browser-feasible. The WASM crate must drop them, not paper over them:

| Native feature | Reason | WASM treatment |
|---|---|---|
| `memmap2::Mmap` for vector / graph data | No filesystem in browsers; OPFS exists but isn't the same primitive | Drop the `mmap: Option<Mmap>` field via `#[cfg(not(target_arch = "wasm32"))]`; in-memory `FlatVectors` only |
| Disk-backed rerank (PR #385) reading exact vectors from disk during search | Same — no filesystem | Always rerank against in-memory `FlatVectors` |
| `storage_path` persistence (write graph + PQ codes to disk) | Same | `storage_path` field accepted but ignored on wasm32; no `save()` / `load()` exposed |
| `rayon::par_iter` for parallel Vamana build | wasm32-unknown-unknown is single-threaded by default | Sequential build behind `#[cfg(target_arch = "wasm32")]`, mirroring the pattern landed in `ruvector-rabitq` for #394 |
| `parking_lot::RwLock` | Works in wasm32 but adds bytes; not load-bearing in WASM (no concurrent access) | Keep the type for native; WASM build uses single-threaded path |

Numerical output is bit-identical to the native in-memory path — Vamana graph build is deterministic given a seeded RNG, and PQ training is deterministic given the same iteration count and data layout. The on-disk persistence path is what we drop, not the algorithm.

## Decision

Add `crates/ruvector-diskann-wasm` mirroring the rabitq-wasm / acorn-wasm structure, and publish as `@ruvector/diskann-wasm@0.1.0`.

### Crate layout

```
crates/ruvector-diskann-wasm/
├── Cargo.toml          # cdylib + rlib, wasm-bindgen 0.2, depends on ruvector-diskann
├── build.sh            # 3-target wasm-pack (web | nodejs | bundler) → npm/packages/diskann-wasm/
├── src/
│   └── lib.rs          # DiskAnnWasm wrapper class + JS-facing types
└── tests/
    └── web.rs          # wasm-bindgen-test smoke (build → search → recall@10 ≥ 0.7)
```

`Cargo.toml` follows the pattern from `crates/ruvector-rabitq-wasm/Cargo.toml`:

```toml
[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
ruvector-diskann = { path = "../ruvector-diskann", default-features = false }
wasm-bindgen = { workspace = true }
js-sys = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
console_error_panic_hook = { version = "0.1", optional = true }

[target.'cfg(target_arch = "wasm32")'.dependencies]
getrandom = { workspace = true, features = ["wasm_js"] }

[features]
default = ["console_error_panic_hook"]
```

### Public WASM surface (v0.1.0)

```rust
#[wasm_bindgen]
pub struct DiskAnnWasm { /* in-memory only */ }

#[wasm_bindgen]
impl DiskAnnWasm {
    /// Build an in-memory DiskANN/Vamana index over `vectors` (row-major
    /// Float32Array of length `n * dim`).
    pub fn build(
        vectors: &[f32],
        dim: u32,
        max_degree: u32,    // R     (default 64)
        build_beam: u32,    // L_b   (default 128)
        search_beam: u32,   // L_s   (default 64)
        alpha: f32,         // α     (default 1.2)
    ) -> Result<DiskAnnWasm, JsError>;

    /// Top-k search with optional `pq_subspaces` for compressed candidate
    /// distances (0 = no PQ; recommended at high D).
    pub fn search(&self, query: &[f32], k: u32) -> Result<JsValue, JsError>;

    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> u32;

    #[wasm_bindgen(getter)]
    pub fn len(&self) -> u32;

    #[wasm_bindgen(getter, js_name = "memoryBytes")]
    pub fn memory_bytes(&self) -> u32;
}
```

`SearchResult` from native uses `String` ids; the WASM build uses `u32` ids (the row index passed to `build`) to keep allocation per query at zero — same simplification we made in rabitq-wasm. Caller maintains their own external→internal id map.

### npm package

```
npm/packages/diskann-wasm/
├── package.scoped.json     # canonical (committed) — copied to package.json by build.sh
├── README.md               # install, usage (browser / Node / bundler)
├── .gitignore              # excludes generated .wasm/.js/.d.ts and package.json
└── (post-build)
    ├── ruvector_diskann_wasm.js           # web target
    ├── ruvector_diskann_wasm.d.ts
    ├── ruvector_diskann_wasm_bg.wasm
    ├── node/                               # nodejs target
    └── bundler/                            # bundler target
```

`package.scoped.json` mirrors `npm/packages/rabitq-wasm/package.scoped.json` exactly, with name = `@ruvector/diskann-wasm`, version = `0.1.0`, and the SEO keyword set tuned for DiskANN ("diskann", "vamana", "graph-ann", "billion-scale", plus the standard "vector-search", "ann", "embeddings", "wasm", "webassembly", "rag" tail).

### Build workflow

Same `build.sh` shape as rabitq-wasm:

```bash
unset RUSTFLAGS                                  # mold rejects wasm-ld
wasm-pack build --target web      -d .../diskann-wasm
wasm-pack build --target nodejs   -d .../diskann-wasm/node
wasm-pack build --target bundler  -d .../diskann-wasm/bundler
cp package.scoped.json package.json              # restore scoped name after wasm-pack regenerate
```

CI: lean on the existing `check-wasm-dedup` workspace job. Do not add a dedicated wasm-pack build job in this ADR — wasm-pack tooling install dominates CI time, and the rabitq-wasm / acorn-wasm packages aren't gated either. A follow-up ADR can bundle all WASM packages into one wasm-pack matrix job once it pays for itself.

## Versioning

Cargo and npm both start at **0.1.0**. The Rust crate `ruvector-diskann` is at workspace version `2.2.0`, but the WASM wrapper is its own semver track because:

- The native crate exposes `String` ids, mmap, persistence — none of which the WASM API has.
- Sister WASM crates (`rabitq-wasm`, `acorn-wasm`) start at 0.1.0 independent of their parent crate version.
- A consumer pinning `@ruvector/diskann-wasm@^0.1.0` should not be force-bumped every time `ruvector-diskann` adds a server-side feature.

## Out of scope (intentionally)

The first WASM release is in-memory + single-threaded + no persistence. Things that **could** ship in a later 0.2.x but **don't** in 0.1.0:

- **OPFS persistence**: `save(handle)` / `load(handle)` writing graph + PQ codes to a browser OPFS file handle. Real demand exists (large indices in long-lived Workers), but the API design needs a separate ADR — sync-handle vs async, framing/serialization choice, vs IndexedDB.
- **Web Workers + threading**: rayon-on-wasm via `wasm-bindgen-rayon` requires `SharedArrayBuffer` + cross-origin-isolation headers. Out of band; users can wrap `DiskAnnWasm.build(...)` in their own Worker today.
- **PQ in the WASM build**: PQ training (k-means) and PQ-distance candidate filtering work in wasm32 in principle, but the 0.1.0 surface keeps `pq_subspaces = 0`. Once recall + speed are validated for the brute-rerank path, we can expose PQ in 0.2.0.
- **Disk-backed rerank**: dropped entirely from the WASM build. The native PR #385 path stays Node/native only.

These are listed so consumers know the v0.1.0 ceiling; they aren't promises.

## Alternatives considered

- **Don't publish; keep DiskANN Node-only.** Loses the consistency win — every other ANN backend in the repo has a WASM build. Browser/edge users have to swap implementations when they want a graph-based index instead of HNSW or RaBitQ.
- **Publish a single `@ruvector/wasm` mega-package containing all backends.** Bundle size becomes a problem fast — even one backend is ~70–85 KB compressed. Users running edge functions pay for backends they don't use. The per-backend split is what graph-wasm / rabitq-wasm / acorn-wasm already standardize on.
- **Wait for OPFS persistence to be ready and ship 0.1.0 with persistence built in.** Couples two separate decisions. The brute-rerank in-memory build is useful on its own (small-N RAG, on-the-fly re-indexing of session data); persistence design can take its time.
- **Reuse `crates/ruvector-diskann-node` and target both via NAPI's Node + napi-rs's experimental wasm32 path.** napi-rs's wasm32 path exists but is immature; wasm-bindgen is the established route this repo uses for every other WASM crate. Stay consistent.

## Consequences

- Closes the WASM-coverage gap — every browser-relevant ANN backend in the repo (`graph`, `rabitq`, `acorn`, `diskann`) has a parallel `@ruvector/*-wasm` package on npm.
- One more wasm-pack build in the publish process. Mitigated by mirroring rabitq-wasm's `build.sh` so the release runner has a uniform shape.
- A `ruvector-diskann-wasm` workspace member is added; it's small (re-exports, no new test infrastructure beyond `wasm-bindgen-test` smoke).
- The on-disk DiskANN feature set (mmap, persistence, disk-backed rerank) explicitly **does not regress** in this ADR — those code paths stay native-only via `cfg(not(target_arch = "wasm32"))`. The WASM build is a strict subset, not a re-implementation.
- Consumers reading the Rust API and the WASM API will see different surfaces (notably `u32` ids in WASM vs `String` in native). Documented in the README.

## See also

- ADR-161 — `ruvector-rabitq-wasm` packaging (sibling, the closest precedent — same RNG-determinism, same in-memory simplification)
- ADR-162 — `ruvector-acorn-wasm` packaging (sibling, predicate-agnostic filtered HNSW)
- ADR-143 — DiskANN/Vamana adoption (the parent algorithm decision)
- `crates/ruvector-rabitq-wasm/` — the directory layout we mirror
- `npm/packages/rabitq-wasm/` — the npm-package layout we mirror
- `crates/ruvector-graph-wasm/build.sh` — the original 3-target wasm-pack pattern
