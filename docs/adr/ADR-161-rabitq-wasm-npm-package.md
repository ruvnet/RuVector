# ADR-161: Publish `ruvector-rabitq-wasm` as `@ruvector/rabitq-wasm` on npm

**Status**: Proposed
**Date**: 2026-04-26
**Driver**: User-flagged gap — the `ruvector-rabitq-wasm` Rust crate
shipped in commit `a674d6eba` but has no `package.json`, README, or
npm publication. The rotation-based 1-bit RaBitQ index (ADR-154) is
the most browser-relevant of the ruvector backends because it shrinks
embeddings 32× — exactly what edge / WebGPU / Cloudflare-Worker
deployments need. Letting the WASM bindings sit dark wastes the work.

## Context

ruvector already publishes one WASM package — `@ruvector/graph-wasm`
(v2.0.3, ~50 K monthly downloads) — built from
`crates/ruvector-graph-wasm/build.sh` via three `wasm-pack` targets
(`web`, `nodejs`, `bundler`) emitting into `npm/packages/graph-wasm/`.
The package is wired into npm via:

- `package.json` with `name = "@ruvector/graph-wasm"`,
  `publishConfig.access = "public"`, `files` listing the `.wasm` and
  `.js`/`.d.ts` artifacts that wasm-pack emits, and a homepage /
  repository pointer back into the Rust crate.
- `index.js` and `index.d.ts` shims that re-export the wasm-pack
  output.
- `README.md` describing usage in browser / Node / bundler contexts.

`ruvector-rabitq-wasm` already exposes the public surface (commit
`a674d6eba`):

- `RabitqIndex.build(vectors: Float32Array, dim: u32, seed: u64,
  rerank_factor: u32) -> RabitqIndex`
- `RabitqIndex.search(query: Float32Array, k: u32) -> SearchResult[]`
- `SearchResult { id: u32, distance: f32 }`
- `version()` for build-time crate version.
- `wasm-bindgen-test` suite under `#[cfg(target_arch = "wasm32")]`.

The native build is bit-identical to the wasm32 build because RaBitQ
rotation is deterministic by construction (`(seed, dim, vectors)` →
fixed codes — ADR-154 invariant).

## Decision

Mirror the `graph-wasm` packaging pattern for `rabitq-wasm`:

1. Add `crates/ruvector-rabitq-wasm/build.sh` — the standard 3-target
   `wasm-pack build` script that emits into
   `npm/packages/rabitq-wasm/{,node/,bundler/}`.
2. Add `npm/packages/rabitq-wasm/package.json`:
   - `name`: `@ruvector/rabitq-wasm`
   - `version`: `0.1.0` (matches Cargo)
   - `description`: 1-bit quantized vector index (RaBitQ) for browsers and edge runtimes
   - `keywords`: rabitq, vector-search, quantization, hnsw, ann, embeddings, wasm, webassembly, rust
   - `files`: just the wasm-pack-generated artifacts
   - `publishConfig.access = "public"`
3. Add `npm/packages/rabitq-wasm/README.md` — minimal install + usage
   example matching the doctest at the top of `lib.rs`.
4. Add a `Cargo.toml` `[lib] crate-type = ["cdylib", "rlib"]` if not
   already present (it is — verified before this ADR).
5. CI: leave the existing `check-wasm-dedup` job in place; do not add
   a wasm-pack-build CI job initially because wasm-pack downloads
   tooling at job start and we want to keep PR #391 / #393 unblocked.
   A follow-up ADR can wire it into `.github/workflows/ci.yml`.
6. Publish manually for now: `wasm-pack publish` after a clean `npm
   pack` review. Future ADR can switch to a release-please workflow.

## Versioning

The Cargo crate is at `0.1.0`. The npm package starts at `0.1.0` and
tracks Cargo. Because RaBitQ codes are stable across architectures
(rotation determinism), there is no separate semver story for the
WASM build versus the Rust build — same `0.1.0` ships everywhere.

## Alternatives considered

- **Don't publish; keep the crate internal.** Leaves a working WASM
  artifact unused. RaBitQ's primary value proposition (32× memory
  reduction for embedding indices) is most relevant at the edge —
  exactly the deployment target that needs npm distribution.
- **Publish under `ruvector-rabitq` (no scope).** The graph-wasm
  precedent uses `@ruvector/*`; mixing scoped and unscoped names is
  noise.
- **Bundle into `@ruvector/core`.** The NAPI-RS `core` package is
  Node-only (loads `.node` native binaries). WASM is a different
  delivery mechanism and a different audience — keeping them in
  separate npm packages lets browser and Worker users avoid the
  Node-only bits.

## Consequences

- Edge / browser users can `npm install @ruvector/rabitq-wasm` and
  get a 1-bit index without dragging in any of the workspace's
  Node-only crates.
- One more npm publish surface to maintain. Mitigated by reusing the
  exact directory layout / build.sh pattern from graph-wasm so
  release tooling treats them uniformly.
- The crate's existing `wasm_bindgen_test` suite remains the primary
  correctness gate for the JS surface; numerical correctness is
  covered by the parent `ruvector-rabitq` test suite.

## See also

- ADR-154 — RaBitQ rotation-based 1-bit quantization
- ADR-162 — `ruvector-acorn-wasm` packaging (sibling ADR)
- `crates/ruvector-graph-wasm/build.sh` — the script we mirror
- `npm/packages/graph-wasm/` — the npm structure we mirror
