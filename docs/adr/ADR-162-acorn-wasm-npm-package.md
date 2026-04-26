# ADR-162: Add `ruvector-acorn-wasm` crate and publish as `@ruvector/acorn-wasm` on npm

**Status**: Proposed
**Date**: 2026-04-26
**Driver**: ADR-160 ships a pure-Rust ACORN filtered HNSW with 96%
recall@10 at 1% selectivity. Filtered vector search is the dominant
production access pattern (RAG with metadata filters, ACL-gated
retrieval, e-commerce attribute filters), and the most useful place
for it is *closer to the user*: at the edge, in the browser, or in a
worker. Today the crate is workspace-internal and the only Rust-to-JS
delivery for the workspace is `@ruvector/graph-wasm`. Add a sibling
WASM crate + npm package so browser/edge users can consume ACORN
without a server.

## Context

ADR-160 introduces `crates/ruvector-acorn` with a `FilteredIndex`
trait and three variants: `FlatFilteredIndex`, `AcornIndex1` (γ=1,
M=16), `AcornIndexGamma` (γ=2, M=32). The optimization round (PR
#391, commit `eb88176`) added:

- **Bounded-beam fix** in `acorn_search` (correctness)
- **Parallel build** with rayon (≈80× faster index construction)
- **Flat row-major data layout** (cache locality + SIMD)
- **`Vec<bool>` visited** (no hashing on the hot path)
- **Hand-unrolled L2²** (3-5× faster distance kernel for D ≥ 64)

The crate has 12/12 unit tests passing and a `cargo run --release`
benchmark binary that produces a recall/QPS table.

ADR-161 covers the sibling `ruvector-rabitq-wasm` packaging. This ADR
is the parallel decision for the *missing* acorn WASM crate — the
Rust crate exists but has no `wasm-bindgen` wrapper and no npm
package.

## Decision

1. **Add `crates/ruvector-acorn-wasm`** — new workspace member. Mirrors
   the layout of `crates/ruvector-rabitq-wasm`:
   - `Cargo.toml` with `crate-type = ["cdylib", "rlib"]`, `wasm-bindgen`,
     `js-sys`, `serde-wasm-bindgen`, `console_error_panic_hook`
     (default-feature), `getrandom` with `js` feature behind a
     `cfg(target_arch = "wasm32")` block. Depends on
     `ruvector-acorn` from the workspace.
   - `src/lib.rs` exposing:
     - `AcornIndex` (default = γ=2, M=32 — best recall) with
       `build(vectors: &[f32], dim: u32, gamma: u32) -> AcornIndex`.
     - `search(query: &[f32], k: u32, predicate: &js_sys::Function) -> SearchResult[]`.
       The predicate is a JS callback `(id: number) => boolean` so
       browser callers can plug in arbitrary filter logic without
       crossing the FFI boundary on every vector.
     - `SearchResult { id: u32, distance: f32 }` mirroring the RaBitQ
       binding for shape-symmetric SDKs.
     - `version()` for the build-time crate version.
   - `wasm-bindgen-test` smoke test under `#[cfg(target_arch =
     "wasm32")]` (the same gate the rabitq-wasm crate uses to dodge
     wasm-bindgen 0.2.117's native-context panics).

2. **Add `npm/packages/acorn-wasm/`** — three-target wasm-pack output
   (`web`, `nodejs`, `bundler`) plus:
   - `package.json` named `@ruvector/acorn-wasm`, version `0.1.0`,
     `publishConfig.access = "public"`, identical structure to
     `npm/packages/graph-wasm/package.json`.
   - `README.md` with install + minimal usage example.

3. **Add `crates/ruvector-acorn-wasm/build.sh`** — the standard 3-target
   `wasm-pack build` script that emits into `npm/packages/acorn-wasm/`.

4. **Don't add a CI wasm-pack job yet** — same reasoning as ADR-161.
   `check-wasm-dedup` keeps the build honest; a follow-up ADR can
   wire the publish step into release-please.

5. **Default the JS class to ACORN-γ.** The trait + three variants in
   the Rust crate are useful for benchmarking; for npm consumers,
   ship the variant with the best recall/cost trade-off. ACORN-γ at
   γ=2 doubles edges (≈3 MB for n=5K, D=128) but maintains 96%
   recall@10 at 1% selectivity. We expose `gamma: u32` as an explicit
   parameter so callers can pick γ=1 if they need a smaller graph.

## Predicate boundary

The Rust crate accepts `&dyn Fn(u32) -> bool`. In WASM we expose the
predicate as a `js_sys::Function` so the JavaScript runtime evaluates
each filter test. This crosses the FFI boundary once per node visited
during search (≤ ef nodes ≈ 150 default), not once per vector — the
overhead is bounded and predictable. The alternative (compiling
predicates as a closure in WASM via macros) is significantly more
complex and offers no real perf win at the scales where browser-side
ACORN makes sense.

## Versioning

The Rust crate starts at `0.1.0` to match its sibling.
`@ruvector/acorn-wasm@0.1.0` ships in lockstep. ACORN itself is
deterministic given a fixed graph build seed (the greedy NN-descent
isn't seeded today — listed as roadmap), so wasm32 and native
produce identical search output for an identical input set.

## Alternatives considered

- **Bundle ACORN into `@ruvector/graph-wasm`.** That package targets
  Cypher-style graph DB use, not ANN search. Combining doubles the
  WASM bundle size and confuses keyword discovery (graph DB users
  searching for it now have to wade through filter-search content).
- **Don't ship; let users compile their own.** Only realistic for
  Rust users. Browser/Worker consumers would have to set up
  wasm-pack + a build pipeline themselves, which is a deal-breaker
  for "I just want to add filtered search to my page" scenarios.
- **Predicate as a Rust closure encoded as an opcode tape.** Would
  let us avoid the JS-call-per-node FFI hop, but adds a mini-DSL
  surface. Not worth the complexity at filter-cost ≪ distance-cost.

## Consequences

- A second WASM npm package the project maintains. Mitigated by
  using the same directory layout / build.sh pattern as graph-wasm
  and rabitq-wasm so release tooling sees them all uniformly.
- The Rust trait surface stays the same; the WASM crate is a
  thin façade. Future Rust-side optimizations (parallel queries,
  simsimd kernel, NN-descent build) flow to the WASM build for free.
- Browser and edge-runtime users can `npm install
  @ruvector/acorn-wasm` and get filtered ANN search with no server.

## See also

- ADR-160 — ACORN predicate-agnostic filtered HNSW
- ADR-161 — `ruvector-rabitq-wasm` npm packaging (sibling ADR)
- `crates/ruvector-rabitq-wasm/src/lib.rs` — the sibling crate we
  mirror
- `npm/packages/graph-wasm/` — the npm structure pattern
