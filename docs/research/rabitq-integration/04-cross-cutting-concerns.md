# 04 — Cross-Cutting Concerns

The invariants every new RaBitQ integration must hold. These come from
reading the existing call sites and the ADRs that govern them; if a
new integration breaks any of these, it almost certainly invalidates
ADR-154/155/157 by side effect.

---

## 1. Determinism across architectures

**The contract.** `(dim, seed, items) → bit-identical rotation matrix
+ packed codes + index build + search output across runs and across
machines.` Stated explicitly at
`crates/ruvector-rabitq/src/persist.rs:14-17` and re-stated at
`crates/ruvector-rabitq/src/lib.rs:34-37`. Tested by
`persist::tests::serialize_roundtrip_preserves_search_results`
(`persist.rs:258`) which compares score bits with `to_bits()` — not a
tolerance compare.

**Why it matters for new integrations.** ruLake's witness chain
(ADR-155) and cross-backend cache sharing (the
`two_backends_share_cache_when_witness_matches` test) depend on
this. So does the rabitq-by-reference handoff in
`ArtifactKind::RuLakeWitness`
(`crates/rvAgent/rvagent-a2a/src/artifact_types.rs:64`) — agents on
different boxes reading the same witness must compute the same
top-k.

**The trap.** Floating-point reduction order is not stable across
SIMD widths or GPU lane counts. ADR-157 already calls this out:
the **scan phase** (1-bit popcount) is integer math and trivially
deterministic; the **rerank phase** (exact L2²) is float reduction
and can diverge in the last ulp on GPU. ADR-157's resolution: kernels
that can't guarantee identical rerank set `caps().deterministic =
false`, and the dispatch policy refuses to use them on Fresh/Frozen
paths.

**Enforcement.** Every integration adds a regression test of the
shape "build same data twice, different threads/seeds-with-same-value,
asserting `to_bits()` match on at least 100 query results". The
existing test at `persist.rs:258` is the model.

---

## 2. Witness format compatibility

**The contract.** `.rbpx` v1 is the on-disk and on-wire format
(`crates/ruvector-rabitq/src/persist.rs:23-33`). It carries
`(magic, version, dim, seed, rerank_factor, n, items)`. The format is
**deliberately seed-based** rather than field-based — it stores the
*replay inputs*, not the index internals, because the deterministic
build is cheaper to re-run than the rotation matrix is to ship.

**Why it matters.** Every cross-process integration that wants
witness-sealed memory rides this format. `ruvector-rulake`'s
`save_index`/`load_index` calls (`lake.rs:304,399`) are the only
producer/consumer today, but ADR-159's `RuLakeWitness` artifact (and
its `data_ref` field) implicitly depends on this format being stable.

**The trap.** A consumer that needs a fielded format (e.g. for a
columnar store like Parquet) will be tempted to widen `.rbpx` v1 with
extra fields. Don't. The right shape is:

- For a richer container, wrap `.rbpx` inside another format
  (e.g. a tar-like bundle that holds `.rbpx` + a sidecar metadata file).
- For a different field set entirely, bump to `.rbpx` v2 in the same
  module, with a feature flag, and keep v1 readable.
- Never extend v1 in place. The persist format's `MAGIC` + `VERSION`
  bytes (`persist.rs:49-51`) are a contract.

**Enforcement.** PR review on every change touching `persist.rs`. The
`reject_version_too_new` test (`persist.rs:425`) defends this.

---

## 3. Memory ownership: who holds the codes

**The lesson from PR #381 (Python SDK).** `RabitqPlusIndex` does not
expose `originals_flat` directly — the field is private at
`crates/ruvector-rabitq/src/index.rs:546`. Consumers that need to
re-export the originals (e.g. for `save_index`) call
`export_items()` (`src/index.rs:589`), which **clones**
`n*dim*sizeof(f32)` bytes. This is documented in
`crates/ruvector-py/src/rabitq.rs:35-43` as a deliberate cost trade.

**The contract.** Three rules.

a. The cache (or the consumer's struct) owns the `Arc<RabitqPlusIndex>`.
   `ruvector-rulake::cache.rs:213` is the model.

b. New consumers that need raw vectors call `export_items()`. They do
   not get a borrowed slice; the wrapper is intentional.

c. New consumers that need to *avoid* the export-items copy need to
   restructure to keep the source-of-truth `Vec<f32>` themselves and
   use `RabitqPlusIndex` only for the codes + search. The Python SDK
   chose to do the copy; ruLake chose to keep the source-of-truth in
   `LocalBackend::PulledBatch`.

**The trap.** A PR that "adds a `pub fn raw_vector(&self, i: usize) ->
&[f32]` to `RabitqPlusIndex` for performance" — see Anti-pattern C in
§03. Refuse it. If the perf is real, the right move is to widen
`AnnIndex`, not the struct internals.

---

## 4. API stability and version pinning

**The state.** `ruvector-rabitq` is at `2.2.0` on crates.io
(`Cargo.toml:215` workspace version). Both consumer Cargo.tomls
(`ruvector-rulake/Cargo.toml:16`, `ruvector-py/Cargo.toml:25`) pin via
`path = "../ruvector-rabitq"`. The rulake Cargo.toml also adds a
`version = "2.2"` constraint, the Python SDK doesn't yet — that's
worth normalising.

**The contract.** New integrations pin `ruvector-rabitq = { path =
"../ruvector-rabitq", version = "^2.2" }` to allow patch + minor
upgrades but block major ones. This is what semver bought: anything
that needs to break the persist format or the index trait surface
becomes a major bump and forces a synchronised upgrade across all
consumers.

**The trap.** Workspace-only `path` deps without a version constraint
work locally, but the moment the supplier crate publishes a major
version on crates.io and a downstream user pulls
`ruvector-rabitq = "3"` the workspace is silently inconsistent. Add
the version constraint at integration time.

---

## 5. Performance footprint on small targets

**The numbers.** `ruvector-rabitq`'s `Cargo.toml` deps are `rand`,
`rand_distr`, `rayon`, `serde`, `serde_json`, `thiserror` — small.
But the rotation tables, the cos-LUT, and the binary code paths add
~50 KB to a release WASM bundle (estimated; not yet measured for
ruvector-py wheel). The crate explicitly disables `unsafe` and pulls
no BLAS, which keeps it portable.

**The contract.** WASM, embedded, and `wasm32-*` consumers must
feature-gate the rabitq dep. The `Cargo.toml` excludes list at
`/home/ruvultra/projects/ruvector/Cargo.toml:1-8` already keeps things
out of `cargo build --workspace` selectively; new WASM consumers
should follow that pattern.

**The trap.** Adding `ruvector-rabitq` as a default dep on a
hypothetical `ruvector-edge-something` crate, then discovering the
WASM build is 50 KB heavier and the embedded ESP32 build (cf.
`examples/ruvLLM/esp32-flash` excluded list) doesn't link. Feature-
gate before integrating, not after.

---

## 6. Cross-language story

**The state today.** `ruvector-py` is the only non-Rust consumer
(M1 shipped, commit `e7f5a391f`). Wheel binding via PyO3 + maturin,
ABI3 across Python 3.9..3.13 (`crates/ruvector-py/Cargo.toml:21`).

**The contract for future bindings (Node, WASM, Java).**

- Bindings expose **only the `AnnIndex` trait surface plus persist**.
  Internal types (`BinaryCode`, `RandomRotation`) stay Rust-only —
  exposing them widens the FFI surface beyond what the determinism
  contract can survive across language runtimes.

- Persist roundtrip is the cross-language compatibility test. A `.rbpx`
  written by Rust must load identically in Python; a `.rbpx` written
  by Python must load identically in Rust. The
  `persist::tests::serialize_roundtrip_preserves_search_results` test
  is the in-Rust version; the cross-language version is a
  cross-runtime test (the Python SDK already does the round-trip in
  its test suite, just within Python).

- WASM bindings inherit the §5 footprint constraint: no rabitq in the
  default WASM bundle unless the consumer opts in.

**The trap.** Each new binding tempted to expose more of the API. The
Python SDK got this right by exposing exactly one class
(`crates/ruvector-py/src/rabitq.rs:36`); future bindings should match.

---

## 7. The `VectorKernel` story is asymmetrical

**The state.** Trait shipped in `ruvector-rabitq` (`src/kernel.rs`).
One implementation (`CpuKernel`). **Zero callers** that wire dispatch
— only a doc comment at `crates/ruvector-rulake/src/lake.rs:595`.
That's a real gap.

**The implication for new integrations.** A consumer that uses
Pattern 2 (§03) is **the first non-test caller of `VectorKernel`**.
That consumer must:

- Implement the dispatch policy from ADR-157 §"Dispatch policy
  normative" (preference order, batch-size + dim + determinism filter).
- Decide where to surface kernel identity in stats (the comment in
  `src/kernel.rs:23-25` says "kernel identity is surfaced in caps +
  stats, not in the witness" — caller's responsibility).
- Write the test that verifies determinism across two registered
  kernels on Fresh/Frozen consistency.

This is real engineering — Phase 2 of §05 explicitly budgets it. A
consumer that thinks it's getting "free GPU" by adopting the trait
is going to be disappointed unless someone has done this work first.

**The graceful path.** `ruvector-rulake` should be that someone. It
already references the trait in the doc comment; making the dispatch
real in rulake first means every other Pattern-2 consumer inherits a
working pattern and a test harness.

---

## 8. The witness chain is anchored on data, not on kernels

**Restated from ADR-157 §"Determinism as a hard gate":** the
witness is computed over `(data_ref, dim, rotation_seed,
rerank_factor, generation)`. Kernel identity is **not** in the
witness — kernels are execution substrate.

**The contract for new integrations.** A consumer that adds a new
kernel does *not* invalidate any existing witness. A consumer that
changes the rotation seed, the rerank factor, or the data does. New
integrations must not couple kernel selection to data identity — that
includes "use a different rotation seed for the GPU path because it
benchmarks better at that seed", which is a ruled-out direction.

This is what makes Phase 2's GPU work safe: a CUDA kernel that ships
later does not break already-published bundles.
