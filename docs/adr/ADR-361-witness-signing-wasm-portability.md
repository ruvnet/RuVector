# ADR-361: WASM32 Portability of the ADR-347 Span-Signed Witness Chain

## Status

Accepted (as a measured, opt-in library capability -- not a default
behavior change to any existing crate). 2026-11-16 nightly research run.
See `docs/research/nightly/2026-11-16-witness-wasm-portability/README.md`
for the full hypothesis, methodology, and raw benchmark output.

## Context

`ruvector-agent-memory`'s ADR-347 `WitnessSigner` (span-signed,
SHA-256-chained, Ed25519-signed TARL ledger witnesses) was benchmarked
natively three times (2026-09-16 nightly and its 2026-09-26 post-review
hardening) and twice explicitly deferred the question of WASM
compatibility and cost:

- 2026-08-31/2026-09-01 nightlies (`ruvector-retrieval-receipt`,
  ADR-340/343) flagged WASM binary-size and signing-latency measurement
  as an open item for the sibling signed-retrieval-receipt lineage.
- 2026-09-16 nightly (ADR-347, `witness_signing.rs`) flagged the
  identical open item a third time, against `ruvector-agent-memory`,
  explicitly deferring it ("Next Research item 4").

No prior nightly run has actually built and measured a WASM target for
either crate. `ruvector-agent-memory` itself is not easily portable to
`wasm32-unknown-unknown` as a whole (it depends on `DashMap` and
`roaring`, neither audited here for WASM compatibility), so this ADR does
not attempt to port that crate. It instead ports the specific primitive
the open question is actually about -- the span-signing scheme itself
(SHA-256 chaining + Ed25519 signing of `PerRecord` vs `BatchTail` spans)
-- as a new, minimal, self-contained crate with no dependency beyond
`rvf-types`, which is already `no_std`/`alloc`-compatible.

## Hypothesis

```text
Given the same span-signing design validated natively by ADR-347
(SHA-256-chained spans, Ed25519-signed, PerRecord vs BatchTail{64}
strategies), re-implemented standalone against a synthetic deterministic
record so it depends only on rvf-types,

when compiled to wasm32-unknown-unknown and executed inside a wasmtime
sandbox with no host imports beyond what the Rust std target itself
requires,

then (a) the released .wasm module is small enough for edge/browser
distribution (< 1 MiB, unoptimized), (b) BatchTail{64}'s per-op latency
advantage over PerRecord survives inside the wasm sandbox at >= 5x (the
same order of magnitude as ADR-347's native 24x-36x finding), and (c)
wasmtime-sandboxed execution costs no more than 10x the native per-op
latency for either strategy,

subject to: zero verification failures across all honest runs, and 100%
diligent-forgery detection (a tampered record's recomputed span digest
must differ from the signed one) in both native and wasm execution, for
both strategies.
```

## Decision

1. Add `crates/ruvector-witness-wasm`, a new, minimal, `cdylib` + `rlib`
   crate depending only on `rvf-types` (`alloc`, `ed25519` features,
   `default-features = false`). It ports ADR-347's span-signing message
   layout (`MSG_TAG`/`RECORDS_TAG`/span fields) under a distinct,
   crate-specific domain tag -- a signature produced here can never be
   confused with a real TARL ledger signature.
2. The crate exposes `run_workload(strategy, n, batch_size) ->
   WorkloadResult` and `tamper_check(strategy, n, batch_size) -> bool` as
   plain Rust functions (used directly by the native benchmark variant
   and unit tests), plus two `#[no_mangle] extern "C"` exports
   (`wasm_run_workload`, `wasm_tamper_check`), gated to
   `target_arch = "wasm32"`, that wrap them for the compiled `.wasm`
   module.
3. `examples/wasm_witness_bench.rs` (a normal native binary, built for
   the host target, not wasm32) is the benchmark harness: it loads the
   release `.wasm` artifact's bytes, measures its file size, uses the
   `wasmtime` crate (host-only `dev-dependency`, not a dependency of the
   `cdylib`/`rlib` lib target) to compile, instantiate, and call the
   exported functions, and compares the results against direct native
   calls into the same `run_workload`/`tamper_check` functions. No
   JavaScript, Node, or `wasm-bindgen` glue is involved anywhere in this
   path -- the guest is pure Rust compiled to `wasm32-unknown-unknown`,
   and the host harness that drives it is pure Rust using `wasmtime` as a
   library.
4. No changes to `ruvector-agent-memory` or `rvf-types`. This ADR adds a
   capability, not a modification.

## Evidence

Reproduce with:

```bash
cargo test -p ruvector-witness-wasm
cargo build --target wasm32-unknown-unknown --release -p ruvector-witness-wasm
cargo run --release -p ruvector-witness-wasm --example wasm_witness_bench
```

One real execution (`N=5000` records/call, `REPEATS=15` whole-workload
calls per variant, `BATCH_SIZE=64`, this session's container, Linux
x86_64, `rustc 1.97.0`):

```text
wasm module size (release, no wasm-opt): 85445 bytes (83.4 KiB)
wasmtime module compile time:    150.409 ms
wasmtime instance instantiate time: 0.076 ms
wasm linear memory after instantiate: 17 page(s) = 1088 KiB

native_per_record            mean=     139.7us  signatures=  5000
native_batch_tail64          mean=       2.5us  signatures=    79
wasm_per_record              mean=     533.8us  signatures=  5000
wasm_batch_tail64            mean=       8.3us  signatures=    79

native amortization (per_record / batch_tail64):     55.93x
wasm amortization   (per_record / batch_tail64):     64.16x
wasm/native overhead (per_record):                   3.82x
wasm/native overhead (batch_tail64):                 3.33x

Diligent-forgery rejection: native per_record=true, native batch_tail64=true,
wasm per_record=true, wasm batch_tail64=true

ACCEPTANCE: ACCEPT
```

See the research document for repeated-run evidence and full methodology.
`cargo test -p ruvector-witness-wasm --lib`: 5/5 pass. `cargo clippy -p
ruvector-witness-wasm --all-targets -- -D warnings`: clean. `cargo fmt -p
ruvector-witness-wasm -- --check`: clean.

## Consequences

- Confirms, with real measurement rather than assumption, that
  `rvf-types`'s `ed25519`/`sha256` primitives -- and the ADR-347
  span-signing scheme built on them -- are WASM-portable at a size and
  latency cost compatible with edge/browser deployment.
- `BatchTail`'s amortization advantage is not an artifact of native
  CPU/OS specifics: it survives, and is not eroded by, wasmtime's
  sandboxing and call overhead.
- Does **not** port `ruvector-agent-memory::witness_signing` itself to
  WASM; that would require auditing `DashMap`/`roaring` (used elsewhere
  in that crate) for `wasm32-unknown-unknown` compatibility, which this
  ADR does not attempt or claim.
- Opens a concrete, now-measured path for `RVF` to embed a signed-witness
  verifier as a `WasmRole::Extension`/`ControlPlane` `WASM_SEG` (see
  `rvf_types::wasm_bootstrap`) -- not built here.

## Alternatives

- **Port `ruvector-agent-memory::witness_signing` directly, auditing its
  full dependency tree for wasm32 compatibility.** Rejected for this
  pass: substantially larger scope (DashMap, roaring, the full
  `TransactionalLedger`/`WitnessSink` trait machinery), and the actual
  open question named by three prior nightly runs was specifically about
  the signing *primitive's* WASM cost, not the ledger's.
- **Use `wasm-bindgen` + Node to drive the benchmark.** Rejected: pulls in
  a JavaScript toolchain for no benefit when `wasmtime`'s Rust embedding
  API gives a typed, pure-Rust host harness with equal or better
  precision, and this workspace's engineering constraints prefer Rust
  end to end.
- **Measure only binary size, skip the wasmtime latency comparison.**
  Rejected: binary size alone would not answer whether the `BatchTail`
  amortization result -- the more actionable half of ADR-347's finding --
  survives sandboxing; both were measurable in one pass at acceptable
  cost.

## Implementation Plan

Already implemented in this run:
`crates/ruvector-witness-wasm/{Cargo.toml,src/lib.rs,README.md,examples/wasm_witness_bench.rs}`,
registered in the workspace `Cargo.toml` member list.

## API Shape

```rust
pub enum Strategy { PerRecord = 1, BatchTail = 2 }
pub struct WorkloadResult { pub signature_count: u32, pub verify_failures: u32 }
pub fn run_workload(strategy: Strategy, n: u32, batch_size: u32) -> WorkloadResult;
pub fn tamper_check(strategy: Strategy, n: u32, batch_size: u32) -> bool;
pub fn record_bytes(i: u32) -> [u8; 64]; // deterministic synthetic record

// wasm32 only:
extern "C" fn wasm_run_workload(strategy: i32, n: u32, batch_size: u32) -> i64;
extern "C" fn wasm_tamper_check(strategy: i32, n: u32, batch_size: u32) -> i32;
```

## Feature Flags

None. The crate has no optional features; `rvf-types` is pinned to
`default-features = false, features = ["alloc", "ed25519"]`.

## Benchmark Evidence

See "Evidence" above and the research document's "Benchmark Results"
section for repeated-run numbers.

## Security

- Reuses `rvf-types::ed25519`/`sha256` verbatim; no new cryptographic
  primitive.
- Distinct domain tags (`ruvector-witness-wasm:v1:span`/`:records`) from
  `ruvector-agent-memory::witness_signing`'s `v2` tags prevent any
  cross-crate signature confusion.
- This crate's signing key (`FIXED_SECRET = [11u8; 32]`) is a fixed,
  public, benchmark-only constant -- never use it for anything real. The
  crate is a measurement harness, not a production signing module.
- The `wasm32` exports take only `i32`/`u32` primitives, no pointers into
  wasm linear memory, so there is no host/guest memory-marshaling attack
  surface to audit.

## Governance

None beyond the existing "reuse, don't fork, cryptographic primitives"
convention already established by ADR-320/340/347.

## Failure Modes

- The `wasm_overhead_*` gates (<=10x) could fail on a different
  host CPU/wasmtime version/cranelift optimization level; the thresholds
  and this run's container's hardware are both recorded so a regression
  is checkable, not just assumed stable forever.
- `REPEATS=15` whole-workload calls per variant is modest; see the
  research document's "Limitations" for what this does and does not
  establish about tail-latency stability.

## Migration

None; this is a new, isolated, opt-in crate. No existing crate changes.

## Rollback

Remove `crates/ruvector-witness-wasm` and its workspace member entry;
no other crate depends on it.

## Rejection Criteria

This hypothesis would have been rejected if: the `.wasm` module exceeded
1 MiB, `BatchTail`'s amortization advantage fell below 5x inside wasm,
wasmtime overhead exceeded 10x over native for either strategy, or any
honest run produced a verification failure, or any tamper trial went
undetected. None of these occurred (see Evidence).

## Open Questions

1. Whether a `#![no_std]` build (no `std`, explicit `#[panic_handler]`)
   would shrink the 83.4 KiB module further, and by how much -- not
   measured here (this crate uses `std` for implementation simplicity).
2. Whether `ruvector-agent-memory::witness_signing` itself (not just the
   primitive) could be made wasm32-compatible by swapping `DashMap` for a
   wasm-safe map when the `mincut-forget`/full-ledger features are
   disabled -- not attempted here.
3. Browser (not just `wasmtime`) measurement -- this ADR's harness never
   touches a JS engine; browser `WebAssembly` instantiation/call overhead
   may differ from `wasmtime`'s and was not measured.
