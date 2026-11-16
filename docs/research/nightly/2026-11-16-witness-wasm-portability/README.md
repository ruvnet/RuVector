# Nightly Research: WASM32 Portability of the ADR-347 Span-Signed Witness Chain

**Date:** 2026-11-16
**Slug:** `witness-wasm-portability`
**ADR:** [ADR-361](../../../adr/ADR-361-witness-signing-wasm-portability.md)
**Crate:** `ruvector-witness-wasm` (new)
**Acceptance:** **ACCEPT** -- see [Acceptance Result](#acceptance-result)

## Summary

Three prior nightly runs (`2026-08-31-signed-retrieval-receipts`,
`2026-09-01-signed-receipt-batch-fill-latency`, and most recently
`2026-09-16-witness-signer-agent-memory`, ADR-340/343/347) each flagged
the same open question and each deferred it: is `rvf-types`'s
Ed25519/SHA-256 signing primitive -- and the `PerRecord`-vs-`BatchTail`
span-signing scheme built on it -- actually WASM-portable, and what does
that cost in binary size and sandboxed latency? This is the third time
the question was raised and the first time it was actually measured.

This run builds `crates/ruvector-witness-wasm`, a minimal, self-contained
port of ADR-347's span-signing design (same message layout, same two
strategies, distinct domain tags), compiles it to
`wasm32-unknown-unknown`, and benchmarks it inside a `wasmtime` sandbox
against a native build of the identical Rust functions -- a pure-Rust
host harness, no JavaScript or `wasm-bindgen` anywhere in the path. The
result closes the gap with real numbers: an 83.4 KiB release module,
BatchTail amortization intact at 63-65x inside wasm (even stronger than
native's 55-58x in this run), and wasmtime sandboxing overhead of
3.3x-3.8x over native -- comfortably inside every threshold fixed before
the benchmark ran.

## Abstract

`ruvector-agent-memory`'s ADR-347 `WitnessSigner` gives the TARL ledger
tamper-evident, Ed25519-signed witness spans, with a measured 24x-36x
native latency advantage for batch-amortized signing over per-record
signing. Three nightly documents name "whether this is WASM-compatible,
and at what cost" as an open item and none measure it, because
`ruvector-agent-memory` itself is not a small wasm32 port candidate (it
pulls in `DashMap` and `roaring`, neither audited for `wasm32-unknown-unknown`
here). This run isolates the actual primitive in question -- SHA-256
span-chaining plus Ed25519 span-signing -- into a new, dependency-light
crate (`ruvector-witness-wasm`, depending only on `rvf-types`), compiles
it for `wasm32-unknown-unknown`, and uses the `wasmtime` crate (a normal
Rust host dependency, not a JS toolchain) to load, instantiate, and time
the compiled module against a direct native call into the same source
functions. Three independent `cargo run --release` executions give
consistent binary size, latency, and correctness results.

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
diligent-forgery detection in both native and wasm execution, for both
strategies.
```

Thresholds (1 MiB size, >= 5x amortization retained, <= 10x sandbox
overhead) were fixed in the code before the first benchmark run (see the
`examples/wasm_witness_bench.rs` source, which hard-codes the gates
before printing any result) and were not adjusted after seeing results.

## Why This Matters Now (2026)

Every signed-provenance nightly run to date (retrieval receipts, state
root anchoring, witness signing) has been measured natively only. The
ecosystem's own stated direction -- RVF's self-bootstrapping `WASM_SEG`
format (`rvf_types::wasm_bootstrap`), edge/Cognitum deployment, browser
agent memory -- depends on these primitives actually running outside a
native server process. Leaving that question open a third time without
measuring it would mean three ecosystem components (retrieval receipts,
witness signing, and now this run) all silently assume WASM
compatibility that nobody has verified.

## Long Horizon Thesis

**2036:** agent memory and retrieval provenance move to the edge --
phones, browsers, embedded devices -- where a native Ed25519/SHA-256
binary isn't an option but a small `.wasm` module is. An 83 KiB signed
witness-verification module is a realistic artifact to ship inside a
browser extension or an RVF `WASM_SEG`; a 50 MB native binary is not.

**2046:** if RVF's "runs anywhere compute exists" vision (its own module
doc: embedding a WASM interpreter + microkernel inside the data file
itself) matures, witness verification is exactly the kind of small,
pure-compute, no-host-import module that belongs inside that bootstrap
chain. This run is the first real measurement that a signing/verification
primitive from this codebase fits that role's size and latency budget.

## RuVector Ecosystem Fit

Connects five existing pieces:

1. **`rvf-types::ed25519`/`sha256`** (ADR-320) -- the primitives ported,
   verbatim, with no modification.
2. **`ruvector-agent-memory::witness_signing`** (ADR-347) -- the design
   this run ports and whose open "Next Research" item it closes.
3. **`ruvector-retrieval-receipt`** (ADR-340/343) -- the sibling lineage
   that raised the identical open question twice before ADR-347 raised
   it a third time; this run closes it for the underlying primitive both
   lineages share.
4. **RVF `WASM_SEG`** (`rvf_types::wasm_bootstrap`, ADR-032/063) -- the
   concrete mechanism by which a module like this one could ship inside
   a self-bootstrapping `.rvf` file (not built here; see RVF
   Implications).
5. **`wasmtime`** -- used as a pure-Rust host-embedding library, not a
   browser/Node JS runtime, keeping the entire path (guest + harness) in
   Rust per this process's engineering constraints.

### MetaHarness / Flywheel / Darwin / `ruvector harness` capability discovery

Re-verified, not assumed, per this process's own rule and consistent with
every prior nightly run's finding in this repository:

- `npx metaharness --help` resolves (`metaharness@0.4.23`) to a generic
  project-*scaffolding* CLI -- not an in-repo research orchestrator. Not
  used for the same reason every prior run gave: running it would
  scaffold a new, separate project, not orchestrate work inside
  `ruvector`.
- `npx ruvector harness doctor --json` / `status --json` both fail
  (`npm error could not determine executable to run`) -- no `ruvector`
  CLI package with a `harness` subcommand resolves in this checkout.
  Unchanged from every prior nightly's finding.
- No Darwin evolutionary search was run. This experiment's parameter
  space (two strategies, one batch size, one record count) was swept
  directly and exhaustively (three repeated full runs; see
  `raw-runs.txt`), not through a bounded evolutionary search -- there is
  no "population" of candidate implementations to evolve, only one
  faithful port to measure.
- No Flywheel/witness-chain CLI is installed. This document, the ADR, and
  the crate's own `README.md` (which points back here) are the durable
  record a future nightly run would otherwise get from a Flywheel query.

## Architecture

```mermaid
flowchart TD
    subgraph guest["crates/ruvector-witness-wasm (lib, cdylib+rlib)"]
        RB["record_bytes(i)\n(deterministic xorshift64, no RNG crate)"]
        RW["run_workload(strategy, n, batch_size)\n-> WorkloadResult"]
        TC["tamper_check(strategy, n, batch_size)\n-> bool"]
        SHA["rvf_types::sha256::Sha256\n(span digest, chained link)"]
        ED["rvf_types::ed25519::{ed25519_sign, ed25519_verify}"]
        RB --> RW --> SHA & ED
        RB --> TC --> SHA & ED
        WX["#[cfg(target_arch=\"wasm32\")]\nwasm_run_workload / wasm_tamper_check\n(extern \"C\", i32/u32/i64 only)"]
        RW -.wraps.-> WX
        TC -.wraps.-> WX
    end

    subgraph harness["examples/wasm_witness_bench.rs (native binary, host target)"]
        LOAD["read .wasm bytes\n(binary-size metric)"]
        ENG["wasmtime::Engine / Module::new\n(compile-time metric)"]
        INST["Instance::new\n(instantiate-time metric)"]
        CALL["TypedFunc<(i32,u32,u32), i64/i32>\n.call() x REPEATS, timed"]
        NAT["direct native call into\nrun_workload / tamper_check\n(same source, no wasm)"]
        LOAD --> ENG --> INST --> CALL
    end

    guest -- "cargo build --target wasm32-unknown-unknown --release" --> LOAD
    guest -- "cargo dep (rlib)" --> NAT
    CALL -.compared against.-> NAT
```

## Implementation

New crate `crates/ruvector-witness-wasm/`:

- `src/lib.rs` (~300 lines) -- `Strategy`, `SignedSpan`, deterministic
  `record_bytes`, `build_message` (same field layout as
  `witness_signing.rs`'s `SignedSpan::message`, distinct domain tags),
  `run_workload`, `tamper_check`, and the `wasm32`-gated `extern "C"`
  exports. 5 unit tests (native target): honest runs for both strategies,
  tamper detection for both strategies, record-byte determinism.
- `examples/wasm_witness_bench.rs` (~190 lines) -- the benchmark harness.
  `wasmtime` and `anyhow` are `dev-dependencies` only (not linked into the
  `cdylib`/`rlib` lib target); the example builds for the host target by
  default, exactly where `wasmtime` needs to run.
- `README.md` -- crate-level pointer back to this document.
- Registered in the workspace root `Cargo.toml` member list, next to
  `ruvector-retrieval-receipt`.

No changes to any existing crate.

## Benchmark Methodology

- Workload: `N = 5,000` synthetic 64-byte records per call (smaller than
  ADR-347's native-only `N = 20,000` so that `REPEATS` whole-workload
  calls *through wasmtime* finish in a practical time; still large enough
  for stable microsecond-scale per-op means).
- `REPEATS = 15` independent whole-workload calls per variant; percentiles
  are computed across those 15 whole-call totals (each divided by `N` for
  a per-op figure), not across individual signing operations within one
  call -- a coarser-grained percentile than ADR-347's native-only
  within-run percentiles, chosen because the wasm variants are measured
  as one host-to-guest call per whole workload (crossing the wasm
  boundary once per *record* would measure call overhead, not signing
  cost; see Failure Modes).
- Four variants per run: `native_per_record`, `native_batch_tail64` (both
  direct Rust calls, no sandbox), `wasm_per_record`, `wasm_batch_tail64`
  (both via one `wasmtime` `TypedFunc::call` per repeat, inside a
  `wasmtime::Store` and `Instance` created once and reused across all
  repeats -- instantiation cost is measured separately, once, not folded
  into the per-call timing).
- Deterministic key (`FIXED_SECRET = [11u8; 32]`) and deterministic
  record generation (xorshift64, no RNG crate) -- identical inputs across
  all four variants and across repeated runs.
- Hardware/software: this session's container, Linux x86_64, `rustc
  1.97.0`, `cargo 1.97.0`, `wasmtime 36.0.20` (bumped from the initially
  resolved `27.0.0` after CI's Security audit job flagged several
  RUSTSEC advisories against that version; `36.0.20` clears every
  advisory's fixed-version range and was re-verified to reproduce the
  same results -- see the fourth run appended to `raw-runs.txt`),
  release profile (`opt-level = 3, lto = "fat",
  codegen-units = 1, strip = true`, workspace-wide).
- Binary size: raw `wasm32-unknown-unknown` release `.wasm` file size via
  `std::fs::metadata`-equivalent (`std::fs::read(...).len()`); no
  `wasm-opt` pass applied (not installed in this environment -- flagged
  as a limitation, not fabricated).

Reproduce exactly:

```bash
cargo test -p ruvector-witness-wasm
cargo build --target wasm32-unknown-unknown --release -p ruvector-witness-wasm
cargo run --release -p ruvector-witness-wasm --example wasm_witness_bench
```

## Benchmark Results (raw, three independent executions)

See `raw-runs.txt` for full unedited output. Cross-run summary:

```text
wasm module size:                    85,445 bytes (83.4 KiB), identical all 3 runs
wasmtime compile time:                138 - 182 ms  (run-to-run scheduling noise)
wasmtime instantiate time:            0.05 - 0.08 ms
wasm linear memory after instantiate: 17 pages (1,088 KiB)

native_per_record    mean/op:   138.0 - 145.4 us   (signatures = 5,000/5,000 records)
native_batch_tail64  mean/op:     2.5 us            (signatures =    79/5,000 records)
wasm_per_record      mean/op:   521.7 - 541.4 us   (signatures = 5,000/5,000 records)
wasm_batch_tail64    mean/op:     8.2 - 8.3 us      (signatures =    79/5,000 records)

native amortization (per_record / batch_tail64):  55.4x - 58.2x
wasm amortization   (per_record / batch_tail64):  63.3x - 64.9x
wasm/native overhead (per_record):                 3.72x - 3.82x
wasm/native overhead (batch_tail64):               3.31x - 3.34x

Diligent-forgery detection: 12/12 trials (3 runs x 4 variants) correctly detected, 0 missed.
```

`cargo test -p ruvector-witness-wasm --lib`: 5/5 pass, all three runs.
`cargo clippy -p ruvector-witness-wasm --all-targets -- -D warnings`:
clean. `cargo fmt -p ruvector-witness-wasm -- --check`: clean.

## Acceptance Result

| Gate | Threshold | Measured (range across 3 runs) | Result |
|---|---|---|---|
| WASM module size | < 1 MiB | 83.4 KiB (all 3 runs) | PASS |
| WASM BatchTail amortization retained | >= 5x | 63.3x - 64.9x | PASS |
| WASM/native overhead, PerRecord | <= 10x | 3.72x - 3.82x | PASS |
| WASM/native overhead, BatchTail64 | <= 10x | 3.31x - 3.34x | PASS |
| Diligent-forgery detection | 100% (all variants, all runs) | 12/12 | PASS |
| Honest-run verification failures | 0 | 0 (all 4 variants, all 3 runs) | PASS |
| Pre-existing tests regressed | 0 | 0 (new, isolated crate) | PASS |

**ACCEPT.** All mandatory gates pass on real, reproducible,
`cargo run --release` measured evidence across three independent
executions.

## Memory Math

- `SignedSpan` (this crate's reduced version): 1 (purpose) + 4 + 4 (u32
  `from`/`to`) + 32 (digest) + 64 (signature) = 105 bytes per span, held
  only transiently during `tamper_check`'s span list (not persisted
  anywhere -- this crate has no storage layer).
- `PerRecord` at `N` records: `N` spans (one per record). `BatchTail{64}`
  at `N` records: `ceil(N/64)` spans -- at `N = 5,000`, 79 spans, matching
  the measured `signatures = 79` in every run.
- WASM guest linear memory after instantiation: 17 pages x 64 KiB = 1,088
  KiB -- this is `wasmtime`'s allocated address space for the instance,
  not peak heap usage during the workload (not separately instrumented
  in this pass; flagged under Limitations).

## Performance Math

Native `PerRecord` at ~138-145 us/op for one SHA-256 span-digest + one
Ed25519 sign + one Ed25519 verify per record is consistent with
ADR-347's own native per-op figures (that run measured ~70-137 us for
sign-only, two records per logical "op"; this run's extra per-record
verify call roughly accounts for the difference). The WASM overhead
(3.3x-3.8x) is markedly smaller than the acceptance gate's 10x ceiling,
and is plausibly dominated by `ed25519-dalek`'s scalar-multiplication
inner loop running through `wasmtime`'s Cranelift-compiled code instead
of natively-compiled x86_64 with full LTO inlining across the call
boundary -- consistent with, not contradicted by, published general
findings that WASM arithmetic-heavy code commonly runs within a small
single-digit multiple of native speed on a modern JIT.

## Failure Modes

1. **Per-call granularity, not per-op.** Crossing the wasm host/guest
   boundary once per *workload* (not once per record) was a deliberate
   design choice to avoid measuring `wasmtime` call overhead instead of
   signing cost; it also means this run cannot report true p99 *signing*
   latency inside wasm, only p99 across 15 whole-workload totals. A
   finer-grained (per-record boundary crossing) measurement would answer
   a different, also-useful question (call-overhead-dominated worst case)
   not attempted here.
2. **Compile time is not amortized across the benchmark's own repeats**
   (measured once, outside the timed loop) but *would* be paid on every
   cold start in a real deployment (e.g., a fresh browser tab). 138-182
   ms is a real, if modest, cold-start cost -- not hidden, reported
   explicitly as "wasmtime module compile time."
3. **No fault-injection or concurrent-access measurement** -- this
   benchmark is single-threaded and does not test what happens if a
   `wasmtime::Store` is driven concurrently (not a supported use of that
   API without additional synchronization; not attempted here).

## What This Run Does Not Claim

- That `ruvector-agent-memory::witness_signing` itself (the real TARL
  ledger module, not this crate's reduced port) is WASM-compatible --
  that would require auditing `DashMap`/`roaring`, not attempted here.
- That `wasmtime`'s overhead figures transfer to a browser's native
  `WebAssembly` engine (V8, SpiderMonkey, JavaScriptCore) -- those were
  not measured; `wasmtime` is a server/CLI-oriented WASM runtime, a
  reasonable edge-deployment proxy but not identical to a browser engine.
- That 83.4 KiB is close to a size-optimized floor -- `wasm-opt` was not
  available in this environment to additionally strip the module, and a
  `#![no_std]` variant was not attempted (both flagged as Next Research).

## Rejected Alternatives

- **Port the full `ruvector-agent-memory::witness_signing` module.**
  Rejected for this pass: would require first auditing `DashMap`/
  `roaring` for wasm32 compatibility, a materially larger scope than the
  actually-open question (the primitive's cost), which three prior
  nightly runs named specifically.
- **Drive the benchmark through Node + `wasm-bindgen`.** Rejected: adds a
  JavaScript toolchain dependency for no measurement benefit over
  `wasmtime`'s Rust embedding API, and conflicts with this process's
  preference for an all-Rust path.
- **Skip the native comparison, report only wasm numbers.** Rejected:
  "wasm/native overhead" is the actionable number for a deployment
  deciding whether to accept the WASM path's cost; reporting wasm numbers
  in isolation would not answer that question.

## Security

- No new cryptographic primitive; reuses `rvf-types::ed25519`/`sha256`
  verbatim.
- Distinct domain tags (`ruvector-witness-wasm:v1:*`) from
  `ruvector-agent-memory::witness_signing`'s `v2` tags -- a signature
  produced by this crate's benchmark can never be mistaken for, or
  replayed as, a real TARL ledger signature.
- `FIXED_SECRET = [11u8; 32]` is a public, benchmark-only constant printed
  in this document and the crate's source. It must never be used for any
  real signing; this crate is a measurement harness, not production code.
- The `wasm32` exports exchange only `i32`/`u32`/`i64` primitives across
  the host/guest boundary -- no pointers into wasm linear memory are
  exposed, so there is no memory-marshaling attack surface introduced by
  this ABI.

## Governance

None beyond the existing "reuse don't fork cryptographic primitives"
convention (ADR-320/340/347).

## MCP Implications

Not exposed via MCP. A narrow, read-only
`witness_wasm_bench(n, batch_size) -> BenchmarkReport` tool is a
plausible future wrapper (no mutation authority, pure measurement), but
Step 30 requires the analysis, not that every capability get a tool --
not built here.

## WASM Implications

This *is* the WASM measurement three prior nightly runs deferred: 83.4
KiB release size, 3.3x-3.8x sandboxed latency overhead versus native,
~150 ms one-time module-compile cost, ~1 MiB instance linear memory. All
four numbers are new, real data points this codebase did not have before
this run.

## Edge Implications

An 83.4 KiB module with sub-10us-per-op `BatchTail` signing cost (even
inside a sandbox) is well within budget for a constrained edge device
with a WASM runtime (e.g., a Cognitum-class appliance or a WASI-capable
microcontroller host) -- the ~150 ms one-time compile cost matters most
for a device that cold-starts frequently; a long-lived edge process pays
it once. `PerRecord`'s ~530 us/op inside wasm is likely still too slow
for a high-frequency edge signing workload, mirroring ADR-347's own
native-side guidance to prefer `BatchTail` wherever throughput matters.

## RVF Implications

`rvf_types::wasm_bootstrap` already defines a `WasmRole::Extension`/
`ControlPlane` `WASM_SEG` role for exactly this kind of small, pure-compute
module embedded inside a self-bootstrapping `.rvf` file. This run's 83.4
KiB module is a concrete, measured candidate for that role -- a
witness-chain verifier that an `.rvf` file could carry and execute
without any native dependency on the host. Not implemented (no `.rvf`
packaging was built in this pass); flagged as a direct, now-de-risked
integration path.

## RVM Implications

Marginal, same conclusion ADR-347 reached for its own
`verify_signed_chain`: `run_workload`'s verification half (and
`tamper_check`) are pure, side-effect-free functions, a natural fit for
an RVM proof-gated read path if this primitive is ever promoted into a
coherence domain -- but nothing about this specific result requires RVM
enforcement today. No forced integration proposed.

## ruFlo Implications

A concrete workflow: a scheduled ruFlo job that rebuilds this crate's
`wasm32-unknown-unknown` artifact on every `rvf-types` release and
re-runs this benchmark, alerting if the size or latency gates regress
(e.g., a new `ed25519-dalek` version adds WASM-unfriendly code paths) --
directly analogous to the "scheduled verification job" workflow the
2026-09-16 nightly named for `verify_signed_chain`, now extended to a
build-and-size regression check specific to WASM artifacts.

## Practical Applications

1. **User**: a browser-extension or in-browser agent-memory product.
   **Problem**: needs to verify a signed witness chain client-side
   without a server round trip. **Capability**: this crate's wasm32
   module (or a production equivalent built the same way). **Ecosystem**:
   `rvf-types`. **Path**: compile, ship the 83 KiB module, call
   `wasm_run_workload`/a real verify-only export. **Value**: client-side
   tamper detection with no server trust required. **Risk**: this run
   used `wasmtime`, not a browser engine -- browser-specific measurement
   is unverified. **Horizon**: near-term.
2. **User**: an edge/Cognitum device operator. **Problem**: needs signed
   provenance on a device too constrained for a full native binary.
   **Capability**: the measured 83.4 KiB module + `BatchTail`'s sub-10us
   amortized cost. **Ecosystem**: edge/Cognitum. **Path**: pick
   `batch_size` from this run's (and ADR-347's native) amortization
   curve. **Value**: tamper evidence at edge-appropriate footprint.
   **Risk**: ~150 ms cold-start compile cost on a frequently-restarting
   device. **Horizon**: near-term.
3. **User**: an RVF export author. **Problem**: wants a self-verifying
   `.rvf` file that doesn't depend on the host having any native crypto
   library. **Capability**: this module as a `WasmRole::Extension`
   `WASM_SEG`. **Ecosystem**: RVF. **Path**: wire this (or a hardened
   equivalent) into `rvf_types::wasm_bootstrap`'s packaging. **Value**:
   genuinely portable, host-independent verification. **Risk**: RVF
   packaging itself is unbuilt here. **Horizon**: medium-term.
4. **User**: the `ruvector-retrieval-receipt` maintainers. **Problem**:
   their own ADR-340/343 nightly runs flagged this exact open question
   twice and never measured it. **Capability**: this run's methodology
   (isolate the primitive, port it standalone, measure with `wasmtime`)
   transfers directly. **Ecosystem**: `ruvector-retrieval-receipt`.
   **Path**: apply the same reduced-port pattern to their signing
   primitive. **Value**: closes a two-nightly-old open item by reusing a
   now-proven method. **Risk**: none new. **Horizon**: immediate.
5. **User**: a security researcher auditing WASM supply-chain risk.
   **Problem**: wants to know whether a cryptographic primitive behaves
   identically across native and WASM targets. **Capability**: this
   crate's shared-source design (same `run_workload`/`tamper_check`
   functions compiled for both targets) makes that comparison exact, not
   approximate. **Ecosystem**: general audit tooling. **Path**: `cargo
   test` (native) plus the wasmtime harness (WASM). **Value**: an
   executable, re-runnable equivalence check. **Risk**: none. **Horizon**:
   immediate.
6. **User**: a multi-platform agent SDK vendor. **Problem**: wants one
   signing implementation that runs identically on server, edge, and
   browser. **Capability**: `rvf-types`'s `no_std`/`alloc` design plus
   this run's proof that it compiles and performs acceptably for
   `wasm32-unknown-unknown`. **Ecosystem**: `rvf-types`. **Path**: ship
   the same crate across targets. **Value**: one audited implementation,
   not three. **Risk**: browser-engine-specific behavior unverified.
   **Horizon**: medium-term.
7. **User**: a compliance team needing offline-verifiable records.
   **Problem**: wants a verifier that runs on an auditor's laptop without
   installing the full `ruvector` toolchain. **Capability**: an 83 KiB
   `.wasm` file plus any WASM runtime (`wasmtime`, a browser, etc.).
   **Ecosystem**: general audit tooling. **Path**: ship the module
   standalone. **Value**: minimal-dependency, portable verification.
   **Risk**: none new. **Horizon**: near-term.
8. **User**: the nightly research process itself. **Problem**: three
   runs flagged the same open item without closing it, risking a fourth.
   **Capability**: this document and ADR now give a concrete, cited
   answer. **Ecosystem**: nightly Flywheel (document-based, no CLI
   installed). **Path**: cite this ADR in any future run that would
   otherwise re-flag the same question. **Value**: prevents redundant
   future work. **Risk**: none. **Horizon**: immediate.

## Long Horizon Applications

1. **Thesis**: agent memory that runs entirely client-side (browser or
   edge), with server-independent provenance verification. **Required
   advances**: browser-engine-specific measurement (not just `wasmtime`),
   key distribution. **RuVector role**: this primitive is the verification
   substrate. **Why this experiment matters**: establishes the size/cost
   budget is realistic. **Primary uncertainty**: browser engine overhead
   vs. `wasmtime`'s. **Falsification path**: measuring materially worse
   numbers in a real browser would falsify "wasmtime is a reasonable
   proxy."
2. **Thesis**: RVF files that are fully self-verifying and self-executing
   with zero native host dependency ("runs anywhere compute exists").
   **Required advances**: full `WASM_SEG` packaging, a WASM interpreter
   fallback for hosts without even a native WASM runtime. **RuVector
   role**: this module is a first concrete `Extension`-role candidate.
   **Why this experiment matters**: the self-bootstrap vision needs real
   modules, not just a format spec. **Primary uncertainty**: whether the
   interpreter-of-last-resort tier (`WasmRole::Interpreter`) can run this
   module's cost profile acceptably. **Falsification path**: measuring
   that tier's overhead, not attempted here.
3. **Thesis**: fleet-scale edge cognition where thousands of constrained
   devices verify shared provenance locally. **Required advances**:
   multi-issuer verification (same open item ADR-347 named), fleet key
   distribution. **RuVector role**: the per-device verification primitive
   this run validates fits the footprint. **Why this experiment matters**:
   establishes the single-device cost baseline first. **Primary
   uncertainty**: real embedded-hardware (not container) WASM runtime
   performance. **Falsification path**: measuring on actual constrained
   hardware.
4. **Thesis**: a "write once, verify anywhere" cognitive-state format
   where the same signed witness chain is checked by a server, a browser,
   and an embedded device using the identical compiled artifact. **Required
   advances**: cross-engine WASM behavioral parity guarantees. **RuVector
   role**: `rvf-types`'s shared, dependency-light design is the enabling
   choice. **Why this experiment matters**: proves one specific primitive
   compiles once and runs in two of the three target environments
   (native + wasmtime; browser untested). **Primary uncertainty**:
   browser parity. **Falsification path**: browser measurement.
5. **Thesis**: synthetic nervous systems / agent operating systems where
   every kernel-level memory write carries a WASM-portable proof.
   **Required advances**: kernel-level (RVM) enforcement using this exact
   primitive at syscall frequency. **RuVector role**: `BatchTail`'s
   measured ~8us/op cost (even inside wasm) is compatible with high write
   rates. **Why this experiment matters**: the cost profile under
   sandboxing, not just natively, is now known. **Primary uncertainty**:
   RVM's actual target write frequency. **Falsification path**: measuring
   against that real rate.
6. **Thesis**: robotics / embedded safety certification needing a signed
   sensor-fusion history verifiable on the robot's own constrained
   compute. **Required advances**: real embedded (ARM, no Ed25519
   hardware acceleration) measurement. **RuVector role**: the same
   `WitnessSink`-style composition point this run's design mirrors.
   **Why this experiment matters**: wasmtime's software-only numbers are
   a first proxy for embedded software-only cost. **Primary uncertainty**:
   embedded CPU headroom at real sensor rates. **Falsification path**:
   on-device measurement.
7. **Thesis**: scientific or audit contexts where a third party verifies
   provenance with no access to (or trust in) the original toolchain.
   **Required advances**: distributing the `.wasm` module itself as the
   artifact of record, not just source code. **RuVector role**: this
   run's module is small enough to distribute inline with a dataset or
   paper. **Why this experiment matters**: establishes the artifact is
   genuinely small and portable, a prerequisite for that distribution
   model. **Primary uncertainty**: none specific to this run. **Falsification
   path**: n/a, infrastructural.
8. **Thesis**: cross-organization federated agent memory where each
   organization's edge nodes independently verify a shared signed chain
   with no central verification service. **Required advances**:
   multi-issuer verification (as above), federated key distribution.
   **RuVector role**: the per-node verification cost this run measures
   must stay acceptable at federation scale. **Why this experiment
   matters**: single-node cost is the necessary first data point.
   **Primary uncertainty**: aggregate cost at real federation scale.
   **Falsification path**: a federation-scale load test, not attempted
   here.

## Competitor Comparison

Not directly applicable: this is an internal primitive-portability
measurement, not a retrieval or indexing capability comparable to
Milvus/Qdrant/Weaviate/Pinecone/LanceDB/FAISS/pgvector/Chroma/Vespa/DiskANN.
No comparison attempted, consistent with the equivalent finding in the
2026-09-16 nightly for the same reason.

## Evolution Results

Darwin was not run -- see "MetaHarness / Flywheel / Darwin" above. The
two-strategy, one-batch-size design space was swept directly and
exhaustively (3 full repeated runs), not evolutionarily searched.

## Promotion Decision

**Promote as a new, isolated, opt-in research/measurement crate.** No
existing crate is modified; `ruvector-witness-wasm` has no consumers yet.
Recommend:

- Treat this run's numbers as the WASM-portability answer for
  `rvf-types::ed25519`/`sha256` going forward; future nightly runs should
  cite this ADR rather than re-flagging the open item.
- If `ruvector-retrieval-receipt`'s own signing primitive needs the same
  measurement, apply this run's reduced-port methodology rather than
  attempting a full-crate WASM port.
- A production RVF `WASM_SEG` integration (Practical Application 3) is a
  legitimate follow-up, not started here.

## Witness Evidence

- Commit at run start: recorded in this branch's git history (first
  commit of this nightly run).
- All benchmark output in `raw-runs.txt` is copied verbatim from three
  `cargo run --release -p ruvector-witness-wasm --example
  wasm_witness_bench` executions in this session's container; no numbers
  were hand-edited.
- `cargo test -p ruvector-witness-wasm --lib`: 5/5 pass (run before each
  benchmark execution, same commit).
- No signed/cryptographic witness chain covers this research process
  itself (no Flywheel/witness-chain tooling installed in this
  environment, consistent with every prior nightly's finding) -- this
  document and the git commit history are the durable record.

## Production Path

1. If a real deployment needs WASM-portable witness signing, start from
   this crate's design rather than re-deriving it, but harden it first:
   add the coverage/anchor/unsigned-tail checks ADR-347's own
   post-review hardening added (this crate's `tamper_check` demonstrates
   the core detection property but does not implement the full
   `verify_signed_chain` API surface).
2. Measure a real browser engine before claiming browser-readiness (Next
   Research item 3 below).
3. Consider a `#![no_std]` variant if binary size needs to shrink further
   for a more constrained target than this run's 83.4 KiB already
   satisfies.

## Falsification Criteria

This hypothesis would have been rejected if: the `.wasm` module exceeded
1 MiB; `BatchTail`'s wasm amortization fell below 5x; wasmtime overhead
exceeded 10x over native for either strategy; any honest run produced a
verification failure; or any of the 12 tamper trials (3 runs x 4
variants) went undetected. None of these occurred.

## Limitations

- `wasmtime`, not a browser engine, measured the WASM execution cost.
- Per-call (not per-record) boundary crossing means no true per-signing-op
  tail latency inside wasm was measured (see Failure Modes item 1).
- No `wasm-opt` pass applied (not installed); the 83.4 KiB figure is an
  upper bound on achievable size, not a floor.
- No `#![no_std]` variant attempted.
- No fault-injection, concurrent-access, or real embedded-hardware
  measurement.
- `REPEATS = 15` is modest; this run reports it honestly rather than
  implying deeper statistical confidence than 15 samples support.

## Next Research

1. Measure the same module in an actual browser `WebAssembly` engine
   (not just `wasmtime`), closing the "browser vs. wasmtime" gap named in
   Limitations.
2. Apply this run's reduced-port methodology to
   `ruvector-retrieval-receipt`'s own signing primitive, closing its
   twice-flagged, still-open WASM measurement item (ADR-340/343).
3. Attempt a `#![no_std]` build of this crate and measure the additional
   size reduction.
4. Build the full `verify_signed_chain`-equivalent API (coverage check,
   rollback anchor, unsigned-tail detection) on top of this crate's
   primitive, then re-measure -- this run deliberately scoped to the
   minimal signing/verification/tamper-detection core.
5. Prototype the RVF `WASM_SEG::Extension` packaging path named in "RVF
   Implications."

## References

- `docs/adr/ADR-347-witness-signer-tarl-ledger.md` -- the native design
  this run ports and whose open WASM item it closes.
- `docs/adr/ADR-340-signed-retrieval-receipt-anchoring.md`,
  `docs/adr/ADR-343-signed-receipt-batch-fill-latency-simulation.md` --
  the sibling lineage that first flagged this open item.
- `docs/research/nightly/2026-09-16-witness-signer-agent-memory/README.md`
  -- source of this run's "Next Research item 4" starting point.
- `crates/rvf/rvf-types/src/ed25519.rs`, `src/sha256.rs`,
  `src/wasm_bootstrap.rs` -- the primitives reused and the RVF packaging
  format referenced under RVF Implications.
- `crates/ruvector-agent-memory/src/witness_signing.rs` -- the native
  design this crate's message layout mirrors.
