# Does a Rust Ed25519 witness-signing scheme survive compilation to WebAssembly? We measured it.

## Problem

`ruvector-agent-memory`'s witness-chain signing design (SHA-256-chained
spans, Ed25519-signed, with a `PerRecord` vs `BatchTail` latency
tradeoff measured natively at 24x-36x in favor of batching) has been
built and benchmarked three times on native x86_64. Three separate
nightly research documents, across two related crates, flagged the same
open question and left it unanswered: is this WASM-compatible, and what
does that cost? Nobody had actually built a `wasm32-unknown-unknown`
target and measured it.

## Hypothesis

Port the signing *primitive* (not the full ledger, which depends on
non-trivially-wasm-portable data structures) to a minimal, dependency-light
crate, compile it for `wasm32-unknown-unknown`, and measure it inside a
`wasmtime` sandbox against a native build of the identical source. If the
module stays under 1 MiB, the batching advantage survives sandboxing at
>= 5x, and sandboxing overhead stays under 10x native, the design is
WASM-portable at an edge/browser-appropriate cost. If any of those fail,
it isn't -- and that's a real, useful answer too.

## Technical Design

`crates/ruvector-witness-wasm` depends only on `rvf-types` (which is
already `no_std`/`alloc`-compatible) for its `ed25519` and `sha256`
modules. It re-implements the exact span-signing message layout from
`ruvector-agent-memory::witness_signing` (ADR-347) -- `purpose`, `from`,
`to`, `prev_link`, `records_digest`, chained span-to-span -- against a
deterministic synthetic 64-byte record, under a domain tag distinct from
the real ledger's, so a benchmark signature can never be mistaken for a
production one.

The crate builds two ways from one source:

- As an `rlib`, linked directly into a native benchmark binary (no
  sandbox) for the baseline.
- As a `cdylib` targeting `wasm32-unknown-unknown`, with two
  `#[no_mangle] extern "C"` exports (`wasm_run_workload`,
  `wasm_tamper_check`) that take only `i32`/`u32` primitives -- no
  pointers into wasm linear memory, so there's no cross-boundary memory
  marshaling to get wrong.

The benchmark harness (`examples/wasm_witness_bench.rs`) is itself a
normal native Rust binary. It reads the compiled `.wasm` file's bytes
(binary-size metric), uses the `wasmtime` crate -- a Rust library, not a
JavaScript runtime -- to compile and instantiate the module (startup-cost
metrics), then calls the exported functions through a typed `wasmtime`
handle, timing each call. No JavaScript, Node, or `wasm-bindgen` glue
appears anywhere in this path.

## Actual Implementation

```rust
pub enum Strategy { PerRecord = 1, BatchTail = 2 }

pub fn run_workload(strategy: Strategy, n: u32, batch_size: u32) -> WorkloadResult {
    // chain n deterministic 64-byte records into spans (1 record for
    // PerRecord, batch_size records for BatchTail), SHA-256-digest each
    // span, Ed25519-sign the digest+chain-link, verify immediately.
}

pub fn tamper_check(strategy: Strategy, n: u32, batch_size: u32) -> bool {
    // build the full signed span chain, flip one byte in the middle
    // record, recompute that span's digest from the (now tampered)
    // records, and confirm it no longer matches the signed digest --
    // the same "diligent forgery" property ADR-347's own test checks.
}
```

The `wasm32`-only exports pack the two `u32` results of `run_workload`
into a single `i64` return value, so the benchmark crosses the host/guest
boundary exactly once per whole workload, not once per record -- crossing
it per-record would measure `wasmtime` call overhead, not signing cost.

## Actual Benchmark Evidence

Three independent `cargo run --release` executions, `N = 5,000`
synthetic records per call, 15 repeats per variant, this session's Linux
x86_64 container, `rustc 1.97.0`:

```text
wasm module size (release, no wasm-opt): 85,445 bytes (83.4 KiB)
wasmtime module compile time:             138 - 182 ms
wasmtime instantiate time:                0.05 - 0.08 ms
wasm linear memory after instantiate:     17 pages (1,088 KiB)

native_per_record    mean/op:  138.0 - 145.4 us
native_batch_tail64  mean/op:    2.5 us
wasm_per_record       mean/op:  521.7 - 541.4 us
wasm_batch_tail64     mean/op:    8.2 - 8.3 us

native amortization (per_record / batch_tail64): 55.4x - 58.2x
wasm amortization   (per_record / batch_tail64): 63.3x - 64.9x
wasm/native overhead (per_record):                3.72x - 3.82x
wasm/native overhead (batch_tail64):              3.31x - 3.34x

Diligent-forgery detection: 12/12 trials (3 runs x 4 variants), 0 missed.
```

All acceptance gates fixed before the benchmark ran (size < 1 MiB,
wasm amortization >= 5x, wasm/native overhead <= 10x, 100% tamper
detection) pass, consistently, across all three runs. `cargo test`: 5/5.
`cargo clippy -- -D warnings`: clean.

## Limitations

- `wasmtime` was measured, not a browser `WebAssembly` engine -- a
  reasonable edge-deployment proxy, not a guarantee of browser parity.
- Per-call (not per-record) boundary crossing means no true per-op tail
  latency inside wasm was captured, only whole-workload totals divided
  by record count.
- No `wasm-opt` pass was available in this environment; 83.4 KiB is an
  upper bound on achievable size, not a floor.
- The full `ruvector-agent-memory::witness_signing` module (coverage
  checks, rollback anchors, unsigned-tail detection) was not ported --
  only its core signing/verification/tamper-detection primitive.

## Production Relevance

An 83.4 KiB WASM module with ~8 microsecond-per-record amortized signing
cost is a realistic artifact for a browser extension, an edge appliance,
or an embedded WASI host -- well inside the budget any of those
environments would need for tamper-evident provenance without a native
binary. The measured 3.3x-3.8x sandboxing overhead is the real price of
that portability, and it is small enough that `BatchTail`'s batching
advantage (which this run confirms survives sandboxing, even slightly
amplified: 63-65x inside wasm vs. 55-58x natively in this run) remains
the dominant lever for a deployment that needs both portability and
throughput.

## RuVector Ecosystem Implications

This closes an item three separate nightly research documents flagged
and never measured, across two sibling signed-provenance lineages
(`ruvector-retrieval-receipt` and `ruvector-agent-memory::witness_signing`).
It also gives RVF's self-bootstrapping `WASM_SEG` format
(`rvf_types::wasm_bootstrap`) its first concrete, measured candidate
module: a witness-chain verifier that could ship embedded inside a
`.rvf` file's `Extension` segment, needing no native host dependency to
run.

## Future Direction

The clearest next steps are (1) repeating this exact methodology against
`ruvector-retrieval-receipt`'s own signing primitive, which has the
identical open item on record twice already, and (2) measuring the same
compiled module inside a real browser engine rather than `wasmtime`, to
close the one honest gap this run's own methodology could not close
in a server-side container.

## References

- ADR-361 (this run), ADR-347 (the native design ported), ADR-340/343
  (the sibling lineage that first raised this question).
- `crates/ruvector-witness-wasm/` -- full source.
- `docs/research/nightly/2026-11-16-witness-wasm-portability/raw-runs.txt`
  -- unedited benchmark output, three runs.
