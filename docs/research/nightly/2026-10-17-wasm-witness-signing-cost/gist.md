# What Does Ed25519 Signing Actually Cost on an 8 KB WASM Edge Tile?

## Problem

RuVector's `rvf-wasm` crate is a `#![no_std]` WebAssembly microkernel built
for "Cognitum tiles" — constrained edge devices — with a stated target of
under 8 KB after `wasm-opt`. Two other crates in the same codebase,
`ruvector-retrieval-receipt` and `ruvector-agent-memory`, both sign data
with Ed25519 for tamper-evident audit trails, and both crates' own
benchmark reports have, on three separate occasions, flagged the same
unanswered question: what would this cost if it ran on the WASM tile
itself? Nobody had built the artifact to find out.

## Hypothesis

Add Ed25519 signing to `rvf-wasm` as an opt-in, default-off feature, using
the exact primitive (`rvf_types::ed25519`) the other two crates already
depend on. Measure, against real build output:

1. Does the default (feature-off) tile stay byte-identical? (Regression
   gate.)
2. Does the compiled size grow by less than 60 KB after `wasm-opt -Oz` —
   the tile's own stated sizing methodology?
3. Does sign/verify latency inside a real WASM runtime stay within 5x of
   native?

All three thresholds were fixed before any benchmark ran.

## Technical Design

The feature adds two `#[no_mangle]` exports,
`rvf_witness_ed25519_sign`/`rvf_witness_ed25519_verify`, that read a key,
message, and (for verify) a signature out of the tile's existing
2048-byte "Message I/O buffer" — no new static memory, no change to the
tile's 8 KB `DATA_MEMORY` layout. Both call `rvf_types::ed25519` directly,
the same primitive `ruvector-retrieval-receipt` and `ruvector-agent-memory`
use natively.

To measure latency honestly, a `wasmtime`-hosted Rust benchmark
(`crates/rvf/benches/benches/witness_wasm_bench.rs`) loads the actual
compiled `.wasm` file, instantiates it once, and calls the exported
functions in a loop via Cranelift-JIT-compiled host bindings — not an
estimate, a real WASM execution — compared against the same primitive
called natively in the same process.

## Actual Implementation

- `rvf-wasm/Cargo.toml`: `witness-signing = ["rvf-types/ed25519"]`, off by
  default.
- `rvf-wasm/src/witness.rs`: the two exports plus two tests — a sign→verify
  round trip with a deliberate signature-byte flip that must be rejected,
  and length-validation tests.
- `rvf-benches/benches/witness_wasm_bench.rs`: a criterion benchmark with a
  startup cross-compatibility assertion (native-signed verifies in wasm,
  wasm-signed verifies natively, and the two signatures are byte-identical,
  since Ed25519 is deterministic).

One bug surfaced and fixed during implementation itself: `rvf-wasm` already
has an unrelated `rvf_witness_verify` export (hash-chain verification, a
different concern). The linker caught the name collision on first build;
the new exports were renamed to `rvf_witness_ed25519_{sign,verify}`.

## Actual Benchmark Evidence

Hardware/software: `rustc 1.97.0`, `wasm-opt 108`, `wasmtime 27.0.0`
(Cranelift), Linux x86_64, single-threaded, 96-byte test message.

**Size** (bytes, `wasm-opt -Oz`):

| | Raw | `-Oz` |
|---|---|---|
| Baseline (feature off) | 51,619 | 39,087 |
| Feature on | 126,299 | 103,793 |
| Delta | 74,680 | **64,706** |

**Latency** (criterion median, µs):

| | Native | wasmtime | Ratio |
|---|---|---|---|
| Sign | 32.64 | 286.66 | **8.78x** |
| Verify | 45.24 | 162.92 | **3.60x** |

**Correctness**: `cargo test -p rvf-wasm --lib` — 6/6 passing with the
feature off (unchanged from before this change), 8/8 with it on. Native↔wasm
cross-compatibility assertion: pass.

## Result, Reported Honestly

The regression gate and wire-compatibility claims hold. The size budget
(64.7 KB > 60 KB) and the signing-latency budget (8.78x > 5x) both fail.
The verification-latency budget (3.60x ≤ 5x) holds. This is reported as a
mixed, partially-falsified result — the thresholds were not relaxed after
seeing the numbers.

A side-finding, unrelated to this change but worth recording: the
*baseline* tile (feature off, today's shipped code, re-verified via
`git stash` to be identical to the current main lineage) measures 39 KB
after `wasm-opt -Oz` — not the "<8 KB" its own module doc claims. This run
did not cause and did not investigate that discrepancy.

## Limitations

Single-threaded only. Latency is steady-state per-call cost inside
wasmtime on x86_64 — not cold-start/instantiation cost, not literal
Cognitum hardware, not a browser's `wasm-bindgen`/JS-glue path. One fixed
message size (96 bytes), no payload-size sweep. The "why is signing worse
than verification" question is flagged, not root-caused.

## Production Relevance

A size-budgeted Cognitum tile cannot add Ed25519 signing as built today
without roughly doubling its own size, and should expect signing — not
verification — to carry the larger relative latency penalty. A tile that
only verifies received signatures pays a smaller latency cost but the same
size cost, since both operations link from the same crate today. Shipping
this as an *opt-in* feature (this run's actual deliverable) lets a
deployment make that call deliberately instead of by default.

## RuVector Ecosystem Implications

This closes a three-times-flagged open question (ADR-340, ADR-343, and the
2026-09-16 nightly run) with a number instead of another deferral, using
the exact shared primitive two other in-repo crates already depend on.
Native↔wasm signature determinism, now measured rather than assumed, is a
concrete building block for RVF's deterministic-replay and edge-portability
goals.

## Future Direction

1. Root-cause the sign-vs-verify latency asymmetry.
2. Evaluate a smaller `no_std` Ed25519 implementation against this run's
   exact harness to see if the size budget becomes reachable.
3. Measure cold-start cost separately from per-call cost.
4. Wire this primitive into `rvf-crypto::witness`'s hash-chain format
   end-to-end.
5. Replace the `wasmtime`-on-x86_64 proxy with a real Cognitum-hardware or
   browser measurement.

## References

- ADR-361 (this run), ADR-340, ADR-343.
- `docs/research/nightly/2026-08-31-signed-retrieval-receipts/`,
  `2026-09-01-signed-receipt-batch-fill-latency/`,
  `2026-09-16-witness-signer-agent-memory/` — the lineage this run closes.
- `crates/rvf/rvf-types/src/ed25519.rs`, `crates/rvf/rvf-wasm/src/witness.rs`,
  `crates/rvf/benches/benches/witness_wasm_bench.rs`.
