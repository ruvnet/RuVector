# ADR-361: Opt-In Ed25519 Witness Signing for the `rvf-wasm` Cognitum Tile

## Status

Proposed

## Context

`crates/rvf/rvf-wasm` is the Cognitum-tile WASM microkernel: `#![no_std]`,
no allocator beyond a global `dlmalloc`, 14 `#[no_mangle]` exports, and a
module-doc target of "< 8 KB after wasm-opt." Its `Cargo.toml` already
depends on both `rvf-types` and `rvf-crypto` — the crates that host this
repository's Ed25519 signing primitives — but with
`default-features = false` on both, so **no signing code is compiled into
the shipped tile today**.

Three prior nightly research runs flagged the resulting gap without
closing it:

- `docs/research/nightly/2026-08-31-signed-retrieval-receipts/README.md`
  and `2026-09-01-signed-receipt-batch-fill-latency/README.md` (ADR-340,
  ADR-343) measure Ed25519 signing cost for `ruvector-retrieval-receipt`
  natively and defer "WASM binary-size and signing-latency measurement"
  as future work.
- `2026-09-16-witness-signer-agent-memory/README.md`'s Next Research item
  4 restates the identical open question against a second crate,
  `ruvector-agent-memory`'s `SignedWitnessSink`, noting it as "a third
  open instance of the identical question."

No prior run measured it against the actual production artifact that
would carry this cost at the edge: `rvf-wasm` itself. This ADR closes that
gap directly, using the shared `rvf_types::ed25519` primitive
(`ed25519_sign` / `ed25519_verify`, already depended on by both flagged
consumer crates) and the tile's own build profile and size target as the
measurement methodology.

## Hypothesis

```text
Given the rvf-wasm Cognitum tile, compiled to wasm32-unknown-unknown with
its existing release profile (opt-level=z, lto=true, codegen-units=1,
strip=true) and measured, as the crate's own module doc already does, by
wasm-opt -Oz output size,

when an opt-in `witness-signing` feature (default: off) is added, exposing
two new no_mangle exports (`rvf_witness_ed25519_sign`,
`rvf_witness_ed25519_verify`) that operate on the existing Message I/O
buffer region and call rvf_types::ed25519 directly,

then (a) the default build's exported-function count and compiled size
must be byte-identical to today's shipped tile, (b) the wasm-opt'd size
delta when the feature is enabled must stay under 60 KB, and (c) signing
and verification latency measured inside a real WASM runtime (wasmtime,
Cranelift backend) must stay within 5x of the equivalent native x86_64
call,

subject to: the full pre-existing rvf-wasm test suite remaining green, no
change to any existing export's signature or behavior, and native/wasm
wire-format compatibility (a signature produced on one side must verify on
the other).
```

Claim (a) and wire-format compatibility are **confirmed**. Claim (b) is
**falsified** (measured delta 64.7 KB post-wasm-opt). Claim (c) is
**confirmed for verification** (3.60x) and **falsified for signing**
(8.78x) — see the nightly research report for full measured evidence:
`docs/research/nightly/2026-10-17-wasm-witness-signing-cost/README.md`.

## Decision

1. Add `[features] witness-signing = ["rvf-types/ed25519"]` to
   `rvf-wasm/Cargo.toml`, default off.
2. Add `crates/rvf/rvf-wasm/src/witness.rs`
   (`#[cfg(feature = "witness-signing")]` in `lib.rs`), exposing
   `rvf_witness_ed25519_sign(msg_len: i32) -> i32` and
   `rvf_witness_ed25519_verify(msg_len: i32) -> i32`, both operating on the
   existing 2048-byte Message I/O buffer (no growth of `DATA_MEMORY`).
3. Add `crates/rvf/benches/benches/witness_wasm_bench.rs`, a `wasmtime`-hosted
   criterion benchmark comparing native vs. in-wasm latency for the same
   primitive, plus a startup cross-compatibility assertion.
4. Do **not** enable `witness-signing` by default, and do **not** wire it
   into any existing export. This ADR ships the opt-in primitive and its
   measured cost; it does not decide whether a specific Cognitum tile
   deployment should turn it on.
5. Do **not** attempt to shrink ed25519-dalek's WASM footprint in this ADR
   (e.g. swapping to a smaller no_std implementation). That is recorded as
   follow-up work, not attempted half-finished here (see Consequences).

## Evidence

See `docs/research/nightly/2026-10-17-wasm-witness-signing-cost/README.md`
for the full benchmark methodology, raw criterion output, and wasm-opt
size measurements. Summary:

| Claim | Threshold | Measured | Result |
|---|---|---|---|
| Default build unaffected | byte-identical | 51,619 B raw / 39,087 B `-Oz` (both unchanged) | PASS |
| Wire compatibility | sign/verify interop both directions | native→wasm and wasm→native both verify; signatures byte-identical | PASS |
| Size delta (feature on) | ≤ 60 KB (`-Oz`) | 64,706 B | **FAIL** |
| Sign latency ratio | ≤ 5x native | 8.78x (32.64 µs → 286.66 µs, median) | **FAIL** |
| Verify latency ratio | ≤ 5x native | 3.60x (45.24 µs → 162.92 µs, median) | PASS |
| Existing test suite | green | `rvf-wasm`: 6/6 (off), 8/8 (on); whole rvf workspace native build green | PASS |

## Consequences

- **Positive**: The shared `rvf_types::ed25519` primitive is now proven,
  with real measurements, to be wire-compatible between native and WASM
  execution, and the exact cost of enabling it on the actual Cognitum tile
  artifact is known rather than assumed. The three prior nightly runs'
  open question is closed with a number, not a deferral.
- **Negative**: `ed25519-dalek`'s WASM code size (curve-arithmetic lookup
  tables) is too large for this tile's size budget as currently built. A
  tile that needs witness signing at the stated <8 KB-class budget cannot
  use this feature as shipped; it needs either a smaller Ed25519
  implementation or a relaxed size budget for signing-capable tiles
  specifically.
- **Negative**: Signing latency in a real WASM runtime is materially worse
  than verification's (8.78x vs. 3.60x), which this ADR does not
  root-cause (see Open Questions).
- This ADR does **not** claim real Cognitum hardware behavior, browser
  `wasm-bindgen`/JS-glue overhead, or cold-start/instantiation cost — only
  steady-state wasmtime/Cranelift call latency on this run's x86_64 host
  (see Limitations in the nightly report).

## Alternatives

- **Route `rvf-wasm`'s signing through `rvf-crypto::sign` instead of
  `rvf_types::ed25519` directly.** Rejected for this ADR: `rvf-crypto`'s
  `ed25519-dalek` dependency declaration does not set
  `default-features = false` (unlike `rvf-types`'s, which does), so
  enabling `rvf-crypto/ed25519` in a `#![no_std]` crate risks pulling in
  `std`-only dalek defaults untested here. Using `rvf_types::ed25519`
  directly — already `no_std`-correct — avoids that risk and matches the
  primitive the two flagged consumer crates actually depend on.
- **Pre-committing a looser size/latency budget to make this ADR's
  hypothesis pass.** Rejected per process rule: thresholds were fixed
  before benchmarking and are not weakened after seeing the result ("FAIL"
  is reported as FAIL).

## Security

No new attack surface beyond the primitive's existing one:
`rvf_witness_ed25519_sign`/`verify` take caller-controlled fixed-offset
buffers in the tile's own linear memory, matching every other export's
"host writes memory, calls export, host reads memory" convention already
used throughout `rvf-wasm`. Signing never leaves the secret key resident
longer than one call's stack frame. No new dependency's version pin
changes an existing crate's default behavior (the feature is additive and
off by default).

## Governance

No wire-format, schema, or persisted-data change. No existing export's
signature changed. Standard PR review.

## Open Questions

1. Why is signing (8.78x) measurably worse than verification (3.60x)
   relative to native, when native signing itself is faster than native
   verification? Not root-caused here — see Next Research in the nightly
   report.
2. Would a smaller `no_std` Ed25519 implementation (e.g. a reduced-table
   or `ed25519-compact`-style backend) bring the size delta under budget
   without sacrificing the correctness this run verified?
3. Should a signing-capable Cognitum tile simply carry a larger, explicitly
   documented size budget instead of chasing ed25519-dalek's footprint
   down to the base tile's <8 KB target?

## Rejection Criteria

This ADR's production recommendation (ship `witness-signing` as an
optional Cognitum tile build, documented as size/latency-heavier than the
base tile) is withdrawn if a future run shows the measured size or latency
figures above do not reproduce, or if a smaller-footprint primitive
closes the size gap and should be used instead.
