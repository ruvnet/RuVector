# ADR-352: Ed25519 Edge Witness Signing in WASM

## Status

Accepted (broad hypothesis / `rvf-witness-wasm` as an experimental
crate). Rejected (narrow hypothesis / in-place Cognitum tile signing as
a default capability — retained as an opt-in feature, not promoted).

## Context

`ruvector-agent-memory`'s `SignedWitnessSink` (added in the 2026-09-16
nightly run) and the retrieval-receipt work in ADR-340/343 both sign
witness/receipt records natively using `rvf_types::ed25519` /
`rvf_crypto::sign`. All three of these prior efforts explicitly deferred
the same question without answering it: what does this signing
primitive cost in WASM, and can the existing `rvf-wasm` Cognitum edge
tile microkernel — which documents a "< 8 KB after wasm-opt" size
budget — afford it? This ADR records the run that finally built the
WASM integration and measured it, rather than deferring a fourth time.
Full methodology, raw numbers, and reproduction steps are in
`docs/research/nightly/2026-09-30-edge-witness-signing/README.md`; this
ADR records the decision.

## Hypothesis

1. **H1 (narrow):** Ed25519 signing can be added to the existing
   `rvf-wasm` Cognitum tile without exceeding its documented < 8 KB
   (post-`wasm-opt`) budget.
2. **H2 (broad):** Ed25519 signing is deployable as a standalone WASM
   edge module under 150 KB (raw) and under 1 ms per operation
   in-browser.

Both thresholds were fixed before any WASM artifact was built (H1 from
`rvf-wasm`'s own pre-existing module doc; H2 as an engineering judgment
call for "practical edge/browser deployment").

## Decision

- Add a new, minimal, `#![no_std]` crate `crates/rvf/rvf-witness-wasm`
  wrapping only `rvf_types::ed25519::{ed25519_sign, ed25519_verify}`
  (no `SegmentHeader` canonicalization), following the existing
  `rvf-wasm`/`rvf-solver-wasm` static-scratch-buffer + `dlmalloc`
  convention. **Promoted** as an experimental crate.
- Add an **off-by-default** `witness-sign` Cargo feature to the existing
  `rvf-wasm` crate, exercising the real
  `rvf_crypto::sign_segment`/`verify_segment` path. **Not** wired into
  the default build — H1 is rejected (measured at 108,449 bytes,
  ~13.5x over the 8 KB budget, with no plausible `wasm-opt` pass closing
  that gap).
- Fix `rvf-crypto`'s `ed25519` feature to actually be `no_std`-safe (its
  `ed25519-dalek` dependency was missing `default-features = false`,
  silently pulling in `std` via Cargo feature unification and making
  this integration impossible to even attempt before this fix).
- Fix the release profile for `rvf-wasm`, `rvf-solver-wasm`, and the new
  `rvf-witness-wasm` via `[profile.release.package.*]` overrides at the
  `crates/rvf` workspace root — their own per-crate `[profile.release]`
  blocks were silently ignored by Cargo (non-root workspace member
  profiles are ignored), so these crates had been shipping ~30-45%
  larger than their own manifests claimed. Measured impact on
  `rvf-wasm`: 51,877 → 35,611 bytes (-31.4%).
- Add a permanent, pinned RFC 8032 test vector
  (`rvf-types::ed25519::wasm_interop_tests`) proving the WASM build
  produces bit-identical signatures to the native build for the same
  input, as a regression guard against a future `ed25519-dalek` upgrade
  silently changing output.

## Evidence

See the linked nightly research document for full methodology and raw
numbers. Summary:

| Variant | WASM size (pre-`wasm-opt`) |
|---|---:|
| `rvf-wasm` baseline (no signing) | 35,611 bytes |
| `rvf-wasm` + `witness-sign` (Candidate A) | 108,449 bytes |
| `rvf-witness-wasm` standalone (Candidate B) | 73,031 bytes |

| Operation | Native (Criterion, 128B payload) | WASM (headless Chromium, batched, 3-run avg) |
|---|---:|---:|
| sign | 46.6 µs | ~291.3 µs (~6.2x) |
| verify | 52.8 µs | ~155.2 µs (~2.9x) |

Correctness: WASM-produced signatures are bit-identical to native for
the same key/message; WASM correctly verifies both its own and
natively-produced signatures; WASM correctly rejects a tampered
signature. All four checks passed across 3 independent runs.

## Consequences

- `ruvector-agent-memory`-style witness signing is now proven feasible
  in WASM as a small, separate module, unblocking edge/browser-based
  agent memory signing as a future integration (not yet wired into
  `ruvector-agent-memory` itself — that remains future work).
- The existing `rvf-wasm` Cognitum tile's own size budget is confirmed
  incompatible with in-place Ed25519 signing; a deployment that needs
  both the tile's vector-search capabilities *and* on-tile signing must
  either accept a ~3x larger artifact (the `witness-sign` feature, now
  available but opt-in) or split signing into a separate capsule (an
  RVM coherence-domain boundary, per the research doc's RVM
  Implications section).
- Every consumer of `rvf-wasm` and `rvf-solver-wasm` gets a ~30%+
  smaller default artifact from the profile fix alone, independent of
  anything else in this ADR.
- `rvf-crypto`'s `ed25519` feature is now genuinely usable from any
  `no_std` consumer, not just this one.
- The raw secret key is passed across the WASM/JS FFI boundary via
  linear memory in both new export surfaces (`rvf_ws_sign` and
  `rvf_witness_sign_segment`). This is acceptable for a trusted,
  isolated WASM sandbox but is explicitly **not** a resolved story for
  exposing signing to untrusted browser-page JavaScript — see "Open
  Questions."

## Alternatives Considered

- **Do nothing / defer a fourth time.** Rejected: this is the exact
  failure mode this ADR exists to stop; three prior runs already
  deferred this question, each explicitly citing the ones before it.
- **Only measure size, skip the correctness/latency harness.** Rejected:
  a size number without a working, verified signing path would not
  actually answer "is this feasible," only "is this theoretically
  small."
- **Fix the WASM-incompatibility in `rvf-crypto` but skip building any
  new crate.** Rejected: the `no_std` fix alone doesn't answer the
  headline question; a measurable artifact is required either way, and
  the fix is a prerequisite discovered *while* building that artifact,
  not a substitute for it.
- **Force `lto = true` workspace-wide to fully optimize the three edge
  crates.** Rejected: `lto` cannot be scoped per-package in Cargo, and
  workspace-wide LTO would meaningfully slow down every native crate's
  release build in `crates/rvf` (including production services like
  `rvf-server`) for a benefit that only applies to three WASM-only
  crates. Named as future work (give them standalone one-crate
  workspaces instead, matching existing precedent elsewhere in this
  repository).

## Implementation Plan

Already implemented this run:

1. `rvf-crypto/Cargo.toml`: `ed25519-dalek` dependency now
   `default-features = false, features = ["alloc", "rand_core"]`.
2. `crates/rvf/Cargo.toml`: `[profile.release.package.rvf-wasm]`,
   `[profile.release.package.rvf-solver-wasm]`,
   `[profile.release.package.rvf-witness-wasm]` added; `rvf-witness-wasm`
   added to `members`.
3. `rvf-wasm/Cargo.toml`: new optional `ed25519-dalek` dependency and
   `witness-sign` feature (off by default); dead `[profile.release]`
   block replaced with a pointer comment.
4. `rvf-wasm/src/witness_sign.rs` (new): `witness-sign`-gated exports.
5. `rvf-wasm/src/lib.rs`: registers the new module behind the feature.
6. `rvf-solver-wasm/Cargo.toml`: dead `[profile.release]` block replaced
   with a pointer comment (no functional change).
7. `crates/rvf/rvf-witness-wasm/` (new crate): `Cargo.toml` + `src/lib.rs`.
8. `rvf-types/src/ed25519.rs`: new pinned interop test.
9. `benches/Cargo.toml` + `benches/benches/rvf_benchmarks.rs`: new
   `ed25519_sign_raw_128b`/`ed25519_verify_raw_128b` Criterion
   benchmarks, matching the WASM harness's payload size for a fair
   comparison.

## API Shape

New public WASM exports (all `#[no_mangle] extern "C"`, matching the
existing `rvf-wasm` FFI convention of `i32` pointers into linear
memory):

- `rvf-witness-wasm`: `rvf_ws_message_ptr() -> i32`,
  `rvf_ws_message_capacity() -> i32`,
  `rvf_ws_sign(secret_ptr, msg_len, sig_out_ptr) -> i32`,
  `rvf_ws_verify(pub_ptr, msg_len, sig_ptr) -> i32`.
- `rvf-wasm` (`witness-sign` feature only):
  `rvf_witness_sign_payload_ptr() -> i32`,
  `rvf_witness_sign_payload_capacity() -> i32`,
  `rvf_witness_sign_segment(segment_id, payload_len, secret_ptr, sig_out_ptr) -> i32`,
  `rvf_witness_verify_segment(segment_id, payload_len, pub_ptr, sig_ptr) -> i32`.

## Feature Flags

- `rvf-wasm/witness-sign` — off by default (H1 rejected; this ADR does
  not recommend enabling it in a default/production Cognitum tile
  build).
- `rvf-crypto/ed25519` — unchanged default-on status, now actually
  `no_std`-safe.

## Benchmark Evidence

See "Evidence" above and the full nightly research document for raw
Criterion output, all three headless-Chromium run outputs, and exact
reproduction commands.

## Security

See the nightly research document's "Failure Modes" and "Security
Review" sections. Headline: the signing FFI surface trusts the caller
with raw key material in linear memory; this is appropriate for a
trusted WASM sandbox, not for direct untrusted-browser-JS exposure
without further isolation. No MCP tool is recommended for the signing
path in this ADR; a verification-only MCP tool would be lower-risk
future work.

## Governance

Neither new export surface is wired into any existing production write
path (`ruvector-agent-memory`'s `SignedWitnessSink` is untouched by this
ADR). Promotion of `rvf-witness-wasm` here means "available as an
experimental building block with measured characteristics," not "wired
into a production signing flow" — that remains a distinct, future
decision.

## Failure Modes

See the nightly research document's "Failure Modes Considered" section.

## Migration

None required — both new export surfaces are additive; the existing
default `rvf-wasm` build is unchanged (byte-identical size, confirmed by
rebuild-and-compare).

## Rollback

Revert the commits associated with this ADR; `rvf-witness-wasm` can be
removed from the `crates/rvf` workspace `members` list independent of
the `rvf-crypto`/profile fixes, which are recommended to keep regardless
of any decision on the crate itself.

## Rejection Criteria

This ADR's H2 acceptance would be revisited if: `wasm-opt` measurement
(Next Research item 1) reveals the raw pre-opt numbers were
substantially misleading; the native-vs-WASM latency ratio measured here
fails to hold on real edge hardware (Next Research item 6); or the
key-custody question (Failure Modes) proves unresolvable for the
intended deployment target, making the module unusable regardless of its
favorable size/latency.

## Open Questions

1. Why does the native-vs-WASM sign/verify latency ordering reverse
   (native: sign faster than verify; WASM: verify faster than sign)?
   Not investigated this run.
2. What is `rvf-witness-wasm`'s actual size after a real `wasm-opt -Oz`
   pass? Not measurable in this environment.
3. What is the right key-custody design for exposing edge signing to
   less-trusted callers (a dedicated Worker? non-extractable keys? a
   hardware secure element)? Named, not resolved.

## References

- `docs/research/nightly/2026-09-30-edge-witness-signing/README.md` —
  full methodology, raw numbers, and reproduction steps.
- `docs/research/nightly/2026-09-16-witness-signer-agent-memory/README.md`
  — the run whose deferred Next Research item 4 this ADR closes.
- `docs/adr/ADR-134-witness-schema-log-format.md`
- `docs/adr/ADR-340-signed-retrieval-receipt-anchoring.md`
- `docs/adr/ADR-343-signed-receipt-batch-fill-latency-simulation.md`
- `crates/rvf/rvf-types/src/ed25519.rs`
- `crates/rvf/rvf-crypto/src/sign.rs`
- `crates/rvf/rvf-wasm/src/lib.rs`, `src/witness_sign.rs`
- `crates/rvf/rvf-witness-wasm/`
