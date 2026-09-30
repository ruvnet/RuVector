# What does it cost to sign a witness record in WebAssembly?

## Problem

RuVector's agent-memory subsystem can sign witness chain entries with
Ed25519 natively (Rust, `x86_64`). Three separate nightly research runs
in this project asked, and then deferred without answering, the same
follow-up question: what does that cost in WebAssembly, and can it fit
inside the project's own sub-8-kilobyte edge microkernel? This is a
writeup of the run that finally built it and measured it.

## Hypothesis

Two, fixed before any measurement:

1. **Narrow:** Ed25519 signing fits inside the existing edge tile's own
   documented budget (< 8 KB after `wasm-opt`).
2. **Broad:** Ed25519 signing is practical as a small, separate WASM
   module (< 150 KB raw, < 1 ms per operation in a browser).

## Technical Design

Two implementations, plus the existing unsigned baseline, compiled to
`wasm32-unknown-unknown`:

- **Baseline** — the existing edge tile, no signing capability at all.
- **Candidate A** — the same edge tile, with an opt-in feature that adds
  the project's real segment-level signing/verification code path.
- **Candidate B** — a new, from-scratch minimal module wrapping only the
  raw generic sign/verify primitive, no extra machinery.

All three share `#![no_std]` + a small global allocator, static
scratch buffers, and a plain `extern "C"` FFI boundary — no JS
framework glue, just raw pointers into WASM linear memory, matching the
project's existing edge-module convention.

## Actual Implementation

- `rvf-witness-wasm` (new crate): ~95 lines, two exported functions
  (`sign`, `verify`), no dynamic allocation exposed to callers.
- `rvf-wasm`'s new `witness-sign` feature: ~110 lines, off by default,
  zero effect on the existing shipped artifact.
- A permanent regression test: a fixed RFC 8032 test vector, pinned once
  and printed via `cargo test -- --nocapture`, then hand-carried into a
  browser test harness to prove the WASM build and the native build
  produce **bit-identical signatures** for the same input.

While building this, two unrelated but real bugs surfaced and got fixed:

1. All three WASM edge crates had a `[profile.release]` block in their
   own `Cargo.toml` that Cargo was silently ignoring (a known Cargo
   behavior for non-root workspace members). They'd been shipping with
   the *default* release profile instead of the size-optimized one they
   claimed — about 30% larger than intended. Fixed via a workspace-root
   override scoped to exactly those three crates.
2. The signing crate's Ed25519 dependency wasn't actually safe to use
   from `no_std` code — a missing `default-features = false` let Rust's
   standard library leak in through dependency-feature unification,
   which is what made the original narrow-hypothesis experiment
   impossible to even compile before this run.

## Actual Benchmark Evidence

Binary size, `wasm32-unknown-unknown`, release, before any post-link
`wasm-opt` pass (unavailable in this environment):

| Variant | Size |
|---|---:|
| Baseline (unsigned) | 35,611 bytes |
| Candidate A (in-tile signing) | 108,449 bytes |
| Candidate B (standalone) | 73,031 bytes |

Latency, 128-byte payload:

| Operation | Native (Criterion) | WASM (headless Chromium, batched, 3-run average) |
|---|---:|---:|
| sign | 46.6 µs | ~291 µs |
| verify | 52.8 µs | ~155 µs |

Correctness: the WASM build's signature output is bit-identical to the
native build's for the same key and message; it correctly verifies both
its own and natively-produced signatures; it correctly rejects a
tampered signature. All four checks, three independent runs, zero
failures.

## Result

- **Narrow hypothesis: rejected.** In-tile signing measures 13.5x over
  the tile's own stated budget, even before adding signing the
  unsigned baseline no longer fit that budget either (see below) — no
  realistic amount of post-link optimization closes a gap that size.
- **Broad hypothesis: accepted.** A standalone signing module comes in
  at under half the size threshold and under a third of the latency
  threshold this run set in advance.

An unresolved oddity: natively, signing is faster than verifying; in the
browser, verifying is consistently faster than signing. Reported, not
explained.

## Limitations

- No `wasm-opt` pass was available in this environment; all sizes are
  pre-optimization. The rejection of the narrow hypothesis is robust to
  this (the gap is too large to close), but the exact achievable size
  of the standalone module is likely somewhat smaller than reported
  here.
- `lto = true` could not be scoped to only the affected crates — a
  Cargo limitation, not a choice — so these numbers reflect size
  optimization without full link-time optimization.
- Measured only in headless desktop Chromium, on one machine. No mobile
  browser, no real embedded/edge hardware, no alternative WASM runtime.
- The in-tile signing path uses a synthetic benchmark header, not a
  wired real write path.
- The raw private key crosses the WASM/JS boundary via linear memory —
  fine for a trusted sandbox, not yet a resolved design for exposing
  signing directly to untrusted browser-page code.

## Production Relevance

The standalone module (Candidate B) is small and fast enough to be a
practical building block for edge- or browser-resident agents that need
to sign their own memory locally instead of relaying every write to a
trusted server first — directly extending work already shipped
natively in this project's agent-memory subsystem. It ships as an
experimental crate; production use still needs a real `wasm-opt` pass in
the build pipeline and a resolved key-custody design.

## RuVector Ecosystem Implications

This closes a specific, three-times-deferred gap between RuVector's
native signed-witness-chain work and its WASM edge-deployment story,
and does so by extending, not duplicating, the exact primitive an
earlier nightly run already shipped for native use — one Ed25519
implementation now proven to compile, unmodified, to both targets with
identical output.

## Future Direction

Measure with a real `wasm-opt` pass; give the WASM edge crates their own
isolated one-crate Cargo workspaces so full LTO becomes possible; wire
in-tile signing into an actual witness-chain write path if a concrete
use case wants it; measure on real edge hardware rather than a desktop
browser; and repeat this whole comparison once a post-quantum signing
scheme exists natively, since Ed25519-specific numbers won't transfer.

## References

- Full research document: `docs/research/nightly/2026-09-30-edge-witness-signing/README.md`
- `docs/adr/ADR-352-ed25519-edge-witness-signing-wasm.md`
- The run this one continues: `docs/research/nightly/2026-09-16-witness-signer-agent-memory/README.md`
