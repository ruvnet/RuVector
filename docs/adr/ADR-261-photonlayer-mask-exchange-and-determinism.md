# ADR-261: PhotonLayer — Mask Exchange Format & Determinism Invariant

**Status**: Proposed
**Date**: 2026-06-18
**Deciders**: rUv
**Related**: ADR-260 (PhotonLayer Optical Computing Simulator), ADR-029 (RVF Canonical Format)

---

## Context

PhotonLayer (ADR-260) trains optical phase masks and replays them in three
runtimes: native Rust (`photonlayer-core`/`-bench`/`-cli`), the browser
(`photonlayer-wasm`), and — in the offline reference path — differentiable
Fourier-optics libraries (TorchOptics / waveprop). A trained mask is only
useful if it produces the *same* optical measurement everywhere it runs.
Two risks follow:

1. **Cross-runtime drift.** Floating-point FFT libraries reorder reductions
   under SIMD/threading, so the "same" mask can yield different sensor frames
   on different machines. A demo that cannot be reproduced is not credible.
2. **Untrusted imports.** A mask trained by an external (Python) pipeline must
   be verified against the Rust runtime before it is promoted to a public
   demo, or a swapped/forged mask could fake a result.

This ADR fixes the on-disk mask format and the determinism guarantees that
make imported masks trustworthy.

## Decision

### 1. Mask exchange format

A phase mask is serialized as canonical JSON of the `PhaseMask` type:

```json
{
  "width":  16,
  "height": 16,
  "phase_radians": [ /* width*height f32, row-major, each in [0, 2π) */ ],
  "mask_id": "learned:0xa11ce"
}
```

Rules:

- `phase_radians.len() == width * height`; values are wrapped into `[0, 2π)`.
- Row-major ordering (x fastest), matching `OpticalField` and the detector.
- `mask_id` is advisory provenance, **not** part of any optical computation,
  and therefore **not** hashed into the mask digest (so a re-label does not
  invalidate a replay).
- Floats are encoded/decoded losslessly (`f32`); the binding hash is taken
  over the raw little-endian bytes, not the JSON text.

### 2. Determinism invariant

For any experiment:

```
same input + same mask + same OpticalConfig + same seed  ⇒  same output hash
```

This is enforced structurally, not by convention:

- **In-house FFT.** `photonlayer-core::fft` is an iterative radix-2
  Cooley–Tukey transform restricted to power-of-two sizes. It performs no
  threading, no SIMD reordering, and no library dispatch, so the butterfly
  reduction order is fixed across platforms (including `wasm32`).
- **Seeded noise.** All sensor noise (shot/read) comes from a SplitMix64
  `DeterministicRng` seeded from `OpticalConfig.seed`; no `rand`, no OS entropy.
- **Bit-exact hashing.** `frame_hash`, `mask_hash`, `input_hash`,
  `config_hash` are BLAKE3 over a canonical little-endian byte encoding with a
  domain tag and the dimensions (`hash::hash_f32`). Any change to a value, a
  dimension, or ordering changes the digest.

Tests `receipt::replay_is_deterministic`, `detector::noise_is_deterministic`,
and `simulator::simulation_is_deterministic` assert the invariant directly.

### 3. Import verification

An imported mask is accepted only after a **replay check**:

1. Load the mask JSON; recompute `mask_hash`.
2. Run the canonical Rust pipeline on a fixed probe input + config + seed.
3. Compare the resulting `frame_hash` (and the full `ExperimentReceipt`
   `rvf_receipt_hash`) against the value the exporter recorded.

A mismatch rejects the mask. This closes the "training dependency mismatch"
failure mode (ADR-260 §20.5): the Python trainer and the Rust runtime must
agree on the optical model, or the mask never reaches a demo.

### 4. Receipt binding fields

`ExperimentReceipt` binds the full determinant set (ADR-260 §15):
`experiment_id`, `input_hash`, `mask_hash`, `config_hash`, `output_hash`
(= `frame_hash`), `metrics_hash`, `git_commit`, `rustc_version`,
`feature_flags`, `seed`, and the combined `rvf_receipt_hash`. `verify_receipt`
recomputes the combined digest and rejects any tampered field.

## Consequences

- **Positive.** Masks are portable and auditable; a public demo output can be
  proven to come from a specific mask + config + seed; cross-platform replay
  is bit-identical.
- **Negative / limits.** Power-of-two grids only (non-pow2 must be padded);
  `f32` precision is fixed (a future `f64` mode would be a new format version,
  e.g. `photonlayer.frame.v2`). Differentiable-optics references (TorchOptics)
  use their own FFT and so are validated *against* Rust replay hashes rather
  than assumed bit-identical.

## Acceptance

- Round-trip: serialize → deserialize → re-hash yields the same `mask_hash`.
- Replay: the same mask/input/config/seed reproduces `output_hash` on native
  and `wasm32`.
- Tamper: mutating any bound field fails `verify_receipt`.
