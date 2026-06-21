---
adr: 263
title: "PhotonLayer FiberGate — transmission-matrix optical compression for drift-bound privacy verification"
status: proposed
date: 2026-06-18
authors: [ruvnet, claude-flow]
related: [ADR-260, ADR-261, ADR-262]
tags: [photonlayer, fiber-optics, multimode-fiber, transmission-matrix, mmf, privacy, receipts, drift, ruvector, wasm]
---

# ADR-263 — PhotonLayer FiberGate

> **Decision in one line.** Add a **multimode-fiber (MMF) propagation backend** to
> `photonlayer-core` based on a **calibrated complex transmission matrix (T)**, a
> deterministic intensity-sensor projection, **drift-aware calibration receipts**,
> and ruVector-backed calibration memory — moving PhotonLayer from free-space
> diffraction to guided-wave optics.

## Context

PhotonLayer today models **free-space** scalar diffraction (Fresnel / Fraunhofer /
angular-spectrum; ADR-260). The cutting edge of optical computing is moving to
**guided-wave** substrates — e.g. the Fiber-based Diffractive Deep Neural Network
(Fiber-D2NN, *Opt. Lett.* 50(17):5254) shows high ML accuracy from linear optical
relations **inside** fibers.

The right framing (and the bounded claim we will defend):

> **PhotonLayer Fiber treats the multimode fiber as a calibrated, drifting,
> complex *linear* optical operator. The learned mask shapes the input field so
> that the drifting fiber + sensor produce task-useful compressed measurements.**

This is stronger than "the fiber is just another propagation model." Below
nonlinear power thresholds, the input→output **field** relationship of an MMF is
well modeled by a transmission matrix **T**; the observed **sensor** image is an
intensity speckle pattern. Both deep-learning and intensity-matrix methods are
established for MMF image recovery/classification (arXiv:1805.05134). Bending and
temperature **drift** of T is the central real-world challenge and requires
recalibration (Nat. Commun. s42005-023-01410-x).

### Claim hygiene (no slop)

- **This is NOT zero-knowledge.** The design proves a private input was transformed
  under a specific mask + calibrated fiber state into a specific low-dimensional
  measurement. That is a **cryptographic receipt**, not a ZK proof. We use:
  **receipt-verified privacy gate** / **non-reconstructive verification** /
  **optical biometric commitment** / **fiber-bound biometric proof**. Safe claim:
  *"the server verifies a receipt bound to the current fiber calibration and
  receives only the compressed optical measurement, not the source image."*
- **Consented verification only** — no gallery identification.

## Decision

Add a fiber backend and the supporting machinery, in this **build order** (the
moat is the physics + receipts + memory, not the browser):

1. `photonlayer-core` fiber backend (T-matrix propagation + deterministic sensor).
2. Fiber **drift** benchmark in `photonlayer-bench`.
3. ruVector **calibration memory** in `photonlayer-ruvector`.
4. `photonlayer-wasm` client bindings.

### Mathematical model

**Level 1 — practical T-matrix simulator (implementation target):**

```
E_in' = E_in ⊙ e^{iΦ}            (apply learned phase mask)
E_out = T · E_in'                 (linear mode mixing; T may be non-square)
I_sensor = B · |E_out|² + ε       (intensity-only, binned, deterministic seeded noise ε)
```

**Level 2 — mode-basis simulator (later, for physical interpretability):**

```
E(x,y,z) = Σ_m a_m(z) ψ_m(x,y) e^{iβ_m z}
a_m(0)   = ∬ E_in(x,y) e^{iΦ(x,y)} ψ_m*(x,y) dx dy
a(L)     = C(L, θ, T_env) a(0)     (separates ideal propagation from drift)
```

### Rust design (determinism-first)

Three correctness rules the implementation MUST honor:

1. **Layout contract** — `nalgebra::DMatrix` is column-major; define and document a
   fixed (row-major image) ⇄ (column-vector) mapping so flatten/reshape never
   silently reorders the spatial layout.
2. **Non-square T** — input modes, output field samples, and sensor bins have
   different dimensions (e.g. 256 → 64 → 4). `T` is `output_len × input_len`.
3. **Determinism is not free from `nalgebra`** — pin operation order, no
   uncontrolled parallel reductions, finite checks on every value, stable
   serialization. Bit-identical output across Linux/macOS/WASM is an invariant
   (ADR-261).

Core type (shape per the corrected design):

```rust
pub struct FiberTransmissionMatrix {
    pub t: DMatrix<Complex64>,   // output_len × input_len (non-square allowed)
    pub input_len: usize,
    pub output_len: usize,
    pub version: u64,
    pub calibration_hash: [u8; 32],
}
// propagate(input_field, mask) -> FiberOutput { field, intensity },
//   validating input/mask length + matrix shape + finiteness (FiberError otherwise).
```

New `photonlayer-core` modules: `fiber.rs`, `fiber_matrix.rs`,
`fiber_calibration.rs`, `fiber_sensor.rs`, `fiber_receipt.rs`. Core structs:
`FiberTransmissionMatrix`, `FiberCalibrationState`, `FiberPilotPattern`,
`FiberDriftModel`, `FiberSensorProjector`, `FiberReceipt`.

### Drift-aware training (turns drift from liability into a training distribution)

```
min_{Φ,θ}  E_{T ~ D_fiber} [ L(decoder(B|T(E ⊙ e^{iΦ})|²), y) ]  + λ R(Φ) + γ L_privacy
```

Train the mask against a **family** of likely T states, not one matrix. `R(Φ)` =
smoothness/manufacturability; `L_privacy` = reconstruction/leakage penalty.

Drift metric and accuracy-vs-drift tracking:

```
Δ_T = ‖T_t − T_{t−1}‖_F / ‖T_{t−1}‖_F        A = f(Δ_T, SNR, bins, mask)
```

### ruVector calibration memory (`photonlayer-ruvector`)

Store each calibration as an experiment object (fiber_id, t_version,
calibration_hash, timestamp, temperature, bend_state, snr_db, drift_norm,
mask_id, task, eer, reconstruction_score) and use ruVector for: nearest-calibration
lookup, drift-regime clustering, recalibration prediction, finding masks robust
across multiple T states, and spectral failure explanation.

### Receipt schema + commitment

```rust
pub struct FiberReceipt {
    pub photonlayer_version: String,
    pub fiber_id_hash: [u8; 32],
    pub t_version: u64,
    pub t_hash: [u8; 32],
    pub pilot_hash: [u8; 32],
    pub phase_mask_id: [u8; 32],
    pub input_commitment: [u8; 32],   // salted: C = H(input_hash || nonce || purpose || session)
    pub output_hash: [u8; 32],
    pub decoder_id: [u8; 32],
    pub nonce: [u8; 32],
    pub timestamp_ms: u64,
}
```

Key design choice: **never store the raw biometric hash in a reusable form** — use
the salted commitment `C`.

### Threat model

| Threat | Risk | Mitigation |
|---|---|---|
| Replay old N-pixel output | Medium | bind receipt to T version + nonce + timestamp |
| T-matrix theft | Medium | rotate calibration, encrypt at rest, bind to device |
| Reconstruction attack | High | publish attack suite (ADR-262), train privacy penalty |
| Server spoofing T state | Medium | signed calibration state |
| Browser config DoS | Medium | existing input validation |
| Biometric misuse | High | consented verification only, no gallery |
| Model inversion | High | leakage tests vs attributes + identity |

## Consequences

### Positive
- Unlocks the flagship **medical-endoscope** (optical compression inside a needle-
  thin fiber bundle — no bulky tip sensor) and **drone/edge** sensing paths.
- Drift becomes an **anti-replay signal**, not only a liability.
- The defensible moat: **drift-aware, receipt-verified optical compression with
  experiment memory** — not the browser layer.

### Negative / risks
- Real fiber validation needs hardware + calibration; until then this is a
  **simulator + receipt schema**, claimed as such.
- T-matrix calibration is a new operational burden (pilot wavefronts, drift
  thresholds, recalibration triggers).
- Determinism across native + WASM with `nalgebra` complex linear algebra requires
  care (see rule 3).

### Neutral
- Free-space backend (ADR-260) stays; fiber is an additional `PropagationKind`.

## Acceptance tests

1. Same input + mask + T + seed → identical output hash across Linux, macOS, WASM.
2. Learned mask beats random by ≥ **20 pp** on compressed fiber classification.
3. EER stays below target across ≥ **5 drift states**.
4. Reconstruction-attack similarity stays below the documented threshold.
5. Receipt verification **fails** if any of {T version, phase mask, decoder, nonce,
   output, sensor config} changes.
6. Recalibration trigger fires when `Δ_T` crosses the configured threshold.
7. **Non-square** T passes end-to-end: 256 inputs → 64 output samples → 4 sensor bins.

## Links
- ADR-260 (free-space simulator), ADR-261 (mask exchange & determinism),
  ADR-262 (privacy-preserving optical verification).
- Fiber-D2NN — *Opt. Lett.* 50(17):5254. MMF + deep learning — arXiv:1805.05134.
  Single-ended T recovery / drift — *Nat. Commun.* s42005-023-01410-x.
- Product: **PhotonLayer FiberGate** — calibrated fiber as a physically-bound,
  receipt-verified transformation layer for non-reconstructive verification.
