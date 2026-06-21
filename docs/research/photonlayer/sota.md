# PhotonLayer — State of the Art & Research Basis

**Date**: 2026-06-18
**Scope**: Research grounding for ADR-260/261/262. Where citations could not be
independently fetched in this environment they are marked *(per project brief /
to be verified)*; the technical framing is stated so it can be checked against
the primary sources.

---

## 1. The three converging currents

PhotonLayer sits at the intersection of three lines of work that became
practical together.

### a. Diffractive deep neural networks (D2NN)
Passive diffractive surfaces can be *designed by deep learning* to perform
optical machine-learning functions (e.g. handwritten-digit classification,
lens-like imaging) using diffraction alone, demonstrated at terahertz
wavelengths. Establishes the core premise: **an optical layer can be trained.**
*(Lin et al., Science 2018 — per project brief / to be verified.)*

### b. Differentiable optics libraries
Joint optimization of optics + a digital model is now standard tooling:
- **TorchOptics** — PyTorch, GPU-accelerated, differentiable Fourier optics;
  supports joint optimization of optical systems with ML models.
- **waveprop** — scalar diffraction models and PyTorch training of apertures.
- **diffractsim** — diffraction visualization with a JAX differentiable backend.
These are PhotonLayer's *offline reference* path; the Rust runtime is validated
against them via replay hashes (ADR-261). *(Per project brief / to be verified.)*

### c. Hybrid optical–electronic + modern metasurfaces
- An optimized diffractive optical element placed *before* an electronic CNN
  improves classification while adding little electronic cost; one D2NN hybrid
  study reports input compression >7.8×, and some computational-meta-optics
  systems offload >90% of computation to the optical front end.
- An **edge-enhanced** diffractive network (June 2026) improved single-layer
  MNIST from **64.2% → 80.7%** by adding optical edge extraction.
- A **2026 full-Stokes metasurface** paper jointly optimizes a differentiable
  single-layer metasurface frontend with a **U-Net** backend to reconstruct
  RGB full-Stokes images from a *single monochrome* sensor measurement — the
  clearest modern blueprint for "optical frontend + small digital backend."
- Photonic-neuromorphic metasurface recurrent systems report brain-MRI
  classification and human-action-recognition results (early, but supportive).
- A 2026 computational-meta-optics review frames meta-optics + computational
  imaging as a path to small sensors + lightweight neural nets.
*(All per project brief / to be verified against the primary papers.)*

## 2. Credible vs. overclaimed

| Claim | Verdict |
|-------|---------|
| "Light can perform a trained transformation before digitization." | **Credible** — D2NN + differentiable-optics literature. |
| "A learned optical frontend can shrink sensor pixels / backend size at fixed task accuracy." | **Credible** — hybrid D2NN compression results; PhotonLayer measures it (64× fewer sensor pixels in our harness). |
| "A sensor can capture task-useful but non-human-readable measurements." | **Credible** — encoded/compressive imaging; we measure frame↔input similarity ≈ 0. |
| "Light is running a neural network / replacing the model." | **Overclaimed** — avoid. PhotonLayer's thesis is explicitly *first trained transformation + smaller digital backend*. |
| "Pure optical AI beats digital models." | **Overclaimed** — not the win condition. The honest claim is same accuracy with fewer sensor pixels / smaller decoder. |

## 3. Mapping references → PhotonLayer components

| Reference / idea | PhotonLayer component |
|------------------|------------------------|
| D2NN trainable diffractive layer | `photonlayer-core` phase mask + propagation; `photonlayer-bench` mask learner |
| TorchOptics / waveprop differentiable propagation | offline training reference; Rust replay validation (ADR-261) |
| Hybrid DOE + electronic CNN, >7.8× compression | compression benchmark (`run_compression`): 64× fewer sensor pixels |
| Edge-enhanced D2NN | `photonlayer-cli edge` demo |
| Full-Stokes metasurface + U-Net reconstruction | reconstruction-attack / privacy module direction; future Stokes demo |
| Computational meta-optics for small sensors | the "learned measurement device, not a camera" framing (ADR-260 §22) |

## 4. Application feasibility (project-lead framing)

Optical computing is a **front end** — lower latency, less sensor bandwidth,
lower power, compressed measurements, task-specific sensing — not a replacement
for the perception stack. Feasibility ranking (highest first):

1. **Industrial & scientific sensors** — task-specific learned measurement
   devices (crack/defect/material/polarization detection). Highest feasibility,
   low privacy risk, strong value.
2. **Drone / AV perception preprocessing** — obstacle/wire/landing-zone cues,
   optical flow, glare robustness; grams, watts, bandwidth, and latency
   dominate. High feasibility in simulation, hardware later.
3. **Medical imaging research simulator** — microscopy compression, snapshot
   polarization, reconstruction from fewer measurements. Research tooling, **not
   diagnosis**; careful claims + public datasets.
4. **Consented face verification + liveness** — feasible, high governance
   burden (see ADR-262).
5. **Public / mass-surveillance facial recognition** — **non-goal.** High
   legal/ethical/brand risk; deliberately not built.

## 5. The win condition

Not "beats all neural nets." The defensible, measurable claim is: **a learned
optical frontend preserves task-useful information while reducing sensor pixels,
decoder size, or privacy exposure compared with a direct pixel pipeline** — with
every result reproducible (ADR-261) and auditable (ADR-262).
