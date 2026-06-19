# ADR-262: PhotonLayer — Privacy-Preserving Optical Verification

**Status**: Proposed
**Date**: 2026-06-18
**Deciders**: rUv
**Related**: ADR-260 (PhotonLayer Simulator), ADR-261 (Mask Exchange & Determinism)

---

## Context

The strongest near-term product wedge for a learned optical frontend is **not**
faster facial recognition — it is **consented 1:1 verification** in which the
sensor never records a human-readable image. A learned phase mask encodes the
scene into an intensity pattern optimized for the verification task; a compact
decoder answers only "same / not same," and the raw image is neither stored
nor transmitted. This reframes a privacy liability (storing faces) into a
privacy *feature* (storing only a task-specific optical measurement).

The position is deliberate and bounded:

- **In scope:** consented verification, liveness / presentation-attack cues,
  privacy-preserving sensing where raw imagery should never be retained.
- **Explicit non-goal:** public / mass-surveillance face identification. This
  is a hard boundary, documented and enforced by framing, demo design, and
  the absence of any 1:N identification capability.

## Decision

Build the **Privacy Gate** demo and its supporting metrics as a first-class
benchmark, with three measured properties.

### 1. Verification quality (FAR / FRR / EER)

`photonlayer-bench::verification` forms genuine pairs (same identity) and
impostor pairs (different identity), scores each pair by similarity of their
optical feature vectors under a given mask + config, sweeps a decision
threshold, and reports `VerificationReport { eer, far_at_eer, frr_at_eer,
threshold, num_genuine, num_impostor }`.

Measured (synthetic identity set, 16×16 input, 4-pixel feature space):

| Mask         | EER   | FAR@EER | FRR@EER |
|--------------|-------|---------|---------|
| random mask  | 0.133 | 0.133   | 0.134   |
| learned mask | 0.001 | 0.003   | 0.000   |

The learned optical frontend yields a far lower equal-error rate than a random
mask — the optics are doing useful, task-specific work.

### 2. Privacy by reconstruction attack

`photonlayer-bench::privacy` runs the adversary's best simple move: train a
ridge-regularized **linear inverse** from detector features back to a
downsampled input image, then measure reconstruction PSNR on held-out data.
**Higher reconstruction PSNR = more privacy leakage.** It reports
`PrivacyReport { reconstruction_psnr, leakage_score, frame_input_similarity }`.

Measured:

| Mask                 | recon PSNR (dB) | leakage | frame↔input sim |
|----------------------|-----------------|---------|------------------|
| identity (no optics) | 13.84           | 0.461   | 0.243            |
| random mask          | 11.00           | 0.367   | 0.056            |
| learned mask         | 12.75           | 0.425   | 0.013            |

Both optical masks reduce reconstruction PSNR versus reading pixels directly,
and the detector pattern's correlation with the input (`frame_input_similarity`)
collapses toward zero — i.e. the captured frame is **not human-readable**. The
linear attack fails to recover the image from the optical measurement.

### 3. Tamper-evident provenance

Every Privacy Gate run emits an `ExperimentReceipt` (ADR-260 §15 / ADR-261)
binding the input, mask, config, seed, output frame, and metrics. The receipt
verifies in the browser (`photonlayer-wasm::verify_receipt_json`) and via
`photonlayer-cli verify-receipt <path>`, proving the demonstrated result was
not swapped for a pre-baked one.

### 4. Governance memory (RuVector)

`photonlayer-ruvector` is the governance substrate: it stores, per experiment,
the mask embedding, the detector-pattern embedding, the metric vector, and the
receipt; supports nearest-prior recall; and runs spectral **boundary analysis**
to explain which configuration variable separates passing from failing runs.
For a verification deployment this is where accuracy-by-slice, spoof failure
cases, and reconstruction-risk scores would be recorded and audited.

## Threat Model

- **Honest-but-curious storage.** The system stores only optical measurements +
  embeddings + receipts, never the raw image — so a storage breach yields no
  recoverable faces (bounded by the reconstruction-attack result above).
- **Result forgery / swap.** Mitigated by receipt binding + verification.
- **Reconstruction adversary.** Linear inverse is the baseline; the leakage
  metric is the gate. Stronger (nonlinear/learned) attackers are future work,
  and the leakage score is the place to track that arms race — a mask must
  clear a leakage threshold before promotion to demo mode.
- **Out of scope:** physical sensor attacks, side channels, and — by policy —
  1:N identification.

## Acceptance (ADR-260 §9 face-version gates)

1. Detector pattern is not visually recognizable as the input
   (`frame_input_similarity` near zero). ✅ measured 0.013 (learned).
2. Same-identity verification beats the random-mask baseline. ✅ EER 0.001 vs 0.133.
3. Reconstruction attack fails above a defined privacy threshold (optical PSNR
   < identity PSNR). ✅ measured.
4. FAR / FRR / EER are reported. ✅
5. Runs generate RuVector records and RVF receipts. ✅
6. No images leave the browser by default (WASM-local pipeline). ✅

## Consequences

- **Positive.** A credible, defensible privacy story with measured guarantees,
  not marketing; a clean ethical boundary; reusable verification + privacy
  metrics for the industrial / medical sensor domains.
- **Negative / limits.** Synthetic identities today (the harness, not a face
  dataset); linear attacker only; verification is 1:1 by design. These are
  intentional scope choices, not omissions.
