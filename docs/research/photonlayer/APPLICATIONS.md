# PhotonLayer — Applications & Strategy

> **The category:** not cameras, not neural nets, not "optical computing" — **task-trained sensors**.
> Strategic thesis: *AI created an infinite appetite for visual data; PhotonLayer goes the other way —
> **capture less, decide faster, leak less, and prove what happened.***

Companion to [ASSESSMENT.md](ASSESSMENT.md), ADR-260/261/262/263. Bounded-claim discipline applies:
no diagnosis claims, consented verification only, "receipt-verified" (not zero-knowledge).

## The core shift — image recognition without images

```
Traditional:  scene → camera image → neural net → decision
PhotonLayer:  scene → trained optical transform → tiny sensor measurement → small decoder → decision
```

Recognize / verify / classify / reject from a **compressed optical signature** instead of a stored
image. Aligned with learned optical encoders, lensless privacy cameras, meta-optics, and hybrid
optical-electronic neural nets (arXiv:2406.04129; ACS Photonics 5c02358; PMC12011376).

> **The system sees enough to decide, but not enough to reconstruct.**

## Application areas (positioned by market readiness)

| Area | PhotonLayer role | First market? |
|---|---|---|
| **Industrial inspection** | defect yes/no, barcode/label verify, tamper, scratch, sorting | **Yes — best first market** (low regulatory burden, clear ROI, controlled lighting) |
| Privacy-first machine vision | face *verification* without storage, occupancy without identity, PPE/posture, child-safe presence | High viral / commercial distinctiveness |
| Ultra-low-bandwidth sensors | tiny sensors, always-on vision, few-bin event verification, battery/edge | Reduces the expensive part of edge AI (pixel movement) |
| Drone / robotics pre-perception | landing-pad, horizon, obstacle, marker, motion-cue, terrain class | Medium (safer than full autonomy) |
| Medical imaging **research** | microscopy morphology, lesion compression, endoscopy, cell pre-sort, pathology triage | High risk — **research only, no diagnosis** |
| Fiber-bound security | tamper-evident links, device-bound auth, anti-replay, drift-liveness (ADR-263) | New product class |
| Scientific instruments | design instruments around the *question*, not the image | Biggest long-term idea |

## A new benchmark lane — "accuracy per captured photon / pixel / sensor bin"

Metrics: accuracy/sensor-pixel · accuracy/digital-MAC · EER under reconstruction constraint · drift
robustness · receipt reproducibility · calibration half-life · leakage score · failure boundary. The
receipt system is what makes these **reproducible optical experiments**, not just claims.

## Viral demos (for the Pages UI)

- **A — "The camera that cannot see you"**: face → 4-px measurement → same/different verdict. *It verified the person without storing the face.*
- **B — "The microscope learned what not to measure"**: full image → optical compression → morphology class → failed reconstruction. *The useful signal survived. The image did not.*
- **C — "Drone vision in 4 pixels"**: landing marker detected from a few optical bins. *The drone needs a decision, not a frame.*
- **D — "The fiber is the lock"** (ADR-263): verification fails when T drifts or the receipt is replayed. *The cable became part of the cryptographic boundary.*

## Product path (build order)

1. **PhotonLayer Studio** — browser optical-mask simulator with receipts.
2. **PhotonLayer Bench** — public optical-compression benchmark suite.
3. **PhotonLayer PrivacyGate** — consented verification + reconstruction attacks.
4. **PhotonLayer Industrial** — defect/barcode inspection SDK.
5. **PhotonLayer FiberGate** — drift-aware MMF simulator + lab bridge (ADR-263).
6. **PhotonLayer BioResearch** — microscopy/dermatology research simulator (no diagnostic claims).

## Sharp acceptance test (platform-grade)

> On **3 public datasets**: learned optical mask ≥ full-image baseline − 2%; sensor pixels reduced
> ≥ **16×**; digital MACs reduced ≥ **10×**; reconstruction-attack similarity below threshold;
> receipt verification reproducible across Rust-native **and** WASM.
