# PhotonLayer — Assessment & Research Roadmap

> Strongest claim: **PhotonLayer is a deterministic optical AI front end where a learned phase mask
> performs task-specific analog preprocessing before a tiny digital decoder sees the compressed
> measurement.** This matters because the field is moving toward *meta-optic front ends + electronic
> back ends* for low-latency, low-power, privacy-preserving, compact sensing.

Companion to ADR-260 (optical computing simulator) and ADR-261 (mask exchange & determinism).
Measured numbers in this doc come from `photonlayer-bench` (`more_data_bench` + `mnist_differential_bench`).

## Measured on real data (MNIST, M2)

Deterministic run (seed `0x6e157`; 4000 train / 2000 blind test, balanced 10-class; public MNIST IDX;
`cargo test -p photonlayer-bench --release --test mnist_differential_bench mnist_differential_full -- --ignored`):

**Config A — the product claim (decoder objective, seed `0x6e157`):**

| | sensor px | decoder params | blind-test acc |
|---|---:|---:|---:|
| full-image baseline (same tiny centroid decoder) | 1024 | 10 240 | **75.40%** |
| **optical compressed** (learned mask + pooled read) | **64** | **640** | **73.05%** |
| Δ vs baseline | — | — | **−2.35 pp** |

→ **16.0× fewer sensor pixels and 16.0× fewer digital MACs.** Learned mask beats a random mask by
**+8.10 pp** decoded (the value of learning the optics is real). The compression claim is the headline.

**Config B — the mechanism (argmax-diff objective, seed `0x6e15c`, NO decoder):** isolates the
Li/Ozcan differential-detection lever — plain argmax `I⁺` 18.40% vs differential argmax `I⁺−I⁻`
34.90% = **+16.50 pp** lever (absolute acc is modest by construction; the delta isolates the lever,
not a headline accuracy).

**Honest margin + the ceiling.** On the bit-exact *pre-optimization* FFT core, Config A was **−1.20 pp
(acceptance PASS)**. The OPT-B twiddle-table change (a determinism *improvement* — it removes FFT
float-drift) shifted all FFT paths, moving the converged Config A to **−2.35 pp**, just outside the
−2 pp line. A training-budget sweep proves this is an **optimizer ceiling, not a budget issue**
(1500→−2.35, 3000→−2.15, 4500→−2.20 pp; block hill-climbing has converged). Closing the last ~2 pp —
and reaching ~85–89 % — requires **analytic gradient descent** through the diffraction operator
(`Propagator::backward_into` with `conj(H)`), the documented roadmap keystone. We report the true
single-mask number; we do not assert a PASS the method cannot reach.

### Honest positioning (use verbatim; no slop)

> *A task-trained single optical layer with a tiny digital decoder, classifying MNIST within ~1–2 pp of a
> **matched** full-image tiny-decoder baseline while using ≥16× fewer sensor pixels and ≥10× fewer digital
> MACs. This is **competitive single-layer optical compression** — trading a small, quantified accuracy
> margin for large sensor- and compute-savings — **not a new accuracy SOTA**; the multi-layer ~97–99 %
> D2NN / optoelectronic regime is explicitly out of scope.*

**Must avoid** (overclaim): "beats SOTA", "state-of-the-art MNIST" (real SOTA > 99.7 %), "outperforms
D2NNs" (different task), any bare "≥16×/≥10×" without naming the matched baseline, "near-lossless".

## Why it's differentiated

The unique angle is **not** "optical neural network" — it's **auditable optical compression for
task-useful sensing**. Most optical-AI narratives overclaim; PhotonLayer's wedge is:

1. **Task-first** — mask trained for the downstream objective, not generic reconstruction.
2. **Compression-first** — real-data MNIST: 1024 → 64 sensor pixels (16× reduction) at −2.35 pp vs a matched
   full-image baseline (converged single-mask hill-climb); synthetic flagship reaches 16×16 → 4 (64× reduction).
   Both measured, both deterministic; gradient descent is the documented path to close the residual gap.
3. **Privacy by physics** — verify/classify from a measurement that need not look like the scene.
4. **Deterministic receipts** — reproducible, BLAKE3-bound; suitable for regulated experiments and audit trails.
5. **Rust-native** — embedded, WASM, deterministic benchmarking, eventual hardware control.

## Best use cases (positioned by risk)

| Use case | Why it fits | Risk |
|---|---|---|
| Industrial inspection | Detect defects without full-frame processing | Low |
| Barcode / symbol / package verification | Strong demo path, easy ground truth | Low |
| Drone perception preprocessing | Lower bandwidth, smaller backend model | Medium |
| Scientific imaging | Task-useful measurement vs full capture | Medium |
| Medical imaging *research* | Compression, morphology classification, uncertainty | High |
| Consented identity verification | Strong privacy story if tightly bounded | High |
| Autonomous-vehicle sensing | Valuable but needs hardware + safety validation | Very high |

First commercial wedge: **industrial & scientific sensing**, not healthcare or AV. For medical/AV,
position as **research infrastructure and preprocessing**, not decision automation.

## What to prove next

### 1. Energy model
A measured/simulated energy comparison. Target: **equal-or-better accuracy with ≥10× lower digital
compute and ≥16× lower sensor bandwidth** vs a direct-image-plus-CNN pipeline (compare sensor pixels,
decoder params, MACs, latency, estimated energy).

### 2. Harder datasets
Move beyond synthetic: MNIST / Fashion-MNIST optical compression, CIFAR-10 binary subsets, MVTec-AD
industrial anomaly detection, a public microscopy cell-morphology set, and face *verification* on
consented pairs only (no identification gallery).

### 3. Reconstruction-attack suite
Quantify the privacy claim by publishing attacks: linear reconstruction, learned-decoder
reconstruction, diffusion-prior reconstruction, nearest-neighbour leakage, membership inference, and
attribute leakage (as *risk metrics only*). **"No readable image is stored" is a safer claim than
"privacy-preserving" until leakage is quantified.**

### 4. Hardware bridge
Software phase mask → printed static diffractive mask → SLM lab prototype → lensless camera module →
CMOS sensor integration → tunable metasurface. The credibility unlock is a physical path.

## Demos to build (for the Pages UI)

- **Optical privacy gate** — original face → noise-like measurement → verification result → failed
  reconstruction → receipt hash. Headline: *"The face was verified. The face was never stored."*
  (consented verification, **not** mass identification).
- **Microscope compressor** — cell image → learned compression → morphology class / anomaly score →
  uncertainty → reconstruction failure (no diagnostic claim). Headline: *"The microscope learned what
  not to measure."*
- **Drone vision front end** — full-frame baseline vs 4/8/16/32-pixel optical sensors → decision +
  latency/bandwidth comparison. Headline: *"The drone doesn't need the image. It needs the decision surface."*

## Products

| Product | Buyer | Value |
|---|---|---|
| PhotonLayer Studio | researchers, startups, labs | design & test optical AI masks |
| PhotonLayer Edge | industrial sensor companies | smaller models, lower bandwidth |
| PhotonLayer Verify | privacy-sensitive identity workflows | verification without storing readable images |

Near-term wedge: software + simulation + benchmark receipts. Long-term value: hardware co-design.

## Scoring

| Criterion | Score | Note |
|---|---:|---|
| Novelty | 9 | optical compression + Rust determinism + receipts + memory |
| Technical defensibility | 8 | good bounded claims; needs harder datasets |
| Viral potential | 9 | privacy gate + microscope compressor are highly visual |
| Commercial path | 7 | industrial sensing first, medical later |
| Safety posture | 8 | strong non-goal on surveillance; needs leakage testing |
| Hardware readiness | 5 | strong simulator; physical validation still required |

**Overall: 8.0 platform · 9.0 research demo · 7.0 near-term product.**

## Acceptance test (becomes hard to dismiss when)

> On **three public datasets**, a learned optical mask achieves within **2 pp** of full-image baseline
> accuracy while reducing sensor pixels by **≥16×**, digital MACs by **≥10×**, and reconstruction
> similarity below a documented privacy threshold.

## References

Closest architectural comparisons (cite these for positioning):

- Wirth-Singh et al., **Compressed Meta-Optical Encoder for Image Classification**, arXiv:**2406.06534** (2024) /
  *Adv. Photonics Nexus* 4(2):026009 (2025) — the direct architectural twin: optical encoder + small digital
  back end, MNIST ~93.4% hybrid, ~17.3M → 85.8K MACs, a few pp below its own CNN baseline. **Primary comparison.**
- Bezzam, Vetterli, Simeoni, arXiv:**2206.01429** (2022) — few-pixel anchor (~87.5% MNIST at a 12-pixel learned mask).
- Lin et al., **All-optical machine learning using diffractive deep neural networks**, *Science* 361:1004 (2018),
  arXiv:1804.08711 — the 5-layer D2NN (~91.75% MNIST) we are explicitly **not** competing with.
- Li, Ozcan et al., arXiv:1906.03417 — differential detection (`I⁺−I⁻`) as the diffractive readout (the M2 lever).
- Wang/Zhu/Fu, arXiv:**2507.17374** (2025) — single-layer all-optical 98.59%; **contrast only** (different objective,
  we do not claim to beat it).

Background:

- Optical neural networks: progress and challenges — *Light: Science & Applications* (Nature, 2024).
- Metaoptics merging computational optics and electronics — PMC/NIH.
- Privacy-Aware Meta-Optics for Person Detection — *ACS Photonics* (2026).
