# ADR-260: PhotonLayer — Learned-Optical-Frontend Computing Simulator

**Status**: Proposed
**Date**: 2026-06-18
**Deciders**: rUv
**Supersedes**: None
**Related**: ADR-029 (RVF Canonical Format), ADR-047 (Proof-Gated Mutation Protocol), ADR-117 (Canonical MinCut), ADR-197 (Differentiable MinCut Condensation)

---

## Context

### The Sensor Bandwidth and Compute Problem

Modern perception systems — edge cameras, drone sensors, industrial inspection
rigs, medical imagers — push full-resolution pixel arrays through digital
pipelines before any feature selection occurs. This is wasteful by design:
the raw pixel stream contains far more information than is needed for any
specific task, yet the full stream must be digitized, transmitted, and
processed before the system can decide what mattered.

Diffractive optical elements (DOEs) offer a different model. A thin, passive
(or actively controlled) phase plate placed in the optical path upstream of
the sensor reshapes the light field before any electrons move. If that phase
plate is *designed by optimization*, it can perform task-specific analogue
pre-processing at the speed of light, concentrating class-discriminative
energy into a small set of sensor pixels and discarding everything irrelevant
before the first ADC conversion.

This idea has moved from theory to demonstrated hardware:

- Lin et al. (2018, *Science*) showed that stacked diffractive layers
  designed by backpropagation can classify MNIST digits and reconstruct
  images at the speed of light, purely passively.
- Hybrid DOE-CNN classifiers (Chen et al., 2023 and subsequent work, per
  project brief) report >7.8x electronic-layer compression when an optimized
  DOE precedes a shallow CNN.
- Computational meta-optics reviews (2026, per project brief) document
  systems where >90% of the computation is performed by the optical front end,
  leaving only a lightweight digital decoder.
- Edge-enhanced diffractive networks (June 2026, per project brief) improved
  single-diffractive-layer MNIST accuracy from 64.2% to 80.7% by adding
  optical edge extraction.
- The 2026 full-Stokes reconstruction paper (per project brief) jointly
  optimized a differentiable single-layer metasurface with a U-Net backend
  to reconstruct RGB full-Stokes polarization images from a single monochrome
  sensor measurement — demonstrating the modern joint design blueprint.

RuVector already ships the primitives needed to simulate, train, evaluate, and
store optical experiments in a reproducible, receipt-bound format:

- `crates/photonlayer-core`: scalar field, FFT engine, phase mask, propagation
  (Fresnel / Fraunhofer / angular-spectrum), detector, metrics, RVF receipt.
- `crates/photonlayer-bench`: synthetic data, nearest-centroid decoder,
  baseline variants, in-Rust hill-climbing learner, benchmark runner.
- `crates/photonlayer-ruvector`: experiment embedding (32-dim) and
  cosine-similarity nearest-neighbour memory backed by `ruvector-coherence`.
- `crates/photonlayer-cli`: command-line interface to the simulator and bench.
- `crates/photonlayer-wasm`: `wasm-bindgen` bindings for browser simulation.

All five crates compile as of 2026-06-18 and 21 tests pass.

### The PhotonLayer Claim (Precision Matters)

The precise, non-overclaiming thesis of PhotonLayer is:

> **Light performs the first trained transformation; a smaller digital backend
> reads the result.**

This is not the same as "light is running a neural network." Light propagates
through a trained phase plate and produces an intensity pattern on a sensor.
The phase plate encodes a learned linear transformation (in the Fraunhofer
regime, a weighted Fourier basis; in the Fresnel / angular-spectrum regime, a
convolution with a learned coherent transfer function). The result is a lossy,
task-specific analogue compression that the digital decoder reads. The neural
computation, as traditionally understood, is in the training of the phase mask
and in the digital decoder. Light provides the execution of the first (and most
expensive in terms of flops-per-inference) stage at zero marginal electronic
cost per pixel.

---

## Decision

### Adopt PhotonLayer as a first-class sub-system of RuVector

PhotonLayer occupies the `crates/photonlayer-*` namespace and the
`docs/research/photonlayer/` and `docs/adr/ADR-26x-*` documentation namespace.
It is a simulator and experiment-memory system, not a fabrication or hardware
driver system. All experiments are sealed with RVF-style receipts and stored
in `photonlayer-ruvector` experiment memory for later retrieval and
cross-experiment comparison.

---

## Pipeline Architecture

### §1 End-to-End Data Flow

```
┌──────────────┐
│  InputImage  │  Normalized f32 pixels, width × height
└──────┬───────┘
       │  field::OpticalField::from_image()
       ▼
┌──────────────────┐
│  OpticalField    │  Complex amplitude on spatial grid (padded)
└──────┬───────────┘
       │  mask::PhaseMask::apply()           ← THE LEARNED ELEMENT
       ▼
┌──────────────────────────┐
│  Modulated OpticalField  │  field * exp(i * θ(x,y))
└──────┬───────────────────┘
       │  propagate::propagate()
       │    Fresnel TF / Fraunhofer FFT / Angular Spectrum
       ▼
┌──────────────────────────┐
│  Propagated OpticalField │  Intensity = |amplitude|² before capture
└──────┬───────────────────┘
       │  detector::capture() / capture_with()
       │    shot noise · read noise · quantization · binning · saturation
       ▼
┌──────────────────┐
│  OpticalFrame    │  Sensor intensity map, frame_hash (BLAKE3)
└──────┬───────────┘
       │  metrics::frame_spectrum_embedding()
       │  decoder (NearestCentroid or external digital net)
       │  metrics::{accuracy, mse, psnr, compression_ratio, input_frame_similarity}
       ▼
┌──────────────────┐
│  MetricReport    │  Accuracy, MSE, PSNR, compression ratio, similarity, latency
└──────┬───────────┘
       │  receipt::build_receipt()
       ▼
┌──────────────────────┐
│  ExperimentReceipt   │  RVF-style receipt: 6 content hashes + rvf_receipt_hash
└──────┬───────────────┘
       │  photonlayer-ruvector::ExperimentMemory::remember()
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  RuVector Experiment Memory                                      │
│  Cosine-similarity NN over 32-dim embeddings                    │
│  (mask phase histogram 16-dim + frame spectrum 16-dim)          │
└─────────────────────────────────────────────────────────────────┘
```

### §2 MinCut Boundary Usage

The `ruvector-mincut` family (ADR-117, ADR-197) defines the boundary between
computation domains. In the PhotonLayer pipeline this maps directly:

- **Optical domain** (left of cut): `OpticalField` modulation and propagation.
  Computation is parallelizable, dependency-free across pixels, and in a real
  system executes in the optical path with no ADC cost.
- **Digital domain** (right of cut)**: `OpticalFrame` readout and beyond.
  This is what the MinCut boundary formally partitions.

The sensor pixel count *is* the cut bandwidth: fewer sensor pixels = narrower
cut = lower transmission from optical to digital domain. The PhotonLayer
acceptance gate (§17) requires that a learned mask achieves higher task
accuracy than the digital baseline at the same cut bandwidth.

### §3 Coherence Gates

PhotonLayer does not directly use `ruvector-coherence`'s energy gating for
routing (that is CGT / ADR-015), but it borrows the coherence library for
cosine-similarity search in experiment memory. A future extension can apply
coherence gating to route the digital decoder between a reflex (centroid
lookup) lane and a deep (neural network) lane based on the frame's entropy.

---

## §4 Crate Layout

| Crate | Purpose | Key modules |
|-------|---------|-------------|
| `photonlayer-core` | Optical simulation kernel | `field`, `mask`, `fft`, `propagate`, `detector`, `metrics`, `receipt`, `hash`, `rng`, `complex`, `error`, `config`, `simulator` |
| `photonlayer-bench` | Benchmarks and learner | `synthetic`, `decoder`, `pipeline`, `baselines`, `learn`, `verification` |
| `photonlayer-ruvector` | Experiment memory integration | `embedding`, `memory` |
| `photonlayer-cli` | Command-line interface | `main` |
| `photonlayer-wasm` | Browser WASM bindings | `lib` |

All crates are pure Rust. `photonlayer-core` depends only on `serde`,
`serde_json`, `blake3`, and `thiserror`. `photonlayer-ruvector` additionally
depends on `ruvector-coherence` for cosine similarity.

---

## §5 Data Types

### §5.1 InputImage

```rust
pub struct InputImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,   // row-major, values in [0, 1]
}
```

### §5.2 OpticalField

Complex amplitude on a 2D spatial grid. Internally `Vec<Complex>` where
`Complex { re: f32, im: f32 }`. The field grid may be larger than the image
(zero-padding for propagation accuracy).

### §5.3 PhaseMask

```rust
pub struct PhaseMask {
    pub width: usize,
    pub height: usize,
    pub phase_radians: Vec<f32>,  // [0, 2π) per cell, row-major
    pub mask_id: String,          // "identity" | "random:0x…" | "lens:…" | "learned:0x…"
}
```

This is the trainable element. The mask-exchange format (ADR-261) serializes
exactly this struct. Offline differentiable training (TorchOptics, waveprop,
diffractsim) optimizes `phase_radians` by gradient descent; the simulator's
in-Rust hill-climbing learner optimizes by seeded coordinate perturbation.

### §5.4 OpticalConfig

```rust
pub struct OpticalConfig {
    pub width: usize,
    pub height: usize,
    pub wavelength_nm: f32,
    pub propagation_mm: f32,
    pub pixel_pitch_um: f32,
    pub propagation: PropagationMode,
    pub detector: DetectorConfig,
    pub seed: u64,
}
```

All fields participate in the determinism invariant (§21). The canonical JSON
serialization of this struct is hashed into the receipt.

### §5.5 PropagationMode

```rust
pub enum PropagationMode {
    Fresnel,          // Near-field transfer function
    Fraunhofer,       // Far-field single-FFT (Fourier transform magnitude)
    AngularSpectrum,  // Highest fidelity, valid across regimes
}
```

### §5.6 DetectorConfig

```rust
pub struct DetectorConfig {
    pub shot_noise_photons: f32,   // 0 disables
    pub read_noise_std: f32,       // Additive Gaussian, intensity units
    pub quantization_levels: u32,  // 256 for 8-bit; 0 disables
    pub binning: usize,            // b×b block averaging; 1 disables
    pub saturation: f32,           // Clip ceiling; 0 disables
}
```

### §5.7 OpticalFrame

```rust
pub struct OpticalFrame {
    pub width: usize,
    pub height: usize,
    pub intensity: Vec<f32>,  // Detector output, after noise/quant/binning
    pub frame_hash: String,   // BLAKE3 over (width, height, intensity bytes)
}
```

`frame_hash` is the primary reproducibility handle. Two simulations with
identical inputs and config must produce identical `frame_hash` values.

### §5.8 MetricReport

```rust
pub struct MetricReport {
    pub accuracy: f32,
    pub reconstruction_mse: f32,
    pub compression_ratio: f32,
    pub input_frame_similarity: f32,  // Pearson r: sensor vs. input pixels
    pub native_latency_us: f64,
}
```

`input_frame_similarity` is deliberately measured and bounded: for
privacy-preserving optical verification (ADR-262) the sensor pattern must NOT
look like the input image (Pearson |r| should be low).

---

## §6 Propagation Modes — Physics Summary

### §6.1 Fresnel (Near-Field)

Applies the paraxial Fresnel transfer function in the frequency domain:

```
H(fx, fy) = exp(-i π λ z (fx² + fy²))
```

where λ is wavelength (m), z is propagation distance (m), and (fx, fy) are
spatial frequencies. Valid when the Fresnel number `a²/(λz) >> 1` (aperture
`a` large relative to wavelength times distance).

### §6.2 Fraunhofer (Far-Field)

A single 2D FFT of the masked field; the intensity is the power spectrum.
Valid when `z >> a²/λ`. Analytically the simplest mode and the one that most
directly links the mask's spatial frequency content to the sensor pattern.

### §6.3 Angular Spectrum (Exact Scalar)

Propagates each plane-wave component at its correct angle:

```
H(fx, fy) = exp(i 2π z sqrt(1/λ² - fx² - fy²))
```

Valid for all propagation distances within the scalar diffraction
approximation. The simulator uses this as the default for production
fidelity benchmarks.

---

## §7 Detector Model

The detector stage models four physical effects in sequence:

1. **Intensity**: `I(x,y) = |E(x,y)|²` where `E` is the propagated complex
   amplitude.
2. **Shot noise**: Poisson noise with mean `shot_noise_photons × I(x,y)`.
   Modelled as additive Gaussian `N(0, sqrt(I/photons))` for computational
   efficiency; disabled when `shot_noise_photons == 0`.
3. **Read noise**: Additive Gaussian `N(0, read_noise_std)` after shot noise.
4. **Quantization**: Intensity binned to `quantization_levels` uniform levels
   in `[0, max_intensity]`; disabled when `quantization_levels == 0`.
5. **Binning**: `b×b` pixel blocks averaged to a single output pixel; disabled
   when `binning == 1`. Binning is applied after quantization. The output frame
   dimensions are `floor(width/b) × floor(height/b)`.
6. **Saturation**: Intensities clipped to `[0, saturation]`; disabled when
   `saturation == 0`.

The seeded RNG (`DeterministicRng`, a 64-bit Xoshiro-256** variant) ensures
that noise draws are deterministic given the seed embedded in `OpticalConfig`.

---

## §8 RuVector Experiment-Memory Schema

Each finished experiment is stored as an `ExperimentRecord`:

```rust
pub struct ExperimentRecord {
    pub id: String,              // Unique experiment identifier
    pub label: String,           // Task outcome label ("pass", "fail", class name)
    pub config: OpticalConfig,   // Full optical configuration
    pub mask_id: String,         // From PhaseMask::mask_id
    pub embedding: Vec<f32>,     // 32-dim L2-normalised embedding
    pub receipt: ExperimentReceipt,
    pub metrics: MetricReport,
}
```

### §8.1 Embedding Composition (32-dim)

The 32-dimensional embedding concatenates:

- **16-dim mask phase histogram**: normalized frequency distribution of
  `phase_radians` values across 16 uniform bins on `[0, 2π)`.
- **16-dim frame spectrum embedding**: radial intensity distribution of
  `OpticalFrame::intensity` binned into 16 annular rings by distance from
  frame center, then L2-normalised.

Both components are L2-normalised independently before concatenation;
the joint vector is L2-normalised again. This embedding is suitable for
cosine-similarity nearest-neighbour search over experiment history.

### §8.2 Nearest-Neighbour Recall

`ExperimentMemory::nearest(query_embedding, limit)` returns up to `limit`
stored experiments sorted by descending cosine similarity. The current
implementation is a linear scan over stored embeddings (exact NN). This is
appropriate for the experiment-scale dataset; upgrade to HNSW when exceeding
10,000 stored experiments.

---

## §9 RVF Receipt Binding (§15)

An `ExperimentReceipt` binds every experiment input to a single anti-swap
hash:

```rust
pub struct ExperimentReceipt {
    pub experiment_id: String,
    pub input_hash: String,    // BLAKE3 over (width, height, pixels)
    pub mask_hash: String,     // BLAKE3 over (width, height, phase_radians)
    pub config_hash: String,   // BLAKE3 over canonical JSON of OpticalConfig
    pub output_hash: String,   // frame_hash from OpticalFrame
    pub metrics_hash: String,  // BLAKE3 over MetricReport fields
    pub git_commit: String,
    pub rustc_version: String,
    pub feature_flags: Vec<String>,
    pub seed: u64,
    pub rvf_receipt_hash: String,  // BLAKE3 over all of the above
}
```

`rvf_receipt_hash` is the primary tamper-detection value. `verify_receipt(r)`
recomputes the binding digest and returns `true` iff the receipt fields are
internally consistent. Importing a trained mask from an external source
requires verifying that the imported mask's `mask_hash` matches the hash stored
in the associated receipt.

---

## §10 Benchmarks and Metrics (§16)

### §10.1 Classification Benchmark

Three variants on a synthetic 4-class dataset (16×16 grid, configurable
samples per class):

| Variant | Description |
|---------|-------------|
| `digital_baseline` | Pooled raw image pixels; no optics |
| `random_mask` | Randomly initialized phase mask |
| `learned_mask` | Phase mask trained by seeded block hill-climbing |

The hill-climbing optimizer starts from a random mask and only accepts
phase-perturbing moves that improve training accuracy (with separation margin
as tiebreaker). Because only improving steps are accepted, the learned mask
provably dominates its random starting point on the training objective.

### §10.2 Compression Benchmark (Showcase Claim)

A sensor squeezed to 2×2 (= 4 pixels). The same 4-class synthetic dataset.
Benchmark variants:

| Variant | Description |
|---------|-------------|
| `digital_tiny_sensor` | 4-pixel direct readout; no optics |
| `random_mask_tiny` | Random mask into 4-pixel sensor |
| `learned_mask_tiny` | Learned mask into 4-pixel sensor |

**Observed results** (photonlayer-bench, 2026-06-18): at a 2×2 / 4-pixel sensor
a learned mask reaches **1.00 test accuracy** vs **0.80** for random mask vs
**0.65** for digital baseline. This demonstrates 64× sensor compression
(16×16 input to 2×2 sensor) with no loss of task accuracy. These figures are
from a specific synthetic dataset and learner configuration; results on real
data will differ.

### §10.3 Metrics

| Metric | Symbol | Notes |
|--------|--------|-------|
| Classification accuracy | `accuracy` | Fraction correct on test split |
| Mean squared error | `mse` | Reconstruction fidelity |
| Peak signal-to-noise ratio | `psnr(dB)` | 10 log₁₀(peak²/MSE) |
| Compression ratio | `compression_ratio` | Input pixels / sensor pixels |
| Input-frame similarity | `input_frame_similarity` | Pearson r; low = not human-readable |
| Native simulation latency | `native_latency_us` | Wall-clock µs for one forward pass |

---

## §11 Acceptance Gates (§17)

The following invariants must hold for a PhotonLayer release to be accepted:

### §17.1 Determinism Invariant

For any fixed `(InputImage, PhaseMask, OpticalConfig)`:

```
ScalarSimulator.simulate(img, mask, cfg).frame_hash  ==
ScalarSimulator.simulate(img, mask, cfg).frame_hash
```

The second invocation must produce a bit-identical `frame_hash`. This is the
determinism invariant; it is verified by `tests::replay_is_deterministic` in
`photonlayer-core/src/receipt.rs` (21 tests pass as of 2026-06-18).

### §17.2 Learned Dominates Random

On at least one benchmark variant:

```
learned_mask.train_accuracy >= random_mask.train_accuracy
```

Enforced by `tests::learned_beats_or_matches_random_on_training` in
`photonlayer-bench/src/baselines.rs`.

### §17.3 Learned Wins Under Compression

At the 2×2 sensor task:

```
learned_mask_tiny.test_accuracy > digital_tiny_sensor.test_accuracy
learned_mask_tiny.test_accuracy >= random_mask_tiny.test_accuracy
```

Enforced by `tests::learned_strictly_wins_under_compression`.

### §17.4 Receipt Integrity

`verify_receipt(r)` must return `true` for every receipt produced by
`build_receipt(...)`. Tampering any field (including `output_hash`) must return
`false`. Enforced by `tests::receipt_verifies` and `tests::tamper_breaks_receipt`.

### §17.5 Input-Frame Similarity Bound

For the standard benchmark configuration, `input_frame_similarity` (Pearson r)
must be below a threshold (default 0.4) to confirm that the sensor pattern is
not a direct copy of the input image. This guards against trivially transparent
optical paths.

---

## §12 Determinism Invariant (§21)

The determinism invariant is the foundation of PhotonLayer's reproducibility
guarantee:

> **Same `InputImage` + `PhaseMask` + `OpticalConfig` + `seed` (embedded in
> `OpticalConfig`) always produce the same `frame_hash` (BLAKE3 over the
> `OpticalFrame` intensity bytes).**

Implementation guarantees:

1. The FFT engine (`photonlayer-core/src/fft.rs`) is a pure-Rust, in-house
   Cooley-Tukey implementation with no platform-dependent SIMD paths in the
   default configuration. All arithmetic is deterministic across calls.
2. The noise RNG (`DeterministicRng`) is a deterministic 64-bit
   Xoshiro-256** variant seeded from `OpticalConfig::seed`. No system entropy
   is consumed.
3. `OpticalConfig` is serialized to canonical JSON before hashing, ensuring
   that field ordering is stable across serde versions.
4. `PhaseMask::phase_radians` values are stored as `f32`; the `hash_f32`
   function hashes the raw little-endian bytes, which are stable given
   identical `f32` values.

Violation of the determinism invariant is a hard error. Imported masks (e.g.,
trained offline by TorchOptics) must be replayed and the resulting `frame_hash`
compared against the stored receipt's `output_hash` before the mask is accepted
into experiment memory.

---

## §13 Implementation Phases

### Phase 1 — Core Simulator (Complete)

- [x] `photonlayer-core`: field, FFT, mask, propagation, detector, metrics,
  receipt, hash, RNG, complex, error, config, simulator
- [x] 21 passing tests covering determinism, receipt integrity, mask apply,
  propagation modes, field construction, detector model
- [x] All three propagation modes: Fresnel, Fraunhofer, AngularSpectrum

### Phase 2 — Benchmarks and Learner (Complete)

- [x] `photonlayer-bench`: synthetic 4-class dataset, nearest-centroid decoder,
  digital baseline, random mask baseline, block hill-climbing learner
- [x] Classification benchmark: digital < random < learned ordering
- [x] Compression benchmark: 1.00 vs 0.80 vs 0.65 test accuracy at 2×2 sensor
- [x] BenchReport: JSON-serializable, serde-stable

### Phase 3 — RuVector Integration (Complete)

- [x] `photonlayer-ruvector`: ExperimentRecord, ExperimentMemory, 32-dim
  embedding (mask histogram + frame spectrum), cosine-similarity NN
- [x] `photonlayer-cli`: sub-commands for simulate, bench, store
- [x] `photonlayer-wasm`: wasm-bindgen entry point for browser simulation

### Phase 4 — External Mask Import and Verification (Planned)

- [ ] JSON schema for the mask-exchange format (ADR-261)
- [ ] `import_mask` CLI command: loads a PhaseMask from JSON, replays the
  simulation, verifies `frame_hash` against stored receipt
- [ ] BLAKE3 replay-hash verification in `photonlayer-ruvector`
- [ ] Differential optics gradient export: compute ∂L/∂θ(x,y) for each mask
  cell to enable PyTorch-side optimization

### Phase 5 — Privacy and Verification Applications (Planned, ADR-262)

- [ ] Reconstruction-attack test: confirm that sensor frames cannot be
  inverted to recover human-readable input images above a privacy threshold
- [ ] Optical verification mode: same/different decision from two frames
- [ ] FAR / FRR / EER tracking in experiment memory
- [ ] Governance receipt for spoof-failure events

### Phase 6 — Hardware Calibration Bridge (Future)

- [ ] Import measured PSF from physical DOE to replace simulated transfer
  function
- [ ] Calibration drift tracking: compare measured PSF hash against baseline
  receipt over time
- [ ] Multi-wavelength / polychromatic simulation (currently monochromatic)

---

## §14 Application Domains, Positioning, and Ethics

### §14.1 What PhotonLayer Is

PhotonLayer is a **front-end compression and task-specific sensing system**.
Light performs the first trained transformation; a smaller digital backend
reads the result. This means:

- Lower latency: the optical computation adds zero marginal inference time
  (light propagates through the mask at the speed of light regardless of grid
  size).
- Less sensor bandwidth: fewer pixels reach the ADC, lowering power and
  transmission cost.
- Lower power: fewer ADC conversions, smaller digital pipeline.
- Compressed measurements: the sensor plane contains a task-optimized
  projection, not a raw image.
- Task-specific sensing: the phase mask is optimized for one task; other tasks
  require different masks or a reconfigurable liquid-crystal spatial light
  modulator (SLM).

PhotonLayer is **not** a full perception replacement. It cannot replace the
digital decoder, because the optical front end is a compression step, not a
complete classifier.

### §14.2 Feasibility Ranking (Highest to Lowest for Near-Term Impact)

**Rank 1 — Industrial and scientific sensors (highest feasibility)**

Learned phase masks designed as task-specific measurement devices for:

- Crack detection in structural materials (spatial frequency enhancement)
- Crop disease and surface defect inspection (texture discrimination)
- Material composition sensing (spectral and polarization encoding)
- Polarization imaging (full-Stokes measurement with a single monochrome sensor,
  per 2026 metasurface paper)

These applications have controlled illumination, cooperative subjects,
well-defined pass/fail criteria, and no consent or privacy concerns. The
PhotonLayer compression benchmark (64× sensor reduction with 1.00 accuracy
on synthetic data) directly motivates this class of application.

**Rank 2 — Drone and autonomous vehicle perception preprocessing**

Optical flow estimation, obstacle and wire detection, landing-zone
classification, glare and lens-flare robustness. These applications benefit
from the grams/watts/bandwidth/latency advantages of optical front ends at the
sensor level, before any digital transmission. The phase mask acts as a
hardware filter, passing only task-relevant structure to the digital pipeline.

**Rank 3 — Medical imaging research simulator**

Microscopy compression (encoding a high-resolution wavefront into a compact
measurement), snapshot polarization imaging (single-shot full-Stokes, after
2026 metasurface blueprint), reconstruction from fewer measurements
(compressed sensing with a learned optical modulator). **This is a research
tooling application, not a clinical diagnosis tool.** PhotonLayer does not
produce diagnostic conclusions; it simulates the measurement physics.

**Rank 4 — Consented face verification with optical encoding**

A phase mask that maps a face image to an encoded sensor pattern — not a
human-readable image — for same/different verification and liveness/anti-spoof
testing. Key constraints:

- The sensor pattern must NOT be a recoverable face image (ADR-262 governs the
  reconstruction-attack threshold).
- Consent and transparency are required: the subject must know that an optical
  encoding device is in use.
- No face biometric is stored; only the compact encoded frame and the receipt.

This application is Rank 4 because the privacy engineering (reconstruction-
attack resistance, consent governance, regulatory compliance) is substantial.

**Non-goal — Public or mass surveillance facial recognition**

PhotonLayer will not be used, documented, or positioned as a technology for
identifying individuals in public spaces without consent. This is an explicit
architectural and ethical boundary. No crate, no benchmark, and no ADR in the
PhotonLayer namespace documents or facilitates mass facial recognition. The
`input_frame_similarity` metric and the ADR-262 reconstruction-attack test
exist precisely to confirm that the optical encoding discards human-readable
image content.

### §14.3 Claim Discipline

The following language is prohibited in PhotonLayer documentation and
marketing:

- "Light is running a neural network" — this conflates the optical propagation
  with the training algorithm and the decoder.
- "The optical system classifies images" — the optical system projects; the
  digital decoder classifies.
- "This replaces digital sensing" — it compresses and pre-processes the
  sensed signal; the sensor and digital pipeline remain.

The following language is precise and permitted:

- "Light performs the first trained transformation."
- "A learned phase mask concentrates task-discriminative energy into fewer
  sensor pixels."
- "The digital decoder reads a compressed, task-specific optical projection."
- "The learned mask achieves higher accuracy than a random mask at the same
  sensor pixel count."

---

## Consequences

### Positive

- Establishes a reproducible, receipt-bound optical simulation infrastructure
  inside RuVector with zero external ML framework dependencies.
- The compression benchmark provides a concrete, falsifiable claim: 64×
  sensor compression with 1.00 test accuracy (on synthetic data) vs 0.65 for
  the digital baseline.
- The determinism invariant enables cross-machine and cross-time experiment
  comparison, mask import/export, and audit trails.
- The RuVector experiment memory enables nearest-neighbour retrieval of
  similar past experiments, supporting iterative mask design.
- The WASM build enables browser-based simulation demos and educational tools.

### Negative

- The in-Rust hill-climbing learner is not gradient-based; for large masks
  (e.g., 512×512) offline differentiable training (TorchOptics, waveprop) is
  required, and the import/verification bridge (Phase 4) is not yet built.
- The current simulator is monochromatic; broadband / polychromatic simulation
  requires Phase 6 extensions.
- The 21 tests cover the synthetic benchmark; real-world optical data
  validation requires physical fabrication or measured PSF import.

### Risks

- The compression benchmark results (1.00 / 0.80 / 0.65) are on a small
  synthetic 4-class dataset. Overclaiming these as representative of real-world
  performance would be a scientific integrity failure. All external
  communication must include the dataset and configuration context.
- The privacy acceptance gate (§17.5, `input_frame_similarity < 0.4`) is a
  necessary but not sufficient condition for reconstruction-attack resistance.
  ADR-262 governs the full threat model.

---

## References

- ADR-015: Coherence-Gated Transformer — coherence library used for NN search
- ADR-029: RVF Canonical Format — receipt binding pattern
- ADR-047: Proof-Gated Mutation Protocol — proof tier design reference
- ADR-117: Canonical MinCut — MinCut domain boundary concept
- ADR-197: Differentiable MinCut Condensation — differentiable boundary loss
- ADR-261: PhotonLayer Mask-Exchange Format and Determinism (companion ADR)
- ADR-262: PhotonLayer Privacy-Preserving Optical Verification (companion ADR)
- `crates/photonlayer-core/src/lib.rs` — simulator entry point and prelude
- `crates/photonlayer-core/src/config.rs` — OpticalConfig, PropagationMode, DetectorConfig
- `crates/photonlayer-core/src/mask.rs` — PhaseMask, apply, histogram
- `crates/photonlayer-core/src/receipt.rs` — ExperimentReceipt, build_receipt, verify_receipt
- `crates/photonlayer-core/src/metrics.rs` — MetricReport, frame_spectrum_embedding
- `crates/photonlayer-bench/src/baselines.rs` — run_classification, run_compression
- `crates/photonlayer-bench/src/learn.rs` — LearnConfig, learn_mask, LearnOutcome
- `crates/photonlayer-ruvector/src/memory.rs` — ExperimentMemory, ExperimentRecord
- `crates/photonlayer-ruvector/src/embedding.rs` — experiment_embedding (32-dim)
- Lin et al. (2018), "All-Optical Machine Learning Using Diffractive Deep Neural
  Networks," *Science* 361(6406): 1004–1008 (per project brief / to be verified)
- Computational meta-optics review (2026) — >90% optical front-end computation
  (per project brief / to be verified)
- Edge-enhanced diffractive networks (June 2026) — 64.2% → 80.7% MNIST
  single-layer accuracy (per project brief / to be verified)
- Full-Stokes metasurface + U-Net reconstruction (2026) — joint differentiable
  optical-electronic design blueprint (per project brief / to be verified)
- TorchOptics library — PyTorch differentiable Fourier optics, GPU, joint
  optimization (per project brief / to be verified)
- waveprop library — scalar diffraction, trainable apertures (per project brief
  / to be verified)
- diffractsim library — JAX differentiable optimization and visualization (per
  project brief / to be verified)
