# ADR-011: Acoustic Feature Extraction

## Status
Accepted

## Date
2026-08-03

## Context

The platform computes Perch 2.0 embeddings, which are excellent for retrieval and
useless for explanation: a 1536-dimensional vector tells a researcher nothing about
*why* two calls are neighbors. Interpretable acoustic descriptors are needed for three
distinct consumers:

1. **Visualization** — colouring, sizing, and positioning points in the manifold view,
   and driving the per-segment feature panel (ADR-012).
2. **Interpretation** — `sevensense-interpretation` currently emits hardcoded
   `SharedFeature { name: "frequency_modulation", … }` strings. Evidence packs claiming
   two calls share a "rapid upward frequency sweep" must measure that, not assert it.
3. **Quality gating** — rejecting wind, rain, and clipping before spending inference
   budget on them.

What exists today is thin. `CallSegment` carries `zero_crossing_rate`, `rms_energy`,
`peak_amplitude`, and an `Option<f32> spectral_centroid` whose builder
(`with_spectral_centroid`) has no callers anywhere in the workspace — the field is
always `None`. Spectral spread, crest, flatness, entropy, slope, rolloff, and
amplitude/frequency modulation do not exist in any form.

`MelSpectrogram` is real and correct, but it is the wrong substrate for these
descriptors. Mel bins are perceptually spaced and, as configured
(`for_5s_segment`), band-limited to 500–15000 Hz with log scaling applied in place.
A centroid computed over log-scaled mel bins is not a centroid in Hz and is not
comparable across configurations.

## Decision

We add a `features` module to `sevensense-audio` that computes descriptors from a
**linear-frequency power spectrum**, independent of the mel path, and we populate the
existing `CallSegment` fields from it.

### 1. Two-level output

```rust
pub struct AcousticFeatures {
    pub frames: Vec<SpectralFrame>,   // per-frame, drives live visualization
    pub summary: FeatureSummary,      // per-segment, drives radar plot and evidence
    pub config: FeatureConfig,
}
```

Per-frame values are what a live display needs. Per-segment summaries are what a
comparison, a label, or an evidence pack needs. Computing only one of the two forces
consumers to re-derive the other badly.

### 2. Per-frame descriptors

For frame magnitude spectrum `m[k]` at bin frequencies `f[k]`, with power
`p[k] = m[k]²` and normalized power `p̂[k] = p[k] / Σp`:

| Descriptor | Definition | Unit |
|---|---|---|
| `centroid_hz` | `Σ f[k]·p̂[k]` | Hz |
| `spread_hz` | `sqrt(Σ (f[k] − centroid)²·p̂[k])` | Hz |
| `skewness` | third standardized moment of `p̂` over `f` | — |
| `rolloff_hz` | smallest `f` with cumulative power ≥ 85% | Hz |
| `flatness` | `exp(mean(ln p)) / mean(p)` (geometric ÷ arithmetic mean) | 0–1 |
| `tonality` | `1 − flatness` | 0–1 |
| `crest` | `max(m) / rms(m)`, normalized to 0–1 by `1 − 1/crest` | 0–1 |
| `entropy` | `−Σ p̂ ln p̂ / ln(n_bins)` | 0–1 |
| `slope` | least-squares slope of `m[k]` against `f[k]` | 1/Hz |
| `dominant_hz` | `argmax_f m[k]`, parabolically interpolated | Hz |
| `energy_db` | `20·log10(rms(frame))` | dBFS |
| `zcr` | zero crossings ÷ frame length | 0–1 |

Flatness, crest, and entropy are each normalized to 0–1 so that a radar plot can show
them on shared axes without per-axis rescaling that would hide cross-feature structure.

Two numerical details matter and are easy to get wrong. Flatness uses
`exp(mean(ln p))` rather than a direct product, because the product of ~1000 bin
powers underflows `f32` immediately. Entropy skips zero-power bins rather than
relying on `0·ln 0` evaluating to `NaN`.

Dominant frequency uses parabolic interpolation over the peak bin and its two
neighbours. Without it, resolution is quantized to `sample_rate / n_fft` ≈ 15.6 Hz at
the default configuration, which is coarse enough to be visible as banding in a
frequency track.

### 3. Modulation descriptors

Amplitude and frequency modulation are properties of a *sequence* of frames, not of a
frame, and are computed over the segment:

- **Amplitude modulation**: take the frame energy envelope, remove its mean, and take
  the FFT. `am_rate_hz` is the peak modulation frequency within 2–200 Hz;
  `am_depth` is the peak magnitude normalized by the mean envelope, clamped to 0–1.
- **Frequency modulation**: the same procedure applied to the `centroid_hz` track.
  `fm_rate_hz` is the peak; `fm_extent_hz` is the standard deviation of the centroid
  track, which is the more robust measure of sweep width.

The lower bound of 2 Hz excludes the DC component and the slow drift of the noise
floor. The upper bound is `min(200 Hz, frame_rate / 2)` — the envelope is sampled at
the frame rate, so Nyquist applies to it. At the default hop of 320 (100 fps) the
effective ceiling is **50 Hz**, not 200 Hz.

This matters for fast trills. Most passerine trills fall in 10–50 Hz and are measured
correctly at the default hop, but a trill above 50 Hz aliases and will be reported as a
lower rate. Measuring those requires a smaller hop: hop 160 gives 200 fps and a 100 Hz
ceiling. The 200 Hz constant is the band limit for such configurations, not a promise
about the default one.

### 4. Summary statistics

`FeatureSummary` carries mean and standard deviation for each per-frame descriptor,
plus the modulation values and a `voiced_fraction` (proportion of frames above the
energy gate). Means are computed over **voiced frames only**. Averaging the centroid
across silence pulls it toward whatever the noise floor happens to be, which is the
single most common way this kind of summary becomes meaningless.

### 5. Configuration

```rust
pub struct FeatureConfig {
    pub n_fft: usize,        // 1024 — 31.25 Hz resolution at 32 kHz
    pub hop_length: usize,   // 320  — 100 fps, aligns with the mel path
    pub sample_rate: u32,    // 32_000
    pub f_min: f32,          // 200.0 — below most avian fundamentals, above rumble
    pub f_max: f32,          // 16_000.0 — Nyquist
    pub rolloff_percent: f32,// 0.85
    pub energy_gate_db: f32, // -60.0
}
```

`n_fft` is 1024 rather than the mel path's 2048: the descriptors need time resolution
more than frequency resolution, and a 32 ms window tracks a fast trill that a 64 ms
window smears. The hop matches the mel path so frame indices are directly comparable.

Analysis is restricted to `[f_min, f_max]`. Including DC and near-DC bins drags the
centroid toward zero whenever there is any low-frequency rumble, which is always.

### 6. Integration

`EnergySegmenter` gains a feature pass that populates `spectral_centroid` and
`dominant_frequency` on `CallSegment` — the fields that already exist and are always
`None` today. `AcousticFeatures` is attached to the segment record for the API to
serve.

### 7. Performance

Extraction runs at the same hop as the mel path and shares its FFT planner strategy
(`realfft`, plan created once per extractor, reused per frame). Per-frame cost is
dominated by one real FFT of size `n_fft`. Frames are processed with `rayon` for
offline batches; the streaming path processes frames singly as they arrive, since
per-frame parallelism at 100 fps costs more in scheduling than it saves.

Target: **< 5 ms per 5-second segment** on one core, which is ~2% of the embedding
inference budget and therefore not worth optimizing further.

## Consequences

### Positive
- The visualization has real data behind every axis.
- Evidence packs can cite measured values instead of hardcoded strings.
- Quality gating can reject wind (high energy, low tonality, low centroid) and clipping
  (crest → 1) before inference.
- Fields already in `CallSegment` stop being permanently `None`.

### Negative
- A second spectral transform alongside the mel path. Sharing one STFT would couple the
  two configurations, and they legitimately want different windows; the duplicated FFT
  is the cheaper cost.
- Descriptor values depend on `FeatureConfig`, so cross-corpus comparison requires the
  same config. `FeatureConfig` is therefore recorded in `AcousticFeatures` rather than
  left implicit.

### Risks
- Modulation estimates are unreliable for segments shorter than ~4 modulation periods.
  Below `min_frames_for_modulation` (32), modulation fields are `None` rather than a
  fabricated number.

## Alternatives Considered

**Derive descriptors from the existing mel spectrogram.** Cheaper, and wrong: log-scaled
perceptually-spaced bins do not yield a centroid in Hz, and the 500 Hz mel floor
discards content the descriptors need.

**Use a third-party feature crate (`aubio`, `essentia` bindings).** Both pull C/C++
dependencies, which breaks the WASM target and the project's pure-Rust posture.

**MFCCs instead of individual descriptors.** MFCCs are good model inputs and poor
explanations — no axis of an MFCC vector has a name a biologist would recognize. Perch
already provides the learned representation; this layer exists precisely to be
interpretable.

## References
- ADR-007 (inference pipeline), ADR-010 (streaming ingestion), ADR-012 (visualization)
