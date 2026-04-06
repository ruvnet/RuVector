# ADR-143: HEARmusica — High-Fidelity Rust Port of Tympan Open-Source Hearing Aid

## Status
Accepted

## Date
2026-04-06

## Context

Tympan is an MIT-licensed open-source hearing aid platform built on Arduino/Teensy (ARM Cortex-M7, 600 MHz). Its `AudioStream_F32` abstraction provides a block-graph processing pipeline with ~20 DSP algorithms including WDRC compression, feedback cancellation, and biquad filtering.

The musica project already implements graph-based audio separation (Fiedler vector + dynamic mincut) with sub-millisecond latency. Combining Tympan's proven hearing aid DSP chain with musica's novel separation engine creates a system no commercial hearing aid can match: explainable, graph-based source separation integrated into a complete hearing aid pipeline.

### Why Rust?

| Concern | Tympan (C++) | HEARmusica (Rust) |
|---------|-------------|-------------------|
| Memory safety | Manual (buffer overruns possible) | Compile-time guaranteed |
| Concurrency | Interrupt-based (race conditions possible) | Ownership model prevents data races |
| Targets | Teensy only | Embedded (`no_std`), WASM, desktop, cloud |
| Regulatory | Hard to formally verify | Ownership + type system aids certification |
| Performance | Good (ARM CMSIS-DSP) | Equal or better (LLVM auto-vectorization) |

### Why Not Fork OpenMHA?

OpenMHA has 80+ plugins and NAL-NL2 fitting — far richer algorithm library. However:
- **AGPL v3 license** — any derivative must be open-sourced, killing commercial products
- **Complex architecture** — AC variables, template plugins, JACK dependency fight Rust's ownership model
- **200K+ LOC** — porting is impractical; clean-room reimplementation required for any algorithm

Tympan's MIT license and simple `update()` pattern make it the right porting target.

## Decision

Create **HEARmusica** as a Rust hearing aid DSP framework within the musica example crate, porting Tympan's core blocks with a Rust-idiomatic `AudioProcessor` trait and integrating musica's graph-based separation as a first-class processing block.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HEARmusica Pipeline                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input (L/R mic)                                        │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────┐   ┌───────────┐   ┌─────────��────────┐   │
│  │ Biquad   │──▶│ Feedback  │──▶│ Graph Separator  │   │
│  │ Prefilter│   │ Canceller │   │ (Fiedler+MinCut) │   │
│  └──────────┘   └───────────┘   └────────┬─────────┘   │
│                                          │              │
│                            ┌─────────────┼──────────┐   │
│                            ▼             ▼          │   │
│                       [speech]      [noise]         │   │
│                            │                        │   │
│                            ▼                        │   │
│                  ┌──────────────────┐               │   │
│                  │ Multi-Band WDRC  │               │   │
│                  │ Compressor       │               │   │
│                  └────────┬──────���──┘               │   │
│                           │                         │   │
│                           ▼                         │   │
│                  ┌─���────────────────┐               │   │
│                  │ Audiogram Gain   │               │   │
│                  │ (NAL-R/half-gain)│               │   │
│                  └────────┬───���─────┘               │   │
│                           │                         │   │
│                           ▼                         │   │
│                  ┌──���───────────────┐               ���   │
│                  │ Limiter/Output   │               │   │
│                  └──��─────┬─────────┘               │   │
│                           │                         │   │
│                           ▼                         │   │
│                     Output (L/R)                    │   │
│                                                     │   │
└───────��─────────────────────────────────────────────┘
```

### Core Trait (Tympan `AudioStream_F32` → Rust)

```rust
/// Audio processing block — the fundamental unit of HEARmusica.
/// Maps to Tympan's AudioStream_F32 with Rust ownership semantics.
pub trait AudioProcessor: Send {
    /// Configure for given sample rate and block size.
    /// Called once before processing starts (maps to OpenMHA's prepare()).
    fn prepare(&mut self, sample_rate: f32, block_size: usize);

    /// Process one block of audio in-place.
    /// MUST be real-time safe: no allocation, no locks, no syscalls.
    fn process(&mut self, block: &mut AudioBlock);

    /// Release resources (maps to OpenMHA's release()).
    fn release(&mut self) {}

    /// Human-readable name for debugging and replay logging.
    fn name(&self) -> &str;

    /// Current latency contribution in samples.
    fn latency_samples(&self) -> usize { 0 }
}
```

### AudioBlock

```rust
/// Stereo audio block — the data unit passed between processors.
pub struct AudioBlock {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub sample_rate: f32,
    pub block_size: usize,
    pub metadata: BlockMetadata,
}

pub struct BlockMetadata {
    pub frame_index: u64,
    pub timestamp_us: u64,
    pub speech_mask: Option<Vec<f32>>,    // Set by separator
    pub noise_estimate: Option<Vec<f32>>, // Set by noise estimator
}
```

### Processing Blocks (Tympan Port)

| Block | Tympan Source | Rust Module | Key Algorithm |
|-------|-------------|-------------|---------------|
| `BiquadFilter` | `AudioFilterBiquad_F32` | `filter.rs` | IIR biquad (low/high/band/notch/allpass/peaking/shelf) |
| `WDRCompressor` | `AudioEffectCompressor_F32` | `compressor.rs` | Multi-band WDRC with attack/release/ratio/knee |
| `FeedbackCanceller` | `AudioEffectFeedbackCancel_F32` | `feedback.rs` | Normalized LMS adaptive filter |
| `GainProcessor` | `AudioEffectGain_F32` | `gain.rs` | Linear/dB gain + audiogram-shaped frequency response |
| `DelayLine` | `AudioEffectDelay_F32` | `delay.rs` | Sample-accurate circular buffer delay |
| `Mixer` | `AudioMixer_F32` | `mixer.rs` | Weighted sum of N inputs |
| `Limiter` | (custom) | `limiter.rs` | Brick-wall limiter with lookahead |

### Novel Blocks (Musica Integration)

| Block | Module | Key Algorithm |
|-------|--------|---------------|
| `GraphSeparator` | `separator_block.rs` | Fiedler vector + dynamic mincut from musica |
| `BinauralEnhancer` | Uses `hearing_aid.rs` | ILD/IPD/IC features + speech scoring |
| `NeuralRefiner` | Uses `neural_refine.rs` | Tiny MLP mask refinement |

### Pipeline Runner

```rust
pub struct Pipeline {
    blocks: Vec<Box<dyn AudioProcessor>>,
    sample_rate: f32,
    block_size: usize,
}

impl Pipeline {
    pub fn new(sample_rate: f32, block_size: usize) -> Self;
    pub fn add(&mut self, block: Box<dyn AudioProcessor>);
    pub fn prepare(&mut self);
    pub fn process_block(&mut self, block: &mut AudioBlock);
    pub fn total_latency_samples(&self) -> usize;
    pub fn total_latency_ms(&self) -> f32;
}
```

### File Structure

```
docs/examples/musica/src/hearmusica/
├── mod.rs              — Module root, re-exports, Pipeline struct
├── block.rs            — AudioProcessor trait, AudioBlock, BlockMetadata
├── compressor.rs       — Multi-band WDRC compressor
├── feedback.rs         — NLMS adaptive feedback canceller
├── filter.rs           — Biquad IIR filter (all standard types)
├── gain.rs             — Gain processor + audiogram fitting (NAL-R)
├── limiter.rs          — Brick-wall output limiter
├── delay.rs            — Sample-accurate delay line
├── mixer.rs            — Weighted N-input mixer
├── separator_block.rs  — Graph separator as AudioProcessor
├── presets.rs          — Pre-built pipeline configurations
```

### Preset Pipelines

```rust
/// Standard hearing aid: prefilter → feedback cancel → WDRC → audiogram gain → limiter
pub fn standard_hearing_aid(audiogram: &Audiogram) -> Pipeline;

/// Speech-in-noise: prefilter → feedback cancel → graph separator → WDRC (speech only) → gain → limiter
pub fn speech_in_noise(audiogram: &Audiogram) -> Pipeline;

/// Music mode: prefilter → wideband gentle compression → gain → limiter (minimal processing)
pub fn music_mode(audiogram: &Audiogram) -> Pipeline;

/// Maximum clarity: prefilter → feedback cancel → graph separator → neural refine → WDRC → gain → limiter
pub fn maximum_clarity(audiogram: &Audiogram) -> Pipeline;
```

## Performance Targets

| Metric | Target | Tympan Reference |
|--------|--------|-----------------|
| Block latency | < 0.5 ms per block | ~1.3 ms (block 16 @ 24 kHz) |
| Total pipeline latency | < 4 ms | 5.7 ms measured |
| Memory usage | < 64 KB working set | ~50 KB on Teensy |
| Binary size (WASM) | < 200 KB | N/A |
| Sample rates | 8-96 kHz | 8-96 kHz |
| Block sizes | 16-256 samples | 1-128 samples |

## Testing Strategy

1. **Unit tests per block** — Verify frequency response, gain curves, convergence
2. **Pipeline integration tests** — End-to-end with synthetic signals
3. **Latency validation** — Every block stays within budget
4. **Preset validation** — Each preset processes without clipping or artifacts
5. **Comparison test** — Same input through Tympan WDRC params vs HEARmusica, verify SDR > 30 dB

## Consequences

### Positive
- MIT-licensed Rust hearing aid DSP — first of its kind
- Runs everywhere (MCU, WASM, desktop) from single codebase
- Graph-based separation integrated as native pipeline block
- Fully auditable for FDA/CE regulatory compliance
- Sub-millisecond block processing enables ultra-low-latency configurations

### Negative
- Initial algorithm library is smaller than OpenMHA (8 blocks vs 30+ plugins)
- No hardware board (depends on external audio I/O)
- Beamforming requires multi-mic arrays (not in scope for v1)

### Risks
- WDRC parameter tuning requires audiological expertise
- Real-world validation needs clinical testing with hearing-impaired users
- Feedback cancellation convergence depends on acoustic coupling

## References

- Tympan Library: https://github.com/Tympan/Tympan_Library (MIT)
- OpenAudio ArduinoLibrary: https://github.com/chipaudette/OpenAudio_ArduinoLibrary
- ANSI S3.22 Hearing Aid Testing Standard
- NAL-R Prescription Rule (Byrne & Dillon, 1986)
- WDRC: Villchur (1973), compression ratios and kneepoints
- NLMS Adaptive Filtering: Haykin, Adaptive Filter Theory
