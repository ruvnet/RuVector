# Musica — Structure-First Audio Source Separation

Dynamic mincut graph partitioning for audio source separation, hearing aid enhancement, multitrack stem splitting, and crowd-scale speaker identity tracking.

## Core Idea

Traditional audio separation is **frequency-first**: FFT masking, ICA, NMF.

Musica is **structure-first**: reframe audio as a graph partitioning problem.

- **Nodes** = time-frequency atoms (STFT bins, critical bands, or learned embeddings)
- **Edges** = similarity (spectral proximity, phase coherence, harmonic alignment, temporal continuity, spatial cues)
- **Weights** = how strongly two elements "belong together"

Dynamic mincut finds the **minimum boundary** where signals naturally separate, preserving **maximum internal coherence** within each partition.

*What breaks the null is the signal.*

## Architecture

```
Raw Audio
    |
    v
STFT / Filterbank
    |
    v
Graph Construction (spectral + temporal + harmonic + spatial edges)
    |
    v
Laplacian Eigenvectors (Fiedler vector via Lanczos)
    |
    v
Spectral Clustering (balanced initial partition)
    |
    v
Dynamic MinCut Refinement (boundary optimization)
    |
    v
Soft Mask Generation (distance-weighted)
    |
    v
Overlap-Add Reconstruction
```

## Modules

| Module | Purpose | Key Feature |
|--------|---------|-------------|
| `stft` | Time-frequency decomposition | Zero-dep radix-2 FFT + Hann window |
| `lanczos` | Sparse Laplacian eigensolver | SIMD-optimized Lanczos iteration |
| `audio_graph` | Graph construction from STFT | Spectral, temporal, harmonic, phase edges |
| `separator` | Spectral clustering + mincut | Fiedler vector + balanced partitions |
| `hearing_aid` | Binaural streaming enhancer | <8ms latency, audiogram gain shaping |
| `multitrack` | 6-stem music separator | Vocals/bass/drums/guitar/piano/other |
| `crowd` | Distributed identity tracker | Hierarchical sensor fusion at scale |
| `wav` | WAV file I/O | 16/24-bit PCM, mono/stereo |
| `benchmark` | SDR/SIR/SAR evaluation | Comparison against baselines |

## Usage

```bash
# Build
cargo build --release

# Run full benchmark suite
cargo run --release

# Run tests
cargo test
```

## Hearing Aid Mode

Streaming binaural speech enhancement targeting:
- **Latency**: <8ms algorithmic delay
- **Input**: Left + right microphone streams
- **Output**: Enhanced binaural audio preserving spatial cues
- **Features**: 32-64 critical bands, ILD/IPD/IC features, audiogram fitting

```rust
use musica::hearing_aid::{HearingAidConfig, StreamingState};

let config = HearingAidConfig::default();
let mut state = StreamingState::new(&config);

// Process each hop
let result = state.process_frame(&left_samples, &right_samples, &config);
// result.mask, result.speech_score, result.latency_us
```

## Multitrack Mode

6-stem music source separation:
- Vocals, Bass, Drums, Guitar, Piano, Other
- Band-split spectral priors per instrument
- Graph-based coherence refinement
- Wiener-style soft masking with temporal smoothing

```rust
use musica::multitrack::{separate_multitrack, MultitrackConfig};

let config = MultitrackConfig::default();
let result = separate_multitrack(&audio_signal, &config);
for stem in &result.stems {
    println!("{:?}: confidence={:.2}", stem.stem, stem.confidence);
}
```

## Crowd-Scale Mode

Distributed speaker identity tracking across thousands of speakers:
- Hierarchical: local events → local speakers → regional association → global identity
- Handles reappearance, merging, and identity persistence
- Scales via hypothesis compression, not raw waveform processing

## Benchmark Targets

| Category | Metric | Baseline | Target |
|----------|--------|----------|--------|
| Two-tone separation | SDR | 0 dB | >6 dB |
| Hearing aid latency | Algorithmic delay | N/A | <8 ms |
| Multitrack vocals | SDR | 5-7 dB | 6-9 dB |
| Crowd tracking | Identities maintained | N/A | 100-300 |

## Why This Beats Traditional Methods

| Method | Weakness | Musica Advantage |
|--------|----------|-----------------|
| FFT masking | Struggles with spectral overlap | Cuts by structure, not amplitude |
| ICA | Needs multiple channels | Works single-channel |
| Deep learning | Brittle, hallucination, opaque | Deterministic + explainable |
| NMF | Slow, approximate | Real-time incremental |

## Stack Integration

- **RuVector** → embedding + similarity graph
- **Dynamic MinCut** → partition engine
- **Lanczos** → spectral structural analysis
- **RVF** → temporal partitions + witness logs

## References

- Stoer-Wagner minimum cut algorithm
- Spectral clustering via graph Laplacian (Shi & Malik, 2000)
- BS-RoFormer (Sound Demixing Challenge 2023)
- MUSDB18 benchmark dataset
- Pseudo-deterministic canonical minimum cut (Kenneth-Mordoch, 2026)
