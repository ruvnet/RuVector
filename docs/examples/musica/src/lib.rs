//! # Musica — Structure-First Audio Source Separation
//!
//! Audio source separation via dynamic mincut graph partitioning.
//!
//! Instead of frequency-first separation (FFT masking, ICA, NMF), this approach
//! reframes audio as a **graph partitioning problem**:
//!
//! - **Nodes** = time-frequency atoms (STFT bins, critical bands)
//! - **Edges** = similarity (spectral, phase, harmonic, temporal, spatial)
//! - **Weights** = how strongly two elements "belong together"
//!
//! Dynamic mincut finds the **minimum boundary** where signals naturally separate,
//! preserving **maximum internal coherence** within each partition.
//!
//! ## Modules
//!
//! - `stft` — STFT/ISTFT with radix-2 FFT (zero dependencies)
//! - `lanczos` — SIMD-optimized sparse Lanczos eigensolver
//! - `audio_graph` — Weighted graph construction from STFT
//! - `separator` — Spectral clustering + mincut partitioning
//! - `hearing_aid` — Binaural streaming enhancer (<8ms latency)
//! - `multitrack` — 6-stem music separator (vocals/bass/drums/guitar/piano/other)
//! - `crowd` — Distributed speaker identity tracker (thousands of speakers)
//! - `wav` — WAV file I/O (16/24-bit PCM)
//! - `benchmark` — SDR/SIR/SAR evaluation

pub mod enhanced_separator;
pub mod hearmusica;
pub mod adaptive;
pub mod audio_graph;
pub mod benchmark;
pub mod crowd;
pub mod hearing_aid;
pub mod lanczos;
pub mod multi_res;
pub mod multitrack;
pub mod neural_refine;
pub mod phase;
pub mod separator;
pub mod stft;
pub mod streaming_multi;
pub mod evaluation;
pub mod wasm_bridge;
pub mod wav;
