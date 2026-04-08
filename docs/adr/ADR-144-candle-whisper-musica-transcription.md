# ADR-144: Candle-Whisper Integration with Musica for Pure-Rust Transcription

## Status
Accepted

## Date
2026-04-06

## Context

Musica performs audio source separation via dynamic mincut graph partitioning, producing clean per-source audio tracks. The natural next step is transcription — converting separated speech to text. Current transcription systems (Whisper, Deepgram) suffer significant accuracy degradation with overlapping speakers and background noise:

- **Clean speech**: ~5% WER (Word Error Rate)
- **2 overlapping speakers**: ~25-35% WER
- **Cocktail party (4+ speakers + noise)**: ~40-60% WER

By separating sources first with Musica, then transcribing each clean track independently, we can maintain near-clean-speech accuracy even in challenging scenarios.

### Why candle-whisper over whisper-rs?

| Criterion | candle-whisper | whisper-rs |
|-----------|---------------|------------|
| **Language** | Pure Rust | C++ FFI bindings |
| **Build** | `cargo build` only | Needs C++ compiler + cmake |
| **Dependencies** | candle-core/nn/transformers | whisper.cpp (compiled from source) |
| **Cross-compile** | Easy (pure Rust) | Hard (C++ toolchain per target) |
| **WASM potential** | Possible via candle WASM | Not feasible (C++ FFI) |
| **Inference speed** | 1.5-3x slower on CPU | Fastest (GGML optimized) |
| **GPU support** | CUDA + Metal via features | CUDA + Metal + CoreML |
| **Alignment** | Matches Musica's zero-C-dep philosophy | External C++ dependency |

**Decision**: Use candle-whisper for architectural purity. The speed penalty is acceptable because:
1. Musica's separation is the bottleneck, not transcription
2. The `tiny` model (39M params) runs 5-10x real-time even via candle on CPU
3. Pure Rust enables WASM deployment for browser-based transcription
4. No cmake/C++ build complexity

## Decision

Integrate candle-whisper as an optional feature (`transcribe`) in Musica, providing:

1. **TranscriberConfig** — model size, language, task (transcribe/translate), beam size
2. **Transcriber** — loads Whisper model via candle, accepts `&[f32]` PCM at 16kHz
3. **TranscriptionResult** — segments with text, timestamps, confidence
4. **Pipeline integration** — `separate_and_transcribe()` combining Musica + Whisper
5. **Before/after benchmark** — measures SNR improvement and simulated WER reduction

### Architecture

```
                    ┌─────────────────┐
Raw Mixed Audio ──> │  Musica Separator │
                    │  (graph mincut)   │
                    └──┬──┬──┬──┬──────┘
                       │  │  │  │
              Speaker1 │  │  │  │ Noise
              Speaker2 │  │  │ (discard)
              Speaker3 │  │
                       ▼  ▼  ▼
                    ┌─────────────────┐
                    │ candle-whisper   │
                    │ (per-track)      │
                    └──┬──┬──┬────────┘
                       │  │  │
                       ▼  ▼  ▼
              Transcript per speaker
              with timestamps + confidence
```

### Audio Format Flow

```
Musica output: Vec<f64> (any sample rate)
    → resample to 16kHz if needed
    → cast f64 → f32
    → pad/trim to 30-second chunks
    → feed to Whisper encoder
    → decode tokens → text segments
```

### Feature Flag Design

```toml
[features]
transcribe = ["candle-core", "candle-nn", "candle-transformers"]
```

When `transcribe` is disabled, the module compiles as a stub with the same public API but returns a "candle not available" error. This keeps the base Musica build lightweight.

## Consequences

### Positive
- Pure Rust end-to-end: capture → separate → transcribe → index
- No C/C++ toolchain required
- WASM-deployable transcription pipeline
- Dramatically improved transcription accuracy via pre-separation
- Optional dependency — doesn't bloat base build

### Negative
- candle inference ~1.5-3x slower than whisper.cpp on CPU
- Model weights must be downloaded at runtime (~75MB for tiny, ~500MB for base)
- candle ecosystem less mature than PyTorch/whisper.cpp
- Large dependency tree when enabled (~50 crates)

### Mitigations
- Default to `tiny` model for real-time use cases
- Cache model weights locally after first download
- GPU acceleration via `cuda`/`metal` feature flags when available
- Benchmark to validate acceptable latency

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| WER (clean, tiny model) | <8% | Baseline Whisper tiny accuracy |
| WER (separated track) | <12% | After Musica separation |
| WER (raw mixed, no separation) | >30% | Demonstrates improvement |
| Inference RTF (tiny, CPU) | <0.2x | 5x faster than real-time |
| Separation + transcription latency | <5s per 30s audio | End-to-end |

## References

- [candle](https://github.com/huggingface/candle) — HuggingFace's minimalist Rust ML framework
- [candle-whisper example](https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper)
- [OpenAI Whisper](https://github.com/openai/whisper) — Original model
- ADR-143: HEARmusica Tympan Rust Port
