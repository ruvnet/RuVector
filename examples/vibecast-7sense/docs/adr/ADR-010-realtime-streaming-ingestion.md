# ADR-010: Real-Time Streaming Audio Ingestion

## Status
Accepted

## Date
2026-08-03

## Updated
2026-08-08

Implementation correction: ring slots now use publication stamps around each
sample, so a consumer detects and accounts for an overwrite that races the copy
instead of silently accepting a value from a newer generation.

## Context

ADR-001 through ADR-009 describe an offline, file-oriented pipeline: a recording is
uploaded, decoded in full, segmented, embedded, indexed, and analyzed as a batch job.
The implementation matches that description. `AudioIngestionService::ingest_file`
takes a `&Path`, and `SymphoniaFileReader` decodes every packet into a single
`Vec<f32>` before any downstream stage runs. Nothing in the workspace opens an audio
input device.

Two documents nonetheless promise live operation:

- `crates/sevensense-audio/README.md` advertises `AudioStream::new()` for "real-time
  microphone/line-in capture". No such type exists.
- `crates/sevensense-api/README.md` and the root `README.md` advertise
  `ws://…/ws/stream` carrying binary audio chunks and returning `detection` messages
  with species and confidence. No such route is registered, and the WebSocket receive
  task discards every inbound frame.

The gap is architectural, not cosmetic. Whole-buffer processing is incompatible with
live audio in three specific ways:

1. **Decode** materializes the entire signal before returning.
2. **Segmentation** (`EnergySegmenter`) computes an adaptive noise floor from global
   statistics over the complete sample buffer, so it cannot emit a segment until the
   recording ends.
3. **Resampling** (`rubato`) is constructed per-call rather than held across chunks,
   so filter state is discarded at every boundary.

We need a design that supports continuous input without forking the codebase into
"live" and "offline" halves that drift apart.

## Decision

We introduce a streaming ingestion path built on a bounded ring buffer and an
incremental segmenter, and we make the existing offline path a thin adapter over it.
Offline processing becomes "streaming, with a file as the source" rather than a
parallel implementation.

### 1. Capture abstraction

Device access sits behind a trait so that the core crate stays testable without
hardware and so WASM builds can supply a Web Audio source:

```rust
pub trait AudioSource: Send {
    /// Sample rate of the samples this source yields.
    fn sample_rate(&self) -> u32;
    /// Number of interleaved channels.
    fn channels(&self) -> u16;
    /// Fill `out` with the next available samples; returns count written.
    /// Returns `Ok(0)` when the source is exhausted (files) or idle (devices).
    fn read(&mut self, out: &mut [f32]) -> Result<usize, AudioError>;
}
```

Three implementations: `DeviceSource` (cpal, native only, behind the `capture`
feature), `FileSource` (symphonia, incremental — it already decodes packet by packet
and simply stops buffering everything), and `MemorySource` (tests and benchmarks).

`cpal` is an optional dependency gated behind a non-default `capture` feature. The
core crate must continue to build on targets without an audio backend, including
`wasm32-unknown-unknown`.

### 2. Ring buffer and back-pressure

The capture callback runs on a real-time audio thread and must never allocate, lock,
or block. It writes into a fixed-capacity SPSC ring buffer sized to
`ring_seconds × sample_rate` (default 30 s). The analysis thread drains it.

When the consumer falls behind, the producer **overwrites the oldest samples and
increments a dropped-sample counter**. We deliberately choose lossy overwrite over
blocking: stalling the audio callback causes device-level glitches and, in a
monitoring deployment, stale audio is worth less than current audio. The counter is
surfaced in `StreamStats` so callers can detect and report underruns rather than
silently trusting a degraded stream.

### 3. Incremental segmentation

`StreamSegmenter` replaces global statistics with an exponential moving average noise
floor:

```
noise_floor ← α · noise_floor + (1 − α) · frame_rms      when frame is below threshold
```

with α derived from a configurable time constant (default 2 s). A segment opens when
frame RMS exceeds `noise_floor × open_ratio` for `min_open_frames` consecutive frames,
and closes after `hangover_ms` below `noise_floor × close_ratio`. Distinct open and
close ratios give hysteresis, which prevents a segment from flickering at the
threshold.

Two bounds keep latency and memory predictable: a segment is force-closed at
`max_segment_ms` (default 10 s), and segments shorter than `min_segment_ms`
(default 120 ms) are discarded as transients.

The offline `EnergySegmenter` is retained. Its global noise floor is genuinely better
when the whole signal is available, and changing it would alter existing test
expectations for no benefit.

### 4. Overlapping analysis windows

Perch 2.0 expects 5-second windows at 32 kHz. A detected segment is emitted as one or
more analysis windows with 50% overlap (`hop = 2.5 s`), so a vocalization straddling a
window boundary still lands near the center of some window. Windows shorter than 5 s
are zero-padded; longer segments yield multiple windows.

### 5. Stateful resampling

`StreamResampler` holds one `rubato` instance for the lifetime of the stream and
processes fixed-size blocks, preserving filter state across chunk boundaries. The
per-call construction in the offline path is a latent quality bug — it produces a
discontinuity at every call — and the streaming path must not reproduce it.

### 6. Pipeline topology

```
AudioSource ──► RingBuffer ──► StreamResampler ──► StreamSegmenter ──► WindowEmitter
   (rt thread)     (lock-free)     (stateful)         (EMA + hysteresis)     │
                                                                             ▼
                                                        FeatureExtractor + Embedder
                                                                             │
                                                                             ▼
                                                              HNSW k-NN ──► StreamEvent
```

Each stage is a `tokio` task connected by bounded `mpsc` channels. Bounded channels
mean back-pressure propagates to the ring buffer, where the drop policy is explicit,
rather than causing unbounded memory growth somewhere in the middle.

### 7. Transport

A new `/ws/stream` route accepts binary frames of little-endian `f32` PCM, preceded by
a JSON `StreamConfig { sample_rate, channels, format }` handshake. The server replies
with `StreamEvent` messages:

| Event | Payload |
|-------|---------|
| `ready` | negotiated config, session id |
| `frame` | per-frame acoustic features for live visualization (see ADR-011) |
| `segment` | segment opened/closed, timing, quality |
| `detection` | k-NN candidates with distances and labels (see ADR-013) |
| `stats` | throughput, dropped samples, queue depth |
| `error` | code and message |

`frame` events are sent at the feature hop rate but coalesced into batches of up to 16
frames to bound WebSocket message overhead. This is the data feeding the live
visualization; sending one message per frame at ~100 Hz per client is wasteful.

## Consequences

### Positive
- Live capture, file replay, and tests share one pipeline, so they cannot drift.
- Bounded memory regardless of stream duration.
- Latency is dominated by the analysis window, not by file length.
- The `AudioSource` trait keeps `cpal` out of the default build, preserving WASM support.

### Negative
- The EMA noise floor is less accurate than global statistics on short clips, so
  offline results from the two segmenters will not be bit-identical. We keep both
  rather than pretending one supersedes the other.
- Lossy overwrite means a slow consumer loses audio. This is a deliberate trade and
  must be visible in `StreamStats`, not silent.
- Real-time correctness (no allocation in the capture callback) is a property tests
  can only approximate; it needs review discipline.

### Risks
- `cpal` device enumeration differs across platforms and CI has no audio device. All
  device-dependent tests are gated behind the `capture` feature and excluded from
  default CI runs; the pipeline itself is tested with `MemorySource`.

## Alternatives Considered

**Block on a full ring buffer.** Rejected: blocking the real-time audio callback
causes device glitches and, on some backends, disconnects.

**Fixed-size tumbling windows with no segmentation.** Simpler, and it is what many
demo systems do, but it embeds silence and splits calls arbitrarily, degrading both
embedding quality and the visualization.

**Reuse `EnergySegmenter` over a sliding buffer.** Recomputing global statistics per
chunk is O(n) per chunk and still yields a threshold that shifts discontinuously as
the window slides.

## References
- ADR-001 (system architecture), ADR-007 (inference pipeline)
- ADR-011 (acoustic features), ADR-013 (retrieval-based identification)
