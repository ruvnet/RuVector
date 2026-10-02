# ADR-0005: Streaming MP4 + PNG Output Pipeline

- **Status**: Accepted
- **Date**: 2026-06-27

## Context

The user requires the demo to emit **streaming video output** as MP4 and to save
individual frames as images (PNG) — both the input frames and the rendered
reconstruction.

## Decision

`streamcloud-io` owns all encoding:

- **PNG**: per-frame export via the `image` crate (`RgbaImage::save`). Used for
  keyframe snapshots and the final still.
- **MP4**: an H.264 stream muxed incrementally as frames are rendered, so output
  begins before the full sequence is processed (true streaming, matching the
  model's ~20 FPS streaming nature).
  - Primary path: `ffmpeg` via a pipe (`-f rawvideo -pix_fmt rgba` → libx264),
    detected at runtime.
  - Fallback path: pure-Rust muxing when `ffmpeg` is unavailable, so the demo
    never hard-depends on a system binary in CI.
- A `FrameSink` trait abstracts "where rendered frames go" (PNG dir, MP4 stream,
  or in-memory for tests), so the pipeline is encoder-agnostic.

## Rationale

- The `image` crate is pure-Rust and trivial for PNG.
- Piping raw frames to ffmpeg is the simplest robust path to broadly-compatible
  H.264 MP4; the pure-Rust fallback keeps CI hermetic.
- A `FrameSink` trait keeps `streamcloud-pipeline` independent of the sink and makes
  the encoders unit-testable (write a 2-frame clip to a temp dir).

## Consequences

- ffmpeg, when used, must be on `PATH`; absence triggers the fallback with a log
  warning and (if needed) reduced codec options.
- MP4 timestamps are derived from the configured FPS; variable-rate capture is
  out of scope for v0.1.
- Tests assert a non-empty, well-formed PNG and a non-empty MP4 are produced;
  they do not assert visual fidelity.
