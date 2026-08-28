# ADR-340: Native Apple Edge Runtime for RuVector and RuVLLM Consumers

**Status:** Accepted

**Date:** 2026-08-28
**Owners:** RuVector and RuVLLM maintainers

## Context

RuView needs a reusable iPhone runtime for small sensor-fusion models, local
vector search, and governed on-device training. Other RuVector applications
need the same primitives without inheriting RuView's RF schema, room state,
privacy consent, model-promotion policy, Expo bridge, or UI.

The existing Apple surfaces do not provide that package boundary:

- `crates/ruvllm` is a server/workstation runtime. Its default feature set
  includes asynchronous serving, Candle, downloads, routing metrics and
  quantization. It also forces storage, HNSW, parallel and SIMD features on its
  `ruvector-core` dependency.
- `crates/ruvllm/src/kernels/accelerate.rs`, the native Metal module and the
  implemented portions of the Core ML backend are macOS-gated. Meanwhile,
  `autodetect.rs` reports Metal for iOS. That is a capability-reporting
  mismatch, not an iPhone runtime.
- `crates/ruvllm/src/backends/coreml_backend.rs` does not implement GGUF to
  Core ML conversion and does not provide a stateful mobile autoregressive
  session. `hybrid_pipeline.rs` still contains unimplemented generation,
  streaming and embedding paths.
- `npm/packages/ruvllm` uses Node-specific process, filesystem, module and
  native-addon APIs. It is not a React Native package.
- `examples/wasm/ios` is an incomplete example with placeholder allocation,
  persistence and learner bindings. Loading bytes there does not establish a
  working inference runtime.
- `crates/ruvector-cnn` constructs the inspected simplified embedder with
  generated weights. It must not be described as a pretrained camera detector.

The useful, already proven implementation direction is Apple-native:
MPSGraph/Metal for bounded training, Accelerate for small dense inference, Core
ML for application-supplied compiled models, and Rust/AArch64 for a portable
vector kernel. RuView's existing pose-student trainer demonstrates that shape,
but it combines generic math with application-specific pose and RF governance
in one large native module.

## Decision

Create two upstream packages and keep their authority deliberately narrow.

### 1. `RuVectorAppleML` Swift package

The root `Package.swift` exports a source library for iOS 16+ and macOS 13+.
`RuVectorAppleML.podspec` exposes the same reviewed sources to CocoaPods
consumers after a tagged release.

The package owns:

- validated temporal model shapes and tensors;
- deterministic full-window temporal projection training through MPSGraph on
  an explicit Metal command queue;
- low-allocation, lock-safe Accelerate inference;
- thermal, Low Power Mode and foreground/background compute policy;
- actor-owned Core ML sessions with explicit requested compute-unit policy and
  bounded feature types and counts;
- bounded model manifests and point-in-time streaming SHA-256 and Ed25519
  verification of regular-file distribution artifacts; and
- privacy-manifest, test and benchmark artifacts.

The package does not own dataset meaning, labels, consent, quality thresholds,
promotion, retention, networking, downloads, or application state.

### 2. `ruvector-apple-core` Rust package

The Rust package is a memory-only exact vector index with `rlib` and
`staticlib` outputs. Its versioned C ABI uses opaque handles, fixed-width
fields, explicit result/buffer destruction, bounded dimensions, finite-value
validation, corruption-resistant snapshots and panic containment. No Rust
layout type crosses the ABI.

Cosine, squared-L2 and dot-product search are the initial algorithms. HNSW,
quantization and persistent storage remain separate opt-in candidates until
numerical parity, memory and physical-device evidence exist.

An XCFramework and actor-isolated Swift `VectorStore` are the release packaging
target. The source static library and ABI are delivered first so the ABI can be
fuzzed and frozen before binary distribution.

### 3. RuVLLM mobile boundary

Do not embed the current `ruvllm` crate or `@ruvector/ruvllm` package in iOS.
RuVLLM models may be exported into an application-reviewed Core ML asset and
run by `CoreMLModelSession`; that is model execution, not a port of RuVLLM's
server, download, tokenizer, KV-cache or training stack. The session does not
authenticate a compiled model directory. The consuming application must bind
distribution-artifact verification, protected staging, transformation, Core
ML compilation and activation in one reviewed workflow.

Future text generation requires a separate accepted ADR and must provide:

- a signed model/tokenizer/transform manifest;
- actor-owned stateful KV cache using supported Apple APIs;
- explicit context, memory and output bounds;
- cancellation, memory-warning recovery and atomic rollback; and
- quality, latency, energy and thermal evidence from physical devices.

Until those gates pass, the package truthfully supports small temporal models
and application-supplied Core ML inference, not a complete on-device LLM.

## Optimization Contract

1. Accelerate inference stores reusable normalization, hidden and output
   buffers. Access is lock-safe; callers can share one predictor without
   racing its scratch space.
2. Weight tensors are row-major and passed directly to `cblas_sgemv`; no
   per-call model conversion or bridge through JavaScript occurs.
3. MPSGraph receives one bounded batch and reuses immutable input, target and
   mask data across epochs. Per-epoch result objects are drained in an
   autorelease pool.
4. Training checks cancellation and elapsed time at every MPSGraph epoch
   boundary, and thermal policy every other epoch. Cancellation and maximum
   duration are cooperative: an in-flight graph invocation is not preempted,
   and an over-time result is rejected when the epoch returns. Serious/critical
   thermal state and Low Power Mode reject training rather than silently
   degrading it.
5. Interactive Core ML requests `.all`, allowing the operating system to
   select or change supported compute devices. Background/efficiency work
   requests `.cpuOnly` to avoid GPU contention. The requested policy is
   explicit; Apple's actual CPU, GPU or Neural Engine choice is not exposed by
   this package and is never inferred from a marketing device name.
6. Model and batch dimensions have fixed upper bounds and overflow checks.
   NaN and infinity are rejected before native kernels execute.

No fixed throughput, ANE TOPS, speedup or iPhone 17 Pro advantage is part of
this decision. Those are measurements, not architecture.

## RuView Integration

RuView replaces only the generic training, prediction and runtime-policy
portions of its native pose-student module. It retains:

- the existing protected artifact and consumed-holdout-ledger paths;
- RF feature, room, calibration and joint schemas;
- explicit opt-in and physical-device-only capture;
- context-digest binding and source provenance;
- leakage-free train/holdout separation;
- the mean-pose and RF baselines;
- the 25 percent PCK-at-20-centimetres selection gate, abstention and rollback;
  and
- honest limitations: coarse pose is not DensePose, identity, or proof of
  physical vision through walls.

The Expo layer receives throttled summaries. Camera, LiDAR, Metal buffers and
full-rate sensor windows stay native.

## Security and Privacy Gates

- No networking, telemetry, background upload, model catalog, URL fetch or
  credential storage exists in either package.
- The artifact verifier opens the final path component without following a
  symbolic link and authenticates the regular-file bytes read at verification
  time against signed byte count and SHA-256 metadata.
- A verification receipt is not an activation token, does not pin a mutable
  path, does not verify an `.mlmodelc` directory, and cannot establish
  provenance for an independently loaded Core ML URL. Applications own the
  protected staging, compilation and atomic activation transaction.
- A digest without a trusted signature is integrity evidence only, not model
  provenance.
- C callers must pass a live, matching opaque index handle and must not destroy
  it concurrently with another call. Within that standard lifetime contract,
  the ABI rejects null pointers, invalid lengths and dimensions, arithmetic
  overflow and non-finite values. Rust panics cannot cross FFI.
- Returned C allocations have one documented owner and matching free function.
- Snapshots are versioned and rejected on truncation, corruption, unknown
  metric or trailing data.
- Applications supply a protected persistence URL and own retention/deletion.
  The packages never persist raw sensor samples.
- Learning candidates cannot self-promote. Applications retain frozen
  baselines, held-out evaluation, explicit approval and rollback.

## Validation and Release Gates

### Automated on every change

- `swift test` including tensor-shape, non-finite, decoding, concurrency,
  cancellation, MPSGraph execution, bounded Core ML features, signature,
  symbolic-link and tamper tests;
- SwiftPM builds for generic arm64 iPhoneOS and arm64/x86_64 iOS Simulator;
- Rust unit tests, formatting and Clippy with warnings denied;
- Rust builds for `aarch64-apple-ios` and `aarch64-apple-ios-sim`;
- scalar-versus-index numerical parity, snapshot corruption and C ABI misuse
  tests; and
- secret scan, privacy-manifest presence and public-symbol review.

### Required before a binary release or replacement of an existing path

- generated C header ABI diff and XCFramework device/simulator slice audit;
- deterministic consumer parity for scores and result ordering;
- interrupted activation and rollback testing;
- a 10 to 30 minute camera/LiDAR/RF concurrency burn-in;
- foreground/background, cancellation and memory-warning recovery; and
- a physical iPhone matrix recording hardware, OS, model digest, p50/p95/p99,
  peak/resident memory, energy and thermal state.

Task-specific learning additionally requires a leakage-free held-out dataset.
Pose evaluation includes PCK and a mean-pose baseline. A simulator build or Mac
benchmark is never reported as iPhone validation.

Publication is authorized only through
`.github/workflows/apple-edge-release.yml`. That workflow binds a full main
commit SHA and matching version, seals and attests the source, crate, device and
simulator archives, requires protected-environment approval plus a reviewed
physical-iPhone evidence digest, and verifies the public GitHub, crates.io and
CocoaPods bytes. Workstation publication is outside this decision.

## Initial Evidence

At acceptance, the source package has automated tests for validation,
inference, concurrent access, MPSGraph training, cancellation and signed-model
tampering. It builds against the generic iPhoneOS and iOS Simulator SDKs.

The package benchmark emits `MEASURED_ON_CURRENT_RUNTIME` with the platform and
model shape. Its first result is a Mac runtime measurement and therefore says
nothing about iPhone 17 Pro latency, energy or thermal behavior. Physical
device benchmarking remains a release gate.

## Consequences

### Positive

- RuView and other RuVector applications share optimized, reviewed Apple math
  without sharing domain authority.
- The package can be consumed with SwiftPM or CocoaPods.
- Requested compute policy is explicit while actual Core ML device selection
  remains correctly described as operating-system controlled.
- Small sensor models can train and infer without copying full-rate data through
  React Native or a WebView.
- The Rust ABI creates a stable path to NEON-backed vector search and an
  XCFramework without importing the server runtime.

### Negative

- A new Swift and Rust compatibility surface requires ABI and artifact-release
  discipline.
- Source SwiftPM distribution does not automatically include the Rust static
  library; the XCFramework is a separate release milestone.
- RuView temporarily carries an adapter while preserving its prior artifact
  format and holdout ledger.
- Full RuVLLM text generation, HNSW and quantization are explicitly deferred.

## Alternatives Considered

**Embed `@ruvector/ruvllm` in React Native.** Rejected. It is Node-specific and
silently falls back on unsupported platforms.

**Cross-compile the full default `ruvllm` crate.** Rejected. It pulls server,
download and desktop backend concerns, while important iOS paths remain
unimplemented or macOS-gated.

**Use the existing iOS WASM example.** Rejected. Its essential runtime and
persistence bindings are placeholders, and it adds a bridge on the hottest
sensor path.

**Keep the implementation inside RuView.** Rejected. It duplicates generic
Apple execution code across applications and makes domain policy difficult to
review separately from numerical kernels.

**Train a full language or vision foundation model on-device.** Rejected for
this package. Bounded task heads and adapters fit the resource and governance
contract; full-model training does not.

## Primary References

- Apple Core ML compute units:
  <https://developer.apple.com/documentation/coreml/mlcomputeunits>
- Apple MPSGraph:
  <https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph>
- Apple MPSGraph training sample:
  <https://developer.apple.com/documentation/metalperformanceshadersgraph/training-a-neural-network-using-mps-graph>
- Apple Accelerate:
  <https://developer.apple.com/documentation/accelerate>
- Apple XCFramework distribution:
  <https://developer.apple.com/documentation/xcode/creating-a-multi-platform-binary-framework-bundle>
