# RuVectorAppleML

`RuVectorAppleML` is a semantics-neutral Swift package for bounded on-device
training and inference in RuVector applications. It is designed for iOS 16+
and macOS 13+ and has no networking, analytics, model-download, UI, camera,
LiDAR, RF, pose, identity, or application-state authority.

The package provides:

- full-window temporal projection training through MPSGraph and Metal;
- low-allocation, lock-safe inference through Accelerate;
- explicit thermal and Low Power Mode decisions;
- bounded, actor-isolated hardware-adaptive route selection with runtime-scoped EWMA
  latency and dimensionless relative-energy-proxy learning;
- Core ML sessions with an explicit requested compute-unit policy and bounded
  feature types and counts; and
- point-in-time verification of size- and digest-bound, Ed25519-signed
  regular-file distribution artifacts.

Applications own consent, feature semantics, dataset splits, selection gates,
artifact retention, model promotion and rollback. In particular, this package
does not make a model accurate, does not prove physical sensing through walls,
and is not a complete mobile port of RuVLLM.

## Adaptive execution

`AdaptiveExecutionPlanner` chooses among caller-supplied implementations that
must already be numerically compatible. It begins with the most conservative
eligible caller-supplied route, learns bounded hardware/runtime-scoped EWMA
costs from explicit observations, and can sample alternatives with deterministic
exploration under nominal foreground conditions. Thermal state, Low Power Mode,
background execution, failure cooldowns, and route hysteresis are part of every
decision.

Planner construction requires a bounded caller-owned
`AdaptiveOptimizationContextRevision`. Applications must change it when their
measurement method, calibration, room policy, or scheduling policy changes.
Candidate implementation revisions separately identify exact kernels or model
artifacts. Exploration, cooldown, and energy-evidence age use workload-local
operation clocks, so unrelated workloads do not change another workload's
learned policy.

Multiple decisions for one workload/candidate may remain in flight and finish
out of order. Outstanding receipts are bounded by `maximumProfiles`; selection
refuses explicitly at that bound and never evicts a still-valid receipt.

The optional `AdaptiveRelativeEnergyProxy` is dimensionless and meaningful
only within one application's measurement method; it is never joules. Planner
snapshots contain cost summaries, not sensor samples, and restore only when the
non-unique hardware/runtime fingerprint, optimization-context revision, and
configuration match exactly. Snapshot schema v3 invalidates v2 learned state;
the application owns persistence.

Core ML candidates declare requested allowed compute units. Apple controls the
actual CPU, GPU, or Neural Engine placement, which remains opaque to this
package. Selecting a `.cpuAndNeuralEngine` or `.all` candidate is not evidence
that the Neural Engine ran the workload.

## Core ML and artifact provenance

`CoreMLModelSession` loads a caller-supplied compiled model. Interactive work
requests Core ML's `.all` compute-unit policy; Apple may choose or change the
actual CPU, GPU, or Neural Engine execution path, and that device identity is
not exposed or reported by this package. Efficiency work requests `.cpuOnly`.

Inputs are limited to bounded scalar, string, multi-array, and image values.
Dictionary, sequence, state, undefined, and unknown feature types are rejected.
Models and predictions with more than 256 outputs are rejected rather than
partially returned.

`ModelArtifactVerifier.verifyRegularFileDistributionArtifact` authenticates
the bytes read from one local, non-symbolic regular file at that point in time.
Its receipt contains no URL and is not an activation token. It does not verify
an `.mlmodelc` directory or bind an independently supplied model URL to the
signed file. A consuming application must own a protected, atomic workflow for
staging, transformation, Core ML compilation, activation, and rollback.

Training cancellation and maximum-duration enforcement are cooperative at
MPSGraph epoch boundaries. An in-flight graph invocation is not preempted; the
trainer rejects an over-time result when that epoch returns.

## Swift Package Manager

```swift
.package(url: "https://github.com/ruvnet/ruvector.git", exact: "2.3.0")
```

Add the `RuVectorAppleML` product to the consuming target. The source package
can also be consumed through `RuVectorAppleML.podspec` once a reviewed tag is
published.

## Example

```swift
let shape = try TemporalModelShape(
    windowLength: 8,
    inputWidth: 30,
    hiddenWidth: 16,
    outputWidth: 45
)
let options = try TemporalTrainingOptions()
let report = try MPSGraphTemporalTrainer().train(
    batch: trainingBatch,
    shape: shape,
    options: options
)
let predictor = AccelerateTemporalPredictor(model: report.model)
let output = try predictor.predict(window: latestWindow)
```

Run package tests and the local benchmark with:

```bash
swift test
swift run -c release ruvector-apple-benchmark
```

Benchmark output is labelled `MEASURED_ON_CURRENT_RUNTIME`. A Mac or simulator
result is not an iPhone result. Release claims require a physical-device run,
quality evaluation on a leakage-free held-out dataset, and application-level
governance tests.

## Native vector core

Applications that also need a small local embedding index can consume the
separate `ruvector-apple-core` Rust crate or its versioned C ABI. It provides a
bounded exact index and self-describing result/snapshot owners; it is not linked
into the Swift ML product automatically. See
`crates/ruvector-apple-core/README.md` and ADR-340 for the ownership and release
contract.
