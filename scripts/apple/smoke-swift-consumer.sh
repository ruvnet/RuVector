#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

ln -s "$repo_root" "$work_dir/ruvector"
mkdir -p "$work_dir/consumer/Sources/Consumer"

cat > "$work_dir/consumer/Package.swift" <<'SWIFT_PACKAGE'
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RuVectorAppleConsumer",
    platforms: [.macOS(.v13)],
    dependencies: [.package(path: "../ruvector")],
    targets: [
        .executableTarget(
            name: "Consumer",
            dependencies: [
                .product(name: "RuVectorAppleML", package: "ruvector")
            ]
        )
    ]
)
SWIFT_PACKAGE

cat > "$work_dir/consumer/Sources/Consumer/main.swift" <<'SWIFT_SOURCE'
import RuVectorAppleML

let shape = try TemporalModelShape(
    windowLength: 2,
    inputWidth: 3,
    hiddenWidth: 4,
    outputWidth: 2
)
guard shape.inputCount == 6 else {
    fatalError("public model-shape contract is unavailable")
}
let decision = RuntimeResourcePolicy.decision(for: .interactiveInference)
let optimizationContext = try AdaptiveOptimizationContextRevision(
    "consumer-measurement-v1|calibration-v1|policy-v1"
)
let planner = try AdaptiveExecutionPlanner(
    optimizationContextRevision: optimizationContext,
    configuration: .standard
)
let workload = try AdaptiveWorkloadDescriptor(
    identifier: "consumer-smoke",
    kind: .vectorSearch
)
let candidate = try AdaptiveExecutionCandidate(
    identifier: "accelerate-smoke",
    implementationRevision: "consumer-kernel-v1",
    backend: .accelerateCPU
)
let runtimeState = AdaptiveRuntimeState(
    thermalState: .nominal,
    lowPowerModeEnabled: false,
    appIsForeground: true,
    simulator: false
)
_ = planner
print(
    "RuVectorAppleML consumer ready: \(shape.inputCount), "
        + "\(decision.profile.rawValue), \(workload.identifier), "
        + "\(candidate.implementationRevision), \(runtimeState.thermalState.rawValue), "
        + optimizationContext.value
)
SWIFT_SOURCE

swift run --package-path "$work_dir/consumer" Consumer
