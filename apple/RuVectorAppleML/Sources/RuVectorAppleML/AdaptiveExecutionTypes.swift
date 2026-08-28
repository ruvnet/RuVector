import CoreML
import Foundation

public enum AdaptiveWorkloadKind: String, Codable, CaseIterable, Sendable {
    case temporalFusion
    case vectorSearch
    case pointCloud
    case visionInference
    case poseInference
    case modelInference
    case modelTraining
    case generic
}

/// Caller-owned revision for every non-hardware input that changes how costs
/// are measured or interpreted.
///
/// Applications should change this value when instrumentation, calibration,
/// room policy, scheduling policy, or another optimization-context input
/// changes. It is intentionally separate from a candidate's implementation
/// revision, which identifies the executable kernel or model artifact.
public struct AdaptiveOptimizationContextRevision: Codable, Equatable, Hashable, Sendable {
    public static let maximumByteCount = 128

    public let value: String

    public init(_ value: String) throws {
        guard value == value.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty,
              value.utf8.count <= Self.maximumByteCount else {
            throw RuVectorEdgeMLError.invalidConfiguration(
                "Adaptive optimization context revision must be trimmed and contain 1...128 UTF-8 bytes"
            )
        }
        self.value = value
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(container.decode(String.self))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

public struct AdaptiveWorkloadDescriptor: Codable, Hashable, Sendable {
    public static let maximumIdentifierByteCount = 128

    public let identifier: String
    public let kind: AdaptiveWorkloadKind

    public init(identifier: String, kind: AdaptiveWorkloadKind) throws {
        try Self.validateIdentifier(identifier)
        self.identifier = identifier
        self.kind = kind
    }

    static func validateIdentifier(_ identifier: String) throws {
        guard identifier == identifier.trimmingCharacters(in: .whitespacesAndNewlines),
              !identifier.isEmpty,
              identifier.utf8.count <= maximumIdentifierByteCount else {
            throw RuVectorEdgeMLError.invalidConfiguration(
                "Adaptive workload identifier must be trimmed and contain 1...128 UTF-8 bytes"
            )
        }
    }
}

/// The Core ML compute units a caller permits the operating system to use.
///
/// This is a request, not execution-device telemetry. Core ML may choose or
/// change the actual CPU, GPU, or Neural Engine placement within the allowed
/// set, and this package does not report that opaque choice.
public enum AdaptiveCoreMLComputePolicy: String, Codable, CaseIterable, Sendable {
    case cpuOnly
    case cpuAndGPU
    case cpuAndNeuralEngine
    case all

    public var requestedMLComputeUnits: MLComputeUnits {
        switch self {
        case .cpuOnly: return .cpuOnly
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .all: return .all
        }
    }

    public var actualPlacementIsOpaque: Bool { true }
}

public enum AdaptiveExecutionBackend: Codable, Hashable, Sendable {
    case accelerateCPU
    case metalGPU
    case coreML(requestedComputeUnits: AdaptiveCoreMLComputePolicy)

    var stableKey: String {
        switch self {
        case .accelerateCPU: return "accelerate-cpu"
        case .metalGPU: return "metal-gpu"
        case .coreML(let policy): return "coreml-requested-\(policy.rawValue)"
        }
    }

    var permitsGPU: Bool {
        switch self {
        case .metalGPU: return true
        case .accelerateCPU: return false
        case .coreML(let policy):
            return policy == .cpuAndGPU || policy == .all
        }
    }

    var isStrictCPU: Bool {
        switch self {
        case .accelerateCPU, .coreML(requestedComputeUnits: .cpuOnly):
            return true
        case .metalGPU,
             .coreML(requestedComputeUnits: .cpuAndGPU),
             .coreML(requestedComputeUnits: .cpuAndNeuralEngine),
             .coreML(requestedComputeUnits: .all):
            return false
        }
    }

    var conservativeRank: Int {
        switch self {
        case .accelerateCPU: return 0
        case .coreML(requestedComputeUnits: .cpuOnly): return 1
        case .coreML(requestedComputeUnits: .cpuAndNeuralEngine): return 2
        case .coreML(requestedComputeUnits: .cpuAndGPU): return 3
        case .coreML(requestedComputeUnits: .all): return 4
        case .metalGPU: return 5
        }
    }
}

enum AdaptiveRuntimeEligibility {
    static func permits(
        backend: AdaptiveExecutionBackend,
        workload: AdaptiveWorkloadKind,
        state: AdaptiveRuntimeState,
        allowSimulatorTraining: Bool
    ) -> Bool {
        if workload == .modelTraining {
            if state.simulator && !allowSimulatorTraining { return false }
            if !state.appIsForeground || state.lowPowerModeEnabled { return false }
            if state.thermalState != .nominal { return false }
        }
        switch state.thermalState {
        case .serious, .critical, .unknown:
            return backend.isStrictCPU
        case .fair:
            if backend.permitsGPU { return false }
        case .nominal:
            break
        }
        if (state.lowPowerModeEnabled || !state.appIsForeground), backend.permitsGPU {
            return false
        }
        return true
    }
}

public enum AdaptiveNumericPrecision: String, Codable, CaseIterable, Sendable {
    case float32
    case float16
    case int8
}

public enum AdaptiveTensorLayout: String, Codable, CaseIterable, Sendable {
    case rowMajor
    case channelsFirst
    case channelsLast
    case implementationDefined
}

public struct AdaptiveExecutionCandidate: Codable, Hashable, Sendable {
    public static let maximumIdentifierByteCount = 128
    public static let maximumBatchSize = 4_096

    public let identifier: String
    /// Caller-owned revision/digest for the exact kernel or model artifact.
    /// Reusing a candidate identifier with new code must use a new revision.
    public let implementationRevision: String
    public let backend: AdaptiveExecutionBackend
    public let precision: AdaptiveNumericPrecision
    public let layout: AdaptiveTensorLayout
    public let batchSize: Int

    public init(
        identifier: String,
        implementationRevision: String,
        backend: AdaptiveExecutionBackend,
        precision: AdaptiveNumericPrecision = .float32,
        layout: AdaptiveTensorLayout = .rowMajor,
        batchSize: Int = 1
    ) throws {
        guard identifier == identifier.trimmingCharacters(in: .whitespacesAndNewlines),
              !identifier.isEmpty,
              identifier.utf8.count <= Self.maximumIdentifierByteCount else {
            throw RuVectorEdgeMLError.invalidConfiguration(
                "Adaptive candidate identifier must be trimmed and contain 1...128 UTF-8 bytes"
            )
        }
        guard implementationRevision
            == implementationRevision.trimmingCharacters(in: .whitespacesAndNewlines),
              !implementationRevision.isEmpty,
              implementationRevision.utf8.count <= Self.maximumIdentifierByteCount else {
            throw RuVectorEdgeMLError.invalidConfiguration(
                "Adaptive implementation revision must be trimmed and contain 1...128 UTF-8 bytes"
            )
        }
        guard (1...Self.maximumBatchSize).contains(batchSize) else {
            throw RuVectorEdgeMLError.invalidConfiguration(
                "Adaptive candidate batch size must be in 1...4096"
            )
        }
        self.identifier = identifier
        self.implementationRevision = implementationRevision
        self.backend = backend
        self.precision = precision
        self.layout = layout
        self.batchSize = batchSize
    }

    var stableKey: String {
        [
            identifier,
            implementationRevision,
            backend.stableKey,
            precision.rawValue,
            layout.rawValue,
            String(batchSize),
        ]
            .joined(separator: "|")
    }
}

/// A non-negative, dimensionless proxy supplied by the application.
///
/// Values are meaningful only relative to other candidates measured by the
/// same caller and instrumentation. They are never interpreted as joules.
public struct AdaptiveRelativeEnergyProxy: Codable, Equatable, Sendable {
    public static let maximumValue = 1_000_000_000_000.0

    public let value: Double

    public init(_ value: Double) throws {
        guard value.isFinite, value >= 0, value <= Self.maximumValue else {
            throw RuVectorEdgeMLError.invalidInput(
                "Relative energy proxy must be finite and in 0...1e12"
            )
        }
        self.value = value
    }

    private enum CodingKeys: String, CodingKey {
        case value
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(container.decode(Double.self, forKey: .value))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(value, forKey: .value)
    }
}

public enum AdaptiveThermalState: String, Codable, CaseIterable, Sendable {
    case nominal
    case fair
    case serious
    case critical
    case unknown

    static func current(_ state: ProcessInfo.ThermalState) -> Self {
        switch state {
        case .nominal: return .nominal
        case .fair: return .fair
        case .serious: return .serious
        case .critical: return .critical
        @unknown default: return .unknown
        }
    }
}

public struct AdaptiveRuntimeState: Codable, Equatable, Sendable {
    public let thermalState: AdaptiveThermalState
    public let lowPowerModeEnabled: Bool
    public let appIsForeground: Bool
    public let simulator: Bool

    public init(
        thermalState: AdaptiveThermalState,
        lowPowerModeEnabled: Bool,
        appIsForeground: Bool,
        simulator: Bool
    ) {
        self.thermalState = thermalState
        self.lowPowerModeEnabled = lowPowerModeEnabled
        self.appIsForeground = appIsForeground
        self.simulator = simulator
    }

    public static func current(appIsForeground: Bool) -> Self {
        #if targetEnvironment(simulator)
        let simulator = true
        #else
        let simulator = false
        #endif
        let process = ProcessInfo.processInfo
        return .init(
            thermalState: .current(process.thermalState),
            lowPowerModeEnabled: process.isLowPowerModeEnabled,
            appIsForeground: appIsForeground,
            simulator: simulator
        )
    }
}

/// Opaque planner-issued identity for exactly one selected candidate.
///
/// Reset or restore invalidates outstanding receipts. Multiple decisions for
/// the same workload/candidate can remain in flight concurrently until the
/// planner's explicit pending-receipt bound is reached. There is intentionally
/// no public initializer.
public struct AdaptiveDecisionReceipt: Equatable, Sendable {
    public let sessionIdentifier: UUID
    public let generation: UInt64
    public let selectionSequence: UInt64
    public let workload: AdaptiveWorkloadDescriptor
    public let candidate: AdaptiveExecutionCandidate

    init(
        sessionIdentifier: UUID,
        generation: UInt64,
        selectionSequence: UInt64,
        workload: AdaptiveWorkloadDescriptor,
        candidate: AdaptiveExecutionCandidate
    ) {
        self.sessionIdentifier = sessionIdentifier
        self.generation = generation
        self.selectionSequence = selectionSequence
        self.workload = workload
        self.candidate = candidate
    }
}

public struct AdaptiveExecutionObservation: Sendable {
    public static let minimumLatencyMilliseconds = 0.000_001
    public static let maximumLatencyMilliseconds = 3_600_000.0

    public let decisionReceipt: AdaptiveDecisionReceipt
    public let latencyMilliseconds: Double
    public let relativeEnergyProxy: AdaptiveRelativeEnergyProxy?
    public let succeeded: Bool

    public init(
        decision: AdaptiveExecutionDecision,
        latencyMilliseconds: Double,
        relativeEnergyProxy: AdaptiveRelativeEnergyProxy? = nil,
        succeeded: Bool
    ) throws {
        guard latencyMilliseconds.isFinite,
              (Self.minimumLatencyMilliseconds...Self.maximumLatencyMilliseconds)
                .contains(latencyMilliseconds) else {
            throw RuVectorEdgeMLError.invalidInput(
                "Adaptive observation latency must be finite and in 1e-6...3600000 milliseconds"
            )
        }
        if let relativeEnergyProxy {
            _ = try AdaptiveRelativeEnergyProxy(relativeEnergyProxy.value)
        }
        decisionReceipt = decision.receipt
        self.latencyMilliseconds = latencyMilliseconds
        self.relativeEnergyProxy = relativeEnergyProxy
        self.succeeded = succeeded
    }
}

public enum AdaptiveSelectionReason: String, Codable, Sendable {
    case conservativeColdStart
    case learnedCost
    case hysteresis
    case deterministicExploration
    case constrainedFallback
    case allCandidatesCoolingDownFallback
}

public struct AdaptiveExecutionDecision: Equatable, Sendable {
    /// Planner-issued receipt required to record the eventual observation.
    public let receipt: AdaptiveDecisionReceipt
    public let candidate: AdaptiveExecutionCandidate
    public let reason: AdaptiveSelectionReason
    public let estimatedLatencyMilliseconds: Double?
    public let estimatedRelativeEnergyProxy: AdaptiveRelativeEnergyProxy?
    public let eligibleCandidateCount: Int
}
