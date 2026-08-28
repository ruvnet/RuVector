import Foundation

public struct AdaptivePlannerConfiguration: Codable, Equatable, Sendable {
    public static let standard = try! AdaptivePlannerConfiguration()

    public let maximumProfiles: Int
    public let ewmaAlpha: Double
    public let latencyWeight: Double
    public let relativeEnergyProxyWeight: Double
    public let explorationInterval: UInt64
    public let hysteresisFraction: Double
    public let failureCooldownSelections: UInt64
    public let minimumEnergySamples: UInt32
    public let energyObservationMaxAge: UInt64
    public let allowSimulatorTraining: Bool

    public init(
        maximumProfiles: Int = 256,
        ewmaAlpha: Double = 0.20,
        latencyWeight: Double = 0.75,
        relativeEnergyProxyWeight: Double = 0.25,
        explorationInterval: UInt64 = 16,
        hysteresisFraction: Double = 0.10,
        failureCooldownSelections: UInt64 = 4,
        minimumEnergySamples: UInt32 = 2,
        energyObservationMaxAge: UInt64 = 128,
        allowSimulatorTraining: Bool = false
    ) throws {
        guard (1...1_024).contains(maximumProfiles),
              ewmaAlpha.isFinite, ewmaAlpha > 0, ewmaAlpha <= 1,
              latencyWeight.isFinite, latencyWeight > 0, latencyWeight <= 1,
              relativeEnergyProxyWeight.isFinite,
              (0...1).contains(relativeEnergyProxyWeight),
              (latencyWeight + relativeEnergyProxyWeight).isFinite,
              (2...1_024).contains(explorationInterval),
              hysteresisFraction.isFinite,
              (0...1).contains(hysteresisFraction),
              (1...128).contains(failureCooldownSelections),
              (1...1_024).contains(minimumEnergySamples),
              (1...1_000_000).contains(energyObservationMaxAge) else {
            throw RuVectorEdgeMLError.invalidConfiguration(
                "Adaptive planner configuration is outside its bounded limits"
            )
        }
        self.maximumProfiles = maximumProfiles
        self.ewmaAlpha = ewmaAlpha
        self.latencyWeight = latencyWeight
        self.relativeEnergyProxyWeight = relativeEnergyProxyWeight
        self.explorationInterval = explorationInterval
        self.hysteresisFraction = hysteresisFraction
        self.failureCooldownSelections = failureCooldownSelections
        self.minimumEnergySamples = minimumEnergySamples
        self.energyObservationMaxAge = energyObservationMaxAge
        self.allowSimulatorTraining = allowSimulatorTraining
    }

    static func validate(_ configuration: Self) throws {
        _ = try Self(
            maximumProfiles: configuration.maximumProfiles,
            ewmaAlpha: configuration.ewmaAlpha,
            latencyWeight: configuration.latencyWeight,
            relativeEnergyProxyWeight: configuration.relativeEnergyProxyWeight,
            explorationInterval: configuration.explorationInterval,
            hysteresisFraction: configuration.hysteresisFraction,
            failureCooldownSelections: configuration.failureCooldownSelections,
            minimumEnergySamples: configuration.minimumEnergySamples,
            energyObservationMaxAge: configuration.energyObservationMaxAge,
            allowSimulatorTraining: configuration.allowSimulatorTraining
        )
    }
}

public struct AdaptiveExecutionProfileSnapshot: Codable, Equatable, Sendable {
    public let workload: AdaptiveWorkloadDescriptor
    public let candidate: AdaptiveExecutionCandidate
    public let sampleCount: UInt32
    public let latencyMillisecondsEWMA: Double
    public let relativeEnergyProxyEWMA: AdaptiveRelativeEnergyProxy?
    public let energySampleCount: UInt32
    public let energyLastObservedSequence: UInt64
    public let failureCount: UInt32
    public let consecutiveFailures: UInt16
    public let cooldownUntilSelection: UInt64
    public let lastTouchedSequence: UInt64
}

public struct AdaptiveWorkloadClockSnapshot: Codable, Equatable, Sendable {
    public let workload: AdaptiveWorkloadDescriptor
    public let selectionSequence: UInt64
    public let operationSequence: UInt64
    public let lastTouchedSequence: UInt64
}

public struct AdaptiveExecutionSnapshot: Codable, Equatable, Sendable {
    public static let schemaVersion: UInt16 = 3

    public let version: UInt16
    public let fingerprint: AppleHardwareFingerprint
    public let optimizationContextRevision: AdaptiveOptimizationContextRevision
    public let configuration: AdaptivePlannerConfiguration
    public let selectionSequence: UInt64
    public let operationSequence: UInt64
    public let workloadClocks: [AdaptiveWorkloadClockSnapshot]
    public let profiles: [AdaptiveExecutionProfileSnapshot]

    init(
        fingerprint: AppleHardwareFingerprint,
        optimizationContextRevision: AdaptiveOptimizationContextRevision,
        configuration: AdaptivePlannerConfiguration,
        selectionSequence: UInt64,
        operationSequence: UInt64,
        workloadClocks: [AdaptiveWorkloadClockSnapshot],
        profiles: [AdaptiveExecutionProfileSnapshot]
    ) {
        version = Self.schemaVersion
        self.fingerprint = fingerprint
        self.optimizationContextRevision = optimizationContextRevision
        self.configuration = configuration
        self.selectionSequence = selectionSequence
        self.operationSequence = operationSequence
        self.workloadClocks = workloadClocks
        self.profiles = profiles
    }

    private enum CodingKeys: String, CodingKey {
        case version
        case fingerprint
        case optimizationContextRevision
        case configuration
        case selectionSequence
        case operationSequence
        case workloadClocks
        case profiles
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(UInt16.self, forKey: .version)
        fingerprint = try container.decode(AppleHardwareFingerprint.self, forKey: .fingerprint)
        configuration = try container.decode(AdaptivePlannerConfiguration.self, forKey: .configuration)
        selectionSequence = try container.decode(UInt64.self, forKey: .selectionSequence)
        operationSequence = try container.decode(UInt64.self, forKey: .operationSequence)
        profiles = try container.decode([AdaptiveExecutionProfileSnapshot].self, forKey: .profiles)

        if version < Self.schemaVersion {
            optimizationContextRevision = try container.decodeIfPresent(
                AdaptiveOptimizationContextRevision.self,
                forKey: .optimizationContextRevision
            ) ?? AdaptiveOptimizationContextRevision("legacy-unscoped-snapshot")
            workloadClocks = try container.decodeIfPresent(
                [AdaptiveWorkloadClockSnapshot].self,
                forKey: .workloadClocks
            ) ?? []
        } else {
            optimizationContextRevision = try container.decode(
                AdaptiveOptimizationContextRevision.self,
                forKey: .optimizationContextRevision
            )
            workloadClocks = try container.decode(
                [AdaptiveWorkloadClockSnapshot].self,
                forKey: .workloadClocks
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(version, forKey: .version)
        try container.encode(fingerprint, forKey: .fingerprint)
        try container.encode(optimizationContextRevision, forKey: .optimizationContextRevision)
        try container.encode(configuration, forKey: .configuration)
        try container.encode(selectionSequence, forKey: .selectionSequence)
        try container.encode(operationSequence, forKey: .operationSequence)
        try container.encode(workloadClocks, forKey: .workloadClocks)
        try container.encode(profiles, forKey: .profiles)
    }
}

public enum AdaptiveSnapshotRestoreResult: Equatable, Sendable {
    case restored(profileCount: Int)
    case invalidatedSchemaMismatch
    case invalidatedFingerprintMismatch
    case invalidatedOptimizationContextMismatch
    case invalidatedConfigurationMismatch
}
