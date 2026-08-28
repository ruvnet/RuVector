import Foundation

/// Learns a bounded cost profile for compatible execution candidates.
///
/// The caller remains responsible for correctness parity between candidates
/// and for measuring latency and any dimensionless relative-energy proxy. The
/// planner never treats a requested Core ML compute policy as proof of the
/// actual execution device selected by the operating system.
public actor AdaptiveExecutionPlanner {
    private static let maximumRestorableCounter = UInt64.max - 1_000_000

    private struct ProfileKey: Hashable {
        let workload: AdaptiveWorkloadDescriptor
        let candidate: AdaptiveExecutionCandidate
    }

    private struct Profile {
        var sampleCount: UInt32 = 0
        var latencyEWMA: Double = 0
        var relativeEnergyProxyEWMA: Double?
        var energySampleCount: UInt32 = 0
        var energyLastObservedSequence: UInt64 = 0
        var failureCount: UInt32 = 0
        var consecutiveFailures: UInt16 = 0
        var cooldownUntilSelection: UInt64 = 0
        var healthStateSelectionSequence: UInt64 = 0
        var lastTouchedSequence: UInt64 = 0

        init() {}

        init(
            snapshot: AdaptiveExecutionProfileSnapshot,
            healthStateSelectionSequence: UInt64
        ) {
            sampleCount = snapshot.sampleCount
            latencyEWMA = snapshot.latencyMillisecondsEWMA
            relativeEnergyProxyEWMA = snapshot.relativeEnergyProxyEWMA?.value
            energySampleCount = snapshot.energySampleCount
            energyLastObservedSequence = snapshot.energyLastObservedSequence
            failureCount = snapshot.failureCount
            consecutiveFailures = snapshot.consecutiveFailures
            cooldownUntilSelection = snapshot.cooldownUntilSelection
            self.healthStateSelectionSequence = healthStateSelectionSequence
            lastTouchedSequence = snapshot.lastTouchedSequence
        }
    }

    private struct WorkloadState {
        var selectionSequence: UInt64 = 0
        var operationSequence: UInt64 = 0
        var lastChoice: AdaptiveExecutionCandidate?
        var lastTouchedSequence: UInt64

        init(lastTouchedSequence: UInt64 = 0) {
            self.lastTouchedSequence = lastTouchedSequence
        }
    }

    private struct PendingDecision {
        let key: ProfileKey
    }

    public let fingerprint: AppleHardwareFingerprint
    public let optimizationContextRevision: AdaptiveOptimizationContextRevision
    public let configuration: AdaptivePlannerConfiguration

    private var profiles: [ProfileKey: Profile] = [:]
    private var workloadStates: [AdaptiveWorkloadDescriptor: WorkloadState] = [:]
    private var pendingDecisions: [UInt64: PendingDecision] = [:]
    private var sessionIdentifier = UUID()
    private var generation: UInt64 = 1
    private var selectionSequence: UInt64 = 0
    private var operationSequence: UInt64 = 0

    public init(
        optimizationContextRevision: AdaptiveOptimizationContextRevision,
        fingerprint: AppleHardwareFingerprint = .current(),
        configuration: AdaptivePlannerConfiguration = .standard
    ) throws {
        try AppleHardwareFingerprint.validate(fingerprint)
        _ = try AdaptiveOptimizationContextRevision(optimizationContextRevision.value)
        try AdaptivePlannerConfiguration.validate(configuration)
        self.fingerprint = fingerprint
        self.optimizationContextRevision = optimizationContextRevision
        self.configuration = configuration
    }

    public func select(
        workload: AdaptiveWorkloadDescriptor,
        candidates: [AdaptiveExecutionCandidate],
        runtimeState: AdaptiveRuntimeState
    ) throws -> AdaptiveExecutionDecision {
        try Self.validate(workload: workload, candidates: candidates)
        let eligible = candidates.filter {
            isEligible(candidate: $0, workload: workload, runtimeState: runtimeState)
        }
        guard !eligible.isEmpty else {
            throw RuVectorEdgeMLError.resourceUnavailable(
                "No adaptive execution candidate is permitted by the current runtime state"
            )
        }
        guard pendingDecisions.count < configuration.maximumProfiles else {
            throw RuVectorEdgeMLError.resourceUnavailable(
                "Adaptive planner has reached its bounded in-flight decision limit"
            )
        }

        let nextSelectionSequence = try incremented(selectionSequence)
        let nextOperationSequence = try incremented(operationSequence)
        let priorWorkloadState = workloadStates[workload] ?? WorkloadState()
        let nextWorkloadSelection = try incremented(priorWorkloadState.selectionSequence)
        let nextWorkloadOperation = try incremented(priorWorkloadState.operationSequence)
        try prepareWorkloadState(for: workload)

        let ready = eligible.filter {
            let profile = profiles[ProfileKey(workload: workload, candidate: $0)]
            return profile == nil || nextWorkloadSelection > profile!.cooldownUntilSelection
        }
        let allCoolingDown = ready.isEmpty
        let active = allCoolingDown ? eligible : ready
        let constrained = runtimeState.thermalState != .nominal
            || runtimeState.lowPowerModeEnabled
            || !runtimeState.appIsForeground

        let selected: AdaptiveExecutionCandidate
        var reason: AdaptiveSelectionReason
        let measured = active.filter {
            profiles[ProfileKey(workload: workload, candidate: $0)]?.sampleCount ?? 0 > 0
        }

        if allCoolingDown {
            selected = Self.mostConservative(active)
            reason = .allCandidatesCoolingDownFallback
        } else if measured.isEmpty {
            selected = Self.mostConservative(active)
            reason = constrained ? .constrainedFallback : .conservativeColdStart
        } else if shouldExplore(
            runtimeState: runtimeState,
            workloadSelectionSequence: nextWorkloadSelection
        ),
                  let exploration = explorationCandidate(
                    workload: workload,
                    candidates: active,
                    excluding: priorWorkloadState.lastChoice
                  ) {
            selected = exploration
            reason = .deterministicExploration
        } else {
            let scores = normalizedScores(
                workload: workload,
                candidates: measured,
                workloadOperationSequence: nextWorkloadOperation
            )
            let best = measured.min {
                Self.compare($0, $1, scores: scores)
            }!
            if let previous = priorWorkloadState.lastChoice,
               active.contains(previous),
               let previousScore = scores[previous],
               let bestScore = scores[best],
                previousScore <= bestScore * (1 + configuration.hysteresisFraction) {
                selected = previous
                reason = constrained ? .constrainedFallback : .hysteresis
            } else {
                selected = best
                reason = constrained ? .constrainedFallback : .learnedCost
            }
        }

        selectionSequence = nextSelectionSequence
        operationSequence = nextOperationSequence
        var workloadState = workloadStates[workload] ?? WorkloadState()
        workloadState.selectionSequence = nextWorkloadSelection
        workloadState.operationSequence = nextWorkloadOperation
        workloadState.lastChoice = selected
        workloadState.lastTouchedSequence = operationSequence
        workloadStates[workload] = workloadState

        var selectedProfile = profiles[ProfileKey(workload: workload, candidate: selected)]
        if selectedProfile != nil {
            selectedProfile!.lastTouchedSequence = operationSequence
            profiles[ProfileKey(workload: workload, candidate: selected)] = selectedProfile
        }
        let key = ProfileKey(workload: workload, candidate: selected)
        pendingDecisions[selectionSequence] = PendingDecision(key: key)
        let receipt = AdaptiveDecisionReceipt(
            sessionIdentifier: sessionIdentifier,
            generation: generation,
            selectionSequence: selectionSequence,
            workload: workload,
            candidate: selected
        )
        return decision(
            candidate: selected,
            reason: reason,
            eligibleCount: eligible.count,
            workload: workload,
            receipt: receipt,
            workloadOperationSequence: nextWorkloadOperation
        )
    }

    public func record(_ observation: AdaptiveExecutionObservation) throws {
        let receipt = observation.decisionReceipt
        try Self.validate(workload: receipt.workload, candidates: [receipt.candidate])
        let key = ProfileKey(workload: receipt.workload, candidate: receipt.candidate)
        guard receipt.sessionIdentifier == sessionIdentifier,
              receipt.generation == generation,
              let pending = pendingDecisions[receipt.selectionSequence],
              pending.key == key else {
            throw RuVectorEdgeMLError.invalidInput(
                "Adaptive observation is stale, replayed, or not issued by this planner"
            )
        }
        if let proxy = observation.relativeEnergyProxy {
            _ = try AdaptiveRelativeEnergyProxy(proxy.value)
        }
        guard var workloadState = workloadStates[receipt.workload] else {
            throw RuVectorEdgeMLError.invalidInput(
                "Adaptive observation refers to an unavailable workload state"
            )
        }
        try prepareProfileSlot(for: key)
        let nextOperationSequence = try incremented(operationSequence)
        let nextWorkloadOperation = try incremented(workloadState.operationSequence)

        operationSequence = nextOperationSequence
        workloadState.operationSequence = nextWorkloadOperation
        workloadState.lastTouchedSequence = operationSequence
        workloadStates[receipt.workload] = workloadState
        pendingDecisions.removeValue(forKey: receipt.selectionSequence)

        var profile = profiles[key] ?? Profile()
        profile.lastTouchedSequence = operationSequence

        if observation.succeeded {
            if profile.sampleCount == 0 {
                profile.latencyEWMA = observation.latencyMilliseconds
            } else {
                profile.latencyEWMA = ewma(
                    previous: profile.latencyEWMA,
                    observation: observation.latencyMilliseconds
                )
            }
            if let proxy = observation.relativeEnergyProxy?.value {
                profile.relativeEnergyProxyEWMA = profile.relativeEnergyProxyEWMA.map {
                    ewma(previous: $0, observation: proxy)
                } ?? proxy
                if profile.energySampleCount < UInt32.max { profile.energySampleCount += 1 }
                profile.energyLastObservedSequence = nextWorkloadOperation
            }
            if profile.sampleCount < UInt32.max { profile.sampleCount += 1 }
            if receipt.selectionSequence > profile.healthStateSelectionSequence {
                profile.consecutiveFailures = 0
                profile.cooldownUntilSelection = 0
                profile.healthStateSelectionSequence = receipt.selectionSequence
            }
        } else {
            if profile.failureCount < UInt32.max { profile.failureCount += 1 }
            if receipt.selectionSequence > profile.healthStateSelectionSequence {
                if profile.consecutiveFailures < UInt16.max { profile.consecutiveFailures += 1 }
                profile.cooldownUntilSelection = saturatingAdd(
                    workloadState.selectionSequence,
                    configuration.failureCooldownSelections
                )
                profile.healthStateSelectionSequence = receipt.selectionSequence
            }
        }
        profiles[key] = profile
    }

    public func makeSnapshot() -> AdaptiveExecutionSnapshot {
        let workloadClocks = workloadStates.map { workload, state in
            AdaptiveWorkloadClockSnapshot(
                workload: workload,
                selectionSequence: state.selectionSequence,
                operationSequence: state.operationSequence,
                lastTouchedSequence: state.lastTouchedSequence
            )
        }.sorted { lhs, rhs in
            Self.workloadCompare(lhs.workload, rhs.workload)
        }
        let entries = profiles.map { key, profile in
            AdaptiveExecutionProfileSnapshot(
                workload: key.workload,
                candidate: key.candidate,
                sampleCount: profile.sampleCount,
                latencyMillisecondsEWMA: profile.latencyEWMA,
                relativeEnergyProxyEWMA: profile.relativeEnergyProxyEWMA.flatMap {
                    try? AdaptiveRelativeEnergyProxy($0)
                },
                energySampleCount: profile.energySampleCount,
                energyLastObservedSequence: profile.energyLastObservedSequence,
                failureCount: profile.failureCount,
                consecutiveFailures: profile.consecutiveFailures,
                cooldownUntilSelection: profile.cooldownUntilSelection,
                lastTouchedSequence: profile.lastTouchedSequence
            )
        }.sorted { lhs, rhs in
            Self.snapshotCompare(lhs, rhs)
        }
        return .init(
            fingerprint: fingerprint,
            optimizationContextRevision: optimizationContextRevision,
            configuration: configuration,
            selectionSequence: selectionSequence,
            operationSequence: operationSequence,
            workloadClocks: workloadClocks,
            profiles: entries
        )
    }

    public func restore(_ snapshot: AdaptiveExecutionSnapshot) throws -> AdaptiveSnapshotRestoreResult {
        guard snapshot.version == AdaptiveExecutionSnapshot.schemaVersion else {
            reset()
            return .invalidatedSchemaMismatch
        }
        try AppleHardwareFingerprint.validate(snapshot.fingerprint)
        guard snapshot.fingerprint == fingerprint else {
            reset()
            return .invalidatedFingerprintMismatch
        }
        _ = try AdaptiveOptimizationContextRevision(snapshot.optimizationContextRevision.value)
        guard snapshot.optimizationContextRevision == optimizationContextRevision else {
            reset()
            return .invalidatedOptimizationContextMismatch
        }
        try AdaptivePlannerConfiguration.validate(snapshot.configuration)
        guard snapshot.configuration == configuration else {
            reset()
            return .invalidatedConfigurationMismatch
        }
        guard snapshot.profiles.count <= configuration.maximumProfiles,
              snapshot.workloadClocks.count <= configuration.maximumProfiles else {
            throw RuVectorEdgeMLError.invalidInput("Adaptive snapshot exceeds the profile limit")
        }
        guard snapshot.selectionSequence <= snapshot.operationSequence,
              snapshot.selectionSequence < Self.maximumRestorableCounter,
              snapshot.operationSequence < Self.maximumRestorableCounter else {
            throw RuVectorEdgeMLError.invalidInput(
                "Adaptive snapshot sequence counters are inconsistent"
            )
        }

        var restoredWorkloadStates: [AdaptiveWorkloadDescriptor: WorkloadState] = [:]
        for clock in snapshot.workloadClocks {
            try AdaptiveWorkloadDescriptor.validateIdentifier(clock.workload.identifier)
            guard clock.selectionSequence > 0,
                  clock.operationSequence > 0,
                  clock.selectionSequence <= clock.operationSequence,
                  clock.selectionSequence <= snapshot.selectionSequence,
                  clock.operationSequence <= snapshot.operationSequence,
                  clock.selectionSequence < Self.maximumRestorableCounter,
                  clock.operationSequence < Self.maximumRestorableCounter,
                  clock.lastTouchedSequence > 0,
                  clock.lastTouchedSequence <= snapshot.operationSequence,
                  restoredWorkloadStates[clock.workload] == nil else {
                throw RuVectorEdgeMLError.invalidInput(
                    "Adaptive snapshot contains an invalid workload clock"
                )
            }
            var state = WorkloadState(lastTouchedSequence: clock.lastTouchedSequence)
            state.selectionSequence = clock.selectionSequence
            state.operationSequence = clock.operationSequence
            restoredWorkloadStates[clock.workload] = state
        }

        var restored: [ProfileKey: Profile] = [:]
        for entry in snapshot.profiles {
            try Self.validate(workload: entry.workload, candidates: [entry.candidate])
            guard let workloadState = restoredWorkloadStates[entry.workload],
                  entry.latencyMillisecondsEWMA.isFinite,
                  entry.latencyMillisecondsEWMA >= 0,
                  (entry.sampleCount == 0
                    ? entry.latencyMillisecondsEWMA == 0
                        && entry.relativeEnergyProxyEWMA == nil
                    : entry.latencyMillisecondsEWMA
                        >= AdaptiveExecutionObservation.minimumLatencyMilliseconds),
                  entry.sampleCount > 0 || entry.failureCount > 0,
                  UInt32(entry.consecutiveFailures) <= entry.failureCount,
                  (entry.consecutiveFailures > 0 || entry.cooldownUntilSelection == 0),
                  entry.relativeEnergyProxyEWMA?.value.isFinite != false,
                  (entry.relativeEnergyProxyEWMA?.value ?? 0) >= 0,
                  (entry.relativeEnergyProxyEWMA?.value ?? 0)
                    <= AdaptiveRelativeEnergyProxy.maximumValue,
                  (entry.relativeEnergyProxyEWMA == nil
                    ? entry.energySampleCount == 0 && entry.energyLastObservedSequence == 0
                    : entry.energySampleCount > 0
                        && entry.energySampleCount <= entry.sampleCount
                        && entry.energyLastObservedSequence > 0
                        && entry.energyLastObservedSequence <= workloadState.operationSequence),
                  entry.cooldownUntilSelection <= workloadState.selectionSequence
                    || entry.cooldownUntilSelection - workloadState.selectionSequence
                        <= configuration.failureCooldownSelections,
                  entry.lastTouchedSequence > 0,
                  entry.lastTouchedSequence <= snapshot.operationSequence else {
                throw RuVectorEdgeMLError.invalidInput("Adaptive snapshot contains an invalid profile")
            }
            let key = ProfileKey(workload: entry.workload, candidate: entry.candidate)
            guard restored[key] == nil else {
                throw RuVectorEdgeMLError.invalidInput("Adaptive snapshot contains a duplicate profile")
            }
            restored[key] = Profile(
                snapshot: entry,
                healthStateSelectionSequence: snapshot.selectionSequence
            )
        }
        profiles = restored
        workloadStates = restoredWorkloadStates
        selectionSequence = snapshot.selectionSequence
        operationSequence = snapshot.operationSequence
        pendingDecisions.removeAll(keepingCapacity: false)
        advanceGeneration()
        return .restored(profileCount: restored.count)
    }

    public func reset() {
        profiles.removeAll(keepingCapacity: false)
        workloadStates.removeAll(keepingCapacity: false)
        pendingDecisions.removeAll(keepingCapacity: false)
        selectionSequence = 0
        operationSequence = 0
        advanceGeneration()
    }

    private func isEligible(
        candidate: AdaptiveExecutionCandidate,
        workload: AdaptiveWorkloadDescriptor,
        runtimeState: AdaptiveRuntimeState
    ) -> Bool {
        AdaptiveRuntimeEligibility.permits(
            backend: candidate.backend,
            workload: workload.kind,
            state: runtimeState,
            allowSimulatorTraining: configuration.allowSimulatorTraining
        )
    }

    private func shouldExplore(
        runtimeState: AdaptiveRuntimeState,
        workloadSelectionSequence: UInt64
    ) -> Bool {
        runtimeState.thermalState == .nominal
            && !runtimeState.lowPowerModeEnabled
            && runtimeState.appIsForeground
            && workloadSelectionSequence.isMultiple(of: configuration.explorationInterval)
    }

    private func explorationCandidate(
        workload: AdaptiveWorkloadDescriptor,
        candidates: [AdaptiveExecutionCandidate],
        excluding previous: AdaptiveExecutionCandidate?
    ) -> AdaptiveExecutionCandidate? {
        let alternatives = candidates.filter { $0 != previous }
        guard !alternatives.isEmpty else { return nil }
        return alternatives.min { lhs, rhs in
            let leftSamples = profiles[ProfileKey(workload: workload, candidate: lhs)]?.sampleCount ?? 0
            let rightSamples = profiles[ProfileKey(workload: workload, candidate: rhs)]?.sampleCount ?? 0
            if leftSamples != rightSamples { return leftSamples < rightSamples }
            return Self.conservativeCompare(lhs, rhs)
        }
    }

    private func normalizedScores(
        workload: AdaptiveWorkloadDescriptor,
        candidates: [AdaptiveExecutionCandidate],
        workloadOperationSequence: UInt64
    ) -> [AdaptiveExecutionCandidate: Double] {
        let candidateProfiles = candidates.compactMap { candidate -> (AdaptiveExecutionCandidate, Profile)? in
            guard let profile = profiles[ProfileKey(workload: workload, candidate: candidate)],
                  profile.sampleCount > 0 else { return nil }
            return (candidate, profile)
        }
        let minimumLatency = candidateProfiles.map(\.1.latencyEWMA).min() ?? 1
        let energyValues = candidateProfiles.compactMap {
            freshEnergy(for: $0.1, workloadOperationSequence: workloadOperationSequence)
        }
        let minimumEnergy = energyValues.count >= 2 ? energyValues.min() : nil
        let maximumEnergy = energyValues.count >= 2 ? energyValues.max() : nil

        return Dictionary(uniqueKeysWithValues: candidateProfiles.map { candidate, profile in
            let latencyScore = profile.latencyEWMA / max(minimumLatency, .leastNonzeroMagnitude)
            guard let minimumEnergy else { return (candidate, latencyScore) }
            let conservativeMissingEnergy = (maximumEnergy ?? minimumEnergy) * 1.10 + 0.000_001
            let energy = freshEnergy(
                for: profile,
                workloadOperationSequence: workloadOperationSequence
            ) ?? conservativeMissingEnergy
            let energyScore = (energy + 0.000_001) / (minimumEnergy + 0.000_001)
            let weight = configuration.latencyWeight + configuration.relativeEnergyProxyWeight
            let score = (
                latencyScore * configuration.latencyWeight
                    + energyScore * configuration.relativeEnergyProxyWeight
            ) / weight
            return (candidate, score)
        })
    }

    private func decision(
        candidate: AdaptiveExecutionCandidate,
        reason: AdaptiveSelectionReason,
        eligibleCount: Int,
        workload: AdaptiveWorkloadDescriptor,
        receipt: AdaptiveDecisionReceipt,
        workloadOperationSequence: UInt64
    ) -> AdaptiveExecutionDecision {
        let profile = profiles[ProfileKey(workload: workload, candidate: candidate)]
        return .init(
            receipt: receipt,
            candidate: candidate,
            reason: reason,
            estimatedLatencyMilliseconds: profile.flatMap { $0.sampleCount > 0 ? $0.latencyEWMA : nil },
            estimatedRelativeEnergyProxy: profile.flatMap {
                freshEnergy(
                    for: $0,
                    workloadOperationSequence: workloadOperationSequence
                )
            }.flatMap {
                try? AdaptiveRelativeEnergyProxy($0)
            },
            eligibleCandidateCount: eligibleCount
        )
    }

    private func ewma(previous: Double, observation: Double) -> Double {
        configuration.ewmaAlpha * observation + (1 - configuration.ewmaAlpha) * previous
    }

    private func freshEnergy(
        for profile: Profile,
        workloadOperationSequence: UInt64
    ) -> Double? {
        guard profile.energySampleCount >= configuration.minimumEnergySamples,
              profile.energyLastObservedSequence > 0,
              workloadOperationSequence >= profile.energyLastObservedSequence,
              workloadOperationSequence - profile.energyLastObservedSequence
                <= configuration.energyObservationMaxAge else {
            return nil
        }
        return profile.relativeEnergyProxyEWMA
    }

    func stateCountsForTesting() -> (profiles: Int, workloadStates: Int, pendingDecisions: Int) {
        (profiles.count, workloadStates.count, pendingDecisions.count)
    }

    private func prepareWorkloadState(for workload: AdaptiveWorkloadDescriptor) throws {
        guard workloadStates[workload] == nil else { return }
        if workloadStates.count >= configuration.maximumProfiles {
            let pendingWorkloads = Set(pendingDecisions.values.map(\.key.workload))
            let victim = workloadStates.filter { !pendingWorkloads.contains($0.key) }.min { lhs, rhs in
                if lhs.value.lastTouchedSequence != rhs.value.lastTouchedSequence {
                    return lhs.value.lastTouchedSequence < rhs.value.lastTouchedSequence
                }
                return Self.workloadCompare(lhs.key, rhs.key)
            }?.key
            guard let victim else {
                throw RuVectorEdgeMLError.resourceUnavailable(
                    "Adaptive planner workload-state capacity is pinned by in-flight decisions"
                )
            }
            let profileKeysToRemove = profiles.keys.filter { $0.workload == victim }
            for key in profileKeysToRemove {
                profiles.removeValue(forKey: key)
            }
            workloadStates.removeValue(forKey: victim)
        }
        workloadStates[workload] = WorkloadState()
    }

    private func prepareProfileSlot(for key: ProfileKey) throws {
        guard profiles[key] == nil,
              profiles.count >= configuration.maximumProfiles else { return }
        let pendingProfileKeys = Set(pendingDecisions.values.map(\.key))
        guard let victim = profiles.filter({ !pendingProfileKeys.contains($0.key) }).min(by: { lhs, rhs in
            if lhs.value.lastTouchedSequence != rhs.value.lastTouchedSequence {
                return lhs.value.lastTouchedSequence < rhs.value.lastTouchedSequence
            }
            return Self.profileKeyCompare(lhs.key, rhs.key)
        })?.key else {
            throw RuVectorEdgeMLError.resourceUnavailable(
                "Adaptive planner profile capacity is pinned by in-flight decisions"
            )
        }
        profiles.removeValue(forKey: victim)
    }

    private static func validate(
        workload: AdaptiveWorkloadDescriptor,
        candidates: [AdaptiveExecutionCandidate]
    ) throws {
        try AdaptiveWorkloadDescriptor.validateIdentifier(workload.identifier)
        guard !candidates.isEmpty, candidates.count <= 64 else {
            throw RuVectorEdgeMLError.invalidConfiguration(
                "Adaptive selection requires 1...64 compatible candidates"
            )
        }
        var identifiers = Set<String>()
        for candidate in candidates {
            _ = try AdaptiveExecutionCandidate(
                identifier: candidate.identifier,
                implementationRevision: candidate.implementationRevision,
                backend: candidate.backend,
                precision: candidate.precision,
                layout: candidate.layout,
                batchSize: candidate.batchSize
            )
            guard identifiers.insert(candidate.identifier).inserted else {
                throw RuVectorEdgeMLError.invalidConfiguration(
                    "Adaptive candidate identifiers must be unique within a selection"
                )
            }
        }
    }

    private static func mostConservative(
        _ candidates: [AdaptiveExecutionCandidate]
    ) -> AdaptiveExecutionCandidate {
        candidates.min(by: conservativeCompare)!
    }

    private static func conservativeCompare(
        _ lhs: AdaptiveExecutionCandidate,
        _ rhs: AdaptiveExecutionCandidate
    ) -> Bool {
        if lhs.backend.conservativeRank != rhs.backend.conservativeRank {
            return lhs.backend.conservativeRank < rhs.backend.conservativeRank
        }
        if lhs.identifier != rhs.identifier { return lhs.identifier < rhs.identifier }
        if lhs.implementationRevision != rhs.implementationRevision {
            return lhs.implementationRevision < rhs.implementationRevision
        }
        if lhs.backend.stableKey != rhs.backend.stableKey {
            return lhs.backend.stableKey < rhs.backend.stableKey
        }
        if lhs.precision.rawValue != rhs.precision.rawValue {
            return lhs.precision.rawValue < rhs.precision.rawValue
        }
        if lhs.layout.rawValue != rhs.layout.rawValue {
            return lhs.layout.rawValue < rhs.layout.rawValue
        }
        return lhs.batchSize < rhs.batchSize
    }

    private static func compare(
        _ lhs: AdaptiveExecutionCandidate,
        _ rhs: AdaptiveExecutionCandidate,
        scores: [AdaptiveExecutionCandidate: Double]
    ) -> Bool {
        let left = scores[lhs] ?? .infinity
        let right = scores[rhs] ?? .infinity
        if left != right { return left < right }
        return conservativeCompare(lhs, rhs)
    }

    private static func profileKeyCompare(_ lhs: ProfileKey, _ rhs: ProfileKey) -> Bool {
        if lhs.workload != rhs.workload {
            return workloadCompare(lhs.workload, rhs.workload)
        }
        return conservativeCompare(lhs.candidate, rhs.candidate)
    }

    private static func snapshotCompare(
        _ lhs: AdaptiveExecutionProfileSnapshot,
        _ rhs: AdaptiveExecutionProfileSnapshot
    ) -> Bool {
        if lhs.workload != rhs.workload {
            return workloadCompare(lhs.workload, rhs.workload)
        }
        return conservativeCompare(lhs.candidate, rhs.candidate)
    }

    private static func workloadCompare(
        _ lhs: AdaptiveWorkloadDescriptor,
        _ rhs: AdaptiveWorkloadDescriptor
    ) -> Bool {
        if lhs.kind.rawValue != rhs.kind.rawValue {
            return lhs.kind.rawValue < rhs.kind.rawValue
        }
        return lhs.identifier < rhs.identifier
    }

    private func incremented(_ value: UInt64) throws -> UInt64 {
        guard value < Self.maximumRestorableCounter else {
            throw RuVectorEdgeMLError.resourceUnavailable(
                "Adaptive planner sequence budget is exhausted; reset the planner"
            )
        }
        return value + 1
    }

    private func advanceGeneration() {
        if generation < Self.maximumRestorableCounter {
            generation += 1
        } else {
            sessionIdentifier = UUID()
            generation = 1
        }
    }

    private func saturatingAdd(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? UInt64.max : sum
    }
}
