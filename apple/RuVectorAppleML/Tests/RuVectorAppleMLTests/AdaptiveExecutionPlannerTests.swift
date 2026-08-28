import CoreML
import Foundation
import XCTest
@testable import RuVectorAppleML

final class AdaptiveExecutionPlannerTests: XCTestCase {
    func testColdStartUsesConservativeCPUCandidate() async throws {
        let planner = try makePlanner()
        let decision = try await planner.select(
            workload: workload(),
            candidates: [metal(), cpu()],
            runtimeState: nominalState
        )

        let expected = try cpu()
        XCTAssertEqual(decision.candidate, expected)
        XCTAssertEqual(decision.reason, .conservativeColdStart)
        XCTAssertEqual(decision.eligibleCandidateCount, 2)
    }

    func testLearnsBoundedEWMAForLatencyAndRelativeEnergyProxy() async throws {
        let configuration = try AdaptivePlannerConfiguration(ewmaAlpha: 0.25)
        let planner = try makePlanner(configuration: configuration)
        try await record(on: planner, candidate: cpu(), latency: 100, energy: 8)
        try await record(on: planner, candidate: cpu(), latency: 20, energy: 4)

        let snapshot = await planner.makeSnapshot()
        let profile = try XCTUnwrap(snapshot.profiles.first)
        XCTAssertEqual(profile.sampleCount, 2)
        XCTAssertEqual(profile.latencyMillisecondsEWMA, 80, accuracy: 0.000_001)
        XCTAssertEqual(profile.relativeEnergyProxyEWMA?.value ?? -1, 7, accuracy: 0.000_001)
    }

    func testDeterministicExplorationSamplesAnUnmeasuredAlternative() async throws {
        let configuration = try AdaptivePlannerConfiguration(explorationInterval: 2)
        let planner = try makePlanner(configuration: configuration)
        let first = try await planner.select(
            workload: workload(),
            candidates: [metal(), cpu()],
            runtimeState: nominalState
        )
        try await planner.record(observation(decision: first, latency: 12))

        let second = try await planner.select(
            workload: workload(),
            candidates: [metal(), cpu()],
            runtimeState: nominalState
        )
        let expected = try metal()
        XCTAssertEqual(second.candidate, expected)
        XCTAssertEqual(second.reason, .deterministicExploration)
    }

    func testExplorationCadenceIsLocalToEachWorkload() async throws {
        let planner = try makePlanner(
            configuration: try .init(explorationInterval: 2)
        )
        let primary = try workload()
        let firstPrimary = try await planner.select(
            workload: primary,
            candidates: [cpu(), metal()],
            runtimeState: nominalState
        )
        try await planner.record(observation(decision: firstPrimary, latency: 10))

        let unrelated = try AdaptiveWorkloadDescriptor(identifier: "unrelated", kind: .generic)
        let unrelatedDecision = try await planner.select(
            workload: unrelated,
            candidates: [cpu()],
            runtimeState: nominalState
        )
        try await planner.record(observation(decision: unrelatedDecision, latency: 1))

        let secondPrimary = try await planner.select(
            workload: primary,
            candidates: [cpu(), metal()],
            runtimeState: nominalState
        )
        XCTAssertEqual(secondPrimary.candidate, try metal())
        XCTAssertEqual(secondPrimary.reason, .deterministicExploration)
    }

    func testHysteresisAvoidsSwitchingForSmallPredictedGain() async throws {
        let planner = try makePlanner(
            configuration: try .init(hysteresisFraction: 0.10)
        )
        let initial = try await planner.select(
            workload: workload(),
            candidates: [cpu(), metal()],
            runtimeState: nominalState
        )
        try await planner.record(observation(decision: initial, latency: 10))
        try await record(on: planner, candidate: metal(), latency: 9.5)
        _ = try await planner.select(
            workload: workload(),
            candidates: [cpu()],
            runtimeState: nominalState
        )

        let decision = try await planner.select(
            workload: workload(),
            candidates: [cpu(), metal()],
            runtimeState: nominalState
        )
        let expected = try cpu()
        XCTAssertEqual(decision.candidate, expected)
        XCTAssertEqual(decision.reason, .hysteresis)
    }

    func testRelativeEnergyProxyCanFavorLowerCostCandidate() async throws {
        let configuration = try AdaptivePlannerConfiguration(
            latencyWeight: 0.5,
            relativeEnergyProxyWeight: 0.5
        )
        let planner = try makePlanner(configuration: configuration)
        try await record(on: planner, candidate: cpu(), latency: 11, energy: 1)
        try await record(on: planner, candidate: cpu(), latency: 11, energy: 1)
        try await record(on: planner, candidate: metal(), latency: 10, energy: 10)
        try await record(on: planner, candidate: metal(), latency: 10, energy: 10)

        let decision = try await planner.select(
            workload: workload(),
            candidates: [metal(), cpu()],
            runtimeState: nominalState
        )
        let expected = try cpu()
        XCTAssertEqual(decision.candidate, expected)
        XCTAssertEqual(decision.reason, .learnedCost)
    }

    func testEnergyEvidenceExpiresAfterSameWorkloadOperations() async throws {
        let planner = try makePlanner(
            configuration: try .init(
                minimumEnergySamples: 1,
                energyObservationMaxAge: 2
            )
        )
        try await record(on: planner, candidate: cpu(), latency: 10, energy: 1)
        for _ in 0..<3 {
            _ = try await planner.select(
                workload: workload(),
                candidates: [cpu()],
                runtimeState: nominalState
            )
        }
        let decision = try await planner.select(
            workload: workload(),
            candidates: [cpu()],
            runtimeState: nominalState
        )
        XCTAssertNil(decision.estimatedRelativeEnergyProxy)
    }

    func testUnrelatedWorkloadOperationsDoNotExpireEnergyEvidence() async throws {
        let planner = try makePlanner(
            configuration: try .init(
                minimumEnergySamples: 1,
                energyObservationMaxAge: 2
            )
        )
        try await record(on: planner, candidate: cpu(), latency: 10, energy: 1)
        for index in 0..<8 {
            let unrelated = try AdaptiveWorkloadDescriptor(
                identifier: "unrelated-energy-\(index)",
                kind: .generic
            )
            let decision = try await planner.select(
                workload: unrelated,
                candidates: [cpu()],
                runtimeState: nominalState
            )
            try await planner.record(observation(decision: decision, latency: 1))
        }

        let decision = try await planner.select(
            workload: workload(),
            candidates: [cpu()],
            runtimeState: nominalState
        )
        XCTAssertEqual(decision.estimatedRelativeEnergyProxy?.value, 1)
    }

    func testFailureCooldownFallsBackToHealthyCandidate() async throws {
        let planner = try makePlanner(
            configuration: try .init(failureCooldownSelections: 2)
        )
        try await record(on: planner, candidate: cpu(), latency: 20)
        try await record(on: planner, candidate: metal(), latency: 5)
        let selected = try await planner.select(
            workload: workload(),
            candidates: [cpu(), metal()],
            runtimeState: nominalState
        )
        let expectedMetal = try metal()
        XCTAssertEqual(selected.candidate, expectedMetal)
        try await planner.record(observation(decision: selected, latency: 1, succeeded: false))

        let fallback = try await planner.select(
            workload: workload(),
            candidates: [cpu(), metal()],
            runtimeState: nominalState
        )
        let expectedCPU = try cpu()
        XCTAssertEqual(fallback.candidate, expectedCPU)
    }

    func testUnrelatedWorkloadsDoNotAdvanceFailureCooldown() async throws {
        let planner = try makePlanner(
            configuration: try .init(failureCooldownSelections: 2)
        )
        try await record(on: planner, candidate: metal(), latency: 1, succeeded: false)

        for index in 0..<8 {
            let unrelated = try AdaptiveWorkloadDescriptor(
                identifier: "unrelated-cooldown-\(index)",
                kind: .generic
            )
            let decision = try await planner.select(
                workload: unrelated,
                candidates: [cpu()],
                runtimeState: nominalState
            )
            try await planner.record(observation(decision: decision, latency: 1))
        }

        let primary = try await planner.select(
            workload: workload(),
            candidates: [metal(), cpu()],
            runtimeState: nominalState
        )
        XCTAssertEqual(primary.candidate, try cpu())
    }

    func testAllCoolingCandidatesUseExplicitConservativeFallback() async throws {
        let planner = try makePlanner()
        try await record(on: planner, candidate: cpu(), latency: 1, succeeded: false)
        try await record(on: planner, candidate: metal(), latency: 1, succeeded: false)

        let decision = try await planner.select(
            workload: workload(),
            candidates: [metal(), cpu()],
            runtimeState: nominalState
        )
        let expected = try cpu()
        XCTAssertEqual(decision.candidate, expected)
        XCTAssertEqual(decision.reason, .allCandidatesCoolingDownFallback)
    }

    func testThermalLowPowerAndBackgroundStatesGateGPUCandidates() async throws {
        let planner = try makePlanner()
        try await record(on: planner, candidate: cpu(), latency: 50)
        try await record(on: planner, candidate: metal(), latency: 1)

        for state in [
            AdaptiveRuntimeState(
                thermalState: .critical,
                lowPowerModeEnabled: false,
                appIsForeground: true,
                simulator: false
            ),
            AdaptiveRuntimeState(
                thermalState: .nominal,
                lowPowerModeEnabled: true,
                appIsForeground: true,
                simulator: false
            ),
            AdaptiveRuntimeState(
                thermalState: .nominal,
                lowPowerModeEnabled: false,
                appIsForeground: false,
                simulator: false
            ),
        ] {
            let decision = try await planner.select(
                workload: workload(),
                candidates: [metal(), cpu()],
                runtimeState: state
            )
            let expected = try cpu()
            XCTAssertEqual(decision.candidate, expected)
            XCTAssertEqual(decision.eligibleCandidateCount, 1)
            XCTAssertEqual(decision.reason, .constrainedFallback)
        }
    }

    func testTrainingIsRejectedOnSimulatorByDefault() async throws {
        let planner = try makePlanner()
        let training = try AdaptiveWorkloadDescriptor(identifier: "trainer", kind: .modelTraining)
        do {
            _ = try await planner.select(
                workload: training,
                candidates: [cpu()],
                runtimeState: .init(
                    thermalState: .nominal,
                    lowPowerModeEnabled: false,
                    appIsForeground: true,
                    simulator: true
                )
            )
            XCTFail("Simulator training should be rejected")
        } catch let error as RuVectorEdgeMLError {
            guard case .resourceUnavailable = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testSeriousThermalStateDoesNotTreatRequestedNeuralEngineAsStrictCPU() async throws {
        let planner = try makePlanner()
        let coreML = try AdaptiveExecutionCandidate(
            identifier: "coreml-ne",
            implementationRevision: "model-sha256-v1",
            backend: .coreML(requestedComputeUnits: .cpuAndNeuralEngine)
        )
        let decision = try await planner.select(
            workload: workload(),
            candidates: [coreML, cpu()],
            runtimeState: .init(
                thermalState: .serious,
                lowPowerModeEnabled: false,
                appIsForeground: true,
                simulator: false
            )
        )

        let expected = try cpu()
        XCTAssertEqual(decision.candidate, expected)
        XCTAssertEqual(decision.eligibleCandidateCount, 1)
    }

    func testSnapshotRestoresOnlyForExactFingerprintContextAndConfiguration() async throws {
        let source = try makePlanner()
        try await record(on: source, candidate: cpu(), latency: 12, energy: 3)
        let snapshot = await source.makeSnapshot()

        let matching = try makePlanner()
        let restored = try await matching.restore(snapshot)
        XCTAssertEqual(restored, .restored(profileCount: 1))
        let matchingSnapshot = await matching.makeSnapshot()
        XCTAssertEqual(matchingSnapshot.profiles.count, 1)

        let changedFingerprint = try AppleHardwareFingerprint(
            platform: "ios",
            machineIdentifier: "iPhone-different-class",
            operatingSystemVersion: "20.0.0",
            logicalProcessorCount: 6,
            memoryClassMegabytes: 8_192
        )
        let mismatched = try makePlanner(fingerprint: changedFingerprint)
        let invalidated = try await mismatched.restore(snapshot)
        XCTAssertEqual(invalidated, .invalidatedFingerprintMismatch)
        let mismatchedSnapshot = await mismatched.makeSnapshot()
        XCTAssertTrue(mismatchedSnapshot.profiles.isEmpty)

        let changedContext = try makePlanner(
            contextRevision: contextRevision("measurement-v2|calibration-v1|policy-v1")
        )
        let contextInvalidated = try await changedContext.restore(snapshot)
        XCTAssertEqual(contextInvalidated, .invalidatedOptimizationContextMismatch)
        let contextInvalidatedSnapshot = await changedContext.makeSnapshot()
        XCTAssertTrue(contextInvalidatedSnapshot.profiles.isEmpty)
    }

    func testSnapshotCodableRoundTripPreservesRestorableProfile() async throws {
        let source = try makePlanner()
        try await record(on: source, candidate: cpu(), latency: 7, energy: 2)
        let encoded = try JSONEncoder().encode(await source.makeSnapshot())
        let decoded = try JSONDecoder().decode(AdaptiveExecutionSnapshot.self, from: encoded)
        let destination = try makePlanner()

        let result = try await destination.restore(decoded)
        XCTAssertEqual(result, .restored(profileCount: 1))
        let restored = await destination.makeSnapshot()
        XCTAssertEqual(restored.profiles.first?.latencyMillisecondsEWMA, 7)
    }

    func testLegacyV2SnapshotDecodesAndInvalidatesBySchema() async throws {
        let source = try makePlanner()
        try await record(on: source, candidate: cpu(), latency: 7, energy: 2)
        let encoded = try JSONEncoder().encode(await source.makeSnapshot())
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        object["version"] = 2
        object.removeValue(forKey: "optimizationContextRevision")
        object.removeValue(forKey: "workloadClocks")
        let legacyData = try JSONSerialization.data(withJSONObject: object)
        let decoded = try JSONDecoder().decode(AdaptiveExecutionSnapshot.self, from: legacyData)
        let destination = try makePlanner()

        let result = try await destination.restore(decoded)
        XCTAssertEqual(result, .invalidatedSchemaMismatch)
        let invalidatedSnapshot = await destination.makeSnapshot()
        XCTAssertTrue(invalidatedSnapshot.profiles.isEmpty)
    }

    func testProfilesAreBoundedAndEvictLeastRecentlyTouched() async throws {
        let planner = try makePlanner(configuration: try .init(maximumProfiles: 2))
        let first = try AdaptiveExecutionCandidate(
            identifier: "first",
            implementationRevision: "kernel-v1",
            backend: .accelerateCPU
        )
        let second = try AdaptiveExecutionCandidate(
            identifier: "second",
            implementationRevision: "kernel-v1",
            backend: .accelerateCPU
        )
        let third = try AdaptiveExecutionCandidate(
            identifier: "third",
            implementationRevision: "kernel-v1",
            backend: .accelerateCPU
        )
        try await record(on: planner, candidate: first, latency: 1)
        try await record(on: planner, candidate: second, latency: 1)
        try await record(on: planner, candidate: third, latency: 1)

        let identifiers = Set(await planner.makeSnapshot().profiles.map(\.candidate.identifier))
        XCTAssertEqual(identifiers, Set(["second", "third"]))
    }

    func testMultipleInFlightReceiptsRemainValidAndCapacityRefusesWithoutEviction() async throws {
        let planner = try makePlanner(configuration: try .init(maximumProfiles: 2))
        let first = try await planner.select(
            workload: workload(),
            candidates: [cpu()],
            runtimeState: nominalState
        )
        let second = try await planner.select(
            workload: workload(),
            candidates: [cpu()],
            runtimeState: nominalState
        )
        do {
            _ = try await planner.select(
                workload: workload(),
                candidates: [cpu()],
                runtimeState: nominalState
            )
            XCTFail("The bounded in-flight decision limit should refuse a new selection")
        } catch let error as RuVectorEdgeMLError {
            guard case .resourceUnavailable = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
        var counts = await planner.stateCountsForTesting()
        XCTAssertEqual(counts.pendingDecisions, 2)

        try await planner.record(observation(decision: second, latency: 2))
        try await planner.record(observation(decision: first, latency: 1))
        counts = await planner.stateCountsForTesting()
        XCTAssertEqual(counts.pendingDecisions, 0)
        XCTAssertEqual(counts.profiles, 1)

        do {
            try await planner.record(observation(decision: first, latency: 1))
            XCTFail("A consumed decision receipt must not be replayable")
        } catch let error as RuVectorEdgeMLError {
            guard case .invalidInput = error else { return XCTFail("Unexpected error: \(error)") }
        }
    }

    func testWorkloadClockAndProfileStateRemainBounded() async throws {
        let planner = try makePlanner(configuration: try .init(maximumProfiles: 2))
        for index in 0..<20 {
            let selectedWorkload = try AdaptiveWorkloadDescriptor(
                identifier: "bounded-workload-\(index)",
                kind: .generic
            )
            let decision = try await planner.select(
                workload: selectedWorkload,
                candidates: [cpu()],
                runtimeState: nominalState
            )
            try await planner.record(observation(decision: decision, latency: 1))
        }

        let counts = await planner.stateCountsForTesting()
        XCTAssertLessThanOrEqual(counts.profiles, 2)
        XCTAssertLessThanOrEqual(counts.workloadStates, 2)
        XCTAssertEqual(counts.pendingDecisions, 0)
        let snapshot = await planner.makeSnapshot()
        XCTAssertLessThanOrEqual(snapshot.workloadClocks.count, 2)
    }

    func testOlderInFlightSuccessCannotClearNewerFailureCooldown() async throws {
        let planner = try makePlanner()
        let older = try await planner.select(
            workload: workload(),
            candidates: [metal()],
            runtimeState: nominalState
        )
        let newer = try await planner.select(
            workload: workload(),
            candidates: [metal()],
            runtimeState: nominalState
        )
        try await planner.record(observation(decision: newer, latency: 1, succeeded: false))
        try await planner.record(observation(decision: older, latency: 1))
        do {
            try await planner.record(observation(decision: newer, latency: 1))
            XCTFail("A consumed decision receipt must not be replayable")
        } catch let error as RuVectorEdgeMLError {
            guard case .invalidInput = error else { return XCTFail("Unexpected error: \(error)") }
        }
        let snapshot = await planner.makeSnapshot()
        let profile = try XCTUnwrap(snapshot.profiles.first)
        XCTAssertEqual(profile.sampleCount, 1)
        XCTAssertEqual(profile.consecutiveFailures, 1)
        XCTAssertGreaterThan(profile.cooldownUntilSelection, 0)
    }

    func testResetInvalidatesOutstandingDecisionReceipts() async throws {
        let planner = try makePlanner()
        let decision = try await planner.select(
            workload: workload(),
            candidates: [cpu()],
            runtimeState: nominalState
        )
        await planner.reset()
        do {
            try await planner.record(observation(decision: decision, latency: 1))
            XCTFail("Reset must invalidate outstanding decision receipts")
        } catch let error as RuVectorEdgeMLError {
            guard case .invalidInput = error else { return XCTFail("Unexpected error: \(error)") }
        }
    }

    func testImplementationRevisionSeparatesLearnedProfiles() async throws {
        let planner = try makePlanner()
        let first = try AdaptiveExecutionCandidate(
            identifier: "cpu",
            implementationRevision: "kernel-v1",
            backend: .accelerateCPU
        )
        let second = try AdaptiveExecutionCandidate(
            identifier: "cpu",
            implementationRevision: "kernel-v2",
            backend: .accelerateCPU
        )
        try await record(on: planner, candidate: first, latency: 10)
        try await record(on: planner, candidate: second, latency: 20)
        let revisions = Set(await planner.makeSnapshot().profiles.map(
            \.candidate.implementationRevision
        ))
        XCTAssertEqual(revisions, Set(["kernel-v1", "kernel-v2"]))
    }

    func testRestoreRejectsCounterValuesThatCouldFreezeFutureLearning() async throws {
        let planner = try makePlanner()
        let snapshot = AdaptiveExecutionSnapshot(
            fingerprint: try fingerprint(),
            optimizationContextRevision: try contextRevision(),
            configuration: .standard,
            selectionSequence: UInt64.max,
            operationSequence: UInt64.max,
            workloadClocks: [],
            profiles: []
        )
        do {
            _ = try await planner.restore(snapshot)
            XCTFail("Pathological counters should be rejected")
        } catch let error as RuVectorEdgeMLError {
            guard case .invalidInput = error else { return XCTFail("Unexpected error: \(error)") }
        }
    }

    func testAdaptiveCoreMLPolicyUsesTheSameImmediateThermalMatrixAsPlanner() {
        let serious = AdaptiveRuntimeState(
            thermalState: .serious,
            lowPowerModeEnabled: false,
            appIsForeground: true,
            simulator: false
        )
        XCTAssertTrue(CoreMLModelSession.adaptivePolicyPermits(
            .cpuOnly,
            workload: .interactiveInference,
            runtimeState: serious
        ))
        XCTAssertFalse(CoreMLModelSession.adaptivePolicyPermits(
            .all,
            workload: .interactiveInference,
            runtimeState: serious
        ))
        XCTAssertFalse(CoreMLModelSession.adaptivePolicyPermits(
            .cpuAndNeuralEngine,
            workload: .training,
            runtimeState: serious
        ))
        let nominalForeground = AdaptiveRuntimeState(
            thermalState: .nominal,
            lowPowerModeEnabled: false,
            appIsForeground: true,
            simulator: false
        )
        XCTAssertTrue(CoreMLModelSession.adaptivePolicyPermits(
            .cpuOnly,
            workload: .backgroundInference,
            runtimeState: nominalForeground
        ))
        XCTAssertFalse(CoreMLModelSession.adaptivePolicyPermits(
            .all,
            workload: .backgroundInference,
            runtimeState: nominalForeground
        ))
    }

    func testCoreMLPolicyDescribesRequestedUnitsWithoutClaimingPlacement() throws {
        XCTAssertEqual(
            AdaptiveCoreMLComputePolicy.cpuAndNeuralEngine.requestedMLComputeUnits,
            MLComputeUnits.cpuAndNeuralEngine
        )
        XCTAssertTrue(AdaptiveCoreMLComputePolicy.cpuAndNeuralEngine.actualPlacementIsOpaque)
        let candidate = try AdaptiveExecutionCandidate(
            identifier: "coreml-flexible",
            implementationRevision: "model-sha256-v1",
            backend: .coreML(requestedComputeUnits: .all)
        )
        guard case .coreML(let requestedPolicy) = candidate.backend else {
            return XCTFail("Expected Core ML candidate")
        }
        XCTAssertEqual(requestedPolicy, .all)
    }

    func testInputBoundsRejectInvalidCandidatesAndDuplicateIdentifiers() async throws {
        XCTAssertThrowsError(try AdaptiveExecutionCandidate(
            identifier: "",
            implementationRevision: "kernel-v1",
            backend: .accelerateCPU
        ))
        XCTAssertThrowsError(try AdaptiveExecutionCandidate(
            identifier: "too-large",
            implementationRevision: "kernel-v1",
            backend: .accelerateCPU,
            batchSize: 4_097
        ))
        XCTAssertThrowsError(try AdaptiveRelativeEnergyProxy(
            AdaptiveRelativeEnergyProxy.maximumValue + 1
        ))
        XCTAssertThrowsError(try JSONDecoder().decode(
            AdaptiveRelativeEnergyProxy.self,
            from: Data("{\"value\":-1}".utf8)
        ))
        XCTAssertThrowsError(try JSONDecoder().decode(
            AdaptiveRelativeEnergyProxy.self,
            from: Data("{\"value\":1000000000001}".utf8)
        ))
        XCTAssertThrowsError(try AdaptiveOptimizationContextRevision(""))
        XCTAssertThrowsError(try AdaptivePlannerConfiguration(
            latencyWeight: 0,
            relativeEnergyProxyWeight: 1
        ))
        XCTAssertThrowsError(try AdaptivePlannerConfiguration(
            latencyWeight: 1,
            relativeEnergyProxyWeight: .greatestFiniteMagnitude
        ))
        let boundedDecision = try await makePlanner().select(
            workload: workload(),
            candidates: [cpu()],
            runtimeState: nominalState
        )
        XCTAssertThrowsError(try AdaptiveExecutionObservation(
            decision: boundedDecision,
            latencyMilliseconds: .leastNonzeroMagnitude,
            succeeded: true
        ))

        let planner = try makePlanner()
        let duplicate = try AdaptiveExecutionCandidate(
            identifier: "cpu",
            implementationRevision: "kernel-v2",
            backend: .metalGPU
        )
        do {
            _ = try await planner.select(
                workload: workload(),
                candidates: [cpu(), duplicate],
                runtimeState: nominalState
            )
            XCTFail("Duplicate identifiers should be rejected")
        } catch let error as RuVectorEdgeMLError {
            guard case .invalidConfiguration = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testRestoreRejectsNoncanonicalDecodedProfilesAndCounters() async throws {
        let planner = try makePlanner()
        let candidate = try cpu()
        let invalidUnmeasured = AdaptiveExecutionProfileSnapshot(
            workload: try workload(),
            candidate: candidate,
            sampleCount: 0,
            latencyMillisecondsEWMA: 10,
            relativeEnergyProxyEWMA: nil,
            energySampleCount: 0,
            energyLastObservedSequence: 0,
            failureCount: 0,
            consecutiveFailures: 0,
            cooldownUntilSelection: 0,
            lastTouchedSequence: 1
        )
        let invalidProfileSnapshot = AdaptiveExecutionSnapshot(
            fingerprint: try fingerprint(),
            optimizationContextRevision: try contextRevision(),
            configuration: .standard,
            selectionSequence: 1,
            operationSequence: 2,
            workloadClocks: [
                AdaptiveWorkloadClockSnapshot(
                    workload: try workload(),
                    selectionSequence: 1,
                    operationSequence: 2,
                    lastTouchedSequence: 2
                ),
            ],
            profiles: [invalidUnmeasured]
        )
        do {
            _ = try await planner.restore(invalidProfileSnapshot)
            XCTFail("Noncanonical unmeasured profile should be rejected")
        } catch {
            // Expected.
        }

        let invalidCounters = AdaptiveExecutionSnapshot(
            fingerprint: try fingerprint(),
            optimizationContextRevision: try contextRevision(),
            configuration: .standard,
            selectionSequence: 2,
            operationSequence: 1,
            workloadClocks: [],
            profiles: []
        )
        do {
            _ = try await planner.restore(invalidCounters)
            XCTFail("Inconsistent sequence counters should be rejected")
        } catch {
            // Expected.
        }
    }

    private var nominalState: AdaptiveRuntimeState {
        .init(
            thermalState: .nominal,
            lowPowerModeEnabled: false,
            appIsForeground: true,
            simulator: false
        )
    }

    private func makePlanner(
        configuration: AdaptivePlannerConfiguration = .standard,
        fingerprint suppliedFingerprint: AppleHardwareFingerprint? = nil,
        contextRevision suppliedContextRevision: AdaptiveOptimizationContextRevision? = nil
    ) throws -> AdaptiveExecutionPlanner {
        let resolvedFingerprint: AppleHardwareFingerprint
        if let suppliedFingerprint {
            resolvedFingerprint = suppliedFingerprint
        } else {
            resolvedFingerprint = try fingerprint()
        }
        let resolvedContextRevision: AdaptiveOptimizationContextRevision
        if let suppliedContextRevision {
            resolvedContextRevision = suppliedContextRevision
        } else {
            resolvedContextRevision = try contextRevision()
        }
        return try AdaptiveExecutionPlanner(
            optimizationContextRevision: resolvedContextRevision,
            fingerprint: resolvedFingerprint,
            configuration: configuration
        )
    }

    private func contextRevision(
        _ value: String = "measurement-v1|calibration-v1|policy-v1"
    ) throws -> AdaptiveOptimizationContextRevision {
        try .init(value)
    }

    private func fingerprint() throws -> AppleHardwareFingerprint {
        try .init(
            platform: "ios",
            machineIdentifier: "iPhone-hardware-class",
            operatingSystemVersion: "20.0.0",
            logicalProcessorCount: 6,
            memoryClassMegabytes: 8_192
        )
    }

    private func workload() throws -> AdaptiveWorkloadDescriptor {
        try .init(identifier: "sensor-fusion", kind: .temporalFusion)
    }

    private func cpu() throws -> AdaptiveExecutionCandidate {
        try .init(
            identifier: "cpu",
            implementationRevision: "accelerate-kernel-v1",
            backend: .accelerateCPU
        )
    }

    private func metal() throws -> AdaptiveExecutionCandidate {
        try .init(
            identifier: "metal",
            implementationRevision: "metallib-sha256-v1",
            backend: .metalGPU,
            precision: .float16,
            layout: .channelsLast
        )
    }

    private func observation(
        decision: AdaptiveExecutionDecision,
        latency: Double,
        energy: Double? = nil,
        succeeded: Bool = true
    ) throws -> AdaptiveExecutionObservation {
        try .init(
            decision: decision,
            latencyMilliseconds: latency,
            relativeEnergyProxy: try energy.map(AdaptiveRelativeEnergyProxy.init),
            succeeded: succeeded
        )
    }

    @discardableResult
    private func record(
        on planner: AdaptiveExecutionPlanner,
        candidate: AdaptiveExecutionCandidate,
        latency: Double,
        energy: Double? = nil,
        succeeded: Bool = true
    ) async throws -> AdaptiveExecutionDecision {
        let decision = try await planner.select(
            workload: workload(),
            candidates: [candidate],
            runtimeState: nominalState
        )
        try await planner.record(
            observation(
                decision: decision,
                latency: latency,
                energy: energy,
                succeeded: succeeded
            )
        )
        return decision
    }
}
