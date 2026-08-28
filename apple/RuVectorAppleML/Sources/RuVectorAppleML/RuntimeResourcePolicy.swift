import CoreML
import Foundation

public enum EdgeMLWorkload: String, Codable, Sendable {
    case interactiveInference
    case backgroundInference
    case training
}

public enum EdgeMLPerformanceProfile: String, Codable, Sendable {
    case automatic
    case efficiency
    case balanced
    case performance
}

public struct EdgeMLResourceDecision: Equatable, Sendable {
    public let permitted: Bool
    public let profile: EdgeMLPerformanceProfile
    public let reason: String?
    public let thermalState: String
    public let lowPowerModeEnabled: Bool
    public let simulator: Bool
}

public enum RuntimeResourcePolicy {
    public static func decision(
        for workload: EdgeMLWorkload,
        requestedProfile: EdgeMLPerformanceProfile = .automatic,
        allowSimulatorTraining: Bool = false
    ) -> EdgeMLResourceDecision {
        let info = ProcessInfo.processInfo
        let thermalState = thermalLabel(info.thermalState)
        let lowPower = info.isLowPowerModeEnabled
        #if targetEnvironment(simulator)
        let simulator = true
        #else
        let simulator = false
        #endif

        if workload == .training && simulator && !allowSimulatorTraining {
            return .init(permitted: false, profile: .efficiency,
                         reason: "Physical Apple hardware is required for governed training.",
                         thermalState: thermalState, lowPowerModeEnabled: lowPower, simulator: simulator)
        }
        if matchesUnsafeThermalState(info.thermalState) {
            return .init(permitted: false, profile: .efficiency,
                         reason: "The device thermal state is \(thermalState).",
                         thermalState: thermalState, lowPowerModeEnabled: lowPower, simulator: simulator)
        }
        if workload == .training && lowPower {
            return .init(permitted: false, profile: .efficiency,
                         reason: "Low Power Mode blocks on-device training.",
                         thermalState: thermalState, lowPowerModeEnabled: lowPower, simulator: simulator)
        }
        let resolved: EdgeMLPerformanceProfile
        if lowPower || info.thermalState == .fair {
            resolved = .efficiency
        } else if requestedProfile == .automatic {
            resolved = workload == .interactiveInference ? .performance : .balanced
        } else {
            resolved = requestedProfile
        }
        return .init(permitted: true, profile: resolved, reason: nil,
                     thermalState: thermalState, lowPowerModeEnabled: lowPower, simulator: simulator)
    }

    public static func coreMLComputeUnits(
        for workload: EdgeMLWorkload,
        profile: EdgeMLPerformanceProfile = .automatic
    ) -> MLComputeUnits {
        let decision = decision(for: workload, requestedProfile: profile)
        if workload == .backgroundInference || decision.profile == .efficiency {
            return .cpuOnly
        }
        return .all
    }

    private static func thermalLabel(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private static func matchesUnsafeThermalState(_ state: ProcessInfo.ThermalState) -> Bool {
        switch state {
        case .nominal, .fair: return false
        case .serious, .critical: return true
        @unknown default: return true
        }
    }
}
