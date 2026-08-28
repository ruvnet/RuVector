import Darwin
import Foundation

/// A non-unique hardware/runtime class used to scope local cost observations.
///
/// It intentionally excludes serial numbers, advertising identifiers, and
/// other stable user or device identifiers. A profile is invalidated when any
/// field changes because costs learned on another OS or hardware class are not
/// assumed to transfer safely.
public struct AppleHardwareFingerprint: Codable, Equatable, Hashable, Sendable {
    public static let plannerSchemaVersion: UInt16 = 1

    public let schemaVersion: UInt16
    public let platform: String
    public let machineIdentifier: String
    public let operatingSystemVersion: String
    public let logicalProcessorCount: Int
    public let memoryClassMegabytes: Int

    public init(
        schemaVersion: UInt16 = Self.plannerSchemaVersion,
        platform: String,
        machineIdentifier: String,
        operatingSystemVersion: String,
        logicalProcessorCount: Int,
        memoryClassMegabytes: Int
    ) throws {
        guard schemaVersion == Self.plannerSchemaVersion,
              Self.validBoundedText(platform),
              Self.validBoundedText(machineIdentifier),
              Self.validBoundedText(operatingSystemVersion),
              (1...1_024).contains(logicalProcessorCount),
              (128...1_048_576).contains(memoryClassMegabytes) else {
            throw RuVectorEdgeMLError.invalidConfiguration(
                "Apple hardware fingerprint contains an invalid or unbounded field"
            )
        }
        self.schemaVersion = schemaVersion
        self.platform = platform
        self.machineIdentifier = machineIdentifier
        self.operatingSystemVersion = operatingSystemVersion
        self.logicalProcessorCount = logicalProcessorCount
        self.memoryClassMegabytes = memoryClassMegabytes
    }

    public static func current() -> Self {
        let process = ProcessInfo.processInfo
        let version = process.operatingSystemVersion
        let memoryMiB = max(128, Int(process.physicalMemory / 1_048_576))
        let memoryClass = min(1_048_576, max(128, (memoryMiB / 512) * 512))
        let machine = boundedTextOrFallback(
            sysctlString("hw.machine"),
            fallback: "unknown-apple-machine"
        )
        let osVersion = boundedTextOrFallback(
            "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)",
            fallback: "unknown-apple-os"
        )
        return (try? .init(
            platform: currentPlatform,
            machineIdentifier: machine,
            operatingSystemVersion: osVersion,
            logicalProcessorCount: min(1_024, max(1, process.processorCount)),
            memoryClassMegabytes: memoryClass
        )) ?? Self(
            validatedPlatform: "unknown-apple-platform",
            machineIdentifier: "unknown-apple-machine",
            operatingSystemVersion: "unknown-apple-os",
            logicalProcessorCount: 1,
            memoryClassMegabytes: 128
        )
    }

    static func validate(_ fingerprint: Self) throws {
        _ = try Self(
            schemaVersion: fingerprint.schemaVersion,
            platform: fingerprint.platform,
            machineIdentifier: fingerprint.machineIdentifier,
            operatingSystemVersion: fingerprint.operatingSystemVersion,
            logicalProcessorCount: fingerprint.logicalProcessorCount,
            memoryClassMegabytes: fingerprint.memoryClassMegabytes
        )
    }

    private static var currentPlatform: String {
        #if targetEnvironment(simulator)
        return "apple-simulator"
        #elseif os(iOS)
        return "ios"
        #elseif os(macOS)
        return "macos"
        #else
        return "unknown-apple-platform"
        #endif
    }

    private static func validBoundedText(_ value: String) -> Bool {
        value == value.trimmingCharacters(in: .whitespacesAndNewlines)
            && !value.isEmpty
            && value.utf8.count <= 128
    }

    private static func boundedTextOrFallback(_ value: String?, fallback: String) -> String {
        guard let value, validBoundedText(value) else { return fallback }
        return value
    }

    private init(
        validatedPlatform: String,
        machineIdentifier: String,
        operatingSystemVersion: String,
        logicalProcessorCount: Int,
        memoryClassMegabytes: Int
    ) {
        schemaVersion = Self.plannerSchemaVersion
        platform = validatedPlatform
        self.machineIdentifier = machineIdentifier
        self.operatingSystemVersion = operatingSystemVersion
        self.logicalProcessorCount = logicalProcessorCount
        self.memoryClassMegabytes = memoryClassMegabytes
    }

    private static func sysctlString(_ key: String) -> String? {
        key.withCString { keyPointer in
            var size = 0
            guard sysctlbyname(keyPointer, nil, &size, nil, 0) == 0,
                  size > 1, size <= 1_024 else { return nil }
            var value = [CChar](repeating: 0, count: size)
            let result = value.withUnsafeMutableBufferPointer { buffer in
                sysctlbyname(keyPointer, buffer.baseAddress, &size, nil, 0)
            }
            guard result == 0 else { return nil }
            return String(cString: value)
        }
    }
}
