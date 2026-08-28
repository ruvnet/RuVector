import CoreML
import Foundation

/// An actor-isolated session for a caller-supplied, compiled Core ML model.
///
/// This type does not authenticate the model directory. In particular, a
/// point-in-time receipt from `ModelArtifactVerifier` for a regular-file
/// distribution artifact is not provenance for an independently supplied
/// `.mlmodelc` URL.
public actor CoreMLModelSession {
    private static let maximumFeatureCount = 256
    private static let maximumFeatureNameByteCount = 1_024
    private static let maximumTensorElementCount = 16_777_216
    private static let maximumStringByteCount = 1_048_576

    public let workload: EdgeMLWorkload
    public let performanceProfile: EdgeMLPerformanceProfile
    private var model: MLModel?

    public init(
        compiledModelURL: URL,
        workload: EdgeMLWorkload = .interactiveInference,
        performanceProfile: EdgeMLPerformanceProfile = .automatic
    ) throws {
        guard compiledModelURL.isFileURL else {
            throw RuVectorEdgeMLError.invalidInput("Core ML model URL must be local")
        }
        let decision = RuntimeResourcePolicy.decision(for: workload, requestedProfile: performanceProfile)
        guard decision.permitted else {
            throw RuVectorEdgeMLError.resourceUnavailable(decision.reason ?? "Core ML is unavailable")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = RuntimeResourcePolicy.coreMLComputeUnits(
            for: workload,
            profile: decision.profile
        )
        let loadedModel = try MLModel(contentsOf: compiledModelURL, configuration: configuration)
        _ = try Self.validatedOutputNames(
            Set(loadedModel.modelDescription.outputDescriptionsByName.keys)
        )
        model = loadedModel
        self.workload = workload
        self.performanceProfile = decision.profile
    }

    public func prediction(features: [String: MLFeatureValue]) throws -> [String: MLFeatureValue] {
        try Self.validateInputFeatures(features)
        guard let model else {
            throw RuVectorEdgeMLError.resourceUnavailable("Core ML session has been unloaded")
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let result = try model.prediction(from: provider)
        let names = try Self.validatedOutputNames(result.featureNames)
        var output: [String: MLFeatureValue] = [:]
        output.reserveCapacity(names.count)
        for name in names {
            guard let value = result.featureValue(for: name) else {
                throw RuVectorEdgeMLError.invalidInput("Core ML prediction omitted a declared output")
            }
            output[name] = value
        }
        return output
    }

    public func unload() { model = nil }

    static func validateInputFeatures(_ features: [String: MLFeatureValue]) throws {
        guard features.count <= 128 else {
            throw RuVectorEdgeMLError.invalidInput("Core ML input count exceeds the bounded limit")
        }
        for (name, value) in features {
            guard !name.isEmpty, name.utf8.count <= maximumFeatureNameByteCount else {
                throw RuVectorEdgeMLError.invalidInput("Core ML input name exceeds the bounded limit")
            }
            guard !value.isUndefined else {
                throw RuVectorEdgeMLError.invalidInput("undefined Core ML inputs are not supported")
            }
            switch value.type {
            case .int64:
                break
            case .double:
                guard value.doubleValue.isFinite else {
                    throw RuVectorEdgeMLError.invalidInput("Core ML numeric input must be finite")
                }
            case .multiArray:
                guard let array = value.multiArrayValue, array.count <= maximumTensorElementCount else {
                    throw RuVectorEdgeMLError.invalidInput("Core ML multi-array input exceeds the element limit")
                }
            case .string:
                guard value.stringValue.utf8.count <= maximumStringByteCount else {
                    throw RuVectorEdgeMLError.invalidInput("Core ML string input exceeds the byte limit")
                }
            case .image:
                guard let buffer = value.imageBufferValue else {
                    throw RuVectorEdgeMLError.invalidInput("Core ML image input is unavailable")
                }
                let (pixels, overflow) = CVPixelBufferGetWidth(buffer)
                    .multipliedReportingOverflow(by: CVPixelBufferGetHeight(buffer))
                if overflow || pixels > maximumTensorElementCount {
                    throw RuVectorEdgeMLError.invalidInput("Core ML image input exceeds the pixel limit")
                }
            default:
                throw RuVectorEdgeMLError.invalidInput(
                    "Core ML input type \(value.type.rawValue) is not supported by this bounded session"
                )
            }
        }
    }

    static func validatedOutputNames(_ names: Set<String>) throws -> [String] {
        guard names.count <= maximumFeatureCount else {
            throw RuVectorEdgeMLError.invalidInput("Core ML output count exceeds the bounded limit")
        }
        guard names.allSatisfy({ !$0.isEmpty && $0.utf8.count <= maximumFeatureNameByteCount }) else {
            throw RuVectorEdgeMLError.invalidInput("Core ML output name exceeds the bounded limit")
        }
        return names.sorted()
    }
}
