import Foundation

public struct TemporalModelShape: Codable, Equatable, Sendable {
    public let windowLength: Int
    public let inputWidth: Int
    public let hiddenWidth: Int
    public let outputWidth: Int

    public init(windowLength: Int, inputWidth: Int, hiddenWidth: Int, outputWidth: Int) throws {
        guard (1...256).contains(windowLength) else {
            throw RuVectorEdgeMLError.invalidConfiguration("windowLength must be between 1 and 256")
        }
        guard (1...4_096).contains(inputWidth) else {
            throw RuVectorEdgeMLError.invalidConfiguration("inputWidth must be between 1 and 4096")
        }
        guard (1...1_024).contains(hiddenWidth) else {
            throw RuVectorEdgeMLError.invalidConfiguration("hiddenWidth must be between 1 and 1024")
        }
        guard (1...4_096).contains(outputWidth) else {
            throw RuVectorEdgeMLError.invalidConfiguration("outputWidth must be between 1 and 4096")
        }
        let (inputCount, overflowA) = windowLength.multipliedReportingOverflow(by: inputWidth)
        let (temporalCount, overflowB) = inputCount.multipliedReportingOverflow(by: hiddenWidth)
        let (projectionCount, overflowC) = hiddenWidth.multipliedReportingOverflow(by: outputWidth)
        let (weightCount, overflowD) = temporalCount.addingReportingOverflow(projectionCount)
        guard !overflowA, !overflowB, !overflowC, !overflowD,
              inputCount <= 262_144, weightCount <= 16_777_216 else {
            throw RuVectorEdgeMLError.invalidConfiguration("model dimensions exceed the bounded element budget")
        }
        self.windowLength = windowLength
        self.inputWidth = inputWidth
        self.hiddenWidth = hiddenWidth
        self.outputWidth = outputWidth
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            windowLength: container.decode(Int.self, forKey: .windowLength),
            inputWidth: container.decode(Int.self, forKey: .inputWidth),
            hiddenWidth: container.decode(Int.self, forKey: .hiddenWidth),
            outputWidth: container.decode(Int.self, forKey: .outputWidth)
        )
    }

    public var inputCount: Int { windowLength * inputWidth }
}

public struct TemporalProjectionModel: Codable, Equatable, Sendable {
    public static let currentSchema = "ruvector.apple.temporal-projection.v1"

    public let schema: String
    public let shape: TemporalModelShape
    public let featureMean: [Float]
    public let featureStandardDeviation: [Float]
    public let temporalWeights: [Float]
    public let temporalBias: [Float]
    public let projectionWeights: [Float]
    public let projectionBias: [Float]

    public init(
        shape: TemporalModelShape,
        featureMean: [Float],
        featureStandardDeviation: [Float],
        temporalWeights: [Float],
        temporalBias: [Float],
        projectionWeights: [Float],
        projectionBias: [Float]
    ) throws {
        try Self.validate(
            schema: Self.currentSchema,
            shape: shape,
            featureMean: featureMean,
            featureStandardDeviation: featureStandardDeviation,
            temporalWeights: temporalWeights,
            temporalBias: temporalBias,
            projectionWeights: projectionWeights,
            projectionBias: projectionBias
        )
        self.schema = Self.currentSchema
        self.shape = shape
        self.featureMean = featureMean
        self.featureStandardDeviation = featureStandardDeviation
        self.temporalWeights = temporalWeights
        self.temporalBias = temporalBias
        self.projectionWeights = projectionWeights
        self.projectionBias = projectionBias
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let schema = try container.decode(String.self, forKey: .schema)
        let shape = try container.decode(TemporalModelShape.self, forKey: .shape)
        let featureMean = try container.decode([Float].self, forKey: .featureMean)
        let featureStandardDeviation = try container.decode([Float].self, forKey: .featureStandardDeviation)
        let temporalWeights = try container.decode([Float].self, forKey: .temporalWeights)
        let temporalBias = try container.decode([Float].self, forKey: .temporalBias)
        let projectionWeights = try container.decode([Float].self, forKey: .projectionWeights)
        let projectionBias = try container.decode([Float].self, forKey: .projectionBias)
        try Self.validate(
            schema: schema,
            shape: shape,
            featureMean: featureMean,
            featureStandardDeviation: featureStandardDeviation,
            temporalWeights: temporalWeights,
            temporalBias: temporalBias,
            projectionWeights: projectionWeights,
            projectionBias: projectionBias
        )
        self.schema = schema
        self.shape = shape
        self.featureMean = featureMean
        self.featureStandardDeviation = featureStandardDeviation
        self.temporalWeights = temporalWeights
        self.temporalBias = temporalBias
        self.projectionWeights = projectionWeights
        self.projectionBias = projectionBias
    }

    private static func validate(
        schema: String,
        shape: TemporalModelShape,
        featureMean: [Float],
        featureStandardDeviation: [Float],
        temporalWeights: [Float],
        temporalBias: [Float],
        projectionWeights: [Float],
        projectionBias: [Float]
    ) throws {
        guard schema == currentSchema else {
            throw RuVectorEdgeMLError.modelShapeMismatch("unsupported model schema")
        }
        guard featureMean.count == shape.inputWidth,
              featureStandardDeviation.count == shape.inputWidth,
              temporalWeights.count == shape.inputCount * shape.hiddenWidth,
              temporalBias.count == shape.hiddenWidth,
              projectionWeights.count == shape.hiddenWidth * shape.outputWidth,
              projectionBias.count == shape.outputWidth else {
            throw RuVectorEdgeMLError.modelShapeMismatch("model tensors do not match the declared shape")
        }
        guard featureMean.allSatisfy(\.isFinite),
              featureStandardDeviation.allSatisfy(\.isFinite),
              temporalWeights.allSatisfy(\.isFinite), temporalBias.allSatisfy(\.isFinite),
              projectionWeights.allSatisfy(\.isFinite), projectionBias.allSatisfy(\.isFinite),
              featureStandardDeviation.allSatisfy({ $0 > 0 }) else {
            throw RuVectorEdgeMLError.modelShapeMismatch("model tensors must be finite and standard deviations positive")
        }
    }
}
