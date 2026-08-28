import Foundation

public struct TemporalTrainingBatch: Sendable {
    public let windows: [[Float]]
    public let targets: [[Float]]
    public let masks: [[Float]]

    public init(windows: [[Float]], targets: [[Float]], masks: [[Float]]? = nil) {
        self.windows = windows
        self.targets = targets
        self.masks = masks ?? targets.map { [Float](repeating: 1, count: $0.count) }
    }

    func validate(shape: TemporalModelShape) throws {
        let (inputElements, inputOverflow) = windows.count.multipliedReportingOverflow(by: shape.inputCount)
        let (outputElements, outputOverflow) = targets.count.multipliedReportingOverflow(by: shape.outputWidth)
        guard (1...4_096).contains(windows.count), targets.count == windows.count,
              masks.count == windows.count, !inputOverflow, !outputOverflow,
              inputElements <= 8_388_608, outputElements <= 4_194_304 else {
            throw RuVectorEdgeMLError.invalidInput("training batch must contain 1...4096 aligned samples")
        }
        for index in windows.indices {
            guard windows[index].count == shape.inputCount,
                  targets[index].count == shape.outputWidth,
                  masks[index].count == shape.outputWidth else {
                throw RuVectorEdgeMLError.invalidInput("training tensors do not match the declared shape")
            }
            guard windows[index].allSatisfy({ $0.isFinite && abs($0) <= 1_000_000 }),
                  targets[index].allSatisfy({ $0.isFinite && abs($0) <= 1_000_000 }),
                  masks[index].allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
                throw RuVectorEdgeMLError.invalidInput("training values must be finite and bounded")
            }
        }
        guard masks.joined().contains(where: { $0 > 0 }) else {
            throw RuVectorEdgeMLError.invalidInput("at least one training target must be enabled")
        }
    }
}

public struct TemporalTrainingOptions: Sendable {
    public let epochs: Int
    public let learningRate: Float
    public let l2Regularization: Float
    public let maximumDurationSeconds: Double
    public let deterministicSeed: UInt64
    public let requirePhysicalDevice: Bool

    public init(
        epochs: Int = 16,
        learningRate: Float = 0.002,
        l2Regularization: Float = 0.0001,
        maximumDurationSeconds: Double = 60,
        deterministicSeed: UInt64 = 0x5255564543544F52,
        requirePhysicalDevice: Bool = true
    ) throws {
        guard (1...128).contains(epochs) else {
            throw RuVectorEdgeMLError.invalidConfiguration("epochs must be between 1 and 128")
        }
        guard learningRate.isFinite, (0.00001...0.1).contains(learningRate) else {
            throw RuVectorEdgeMLError.invalidConfiguration("learningRate must be between 0.00001 and 0.1")
        }
        guard l2Regularization.isFinite, (0...0.1).contains(l2Regularization) else {
            throw RuVectorEdgeMLError.invalidConfiguration("l2Regularization must be between 0 and 0.1")
        }
        guard maximumDurationSeconds.isFinite, (1...600).contains(maximumDurationSeconds) else {
            throw RuVectorEdgeMLError.invalidConfiguration("maximumDurationSeconds must be between 1 and 600")
        }
        self.epochs = epochs
        self.learningRate = learningRate
        self.l2Regularization = l2Regularization
        self.maximumDurationSeconds = maximumDurationSeconds
        self.deterministicSeed = deterministicSeed
        self.requirePhysicalDevice = requirePhysicalDevice
    }
}

public struct TemporalTrainingProgress: Sendable {
    public let epoch: Int
    public let epochCount: Int
    public let loss: Float
}

public struct TemporalTrainingReport: Sendable {
    public let model: TemporalProjectionModel
    public let backend: String
    public let deviceName: String
    public let durationSeconds: Double
    public let finalLoss: Float
    public let sampleEpochsPerSecond: Double
    public let evidence: String
}
