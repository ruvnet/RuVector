import Accelerate
import Foundation

public final class AccelerateTemporalPredictor: @unchecked Sendable {
    public let model: TemporalProjectionModel
    private let lock = NSLock()
    private var normalized: [Float]
    private var hidden: [Float]
    private var output: [Float]

    public init(model: TemporalProjectionModel) {
        self.model = model
        normalized = .init(repeating: 0, count: model.shape.inputCount)
        hidden = .init(repeating: 0, count: model.shape.hiddenWidth)
        output = .init(repeating: 0, count: model.shape.outputWidth)
    }

    public func predict(window: [Float]) throws -> [Float] {
        guard window.count == model.shape.inputCount else {
            throw RuVectorEdgeMLError.invalidInput("window does not match model input shape")
        }
        guard window.allSatisfy(\.isFinite) else {
            throw RuVectorEdgeMLError.invalidInput("window values must be finite")
        }
        lock.lock()
        defer { lock.unlock() }
        normalize(window)
        for index in hidden.indices { hidden[index] = model.temporalBias[index] }
        model.temporalWeights.withUnsafeBufferPointer { weights in
            normalized.withUnsafeBufferPointer { input in
                hidden.withUnsafeMutableBufferPointer { result in
                    cblas_sgemv(
                        CblasRowMajor, CblasTrans,
                        Int32(model.shape.inputCount), Int32(model.shape.hiddenWidth),
                        1, weights.baseAddress!, Int32(model.shape.hiddenWidth),
                        input.baseAddress!, 1, 1, result.baseAddress!, 1
                    )
                }
            }
        }
        vDSP_vthres(hidden, 1, [0], &hidden, 1, vDSP_Length(hidden.count))
        for index in output.indices { output[index] = model.projectionBias[index] }
        model.projectionWeights.withUnsafeBufferPointer { weights in
            hidden.withUnsafeBufferPointer { input in
                output.withUnsafeMutableBufferPointer { result in
                    cblas_sgemv(
                        CblasRowMajor, CblasTrans,
                        Int32(model.shape.hiddenWidth), Int32(model.shape.outputWidth),
                        1, weights.baseAddress!, Int32(model.shape.outputWidth),
                        input.baseAddress!, 1, 1, result.baseAddress!, 1
                    )
                }
            }
        }
        guard output.allSatisfy(\.isFinite) else { throw RuVectorEdgeMLError.numericalFailure }
        return output
    }

    private func normalize(_ window: [Float]) {
        for frame in 0..<model.shape.windowLength {
            let offset = frame * model.shape.inputWidth
            for feature in 0..<model.shape.inputWidth {
                let index = offset + feature
                normalized[index] = (window[index] - model.featureMean[feature])
                    / model.featureStandardDeviation[feature]
            }
        }
    }
}
