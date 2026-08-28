import XCTest
@testable import RuVectorAppleML

final class TemporalModelTests: XCTestCase {
    func testShapeRejectsUnboundedDimensions() throws {
        XCTAssertThrowsError(try TemporalModelShape(windowLength: 0, inputWidth: 2, hiddenWidth: 2, outputWidth: 1))
        XCTAssertThrowsError(try TemporalModelShape(windowLength: 256, inputWidth: 4_096, hiddenWidth: 2, outputWidth: 1))
    }

    func testPredictorNormalizesAndProjects() throws {
        let shape = try TemporalModelShape(windowLength: 2, inputWidth: 2, hiddenWidth: 2, outputWidth: 1)
        let model = try TemporalProjectionModel(
            shape: shape,
            featureMean: [1, 2],
            featureStandardDeviation: [2, 4],
            temporalWeights: [
                1, 0,
                0, 1,
                1, 0,
                0, 1,
            ],
            temporalBias: [0, 0],
            projectionWeights: [2, 3],
            projectionBias: [1]
        )
        let prediction = try AccelerateTemporalPredictor(model: model).predict(window: [3, 6, 5, 10])
        XCTAssertEqual(prediction[0], 16, accuracy: 0.0001)
    }

    func testPredictorRejectsNonFiniteInput() throws {
        let model = try identityModel()
        XCTAssertThrowsError(try AccelerateTemporalPredictor(model: model).predict(window: [.nan]))
    }

    func testPredictorMatchesScalarReferenceAcrossNegativeReLUPath() throws {
        let shape = try TemporalModelShape(windowLength: 2, inputWidth: 2, hiddenWidth: 2, outputWidth: 2)
        let model = try TemporalProjectionModel(
            shape: shape,
            featureMean: [0.5, -0.5],
            featureStandardDeviation: [2, 0.5],
            temporalWeights: [
                -2, 1,
                1, -3,
                0.5, 2,
                -1, 0.25,
            ],
            temporalBias: [-0.25, 0.5],
            projectionWeights: [1.5, -0.5, -2, 3],
            projectionBias: [0.75, -1.25]
        )
        let window: [Float] = [-3, 0.5, 2.5, -1]
        let actual = try AccelerateTemporalPredictor(model: model).predict(window: window)
        let expected = scalarPrediction(model: model, window: window)
        XCTAssertEqual(actual.count, expected.count)
        for index in actual.indices {
            XCTAssertEqual(actual[index], expected[index], accuracy: 0.000_01)
        }
    }

    func testDecodingRevalidatesTensorShapes() throws {
        let invalid = """
        {"schema":"ruvector.apple.temporal-projection.v1","shape":{"windowLength":1,"inputWidth":1,"hiddenWidth":1,"outputWidth":1},"featureMean":[],"featureStandardDeviation":[1],"temporalWeights":[1],"temporalBias":[0],"projectionWeights":[1],"projectionBias":[0]}
        """.data(using: .utf8)!
        XCTAssertThrowsError(try JSONDecoder().decode(TemporalProjectionModel.self, from: invalid))

        let unboundedShape = """
        {"windowLength":9223372036854775807,"inputWidth":9223372036854775807,"hiddenWidth":1,"outputWidth":1}
        """.data(using: .utf8)!
        XCTAssertThrowsError(try JSONDecoder().decode(TemporalModelShape.self, from: unboundedShape))
    }

    func testConcurrentPredictionsRemainDeterministic() throws {
        let predictor = AccelerateTemporalPredictor(model: try identityModel())
        let expected = try predictor.predict(window: [3])
        let lock = NSLock()
        var failures = 0
        DispatchQueue.concurrentPerform(iterations: 100) { _ in
            let actual = try? predictor.predict(window: [3])
            if actual != expected {
                lock.lock()
                failures += 1
                lock.unlock()
            }
        }
        XCTAssertEqual(failures, 0)
    }

    private func identityModel() throws -> TemporalProjectionModel {
        let shape = try TemporalModelShape(windowLength: 1, inputWidth: 1, hiddenWidth: 1, outputWidth: 1)
        return try .init(
            shape: shape, featureMean: [0], featureStandardDeviation: [1],
            temporalWeights: [1], temporalBias: [0], projectionWeights: [1], projectionBias: [0]
        )
    }


    private func scalarPrediction(model: TemporalProjectionModel, window: [Float]) -> [Float] {
        let normalized = window.enumerated().map { index, value in
            let feature = index % model.shape.inputWidth
            return (value - model.featureMean[feature]) / model.featureStandardDeviation[feature]
        }
        var hidden = model.temporalBias
        for input in normalized.indices {
            for channel in hidden.indices {
                hidden[channel] += normalized[input]
                    * model.temporalWeights[input * model.shape.hiddenWidth + channel]
            }
        }
        hidden = hidden.map { max(0, $0) }
        var output = model.projectionBias
        for channel in hidden.indices {
            for coordinate in output.indices {
                output[coordinate] += hidden[channel]
                    * model.projectionWeights[channel * model.shape.outputWidth + coordinate]
            }
        }
        return output
    }
}
