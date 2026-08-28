import XCTest
@testable import RuVectorAppleML

final class MPSGraphTrainerTests: XCTestCase {
    func testSmallTrainingRunProducesUsableFiniteModel() throws {
        let shape = try TemporalModelShape(windowLength: 1, inputWidth: 2, hiddenWidth: 4, outputWidth: 1)
        let batch = TemporalTrainingBatch(
            windows: [[0, 0], [0, 1], [1, 0], [1, 1]],
            targets: [[0], [1], [1], [2]]
        )
        let options = try TemporalTrainingOptions(
            epochs: 8, learningRate: 0.02, l2Regularization: 0,
            maximumDurationSeconds: 30, requirePhysicalDevice: false
        )
        let report = try MPSGraphTemporalTrainer().train(batch: batch, shape: shape, options: options)
        XCTAssertTrue(report.finalLoss.isFinite)
        XCTAssertEqual(report.backend, "apple-mpsgraph-metal")
        XCTAssertEqual(report.evidence, "MEASURED_ON_CURRENT_RUNTIME")
        let prediction = try AccelerateTemporalPredictor(model: report.model).predict(window: [1, 1])
        XCTAssertEqual(prediction.count, 1)
        XCTAssertTrue(prediction[0].isFinite)

        let predictor = AccelerateTemporalPredictor(model: report.model)
        let squaredErrors = try zip(batch.windows, batch.targets).map { window, target in
            let difference = try predictor.predict(window: window)[0] - target[0]
            return difference * difference
        }
        let returnedModelLoss = squaredErrors.reduce(0, +) / Float(squaredErrors.count)
        XCTAssertEqual(report.finalLoss, returnedModelLoss, accuracy: max(0.000_01, returnedModelLoss * 0.000_1))
    }

    func testCancellationStopsBeforeFirstEpoch() throws {
        let shape = try TemporalModelShape(windowLength: 1, inputWidth: 1, hiddenWidth: 1, outputWidth: 1)
        let options = try TemporalTrainingOptions(epochs: 1, maximumDurationSeconds: 5, requirePhysicalDevice: false)
        XCTAssertThrowsError(try MPSGraphTemporalTrainer().train(
            batch: .init(windows: [[0]], targets: [[0]]),
            shape: shape,
            options: options,
            isCancelled: { true }
        )) { error in
            XCTAssertEqual(error as? RuVectorEdgeMLError, .cancelled)
        }
    }
}
