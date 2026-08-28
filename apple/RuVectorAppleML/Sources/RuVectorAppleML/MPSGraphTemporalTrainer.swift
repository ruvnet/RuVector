import Foundation
import Metal
import MetalPerformanceShadersGraph

public final class MPSGraphTemporalTrainer: @unchecked Sendable {
    public init() {}

    public func train(
        batch: TemporalTrainingBatch,
        shape: TemporalModelShape,
        options: TemporalTrainingOptions,
        progress: (@Sendable (TemporalTrainingProgress) -> Void)? = nil,
        isCancelled: (@Sendable () -> Bool)? = nil
    ) throws -> TemporalTrainingReport {
        try batch.validate(shape: shape)
        let initialDecision = RuntimeResourcePolicy.decision(
            for: .training,
            allowSimulatorTraining: !options.requirePhysicalDevice
        )
        guard initialDecision.permitted else {
            throw RuVectorEdgeMLError.resourceUnavailable(initialDecision.reason ?? "training is unavailable")
        }
        guard let metalDevice = MTLCreateSystemDefaultDevice(),
              let commandQueue = metalDevice.makeCommandQueue() else {
            throw RuVectorEdgeMLError.resourceUnavailable("Metal command queue is unavailable")
        }

        let started = ProcessInfo.processInfo.systemUptime
        let statistics = featureStatistics(windows: batch.windows, shape: shape)
        let normalizedWindows = normalize(batch.windows, shape: shape, statistics: statistics)
        let graphState = try makeGraph(sampleCount: batch.windows.count, shape: shape, options: options)
        let graphDevice = MPSGraphDevice(mtlDevice: metalDevice)
        let inputData = tensorData(normalizedWindows.flatMap { $0 }, shape: graphState.inputShape, device: graphDevice)
        let targetData = tensorData(batch.targets.flatMap { $0 }, shape: graphState.outputShape, device: graphDevice)
        let maskData = tensorData(batch.masks.flatMap { $0 }, shape: graphState.outputShape, device: graphDevice)
        var generator = DeterministicGenerator(seed: options.deterministicSeed)
        var temporal = generator.values(count: shape.inputCount * shape.hiddenWidth, scale: 0.08)
        var temporalBias = [Float](repeating: 0, count: shape.hiddenWidth)
        var projection = generator.values(count: shape.hiddenWidth * shape.outputWidth, scale: 0.08)
        var projectionBias = initialProjectionBias(targets: batch.targets, masks: batch.masks, width: shape.outputWidth)
        var latestTrainingLoss = Float.nan

        for epoch in 0..<options.epochs {
            if isCancelled?() == true { throw RuVectorEdgeMLError.cancelled }
            guard ProcessInfo.processInfo.systemUptime - started <= options.maximumDurationSeconds else {
                throw RuVectorEdgeMLError.timeLimitExceeded
            }
            if epoch.isMultiple(of: 2) {
                let decision = RuntimeResourcePolicy.decision(
                    for: .training,
                    allowSimulatorTraining: !options.requirePhysicalDevice
                )
                guard decision.permitted else {
                    throw RuVectorEdgeMLError.resourceUnavailable(decision.reason ?? "training was paused")
                }
            }
            let result = try autoreleasepool {
                try runEpoch(
                    graphState: graphState,
                    commandQueue: commandQueue,
                    graphDevice: graphDevice,
                    inputData: inputData,
                    targetData: targetData,
                    maskData: maskData,
                    temporal: temporal,
                    temporalBias: temporalBias,
                    projection: projection,
                    projectionBias: projectionBias
                )
            }
            temporal = result.temporal
            temporalBias = result.temporalBias
            projection = result.projection
            projectionBias = result.projectionBias
            latestTrainingLoss = result.loss
            guard latestTrainingLoss.isFinite, temporal.allSatisfy(\.isFinite), temporalBias.allSatisfy(\.isFinite),
                  projection.allSatisfy(\.isFinite), projectionBias.allSatisfy(\.isFinite) else {
                throw RuVectorEdgeMLError.numericalFailure
            }
            guard ProcessInfo.processInfo.systemUptime - started <= options.maximumDurationSeconds else {
                throw RuVectorEdgeMLError.timeLimitExceeded
            }
            progress?(.init(epoch: epoch + 1, epochCount: options.epochs, loss: latestTrainingLoss))
        }

        let finalLoss = try autoreleasepool {
            try evaluateLoss(
                graphState: graphState,
                commandQueue: commandQueue,
                graphDevice: graphDevice,
                inputData: inputData,
                targetData: targetData,
                maskData: maskData,
                temporal: temporal,
                temporalBias: temporalBias,
                projection: projection,
                projectionBias: projectionBias
            )
        }
        guard finalLoss.isFinite else { throw RuVectorEdgeMLError.numericalFailure }
        let duration = ProcessInfo.processInfo.systemUptime - started
        guard duration <= options.maximumDurationSeconds else {
            throw RuVectorEdgeMLError.timeLimitExceeded
        }
        let model = try TemporalProjectionModel(
            shape: shape,
            featureMean: statistics.mean,
            featureStandardDeviation: statistics.standardDeviation,
            temporalWeights: temporal,
            temporalBias: temporalBias,
            projectionWeights: projection,
            projectionBias: projectionBias
        )
        return .init(
            model: model,
            backend: "apple-mpsgraph-metal",
            deviceName: metalDevice.name,
            durationSeconds: duration,
            finalLoss: finalLoss,
            sampleEpochsPerSecond: Double(batch.windows.count * options.epochs) / max(duration, 0.000_001),
            evidence: "MEASURED_ON_CURRENT_RUNTIME"
        )
    }
}

private struct GraphState {
    let graph: MPSGraph
    let input: MPSGraphTensor
    let target: MPSGraphTensor
    let mask: MPSGraphTensor
    let parameters: [MPSGraphTensor]
    let updated: [MPSGraphTensor]
    let loss: MPSGraphTensor
    let inputShape: [NSNumber]
    let outputShape: [NSNumber]
    let temporalShape: [NSNumber]
    let temporalBiasShape: [NSNumber]
    let projectionShape: [NSNumber]
    let projectionBiasShape: [NSNumber]
}

private struct EpochResult {
    let temporal: [Float]
    let temporalBias: [Float]
    let projection: [Float]
    let projectionBias: [Float]
    let loss: Float
}

private extension MPSGraphTemporalTrainer {
    func makeGraph(
        sampleCount: Int,
        shape: TemporalModelShape,
        options: TemporalTrainingOptions
    ) throws -> GraphState {
        let graph = MPSGraph()
        let inputShape = numbers([sampleCount, shape.windowLength, shape.inputWidth, 1])
        let outputShape = numbers([sampleCount, shape.outputWidth])
        let temporalShape = numbers([shape.windowLength, shape.inputWidth, 1, shape.hiddenWidth])
        let temporalBiasShape = numbers([1, 1, 1, shape.hiddenWidth])
        let projectionShape = numbers([shape.hiddenWidth, shape.outputWidth])
        let projectionBiasShape = numbers([1, shape.outputWidth])
        let input = graph.placeholder(shape: inputShape, dataType: .float32, name: "input")
        let target = graph.placeholder(shape: outputShape, dataType: .float32, name: "target")
        let mask = graph.placeholder(shape: outputShape, dataType: .float32, name: "mask")
        let temporal = graph.placeholder(shape: temporalShape, dataType: .float32, name: "temporal_weights")
        let temporalBias = graph.placeholder(shape: temporalBiasShape, dataType: .float32, name: "temporal_bias")
        let projection = graph.placeholder(shape: projectionShape, dataType: .float32, name: "projection_weights")
        let projectionBias = graph.placeholder(shape: projectionBiasShape, dataType: .float32, name: "projection_bias")
        guard let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1,
            groups: 1, paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0,
            paddingStyle: .explicit, dataLayout: .NHWC, weightsLayout: .HWIO
        ) else { throw RuVectorEdgeMLError.resourceUnavailable("MPSGraph convolution is unavailable") }
        let convolved = graph.convolution2D(input, weights: temporal, descriptor: descriptor, name: "temporal_projection")
        let hidden = graph.reLU(with: graph.addition(convolved, temporalBias, name: nil), name: "relu")
        let flattened = graph.reshape(hidden, shape: numbers([sampleCount, shape.hiddenWidth]), name: "flatten")
        let prediction = graph.addition(
            graph.matrixMultiplication(primary: flattened, secondary: projection, name: nil),
            projectionBias,
            name: "prediction"
        )
        let squared = graph.square(with: graph.subtraction(prediction, target, name: nil), name: nil)
        let masked = graph.multiplication(squared, mask, name: nil)
        let maskTotal = graph.reductionSum(with: mask, axes: nil, name: nil)
        let minimum = graph.constant(1.0, dataType: .float32)
        let denominator = graph.maximum(maskTotal, minimum, name: nil)
        var loss = graph.division(graph.reductionSum(with: masked, axes: nil, name: nil), denominator, name: "masked_mse")
        let parameters = [temporal, temporalBias, projection, projectionBias]
        if options.l2Regularization > 0 {
            let l2 = graph.addition(
                graph.reductionSum(with: graph.square(with: temporal, name: nil), axes: nil, name: nil),
                graph.reductionSum(with: graph.square(with: projection, name: nil), axes: nil, name: nil),
                name: nil
            )
            loss = graph.addition(
                loss,
                graph.multiplication(l2, graph.constant(Double(options.l2Regularization), dataType: .float32), name: nil),
                name: "regularized_loss"
            )
        }
        let gradients = graph.gradients(of: loss, with: parameters, name: "gradients")
        let rate = graph.constant(Double(options.learningRate), dataType: .float32)
        let updated = parameters.map {
            graph.stochasticGradientDescent(learningRate: rate, values: $0, gradient: gradients[$0]!, name: nil)
        }
        return .init(
            graph: graph, input: input, target: target, mask: mask, parameters: parameters,
            updated: updated, loss: loss, inputShape: inputShape, outputShape: outputShape,
            temporalShape: temporalShape, temporalBiasShape: temporalBiasShape,
            projectionShape: projectionShape, projectionBiasShape: projectionBiasShape
        )
    }

    func runEpoch(
        graphState state: GraphState,
        commandQueue: MTLCommandQueue,
        graphDevice: MPSGraphDevice,
        inputData: MPSGraphTensorData,
        targetData: MPSGraphTensorData,
        maskData: MPSGraphTensorData,
        temporal: [Float],
        temporalBias: [Float],
        projection: [Float],
        projectionBias: [Float]
    ) throws -> EpochResult {
        let feeds = feeds(
            state: state, graphDevice: graphDevice, inputData: inputData,
            targetData: targetData, maskData: maskData, temporal: temporal,
            temporalBias: temporalBias, projection: projection, projectionBias: projectionBias
        )
        let results = state.graph.run(
            with: commandQueue,
            feeds: feeds,
            targetTensors: state.updated + [state.loss],
            targetOperations: nil
        )
        guard let temporalData = results[state.updated[0]],
              let temporalBiasData = results[state.updated[1]],
              let projectionData = results[state.updated[2]],
              let projectionBiasData = results[state.updated[3]],
              let lossData = results[state.loss] else { throw RuVectorEdgeMLError.numericalFailure }
        return .init(
            temporal: floats(temporalData, count: temporal.count),
            temporalBias: floats(temporalBiasData, count: temporalBias.count),
            projection: floats(projectionData, count: projection.count),
            projectionBias: floats(projectionBiasData, count: projectionBias.count),
            loss: floats(lossData, count: 1).first ?? .nan
        )
    }

    func evaluateLoss(
        graphState state: GraphState,
        commandQueue: MTLCommandQueue,
        graphDevice: MPSGraphDevice,
        inputData: MPSGraphTensorData,
        targetData: MPSGraphTensorData,
        maskData: MPSGraphTensorData,
        temporal: [Float],
        temporalBias: [Float],
        projection: [Float],
        projectionBias: [Float]
    ) throws -> Float {
        let values = state.graph.run(
            with: commandQueue,
            feeds: feeds(
                state: state, graphDevice: graphDevice, inputData: inputData,
                targetData: targetData, maskData: maskData, temporal: temporal,
                temporalBias: temporalBias, projection: projection, projectionBias: projectionBias
            ),
            targetTensors: [state.loss],
            targetOperations: nil
        )
        guard let lossData = values[state.loss] else { throw RuVectorEdgeMLError.numericalFailure }
        return floats(lossData, count: 1).first ?? .nan
    }

    func feeds(
        state: GraphState,
        graphDevice: MPSGraphDevice,
        inputData: MPSGraphTensorData,
        targetData: MPSGraphTensorData,
        maskData: MPSGraphTensorData,
        temporal: [Float],
        temporalBias: [Float],
        projection: [Float],
        projectionBias: [Float]
    ) -> [MPSGraphTensor: MPSGraphTensorData] {
        [
            state.input: inputData,
            state.target: targetData,
            state.mask: maskData,
            state.parameters[0]: tensorData(temporal, shape: state.temporalShape, device: graphDevice),
            state.parameters[1]: tensorData(temporalBias, shape: state.temporalBiasShape, device: graphDevice),
            state.parameters[2]: tensorData(projection, shape: state.projectionShape, device: graphDevice),
            state.parameters[3]: tensorData(projectionBias, shape: state.projectionBiasShape, device: graphDevice),
        ]
    }
}

private struct FeatureStatistics {
    let mean: [Float]
    let standardDeviation: [Float]
}

private func featureStatistics(windows: [[Float]], shape: TemporalModelShape) -> FeatureStatistics {
    let count = Float(windows.count * shape.windowLength)
    var mean = [Float](repeating: 0, count: shape.inputWidth)
    for window in windows {
        for frame in 0..<shape.windowLength {
            for feature in 0..<shape.inputWidth { mean[feature] += window[frame * shape.inputWidth + feature] / count }
        }
    }
    var variance = [Float](repeating: 0, count: shape.inputWidth)
    for window in windows {
        for frame in 0..<shape.windowLength {
            for feature in 0..<shape.inputWidth {
                let difference = window[frame * shape.inputWidth + feature] - mean[feature]
                variance[feature] += difference * difference / count
            }
        }
    }
    return .init(mean: mean, standardDeviation: variance.map { max(sqrt($0), 0.000_001) })
}

private func normalize(
    _ windows: [[Float]],
    shape: TemporalModelShape,
    statistics: FeatureStatistics
) -> [[Float]] {
    windows.map { window in
        window.enumerated().map { index, value in
            let feature = index % shape.inputWidth
            return (value - statistics.mean[feature]) / statistics.standardDeviation[feature]
        }
    }
}

private func initialProjectionBias(targets: [[Float]], masks: [[Float]], width: Int) -> [Float] {
    (0..<width).map { column in
        var sum: Float = 0
        var count: Float = 0
        for row in targets.indices where masks[row][column] > 0 {
            sum += targets[row][column]
            count += 1
        }
        return count > 0 ? sum / count : 0
    }
}

private func tensorData(_ values: [Float], shape: [NSNumber], device: MPSGraphDevice) -> MPSGraphTensorData {
    values.withUnsafeBufferPointer { buffer in
        MPSGraphTensorData(device: device, data: Data(buffer: buffer), shape: shape, dataType: .float32)
    }
}

private func floats(_ data: MPSGraphTensorData, count: Int) -> [Float] {
    var values = [Float](repeating: 0, count: count)
    data.mpsndarray().readBytes(&values, strideBytes: nil)
    return values
}

private func numbers(_ values: [Int]) -> [NSNumber] { values.map(NSNumber.init) }

private struct DeterministicGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed == 0 ? 1 : seed }
    mutating func values(count: Int, scale: Float) -> [Float] {
        (0..<count).map { _ in
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return (Float(state & 0xffff) / Float(0xffff) * 2 - 1) * scale
        }
    }
}
