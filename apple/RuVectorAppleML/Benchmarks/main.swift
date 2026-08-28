import Foundation
import RuVectorAppleML

struct BenchmarkResult: Encodable {
    let evidence: String
    let platform: String
    let iterations: Int
    let inputCount: Int
    let hiddenWidth: Int
    let outputWidth: Int
    let p50Milliseconds: Double
    let p95Milliseconds: Double
    let predictionsPerSecond: Double
}

func percentile(_ sorted: [Double], _ fraction: Double) -> Double {
    sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * fraction))]
}

do {
    let shape = try TemporalModelShape(windowLength: 8, inputWidth: 30, hiddenWidth: 16, outputWidth: 45)
    let model = try TemporalProjectionModel(
        shape: shape,
        featureMean: .init(repeating: 0, count: shape.inputWidth),
        featureStandardDeviation: .init(repeating: 1, count: shape.inputWidth),
        temporalWeights: .init(repeating: 0.01, count: shape.inputCount * shape.hiddenWidth),
        temporalBias: .init(repeating: 0, count: shape.hiddenWidth),
        projectionWeights: .init(repeating: 0.01, count: shape.hiddenWidth * shape.outputWidth),
        projectionBias: .init(repeating: 0, count: shape.outputWidth)
    )
    let predictor = AccelerateTemporalPredictor(model: model)
    let window = [Float](repeating: 0.25, count: shape.inputCount)
    for _ in 0..<100 { _ = try predictor.predict(window: window) }
    let iterations = 5_000
    var durations = [Double]()
    durations.reserveCapacity(iterations)
    for _ in 0..<iterations {
        let started = ProcessInfo.processInfo.systemUptime
        _ = try predictor.predict(window: window)
        durations.append((ProcessInfo.processInfo.systemUptime - started) * 1_000)
    }
    durations.sort()
    let totalSeconds = durations.reduce(0, +) / 1_000
    let result = BenchmarkResult(
        evidence: "MEASURED_ON_CURRENT_RUNTIME",
        platform: ProcessInfo.processInfo.operatingSystemVersionString,
        iterations: iterations,
        inputCount: shape.inputCount,
        hiddenWidth: shape.hiddenWidth,
        outputWidth: shape.outputWidth,
        p50Milliseconds: percentile(durations, 0.50),
        p95Milliseconds: percentile(durations, 0.95),
        predictionsPerSecond: Double(iterations) / max(totalSeconds, 0.000_001)
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    FileHandle.standardOutput.write(try encoder.encode(result))
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    FileHandle.standardError.write(Data("benchmark failed: \(error)\n".utf8))
    exit(1)
}
