// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "RuVectorApple",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(name: "RuVectorAppleML", targets: ["RuVectorAppleML"]),
        .executable(name: "ruvector-apple-benchmark", targets: ["RuVectorAppleBenchmark"]),
    ],
    targets: [
        .target(
            name: "RuVectorAppleML",
            path: "apple/RuVectorAppleML/Sources/RuVectorAppleML",
            resources: [.process("PrivacyInfo.xcprivacy")]
        ),
        .executableTarget(
            name: "RuVectorAppleBenchmark",
            dependencies: ["RuVectorAppleML"],
            path: "apple/RuVectorAppleML/Benchmarks"
        ),
        .testTarget(
            name: "RuVectorAppleMLTests",
            dependencies: ["RuVectorAppleML"],
            path: "apple/RuVectorAppleML/Tests/RuVectorAppleMLTests"
        ),
    ]
)
