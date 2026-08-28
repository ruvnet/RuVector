import CryptoKit
import XCTest
@testable import RuVectorAppleML

final class ModelArtifactTests: XCTestCase {
    func testSignedArtifactVerificationAndTamperRejection() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let asset = directory.appendingPathComponent("model.bin")
        let bytes = Data("bounded-model".utf8)
        try bytes.write(to: asset)
        let manifest = try RuVectorModelManifest(
            identifier: "test.model", version: "1.0.0", purpose: "unit-test",
            assetSHA256: try ModelArtifactVerifier.sha256(url: asset),
            assetByteCount: UInt64(bytes.count), minimumOSVersion: "16.0"
        )
        let key = Curve25519.Signing.PrivateKey()
        let signature = try key.signature(for: manifest.canonicalData())
        let receipt = try ModelArtifactVerifier.verifyRegularFileDistributionArtifact(
            assetURL: asset,
            manifest: manifest,
            signature: signature,
            publicKey: key.publicKey.rawRepresentation
        )
        XCTAssertEqual(receipt.manifest, manifest)
        XCTAssertEqual(receipt.contentSHA256, manifest.assetSHA256)
        XCTAssertEqual(receipt.byteCount, manifest.assetByteCount)
        try Data("tampered-data".utf8).write(to: asset)
        XCTAssertThrowsError(try ModelArtifactVerifier.verifyRegularFileDistributionArtifact(
            assetURL: asset,
            manifest: manifest,
            signature: signature,
            publicKey: key.publicKey.rawRepresentation
        ))
    }

    func testArtifactVerifierRejectsSymbolicLink() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let target = directory.appendingPathComponent("model.bin")
        let link = directory.appendingPathComponent("model-link.bin")
        let bytes = Data("bounded-model".utf8)
        try bytes.write(to: target)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        let manifest = try RuVectorModelManifest(
            identifier: "test.model", version: "1.0.0", purpose: "unit-test",
            assetSHA256: try ModelArtifactVerifier.sha256(url: target),
            assetByteCount: UInt64(bytes.count), minimumOSVersion: "16.0"
        )
        let key = Curve25519.Signing.PrivateKey()
        let signature = try key.signature(for: manifest.canonicalData())

        XCTAssertThrowsError(try ModelArtifactVerifier.verifyRegularFileDistributionArtifact(
            assetURL: link,
            manifest: manifest,
            signature: signature,
            publicKey: key.publicKey.rawRepresentation
        ))
    }

    func testManifestRejectsUnsafeIdentifier() {
        XCTAssertThrowsError(try RuVectorModelManifest(
            identifier: "../escape", version: "1", purpose: "test",
            assetSHA256: String(repeating: "a", count: 64), assetByteCount: 1,
            minimumOSVersion: "16.0"
        ))
    }
}
