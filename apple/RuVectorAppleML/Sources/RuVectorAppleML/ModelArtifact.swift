import CryptoKit
import Darwin
import Foundation

public struct RuVectorModelManifest: Codable, Equatable, Sendable {
    public static let currentSchema = "ruvector.apple.model-manifest.v1"

    public let schema: String
    public let identifier: String
    public let version: String
    public let purpose: String
    public let assetSHA256: String
    public let assetByteCount: UInt64
    public let minimumOSVersion: String

    public init(
        identifier: String,
        version: String,
        purpose: String,
        assetSHA256: String,
        assetByteCount: UInt64,
        minimumOSVersion: String
    ) throws {
        try Self.validate(
            schema: Self.currentSchema,
            identifier: identifier,
            version: version,
            purpose: purpose,
            assetSHA256: assetSHA256,
            assetByteCount: assetByteCount,
            minimumOSVersion: minimumOSVersion
        )
        schema = Self.currentSchema
        self.identifier = identifier
        self.version = version
        self.purpose = purpose
        self.assetSHA256 = assetSHA256.lowercased()
        self.assetByteCount = assetByteCount
        self.minimumOSVersion = minimumOSVersion
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let schema = try container.decode(String.self, forKey: .schema)
        let identifier = try container.decode(String.self, forKey: .identifier)
        let version = try container.decode(String.self, forKey: .version)
        let purpose = try container.decode(String.self, forKey: .purpose)
        let assetSHA256 = try container.decode(String.self, forKey: .assetSHA256)
        let assetByteCount = try container.decode(UInt64.self, forKey: .assetByteCount)
        let minimumOSVersion = try container.decode(String.self, forKey: .minimumOSVersion)
        try Self.validate(
            schema: schema,
            identifier: identifier,
            version: version,
            purpose: purpose,
            assetSHA256: assetSHA256,
            assetByteCount: assetByteCount,
            minimumOSVersion: minimumOSVersion
        )
        self.schema = schema
        self.identifier = identifier
        self.version = version
        self.purpose = purpose
        self.assetSHA256 = assetSHA256.lowercased()
        self.assetByteCount = assetByteCount
        self.minimumOSVersion = minimumOSVersion
    }

    public func canonicalData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    private static func validate(
        schema: String,
        identifier: String,
        version: String,
        purpose: String,
        assetSHA256: String,
        assetByteCount: UInt64,
        minimumOSVersion: String
    ) throws {
        let safe = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        guard schema == currentSchema else {
            throw RuVectorEdgeMLError.invalidConfiguration("unsupported model manifest schema")
        }
        guard (1...128).contains(identifier.count),
              identifier.unicodeScalars.allSatisfy({ safe.contains($0) }) else {
            throw RuVectorEdgeMLError.invalidConfiguration("model identifier is invalid")
        }
        guard (1...64).contains(version.count), (1...256).contains(purpose.count),
              (1...32).contains(minimumOSVersion.count) else {
            throw RuVectorEdgeMLError.invalidConfiguration("model metadata is invalid")
        }
        let digest = assetSHA256.lowercased()
        guard digest.count == 64, digest.allSatisfy({ $0.isHexDigit }),
              (1...2_147_483_648).contains(assetByteCount) else {
            throw RuVectorEdgeMLError.invalidConfiguration("model asset digest or size is invalid")
        }
    }
}

/// A point-in-time receipt for bytes read from one regular-file descriptor.
///
/// The receipt intentionally contains no URL. It does not pin a path, attest a
/// compiled Core ML directory, authorize activation, or prove that a later read
/// observes the same bytes.
public struct VerifiedRegularFileArtifact: Equatable, Sendable {
    public let manifest: RuVectorModelManifest
    public let contentSHA256: String
    public let byteCount: UInt64

    fileprivate init(manifest: RuVectorModelManifest) {
        self.manifest = manifest
        contentSHA256 = manifest.assetSHA256
        byteCount = manifest.assetByteCount
    }
}

/// Verifies signed metadata and the bytes currently read from a local,
/// non-symbolic regular file.
///
/// This utility authenticates a distribution artifact at verification time.
/// It does not bind that artifact to an independently compiled or loaded
/// `.mlmodelc` directory. A consuming application owns protected staging,
/// transformation, compilation, activation, and rollback.
public enum ModelArtifactVerifier {
    private static let maximumArtifactByteCount: UInt64 = 2_147_483_648

    public static func verifyRegularFileDistributionArtifact(
        assetURL: URL,
        manifest: RuVectorModelManifest,
        signature: Data,
        publicKey: Data
    ) throws -> VerifiedRegularFileArtifact {
        let key = try Curve25519.Signing.PublicKey(rawRepresentation: publicKey)
        guard key.isValidSignature(signature, for: try manifest.canonicalData()) else {
            throw RuVectorEdgeMLError.invalidInput("model manifest signature is invalid")
        }
        return try withOpenRegularFile(at: assetURL) { handle, fileSize in
            guard fileSize == manifest.assetByteCount else {
                throw RuVectorEdgeMLError.invalidInput("model asset size does not match the signed manifest")
            }
            guard try sha256(handle: handle, expectedByteCount: fileSize) == manifest.assetSHA256 else {
                throw RuVectorEdgeMLError.invalidInput("model asset digest does not match the signed manifest")
            }
            return VerifiedRegularFileArtifact(manifest: manifest)
        }
    }

    public static func sha256(url: URL) throws -> String {
        try withOpenRegularFile(at: url) { handle, fileSize in
            guard fileSize <= maximumArtifactByteCount else {
                throw RuVectorEdgeMLError.invalidInput("model artifact exceeds the byte limit")
            }
            return try sha256(handle: handle, expectedByteCount: fileSize)
        }
    }

    private static func sha256(handle: FileHandle, expectedByteCount: UInt64) throws -> String {
        var hasher = SHA256()
        var remaining = expectedByteCount
        while remaining > 0 {
            let readCount = Int(min(remaining, 1_048_576))
            guard let data = try handle.read(upToCount: readCount), data.count == readCount else {
                throw RuVectorEdgeMLError.invalidInput("model artifact changed while it was being read")
            }
            hasher.update(data: data)
            remaining -= UInt64(data.count)
        }
        let trailingByte = try handle.read(upToCount: 1)
        guard trailingByte?.isEmpty ?? true else {
            throw RuVectorEdgeMLError.invalidInput("model artifact changed while it was being read")
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func withOpenRegularFile<T>(
        at url: URL,
        _ body: (FileHandle, UInt64) throws -> T
    ) throws -> T {
        guard url.isFileURL else {
            throw RuVectorEdgeMLError.invalidInput("model artifact must be a local file")
        }
        let descriptor = url.withUnsafeFileSystemRepresentation { path -> Int32 in
            guard let path else { return -1 }
            return Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        }
        guard descriptor >= 0 else {
            throw RuVectorEdgeMLError.invalidInput("model artifact could not be opened as a non-symbolic file")
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }

        var metadata = stat()
        guard Darwin.fstat(descriptor, &metadata) == 0,
              metadata.st_mode & S_IFMT == S_IFREG,
              metadata.st_size >= 0 else {
            throw RuVectorEdgeMLError.invalidInput("model artifact must be a regular file")
        }
        return try body(handle, UInt64(metadata.st_size))
    }
}
