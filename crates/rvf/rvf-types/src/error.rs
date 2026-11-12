//! Error codes and error types for the RVF format.
//!
//! Error codes are 16-bit unsigned integers where the high byte identifies
//! the category and the low byte the specific error.

/// Wire-format error code (u16). The high byte is the category, the low byte is
/// the specific error within that category.
///
/// This enum is `#[non_exhaustive]`: new codes may be added in minor releases,
/// so downstream `match`es must include a wildcard arm (fail closed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[non_exhaustive]
pub enum ErrorCode {
    // ---- Category 0x00: Success ----
    /// Operation succeeded.
    Ok = 0x0000,
    /// Partial success (some items failed).
    OkPartial = 0x0001,

    // ---- Category 0x01: Format Errors ----
    /// Segment magic mismatch (expected 0x52564653).
    InvalidMagic = 0x0100,
    /// Unsupported segment version.
    InvalidVersion = 0x0101,
    /// Segment hash verification failed.
    InvalidChecksum = 0x0102,
    /// Cryptographic signature invalid.
    InvalidSignature = 0x0103,
    /// Segment payload shorter than declared length.
    TruncatedSegment = 0x0104,
    /// Root manifest validation failed.
    InvalidManifest = 0x0105,
    /// No valid MANIFEST_SEG in file.
    ManifestNotFound = 0x0106,
    /// Segment type not recognized (advisory, not fatal).
    UnknownSegmentType = 0x0107,
    /// Data not at expected 64-byte boundary.
    AlignmentError = 0x0108,
    /// Authoritative metadata payload or resulting vector binding is invalid.
    InvalidMetadata = 0x0109,
    /// Metadata replay exceeded the normal-open delta or byte ceiling.
    MetadataReplayLimitExceeded = 0x010A,
    /// Metadata base generations are missing, cyclic, or non-monotonic.
    MetadataAncestryInvalid = 0x010B,

    // ---- Category 0x02: Query Errors ----
    /// Query vector dimension != index dimension.
    DimensionMismatch = 0x0200,
    /// No index segments available.
    EmptyIndex = 0x0201,
    /// Requested distance metric not available.
    MetricUnsupported = 0x0202,
    /// Invalid filter expression.
    FilterParseError = 0x0203,
    /// Requested K exceeds available vectors.
    KTooLarge = 0x0204,
    /// Query exceeded time budget.
    Timeout = 0x0205,
    /// Linear metadata filtering exceeded an explicit resource budget.
    FilterBudgetExceeded = 0x0206,
    /// Query vector contains NaN or an infinity, or its distances overflow to
    /// a non-finite value (invalid input).
    NonFiniteQuery = 0x0207,

    // ---- Category 0x03: Write Errors ----
    /// Another writer holds the lock.
    LockHeld = 0x0300,
    /// Lock file exists but owner process is dead.
    LockStale = 0x0301,
    /// Insufficient space for write.
    DiskFull = 0x0302,
    /// Durable write (fsync) failed.
    FsyncFailed = 0x0303,
    /// Segment exceeds 4 GB limit.
    SegmentTooLarge = 0x0304,
    /// File opened in read-only mode.
    ReadOnly = 0x0305,

    // ---- Category 0x04: Tile Errors (WASM Microkernel) ----
    /// WASM trap (OOB, unreachable, stack overflow).
    TileTrap = 0x0400,
    /// Tile exceeded scratch memory (64 KB).
    TileOom = 0x0401,
    /// Tile computation exceeded time budget.
    TileTimeout = 0x0402,
    /// Malformed hub-tile message.
    TileInvalidMsg = 0x0403,
    /// Operation not available on this profile.
    TileUnsupportedOp = 0x0404,

    // ---- Category 0x05: Crypto Errors ----
    /// Referenced key_id not in CRYPTO_SEG.
    KeyNotFound = 0x0500,
    /// Key past valid_until timestamp.
    KeyExpired = 0x0501,
    /// Decryption or auth tag verification failed.
    DecryptFailed = 0x0502,
    /// Cryptographic algorithm not implemented.
    AlgoUnsupported = 0x0503,
    /// Attestation quote verification failed.
    AttestationInvalid = 0x0504,
    /// TEE platform not supported.
    PlatformUnsupported = 0x0505,
    /// Attestation quote expired or nonce mismatch.
    AttestationExpired = 0x0506,
    /// Key is not bound to the current TEE measurement.
    KeyNotBound = 0x0507,

    // ---- Category 0x06: Lineage Errors ----
    /// Referenced parent file not found.
    ParentNotFound = 0x0600,
    /// Parent file hash does not match recorded parent_hash.
    ParentHashMismatch = 0x0601,
    /// Lineage chain is broken (missing link).
    LineageBroken = 0x0602,
    /// Lineage chain contains a cycle.
    LineageCyclic = 0x0603,

    // ---- Category 0x08: Security Errors (ADR-033) ----
    /// Level 0 manifest has no signature in Strict/Paranoid mode.
    UnsignedManifest = 0x0800,
    /// Content hash mismatch on a hotset-referenced segment.
    ContentHashMismatch = 0x0801,
    /// Manifest signer is not in the trust store.
    UnknownSigner = 0x0802,
    /// Centroid epoch drift exceeds maximum allowed.
    EpochDriftExceeded = 0x0803,
    /// Level 1 manifest signature invalid (Paranoid mode).
    Level1InvalidSignature = 0x0804,
    /// RVSD descriptor is shorter or longer than its fixed layout requires.
    StreamDescriptorTruncated = 0x0805,
    /// RVSD descriptor magic mismatch.
    StreamBadMagic = 0x0806,
    /// RVSD descriptor version is not supported.
    StreamUnsupportedVersion = 0x0807,
    /// RVSD descriptor has unknown flags, algorithms, chunk size or non-zero reserved bytes.
    StreamBadFlags = 0x0808,
    /// RVSD descriptor exceeds a receiver policy limit.
    StreamLimitExceeded = 0x0809,
    /// RVSD segment table violates an invariant (order, contiguity, uniqueness, sizes).
    StreamTableInvalid = 0x080A,
    /// RVSD descriptor signature does not verify.
    StreamBadSignature = 0x080B,
    /// RVSD descriptor epoch is lower than the pinned epoch (replay).
    StreamStaleEpoch = 0x080C,
    /// RVSD descriptor has the pinned epoch but different content.
    StreamEpochConflict = 0x080D,
    /// RVSD descriptor has expired.
    StreamExpired = 0x080E,
    /// RVSD descriptor policy hash differs from the expected policy hash.
    StreamPolicyMismatch = 0x080F,
    /// Stream chunk does not match its committed leaf hash.
    StreamChunkHashMismatch = 0x0810,
    /// Stream chunk table does not match the committed chunk table hash.
    StreamTableHashMismatch = 0x0811,
    /// Stream frame is a duplicate, out of order, unknown or malformed.
    StreamUnexpectedFrame = 0x0812,
    /// Stream frame declares a size that contradicts the signed descriptor.
    StreamFrameSizeLie = 0x0813,
    /// Chunk arrived before the stages named in its requires_mask were verified.
    StreamRequirementsUnmet = 0x0814,
    /// SHA-256 of the verified body differs from the signed rvf_identity.
    StreamIdentityMismatch = 0x0815,
    /// Executable or index segment uses a non-cryptographic checksum algorithm.
    StreamWeakChecksum = 0x0816,
    /// Stream is poisoned; no further frames are accepted.
    StreamPoisoned = 0x0817,
    // 0x0818 is reserved (never assigned).
    /// Frame pushed in a state that cannot accept it.
    StreamBadState = 0x0819,
    /// Too many idempotent duplicate frames (retransmit flood).
    StreamDuplicateFlood = 0x081A,
    /// A signature-checking open was requested without an explicit freshness choice.
    FreshnessRequired = 0x081B,

    // ---- Category 0x09: Quality Errors (ADR-033) ----
    /// Query result quality is below threshold and AcceptDegraded not set.
    QualityBelowThreshold = 0x0900,
    /// Per-connection budget tokens exhausted (DoS protection).
    BudgetTokensExhausted = 0x0901,
    /// Query signature is blacklisted (repeated degenerate queries).
    QueryBlacklisted = 0x0902,

    // ---- Category 0x07: COW Errors ----
    /// COW cluster map is corrupt or unreadable.
    CowMapCorrupt = 0x0700,
    /// Referenced cluster not found in COW map.
    ClusterNotFound = 0x0701,
    /// Parent chain is broken (missing ancestor).
    ParentChainBroken = 0x0702,
    /// Delta patch exceeds compaction threshold.
    DeltaThresholdExceeded = 0x0703,
    /// Snapshot is frozen and cannot be modified.
    SnapshotFrozen = 0x0704,
    /// Membership filter is invalid or corrupt.
    MembershipInvalid = 0x0705,
    /// Generation counter is stale (concurrent modification).
    GenerationStale = 0x0706,
    /// Kernel binding hash does not match manifest.
    KernelBindingMismatch = 0x0707,
    /// Double-root manifest is corrupt.
    DoubleRootCorrupt = 0x0708,
}

impl ErrorCode {
    /// Return the error category (high byte).
    #[inline]
    pub const fn category(self) -> u8 {
        (self as u16 >> 8) as u8
    }

    /// Return true if this code indicates success (category 0x00).
    #[inline]
    pub const fn is_success(self) -> bool {
        self.category() == 0x00
    }

    /// Return true if this is a format error (category 0x01), which is generally fatal.
    #[inline]
    pub const fn is_format_error(self) -> bool {
        self.category() == 0x01
    }

    /// Return true if this is a security error (category 0x08).
    #[inline]
    pub const fn is_security_error(self) -> bool {
        self.category() == 0x08
    }

    /// Return true if this is a quality error (category 0x09).
    #[inline]
    pub const fn is_quality_error(self) -> bool {
        self.category() == 0x09
    }
}

impl TryFrom<u16> for ErrorCode {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0x0000 => Ok(Self::Ok),
            0x0001 => Ok(Self::OkPartial),

            0x0100 => Ok(Self::InvalidMagic),
            0x0101 => Ok(Self::InvalidVersion),
            0x0102 => Ok(Self::InvalidChecksum),
            0x0103 => Ok(Self::InvalidSignature),
            0x0104 => Ok(Self::TruncatedSegment),
            0x0105 => Ok(Self::InvalidManifest),
            0x0106 => Ok(Self::ManifestNotFound),
            0x0107 => Ok(Self::UnknownSegmentType),
            0x0108 => Ok(Self::AlignmentError),
            0x0109 => Ok(Self::InvalidMetadata),
            0x010A => Ok(Self::MetadataReplayLimitExceeded),
            0x010B => Ok(Self::MetadataAncestryInvalid),

            0x0200 => Ok(Self::DimensionMismatch),
            0x0201 => Ok(Self::EmptyIndex),
            0x0202 => Ok(Self::MetricUnsupported),
            0x0203 => Ok(Self::FilterParseError),
            0x0204 => Ok(Self::KTooLarge),
            0x0205 => Ok(Self::Timeout),
            0x0206 => Ok(Self::FilterBudgetExceeded),
            0x0207 => Ok(Self::NonFiniteQuery),

            0x0300 => Ok(Self::LockHeld),
            0x0301 => Ok(Self::LockStale),
            0x0302 => Ok(Self::DiskFull),
            0x0303 => Ok(Self::FsyncFailed),
            0x0304 => Ok(Self::SegmentTooLarge),
            0x0305 => Ok(Self::ReadOnly),

            0x0400 => Ok(Self::TileTrap),
            0x0401 => Ok(Self::TileOom),
            0x0402 => Ok(Self::TileTimeout),
            0x0403 => Ok(Self::TileInvalidMsg),
            0x0404 => Ok(Self::TileUnsupportedOp),

            0x0500 => Ok(Self::KeyNotFound),
            0x0501 => Ok(Self::KeyExpired),
            0x0502 => Ok(Self::DecryptFailed),
            0x0503 => Ok(Self::AlgoUnsupported),
            0x0504 => Ok(Self::AttestationInvalid),
            0x0505 => Ok(Self::PlatformUnsupported),
            0x0506 => Ok(Self::AttestationExpired),
            0x0507 => Ok(Self::KeyNotBound),

            0x0600 => Ok(Self::ParentNotFound),
            0x0601 => Ok(Self::ParentHashMismatch),
            0x0602 => Ok(Self::LineageBroken),
            0x0603 => Ok(Self::LineageCyclic),

            0x0800 => Ok(Self::UnsignedManifest),
            0x0801 => Ok(Self::ContentHashMismatch),
            0x0802 => Ok(Self::UnknownSigner),
            0x0803 => Ok(Self::EpochDriftExceeded),
            0x0804 => Ok(Self::Level1InvalidSignature),
            0x0805 => Ok(Self::StreamDescriptorTruncated),
            0x0806 => Ok(Self::StreamBadMagic),
            0x0807 => Ok(Self::StreamUnsupportedVersion),
            0x0808 => Ok(Self::StreamBadFlags),
            0x0809 => Ok(Self::StreamLimitExceeded),
            0x080A => Ok(Self::StreamTableInvalid),
            0x080B => Ok(Self::StreamBadSignature),
            0x080C => Ok(Self::StreamStaleEpoch),
            0x080D => Ok(Self::StreamEpochConflict),
            0x080E => Ok(Self::StreamExpired),
            0x080F => Ok(Self::StreamPolicyMismatch),
            0x0810 => Ok(Self::StreamChunkHashMismatch),
            0x0811 => Ok(Self::StreamTableHashMismatch),
            0x0812 => Ok(Self::StreamUnexpectedFrame),
            0x0813 => Ok(Self::StreamFrameSizeLie),
            0x0814 => Ok(Self::StreamRequirementsUnmet),
            0x0815 => Ok(Self::StreamIdentityMismatch),
            0x0816 => Ok(Self::StreamWeakChecksum),
            0x0817 => Ok(Self::StreamPoisoned),
            0x0819 => Ok(Self::StreamBadState),
            0x081A => Ok(Self::StreamDuplicateFlood),
            0x081B => Ok(Self::FreshnessRequired),

            0x0900 => Ok(Self::QualityBelowThreshold),
            0x0901 => Ok(Self::BudgetTokensExhausted),
            0x0902 => Ok(Self::QueryBlacklisted),

            0x0700 => Ok(Self::CowMapCorrupt),
            0x0701 => Ok(Self::ClusterNotFound),
            0x0702 => Ok(Self::ParentChainBroken),
            0x0703 => Ok(Self::DeltaThresholdExceeded),
            0x0704 => Ok(Self::SnapshotFrozen),
            0x0705 => Ok(Self::MembershipInvalid),
            0x0706 => Ok(Self::GenerationStale),
            0x0707 => Ok(Self::KernelBindingMismatch),
            0x0708 => Ok(Self::DoubleRootCorrupt),

            other => Err(other),
        }
    }
}

/// Rust-idiomatic error type wrapping format-level failures.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum RvfError {
    /// A wire-level error code was returned.
    Code(ErrorCode),
    /// The raw u16 did not map to a known error code.
    UnknownCode(u16),
    /// A segment header had an invalid magic number.
    BadMagic { expected: u32, got: u32 },
    /// A struct size assertion failed.
    SizeMismatch { expected: usize, got: usize },
    /// A value was outside the valid enum range.
    InvalidEnumValue { type_name: &'static str, value: u64 },
    /// Security policy violation during file open (ADR-033 §4).
    Security(crate::security::SecurityError),
    /// Query result quality is below threshold (ADR-033 §2.4).
    /// Contains the QualityEnvelope with partial results and diagnostics.
    QualityBelowThreshold {
        quality: crate::quality::ResponseQuality,
        reason: &'static str,
    },
}

impl core::fmt::Display for RvfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Code(c) => write!(f, "RVF error code 0x{:04X} ({:?})", *c as u16, c),
            Self::UnknownCode(v) => write!(f, "unknown RVF error code 0x{v:04X}"),
            Self::BadMagic { expected, got } => {
                write!(f, "bad magic: expected 0x{expected:08X}, got 0x{got:08X}")
            }
            Self::SizeMismatch { expected, got } => {
                write!(f, "size mismatch: expected {expected}, got {got}")
            }
            Self::InvalidEnumValue { type_name, value } => {
                write!(f, "invalid {type_name} value: {value}")
            }
            Self::Security(e) => write!(f, "security error: {e}"),
            Self::QualityBelowThreshold { quality, reason } => {
                write!(f, "quality below threshold ({quality:?}): {reason}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    /// #1276 item 5: errors must name the code, not just print the number.
    #[test]
    fn display_names_the_error_code() {
        let s = format!("{}", RvfError::Code(ErrorCode::InvalidManifest));
        assert!(s.contains("0x0105") && s.contains("InvalidManifest"), "{s}");
        let s = format!("{}", RvfError::Code(ErrorCode::DimensionMismatch));
        assert!(s.contains("DimensionMismatch"), "{s}");
    }

    #[test]
    fn error_code_round_trip_all() {
        let codes: &[(u16, ErrorCode)] = &[
            (0x0000, ErrorCode::Ok),
            (0x0001, ErrorCode::OkPartial),
            (0x0100, ErrorCode::InvalidMagic),
            (0x0101, ErrorCode::InvalidVersion),
            (0x0102, ErrorCode::InvalidChecksum),
            (0x0103, ErrorCode::InvalidSignature),
            (0x0104, ErrorCode::TruncatedSegment),
            (0x0105, ErrorCode::InvalidManifest),
            (0x0106, ErrorCode::ManifestNotFound),
            (0x0107, ErrorCode::UnknownSegmentType),
            (0x0108, ErrorCode::AlignmentError),
            (0x0200, ErrorCode::DimensionMismatch),
            (0x0201, ErrorCode::EmptyIndex),
            (0x0202, ErrorCode::MetricUnsupported),
            (0x0203, ErrorCode::FilterParseError),
            (0x0204, ErrorCode::KTooLarge),
            (0x0205, ErrorCode::Timeout),
            (0x0206, ErrorCode::FilterBudgetExceeded),
            (0x0207, ErrorCode::NonFiniteQuery),
            (0x0300, ErrorCode::LockHeld),
            (0x0301, ErrorCode::LockStale),
            (0x0302, ErrorCode::DiskFull),
            (0x0303, ErrorCode::FsyncFailed),
            (0x0304, ErrorCode::SegmentTooLarge),
            (0x0305, ErrorCode::ReadOnly),
            (0x0400, ErrorCode::TileTrap),
            (0x0401, ErrorCode::TileOom),
            (0x0402, ErrorCode::TileTimeout),
            (0x0403, ErrorCode::TileInvalidMsg),
            (0x0404, ErrorCode::TileUnsupportedOp),
            (0x0500, ErrorCode::KeyNotFound),
            (0x0501, ErrorCode::KeyExpired),
            (0x0502, ErrorCode::DecryptFailed),
            (0x0503, ErrorCode::AlgoUnsupported),
            (0x0504, ErrorCode::AttestationInvalid),
            (0x0505, ErrorCode::PlatformUnsupported),
            (0x0506, ErrorCode::AttestationExpired),
            (0x0507, ErrorCode::KeyNotBound),
            (0x0600, ErrorCode::ParentNotFound),
            (0x0601, ErrorCode::ParentHashMismatch),
            (0x0602, ErrorCode::LineageBroken),
            (0x0603, ErrorCode::LineageCyclic),
            (0x0800, ErrorCode::UnsignedManifest),
            (0x0801, ErrorCode::ContentHashMismatch),
            (0x0802, ErrorCode::UnknownSigner),
            (0x0803, ErrorCode::EpochDriftExceeded),
            (0x0804, ErrorCode::Level1InvalidSignature),
            (0x0805, ErrorCode::StreamDescriptorTruncated),
            (0x0806, ErrorCode::StreamBadMagic),
            (0x0807, ErrorCode::StreamUnsupportedVersion),
            (0x0808, ErrorCode::StreamBadFlags),
            (0x0809, ErrorCode::StreamLimitExceeded),
            (0x080A, ErrorCode::StreamTableInvalid),
            (0x080B, ErrorCode::StreamBadSignature),
            (0x080C, ErrorCode::StreamStaleEpoch),
            (0x080D, ErrorCode::StreamEpochConflict),
            (0x080E, ErrorCode::StreamExpired),
            (0x080F, ErrorCode::StreamPolicyMismatch),
            (0x0810, ErrorCode::StreamChunkHashMismatch),
            (0x0811, ErrorCode::StreamTableHashMismatch),
            (0x0812, ErrorCode::StreamUnexpectedFrame),
            (0x0813, ErrorCode::StreamFrameSizeLie),
            (0x0814, ErrorCode::StreamRequirementsUnmet),
            (0x0815, ErrorCode::StreamIdentityMismatch),
            (0x0816, ErrorCode::StreamWeakChecksum),
            (0x0817, ErrorCode::StreamPoisoned),
            (0x0819, ErrorCode::StreamBadState),
            (0x081A, ErrorCode::StreamDuplicateFlood),
            (0x081B, ErrorCode::FreshnessRequired),
            (0x0900, ErrorCode::QualityBelowThreshold),
            (0x0901, ErrorCode::BudgetTokensExhausted),
            (0x0902, ErrorCode::QueryBlacklisted),
            (0x0700, ErrorCode::CowMapCorrupt),
            (0x0701, ErrorCode::ClusterNotFound),
            (0x0702, ErrorCode::ParentChainBroken),
            (0x0703, ErrorCode::DeltaThresholdExceeded),
            (0x0704, ErrorCode::SnapshotFrozen),
            (0x0705, ErrorCode::MembershipInvalid),
            (0x0706, ErrorCode::GenerationStale),
            (0x0707, ErrorCode::KernelBindingMismatch),
            (0x0708, ErrorCode::DoubleRootCorrupt),
        ];
        for &(raw, expected) in codes {
            assert_eq!(ErrorCode::try_from(raw), Ok(expected), "code 0x{raw:04X}");
            assert_eq!(expected as u16, raw);
        }
    }

    #[test]
    fn unknown_code() {
        assert_eq!(ErrorCode::try_from(0x9999), Err(0x9999));
    }

    #[test]
    fn category_extraction() {
        assert_eq!(ErrorCode::Ok.category(), 0x00);
        assert_eq!(ErrorCode::InvalidMagic.category(), 0x01);
        assert_eq!(ErrorCode::DimensionMismatch.category(), 0x02);
        assert_eq!(ErrorCode::LockHeld.category(), 0x03);
        assert_eq!(ErrorCode::TileTrap.category(), 0x04);
        assert_eq!(ErrorCode::KeyNotFound.category(), 0x05);
        assert_eq!(ErrorCode::ParentNotFound.category(), 0x06);
        assert_eq!(ErrorCode::CowMapCorrupt.category(), 0x07);
        assert_eq!(ErrorCode::UnsignedManifest.category(), 0x08);
        assert_eq!(ErrorCode::QualityBelowThreshold.category(), 0x09);
    }

    #[test]
    fn security_error_check() {
        assert!(ErrorCode::UnsignedManifest.is_security_error());
        assert!(!ErrorCode::Ok.is_security_error());
    }

    #[test]
    fn quality_error_check() {
        assert!(ErrorCode::QualityBelowThreshold.is_quality_error());
        assert!(!ErrorCode::Ok.is_quality_error());
    }

    #[test]
    fn success_check() {
        assert!(ErrorCode::Ok.is_success());
        assert!(ErrorCode::OkPartial.is_success());
        assert!(!ErrorCode::InvalidMagic.is_success());
    }

    #[test]
    fn format_error_check() {
        assert!(ErrorCode::InvalidMagic.is_format_error());
        assert!(!ErrorCode::Ok.is_format_error());
        assert!(!ErrorCode::DimensionMismatch.is_format_error());
    }

    #[test]
    fn rvf_error_display() {
        let e = RvfError::BadMagic {
            expected: 0x52564653,
            got: 0x00000000,
        };
        let s = format!("{e}");
        assert!(s.contains("bad magic"));
        assert!(s.contains("52564653"));
    }

    #[test]
    fn cow_error_check() {
        assert_eq!(ErrorCode::CowMapCorrupt as u16, 0x0700);
        assert_eq!(ErrorCode::ClusterNotFound as u16, 0x0701);
        assert_eq!(ErrorCode::ParentChainBroken as u16, 0x0702);
        assert_eq!(ErrorCode::DeltaThresholdExceeded as u16, 0x0703);
        assert_eq!(ErrorCode::SnapshotFrozen as u16, 0x0704);
        assert_eq!(ErrorCode::MembershipInvalid as u16, 0x0705);
        assert_eq!(ErrorCode::GenerationStale as u16, 0x0706);
        assert_eq!(ErrorCode::KernelBindingMismatch as u16, 0x0707);
        assert_eq!(ErrorCode::DoubleRootCorrupt as u16, 0x0708);
        // All COW errors should be category 0x07
        assert_eq!(ErrorCode::CowMapCorrupt.category(), 0x07);
        assert_eq!(ErrorCode::DoubleRootCorrupt.category(), 0x07);
    }

    #[test]
    fn error_codes_match_spec() {
        assert_eq!(ErrorCode::InvalidMagic as u16, 0x0100);
        assert_eq!(ErrorCode::InvalidChecksum as u16, 0x0102);
        assert_eq!(ErrorCode::ManifestNotFound as u16, 0x0106);
        assert_eq!(ErrorCode::AlgoUnsupported as u16, 0x0503);
    }
}
