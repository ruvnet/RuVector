//! Regression test for issue #1274: `ErrorCode` and `StreamClass` must be
//! `#[non_exhaustive]` so adding variants is not a semver break.
//!
//! This integration test is an *external* crate from rvf-types' point of view.
//! `deny(unreachable_patterns)` makes the wildcard arm an error when the enum
//! is exhaustive (every variant listed + `_` = unreachable) and mandatory when
//! it is `#[non_exhaustive]`. The test therefore fails to compile before the fix.
#![deny(unreachable_patterns)]

use rvf_types::stream::StreamClass;
use rvf_types::ErrorCode;

/// Lists every current variant plus a fail-closed wildcard.
fn classify(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::Ok => "known",
        ErrorCode::OkPartial => "known",
        ErrorCode::InvalidMagic => "known",
        ErrorCode::InvalidVersion => "known",
        ErrorCode::InvalidChecksum => "known",
        ErrorCode::InvalidSignature => "known",
        ErrorCode::TruncatedSegment => "known",
        ErrorCode::InvalidManifest => "known",
        ErrorCode::ManifestNotFound => "known",
        ErrorCode::UnknownSegmentType => "known",
        ErrorCode::AlignmentError => "known",
        ErrorCode::InvalidMetadata => "known",
        ErrorCode::MetadataReplayLimitExceeded => "known",
        ErrorCode::MetadataAncestryInvalid => "known",
        ErrorCode::DimensionMismatch => "known",
        ErrorCode::EmptyIndex => "known",
        ErrorCode::MetricUnsupported => "known",
        ErrorCode::FilterParseError => "known",
        ErrorCode::KTooLarge => "known",
        ErrorCode::Timeout => "known",
        ErrorCode::FilterBudgetExceeded => "known",
        ErrorCode::LockHeld => "known",
        ErrorCode::LockStale => "known",
        ErrorCode::DiskFull => "known",
        ErrorCode::FsyncFailed => "known",
        ErrorCode::SegmentTooLarge => "known",
        ErrorCode::ReadOnly => "known",
        ErrorCode::TileTrap => "known",
        ErrorCode::TileOom => "known",
        ErrorCode::TileTimeout => "known",
        ErrorCode::TileInvalidMsg => "known",
        ErrorCode::TileUnsupportedOp => "known",
        ErrorCode::KeyNotFound => "known",
        ErrorCode::KeyExpired => "known",
        ErrorCode::DecryptFailed => "known",
        ErrorCode::AlgoUnsupported => "known",
        ErrorCode::AttestationInvalid => "known",
        ErrorCode::PlatformUnsupported => "known",
        ErrorCode::AttestationExpired => "known",
        ErrorCode::KeyNotBound => "known",
        ErrorCode::ParentNotFound => "known",
        ErrorCode::ParentHashMismatch => "known",
        ErrorCode::LineageBroken => "known",
        ErrorCode::LineageCyclic => "known",
        ErrorCode::UnsignedManifest => "known",
        ErrorCode::ContentHashMismatch => "known",
        ErrorCode::UnknownSigner => "known",
        ErrorCode::EpochDriftExceeded => "known",
        ErrorCode::Level1InvalidSignature => "known",
        ErrorCode::StreamDescriptorTruncated => "known",
        ErrorCode::StreamBadMagic => "known",
        ErrorCode::StreamUnsupportedVersion => "known",
        ErrorCode::StreamBadFlags => "known",
        ErrorCode::StreamLimitExceeded => "known",
        ErrorCode::StreamTableInvalid => "known",
        ErrorCode::StreamBadSignature => "known",
        ErrorCode::StreamStaleEpoch => "known",
        ErrorCode::StreamEpochConflict => "known",
        ErrorCode::StreamExpired => "known",
        ErrorCode::StreamPolicyMismatch => "known",
        ErrorCode::StreamChunkHashMismatch => "known",
        ErrorCode::StreamTableHashMismatch => "known",
        ErrorCode::StreamUnexpectedFrame => "known",
        ErrorCode::StreamFrameSizeLie => "known",
        ErrorCode::StreamRequirementsUnmet => "known",
        ErrorCode::StreamIdentityMismatch => "known",
        ErrorCode::StreamWeakChecksum => "known",
        ErrorCode::StreamPoisoned => "known",
        ErrorCode::StreamBadState => "known",
        ErrorCode::StreamDuplicateFlood => "known",
        ErrorCode::FreshnessRequired => "known",
        ErrorCode::NonFiniteQuery => "known",
        ErrorCode::QualityBelowThreshold => "known",
        ErrorCode::BudgetTokensExhausted => "known",
        ErrorCode::QueryBlacklisted => "known",
        ErrorCode::CowMapCorrupt => "known",
        ErrorCode::ClusterNotFound => "known",
        ErrorCode::ParentChainBroken => "known",
        ErrorCode::DeltaThresholdExceeded => "known",
        ErrorCode::SnapshotFrozen => "known",
        ErrorCode::MembershipInvalid => "known",
        ErrorCode::GenerationStale => "known",
        ErrorCode::KernelBindingMismatch => "known",
        ErrorCode::DoubleRootCorrupt => "known",
        // Required by #[non_exhaustive]; unknown future codes fail closed.
        _ => "unknown",
    }
}

fn class_name(c: StreamClass) -> &'static str {
    match c {
        StreamClass::Executable => "executable",
        StreamClass::Model => "model",
        StreamClass::Index => "index",
        StreamClass::State => "state",
        _ => "unknown",
    }
}

#[test]
fn error_code_is_non_exhaustive_and_wildcard_is_reachable() {
    assert_eq!(classify(ErrorCode::Ok), "known");
    assert_eq!(classify(ErrorCode::FreshnessRequired), "known");
}

#[test]
fn stream_class_is_non_exhaustive() {
    assert_eq!(class_name(StreamClass::of(0x06)), "model");
}

#[test]
fn rvf_error_and_friends_are_non_exhaustive() {
    use rvf_types::RvfError;
    let name = |e: RvfError| match e {
        RvfError::Code(_) => "code",
        RvfError::UnknownCode(_) => "unknown",
        RvfError::BadMagic { .. } => "magic",
        RvfError::SizeMismatch { .. } => "size",
        RvfError::InvalidEnumValue { .. } => "enum",
        RvfError::Security(_) => "security",
        RvfError::QualityBelowThreshold { .. } => "quality",
        _ => "future",
    };
    assert_eq!(name(RvfError::UnknownCode(1)), "unknown");
}
