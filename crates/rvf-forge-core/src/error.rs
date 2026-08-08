//! Stable error codes shared with the `@ruvector/forge` CLI.
//!
//! The CLI is required to return stable machine-readable error codes
//! (ADR-283 §4). [`ErrorCode`] is that vocabulary; [`ForgeError`] is the Rust
//! carrier. The string form of a code is part of the public contract and must
//! not change once released.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Machine-readable error code. The wire form is the SCREAMING_SNAKE_CASE
/// string returned by [`ErrorCode::as_str`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    /// The bytes are not a well-formed RVF container.
    InvalidRvf,
    /// An executable segment carries no signature and unsigned executables are
    /// not permitted (the default).
    UnsignedSegment,
    /// A hash or signature check failed.
    VerifyFailed,
    /// A build manifest could not be produced or canonicalized.
    Manifest,
    /// A requested target is absent from the published compatibility matrix.
    UnsupportedTarget,
    /// Filesystem or other I/O failure.
    Io,
}

impl ErrorCode {
    /// The stable wire string for this code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRvf => "INVALID_RVF",
            Self::UnsignedSegment => "UNSIGNED_SEGMENT",
            Self::VerifyFailed => "VERIFY_FAILED",
            Self::Manifest => "MANIFEST",
            Self::UnsupportedTarget => "UNSUPPORTED_TARGET",
            Self::Io => "IO",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Every failure surfaced by this crate.
///
/// Each variant maps to exactly one [`ErrorCode`], reachable via
/// [`ForgeError::code`].
#[derive(Debug, thiserror::Error)]
pub enum ForgeError {
    /// The bytes are not a well-formed RVF container.
    #[error("INVALID_RVF: {detail}")]
    InvalidRvf {
        /// What was malformed.
        detail: String,
    },

    /// An executable segment (WASM, kernel, or eBPF) is unsigned.
    #[error("UNSIGNED_SEGMENT: {segment_type} segment {segment_id} is unsigned")]
    UnsignedSegment {
        /// The segment's ordinal from its header.
        segment_id: u64,
        /// Human-readable segment type name.
        segment_type: String,
    },

    /// A hash or signature verification failed.
    #[error("VERIFY_FAILED: {detail}")]
    VerifyFailed {
        /// Which check failed and why.
        detail: String,
    },

    /// A build manifest could not be produced or canonicalized.
    #[error("MANIFEST: {detail}")]
    Manifest {
        /// What made the manifest invalid.
        detail: String,
    },

    /// A requested target is absent from the published compatibility matrix.
    #[error("UNSUPPORTED_TARGET: {target} ({detail})")]
    UnsupportedTarget {
        /// The rejected target identifier as supplied by the caller.
        target: String,
        /// Why it was rejected, naming the nearest supported alternative where
        /// one exists (ADR-291 §2).
        detail: String,
    },

    /// Filesystem or other I/O failure.
    #[error("IO: {path}: {detail}")]
    Io {
        /// The path being operated on.
        path: String,
        /// The underlying error.
        detail: String,
    },
}

impl ForgeError {
    /// The stable code for this error.
    pub const fn code(&self) -> ErrorCode {
        match self {
            Self::InvalidRvf { .. } => ErrorCode::InvalidRvf,
            Self::UnsignedSegment { .. } => ErrorCode::UnsignedSegment,
            Self::VerifyFailed { .. } => ErrorCode::VerifyFailed,
            Self::Manifest { .. } => ErrorCode::Manifest,
            Self::UnsupportedTarget { .. } => ErrorCode::UnsupportedTarget,
            Self::Io { .. } => ErrorCode::Io,
        }
    }

    /// Build an [`ForgeError::Io`] from a `std::io::Error` and the path it
    /// applies to.
    pub fn io(path: impl AsRef<std::path::Path>, err: &std::io::Error) -> Self {
        Self::Io {
            path: path.as_ref().display().to_string(),
            detail: err.to_string(),
        }
    }

    /// Build an [`ForgeError::InvalidRvf`] from any displayable cause.
    pub fn invalid_rvf(detail: impl fmt::Display) -> Self {
        Self::InvalidRvf {
            detail: detail.to_string(),
        }
    }
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ForgeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_stable_strings() {
        assert_eq!(ErrorCode::InvalidRvf.as_str(), "INVALID_RVF");
        assert_eq!(ErrorCode::UnsignedSegment.as_str(), "UNSIGNED_SEGMENT");
        assert_eq!(ErrorCode::VerifyFailed.as_str(), "VERIFY_FAILED");
        assert_eq!(ErrorCode::Manifest.as_str(), "MANIFEST");
        assert_eq!(ErrorCode::UnsupportedTarget.as_str(), "UNSUPPORTED_TARGET");
        assert_eq!(ErrorCode::Io.as_str(), "IO");
    }

    #[test]
    fn every_variant_maps_to_its_code() {
        let cases: Vec<(ForgeError, ErrorCode)> = vec![
            (ForgeError::invalid_rvf("x"), ErrorCode::InvalidRvf),
            (
                ForgeError::UnsignedSegment {
                    segment_id: 1,
                    segment_type: "wasm".into(),
                },
                ErrorCode::UnsignedSegment,
            ),
            (
                ForgeError::VerifyFailed { detail: "x".into() },
                ErrorCode::VerifyFailed,
            ),
            (
                ForgeError::Manifest { detail: "x".into() },
                ErrorCode::Manifest,
            ),
            (
                ForgeError::UnsupportedTarget {
                    target: "solaris".into(),
                    detail: "x".into(),
                },
                ErrorCode::UnsupportedTarget,
            ),
            (
                ForgeError::Io {
                    path: "/tmp/x".into(),
                    detail: "x".into(),
                },
                ErrorCode::Io,
            ),
        ];
        for (err, code) in cases {
            assert_eq!(err.code(), code, "{err}");
            // The Display form always leads with the stable code.
            assert!(err.to_string().starts_with(code.as_str()), "{err}");
        }
    }

    #[test]
    fn code_serializes_to_wire_string() {
        let json = serde_json::to_string(&ErrorCode::UnsupportedTarget).unwrap();
        assert_eq!(json, "\"UNSUPPORTED_TARGET\"");
    }
}
