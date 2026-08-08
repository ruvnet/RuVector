//! Stable error codes for the RVForge registry.
//!
//! The vocabulary mirrors `rvf_forge_core::ErrorCode`: a SCREAMING_SNAKE_CASE
//! wire string per failure mode, one enum variant per code, and a `code()`
//! accessor so a caller can branch on the reason without parsing prose. The
//! strings are part of the public contract and must not change once released.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Machine-readable registry error code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RegistryErrorCode {
    /// An object is structurally invalid: wrong `type`, wrong `schemaVersion`,
    /// a missing required field, or a malformed digest.
    InvalidObject,
    /// The caller supplied an id that does not match the id computed from the
    /// object's own content.
    IdMismatch,
    /// A referenced object is not present in the store or the log.
    NotFound,
    /// An Ed25519 signature failed to verify, or a required signature role was
    /// absent.
    SignatureInvalid,
    /// The signing key is revoked, or is not listed in the publisher record it
    /// claims to belong to.
    KeyRevoked,
    /// A referenced dependency — publisher record or capability manifest — does
    /// not resolve in the store.
    MissingDependency,
    /// `predecessor` names a release that does not exist, was never published,
    /// or belongs to a different package.
    BadPredecessor,
    /// A trust level was asserted by the publisher, or a raise was not
    /// accompanied by a valid registry signature.
    TrustLevel,
    /// The subject is revoked and the requested operation is blocked by policy.
    Revoked,
    /// The transparency log is inconsistent, or an inclusion proof failed.
    Log,
    /// Filesystem or other I/O failure.
    Io,
}

impl RegistryErrorCode {
    /// The stable wire string for this code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidObject => "INVALID_OBJECT",
            Self::IdMismatch => "ID_MISMATCH",
            Self::NotFound => "NOT_FOUND",
            Self::SignatureInvalid => "SIGNATURE_INVALID",
            Self::KeyRevoked => "KEY_REVOKED",
            Self::MissingDependency => "MISSING_DEPENDENCY",
            Self::BadPredecessor => "BAD_PREDECESSOR",
            Self::TrustLevel => "TRUST_LEVEL",
            Self::Revoked => "REVOKED",
            Self::Log => "LOG",
            Self::Io => "IO",
        }
    }
}

impl fmt::Display for RegistryErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Every failure surfaced by this crate. Each variant maps to exactly one
/// [`RegistryErrorCode`], reachable via [`RegistryError::code`].
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// An object is structurally invalid.
    #[error("INVALID_OBJECT: {detail}")]
    InvalidObject {
        /// What was malformed.
        detail: String,
    },

    /// A supplied id disagrees with the id computed from the content.
    #[error("ID_MISMATCH: declared {declared}, computed {computed}")]
    IdMismatch {
        /// The id the caller put on the object.
        declared: String,
        /// The id the content actually hashes to.
        computed: String,
    },

    /// A referenced object is absent.
    #[error("NOT_FOUND: {kind} {id}")]
    NotFound {
        /// What kind of object was being looked up.
        kind: String,
        /// The id that did not resolve.
        id: String,
    },

    /// A signature failed to verify or a required role was absent.
    #[error("SIGNATURE_INVALID: {detail}")]
    SignatureInvalid {
        /// Which signature failed and why.
        detail: String,
    },

    /// The signing key is revoked or unknown to its publisher record.
    #[error("KEY_REVOKED: {key_id} ({detail})")]
    KeyRevoked {
        /// The key that was refused.
        key_id: String,
        /// Why it was refused.
        detail: String,
    },

    /// A dependency does not resolve.
    #[error("MISSING_DEPENDENCY: {kind} {id} does not resolve")]
    MissingDependency {
        /// What kind of dependency was missing.
        kind: String,
        /// The id that did not resolve.
        id: String,
    },

    /// `predecessor` is not a published release of the same package.
    #[error("BAD_PREDECESSOR: {predecessor} ({detail})")]
    BadPredecessor {
        /// The rejected predecessor id.
        predecessor: String,
        /// Why it was rejected.
        detail: String,
    },

    /// A trust level was claimed rather than earned.
    #[error("TRUST_LEVEL: {detail}")]
    TrustLevel {
        /// What made the trust-level transition illegal.
        detail: String,
    },

    /// The subject is revoked and execution is blocked by policy.
    #[error("REVOKED: {subject} is revoked ({detail})")]
    Revoked {
        /// The revoked subject.
        subject: String,
        /// Context for the refusal.
        detail: String,
    },

    /// The transparency log is inconsistent or a proof failed.
    #[error("LOG: {detail}")]
    Log {
        /// What was inconsistent.
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

impl RegistryError {
    /// The stable code for this error.
    pub const fn code(&self) -> RegistryErrorCode {
        match self {
            Self::InvalidObject { .. } => RegistryErrorCode::InvalidObject,
            Self::IdMismatch { .. } => RegistryErrorCode::IdMismatch,
            Self::NotFound { .. } => RegistryErrorCode::NotFound,
            Self::SignatureInvalid { .. } => RegistryErrorCode::SignatureInvalid,
            Self::KeyRevoked { .. } => RegistryErrorCode::KeyRevoked,
            Self::MissingDependency { .. } => RegistryErrorCode::MissingDependency,
            Self::BadPredecessor { .. } => RegistryErrorCode::BadPredecessor,
            Self::TrustLevel { .. } => RegistryErrorCode::TrustLevel,
            Self::Revoked { .. } => RegistryErrorCode::Revoked,
            Self::Log { .. } => RegistryErrorCode::Log,
            Self::Io { .. } => RegistryErrorCode::Io,
        }
    }

    /// Build an [`RegistryError::Io`] from a `std::io::Error` and its path.
    pub fn io(path: impl AsRef<std::path::Path>, err: &std::io::Error) -> Self {
        Self::Io {
            path: path.as_ref().display().to_string(),
            detail: err.to_string(),
        }
    }

    /// Build an [`RegistryError::InvalidObject`] from any displayable cause.
    pub fn invalid(detail: impl fmt::Display) -> Self {
        Self::InvalidObject {
            detail: detail.to_string(),
        }
    }
}

/// Canonical-JSON failures from `rvf-forge-core` surface as invalid objects:
/// an object that cannot be canonicalized cannot be content-addressed, so it
/// is not a registry object.
impl From<rvf_forge_core::ForgeError> for RegistryError {
    fn from(err: rvf_forge_core::ForgeError) -> Self {
        Self::InvalidObject {
            detail: err.to_string(),
        }
    }
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, RegistryError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_stable_strings() {
        assert_eq!(RegistryErrorCode::InvalidObject.as_str(), "INVALID_OBJECT");
        assert_eq!(RegistryErrorCode::IdMismatch.as_str(), "ID_MISMATCH");
        assert_eq!(
            RegistryErrorCode::BadPredecessor.as_str(),
            "BAD_PREDECESSOR"
        );
        assert_eq!(RegistryErrorCode::TrustLevel.as_str(), "TRUST_LEVEL");
    }

    #[test]
    fn display_form_leads_with_the_code() {
        let cases: Vec<RegistryError> = vec![
            RegistryError::invalid("x"),
            RegistryError::IdMismatch {
                declared: "a".into(),
                computed: "b".into(),
            },
            RegistryError::NotFound {
                kind: "release".into(),
                id: "x".into(),
            },
            RegistryError::SignatureInvalid { detail: "x".into() },
            RegistryError::KeyRevoked {
                key_id: "k".into(),
                detail: "x".into(),
            },
            RegistryError::MissingDependency {
                kind: "manifest".into(),
                id: "x".into(),
            },
            RegistryError::BadPredecessor {
                predecessor: "p".into(),
                detail: "x".into(),
            },
            RegistryError::TrustLevel { detail: "x".into() },
            RegistryError::Revoked {
                subject: "s".into(),
                detail: "x".into(),
            },
            RegistryError::Log { detail: "x".into() },
            RegistryError::Io {
                path: "/tmp/x".into(),
                detail: "x".into(),
            },
        ];
        for err in cases {
            assert!(err.to_string().starts_with(err.code().as_str()), "{err}");
        }
    }

    #[test]
    fn canonical_json_failure_becomes_invalid_object() {
        let err: RegistryError = rvf_forge_core::ForgeError::Manifest {
            detail: "floating-point".into(),
        }
        .into();
        assert_eq!(err.code(), RegistryErrorCode::InvalidObject);
    }
}
