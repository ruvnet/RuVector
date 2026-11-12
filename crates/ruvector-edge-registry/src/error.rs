//! Registry errors and their RFC 9457 problem codes (ADR-351 §7).

use crate::name::NameError;
use crate::validate::ValidationError;
use crate::version::VersionError;
use ruvector_edge_auth::Capability;
use ruvector_edge_store::StoreError;
use ruvector_edge_tenancy::ProblemCode;
use thiserror::Error;

/// Every registry failure. `detail` strings never carry payload bytes.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RegistryError {
    /// Malformed package name.
    #[error(transparent)]
    Name(#[from] NameError),
    /// Malformed version.
    #[error(transparent)]
    Version(#[from] VersionError),
    /// The uploaded bytes are not an acceptable RVF.
    #[error(transparent)]
    Validation(#[from] ValidationError),
    /// Absent, or invisible to the caller (cross-tenant non-public reads
    /// are indistinguishable from absence).
    #[error("not found")]
    NotFound,
    /// Visible, but the token lacks this capability (step-up challenge).
    #[error("insufficient scope: {} required", .0.satisfying_scope())]
    Forbidden(Capability),
    /// Visible, but owned by another tenant (or, for yank, uploaded by
    /// another member and the caller is not an admin).
    #[error("not the owner")]
    NotOwner,
    /// The version already exists (versions are immutable, even yanked).
    #[error("version already exists")]
    VersionExists,
    /// The package reached its version cap.
    #[error("too many versions")]
    TooManyVersions,
    /// The upload session is finalized, failed or aborted.
    #[error("upload is not open")]
    UploadNotOpen,
    /// The upload session expired.
    #[error("upload expired")]
    UploadExpired,
    /// The object's SHA-256 differs from the digest declared at begin.
    #[error("digest mismatch")]
    DigestMismatch,
    /// The object's size differs from the size declared at begin.
    #[error("size mismatch")]
    SizeMismatch,
    /// Parts are missing, not contiguous from 1, or do not sum to the size.
    #[error("upload parts incomplete: {0}")]
    PartsIncomplete(&'static str),
    /// A size or count limit on the upload.
    #[error("upload limit exceeded: {0}")]
    UploadLimit(&'static str),
    /// The scope has not been claimed (`POST /v1/rvf/scopes/{scope}`).
    #[error("scope not claimed")]
    ScopeUnclaimed,
    /// The scope, or a look-alike of it, is claimed by someone else.
    #[error("scope taken")]
    ScopeTaken,
    /// The tenant holds the maximum number of scopes.
    #[error("too many scopes")]
    TooManyScopes,
    /// The parts streamed at finalize are not the parts the plan froze.
    #[error("upload parts changed after finalize_plan")]
    PartsChanged,
    /// The Worker's measurement of a written blob does not match the
    /// version's digest and size (or is missing where required).
    #[error("blob evidence mismatch")]
    EvidenceMismatch,
    /// Any other malformed request.
    #[error("invalid request: {0}")]
    InvalidRequest(&'static str),
    /// A pagination cursor that was not issued for this listing.
    #[error("invalid cursor")]
    InvalidCursor,
    /// Storage failure.
    #[error(transparent)]
    Store(#[from] StoreError),
}

impl RegistryError {
    /// The ADR §7 problem code.
    pub fn problem_code(&self) -> ProblemCode {
        use RegistryError::*;
        match self {
            Name(_) | Version(_) | DigestMismatch | SizeMismatch | PartsIncomplete(_)
            | InvalidRequest(_) | InvalidCursor | PartsChanged | EvidenceMismatch => {
                ProblemCode::InvalidRequest
            }
            ScopeUnclaimed => ProblemCode::NotClaimed,
            ScopeTaken => ProblemCode::Conflict,
            TooManyScopes => ProblemCode::QuotaExceeded,
            Validation(v) if v.http_status() == 413 => ProblemCode::PayloadTooLarge,
            Validation(ValidationError::DimensionMismatch { .. }) => ProblemCode::DimensionMismatch,
            Validation(_) => ProblemCode::InvalidRequest,
            NotFound | UploadExpired => ProblemCode::NotFound,
            Forbidden(_) => ProblemCode::InsufficientScope,
            NotOwner => ProblemCode::RoleRequired,
            VersionExists | UploadNotOpen => ProblemCode::Conflict,
            TooManyVersions | UploadLimit(_) => ProblemCode::PayloadTooLarge,
            Store(StoreError::Corrupt(_)) => ProblemCode::ServerError,
            Store(_) => ProblemCode::ShardUnavailable,
        }
    }

    /// HTTP status of [`Self::problem_code`].
    pub fn http_status(&self) -> u16 {
        self.problem_code().status_and_code().0
    }

    /// For `Forbidden`, the scope a client should request.
    pub fn step_up_scope(&self) -> Option<&'static str> {
        match self {
            RegistryError::Forbidden(c) => Some(c.satisfying_scope()),
            _ => None,
        }
    }
}

/// Shorthand.
pub type Result<T, E = RegistryError> = core::result::Result<T, E>;
