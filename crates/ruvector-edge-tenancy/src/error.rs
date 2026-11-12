//! Tenancy errors.

use crate::problem::ProblemCode;
use thiserror::Error;

/// Tenancy derivation / validation failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TenancyError {
    /// `iss`, `sub`, `org_id` or `workspace_id` missing or malformed
    /// (maps to `401 invalid_token`). The payload names the claim, never
    /// echoes its value.
    #[error("invalid tenant claim: {0}")]
    InvalidTenantClaim(&'static str),
    /// Collection name outside `^[a-z0-9][a-z0-9_-]{0,62}$` (400).
    #[error("invalid collection name")]
    InvalidCollectionName,
    /// Vector id not 1-256 bytes of UTF-8 without control characters (400).
    #[error("invalid vector id")]
    InvalidVectorId,
    /// A stored identifier (tenant key, uid, service, shard, DO meta row)
    /// failed strict parsing. Storage corruption or a forged value; the
    /// caller must fail closed (500).
    #[error("malformed stored identifier: {0}")]
    MalformedIdentifier(&'static str),
    /// Shard count outside `1..=MAX_SHARDS` or shard index out of range (400).
    #[error("invalid shard")]
    InvalidShard,
    /// The collection-uid sequence is exhausted (500; never expected).
    #[error("collection uid sequence exhausted")]
    UidSequenceExhausted,
    /// The caller's context does not match the object's identity (ADR §4.3).
    /// Deliberately indistinguishable from a missing resource: `404`.
    #[error("not found")]
    NotFound,
}

impl TenancyError {
    /// The RFC 9457 problem code this error maps to.
    pub fn problem_code(&self) -> ProblemCode {
        match self {
            TenancyError::InvalidTenantClaim(_) => ProblemCode::InvalidToken,
            TenancyError::InvalidCollectionName
            | TenancyError::InvalidVectorId
            | TenancyError::InvalidShard => ProblemCode::InvalidRequest,
            TenancyError::NotFound => ProblemCode::NotFound,
            TenancyError::MalformedIdentifier(_) | TenancyError::UidSequenceExhausted => {
                ProblemCode::ServerError
            }
        }
    }
}
