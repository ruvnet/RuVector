//! Tenancy errors.

use thiserror::Error;

/// Tenancy derivation / validation failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TenancyError {
    /// `org_id` / `workspace_id` missing or outside `^[A-Za-z0-9_-]{1,64}$`
    /// (maps to `401 invalid_token`).
    #[error("invalid tenant claim: {0}")]
    InvalidTenantClaim(&'static str),
    /// Collection name outside `^[a-z0-9][a-z0-9_-]{0,62}$` (400).
    #[error("invalid collection name")]
    InvalidCollectionName,
    /// Vector id not 1-256 bytes of UTF-8 without control characters (400).
    #[error("invalid vector id")]
    InvalidVectorId,
    /// Scaffold placeholder. Fails closed.
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
}
