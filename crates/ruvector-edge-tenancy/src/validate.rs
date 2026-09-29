//! Boundary validators (ADR-351 §4.1 step 1, §4.2).

use crate::error::TenancyError;

/// Maximum length of an `org_id` / `workspace_id` claim component.
pub const MAX_CLAIM_COMPONENT_LEN: usize = 64;
/// Maximum collection-name length.
pub const MAX_COLLECTION_NAME_LEN: usize = 63;
/// Maximum vector-id length in bytes.
pub const MAX_VECTOR_ID_BYTES: usize = 256;

/// Validate an `org_id` / `workspace_id` value: `^[A-Za-z0-9_-]{1,64}$`.
/// Rejects `:`, `/`, `|`, NUL, non-ASCII and over-long values, so the `|`
/// delimiter in the tenant-key preimage cannot be forged.
pub fn validate_claim_component(value: &str, what: &'static str) -> Result<(), TenancyError> {
    let ok = !value.is_empty()
        && value.len() <= MAX_CLAIM_COMPONENT_LEN
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if ok {
        Ok(())
    } else {
        Err(TenancyError::InvalidTenantClaim(what))
    }
}

/// A validated collection (or graph) name: `^[a-z0-9][a-z0-9_-]{0,62}$`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CollectionName(String);

impl CollectionName {
    /// Parse and validate.
    pub fn parse(input: &str) -> Result<Self, TenancyError> {
        let bytes = input.as_bytes();
        let first_ok = bytes
            .first()
            .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit());
        let rest_ok = bytes
            .iter()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'_' || *b == b'-');
        if first_ok && rest_ok && bytes.len() <= MAX_COLLECTION_NAME_LEN {
            Ok(CollectionName(input.to_string()))
        } else {
            Err(TenancyError::InvalidCollectionName)
        }
    }

    /// The validated name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A validated vector id: 1-256 bytes of UTF-8, no control characters.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VectorId(String);

impl VectorId {
    /// Parse and validate.
    pub fn parse(input: &str) -> Result<Self, TenancyError> {
        if input.is_empty()
            || input.len() > MAX_VECTOR_ID_BYTES
            || input.chars().any(char::is_control)
        {
            return Err(TenancyError::InvalidVectorId);
        }
        Ok(VectorId(input.to_string()))
    }

    /// The validated id.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
