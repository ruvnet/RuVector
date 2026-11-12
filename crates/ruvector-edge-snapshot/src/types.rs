//! Shared value types, boundary validators and the clock port.

use crate::error::SnapshotError;

/// `tenant_key` length (ADR-351 §4.1: base32lower of sha256, 26 chars).
pub const TENANT_KEY_LEN: usize = 26;
/// `collection_uid` in hex (16 bytes).
pub const COLLECTION_UID_HEX_LEN: usize = 32;
/// Maximum vector id length in bytes (ADR-351 §7: `id ≤ 256B`).
pub const MAX_ID_BYTES: usize = 256;
/// Maximum metadata JSON length in bytes (ADR-351 §7: `metadata ≤ 4KiB`).
pub const MAX_METADATA_BYTES: usize = 4096;
/// Maximum vector dimension accepted anywhere in this crate.
pub const MAX_DIM: u16 = 4096;
/// Maximum service slug length.
pub const MAX_SERVICE_LEN: usize = 32;

/// Collection distance metric (ADR-351 §7: `cosine | l2 | dot`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    /// `1 - cos(a, b)`.
    Cosine,
    /// Euclidean.
    L2,
    /// Negated inner product.
    Dot,
}

impl Metric {
    /// Code used in snapshot manifests.
    pub fn code(self) -> u8 {
        match self {
            Metric::Cosine => 0,
            Metric::L2 => 1,
            Metric::Dot => 2,
        }
    }
    /// Inverse of [`Metric::code`].
    pub fn from_code(c: u8) -> Option<Metric> {
        match c {
            0 => Some(Metric::Cosine),
            1 => Some(Metric::L2),
            2 => Some(Metric::Dot),
            _ => None,
        }
    }
    /// The rvf-runtime manifest byte `[19]` (0 = L2, 1 = InnerProduct, 2 = Cosine).
    pub fn rvf_id(self) -> u8 {
        match self {
            Metric::L2 => 0,
            Metric::Dot => 1,
            Metric::Cosine => 2,
        }
    }
    /// Inverse of [`Metric::rvf_id`]; unknown ids are refused (the runtime
    /// silently maps them to L2, which would import with the wrong metric).
    pub fn from_rvf_id(id: u8) -> Option<Metric> {
        match id {
            0 => Some(Metric::L2),
            1 => Some(Metric::Dot),
            2 => Some(Metric::Cosine),
            _ => None,
        }
    }
    /// Wire form.
    pub fn as_str(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::L2 => "l2",
            Metric::Dot => "dot",
        }
    }
}

/// One stored vector row, as it leaves / enters a `VectorShard`.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// Caller-chosen id (UTF-8, 1..=256 bytes).
    pub id: String,
    /// Vector values (exactly `dim`, finite).
    pub values: Vec<f32>,
    /// Canonical metadata JSON text (≤ 4 KiB), if any.
    pub metadata: Option<String>,
}

impl Row {
    /// Bit-exact equality (`f32` compared by bits, so `-0.0 ≠ 0.0`).
    pub fn bitwise_eq(&self, other: &Row) -> bool {
        self.id == other.id
            && self.metadata == other.metadata
            && self.values.len() == other.values.len()
            && self
                .values
                .iter()
                .zip(&other.values)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    /// Validate against a collection dimension; `Err` carries a static reason.
    pub fn check(&self, dim: u16) -> Result<(), &'static str> {
        check_parts(&self.id, &self.values, self.metadata.as_deref(), dim)
    }
}

/// Row validation on borrowed parts (used by the decoders before allocating a [`Row`]).
pub fn check_parts(
    id: &str,
    values: &[f32],
    metadata: Option<&str>,
    dim: u16,
) -> Result<(), &'static str> {
    if id.is_empty() || id.len() > MAX_ID_BYTES {
        return Err("id length");
    }
    if id.chars().any(char::is_control) {
        return Err("id control character");
    }
    if values.len() != usize::from(dim) {
        return Err("dimension");
    }
    if !values.iter().all(|v| v.is_finite()) {
        return Err("non-finite value");
    }
    if metadata.is_some_and(|m| m.len() > MAX_METADATA_BYTES) {
        return Err("metadata size");
    }
    Ok(())
}

/// Clock port: the Worker passes `Date.now()`; tests pass a fixed value.
pub trait Clock {
    /// Milliseconds since the Unix epoch.
    fn now_ms(&self) -> u64;
}

/// A clock that always returns the same instant.
#[derive(Debug, Clone, Copy)]
pub struct FixedClock(pub u64);

impl Clock for FixedClock {
    fn now_ms(&self) -> u64 {
        self.0
    }
}

/// Where a snapshot belongs: every component is server-derived (ADR-351 §6.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardRef {
    /// Caller tenant (26 chars `[a-z2-7]`).
    pub tenant_key: String,
    /// Service slug (`vector`, …).
    pub service: String,
    /// Collection uid (32 lowercase hex).
    pub collection_uid: String,
    /// Shard number.
    pub shard: u16,
}

impl ShardRef {
    /// Validate every component at the boundary.
    pub fn new(
        tenant_key: &str,
        service: &str,
        collection_uid: &str,
        shard: u16,
    ) -> Result<Self, SnapshotError> {
        validate_tenant_key(tenant_key)?;
        validate_service(service)?;
        validate_collection_uid(collection_uid)?;
        Ok(ShardRef {
            tenant_key: tenant_key.to_string(),
            service: service.to_string(),
            collection_uid: collection_uid.to_string(),
            shard,
        })
    }
}

/// `tenant_key`: exactly 26 chars of `[a-z2-7]`.
pub fn validate_tenant_key(s: &str) -> Result<(), SnapshotError> {
    let ok = s.len() == TENANT_KEY_LEN
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || (b'2'..=b'7').contains(&b));
    ok.then_some(())
        .ok_or(SnapshotError::InvalidIdentifier("tenant_key"))
}

/// `collection_uid`: exactly 32 lowercase hex chars.
pub fn validate_collection_uid(s: &str) -> Result<(), SnapshotError> {
    let ok = s.len() == COLLECTION_UID_HEX_LEN
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    ok.then_some(())
        .ok_or(SnapshotError::InvalidIdentifier("collection_uid"))
}

/// Service slug: `[a-z0-9-]{1,32}`, not starting with `-`.
pub fn validate_service(s: &str) -> Result<(), SnapshotError> {
    let ok = !s.is_empty()
        && s.len() <= MAX_SERVICE_LEN
        && !s.starts_with('-')
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    ok.then_some(())
        .ok_or(SnapshotError::InvalidIdentifier("service"))
}

/// Opaque upload / job id: `[A-Za-z0-9_-]{1,64}`.
pub fn validate_opaque_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// sha256 helper.
pub fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}
