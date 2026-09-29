//! `tenant_key` and Durable Object name derivation (ADR-351 §4.1, §4.3).
//!
//! All preimages are domain-separated by a `v1|` version prefix and a fixed
//! layout; every variable component is validated so it cannot contain the
//! `|` delimiter. Golden-value tests pin the outputs: changing any function
//! here re-keys every tenant and orphans every Durable Object.

use crate::error::TenancyError;
use crate::shard::ShardIndex;
use crate::uid::CollectionUid;
use crate::validate::{validate_claim_component, validate_issuer};
use core::fmt;
use sha2::{Digest, Sha256};

/// Length of a tenant key in base32 characters.
pub const TENANT_KEY_LEN: usize = 26;
/// Length of a DO name in hex characters.
pub const DO_NAME_LEN: usize = 64;

/// Opaque tenant identifier:
/// `base32(sha256("v1|" + iss + "|" + org_id + "|" + workspace_id))[0..26]`,
/// RFC 4648 alphabet, lowercase, unpadded (130 bits). Constructed only by
/// [`derive_tenant_key`] or by strict [`TenantKey::parse`] of a stored value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TenantKey(String);

impl TenantKey {
    /// The 26-char key.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Strictly parse a stored key (exactly 26 chars of `[a-z2-7]`). Used when
    /// reading a DO `meta` row; it does not re-derive anything.
    pub fn parse(stored: &str) -> Result<Self, TenancyError> {
        let ok = stored.len() == TENANT_KEY_LEN
            && stored
                .bytes()
                .all(|b| b.is_ascii_lowercase() || (b'2'..=b'7').contains(&b));
        if ok {
            Ok(TenantKey(stored.to_string()))
        } else {
            Err(TenancyError::MalformedIdentifier("tenant_key"))
        }
    }
}

impl fmt::Display for TenantKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Derive the tenant key from `(iss, org_id, workspace_id)`.
///
/// Validates every component first ([`validate_issuer`],
/// [`validate_claim_component`]), so a `|` in any value cannot shift the
/// preimage boundaries: `("a|b","c")` and `("a","b|c")` are both rejected
/// rather than colliding.
pub fn derive_tenant_key(
    iss: &str,
    org_id: &str,
    workspace_id: &str,
) -> Result<TenantKey, TenancyError> {
    validate_issuer(iss)?;
    validate_claim_component(org_id, "org_id")?;
    validate_claim_component(workspace_id, "workspace_id")?;
    let mut h = Sha256::new();
    h.update(b"v1|");
    h.update(iss.as_bytes());
    h.update(b"|");
    h.update(org_id.as_bytes());
    h.update(b"|");
    h.update(workspace_id.as_bytes());
    let enc = data_encoding::BASE32_NOPAD
        .encode(&h.finalize())
        .to_ascii_lowercase();
    Ok(TenantKey(enc[..TENANT_KEY_LEN].to_string()))
}

/// Service component of a DO name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Service {
    /// rv-vector `VectorShard`.
    Vector,
    /// rv-quant `QuantShard` (M4).
    Quant,
    /// rv-graph `GraphStore` (M4).
    Graph,
    /// rv-mincut `AnalyticsJob` (M4).
    Mincut,
}

impl Service {
    /// Every service, for exhaustive tests.
    pub const ALL: [Service; 4] = [
        Service::Vector,
        Service::Quant,
        Service::Graph,
        Service::Mincut,
    ];

    /// Stable wire name used in the DO-name preimage.
    pub fn as_str(self) -> &'static str {
        match self {
            Service::Vector => "vector",
            Service::Quant => "quant",
            Service::Graph => "graph",
            Service::Mincut => "mincut",
        }
    }

    /// Strictly parse a stored wire name.
    pub fn parse(stored: &str) -> Result<Self, TenancyError> {
        Service::ALL
            .into_iter()
            .find(|s| s.as_str() == stored)
            .ok_or(TenancyError::MalformedIdentifier("service"))
    }
}

/// A Durable Object name for `idFromName` (64 lowercase hex chars). Never
/// use `newUniqueId`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DoName(String);

impl DoName {
    /// The hex name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DoName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// `hex(sha256("v1|" + tenant_key + "|" + service + "|" + collection_uid + "|" + shard))`
/// (ADR §4.3). Keyed by the never-reused `collection_uid`, not the name, so a
/// delete-then-recreate of the same name lands on a fresh, empty DO.
pub fn do_name(
    tenant: &TenantKey,
    service: Service,
    collection: &CollectionUid,
    shard: ShardIndex,
) -> DoName {
    let preimage = format!(
        "v1|{}|{}|{}|{}",
        tenant.as_str(),
        service.as_str(),
        collection.to_hex(),
        shard.get()
    );
    DoName(hex::encode(Sha256::digest(preimage.as_bytes())))
}

/// Name of the per-tenant `TenantLedger` DO:
/// `hex(sha256("v1|" + tenant_key + "|ledger"))`. Cannot collide with a
/// [`do_name`] preimage, which always has five `|`-separated fields.
pub fn ledger_do_name(tenant: &TenantKey) -> DoName {
    let preimage = format!("v1|{}|ledger", tenant.as_str());
    DoName(hex::encode(Sha256::digest(preimage.as_bytes())))
}
