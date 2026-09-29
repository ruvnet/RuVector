//! `tenant_key` and Durable Object name derivation (ADR-351 §4.1, §4.2).

use crate::validate::CollectionName;
use core::fmt;

/// Length of a tenant key in base32 characters.
pub const TENANT_KEY_LEN: usize = 26;

/// Opaque tenant identifier:
/// `base32(sha256("v1|" + iss + "|" + org_id + "|" + workspace_id))[0..26]`,
/// RFC 4648 alphabet, lowercase, unpadded. Only constructible via
/// [`derive_tenant_key`] (called by `TenantContext::from_verified`, i.e.
/// after signature verification and component validation).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TenantKey(String);

impl TenantKey {
    /// The 26-char key.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TenantKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Derive the tenant key. Inputs must already be validated (the `|`
/// delimiter is excluded from components by
/// [`crate::validate_claim_component`]; `iss` is an exact configured issuer).
pub fn derive_tenant_key(iss: &str, org_id: &str, workspace_id: &str) -> TenantKey {
    use sha2::{Digest, Sha256};
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
    TenantKey(enc[..TENANT_KEY_LEN].to_string())
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
    /// Stable wire name used in the DO-name preimage.
    pub fn as_str(self) -> &'static str {
        match self {
            Service::Vector => "vector",
            Service::Quant => "quant",
            Service::Graph => "graph",
            Service::Mincut => "mincut",
        }
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

/// `hex(sha256("v1|" + tenant_key + "|" + service + "|" + collection + "|" + shard))`.
pub fn do_name(
    tenant: &TenantKey,
    service: Service,
    collection: &CollectionName,
    shard: u32,
) -> DoName {
    use sha2::{Digest, Sha256};
    let preimage = format!(
        "v1|{}|{}|{}|{}",
        tenant.as_str(),
        service.as_str(),
        collection.as_str(),
        shard
    );
    DoName(hex::encode(Sha256::digest(preimage.as_bytes())))
}

/// Name of the per-tenant `TenantLedger` DO:
/// `hex(sha256("v1|" + tenant_key + "|ledger"))`.
pub fn ledger_do_name(tenant: &TenantKey) -> DoName {
    use sha2::{Digest, Sha256};
    let preimage = format!("v1|{}|ledger", tenant.as_str());
    DoName(hex::encode(Sha256::digest(preimage.as_bytes())))
}
