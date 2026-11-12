//! `TrustUpdate` — the only way a release's trust level can rise.
//!
//! `registry-model.md` records `trustLevel` on the release but gives no shape
//! for changing it, and a release is immutable, so the field cannot be edited
//! after publication. This object is the missing piece: a separate,
//! registry-signed record naming a release and the level its evidence now
//! supports. A publisher can produce a `TrustUpdate` object, but it will not be
//! accepted without a registry signature, which is what makes ADR-294 §7's
//! "evidence, not endorsement" enforceable rather than advisory.

use super::{
    require_digest, require_non_empty, RegistryObject, Signature, TrustLevel, SCHEMA_VERSION,
};
use crate::error::Result;
use serde::{Deserialize, Serialize};

/// The `type` value for trust updates.
pub const TYPE_TRUST_UPDATE: &str = "trust-update";

/// A registry-signed raise of one release's trust level.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrustUpdate {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Always [`TYPE_TRUST_UPDATE`].
    #[serde(rename = "type")]
    pub object_type: String,
    /// This update's content address.
    pub update_id: String,
    /// The release whose trust level is being raised.
    pub subject: String,
    /// The level the gathered evidence now supports.
    pub trust_level: TrustLevel,
    /// What evidence was gathered.
    pub evidence: TrustEvidence,
    /// RFC 3339 issuance instant.
    pub issued_at: String,
    /// Registry signatures over [`Self::update_id`]. At least one is required.
    #[serde(default)]
    pub signatures: Vec<Signature>,
}

/// What was done to justify the new level.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrustEvidence {
    /// How the evidence was produced: `automated-suite`, `human-review`,
    /// `org-approval`.
    pub method: String,
    /// A pointer to the evidence: a report id, a review case, a ticket.
    pub reference: String,
}

impl Default for TrustUpdate {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            object_type: TYPE_TRUST_UPDATE.into(),
            update_id: String::new(),
            subject: String::new(),
            trust_level: TrustLevel::Tested,
            evidence: TrustEvidence {
                method: String::new(),
                reference: String::new(),
            },
            issued_at: String::new(),
            signatures: Vec::new(),
        }
    }
}

impl RegistryObject for TrustUpdate {
    const TYPE: &'static str = TYPE_TRUST_UPDATE;
    const ID_FIELD: &'static str = "updateId";

    fn schema_version(&self) -> u32 {
        self.schema_version
    }
    fn object_type(&self) -> &str {
        &self.object_type
    }
    fn id(&self) -> &str {
        &self.update_id
    }
    fn set_id(&mut self, id: String) {
        self.update_id = id;
    }
    fn signatures(&self) -> &[Signature] {
        &self.signatures
    }
    fn set_signatures(&mut self, signatures: Vec<Signature>) {
        self.signatures = signatures;
    }

    fn validate_fields(&self) -> Result<()> {
        require_digest("subject", &self.subject)?;
        require_non_empty("evidence.method", &self.evidence.method)?;
        require_non_empty("evidence.reference", &self.evidence.reference)?;
        require_non_empty("issuedAt", &self.issued_at)?;
        Ok(())
    }
}
