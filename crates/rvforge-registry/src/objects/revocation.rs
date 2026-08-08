//! `Revocation` — blocks execution by policy, never deletes.
//!
//! ADR-294 §9 and requirements P12 are explicit that revocation must not remove
//! a locally owned RVF or a user's state. The type reflects that: [`effect`] has
//! exactly one value, [`RevocationEffect::BlockExecution`], and there is no
//! representable "delete" outcome for a caller to reach for.
//!
//! [`effect`]: Revocation::effect

use super::{require_digest, require_non_empty, RegistryObject, Signature, SCHEMA_VERSION};
use crate::crypto::ED25519_KEY_ID_PREFIX;
use crate::error::{RegistryError, Result};
use serde::{Deserialize, Serialize};

/// The `type` value for revocations.
pub const TYPE_REVOCATION: &str = "revocation";

/// A registry-signed statement that a subject must not execute.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Revocation {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Always [`TYPE_REVOCATION`].
    #[serde(rename = "type")]
    pub object_type: String,
    /// This revocation's content address.
    pub revocation_id: String,
    /// What is being revoked: a release id, a publisher id, or a `keyId`.
    pub subject: String,
    /// How to interpret [`Self::subject`].
    pub subject_type: SubjectType,
    /// Why the subject was revoked.
    pub reason: RevocationReason,
    /// Always [`RevocationEffect::BlockExecution`].
    pub effect: RevocationEffect,
    /// RFC 3339 issuance instant.
    pub issued_at: String,
    /// Registry signatures over [`Self::revocation_id`].
    #[serde(default)]
    pub signatures: Vec<Signature>,
}

/// How to interpret a revocation subject.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SubjectType {
    /// A single release id. The publisher's other releases are untouched.
    Release,
    /// One signing key. Releases signed by it stop being executable; the
    /// publisher's other keys keep working.
    PublisherKey,
    /// An entire publisher.
    Publisher,
}

/// Why a subject was revoked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RevocationReason {
    /// Key or build-system compromise.
    Compromise,
    /// The artifact was found to be malicious.
    Malware,
    /// A policy violation short of malware.
    Policy,
    /// The publisher asked for it.
    PublisherRequest,
}

/// What a revocation does. Deliberately a one-variant enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RevocationEffect {
    /// Execution is refused by policy. Reads, exports, and forensic access are
    /// preserved.
    BlockExecution,
}

impl Default for Revocation {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            object_type: TYPE_REVOCATION.into(),
            revocation_id: String::new(),
            subject: String::new(),
            subject_type: SubjectType::Release,
            reason: RevocationReason::Policy,
            effect: RevocationEffect::BlockExecution,
            issued_at: String::new(),
            signatures: Vec::new(),
        }
    }
}

impl RegistryObject for Revocation {
    const TYPE: &'static str = TYPE_REVOCATION;
    const ID_FIELD: &'static str = "revocationId";

    fn schema_version(&self) -> u32 {
        self.schema_version
    }
    fn object_type(&self) -> &str {
        &self.object_type
    }
    fn id(&self) -> &str {
        &self.revocation_id
    }
    fn set_id(&mut self, id: String) {
        self.revocation_id = id;
    }
    fn signatures(&self) -> &[Signature] {
        &self.signatures
    }
    fn set_signatures(&mut self, signatures: Vec<Signature>) {
        self.signatures = signatures;
    }

    fn validate_fields(&self) -> Result<()> {
        require_non_empty("issuedAt", &self.issued_at)?;
        match self.subject_type {
            SubjectType::Release | SubjectType::Publisher => {
                require_digest("subject", &self.subject)?;
            }
            SubjectType::PublisherKey => {
                if !self.subject.starts_with(ED25519_KEY_ID_PREFIX) {
                    return Err(RegistryError::invalid(format!(
                        "publisher-key revocation subject {:?} is not a keyId",
                        self.subject
                    )));
                }
            }
        }
        Ok(())
    }
}
