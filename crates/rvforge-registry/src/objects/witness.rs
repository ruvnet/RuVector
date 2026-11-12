//! `WitnessReceipt` — the per-subject hash chain of registry operations.

use super::{require_digest, require_non_empty, RegistryObject, Signature, SCHEMA_VERSION};
use crate::error::Result;
use serde::{Deserialize, Serialize};

/// The `type` value for witness receipts.
pub const TYPE_WITNESS_RECEIPT: &str = "witness-receipt";

/// One link in a subject's witness chain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WitnessReceipt {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Always [`TYPE_WITNESS_RECEIPT`].
    #[serde(rename = "type")]
    pub object_type: String,
    /// This receipt's content address.
    pub receipt_id: String,
    /// The object this receipt is about.
    pub subject: String,
    /// What happened.
    pub event: WitnessEvent,
    /// How it came out.
    pub outcome: Outcome,
    /// Who performed the operation.
    pub actor: Actor,
    /// Free-form detail shown in the witness viewer.
    pub evidence: WitnessEvidence,
    /// RFC 3339 instant, supplied by the caller.
    pub timestamp: String,
    /// The previous receipt for the same subject, or `null` for the first.
    pub prev_receipt: Option<String>,
    /// Registry signatures over [`Self::receipt_id`], when a signer is
    /// configured.
    #[serde(default)]
    pub signatures: Vec<Signature>,
}

/// The events a receipt can record.
///
/// `publish` and `revoke` extend the list in `registry-model.md`, which names
/// only the reader- and builder-side events. ADR-294 §9 requires the registry's
/// own privileged operations to produce witness records too, and folding them
/// into `update` would make a publication indistinguishable from a version bump
/// in the chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WitnessEvent {
    /// A build produced the artifact.
    Build,
    /// Signatures and hashes were checked.
    Verify,
    /// A user installed the release.
    Install,
    /// An installed release moved to a new version.
    Update,
    /// A capability was granted at install time.
    CapabilityGrant,
    /// A capability request was refused.
    CapabilityDenial,
    /// A revocation check ran.
    RevocationCheck,
    /// The registry accepted a release into the transparency log.
    Publish,
    /// The registry issued a revocation.
    Revoke,
}

/// The outcome of a witnessed operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    /// The operation succeeded.
    Pass,
    /// The operation failed.
    Fail,
    /// The operation was refused by policy.
    Denied,
}

/// Who performed a witnessed operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Actor {
    /// The kind of component acting.
    pub kind: ActorKind,
    /// An identifier for that component.
    pub id: String,
}

/// The component kinds that can produce receipts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActorKind {
    /// A build worker.
    Builder,
    /// The desktop Reader.
    Reader,
    /// The registry itself.
    Registry,
    /// The RVM runtime.
    Rvm,
}

/// Human-readable evidence attached to a receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WitnessEvidence {
    /// What was observed.
    pub details: String,
}

impl Default for WitnessReceipt {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            object_type: TYPE_WITNESS_RECEIPT.into(),
            receipt_id: String::new(),
            subject: String::new(),
            event: WitnessEvent::Verify,
            outcome: Outcome::Pass,
            actor: Actor {
                kind: ActorKind::Registry,
                id: String::new(),
            },
            evidence: WitnessEvidence {
                details: String::new(),
            },
            timestamp: String::new(),
            prev_receipt: None,
            signatures: Vec::new(),
        }
    }
}

impl RegistryObject for WitnessReceipt {
    const TYPE: &'static str = TYPE_WITNESS_RECEIPT;
    const ID_FIELD: &'static str = "receiptId";

    fn schema_version(&self) -> u32 {
        self.schema_version
    }
    fn object_type(&self) -> &str {
        &self.object_type
    }
    fn id(&self) -> &str {
        &self.receipt_id
    }
    fn set_id(&mut self, id: String) {
        self.receipt_id = id;
    }
    fn signatures(&self) -> &[Signature] {
        &self.signatures
    }
    fn set_signatures(&mut self, signatures: Vec<Signature>) {
        self.signatures = signatures;
    }

    fn validate_fields(&self) -> Result<()> {
        require_non_empty("subject", &self.subject)?;
        require_non_empty("actor.id", &self.actor.id)?;
        require_non_empty("timestamp", &self.timestamp)?;
        if let Some(prev) = &self.prev_receipt {
            require_digest("prevReceipt", prev)?;
        }
        Ok(())
    }
}
