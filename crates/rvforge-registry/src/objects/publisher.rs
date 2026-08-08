//! `PublisherRecord` — publisher identity and the keys that may sign for it.

use super::{require_non_empty, RegistryObject, Signature, SCHEMA_VERSION};
use crate::crypto::{verifying_key_from_b64, ED25519_ALG, ED25519_KEY_ID_PREFIX};
use crate::error::{RegistryError, Result};
use serde::{Deserialize, Serialize};

/// The `type` value for publisher records.
pub const TYPE_PUBLISHER: &str = "publisher";

/// A publisher's identity, keys, evidence, and contact details.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublisherRecord {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Always [`TYPE_PUBLISHER`].
    #[serde(rename = "type")]
    pub object_type: String,
    /// This record's content address.
    pub publisher_id: String,
    /// Human-readable publisher name shown in the store.
    pub display_name: String,
    /// Every key that has ever signed for this publisher, revoked or not.
    pub public_keys: Vec<PublicKeyEntry>,
    /// How the identity was established.
    pub identity_evidence: IdentityEvidence,
    /// Support and privacy-policy references.
    pub contact: Contact,
    /// Registry countersignatures over [`Self::publisher_id`].
    #[serde(default)]
    pub signatures: Vec<Signature>,
}

/// One Ed25519 key belonging to a publisher.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicKeyEntry {
    /// Opaque key label, conventionally `ed25519:<hex>`.
    pub key_id: String,
    /// Signature algorithm; only `ed25519` is accepted.
    pub alg: String,
    /// Base64 of the raw 32-byte public key.
    pub public_key: String,
    /// RFC 3339 instant from which this key is valid.
    pub valid_from: String,
    /// RFC 3339 instant at which the key was revoked, or `null` while live.
    pub revoked_at: Option<String>,
}

impl PublicKeyEntry {
    /// Whether the key is still usable for new signatures.
    pub fn is_live(&self) -> bool {
        self.revoked_at.is_none()
    }
}

/// How a publisher's identity was established (ADR-294 §8).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct IdentityEvidence {
    /// The verification method used.
    pub method: IdentityMethod,
    /// The artifact that was checked: a domain, a GitHub handle, a case id.
    pub reference: String,
}

/// The identity-verification methods the model admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IdentityMethod {
    /// A DNS record under a domain the publisher controls.
    Dns,
    /// A GitHub account or organization.
    Github,
    /// A human reviewer signed off.
    ManualReview,
}

/// Publisher contact surfaces shown on every listing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Contact {
    /// Support contact (URL or address).
    pub support: String,
    /// Privacy-policy URL.
    pub privacy_policy: String,
}

impl PublisherRecord {
    /// The live (non-revoked) entry for `key_id`, if this record has one.
    pub fn live_key(&self, key_id: &str) -> Option<&PublicKeyEntry> {
        self.public_keys
            .iter()
            .find(|k| k.key_id == key_id && k.is_live())
    }

    /// Whether `key_id` appears in this record at all, revoked or not.
    pub fn knows_key(&self, key_id: &str) -> bool {
        self.public_keys.iter().any(|k| k.key_id == key_id)
    }
}

impl Default for PublisherRecord {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            object_type: TYPE_PUBLISHER.into(),
            publisher_id: String::new(),
            display_name: String::new(),
            public_keys: Vec::new(),
            identity_evidence: IdentityEvidence {
                method: IdentityMethod::ManualReview,
                reference: String::new(),
            },
            contact: Contact {
                support: String::new(),
                privacy_policy: String::new(),
            },
            signatures: Vec::new(),
        }
    }
}

impl RegistryObject for PublisherRecord {
    const TYPE: &'static str = TYPE_PUBLISHER;
    const ID_FIELD: &'static str = "publisherId";

    fn schema_version(&self) -> u32 {
        self.schema_version
    }
    fn object_type(&self) -> &str {
        &self.object_type
    }
    fn id(&self) -> &str {
        &self.publisher_id
    }
    fn set_id(&mut self, id: String) {
        self.publisher_id = id;
    }
    fn signatures(&self) -> &[Signature] {
        &self.signatures
    }
    fn set_signatures(&mut self, signatures: Vec<Signature>) {
        self.signatures = signatures;
    }

    fn validate_fields(&self) -> Result<()> {
        require_non_empty("displayName", &self.display_name)?;
        require_non_empty(
            "identityEvidence.reference",
            &self.identity_evidence.reference,
        )?;
        require_non_empty("contact.support", &self.contact.support)?;
        require_non_empty("contact.privacyPolicy", &self.contact.privacy_policy)?;
        if self.public_keys.is_empty() {
            return Err(RegistryError::invalid(
                "publisher record has no publicKeys; it could never sign a release",
            ));
        }
        for key in &self.public_keys {
            if key.alg != ED25519_ALG {
                return Err(RegistryError::invalid(format!(
                    "publicKeys[{}].alg is {:?}, only {ED25519_ALG:?} is supported",
                    key.key_id, key.alg
                )));
            }
            if !key.key_id.starts_with(ED25519_KEY_ID_PREFIX) {
                return Err(RegistryError::invalid(format!(
                    "keyId {:?} is not prefixed {ED25519_KEY_ID_PREFIX:?}",
                    key.key_id
                )));
            }
            require_non_empty("publicKeys[].validFrom", &key.valid_from)?;
            verifying_key_from_b64(&key.public_key)?;
        }
        let mut ids: Vec<&str> = self.public_keys.iter().map(|k| k.key_id.as_str()).collect();
        ids.sort_unstable();
        let unique = ids.len();
        ids.dedup();
        if ids.len() != unique {
            return Err(RegistryError::invalid(
                "publicKeys contains a duplicate keyId",
            ));
        }
        Ok(())
    }
}
