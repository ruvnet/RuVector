//! The registry object model from `docs/research/rvf-forge/registry-model.md`.
//!
//! # Identity
//!
//! An object's id is `sha256:<hex>` of its canonical JSON **excluding the
//! `signatures` array**. The model document states that rule but leaves one
//! thing implicit: an object also carries its own id (`releaseId`,
//! `publisherId`, …), and a hash cannot cover the field that holds it. So
//! [`compute_id`] removes *both* `signatures` and the object's own id field
//! before hashing. Everything else — every declared field, including explicit
//! `null`s — is covered.
//!
//! The practical consequence is the one the model wants: two releases that
//! differ in any substantive field have different ids and are different
//! releases, while adding a registry countersignature to an existing release
//! leaves its id alone.

mod capability;
mod publisher;
mod release;
mod revocation;
mod transparency;
mod trust;
mod witness;

pub use capability::{
    CapabilityManifest, CapabilityRequest, DefaultPolicy, BROAD_SCOPES, TYPE_CAPABILITY_MANIFEST,
};
pub use publisher::{
    Contact, IdentityEvidence, IdentityMethod, PublicKeyEntry, PublisherRecord, TYPE_PUBLISHER,
};
pub use release::{
    Compatibility, Listing, ModelLocation, ModelManifest, PackageRef, Release, TrustLevel,
    TYPE_RELEASE,
};
pub use revocation::{
    Revocation, RevocationEffect, RevocationReason, SubjectType, TYPE_REVOCATION,
};
pub use transparency::{
    log_leaf_bytes, LogObjectType, TransparencyLogEntry, TreeHead, TYPE_LOG_ENTRY, TYPE_TREE_HEAD,
};
pub use trust::{TrustEvidence, TrustUpdate, TYPE_TRUST_UPDATE};
pub use witness::{
    Actor, ActorKind, Outcome, WitnessEvent, WitnessEvidence, WitnessReceipt, TYPE_WITNESS_RECEIPT,
};

use crate::error::{RegistryError, Result};
use rvf_forge_core::canonical::canonical_sha256_hex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// The schema version every object in this model carries.
pub const SCHEMA_VERSION: u32 = 1;

/// The prefix on every content address in the model.
pub const DIGEST_PREFIX: &str = "sha256:";

/// An Ed25519 signature over an object id.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Signature {
    /// The `keyId` of the signing key, as listed in the relevant publisher
    /// record (publisher role) or the registry's configured key (registry role).
    pub key_id: String,
    /// Whether this is the publisher's signature or the registry's
    /// countersignature (ADR-294 §9).
    pub role: SignatureRole,
    /// Base64 of the raw 64-byte Ed25519 signature over the object id.
    pub sig: String,
}

/// Who a signature speaks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignatureRole {
    /// The package publisher.
    Publisher,
    /// The registry, countersigning.
    Registry,
}

/// A content-addressed registry object.
///
/// Implementors are plain serde structs; this trait is the small amount of
/// reflection [`compute_id`] and the object store need in order to treat them
/// uniformly.
pub trait RegistryObject: Serialize + DeserializeOwned + Clone {
    /// The value of the object's `type` field.
    const TYPE: &'static str;
    /// The JSON name of the field holding this object's own id.
    const ID_FIELD: &'static str;

    /// The declared `schemaVersion`.
    fn schema_version(&self) -> u32;
    /// The declared `type`.
    fn object_type(&self) -> &str;
    /// The declared id; empty when the object has not been sealed yet.
    fn id(&self) -> &str;
    /// Set the id after computing it.
    fn set_id(&mut self, id: String);
    /// The signature array.
    fn signatures(&self) -> &[Signature];
    /// Replace the signature array (used when merging a countersignature).
    fn set_signatures(&mut self, signatures: Vec<Signature>);
    /// Type-specific structural checks, beyond envelope and id.
    fn validate_fields(&self) -> Result<()>;
}

/// Compute an object's content address.
///
/// # Errors
///
/// [`RegistryError::InvalidObject`] if the object does not serialize to a JSON
/// object or cannot be canonicalized.
pub fn compute_id<T: RegistryObject>(object: &T) -> Result<String> {
    let mut value = serde_json::to_value(object)
        .map_err(|e| RegistryError::invalid(format!("{} is not serializable: {e}", T::TYPE)))?;
    let map = value.as_object_mut().ok_or_else(|| {
        RegistryError::invalid(format!("{} did not serialize to an object", T::TYPE))
    })?;
    map.remove("signatures");
    map.remove(T::ID_FIELD);
    Ok(format!("{DIGEST_PREFIX}{}", canonical_sha256_hex(&value)?))
}

/// Validate an object, compute its id, and return it with the id set.
///
/// If the object already carries an id, it must equal the computed one.
///
/// # Errors
///
/// [`RegistryError::InvalidObject`] for a malformed object,
/// [`RegistryError::IdMismatch`] if a declared id disagrees with the content.
pub fn seal<T: RegistryObject>(mut object: T) -> Result<T> {
    validate(&object)?;
    let computed = compute_id(&object)?;
    if !object.id().is_empty() && object.id() != computed {
        return Err(RegistryError::IdMismatch {
            declared: object.id().to_string(),
            computed,
        });
    }
    object.set_id(computed);
    Ok(object)
}

/// Envelope and field validation, without touching the id.
///
/// # Errors
///
/// [`RegistryError::InvalidObject`] if `schemaVersion` or `type` are wrong, or
/// if a type-specific check fails.
pub fn validate<T: RegistryObject>(object: &T) -> Result<()> {
    if object.schema_version() != SCHEMA_VERSION {
        return Err(RegistryError::invalid(format!(
            "{} has schemaVersion {}, want {SCHEMA_VERSION}",
            T::TYPE,
            object.schema_version()
        )));
    }
    if object.object_type() != T::TYPE {
        return Err(RegistryError::invalid(format!(
            "expected type {:?}, got {:?}",
            T::TYPE,
            object.object_type()
        )));
    }
    for sig in object.signatures() {
        if sig.key_id.is_empty() || sig.sig.is_empty() {
            return Err(RegistryError::invalid(format!(
                "{} carries a signature with an empty keyId or sig",
                T::TYPE
            )));
        }
    }
    object.validate_fields()
}

/// Reject anything that is not `sha256:` followed by 64 lowercase hex digits.
///
/// # Errors
///
/// [`RegistryError::InvalidObject`] naming `field`.
pub fn require_digest(field: &str, value: &str) -> Result<()> {
    let hex = value.strip_prefix(DIGEST_PREFIX).ok_or_else(|| {
        RegistryError::invalid(format!(
            "{field} {value:?} is not prefixed {DIGEST_PREFIX:?}"
        ))
    })?;
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(RegistryError::invalid(format!(
            "{field} {value:?} is not 64 lowercase hex digits"
        )));
    }
    Ok(())
}

/// The 64-hex-digit tail of a content address, for use as a path component.
///
/// # Errors
///
/// As [`require_digest`].
pub fn digest_hex(field: &str, value: &str) -> Result<String> {
    require_digest(field, value)?;
    Ok(value[DIGEST_PREFIX.len()..].to_string())
}

/// Reject an empty required string.
///
/// # Errors
///
/// [`RegistryError::InvalidObject`] naming `field`.
pub fn require_non_empty(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(RegistryError::invalid(format!("{field} is required")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_ignores_signatures_but_not_content() {
        let base = crate::testkit::capability_manifest();
        let a = compute_id(&base).unwrap();

        let mut signed = base.clone();
        signed.signatures.push(Signature {
            key_id: "ed25519:aa".into(),
            role: SignatureRole::Publisher,
            sig: "zz".into(),
        });
        assert_eq!(
            compute_id(&signed).unwrap(),
            a,
            "signatures must not affect the id"
        );

        let mut changed = base;
        changed.denials.push("network".into());
        assert_ne!(
            compute_id(&changed).unwrap(),
            a,
            "content must affect the id"
        );
    }

    #[test]
    fn id_ignores_a_stale_self_id() {
        let mut m = crate::testkit::capability_manifest();
        let real = compute_id(&m).unwrap();
        m.manifest_id = format!("{DIGEST_PREFIX}{}", "9".repeat(64));
        assert_eq!(compute_id(&m).unwrap(), real);
    }

    #[test]
    fn seal_rejects_a_declared_id_that_does_not_match() {
        let mut m = crate::testkit::capability_manifest();
        m.manifest_id = format!("{DIGEST_PREFIX}{}", "0".repeat(64));
        let err = seal(m).unwrap_err();
        assert_eq!(err.code(), crate::RegistryErrorCode::IdMismatch);
    }

    #[test]
    fn seal_accepts_a_declared_id_that_matches() {
        let sealed = seal(crate::testkit::capability_manifest()).unwrap();
        let again = seal(sealed.clone()).unwrap();
        assert_eq!(sealed.manifest_id, again.manifest_id);
    }

    #[test]
    fn envelope_checks_catch_wrong_schema_and_type() {
        let mut m = crate::testkit::capability_manifest();
        m.schema_version = 2;
        assert_eq!(
            validate(&m).unwrap_err().code(),
            crate::RegistryErrorCode::InvalidObject
        );

        let mut m = crate::testkit::capability_manifest();
        m.object_type = "release".into();
        assert_eq!(
            validate(&m).unwrap_err().code(),
            crate::RegistryErrorCode::InvalidObject
        );
    }

    #[test]
    fn digest_shape_is_enforced() {
        let good = format!("{DIGEST_PREFIX}{}", "a".repeat(64));
        require_digest("x", &good).unwrap();
        assert_eq!(digest_hex("x", &good).unwrap(), "a".repeat(64));

        for bad in [
            "a".repeat(64),
            format!("{DIGEST_PREFIX}{}", "a".repeat(63)),
            format!("{DIGEST_PREFIX}{}", "A".repeat(64)),
            format!("{DIGEST_PREFIX}{}", "g".repeat(64)),
            format!("sha512:{}", "a".repeat(64)),
        ] {
            assert!(
                require_digest("x", &bad).is_err(),
                "{bad} should be rejected"
            );
        }
    }
}
