//! Fixtures for exercising the registry without a real publisher pipeline.
//!
//! Keys are derived from fixed 32-byte seeds rather than sampled from an RNG.
//! They are genuine Ed25519 keys — the signatures they produce are verified by
//! the same code path a real publisher's are — and deriving them keeps a failing
//! test reproducible from its seed alone.

use crate::crypto::{b64_encode, key_id_for, Signer, ED25519_ALG};
use crate::error::Result;
use crate::objects::{
    compute_id, CapabilityManifest, CapabilityRequest, Compatibility, Contact, DefaultPolicy,
    IdentityEvidence, IdentityMethod, Listing, ModelLocation, ModelManifest, PackageRef,
    PublicKeyEntry, PublisherRecord, RegistryObject, Release, Signature, SignatureRole, TrustLevel,
    DIGEST_PREFIX,
};
use ed25519_dalek::SigningKey;
use rvf_forge_core::canonical::hex_lower;

/// A deterministic Ed25519 key for `seed`.
pub fn signing_key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

/// A stable stand-in content address derived from `label`.
pub fn digest(label: &str) -> String {
    format!(
        "{DIGEST_PREFIX}{}",
        hex_lower(&crate::merkle::sha256(label.as_bytes()))
    )
}

/// A valid, narrow-scope capability manifest.
pub fn capability_manifest() -> CapabilityManifest {
    CapabilityManifest {
        requests: vec![
            CapabilityRequest {
                class: "filesystem".into(),
                scope: "user-selected".into(),
                rationale: "reads documents you choose".into(),
            },
            CapabilityRequest {
                class: "memory".into(),
                scope: "512MiB".into(),
                rationale: "model working set".into(),
            },
        ],
        denials: vec!["network".into(), "microphone".into(), "process".into()],
        default_policy: DefaultPolicy::Deny,
        ..Default::default()
    }
}

/// A publisher record listing `keys` as live signing keys.
pub fn publisher_record(display_name: &str, keys: &[&SigningKey]) -> PublisherRecord {
    PublisherRecord {
        display_name: display_name.into(),
        public_keys: keys
            .iter()
            .map(|sk| PublicKeyEntry {
                key_id: key_id_for(&sk.verifying_key()),
                alg: ED25519_ALG.into(),
                public_key: b64_encode(sk.verifying_key().as_bytes()),
                valid_from: "2026-08-03T00:00:00Z".into(),
                revoked_at: None,
            })
            .collect(),
        identity_evidence: IdentityEvidence {
            method: IdentityMethod::Dns,
            reference: "example.test".into(),
        },
        contact: Contact {
            support: "support@example.test".into(),
            privacy_policy: "https://example.test/privacy".into(),
        },
        ..Default::default()
    }
}

/// An unsigned release of `name` under `publisher_id`, referencing
/// `manifest_id`.
pub fn release(publisher_id: &str, name: &str, version: &str, manifest_id: &str) -> Release {
    Release {
        package: PackageRef {
            name: name.into(),
            publisher_id: publisher_id.into(),
        },
        version: version.into(),
        predecessor: None,
        rvf_identity: digest(&format!("{name}-{version}-rvf")),
        rvf_size: 4_096,
        capability_manifest: manifest_id.into(),
        runtime_profiles: vec!["wasm".into(), "rvm".into()],
        compatibility: Compatibility {
            rvm_version_min: "0.4.0".into(),
            state_schema_version: 1,
            witness_schema_version: 1,
        },
        model_manifest: ModelManifest {
            location: ModelLocation::Embedded,
            digests: vec![digest(&format!("{name}-model"))],
        },
        software_inventory: digest(&format!("{name}-sbom")),
        evaluation_report: None,
        security_report: None,
        provenance: digest(&format!("{name}-{version}-provenance")),
        trust_level: TrustLevel::Published,
        rollback_safe_until_state_schema: 2,
        listing: Listing {
            category: "Developer Tools".into(),
            description: "An example agent.".into(),
            price_model: "free".into(),
        },
        published_at: "2026-08-03T00:00:00Z".into(),
        ..Default::default()
    }
}

/// Attach a signature over the object's computed id.
///
/// # Errors
///
/// As [`compute_id`].
pub fn sign<T: RegistryObject>(
    mut object: T,
    signer: &dyn Signer,
    role: SignatureRole,
) -> Result<T> {
    let id = compute_id(&object)?;
    let mut signatures = object.signatures().to_vec();
    signatures.push(Signature {
        key_id: signer.key_id(),
        role,
        sig: b64_encode(&signer.sign_bytes(id.as_bytes())),
    });
    object.set_signatures(signatures);
    Ok(object)
}
