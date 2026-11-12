//! Build provenance (requirements §3.10, ADR-283 invariant 5).
//!
//! Invariant 5 requires that build output be traceable to the input RVF,
//! runtime version, source revision, builder, and signing identity.
//! [`ProvenanceRecord`] is that trace, and [`provenance`] is the only way to
//! construct one, so every field the invariant names is non-optional.
//!
//! Timestamps are supplied by the caller rather than read from the clock. A
//! record is hashed and compared across the three workers of one build, so a
//! function that reached for `SystemTime::now` would make two records of the
//! same build differ for no reason and would be untestable besides. The caller
//! — which knows when its job actually started — passes RFC 3339 strings in.

use crate::canonical::{canonical_sha256_hex, to_canonical_json};
use crate::checksums::Sha256Manifest;
use crate::error::{ForgeError, Result};
use serde::{Deserialize, Serialize};

/// Schema identifier carried by every [`ProvenanceRecord`].
pub const PROVENANCE_SCHEMA: &str = "rvforge.provenance/1";

/// Which infrastructure ran the build.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BuilderKind {
    /// The hosted service, on an ephemeral per-job worker (ADR-290 §2).
    Hosted,
    /// An enterprise private builder inside the customer environment, where
    /// models, prompts, data, and signing credentials never leave the
    /// perimeter (ADR-290 §7).
    EnterprisePrivate,
    /// A developer machine.
    Local,
}

/// Which of the four signing postures of ADR-290 §5 produced the artifacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SigningPosture {
    /// A customer-supplied signing identity.
    CustomerSupplied,
    /// KMS or HSM signing, with no private key export.
    KmsOrHsm,
    /// Cognitum signing for verified marketplace packages.
    CognitumMarketplace,
    /// An unsigned development build. Labeled as such here and not eligible
    /// for marketplace distribution.
    UnsignedDevelopment,
}

impl SigningPosture {
    /// Whether artifacts under this posture are signed at all.
    pub const fn is_signed(self) -> bool {
        !matches!(self, Self::UnsignedDevelopment)
    }
}

/// Who built the artifacts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BuilderIdentity {
    /// Stable identifier for the builder, e.g. a service name or hostname.
    pub id: String,
    /// Which infrastructure it is.
    pub kind: BuilderKind,
    /// The ephemeral worker that ran this specific job, when there was one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<String>,
}

/// The signing identity that signed the artifacts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SigningIdentity {
    /// Which posture was used.
    pub posture: SigningPosture,
    /// Certificate subject, key ID, or other public identifier. Never key
    /// material: private signing keys are never exported (ADR-290 §5).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<String>,
}

/// Everything needed to build a [`ProvenanceRecord`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvenanceInputs {
    /// SHA-256 of the input RVF, lowercase hex.
    pub rvf_identity: String,
    /// SHA-256 of the canonical build manifest, from
    /// [`crate::CanonicalBuildManifest::manifest_hash`].
    pub build_manifest_hash: String,
    /// Source revision of the tree the build ran from.
    pub source_revision: String,
    /// Who built it.
    pub builder: BuilderIdentity,
    /// What signed it.
    pub signing: SigningIdentity,
    /// RFC 3339 start timestamp, supplied by the caller.
    pub started_at: String,
    /// RFC 3339 finish timestamp, supplied by the caller.
    pub finished_at: String,
    /// SHA-256 of every produced artifact.
    pub artifacts: Sha256Manifest,
    /// Revision of the compatibility matrix that admitted the build
    /// (ADR-291 acceptance criterion 4).
    pub compatibility_matrix_revision: Option<String>,
}

/// The provenance emitted alongside a build's artifacts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProvenanceRecord {
    /// Schema identifier; always [`PROVENANCE_SCHEMA`].
    pub schema: String,
    /// SHA-256 of the input RVF, lowercase hex.
    pub rvf_identity: String,
    /// Version of this crate, recorded so the producing code is identifiable.
    pub forge_core_version: String,
    /// SHA-256 of the canonical build manifest.
    pub build_manifest_hash: String,
    /// Source revision the build ran from.
    pub source_revision: String,
    /// Who built it.
    pub builder: BuilderIdentity,
    /// What signed it.
    pub signing: SigningIdentity,
    /// RFC 3339 start timestamp.
    pub started_at: String,
    /// RFC 3339 finish timestamp.
    pub finished_at: String,
    /// SHA-256 of every produced artifact, sorted by path.
    pub artifacts: Vec<crate::Sha256Entry>,
    /// Revision of the compatibility matrix that admitted the build.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compatibility_matrix_revision: Option<String>,
}

impl ProvenanceRecord {
    /// The canonical JSON encoding.
    ///
    /// # Errors
    ///
    /// [`ForgeError::Manifest`] if the record cannot be canonicalized.
    pub fn to_canonical_json(&self) -> Result<String> {
        to_canonical_json(self)
    }

    /// SHA-256 of the canonical JSON, lowercase hex.
    ///
    /// # Errors
    ///
    /// As [`ProvenanceRecord::to_canonical_json`].
    pub fn record_hash(&self) -> Result<String> {
        canonical_sha256_hex(self)
    }
}

/// Build a provenance record.
///
/// # Errors
///
/// [`ForgeError::Manifest`] if either hash is not a 64-character hex SHA-256,
/// if a required field is empty, or if the build produced no artifacts — a
/// build that emitted nothing has nothing to be traceable to (invariant 7).
pub fn provenance(inputs: &ProvenanceInputs) -> Result<ProvenanceRecord> {
    require_sha256("rvfIdentity", &inputs.rvf_identity)?;
    require_sha256("buildManifestHash", &inputs.build_manifest_hash)?;
    require_non_empty("sourceRevision", &inputs.source_revision)?;
    require_non_empty("builder.id", &inputs.builder.id)?;
    require_non_empty("startedAt", &inputs.started_at)?;
    require_non_empty("finishedAt", &inputs.finished_at)?;

    if inputs.artifacts.is_empty() {
        return Err(ForgeError::Manifest {
            detail: "provenance needs at least one output artifact".to_string(),
        });
    }

    let mut artifacts = inputs.artifacts.entries.clone();
    artifacts.sort();

    Ok(ProvenanceRecord {
        schema: PROVENANCE_SCHEMA.to_string(),
        rvf_identity: inputs.rvf_identity.to_ascii_lowercase(),
        forge_core_version: env!("CARGO_PKG_VERSION").to_string(),
        build_manifest_hash: inputs.build_manifest_hash.to_ascii_lowercase(),
        source_revision: inputs.source_revision.clone(),
        builder: inputs.builder.clone(),
        signing: inputs.signing.clone(),
        started_at: inputs.started_at.clone(),
        finished_at: inputs.finished_at.clone(),
        artifacts,
        compatibility_matrix_revision: inputs.compatibility_matrix_revision.clone(),
    })
}

fn require_sha256(field: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ForgeError::Manifest {
            detail: format!("{field} must be a 64-character hex SHA-256"),
        });
    }
    Ok(())
}

fn require_non_empty(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(ForgeError::Manifest {
            detail: format!("{field} must not be empty"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Sha256Entry, Sha256Manifest};

    fn artifacts() -> Sha256Manifest {
        Sha256Manifest {
            schema: crate::CHECKSUM_SCHEMA.to_string(),
            entries: vec![
                Sha256Entry {
                    path: "Agent.msi".into(),
                    sha256: "cd".repeat(32),
                    bytes: 10,
                },
                Sha256Entry {
                    path: "Agent.dmg".into(),
                    sha256: "ef".repeat(32),
                    bytes: 20,
                },
            ],
        }
    }

    fn inputs() -> ProvenanceInputs {
        ProvenanceInputs {
            rvf_identity: "ab".repeat(32),
            build_manifest_hash: "12".repeat(32),
            source_revision: "deadbeef".into(),
            builder: BuilderIdentity {
                id: "forge-worker".into(),
                kind: BuilderKind::Hosted,
                worker_id: Some("job-42".into()),
            },
            signing: SigningIdentity {
                posture: SigningPosture::KmsOrHsm,
                identity: Some("CN=Example Inc".into()),
            },
            started_at: "2026-08-03T10:00:00Z".into(),
            finished_at: "2026-08-03T10:04:12Z".into(),
            artifacts: artifacts(),
            compatibility_matrix_revision: Some("matrix-2026-08-01".into()),
        }
    }

    #[test]
    fn record_carries_every_field_invariant_5_names() {
        let r = provenance(&inputs()).unwrap();
        assert_eq!(r.rvf_identity, "ab".repeat(32));
        assert_eq!(r.source_revision, "deadbeef");
        assert_eq!(r.builder.kind, BuilderKind::Hosted);
        assert_eq!(r.signing.posture, SigningPosture::KmsOrHsm);
        assert_eq!(r.forge_core_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(r.artifacts.len(), 2);
        assert_eq!(
            r.compatibility_matrix_revision.as_deref(),
            Some("matrix-2026-08-01")
        );
    }

    #[test]
    fn artifacts_are_sorted_so_the_record_is_stable() {
        let r = provenance(&inputs()).unwrap();
        assert_eq!(r.artifacts[0].path, "Agent.dmg");
        assert_eq!(r.artifacts[1].path, "Agent.msi");

        let mut reversed = inputs();
        reversed.artifacts.entries.reverse();
        assert_eq!(
            provenance(&reversed).unwrap().record_hash().unwrap(),
            r.record_hash().unwrap()
        );
    }

    #[test]
    fn the_same_inputs_produce_byte_identical_json() {
        let a = provenance(&inputs()).unwrap().to_canonical_json().unwrap();
        let b = provenance(&inputs()).unwrap().to_canonical_json().unwrap();
        assert_eq!(a, b);
        assert!(!a.contains(' ') || a.contains("Example Inc"));
    }

    #[test]
    fn a_different_artifact_hash_changes_the_record_hash() {
        let mut changed = inputs();
        changed.artifacts.entries[0].sha256 = "00".repeat(32);
        assert_ne!(
            provenance(&changed).unwrap().record_hash().unwrap(),
            provenance(&inputs()).unwrap().record_hash().unwrap()
        );
    }

    #[test]
    fn empty_artifacts_are_refused() {
        let mut empty = inputs();
        empty.artifacts.entries.clear();
        let err = provenance(&empty).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Manifest);
        assert!(
            err.to_string().contains("at least one output artifact"),
            "{err}"
        );
    }

    #[test]
    fn bad_hashes_and_empty_fields_are_refused() {
        let mut bad_identity = inputs();
        bad_identity.rvf_identity = "short".into();
        assert_eq!(
            provenance(&bad_identity).unwrap_err().code(),
            crate::ErrorCode::Manifest
        );

        let mut no_revision = inputs();
        no_revision.source_revision = "".into();
        assert!(provenance(&no_revision)
            .unwrap_err()
            .to_string()
            .contains("sourceRevision"));

        let mut no_builder = inputs();
        no_builder.builder.id = " ".into();
        assert!(provenance(&no_builder)
            .unwrap_err()
            .to_string()
            .contains("builder.id"));
    }

    #[test]
    fn unsigned_development_builds_are_labeled() {
        let mut dev = inputs();
        dev.signing = SigningIdentity {
            posture: SigningPosture::UnsignedDevelopment,
            identity: None,
        };
        let r = provenance(&dev).unwrap();
        assert!(!r.signing.posture.is_signed());
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("unsigned-development"), "{json}");
    }

    #[test]
    fn record_serializes_with_camel_case_keys() {
        let json = serde_json::to_string(&provenance(&inputs()).unwrap()).unwrap();
        for key in [
            "rvfIdentity",
            "forgeCoreVersion",
            "buildManifestHash",
            "sourceRevision",
            "startedAt",
            "finishedAt",
        ] {
            assert!(json.contains(key), "missing {key}: {json}");
        }
    }
}
