//! The canonical build manifest and the runtime contract it carries.
//!
//! ADR-291 §1 fixes seven fields that every generated package must embed. The
//! build manifest is the superset: it adds the application metadata, targets,
//! and packaging mode that describe *how* the package is produced, and it is
//! what the CLI signs and submits.
//!
//! Two properties make it useful rather than decorative:
//!
//! - **Determinism.** [`CanonicalBuildManifest::to_canonical_json`] is the
//!   hashed form, and the same [`BuildSpec`] always produces the same bytes.
//!   Without that, `buildManifestHash` in a provenance record identifies
//!   nothing.
//! - **Gating before compute.** [`build_manifest`] rejects targets absent from
//!   the compatibility matrix (ADR-291 §2) at manifest-generation time, before
//!   upload and before a worker is allocated.

use crate::canonical::{canonical_sha256_hex, to_canonical_json};
use crate::capability::CapabilityPolicy;
use crate::error::{ForgeError, Result};
use crate::target::Target;
use serde::{Deserialize, Serialize};

/// Schema identifier carried by every [`CanonicalBuildManifest`].
pub const BUILD_MANIFEST_SCHEMA: &str = "rvforge.build-manifest/1";

/// Schema identifier carried by every [`RuntimeContract`].
pub const RUNTIME_CONTRACT_SCHEMA: &str = "rvforge.runtime-contract/1";

/// The state delta and checkpoint schema this crate emits (ADR-291 §1).
pub const STATE_SCHEMA_VERSION: u32 = 1;

/// The witness record schema this crate emits (ADR-291 §1).
pub const WITNESS_SCHEMA_VERSION: u32 = 1;

/// How the RVF reaches the installed machine (ADR-283 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PackagingMode {
    /// The complete RVF plus the reader ship in every installer. Runs with no
    /// network access; duplicates large models across outputs.
    Embedded,
    /// The reader plus a signed RVF locator; the reader downloads and verifies
    /// before execution.
    Thin,
    /// The reader is installed once and `.rvf` is registered as a file type.
    SharedReader,
}

/// The runtime family a package was built and validated for (ADR-291 §1).
///
/// The reader may select a runtime at or below the profile a package was built
/// for and never above it (ADR-291 §4), which is what [`RuntimeProfile::rank`]
/// encodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeProfile {
    /// WebAssembly execution.
    Wasm,
    /// A Linux microVM.
    LinuxMicrovm,
    /// Native execution under OS isolation.
    Native,
    /// Native RVM, the strongest profile.
    Rvm,
}

impl RuntimeProfile {
    /// Position in the runtime preference order of ADR-291 §4, strongest last.
    pub const fn rank(self) -> u8 {
        match self {
            Self::Wasm => 0,
            Self::LinuxMicrovm => 1,
            Self::Native => 2,
            Self::Rvm => 3,
        }
    }

    /// Whether a package built for `self` may select `candidate` at run time.
    ///
    /// A package built for the WASM profile never promotes itself to native
    /// RVM (ADR-291 acceptance criterion 9).
    pub const fn permits(self, candidate: Self) -> bool {
        candidate.rank() <= self.rank()
    }
}

/// Publisher-facing metadata that brands the produced packages (FR007).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppMetadata {
    /// Application name shown to the installing user.
    pub name: String,
    /// Application version.
    pub version: String,
    /// Publisher identity string.
    pub publisher: String,
    /// SPDX license identifier or a license name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Short description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// What the caller asks Forge to build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildSpec {
    /// SHA-256 of the canonical RVF, lowercase hex — normally
    /// [`crate::ForgeInspection::rvf_identity`].
    pub rvf_identity: String,
    /// Application branding.
    pub app: AppMetadata,
    /// Targets to build. Order does not affect the manifest; duplicates are
    /// collapsed.
    pub targets: Vec<Target>,
    /// Packaging mode.
    pub packaging_mode: PackagingMode,
    /// Runtime family this build is validated for.
    pub runtime_profile: RuntimeProfile,
    /// The default-deny capability policy this build is made against.
    pub capability_policy: CapabilityPolicy,
    /// Semantic version of the embedded RVM.
    pub rvm_version: String,
    /// Source revision of the embedded RVM. Required: a semantic version alone
    /// does not identify a build (ADR-291, Alternatives Considered).
    pub rvm_commit: String,
    /// Revision of the compatibility matrix consulted, recorded so a past
    /// admission decision can be reconstructed (ADR-291 §2).
    pub compatibility_matrix_revision: Option<String>,
}

/// The seven-field contract embedded in every generated package (ADR-291 §1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeContract {
    /// Schema identifier; always [`RUNTIME_CONTRACT_SCHEMA`].
    pub schema: String,
    /// SHA-256 of the canonical RVF. Identical across every package from one
    /// build, and unchanged from the submitted artifact.
    pub rvf_identity: String,
    /// Semantic version of the embedded runtime.
    pub rvm_version: String,
    /// Source revision of the embedded runtime.
    pub rvm_commit: String,
    /// Runtime family this package was built for.
    pub runtime_profile: RuntimeProfile,
    /// SHA-256 of the capability policy this package was built against.
    pub capability_policy_hash: String,
    /// State delta and checkpoint schema version.
    pub state_schema_version: u32,
    /// Witness record schema version.
    pub witness_schema_version: u32,
}

/// The full build manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalBuildManifest {
    /// Schema identifier; always [`BUILD_MANIFEST_SCHEMA`].
    pub schema: String,
    /// SHA-256 of the canonical RVF, lowercase hex.
    pub rvf_identity: String,
    /// Application branding.
    pub app: AppMetadata,
    /// Targets to build, deduplicated and sorted so that two specs requesting
    /// the same set produce the same manifest.
    pub targets: Vec<Target>,
    /// Packaging mode.
    pub packaging_mode: PackagingMode,
    /// Runtime family this build is validated for.
    pub runtime_profile: RuntimeProfile,
    /// SHA-256 of the capability policy.
    pub capability_policy_hash: String,
    /// Semantic version of the embedded runtime.
    pub rvm_version: String,
    /// Source revision of the embedded runtime.
    pub rvm_commit: String,
    /// State delta and checkpoint schema version; always
    /// [`STATE_SCHEMA_VERSION`].
    pub state_schema_version: u32,
    /// Witness record schema version; always [`WITNESS_SCHEMA_VERSION`].
    pub witness_schema_version: u32,
    /// Revision of the compatibility matrix that admitted this build.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compatibility_matrix_revision: Option<String>,
}

impl CanonicalBuildManifest {
    /// The canonical JSON encoding: sorted keys, no floats, no whitespace.
    ///
    /// # Errors
    ///
    /// [`ForgeError::Manifest`] if the manifest cannot be canonicalized.
    pub fn to_canonical_json(&self) -> Result<String> {
        to_canonical_json(self)
    }

    /// SHA-256 of the canonical JSON, lowercase hex. This is
    /// `buildManifestHash` in a provenance record.
    ///
    /// # Errors
    ///
    /// As [`CanonicalBuildManifest::to_canonical_json`].
    pub fn manifest_hash(&self) -> Result<String> {
        canonical_sha256_hex(self)
    }

    /// The seven-field runtime contract this manifest implies.
    pub fn runtime_contract(&self) -> RuntimeContract {
        RuntimeContract {
            schema: RUNTIME_CONTRACT_SCHEMA.to_string(),
            rvf_identity: self.rvf_identity.clone(),
            rvm_version: self.rvm_version.clone(),
            rvm_commit: self.rvm_commit.clone(),
            runtime_profile: self.runtime_profile,
            capability_policy_hash: self.capability_policy_hash.clone(),
            state_schema_version: self.state_schema_version,
            witness_schema_version: self.witness_schema_version,
        }
    }
}

/// Build a canonical manifest from `spec`.
///
/// Validation happens here, before upload and before a worker is allocated
/// (ADR-291 §2): the RVF identity must be a 64-character hex SHA-256, at least
/// one target must be requested, every target must be in the compatibility
/// matrix, and `rvmVersion` and `rvmCommit` must both be present.
///
/// # Errors
///
/// [`ForgeError::UnsupportedTarget`] for a target absent from the matrix;
/// [`ForgeError::Manifest`] for every other validation failure.
pub fn build_manifest(spec: &BuildSpec) -> Result<CanonicalBuildManifest> {
    validate_identity(&spec.rvf_identity)?;
    require_non_empty("app.name", &spec.app.name)?;
    require_non_empty("app.version", &spec.app.version)?;
    require_non_empty("app.publisher", &spec.app.publisher)?;
    require_non_empty("rvmVersion", &spec.rvm_version)?;
    require_non_empty("rvmCommit", &spec.rvm_commit)?;

    if spec.targets.is_empty() {
        return Err(ForgeError::Manifest {
            detail: "no targets requested; a build must name at least one target".to_string(),
        });
    }
    for target in &spec.targets {
        target.ensure_supported()?;
    }

    // Deduplicate and sort so that target order at the call site cannot change
    // the manifest bytes.
    let mut targets = spec.targets.clone();
    targets.sort_unstable();
    targets.dedup();

    Ok(CanonicalBuildManifest {
        schema: BUILD_MANIFEST_SCHEMA.to_string(),
        rvf_identity: spec.rvf_identity.to_ascii_lowercase(),
        app: spec.app.clone(),
        targets,
        packaging_mode: spec.packaging_mode,
        runtime_profile: spec.runtime_profile,
        capability_policy_hash: spec.capability_policy.policy_hash()?,
        rvm_version: spec.rvm_version.clone(),
        rvm_commit: spec.rvm_commit.clone(),
        state_schema_version: STATE_SCHEMA_VERSION,
        witness_schema_version: WITNESS_SCHEMA_VERSION,
        compatibility_matrix_revision: spec.compatibility_matrix_revision.clone(),
    })
}

fn validate_identity(identity: &str) -> Result<()> {
    if identity.len() != 64 || !identity.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ForgeError::Manifest {
            detail: format!(
                "rvfIdentity must be a 64-character hex SHA-256, got {} characters",
                identity.len()
            ),
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
    use crate::capability::{CapabilityClass, CapabilityGrant};

    fn spec() -> BuildSpec {
        BuildSpec {
            rvf_identity: "ab".repeat(32),
            app: AppMetadata {
                name: "Example Agent".into(),
                version: "1.2.3".into(),
                publisher: "Example Inc".into(),
                license: Some("MIT".into()),
                description: None,
            },
            targets: vec![Target::WindowsX64, Target::MacosUniversal],
            packaging_mode: PackagingMode::Embedded,
            runtime_profile: RuntimeProfile::Wasm,
            capability_policy: CapabilityPolicy::deny_all(),
            rvm_version: "0.4.0".into(),
            rvm_commit: "abc123".into(),
            compatibility_matrix_revision: Some("matrix-2026-08-01".into()),
        }
    }

    #[test]
    fn manifest_carries_the_adr_291_contract_fields() {
        let m = build_manifest(&spec()).unwrap();
        assert_eq!(m.schema, BUILD_MANIFEST_SCHEMA);
        assert_eq!(m.state_schema_version, 1);
        assert_eq!(m.witness_schema_version, 1);
        assert_eq!(m.rvm_version, "0.4.0");
        assert_eq!(m.rvm_commit, "abc123");
        assert_eq!(m.capability_policy_hash.len(), 64);

        let contract = m.runtime_contract();
        assert_eq!(contract.rvf_identity, m.rvf_identity);
        assert_eq!(contract.runtime_profile, RuntimeProfile::Wasm);
        let json = serde_json::to_string(&contract).unwrap();
        for field in [
            "rvfIdentity",
            "rvmVersion",
            "rvmCommit",
            "runtimeProfile",
            "capabilityPolicyHash",
            "stateSchemaVersion",
            "witnessSchemaVersion",
        ] {
            assert!(json.contains(field), "contract is missing {field}: {json}");
        }
    }

    #[test]
    fn the_same_spec_produces_byte_identical_json() {
        let a = build_manifest(&spec())
            .unwrap()
            .to_canonical_json()
            .unwrap();
        let b = build_manifest(&spec())
            .unwrap()
            .to_canonical_json()
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn target_order_does_not_change_the_manifest() {
        let mut reversed = spec();
        reversed.targets.reverse();
        assert_eq!(
            build_manifest(&spec()).unwrap().manifest_hash().unwrap(),
            build_manifest(&reversed).unwrap().manifest_hash().unwrap()
        );
    }

    #[test]
    fn duplicate_targets_are_collapsed() {
        let mut dupes = spec();
        dupes.targets = vec![
            Target::WindowsX64,
            Target::WindowsX64,
            Target::MacosUniversal,
        ];
        assert_eq!(
            build_manifest(&dupes).unwrap().targets,
            build_manifest(&spec()).unwrap().targets
        );
    }

    #[test]
    fn a_different_capability_policy_changes_the_manifest_hash() {
        let mut opened = spec();
        opened.capability_policy = CapabilityPolicy::deny_all()
            .grant(CapabilityClass::Network, CapabilityGrant::unscoped());
        assert_ne!(
            build_manifest(&spec()).unwrap().manifest_hash().unwrap(),
            build_manifest(&opened).unwrap().manifest_hash().unwrap()
        );
    }

    #[test]
    fn canonical_json_has_sorted_keys_and_no_separator_whitespace() {
        let json = build_manifest(&spec())
            .unwrap()
            .to_canonical_json()
            .unwrap();
        // Whitespace inside string values is content; whitespace around the
        // structural characters is what canonicalization removes.
        assert!(!json.contains("\": "), "{json}");
        assert!(!json.contains(", \""), "{json}");
        assert!(!json.contains('\n'), "{json}");

        let keys = [
            "app",
            "capabilityPolicyHash",
            "rvfIdentity",
            "schema",
            "targets",
        ];
        let positions: Vec<usize> = keys
            .iter()
            .map(|k| {
                json.find(&format!("\"{k}\""))
                    .unwrap_or_else(|| panic!("missing {k}"))
            })
            .collect();
        assert!(
            positions.windows(2).all(|w| w[0] < w[1]),
            "keys are not in sorted order: {json}"
        );
    }

    #[test]
    fn identity_is_lowercased() {
        let mut upper = spec();
        upper.rvf_identity = "AB".repeat(32);
        assert_eq!(
            build_manifest(&upper).unwrap().rvf_identity,
            "ab".repeat(32)
        );
    }

    #[test]
    fn a_bad_identity_is_rejected() {
        for bad in ["", "abc", &"z".repeat(64)] {
            let mut s = spec();
            s.rvf_identity = bad.to_string();
            let err = build_manifest(&s).unwrap_err();
            assert_eq!(err.code(), crate::ErrorCode::Manifest, "{bad}");
        }
    }

    #[test]
    fn empty_required_fields_are_rejected() {
        let mut s = spec();
        s.rvm_commit = "  ".into();
        let err = build_manifest(&s).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Manifest);
        assert!(err.to_string().contains("rvmCommit"), "{err}");
    }

    #[test]
    fn no_targets_is_rejected() {
        let mut s = spec();
        s.targets.clear();
        assert_eq!(
            build_manifest(&s).unwrap_err().code(),
            crate::ErrorCode::Manifest
        );
    }

    #[test]
    fn runtime_profile_never_promotes_above_the_built_profile() {
        assert!(RuntimeProfile::Wasm.permits(RuntimeProfile::Wasm));
        assert!(!RuntimeProfile::Wasm.permits(RuntimeProfile::Rvm));
        assert!(!RuntimeProfile::Wasm.permits(RuntimeProfile::Native));
        assert!(RuntimeProfile::Rvm.permits(RuntimeProfile::Wasm));
        assert!(RuntimeProfile::Native.permits(RuntimeProfile::LinuxMicrovm));
    }
}
