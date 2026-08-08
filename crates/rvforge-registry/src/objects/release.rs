//! `Release` — the immutable, predecessor-linked unit of publication.

use super::{require_digest, require_non_empty, RegistryObject, Signature, SCHEMA_VERSION};
use crate::error::{RegistryError, Result};
use serde::{Deserialize, Serialize};

/// The `type` value for releases.
pub const TYPE_RELEASE: &str = "release";

/// A signed, immutable release of one package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Release {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Always [`TYPE_RELEASE`].
    #[serde(rename = "type")]
    pub object_type: String,
    /// This release's content address.
    pub release_id: String,
    /// Which package this is a release of.
    pub package: PackageRef,
    /// Publisher-chosen version string.
    pub version: String,
    /// The previous release of the same package, or `null` for the first.
    pub predecessor: Option<String>,
    /// Content address of the `.rvf` artifact.
    pub rvf_identity: String,
    /// Size of the artifact in bytes.
    pub rvf_size: u64,
    /// Content address of the release's `CapabilityManifest`.
    pub capability_manifest: String,
    /// Runtime profiles the artifact supports, e.g. `wasm`, `rvm`.
    pub runtime_profiles: Vec<String>,
    /// Runtime and schema compatibility floor.
    pub compatibility: Compatibility,
    /// Where inference comes from and which model digests are involved.
    pub model_manifest: ModelManifest,
    /// Content address of the software inventory.
    pub software_inventory: String,
    /// Content address of the evaluation report, if one exists.
    pub evaluation_report: Option<String>,
    /// Content address of the security report, if one exists.
    pub security_report: Option<String>,
    /// Content address of the provenance record.
    pub provenance: String,
    /// Evidence gathered so far. A publisher may only ever assert
    /// [`TrustLevel::Published`]; see `Registry::raise_trust_level`.
    pub trust_level: TrustLevel,
    /// The state-schema version up to which rolling back remains safe
    /// (ADR-288); declared before installation, never discovered after.
    pub rollback_safe_until_state_schema: u32,
    /// Store listing copy.
    pub listing: Listing,
    /// RFC 3339 publication instant.
    pub published_at: String,
    /// Publisher signature, plus the registry countersignature once added.
    #[serde(default)]
    pub signatures: Vec<Signature>,
}

/// The package a release belongs to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PackageRef {
    /// Package name, unique within a publisher.
    pub name: String,
    /// Content address of the owning `PublisherRecord`.
    pub publisher_id: String,
}

/// Runtime and schema compatibility floor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Compatibility {
    /// Minimum RVM version that can load this release.
    pub rvm_version_min: String,
    /// The state schema this release reads and writes.
    pub state_schema_version: u32,
    /// The witness schema this release emits.
    pub witness_schema_version: u32,
}

/// Where inference happens for this release.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelManifest {
    /// The inference location.
    pub location: ModelLocation,
    /// Content addresses of the models involved.
    pub digests: Vec<String>,
}

/// The five inference sources of ADR-294 §10.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelLocation {
    /// The model ships inside the `.rvf`.
    Embedded,
    /// The publisher runs inference on the user's behalf.
    PublisherInference,
    /// Cognitum Meta LLM.
    MetaLlm,
    /// The user supplies their own model or credentials.
    Byo,
}

/// Store listing copy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Listing {
    /// Store category.
    pub category: String,
    /// One-paragraph description.
    pub description: String,
    /// Commercial model; an open vocabulary per ADR-294 §10.
    pub price_model: String,
}

/// Evidence achieved, never implied safety (ADR-294 §7).
///
/// `Ord` follows the ladder, so "may only be raised" is a `>` comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrustLevel {
    /// Identity verified and package structurally valid.
    Published,
    /// Automated runtime, security, and evaluation tests passed.
    Tested,
    /// Human security and capability review completed.
    Reviewed,
    /// Approved by the customer's security or governance team.
    EnterpriseApproved,
}

impl TrustLevel {
    /// The level every release starts at when the publisher submits it.
    pub const INITIAL: Self = Self::Published;
}

impl Default for Release {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            object_type: TYPE_RELEASE.into(),
            release_id: String::new(),
            package: PackageRef {
                name: String::new(),
                publisher_id: String::new(),
            },
            version: String::new(),
            predecessor: None,
            rvf_identity: String::new(),
            rvf_size: 0,
            capability_manifest: String::new(),
            runtime_profiles: Vec::new(),
            compatibility: Compatibility {
                rvm_version_min: String::new(),
                state_schema_version: 1,
                witness_schema_version: 1,
            },
            model_manifest: ModelManifest {
                location: ModelLocation::Embedded,
                digests: Vec::new(),
            },
            software_inventory: String::new(),
            evaluation_report: None,
            security_report: None,
            provenance: String::new(),
            trust_level: TrustLevel::INITIAL,
            rollback_safe_until_state_schema: 1,
            listing: Listing {
                category: String::new(),
                description: String::new(),
                price_model: String::new(),
            },
            published_at: String::new(),
            signatures: Vec::new(),
        }
    }
}

impl RegistryObject for Release {
    const TYPE: &'static str = TYPE_RELEASE;
    const ID_FIELD: &'static str = "releaseId";

    fn schema_version(&self) -> u32 {
        self.schema_version
    }
    fn object_type(&self) -> &str {
        &self.object_type
    }
    fn id(&self) -> &str {
        &self.release_id
    }
    fn set_id(&mut self, id: String) {
        self.release_id = id;
    }
    fn signatures(&self) -> &[Signature] {
        &self.signatures
    }
    fn set_signatures(&mut self, signatures: Vec<Signature>) {
        self.signatures = signatures;
    }

    fn validate_fields(&self) -> Result<()> {
        require_non_empty("package.name", &self.package.name)?;
        require_digest("package.publisherId", &self.package.publisher_id)?;
        require_non_empty("version", &self.version)?;
        // rvfIdentity is what a revocation ultimately names, so a release
        // without one cannot be revoked precisely and is refused outright.
        require_digest("rvfIdentity", &self.rvf_identity)?;
        require_digest("capabilityManifest", &self.capability_manifest)?;
        require_digest("softwareInventory", &self.software_inventory)?;
        require_digest("provenance", &self.provenance)?;
        if let Some(p) = &self.predecessor {
            require_digest("predecessor", p)?;
        }
        if let Some(r) = &self.evaluation_report {
            require_digest("evaluationReport", r)?;
        }
        if let Some(r) = &self.security_report {
            require_digest("securityReport", r)?;
        }
        for (i, d) in self.model_manifest.digests.iter().enumerate() {
            require_digest(&format!("modelManifest.digests[{i}]"), d)?;
        }
        if self.runtime_profiles.is_empty() {
            return Err(RegistryError::invalid(
                "runtimeProfiles is empty; the release could not run anywhere",
            ));
        }
        require_non_empty(
            "compatibility.rvmVersionMin",
            &self.compatibility.rvm_version_min,
        )?;
        require_non_empty("listing.category", &self.listing.category)?;
        require_non_empty("listing.description", &self.listing.description)?;
        require_non_empty("listing.priceModel", &self.listing.price_model)?;
        require_non_empty("publishedAt", &self.published_at)?;
        Ok(())
    }
}

impl Release {
    /// The publisher-role signature, if the release carries one.
    pub fn publisher_signature(&self) -> Option<&Signature> {
        self.signatures
            .iter()
            .find(|s| s.role == super::SignatureRole::Publisher)
    }

    /// Whether two releases name the same package.
    pub fn same_package_as(&self, other: &Self) -> bool {
        self.package == other.package
    }
}
