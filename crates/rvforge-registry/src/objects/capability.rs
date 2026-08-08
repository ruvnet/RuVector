//! `CapabilityManifest` — the deny-by-default capability contract of a release.

use super::{require_non_empty, RegistryObject, Signature, SCHEMA_VERSION};
use crate::error::{RegistryError, Result};
use serde::{Deserialize, Serialize};

/// The `type` value for capability manifests.
pub const TYPE_CAPABILITY_MANIFEST: &str = "capability-manifest";

/// Scope strings so broad that ADR-294 §8 requires manual review before the
/// release can be published. A manifest requesting one of these must say so in
/// `manualReviewTriggers` rather than slipping through as ordinary prose.
pub const BROAD_SCOPES: &[&str] = &["all-files", "all", "unrestricted", "any", "*"];

/// What a release asks for and what it declares it cannot do.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CapabilityManifest {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Always [`TYPE_CAPABILITY_MANIFEST`].
    #[serde(rename = "type")]
    pub object_type: String,
    /// This manifest's content address.
    pub manifest_id: String,
    /// Always [`DefaultPolicy::Deny`] (ADR-294 §9, ADR-286).
    pub default_policy: DefaultPolicy,
    /// The capabilities the agent asks for.
    pub requests: Vec<CapabilityRequest>,
    /// Capability classes the agent declares it cannot use.
    pub denials: Vec<String>,
    /// Conditions that routed, or would route, this manifest to human review.
    pub manual_review_triggers: Vec<String>,
    /// Publisher signatures over [`Self::manifest_id`].
    #[serde(default)]
    pub signatures: Vec<Signature>,
}

/// The manifest's baseline stance. Only `deny` is representable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DefaultPolicy {
    /// Everything not explicitly requested is refused.
    Deny,
}

/// One requested capability.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CapabilityRequest {
    /// One of the ADR-286 capability classes, e.g. `filesystem`, `memory`.
    pub class: String,
    /// A concrete scope. `user-selected`, `512MiB`, `encrypted-local` — never
    /// a phrase like "your computer".
    pub scope: String,
    /// Why the agent needs it, in the words shown at install time.
    pub rationale: String,
}

impl CapabilityManifest {
    /// Requests whose scope is broad enough to require manual review.
    pub fn broad_requests(&self) -> Vec<&CapabilityRequest> {
        self.requests
            .iter()
            .filter(|r| BROAD_SCOPES.contains(&r.scope.to_ascii_lowercase().as_str()))
            .collect()
    }
}

impl Default for CapabilityManifest {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            object_type: TYPE_CAPABILITY_MANIFEST.into(),
            manifest_id: String::new(),
            default_policy: DefaultPolicy::Deny,
            requests: Vec::new(),
            denials: Vec::new(),
            manual_review_triggers: Vec::new(),
            signatures: Vec::new(),
        }
    }
}

impl RegistryObject for CapabilityManifest {
    const TYPE: &'static str = TYPE_CAPABILITY_MANIFEST;
    const ID_FIELD: &'static str = "manifestId";

    fn schema_version(&self) -> u32 {
        self.schema_version
    }
    fn object_type(&self) -> &str {
        &self.object_type
    }
    fn id(&self) -> &str {
        &self.manifest_id
    }
    fn set_id(&mut self, id: String) {
        self.manifest_id = id;
    }
    fn signatures(&self) -> &[Signature] {
        &self.signatures
    }
    fn set_signatures(&mut self, signatures: Vec<Signature>) {
        self.signatures = signatures;
    }

    fn validate_fields(&self) -> Result<()> {
        for (i, req) in self.requests.iter().enumerate() {
            require_non_empty(&format!("requests[{i}].class"), &req.class)?;
            require_non_empty(&format!("requests[{i}].scope"), &req.scope)?;
            require_non_empty(&format!("requests[{i}].rationale"), &req.rationale)?;
        }
        // A broad scope is permitted, but only if the manifest admits it needs
        // review. Silently broad scopes are the failure ADR-294 §8 exists to
        // prevent, so they are rejected here rather than at install time.
        for req in self.broad_requests() {
            if self.manual_review_triggers.is_empty() {
                return Err(RegistryError::invalid(format!(
                    "request {:?} has broad scope {:?} but manualReviewTriggers is empty",
                    req.class, req.scope
                )));
            }
        }
        Ok(())
    }
}
