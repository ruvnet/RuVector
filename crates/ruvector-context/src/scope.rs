//! Canonical structured context scopes and prefix matching.

use crate::{ContextIndexError, Result};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

const MAX_AUTHORITY_BYTES: usize = 253;
const MAX_SLUG_BYTES: usize = 63;
const MAX_PATH_SEGMENTS: usize = 32;
const MAX_PATH_SEGMENT_BYTES: usize = 128;
const MAX_JOINED_PATH_BYTES: usize = 1_024;

/// Exact non-path namespace fields shared by a group of context shards.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct ContextNamespace {
    authority: String,
    tenant: String,
    subject_kind: String,
    subject_id: String,
    collection: String,
}

#[derive(Deserialize)]
struct ContextNamespaceWire {
    authority: String,
    tenant: String,
    subject_kind: String,
    subject_id: String,
    collection: String,
}

impl<'de> Deserialize<'de> for ContextNamespace {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ContextNamespaceWire::deserialize(deserializer)?;
        Self::new(
            wire.authority,
            wire.tenant,
            wire.subject_kind,
            wire.subject_id,
            wire.collection,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl ContextNamespace {
    /// Construct and validate canonical namespace fields.
    pub fn new(
        authority: impl Into<String>,
        tenant: impl Into<String>,
        subject_kind: impl Into<String>,
        subject_id: impl Into<String>,
        collection: impl Into<String>,
    ) -> Result<Self> {
        let namespace = Self {
            authority: authority.into(),
            tenant: tenant.into(),
            subject_kind: subject_kind.into(),
            subject_id: subject_id.into(),
            collection: collection.into(),
        };
        validate_authority(&namespace.authority)?;
        validate_slug(&namespace.tenant)?;
        validate_slug(&namespace.subject_id)?;
        if !matches!(
            namespace.subject_kind.as_str(),
            "agent" | "user" | "service" | "team"
        ) {
            return Err(ContextIndexError::InvalidScope("unknown subject kind"));
        }
        if !matches!(
            namespace.collection.as_str(),
            "memory" | "resources" | "skills"
        ) {
            return Err(ContextIndexError::InvalidScope("unknown collection"));
        }
        Ok(namespace)
    }

    /// Canonical DNS authority.
    #[must_use]
    pub fn authority(&self) -> &str {
        &self.authority
    }

    /// Canonical tenant slug.
    #[must_use]
    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    /// Canonical subject kind.
    #[must_use]
    pub fn subject_kind(&self) -> &str {
        &self.subject_kind
    }

    /// Canonical subject identifier.
    #[must_use]
    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    /// Canonical collection name.
    #[must_use]
    pub fn collection(&self) -> &str {
        &self.collection
    }
}

/// One exact physical index scope under a structured context namespace.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct ContextScope {
    namespace: ContextNamespace,
    path: Vec<String>,
}

#[derive(Deserialize)]
struct ContextScopeWire {
    namespace: ContextNamespace,
    path: Vec<String>,
}

impl<'de> Deserialize<'de> for ContextScope {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ContextScopeWire::deserialize(deserializer)?;
        Self::new(wire.namespace, wire.path).map_err(serde::de::Error::custom)
    }
}

impl ContextScope {
    /// Construct an exact scope from validated namespace fields and path segments.
    pub fn new(namespace: ContextNamespace, path: Vec<String>) -> Result<Self> {
        if path.len() > MAX_PATH_SEGMENTS {
            return Err(ContextIndexError::InvalidScope("too many path segments"));
        }
        let joined = path
            .iter()
            .map(String::len)
            .sum::<usize>()
            .saturating_add(path.len().saturating_sub(1));
        if joined > MAX_JOINED_PATH_BYTES {
            return Err(ContextIndexError::InvalidScope("path is too long"));
        }
        for segment in &path {
            validate_path_segment(segment)?;
        }
        Ok(Self { namespace, path })
    }

    /// Construct the collection-root scope.
    #[must_use]
    pub const fn root(namespace: ContextNamespace) -> Self {
        Self {
            namespace,
            path: Vec::new(),
        }
    }

    /// Structured non-path namespace fields.
    #[must_use]
    pub const fn namespace(&self) -> &ContextNamespace {
        &self.namespace
    }

    /// Canonical path segments for this exact shard.
    #[must_use]
    pub fn path(&self) -> &[String] {
        &self.path
    }

    /// Whether this exact shard is at or below `root`.
    #[must_use]
    pub fn is_within(&self, root: &Self) -> bool {
        self.namespace == root.namespace
            && self.path.len() >= root.path.len()
            && self.path[..root.path.len()] == root.path
    }

    /// SHA-256 identifier used as the shard filename.
    #[must_use]
    pub fn shard_id(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"ruvector-context-scope-v1\0");
        for component in [
            self.namespace.authority(),
            self.namespace.tenant(),
            self.namespace.subject_kind(),
            self.namespace.subject_id(),
            self.namespace.collection(),
        ] {
            hash_component(&mut hash, component.as_bytes());
        }
        hash.update(
            u16::try_from(self.path.len())
                .unwrap_or(u16::MAX)
                .to_be_bytes(),
        );
        for segment in &self.path {
            hash_component(&mut hash, segment.as_bytes());
        }
        hash.finalize().into()
    }

    pub(crate) fn shard_filename(&self) -> String {
        let mut name = String::with_capacity(69);
        for byte in self.shard_id() {
            use core::fmt::Write as _;
            let _ = write!(name, "{byte:02x}");
        }
        name.push_str(".redb");
        name
    }
}

fn hash_component(hash: &mut Sha256, bytes: &[u8]) {
    hash.update(u16::try_from(bytes.len()).unwrap_or(u16::MAX).to_be_bytes());
    hash.update(bytes);
}

fn validate_authority(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > MAX_AUTHORITY_BYTES || !value.is_ascii() {
        return Err(ContextIndexError::InvalidScope("invalid authority length"));
    }
    for label in value.split('.') {
        if label.is_empty()
            || label.len() > 63
            || label.starts_with('-')
            || label.ends_with('-')
            || !label
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        {
            return Err(ContextIndexError::InvalidScope("noncanonical authority"));
        }
    }
    Ok(())
}

fn validate_slug(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > MAX_SLUG_BYTES
        || value.starts_with('-')
        || value.ends_with('-')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(ContextIndexError::InvalidScope("noncanonical slug"));
    }
    Ok(())
}

fn validate_path_segment(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > MAX_PATH_SEGMENT_BYTES
        || matches!(value, "." | "..")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~'))
    {
        return Err(ContextIndexError::InvalidScope("noncanonical path segment"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn namespace(tenant: &str) -> ContextNamespace {
        ContextNamespace::new("context.example", tenant, "agent", "reader", "memory").unwrap()
    }

    #[test]
    fn scope_prefix_is_segment_aware() {
        let root = ContextScope::new(namespace("acme"), vec!["a".into()]).unwrap();
        let child = ContextScope::new(namespace("acme"), vec!["a".into(), "b".into()]).unwrap();
        let sibling = ContextScope::new(namespace("acme"), vec!["alpha".into()]).unwrap();
        assert!(child.is_within(&root));
        assert!(!sibling.is_within(&root));
    }

    #[test]
    fn shard_name_is_opaque_and_deterministic() {
        let scope = ContextScope::new(namespace("secret-tenant"), vec!["Project".into()]).unwrap();
        let name = scope.shard_filename();
        assert_eq!(name.len(), 69);
        assert!(name.ends_with(".redb"));
        assert!(!name.contains("secret"));
        assert_eq!(name, scope.shard_filename());
    }

    #[test]
    fn noncanonical_fields_are_rejected() {
        assert!(
            ContextNamespace::new("Context.example", "acme", "agent", "reader", "memory").is_err()
        );
        assert!(
            ContextNamespace::new("context.example", "../acme", "agent", "reader", "memory")
                .is_err()
        );
        assert!(ContextScope::new(namespace("acme"), vec!["..".into()]).is_err());
        assert!(serde_json::from_str::<ContextScope>(
            r#"{"namespace":{"authority":"context.example","tenant":"ACME","subject_kind":"agent","subject_id":"reader","collection":"memory"},"path":[]}"#
        )
        .is_err());
    }
}
