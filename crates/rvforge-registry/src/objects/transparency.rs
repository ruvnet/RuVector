//! `TransparencyLogEntry` and the signed tree head.
//!
//! Log entries are not content-addressed the way the signed objects are: their
//! id is the Merkle leaf hash, which covers position as well as content, so
//! two entries for the same object at different indices are different entries.
//! They therefore do not implement `RegistryObject`.

use super::{Signature, SCHEMA_VERSION};
use serde::{Deserialize, Serialize};

/// The `type` value for log entries.
pub const TYPE_LOG_ENTRY: &str = "log-entry";

/// The `type` value for tree heads.
pub const TYPE_TREE_HEAD: &str = "tree-head";

/// One append-only entry in the transparency log.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TransparencyLogEntry {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Always [`TYPE_LOG_ENTRY`].
    #[serde(rename = "type")]
    pub object_type: String,
    /// Zero-based position in the log.
    pub index: u64,
    /// The Merkle leaf hash for this entry, as `sha256:<hex>`.
    pub entry_hash: String,
    /// The tree head after this entry was appended.
    pub tree_head: String,
    /// The content address of the logged object.
    pub object: String,
    /// What kind of object was logged.
    #[serde(rename = "objectType")]
    pub object_type_logged: LogObjectType,
    /// RFC 3339 instant, supplied by the caller.
    pub timestamp: String,
}

/// The object kinds the log covers.
///
/// `registry-model.md` names release, revocation, and publisher. `trust-update`
/// is added because a trust raise changes what the store tells users about a
/// release, and an unlogged trust raise would be the one privileged registry
/// action with no public record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LogObjectType {
    /// A published release.
    Release,
    /// A revocation.
    Revocation,
    /// A publisher record.
    Publisher,
    /// A registry-signed trust-level raise.
    TrustUpdate,
}

/// The current head of the log.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TreeHead {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Always [`TYPE_TREE_HEAD`].
    #[serde(rename = "type")]
    pub object_type: String,
    /// The number of leaves the root covers.
    pub tree_size: u64,
    /// The Merkle tree head, as `sha256:<hex>`.
    pub root_hash: String,
    /// RFC 3339 instant of the append that produced this head.
    pub timestamp: String,
    /// Registry signatures over [`Self::root_hash`], when a signer is
    /// configured.
    #[serde(default)]
    pub signatures: Vec<Signature>,
}

impl TreeHead {
    /// An empty head, for a log with no entries yet.
    pub fn empty(root_hash: String) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            object_type: TYPE_TREE_HEAD.into(),
            tree_size: 0,
            root_hash,
            timestamp: String::new(),
            signatures: Vec::new(),
        }
    }
}

/// The part of an entry that the Merkle leaf hash covers.
///
/// Deliberately excludes `entryHash` and `treeHead`: both are derived from the
/// leaf, so including them would make the hash self-referential.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LogLeaf<'a> {
    index: u64,
    object: &'a str,
    #[serde(rename = "objectType")]
    object_type_logged: LogObjectType,
    timestamp: &'a str,
}

/// The canonical bytes a log entry's Merkle leaf hash is taken over.
///
/// # Errors
///
/// [`crate::RegistryError::InvalidObject`] if the leaf cannot be canonicalized.
pub fn log_leaf_bytes(
    index: u64,
    object: &str,
    object_type: LogObjectType,
    timestamp: &str,
) -> crate::error::Result<Vec<u8>> {
    let leaf = LogLeaf {
        index,
        object,
        object_type_logged: object_type,
        timestamp,
    };
    Ok(rvf_forge_core::canonical::to_canonical_json(&leaf)?.into_bytes())
}
