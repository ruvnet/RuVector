//! The append-only transparency log: `log/entries.jsonl` + `log/tree-head.json`.
//!
//! A release absent from this log is treated as unpublished, so the log — not
//! the object store — is the answer to "was this ever published?". Appends are
//! idempotent per object: logging the same release twice returns the existing
//! entry rather than growing the tree, which keeps re-publication from silently
//! changing every subsequent inclusion proof.

use crate::error::{RegistryError, Result};
use crate::merkle;
use crate::objects::{
    digest_hex, LogObjectType, TransparencyLogEntry, TreeHead, DIGEST_PREFIX, SCHEMA_VERSION,
};
use rvf_forge_core::canonical::{hex_lower, to_canonical_json};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The log directory under the registry root.
pub const LOG_DIR: &str = "log";
/// The append-only entry file.
pub const ENTRIES_FILE: &str = "entries.jsonl";
/// The current tree head.
pub const TREE_HEAD_FILE: &str = "tree-head.json";

/// A file-backed RFC 6962 Merkle transparency log.
#[derive(Clone, Debug)]
pub struct TransparencyLog {
    dir: PathBuf,
}

/// An inclusion proof for one log entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InclusionProof {
    /// The entry's position in the log.
    pub leaf_index: u64,
    /// The number of leaves the proof is against.
    pub tree_size: u64,
    /// The Merkle leaf hash, as `sha256:<hex>`.
    pub leaf_hash: String,
    /// The audit path, root-ward, each as `sha256:<hex>`.
    pub path: Vec<String>,
    /// The tree head this proof reconstructs.
    pub root_hash: String,
}

impl InclusionProof {
    /// Recompute the root from the leaf and path and compare it to
    /// `expected_root`.
    ///
    /// Returns `false` — never an error — for a malformed proof, because a
    /// proof that cannot be parsed is a proof that does not verify.
    pub fn verify(&self, expected_root: &str) -> bool {
        // `root_hash` is what the prover claims; `expected_root` is what the
        // caller trusts. They must agree before the path is even walked.
        if self.root_hash != expected_root {
            return false;
        }
        let Ok(leaf) = parse_hash(&self.leaf_hash) else {
            return false;
        };
        let Ok(root) = parse_hash(expected_root) else {
            return false;
        };
        let mut path = Vec::with_capacity(self.path.len());
        for node in &self.path {
            let Ok(h) = parse_hash(node) else {
                return false;
            };
            path.push(h);
        }
        merkle::verify_inclusion(
            leaf,
            self.leaf_index as usize,
            self.tree_size as usize,
            &path,
            root,
        )
    }
}

fn parse_hash(value: &str) -> Result<merkle::Hash> {
    let hex = digest_hex("hash", value)?;
    let bytes = rvf_forge_core::canonical::hex_decode(&hex)?;
    bytes
        .as_slice()
        .try_into()
        .map_err(|_| RegistryError::invalid("hash is not 32 bytes"))
}

fn format_hash(hash: &merkle::Hash) -> String {
    format!("{DIGEST_PREFIX}{}", hex_lower(hash))
}

impl TransparencyLog {
    /// Open (creating if needed) the log under `registry_root`.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Io`] if the directory cannot be created.
    pub fn open(registry_root: impl AsRef<Path>) -> Result<Self> {
        let dir = registry_root.as_ref().join(LOG_DIR);
        fs::create_dir_all(&dir).map_err(|e| RegistryError::io(&dir, &e))?;
        Ok(Self { dir })
    }

    fn entries_path(&self) -> PathBuf {
        self.dir.join(ENTRIES_FILE)
    }

    fn head_path(&self) -> PathBuf {
        self.dir.join(TREE_HEAD_FILE)
    }

    /// Every entry, in index order.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Io`] on read failure, [`RegistryError::Log`] if a line
    /// does not parse or the indices are not consecutive from zero.
    pub fn entries(&self) -> Result<Vec<TransparencyLogEntry>> {
        let path = self.entries_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(&path).map_err(|e| RegistryError::io(&path, &e))?;
        let mut entries = Vec::new();
        for (line_no, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let entry: TransparencyLogEntry =
                serde_json::from_str(line).map_err(|e| RegistryError::Log {
                    detail: format!("line {} does not parse: {e}", line_no + 1),
                })?;
            if entry.index != entries.len() as u64 {
                return Err(RegistryError::Log {
                    detail: format!(
                        "entry at line {} declares index {}, expected {}",
                        line_no + 1,
                        entry.index,
                        entries.len()
                    ),
                });
            }
            entries.push(entry);
        }
        Ok(entries)
    }

    /// The leaf hashes of every entry, in index order.
    ///
    /// # Errors
    ///
    /// As [`TransparencyLog::entries`], plus [`RegistryError::InvalidObject`]
    /// if a stored `entryHash` is malformed.
    fn leaf_hashes(&self, entries: &[TransparencyLogEntry]) -> Result<Vec<merkle::Hash>> {
        entries.iter().map(|e| parse_hash(&e.entry_hash)).collect()
    }

    /// The entry for `object`, if it has been logged.
    ///
    /// # Errors
    ///
    /// As [`TransparencyLog::entries`].
    pub fn entry_for(&self, object: &str) -> Result<Option<TransparencyLogEntry>> {
        Ok(self.entries()?.into_iter().find(|e| e.object == object))
    }

    /// Every logged object of a given kind, in index order.
    ///
    /// # Errors
    ///
    /// As [`TransparencyLog::entries`].
    pub fn objects_of_type(&self, kind: LogObjectType) -> Result<Vec<String>> {
        Ok(self
            .entries()?
            .into_iter()
            .filter(|e| e.object_type_logged == kind)
            .map(|e| e.object)
            .collect())
    }

    /// Append `object` to the log, or return its existing entry.
    ///
    /// # Errors
    ///
    /// [`RegistryError::InvalidObject`] if `object` is not a digest,
    /// [`RegistryError::Io`] on write failure, [`RegistryError::Log`] if the
    /// existing file is inconsistent.
    pub fn append(
        &self,
        object: &str,
        kind: LogObjectType,
        timestamp: &str,
    ) -> Result<TransparencyLogEntry> {
        digest_hex("logged object", object)?;
        let mut entries = self.entries()?;
        if let Some(existing) = entries.iter().find(|e| e.object == object) {
            return Ok(existing.clone());
        }

        let index = entries.len() as u64;
        let leaf = crate::objects::log_leaf_bytes(index, object, kind, timestamp)?;
        let leaf_hash = merkle::leaf_hash(&leaf);

        let mut leaves = self.leaf_hashes(&entries)?;
        leaves.push(leaf_hash);
        let root = merkle::root(&leaves);

        let entry = TransparencyLogEntry {
            schema_version: SCHEMA_VERSION,
            object_type: crate::objects::TYPE_LOG_ENTRY.into(),
            index,
            entry_hash: format_hash(&leaf_hash),
            tree_head: format_hash(&root),
            object: object.to_string(),
            object_type_logged: kind,
            timestamp: timestamp.to_string(),
        };

        let line = to_canonical_json(&entry)?;
        let path = self.entries_path();
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| RegistryError::io(&path, &e))?;
        writeln!(file, "{line}").map_err(|e| RegistryError::io(&path, &e))?;

        let head = TreeHead {
            schema_version: SCHEMA_VERSION,
            object_type: crate::objects::TYPE_TREE_HEAD.into(),
            tree_size: leaves.len() as u64,
            root_hash: format_hash(&root),
            timestamp: timestamp.to_string(),
            signatures: Vec::new(),
        };
        let head_path = self.head_path();
        fs::write(&head_path, to_canonical_json(&head)?.as_bytes())
            .map_err(|e| RegistryError::io(&head_path, &e))?;

        entries.push(entry.clone());
        Ok(entry)
    }

    /// The current tree head, recomputed from the entries rather than trusted
    /// from `tree-head.json`.
    ///
    /// # Errors
    ///
    /// As [`TransparencyLog::entries`].
    pub fn tree_head(&self) -> Result<TreeHead> {
        let entries = self.entries()?;
        let leaves = self.leaf_hashes(&entries)?;
        let root = format_hash(&merkle::root(&leaves));
        if entries.is_empty() {
            return Ok(TreeHead::empty(root));
        }
        Ok(TreeHead {
            schema_version: SCHEMA_VERSION,
            object_type: crate::objects::TYPE_TREE_HEAD.into(),
            tree_size: leaves.len() as u64,
            root_hash: root,
            timestamp: entries[entries.len() - 1].timestamp.clone(),
            signatures: Vec::new(),
        })
    }

    /// An inclusion proof for `object` against the current tree head.
    ///
    /// # Errors
    ///
    /// [`RegistryError::NotFound`] if the object was never logged, otherwise as
    /// [`TransparencyLog::entries`].
    pub fn inclusion_proof(&self, object: &str) -> Result<InclusionProof> {
        let entries = self.entries()?;
        let entry =
            entries
                .iter()
                .find(|e| e.object == object)
                .ok_or_else(|| RegistryError::NotFound {
                    kind: "log entry".into(),
                    id: object.into(),
                })?;
        let leaves = self.leaf_hashes(&entries)?;
        let index = entry.index as usize;
        let path = merkle::inclusion_path(index, &leaves);
        Ok(InclusionProof {
            leaf_index: entry.index,
            tree_size: leaves.len() as u64,
            leaf_hash: entry.entry_hash.clone(),
            path: path.iter().map(format_hash).collect(),
            root_hash: format_hash(&merkle::root(&leaves)),
        })
    }

    /// Recompute every entry's leaf hash from its own fields and confirm the
    /// chain of tree heads. Detects an entry edited in place.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Log`] naming the first inconsistent entry.
    pub fn verify_consistency(&self) -> Result<()> {
        let entries = self.entries()?;
        let mut leaves: Vec<merkle::Hash> = Vec::with_capacity(entries.len());
        for entry in &entries {
            let bytes = crate::objects::log_leaf_bytes(
                entry.index,
                &entry.object,
                entry.object_type_logged,
                &entry.timestamp,
            )?;
            let recomputed = merkle::leaf_hash(&bytes);
            if format_hash(&recomputed) != entry.entry_hash {
                return Err(RegistryError::Log {
                    detail: format!(
                        "entry {} hashes to {} but claims {}",
                        entry.index,
                        format_hash(&recomputed),
                        entry.entry_hash
                    ),
                });
            }
            leaves.push(recomputed);
            let root = format_hash(&merkle::root(&leaves));
            if root != entry.tree_head {
                return Err(RegistryError::Log {
                    detail: format!(
                        "tree head after entry {} is {root} but the entry claims {}",
                        entry.index, entry.tree_head
                    ),
                });
            }
        }
        Ok(())
    }
}
