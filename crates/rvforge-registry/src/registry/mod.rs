//! The file-backed local registry (ADR-294 MVP).
//!
//! Everything is on disk under one root, laid out as `registry-model.md`
//! specifies:
//!
//! ```text
//! objects/sha256/<2-hex>/<digest>.json   all objects, content-addressed
//! packages/<publisher>/<name>/releases.jsonl
//! log/entries.jsonl
//! log/tree-head.json
//! witness/<subject-hex>.jsonl            per-subject receipt chain index
//! ```
//!
//! `<publisher>` is the 64-hex tail of the publisher id rather than the whole
//! `sha256:…` string, because the colon is not a portable path character. The
//! `witness/` directory is an addition: the model defines the receipt object
//! and its `prevReceipt` chain but no index, and without one, finding a
//! subject's latest receipt would mean rescanning every object in the store.
//!
//! # No clock, no network, no private keys
//!
//! Every operation takes the timestamp it should record as an argument, the
//! same discipline `rvf-forge-core` uses, so results are a pure function of
//! their inputs. There is no network code. Private keys stay with callers: the
//! registry holds a *verifying* key to check countersignatures, and borrows a
//! [`Signer`] only if the caller wants receipts signed.

mod publish;
mod witness;

pub use publish::{PublishOutcome, RevokeOutcome, TrustOutcome};
pub use witness::ReceiptRequest;

use crate::crypto::{verify_over_id, Signer};
use crate::error::{RegistryError, Result};
use crate::log::{InclusionProof, TransparencyLog};
use crate::objects::{
    digest_hex, LogObjectType, PublisherRecord, RegistryObject, Release, Revocation, SignatureRole,
    TransparencyLogEntry, TreeHead, TrustLevel, TrustUpdate,
};
use crate::store::ObjectStore;
use ed25519_dalek::VerifyingKey;
use rvf_forge_core::canonical::to_canonical_json;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The package index directory under the registry root.
pub const PACKAGES_DIR: &str = "packages";
/// The per-package append-only release index.
pub const RELEASES_FILE: &str = "releases.jsonl";
/// The witness-chain index directory under the registry root.
pub const WITNESS_DIR: &str = "witness";

/// The registry's own public key, used to check countersignatures.
#[derive(Clone, Debug)]
pub struct RegistryKey {
    /// The `keyId` a registry-role signature must carry.
    pub key_id: String,
    /// The public half.
    pub verifying_key: VerifyingKey,
}

/// One line of `packages/<publisher>/<name>/releases.jsonl`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PackageIndexEntry {
    /// The release's content address.
    pub release_id: String,
    /// Its version string.
    pub version: String,
    /// Its predecessor, or `null` for the first release.
    pub predecessor: Option<String>,
    /// The publication instant recorded at publish time.
    pub published_at: String,
}

/// Why a release may or may not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionStatus {
    /// Published, and nothing in its chain is revoked.
    Executable,
    /// Absent from the transparency log; treated as never published.
    Unpublished,
    /// Blocked by policy. The artifact and its state remain readable.
    Revoked {
        /// What was revoked: the release, its publisher, or its signing key.
        subject: String,
    },
}

/// A file-backed content-addressed release registry.
#[derive(Clone)]
pub struct Registry {
    root: PathBuf,
    store: ObjectStore,
    log: TransparencyLog,
    registry_key: Option<RegistryKey>,
    receipt_signer: Option<Arc<dyn Signer + Send + Sync>>,
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry")
            .field("root", &self.root)
            .field(
                "registry_key",
                &self.registry_key.as_ref().map(|k| &k.key_id),
            )
            .field("receipt_signer", &self.receipt_signer.is_some())
            .finish()
    }
}

impl Registry {
    /// Open (creating if needed) a registry rooted at `root`.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Io`] if the directory tree cannot be created.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let store = ObjectStore::open(&root)?;
        let log = TransparencyLog::open(&root)?;
        for sub in [PACKAGES_DIR, WITNESS_DIR] {
            let dir = root.join(sub);
            fs::create_dir_all(&dir).map_err(|e| RegistryError::io(&dir, &e))?;
        }
        Ok(Self {
            root,
            store,
            log,
            registry_key: None,
            receipt_signer: None,
        })
    }

    /// Configure the public key that registry-role countersignatures must
    /// verify against. Without it, revocations and trust raises are refused.
    #[must_use]
    pub fn with_registry_key(mut self, key_id: impl Into<String>, key: VerifyingKey) -> Self {
        self.registry_key = Some(RegistryKey {
            key_id: key_id.into(),
            verifying_key: key,
        });
        self
    }

    /// Lend the registry a signer so emitted witness receipts carry a registry
    /// signature. Without it, receipts are emitted unsigned.
    #[must_use]
    pub fn with_receipt_signer(mut self, signer: Arc<dyn Signer + Send + Sync>) -> Self {
        self.receipt_signer = Some(signer);
        self
    }

    /// The registry root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The content-addressed object store.
    pub fn store(&self) -> &ObjectStore {
        &self.store
    }

    /// The transparency log.
    pub fn log(&self) -> &TransparencyLog {
        &self.log
    }

    /// The configured registry key, if any.
    pub fn registry_key(&self) -> Option<&RegistryKey> {
        self.registry_key.as_ref()
    }

    /// Store a publisher record and log it.
    ///
    /// # Errors
    ///
    /// As [`ObjectStore::put`], plus [`RegistryError::Io`] on log failure.
    pub fn put_publisher(&self, record: PublisherRecord, at: &str) -> Result<PublisherRecord> {
        let stored = self.store.put(record)?;
        self.log
            .append(&stored.publisher_id, LogObjectType::Publisher, at)?;
        Ok(stored)
    }

    /// Whether an object appears in the transparency log. A release absent from
    /// the log is unpublished, however complete its stored record looks.
    ///
    /// # Errors
    ///
    /// As [`TransparencyLog::entries`].
    pub fn is_published(&self, object_id: &str) -> Result<bool> {
        Ok(self.log.entry_for(object_id)?.is_some())
    }

    /// Every revocation on file, in log order.
    ///
    /// # Errors
    ///
    /// As [`ObjectStore::get`] and [`TransparencyLog::entries`].
    pub fn revocations(&self) -> Result<Vec<Revocation>> {
        self.log
            .objects_of_type(LogObjectType::Revocation)?
            .iter()
            .map(|id| self.store.get::<Revocation>(id))
            .collect()
    }

    /// Whether `subject` — a release id, publisher id, or `keyId` — is revoked.
    ///
    /// # Errors
    ///
    /// As [`Registry::revocations`].
    pub fn is_revoked(&self, subject: &str) -> Result<bool> {
        Ok(self.revocations()?.iter().any(|r| r.subject == subject))
    }

    /// Why a release may or may not run: published, revoked, or executable.
    ///
    /// A revoked release is still readable through [`Registry::store`] — this
    /// answers an execution question only (ADR-294 §9).
    ///
    /// # Errors
    ///
    /// As [`ObjectStore::get`] and [`Registry::revocations`].
    pub fn execution_status(&self, release_id: &str) -> Result<ExecutionStatus> {
        if !self.is_published(release_id)? {
            return Ok(ExecutionStatus::Unpublished);
        }
        let release: Release = self.store.get(release_id)?;
        let revoked: Vec<String> = self.revocations()?.into_iter().map(|r| r.subject).collect();

        let mut chain = vec![release_id.to_string(), release.package.publisher_id.clone()];
        if let Some(sig) = release.publisher_signature() {
            chain.push(sig.key_id.clone());
        }
        for subject in chain {
            if revoked.contains(&subject) {
                return Ok(ExecutionStatus::Revoked { subject });
            }
        }
        Ok(ExecutionStatus::Executable)
    }

    /// Whether a release is published and unrevoked.
    ///
    /// # Errors
    ///
    /// As [`Registry::execution_status`].
    pub fn is_executable(&self, release_id: &str) -> Result<bool> {
        Ok(self.execution_status(release_id)? == ExecutionStatus::Executable)
    }

    /// The trust level a release has actually earned: its published level,
    /// raised by any registry-signed [`TrustUpdate`] in the log.
    ///
    /// # Errors
    ///
    /// As [`ObjectStore::get`].
    pub fn effective_trust_level(&self, release_id: &str) -> Result<TrustLevel> {
        let release: Release = self.store.get(release_id)?;
        let mut level = release.trust_level;
        for id in self.log.objects_of_type(LogObjectType::TrustUpdate)? {
            let update: TrustUpdate = self.store.get(&id)?;
            if update.subject == release_id && update.trust_level > level {
                level = update.trust_level;
            }
        }
        Ok(level)
    }

    /// The release index for one package, in publication order.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Io`] on read failure, [`RegistryError::InvalidObject`]
    /// if the package name is not a safe path component or a line does not
    /// parse.
    pub fn releases_for(&self, publisher_id: &str, name: &str) -> Result<Vec<PackageIndexEntry>> {
        let path = self.package_index_path(publisher_id, name)?;
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(&path).map_err(|e| RegistryError::io(&path, &e))?;
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|line| {
                serde_json::from_str(line).map_err(|e| {
                    RegistryError::invalid(format!(
                        "{} has an unreadable line: {e}",
                        path.display()
                    ))
                })
            })
            .collect()
    }

    /// Every package the index directory holds, as `(publisherId, name)` pairs
    /// in path order.
    ///
    /// The layout puts the publisher's 64-hex digest above the package name, so
    /// an auditor who wants to walk every release index has no other way in:
    /// [`Registry::releases_for`] needs both halves up front.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Io`] on read failure, [`RegistryError::InvalidObject`]
    /// if a directory name is not a digest or not a safe package component.
    pub fn packages(&self) -> Result<Vec<(String, String)>> {
        let root = self.root.join(PACKAGES_DIR);
        if !root.exists() {
            return Ok(Vec::new());
        }
        let mut packages = Vec::new();
        for publisher_dir in read_dir_sorted(&root)? {
            if !publisher_dir.is_dir() {
                continue;
            }
            let hex = dir_name(&publisher_dir)?;
            let publisher_id = format!("{}{hex}", crate::objects::DIGEST_PREFIX);
            digest_hex("packages/<publisher>", &publisher_id)?;
            for package_dir in read_dir_sorted(&publisher_dir)? {
                if !package_dir.is_dir() {
                    continue;
                }
                let name = dir_name(&package_dir)?;
                safe_component(name)?;
                if package_dir.join(RELEASES_FILE).exists() {
                    packages.push((publisher_id.clone(), name.to_string()));
                }
            }
        }
        Ok(packages)
    }

    /// Every subject with a witness chain, in path order.
    ///
    /// The index filename is `sha256(subject)`, which cannot be inverted, so
    /// the subject is recovered from the first receipt in each chain — and the
    /// filename is then checked against it, which is what catches a chain file
    /// moved or renamed on disk.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Io`] on read failure, [`RegistryError::Log`] if a chain
    /// file is empty or its name does not match the subject it holds,
    /// otherwise as [`ObjectStore::get`].
    pub fn witness_subjects(&self) -> Result<Vec<String>> {
        let root = self.root.join(WITNESS_DIR);
        if !root.exists() {
            return Ok(Vec::new());
        }
        let mut subjects = Vec::new();
        for file in read_dir_sorted(&root)? {
            let Some(stem) = file
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".jsonl"))
            else {
                continue;
            };
            let text = fs::read_to_string(&file).map_err(|e| RegistryError::io(&file, &e))?;
            let first =
                text.lines()
                    .find(|l| !l.trim().is_empty())
                    .ok_or_else(|| RegistryError::Log {
                        detail: format!("witness chain {} is empty", file.display()),
                    })?;
            let receipt: crate::objects::WitnessReceipt = self.store.get(first.trim())?;
            let expected = rvf_forge_core::canonical::hex_lower(&crate::merkle::sha256(
                receipt.subject.as_bytes(),
            ));
            if expected != stem {
                return Err(RegistryError::Log {
                    detail: format!(
                        "witness chain {} holds subject {:?}, which hashes to {expected}",
                        file.display(),
                        receipt.subject
                    ),
                });
            }
            subjects.push(receipt.subject);
        }
        Ok(subjects)
    }

    /// An inclusion proof for a logged object against the current tree head.
    ///
    /// # Errors
    ///
    /// As [`TransparencyLog::inclusion_proof`].
    pub fn inclusion_proof(&self, object_id: &str) -> Result<InclusionProof> {
        self.log.inclusion_proof(object_id)
    }

    /// The current tree head.
    ///
    /// # Errors
    ///
    /// As [`TransparencyLog::tree_head`].
    pub fn tree_head(&self) -> Result<TreeHead> {
        self.log.tree_head()
    }

    /// Every log entry, in index order.
    ///
    /// # Errors
    ///
    /// As [`TransparencyLog::entries`].
    pub fn log_entries(&self) -> Result<Vec<TransparencyLogEntry>> {
        self.log.entries()
    }

    // --- internals shared with the publish and witness modules ---

    fn package_index_path(&self, publisher_id: &str, name: &str) -> Result<PathBuf> {
        let publisher = digest_hex("package.publisherId", publisher_id)?;
        Ok(self
            .root
            .join(PACKAGES_DIR)
            .join(publisher)
            .join(safe_component(name)?)
            .join(RELEASES_FILE))
    }

    fn append_package_index(&self, release: &Release) -> Result<()> {
        let path = self.package_index_path(&release.package.publisher_id, &release.package.name)?;
        let parent = path
            .parent()
            .ok_or_else(|| RegistryError::invalid("package index path has no parent"))?;
        fs::create_dir_all(parent).map_err(|e| RegistryError::io(parent, &e))?;

        let existing = self.releases_for(&release.package.publisher_id, &release.package.name)?;
        if existing.iter().any(|e| e.release_id == release.release_id) {
            return Ok(());
        }
        let entry = PackageIndexEntry {
            release_id: release.release_id.clone(),
            version: release.version.clone(),
            predecessor: release.predecessor.clone(),
            published_at: release.published_at.clone(),
        };
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| RegistryError::io(&path, &e))?;
        writeln!(file, "{}", to_canonical_json(&entry)?).map_err(|e| RegistryError::io(&path, &e))
    }

    /// Check that an object carries a registry-role signature over its id that
    /// verifies against the configured registry key.
    fn require_registry_signature<T: RegistryObject>(&self, object: &T, id: &str) -> Result<()> {
        let key = self
            .registry_key
            .as_ref()
            .ok_or_else(|| RegistryError::SignatureInvalid {
                detail: format!(
                    "no registry key is configured, so a {} cannot be accepted",
                    T::TYPE
                ),
            })?;
        let sig = object
            .signatures()
            .iter()
            .find(|s| s.role == SignatureRole::Registry)
            .ok_or_else(|| RegistryError::SignatureInvalid {
                detail: format!("{} carries no registry-role signature", T::TYPE),
            })?;
        if sig.key_id != key.key_id {
            return Err(RegistryError::SignatureInvalid {
                detail: format!(
                    "{} is countersigned by {:?}, not the registry key {:?}",
                    T::TYPE,
                    sig.key_id,
                    key.key_id
                ),
            });
        }
        verify_over_id(id, &sig.sig, &key.verifying_key)
    }
}

/// Directory entries in name order, so enumeration is reproducible.
fn read_dir_sorted(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| RegistryError::io(dir, &e))? {
        paths.push(entry.map_err(|e| RegistryError::io(dir, &e))?.path());
    }
    paths.sort();
    Ok(paths)
}

/// The final component of a path, as UTF-8.
fn dir_name(path: &Path) -> Result<&str> {
    path.file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| RegistryError::invalid(format!("{} is not a UTF-8 name", path.display())))
}

/// Reject anything that is not a safe single path component.
///
/// # Errors
///
/// [`RegistryError::InvalidObject`] for an empty, over-long, dot-leading, or
/// non-`[A-Za-z0-9._-]` name. Rejecting the leading dot is what makes `..`
/// unrepresentable, so a package name can never walk out of its directory.
pub fn safe_component(name: &str) -> Result<&str> {
    if name.is_empty() || name.len() > 128 {
        return Err(RegistryError::invalid(format!(
            "package name {name:?} must be 1..=128 characters"
        )));
    }
    if name.starts_with('.') {
        return Err(RegistryError::invalid(format!(
            "package name {name:?} may not start with a dot"
        )));
    }
    if !name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    {
        return Err(RegistryError::invalid(format!(
            "package name {name:?} may only contain ASCII alphanumerics, '.', '_', and '-'"
        )));
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_components_reject_traversal() {
        safe_component("my-agent").unwrap();
        safe_component("agent.v2_beta").unwrap();
        for bad in ["", ".", "..", "../etc", "a/b", "a\\b", "a b", ".hidden"] {
            assert!(safe_component(bad).is_err(), "{bad:?} should be rejected");
        }
        assert!(safe_component(&"a".repeat(129)).is_err());
    }

    #[test]
    fn opening_creates_the_documented_layout() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::open(dir.path()).unwrap();
        for sub in ["objects/sha256", "packages", "log", "witness"] {
            assert!(reg.root().join(sub).is_dir(), "{sub} missing");
        }
    }
}
