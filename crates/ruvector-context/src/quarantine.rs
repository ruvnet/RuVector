//! Shard files that failed verification at open time, and their remediation.
//!
//! Quarantine keeps one bad file from denying the whole index, but on its own
//! it converts that into a permanent, silent denial for the one tenant whose
//! name the file occupies. This module is what makes the state both visible
//! and recoverable.

use crate::shard::load_shard;
use crate::{ContextIndexError, ContextIndexOptions, ContextScope, Result};
use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::RwLock;

/// Shard filename to the reason it was not loaded.
pub(crate) type Quarantine = BTreeMap<String, String>;

/// One shard file that failed verification when the index was opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuarantinedShard {
    /// Opaque hash-bound shard filename.
    pub filename: String,
    /// Why the file was not loaded, for operator diagnosis.
    pub reason: String,
}

pub(crate) struct QuarantineList {
    entries: RwLock<Quarantine>,
}

impl QuarantineList {
    pub(crate) const fn new(entries: Quarantine) -> Self {
        Self {
            entries: RwLock::new(entries),
        }
    }

    /// Every quarantined file with the reason it was skipped.
    ///
    /// Filenames are one-way hashes, so the reason is what makes the list
    /// diagnosable; mapping a filename back to a tenant means hashing candidate
    /// scopes with [`ContextScope::shard_id`].
    pub(crate) fn list(&self) -> Result<Vec<QuarantinedShard>> {
        let entries = self.read()?;
        Ok(entries
            .iter()
            .map(|(filename, reason)| QuarantinedShard {
                filename: filename.clone(),
                reason: reason.clone(),
            })
            .collect())
    }

    /// Fail loudly when a scope's shard name is occupied by a quarantined file.
    ///
    /// Only the exact scope is checked. A search rooted at an ancestor still
    /// silently omits a quarantined descendant, because shard filenames are
    /// hashes and cannot be tested for subtree membership.
    pub(crate) fn ensure_absent(&self, scope: &ContextScope) -> Result<()> {
        let filename = scope.shard_filename();
        if self.read()?.contains_key(&filename) {
            return Err(ContextIndexError::ScopeQuarantined(filename));
        }
        Ok(())
    }

    /// Remove the quarantined file occupying one scope's shard name.
    ///
    /// Returns `false` when that name is not quarantined. The file is deleted
    /// only once this call has confirmed it is *not* a valid shard for `scope`:
    /// if it loads and its manifest hashes to this filename, the call fails
    /// with [`ContextIndexError::ShardNotDiscardable`] and the file is left
    /// alone. A symlink is discarded without being opened, because opening it
    /// is the thing quarantine exists to prevent.
    pub(crate) fn discard(
        &self,
        root: &Path,
        scope: &ContextScope,
        options: &ContextIndexOptions,
    ) -> Result<bool> {
        let filename = scope.shard_filename();
        let mut entries = self
            .entries
            .write()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        if !entries.contains_key(&filename) {
            return Ok(false);
        }
        let path = root.join(&filename);
        let is_symlink =
            std::fs::symlink_metadata(&path).is_ok_and(|meta| meta.file_type().is_symlink());
        if !is_symlink {
            if let Ok((loaded, shard)) = load_shard(&path, &filename, options) {
                drop(shard);
                if loaded.shard_filename() == filename {
                    return Err(ContextIndexError::ShardNotDiscardable(filename));
                }
            }
        }
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        entries.remove(&filename);
        Ok(true)
    }

    fn read(&self) -> Result<std::sync::RwLockReadGuard<'_, Quarantine>> {
        self.entries
            .read()
            .map_err(|_| ContextIndexError::LockPoisoned)
    }
}
