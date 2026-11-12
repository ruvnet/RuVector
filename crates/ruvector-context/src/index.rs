//! Persistent scope-sharded vector index.

use crate::layout::{
    acquire_root_lock, create_staging_dir, is_reserved_name, is_shard_name, prepare_root,
};
use crate::quarantine::{Quarantine, QuarantineList, QuarantinedShard};
use crate::shard::{create_shard, load_shard, Shard};
use crate::{ContextIndexError, ContextIndexOptions, ContextNamespace, ContextScope, Result};
use ruvector_core::{SearchQuery, VectorEntry};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

const MAX_POINT_ID_BYTES: usize = 512;

/// Immutable point registered in one exact context scope.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextPoint {
    /// Stable opaque identifier, normally derived from revision and view.
    pub id: String,
    /// Embedding in the configured vector space.
    pub vector: Vec<f32>,
}

/// One globally ranked match from an authorized scope subtree.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextMatch {
    /// Stable opaque point identifier.
    pub id: String,
    /// Distance score where lower values rank first.
    pub score: f32,
    /// Exact physical scope containing the point.
    pub scope: ContextScope,
}

/// Operational counters for one exact scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeStats {
    /// Number of vectors currently retained in the shard.
    pub vectors: usize,
    /// Number of ANN searches issued to this shard since process start.
    pub searches: u64,
}

type PathShards = BTreeMap<Vec<String>, Arc<Shard>>;
type NamespaceShards = BTreeMap<ContextNamespace, PathShards>;

/// Persistent vector index physically partitioned by exact context scope.
///
/// # Caller authorization
///
/// This type is a physical isolation mechanism, not an authorization service.
/// It never decides who may touch a scope: every method assumes the caller has
/// already authorized the supplied [`ContextScope`] for the requesting
/// principal. Passing an unauthorized scope yields that scope's data.
///
/// One live handle owns an index root exclusively. [`ScopedContextIndex::open`]
/// takes an advisory lock on the root directory, so a second handle over the
/// same root fails with [`ContextIndexError::RootLocked`] instead of silently
/// diverging from the first.
///
/// The root directory must be private to this user, the handle is exclusive,
/// and neither guarantee is portable off Unix. See the crate documentation for
/// what those preconditions rest on and where they stop.
pub struct ScopedContextIndex {
    root: PathBuf,
    staging: PathBuf,
    options: ContextIndexOptions,
    shards: RwLock<NamespaceShards>,
    quarantined: QuarantineList,
    _lock: File,
}

impl ScopedContextIndex {
    /// Open an index directory and recover every hash-bound scope shard.
    ///
    /// Takes an exclusive advisory lock on `root`; the lock is released when
    /// the returned handle is dropped, and by the operating system if the
    /// process exits without dropping it.
    ///
    /// Files that match the shard naming convention but fail to load, or whose
    /// stored manifest does not match their filename, are quarantined and
    /// skipped rather than failing the whole open. See
    /// [`ScopedContextIndex::quarantined_shards`].
    pub fn open(root: impl AsRef<Path>, options: ContextIndexOptions) -> Result<Self> {
        options.validate()?;
        let root = root.as_ref().to_path_buf();
        prepare_root(&root)?;
        let lock = acquire_root_lock(&root)?;
        let mut shards: NamespaceShards = BTreeMap::new();
        let mut quarantined = Quarantine::new();
        for entry in std::fs::read_dir(&root)? {
            let entry = entry?;
            let Some(filename) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if is_reserved_name(&filename) {
                // Scratch state from a create or a handle that did not finish,
                // or planted under a reserved name. This handle holds the root
                // lock, so no live writer owns it. Swept before the symlink
                // test so a symlink wearing a reserved name is removed rather
                // than accumulating.
                sweep_reserved_entry(&entry.path());
                continue;
            }
            if entry.file_type()?.is_symlink() {
                // Never hand a symlink to the vector engine. Its API is
                // path-based and it creates the backing file at whatever the
                // path resolves to, so opening one writes a tenant's shard
                // outside this root, at a location the attacker chose.
                // `load_shard` rejects it too; this arm exists to name the
                // cause in the quarantine reason.
                if is_shard_name(&filename) {
                    quarantined.insert(filename, "shard path is a symlink".to_string());
                }
                continue;
            }
            if !is_shard_name(&filename) {
                continue;
            }
            match load_shard(&entry.path(), &filename, &options) {
                Ok((scope, shard)) if filename == scope.shard_filename() => {
                    shards
                        .entry(scope.namespace().clone())
                        .or_default()
                        .insert(scope.path().to_vec(), Arc::new(shard));
                }
                Ok(_) => {
                    quarantined.insert(
                        filename,
                        "manifest scope does not hash to this filename".to_string(),
                    );
                }
                Err(error) => {
                    quarantined.insert(filename, error.to_string());
                }
            }
        }
        let staging = create_staging_dir(&root)?;
        let index = Self {
            root,
            staging,
            options,
            shards: RwLock::new(shards),
            quarantined: QuarantineList::new(quarantined),
            _lock: lock,
        };
        if index.scope_count()? > index.options.max_scopes {
            return Err(ContextIndexError::ScopeCapacity {
                maximum: index.options.max_scopes,
            });
        }
        Ok(index)
    }

    /// Insert a point, returning success for a byte-identical replay.
    ///
    /// A point is immutable only while it is present: once
    /// [`ScopedContextIndex::delete_point`] removes an identifier, the same
    /// identifier may be inserted again with different bytes.
    ///
    /// The caller must already have authorized `scope` for the requesting
    /// principal.
    pub fn insert(&self, scope: &ContextScope, point: ContextPoint) -> Result<()> {
        validate_point(&point, self.options.vector.dimensions)?;
        self.quarantined.ensure_absent(scope)?;
        let mut all = self
            .shards
            .write()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        let exists = all
            .get(scope.namespace())
            .and_then(|paths| paths.get(scope.path()))
            .cloned();
        let shard = if let Some(shard) = exists {
            shard
        } else {
            if count_scopes(&all) >= self.options.max_scopes {
                return Err(ContextIndexError::ScopeCapacity {
                    maximum: self.options.max_scopes,
                });
            }
            let shard = Arc::new(create_shard(
                &self.root,
                &self.staging,
                scope,
                &self.options,
            )?);
            all.entry(scope.namespace().clone())
                .or_default()
                .insert(scope.path().to_vec(), Arc::clone(&shard));
            shard
        };
        if let Some(existing) = shard.get(&point.id)? {
            return if existing.vector == point.vector {
                Ok(())
            } else {
                Err(ContextIndexError::ImmutableConflict)
            };
        }
        shard.insert(VectorEntry {
            id: Some(point.id),
            vector: point.vector,
            metadata: None,
        })
    }

    /// Search only exact shards at or below an already authorized root scope.
    ///
    /// The caller must already have authorized `root` for the requesting
    /// principal; every shard in its subtree is considered readable.
    ///
    /// Results are silently incomplete when a descendant scope is quarantined:
    /// shard filenames are hashes and cannot be tested for subtree membership,
    /// so an ancestor-rooted search cannot detect one. Searching the exact
    /// quarantined scope does fail with
    /// [`ContextIndexError::ScopeQuarantined`]. Callers that need certainty
    /// that a subtree was searched whole must consult
    /// [`ScopedContextIndex::quarantined_shards`].
    pub fn search(
        &self,
        root: &ContextScope,
        vector: &[f32],
        k: usize,
    ) -> Result<Vec<ContextMatch>> {
        if k == 0 || k > self.options.max_results {
            return Err(ContextIndexError::ResultLimit {
                requested: k,
                maximum: self.options.max_results,
            });
        }
        if vector.len() != self.options.vector.dimensions {
            return Err(ruvector_core::RuvectorError::DimensionMismatch {
                expected: self.options.vector.dimensions,
                actual: vector.len(),
            }
            .into());
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(ContextIndexError::NonFiniteVector);
        }
        let all = self
            .shards
            .read()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        let Some(paths) = all.get(root.namespace()) else {
            return Ok(Vec::new());
        };
        let selected = paths
            .range(root.path().to_vec()..)
            .take_while(|(path, _)| path.starts_with(root.path()))
            .collect::<Vec<_>>();
        if selected.len() > self.options.max_search_shards {
            return Err(ContextIndexError::ScopeFanout {
                actual: selected.len(),
                maximum: self.options.max_search_shards,
            });
        }
        let mut matches = Vec::with_capacity(k.min(crate::MAX_RESULT_LIMIT));
        for (path, shard) in selected {
            let results = shard.ann_search(SearchQuery {
                vector: vector.to_vec(),
                k,
                filter: None,
                ef_search: None,
            })?;
            let scope = ContextScope::new(root.namespace().clone(), path.clone())?;
            for result in results {
                push_top_match(
                    &mut matches,
                    k,
                    ContextMatch {
                        id: result.id,
                        score: result.score,
                        scope: scope.clone(),
                    },
                );
            }
        }
        matches.sort_unstable_by(compare_matches);
        Ok(matches)
    }

    /// Delete one point from one exact scope.
    ///
    /// The caller must already have authorized `scope` for the requesting
    /// principal.
    pub fn delete_point(&self, scope: &ContextScope, id: &str) -> Result<bool> {
        validate_point_id(id)?;
        // Without this, deleting from a quarantined scope reports "not
        // present" while the point's bytes sit in the quarantined file.
        self.quarantined.ensure_absent(scope)?;
        let all = self
            .shards
            .read()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        let Some(shard) = all
            .get(scope.namespace())
            .and_then(|paths| paths.get(scope.path()))
        else {
            return Ok(false);
        };
        shard.delete(id)
    }

    /// Erase one exact scope and its persistent vector shard.
    ///
    /// The shard file is unlinked before any in-memory state changes, so a
    /// failure to unlink is reported as an error with the scope still present
    /// rather than as a phantom erase.
    ///
    /// This ordering relies on POSIX semantics, where a file can be unlinked
    /// while it is still open. On Windows the shard is still open at this
    /// point and the unlink fails, so erasure there needs the engine handle
    /// closed first — at the cost of the guarantee above.
    ///
    /// The caller must already have authorized `scope` for the requesting
    /// principal.
    pub fn erase_scope(&self, scope: &ContextScope) -> Result<bool> {
        // A quarantined file still holds this scope's name on disk. Reporting
        // "nothing to erase" would be a false negative on an erasure request.
        self.quarantined.ensure_absent(scope)?;
        let mut all = self
            .shards
            .write()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        let Some(paths) = all.get_mut(scope.namespace()) else {
            return Ok(false);
        };
        if !paths.contains_key(scope.path()) {
            return Ok(false);
        }
        match std::fs::remove_file(self.root.join(scope.shard_filename())) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        drop(paths.remove(scope.path()));
        let namespace_is_empty = paths.is_empty();
        if namespace_is_empty {
            all.remove(scope.namespace());
        }
        Ok(true)
    }

    /// Return counters for one exact scope without exposing other scopes.
    ///
    /// This is an existence oracle over exact scopes and reports exact vector
    /// counts, so the caller must already have authorized `scope` for the
    /// requesting principal exactly as it would for a read.
    pub fn scope_stats(&self, scope: &ContextScope) -> Result<Option<ScopeStats>> {
        self.quarantined.ensure_absent(scope)?;
        let all = self
            .shards
            .read()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        let Some(shard) = all
            .get(scope.namespace())
            .and_then(|paths| paths.get(scope.path()))
        else {
            return Ok(None);
        };
        Ok(Some(ScopeStats {
            vectors: shard.len()?,
            searches: shard.searches(),
        }))
    }

    /// Number of exact-scope shards currently open.
    pub fn scope_count(&self) -> Result<usize> {
        let all = self
            .shards
            .read()
            .map_err(|_| ContextIndexError::LockPoisoned)?;
        Ok(count_scopes(&all))
    }

    /// Shard files skipped at open time, each with the reason it was skipped.
    ///
    /// Operations naming a quarantined scope fail with
    /// [`ContextIndexError::ScopeQuarantined`] rather than silently reporting
    /// the scope as absent.
    pub fn quarantined_shards(&self) -> Result<Vec<QuarantinedShard>> {
        self.quarantined.list()
    }

    /// Discard the quarantined file occupying one scope's shard name.
    ///
    /// Returns `false` when that name is not quarantined. Without this, one
    /// corrupted file denies its tenant every operation permanently, with no
    /// route back short of filesystem surgery. A file that turns out to be a
    /// valid shard for `scope` is never deleted; see
    /// [`ContextIndexError::ShardNotDiscardable`].
    ///
    /// The caller must already have authorized `scope` for the requesting
    /// principal.
    pub fn discard_quarantined(&self, scope: &ContextScope) -> Result<bool> {
        self.quarantined.discard(&self.root, scope, &self.options)
    }
}

impl Drop for ScopedContextIndex {
    fn drop(&mut self) {
        // Best effort: a staging directory left by a crash is swept by the
        // next handle to take the root lock.
        let _ = std::fs::remove_dir_all(&self.staging);
    }
}

/// Remove one reserved entry, whatever kind of thing occupies the name.
///
/// `remove_file` covers regular files and symlinks; a staging directory needs
/// `remove_dir_all`, which does not follow symlinks out of the tree it is
/// removing. Nothing under a reserved name is ever read, so a failure here
/// leaks a file rather than affecting correctness.
fn sweep_reserved_entry(path: &Path) {
    if std::fs::remove_file(path).is_err() {
        let _ = std::fs::remove_dir_all(path);
    }
}

fn validate_point(point: &ContextPoint, dimensions: usize) -> Result<()> {
    validate_point_id(&point.id)?;
    if point.vector.len() != dimensions {
        return Err(ruvector_core::RuvectorError::DimensionMismatch {
            expected: dimensions,
            actual: point.vector.len(),
        }
        .into());
    }
    if point.vector.iter().any(|value| !value.is_finite()) {
        return Err(ContextIndexError::NonFiniteVector);
    }
    Ok(())
}

fn validate_point_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > MAX_POINT_ID_BYTES
        || !id.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b':')
        })
    {
        return Err(ContextIndexError::InvalidPointId);
    }
    Ok(())
}

fn count_scopes(shards: &NamespaceShards) -> usize {
    shards.values().map(BTreeMap::len).sum()
}

fn compare_matches(left: &ContextMatch, right: &ContextMatch) -> core::cmp::Ordering {
    left.score
        .partial_cmp(&right.score)
        .unwrap_or(core::cmp::Ordering::Equal)
        .then_with(|| left.id.cmp(&right.id))
        .then_with(|| left.scope.cmp(&right.scope))
}

fn push_top_match(matches: &mut Vec<ContextMatch>, limit: usize, candidate: ContextMatch) {
    if matches.len() < limit {
        matches.push(candidate);
        return;
    }
    let worst = matches
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| compare_matches(left, right))
        .map_or(0, |(index, _)| index);
    if compare_matches(&candidate, &matches[worst]).is_lt() {
        matches[worst] = candidate;
    }
}
