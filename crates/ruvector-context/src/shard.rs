//! Shard files: naming, creation, recovery, and the root lock.
//!
//! The vector engine backing one exact scope is wrapped here so the per-scope
//! isolation counter cannot be bypassed, and so the rules that decide whether a
//! file on disk may be treated as this scope's shard live in one place.
//!
//! Every path this module hands to the vector engine is one it created itself
//! under a name it claimed atomically. The engine's API is path-based and it
//! creates whatever file the path resolves to, so a symlink anywhere on that
//! path would let an attacker choose where a tenant's vectors are written.

use crate::layout::{file_identity, hard_link_count, scratch_name, FileIdentity};
use crate::{ContextIndexError, ContextIndexOptions, ContextScope, Result};
use ruvector_core::types::DbOptions;
use ruvector_core::{SearchQuery, SearchResult, VectorDB, VectorEntry};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MANIFEST_KEY: &str = "ruvector_context_scope_v1";

/// Handle to the vector engine backing one exact context scope.
///
/// The wrapped [`VectorDB`] is private to this module, so the per-scope
/// isolation counter cannot be bypassed by accident: [`Shard::ann_search`] is
/// the only route to the approximate-nearest-neighbor index and it always
/// counts the traversal. Touching a shard without counting does not compile.
pub(crate) struct Shard {
    db: VectorDB,
    searches: AtomicU64,
}

impl Shard {
    /// Wrap an opened vector database as a countable shard.
    fn new(db: VectorDB) -> Self {
        Self {
            db,
            searches: AtomicU64::new(0),
        }
    }

    /// Configuration the engine actually opened with, stored config included.
    fn options(&self) -> &DbOptions {
        self.db.options()
    }

    /// Fetch one stored point by identifier.
    pub(crate) fn get(&self, id: &str) -> Result<Option<VectorEntry>> {
        Ok(self.db.get(id)?)
    }

    /// Store one point.
    pub(crate) fn insert(&self, entry: VectorEntry) -> Result<()> {
        self.db.insert(entry)?;
        Ok(())
    }

    /// Remove one point, reporting whether it was present.
    pub(crate) fn delete(&self, id: &str) -> Result<bool> {
        Ok(self.db.delete(id)?)
    }

    /// Number of points currently retained.
    pub(crate) fn len(&self) -> Result<usize> {
        Ok(self.db.len()?)
    }

    /// Whether the shard holds no points at all.
    fn is_empty(&self) -> Result<bool> {
        Ok(self.db.is_empty()?)
    }

    /// Persist one engine-level configuration value.
    fn save_config_value(&self, key: &str, value: &str) -> Result<()> {
        Ok(self.db.save_config_value(key, value)?)
    }

    /// Read back one engine-level configuration value.
    fn load_config_value(&self, key: &str) -> Result<Option<String>> {
        Ok(self.db.load_config_value(key)?)
    }

    /// Number of ANN traversals issued to this shard since it was opened.
    pub(crate) fn searches(&self) -> u64 {
        self.searches.load(Ordering::Relaxed)
    }

    /// The only accessor that reaches the ANN index; every call is counted.
    pub(crate) fn ann_search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        self.searches.fetch_add(1, Ordering::Relaxed);
        Ok(self.db.search(query)?)
    }
}

/// Build a new shard for `scope` and publish it under its hash-bound name.
///
/// The shard is built under a scratch name this call claimed with `O_EXCL`,
/// then linked into place. Two properties make this safe, and neither is a
/// check on a path:
///
/// - `link(2)` fails with `EEXIST` against anything already at the destination
///   — regular file, directory, or symlink, dangling or not — and never
///   follows it, so a planted name cannot be adopted or redirected.
/// - The `(device, inode)` of the scratch file is captured from its own file
///   descriptor at the instant `O_EXCL` proved the name was ours. After the
///   link, the published name is required to resolve to *that* inode. An
///   attacker who renames something over the scratch name mid-build steals
///   only the name; the published entry is unlinked and the create fails.
///
/// The second property is what a path check cannot give: names are re-resolved
/// on every syscall, so only the inode identity is stable across the build.
///
/// The build itself happens inside `staging`, a directory only this user may
/// write. That is not defence in depth, it is load-bearing: the engine opens
/// by path, so between claiming a scratch name and the engine opening it,
/// anyone who can write to the containing directory can rename their own file
/// over that name and have the engine initialise it — at a location outside
/// the index root if they aim a symlink there. No check inside this process
/// can prevent that; only taking away the ability to perform the rename can.
pub(crate) fn create_shard(
    root: &Path,
    staging: &Path,
    scope: &ContextScope,
    options: &ContextIndexOptions,
) -> Result<Shard> {
    let filename = scope.shard_filename();
    let (scratch, identity) = claim_scratch_path(staging)?;
    let shard = match build_shard(&scratch, scope, options) {
        Ok(shard) => shard,
        Err(error) => {
            let _ = std::fs::remove_file(&scratch);
            return Err(error);
        }
    };
    let published = root.join(&filename);
    if let Err(error) = std::fs::hard_link(&scratch, &published) {
        drop(shard);
        let _ = std::fs::remove_file(&scratch);
        return Err(match error.kind() {
            ErrorKind::AlreadyExists => ContextIndexError::ShardAdoption(filename),
            _ => error.into(),
        });
    }
    // The scratch name may have been replaced between the claim and the link,
    // in which case the entry just created names someone else's inode. Unlink
    // it: this call created that directory entry, so removing it cannot
    // destroy another scope's shard.
    if let Err(error) = confirm_published_identity(&published, identity) {
        drop(shard);
        let _ = std::fs::remove_file(&published);
        let _ = std::fs::remove_file(&scratch);
        return Err(error);
    }
    // The engine holds the inode open, so dropping the scratch name leaves the
    // shard reachable only under the name its scope hashes to.
    let _ = std::fs::remove_file(&scratch);
    Ok(shard)
}

/// Require the published name to resolve to the inode this call created.
fn confirm_published_identity(published: &Path, identity: FileIdentity) -> Result<()> {
    let metadata = std::fs::symlink_metadata(published)?;
    let name = shard_error_name(published);
    if !metadata.is_file() || file_identity(&metadata) != identity {
        return Err(ContextIndexError::ShardAdoption(name));
    }
    Ok(())
}

fn build_shard(path: &Path, scope: &ContextScope, options: &ContextIndexOptions) -> Result<Shard> {
    let mut vector = options.vector.clone();
    vector.storage_path = path.to_string_lossy().into_owned();
    let shard = Shard::new(VectorDB::new(vector)?);
    // The scratch file was created empty under a name this call claimed with
    // `O_EXCL`, so the engine cannot have adopted anything. Assert it anyway:
    // silent adoption is the single failure this whole path exists to prevent.
    if shard.load_config_value(MANIFEST_KEY)?.is_some()
        || !shard.is_empty()?
        || engine_fingerprint(shard.options())? != engine_fingerprint(&options.vector)?
    {
        return Err(ContextIndexError::ShardAdoption(scope.shard_filename()));
    }
    shard.save_config_value(MANIFEST_KEY, &serde_json::to_string(scope)?)?;
    Ok(shard)
}

/// Claim an unused scratch name inside `root` with `O_CREAT | O_EXCL`, and
/// capture the identity of the file that create produced.
///
/// The exclusive create is what matters, not the name's unpredictability: a
/// pre-planted name — symlink included — fails the create and costs only a
/// retry. The identity comes from the returned descriptor rather than a second
/// lookup of the name, so it describes the file this call made and nothing
/// that later takes its place.
fn claim_scratch_path(staging: &Path) -> Result<(PathBuf, FileIdentity)> {
    for _ in 0..16 {
        let path = staging.join(scratch_name());
        let mut open_options = std::fs::OpenOptions::new();
        open_options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            // The published shard keeps whatever mode the scratch file was
            // created with, so this is where a shard's privacy is decided.
            open_options.mode(0o600);
        }
        match open_options.open(&path) {
            Ok(file) => {
                let identity = file_identity(&file.metadata()?);
                return Ok((path, identity));
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(std::io::Error::new(
        ErrorKind::AlreadyExists,
        "could not claim a scratch name for a new context shard",
    )
    .into())
}

pub(crate) fn load_shard(
    path: &Path,
    filename: &str,
    options: &ContextIndexOptions,
) -> Result<(ContextScope, Shard)> {
    require_exclusive_regular_file(path, filename)?;
    let mut vector = options.vector.clone();
    vector.storage_path = path.to_string_lossy().into_owned();
    let shard = Shard::new(VectorDB::new(vector)?);
    if engine_fingerprint(shard.options())? != engine_fingerprint(&options.vector)? {
        return Err(ContextIndexError::CorruptShard(filename.to_string()));
    }
    let manifest = shard
        .load_config_value(MANIFEST_KEY)?
        .ok_or_else(|| ContextIndexError::CorruptShard(filename.to_string()))?;
    let scope: ContextScope = serde_json::from_str(&manifest)
        .map_err(|_| ContextIndexError::CorruptShard(filename.to_string()))?;
    Ok((scope, shard))
}

/// Refuse to hand the engine anything but an unaliased regular file.
///
/// The engine opens read-write and initialises whatever the path resolves to,
/// so this runs before it sees the path. A symlink is rejected by `is_file`; a
/// hard-link alias — indistinguishable from a regular file by type alone — is
/// rejected by its link count. Every shard this crate publishes settles at one
/// link, because `create_shard` unlinks the scratch name, and an open
/// descriptor does not raise the count.
///
/// A link count of one is not a statement of trust. It says only that no
/// second name existed at the instant of the `lstat`, not that the file is
/// this index's, nor that it will still be unaliased a syscall later. What
/// keeps an attacker from creating that second name is the private root, not
/// this check; this is defence in depth against operator error.
fn require_exclusive_regular_file(path: &Path, filename: &str) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() || hard_link_count(&metadata).is_some_and(|links| links != 1) {
        return Err(ContextIndexError::AliasedShardFile(filename.to_string()));
    }
    Ok(())
}

fn shard_error_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_string()
}

/// Full engine configuration, minus the storage path.
///
/// `VectorDB::new` adopts a stored database's `hnsw_config` and `quantization`
/// along with its dimensions and metric, so comparing only the latter two lets
/// an adopted file dictate engine parameters — lossy quantization, or a
/// degenerate graph. The storage path is excluded because it legitimately
/// differs between the scratch name a shard is built under and its published
/// name.
fn engine_fingerprint(options: &DbOptions) -> Result<String> {
    let normalized = DbOptions {
        storage_path: String::new(),
        ..options.clone()
    };
    Ok(serde_json::to_string(&normalized)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{create_staging_dir, is_reserved_name, is_shard_name};

    #[test]
    fn scratch_names_are_claimed_exclusively_and_distinctly() {
        let root = tempfile::tempdir().unwrap();
        let staging = create_staging_dir(root.path()).unwrap();
        let (first, first_id) = claim_scratch_path(&staging).unwrap();
        let (second, second_id) = claim_scratch_path(&staging).unwrap();
        assert_ne!(first, second);
        for path in [&first, &second] {
            let name = path.file_name().unwrap().to_str().unwrap();
            assert!(is_reserved_name(name));
            assert!(!is_shard_name(name));
        }
        if cfg!(unix) {
            assert!(first_id.is_some(), "unix must capture an inode identity");
            assert_ne!(first_id, second_id, "distinct files, distinct inodes");
        }
    }

    /// The `nlink == 1` rule in `require_exclusive_regular_file` is only sound
    /// if every shard this crate publishes actually settles at one link.
    #[test]
    fn a_published_shard_settles_at_one_hard_link() {
        let root = tempfile::tempdir().unwrap();
        let options = ContextIndexOptions {
            vector: DbOptions {
                dimensions: 3,
                ..DbOptions::default()
            },
            max_scopes: 4,
            max_search_shards: 4,
            max_results: 8,
        };
        let namespace =
            crate::ContextNamespace::new("context.example", "acme", "agent", "reader", "memory")
                .unwrap();
        let scope = ContextScope::new(namespace, vec!["links".into()]).unwrap();
        let staging = create_staging_dir(root.path()).unwrap();
        let shard = create_shard(root.path(), &staging, &scope, &options).unwrap();

        let path = root.path().join(scope.shard_filename());
        let held = std::fs::symlink_metadata(&path).unwrap();
        assert_eq!(hard_link_count(&held), Some(1), "open handle raised nlink");
        drop(shard);
        let closed = std::fs::symlink_metadata(&path).unwrap();
        assert_eq!(hard_link_count(&closed), Some(1));
        require_exclusive_regular_file(&path, "shard").unwrap();
    }
}
