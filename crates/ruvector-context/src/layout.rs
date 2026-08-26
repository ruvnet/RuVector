//! Where an index keeps its files, what it names them, and who may reach them.
//!
//! The root directory is the boundary this crate rests on. Every name beneath
//! it is re-resolved by the kernel on each syscall and none of them is held by
//! descriptor, so a user who can write the root can substitute any of them
//! between one syscall and the next. Nothing checked inside this process
//! closes that; taking away their write access does, which is why `open`
//! refuses a root other users can reach rather than defending the names one
//! at a time.

use crate::{ContextIndexError, Result};
use fs4::TryLockError;
use std::fs::File;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Name of the exclusive root lock file held for the lifetime of one index.
///
/// It is deliberately not a valid shard name, so [`is_shard_name`] skips it.
const ROOT_LOCK_FILENAME: &str = ".lock";

/// Prefix of the scratch name a shard is built under before it is published.
const TEMP_PREFIX: &str = ".create-";

/// Prefix of the private per-handle directory new shards are built inside.
const STAGING_PREFIX: &str = ".staging-";

/// Name for a fresh scratch file, claimed by an atomic create at the call site.
pub(crate) fn scratch_name() -> String {
    format!("{TEMP_PREFIX}{}", unique_suffix())
}

/// Open the exclusive root lock, refusing to follow a symlink at its path.
pub(crate) fn acquire_root_lock(root: &Path) -> Result<File> {
    let path = root.join(ROOT_LOCK_FILENAME);
    let file = match lock_open_options().open(&path) {
        Ok(file) => file,
        Err(error) => return Err(classify_lock_open_error(&path, error)),
    };
    // Called through the trait explicitly: `File` grew an inherent `try_lock`
    // in Rust 1.89, and letting method resolution pick it would mean this code
    // locks via std on new toolchains and via fs4 on the 1.77 MSRV.
    match fs4::FileExt::try_lock(&file) {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(ContextIndexError::RootLocked),
        Err(TryLockError::Error(error)) => Err(error.into()),
    }
}

fn lock_open_options() -> std::fs::OpenOptions {
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        // A symlink planted at the lock path would otherwise let an attacker
        // choose which file this index opens for writing, and which lock the
        // single-handle guarantee actually rests on.
        options.custom_flags(libc::O_NOFOLLOW).mode(0o600);
    }
    options
}

fn classify_lock_open_error(path: &Path, error: std::io::Error) -> ContextIndexError {
    if std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return ContextIndexError::UnsafeRootLock;
    }
    error.into()
}

/// Create the index root if absent and require it to be private.
///
/// The root is the containing directory for every name this crate defends, so
/// its permissions — not any check on those names — are what keep another user
/// from swapping them. `mkdir` applies the mode itself, leaving no window in
/// which the directory exists while still writable by others; an existing root
/// is not modified, only inspected, so an operator's deliberate permissions
/// are reported rather than silently widened or narrowed.
pub(crate) fn prepare_root(root: &Path) -> Result<()> {
    if let Some(parent) = root.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(0o700);
    }
    match builder.create(root) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    require_private_root(root)
}

fn require_private_root(root: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(root)?;
    if !metadata.is_dir() {
        return Err(ContextIndexError::InsecureRoot(
            "not a directory".to_string(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        use std::os::unix::fs::PermissionsExt as _;
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(ContextIndexError::InsecureRoot(format!(
                "mode {mode:04o} grants access to other users"
            )));
        }
        // Mode alone is not ownership: a 0700 directory owned by someone
        // else passes the bit test while every operation inside it is at
        // that other user's mercy (they can chmod it back open, or it can
        // be a root substituted under us by whoever controls the parent).
        // Requiring our own euid closes that: a substituted root would have
        // to be both owned by us AND 0700, which an attacker cannot mint.
        // Every root this newly rejects already failed later with a bare
        // EACCES, so no working deployment changes behavior — the broken
        // ones just fail earlier, with a diagnosable error.
        let owner = metadata.uid();
        let euid = rustix::process::geteuid().as_raw();
        if owner != euid {
            return Err(ContextIndexError::InsecureRoot(format!(
                "owned by uid {owner}, but this process runs as uid {euid}"
            )));
        }
    }
    Ok(())
}

/// Create this handle's private staging directory inside `root`.
///
/// The name is unique per handle and `mkdir` is atomic, so success proves this
/// call created it; the mode is applied by `mkdir` itself rather than by a
/// follow-up `chmod`, leaving no window in which the directory exists while
/// still writable by others.
pub(crate) fn create_staging_dir(root: &Path) -> Result<PathBuf> {
    for _ in 0..16 {
        let path = root.join(format!("{STAGING_PREFIX}{}", unique_suffix()));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt as _;
            builder.mode(0o700);
        }
        match builder.create(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(std::io::Error::new(
        ErrorKind::AlreadyExists,
        "could not claim a staging directory for this context index",
    )
    .into())
}

/// Whether a directory entry is scratch state rather than index content.
///
/// Reserved entries are never read, so any of them left by a crashed process —
/// or planted by someone else — can be swept when a handle takes the root lock.
pub(crate) fn is_reserved_name(name: &str) -> bool {
    name.starts_with(TEMP_PREFIX) || name.starts_with(STAGING_PREFIX)
}

/// A name component unlikely to collide, retried under an atomic create.
pub(crate) fn unique_suffix() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:016x}{nonce:016x}", stamp ^ u64::from(std::process::id()))
}

/// Stable identity of a file, independent of every name that reaches it.
///
/// `None` outside Unix, where the standard library exposes no inode: the
/// identity comparison then degrades to the file-type check alone.
pub(crate) type FileIdentity = Option<(u64, u64)>;

#[cfg(unix)]
pub(crate) fn file_identity(metadata: &std::fs::Metadata) -> FileIdentity {
    use std::os::unix::fs::MetadataExt as _;
    Some((metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
pub(crate) fn file_identity(_metadata: &std::fs::Metadata) -> FileIdentity {
    None
}

#[cfg(unix)]
pub(crate) fn hard_link_count(metadata: &std::fs::Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt as _;
    Some(metadata.nlink())
}

#[cfg(not(unix))]
pub(crate) fn hard_link_count(_metadata: &std::fs::Metadata) -> Option<u64> {
    None
}

/// Whether a directory entry is a shard file belonging to this index.
pub(crate) fn is_shard_name(name: &str) -> bool {
    // The root lock and reserved scratch state are not shards; their names
    // cannot collide with one, but the exclusions are stated rather than
    // inferred from the length check.
    name != ROOT_LOCK_FILENAME
        && !is_reserved_name(name)
        && name.len() == 69
        && name.ends_with(".redb")
        && name[..64]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_names_are_never_mistaken_for_shards() {
        assert!(!is_shard_name(ROOT_LOCK_FILENAME));
        assert!(!is_shard_name(&format!("{TEMP_PREFIX}{}", "a".repeat(61))));
        assert!(!is_shard_name(&format!(
            "{STAGING_PREFIX}{}",
            "a".repeat(60)
        )));
        assert!(is_shard_name(&format!("{}.redb", "a".repeat(64))));
        assert!(!is_shard_name(&format!("{}.redb", "g".repeat(64))));
    }
}
