//! Filesystem helpers for the capsule directory.
//!
//! Only sealed capsule files are touched: a directory listing that swept up
//! anything else would let an unrelated file be counted as state, exported as
//! state, or deleted as state.

use std::path::{Path, PathBuf};

use super::{io, LibraryError, StateTransfer};

/// Sealed capsule files, ignoring anything else in the directory.
pub(super) fn sealed_files(dir: &Path) -> Result<Vec<PathBuf>, LibraryError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(io(dir, &e)),
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    Ok(files)
}

pub(super) fn dir_bytes(dir: &Path) -> Result<u64, LibraryError> {
    let mut total = 0;
    for path in sealed_files(dir)? {
        total += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    }
    Ok(total)
}

pub(super) fn clear_dir(dir: &Path) -> Result<usize, LibraryError> {
    let files = sealed_files(dir)?;
    for path in &files {
        std::fs::remove_file(path).map_err(|e| io(path, &e))?;
    }
    Ok(files.len())
}

pub(super) fn copy_dir(from: &Path, to: &Path, base_identity: &str) -> Result<StateTransfer, LibraryError> {
    std::fs::create_dir_all(to).map_err(|e| io(to, &e))?;
    let mut bytes = 0;
    let files = sealed_files(from)?;
    for path in &files {
        let name = path.file_name().unwrap_or_default();
        bytes += std::fs::copy(path, to.join(name)).map_err(|e| io(path, &e))?;
    }
    Ok(StateTransfer {
        base_identity: base_identity.to_string(),
        files: files.len(),
        bytes,
        path: to.display().to_string(),
    })
}

/// The installed artifact is read-only, so removing its directory needs the
/// bit cleared first on platforms where read-only blocks unlink.
pub(super) fn unlock_dir(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if let Ok(meta) = std::fs::metadata(&path) {
            let mut perms = meta.permissions();
            #[allow(clippy::permissions_set_readonly_false)]
            perms.set_readonly(false);
            let _ = std::fs::set_permissions(&path, perms);
        }
    }
}
