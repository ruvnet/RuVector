//! FilesystemBackend — local disk backend with path traversal protection.
//!
//! Implements the Backend trait for local filesystem operations.
//! Uses `virtual_mode=true` by default (ADR-103 C1/SEC-002).
//! All file operations use `spawn_blocking` (ADR-103 A3).
//! Grep uses literal string matching, not regex (ADR-103 C13).
//! Glob uses walkdir with `follow_links(false)` (ADR-103 C1).
//! Atomic resolve+open via O_NOFOLLOW + /proc/self/fd verification (SEC-001).

use crate::protocol::*;
use crate::utils::format_content_with_line_numbers;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Inner state for `FilesystemBackend`, wrapped in `Arc` for cheap cloning
/// into `spawn_blocking` closures (ADR-103 A3).
#[derive(Debug, Clone)]
struct FilesystemBackendInner {
    cwd: PathBuf,
    virtual_mode: bool,
    max_file_size_bytes: u64,
}

/// Local filesystem backend with security hardening.
///
/// - `virtual_mode` (default `true`): restricts all paths to be relative to `cwd`
/// - Path traversal protection via `resolve_path()`
/// - Atomic resolve+open with post-open verification (SEC-001)
/// - `follow_links(false)` for glob/walkdir operations (SEC-004)
#[derive(Debug, Clone)]
pub struct FilesystemBackend {
    inner: Arc<FilesystemBackendInner>,
}

impl FilesystemBackend {
    /// Create a new filesystem backend rooted at `cwd`.
    ///
    /// `virtual_mode` defaults to `true` per ADR-103 SEC-002.
    pub fn new(cwd: PathBuf) -> Self {
        Self {
            inner: Arc::new(FilesystemBackendInner {
                cwd,
                virtual_mode: true,
                max_file_size_bytes: 10 * 1024 * 1024, // 10 MB
            }),
        }
    }

    /// Create a filesystem backend with explicit options.
    pub fn with_options(cwd: PathBuf, virtual_mode: bool, max_file_size_bytes: u64) -> Self {
        Self {
            inner: Arc::new(FilesystemBackendInner {
                cwd,
                virtual_mode,
                max_file_size_bytes,
            }),
        }
    }

    /// Get the current working directory.
    pub fn cwd(&self) -> &Path {
        &self.inner.cwd
    }

    /// Whether virtual mode is enabled.
    pub fn virtual_mode(&self) -> bool {
        self.inner.virtual_mode
    }

    /// Resolve a user-provided path into a canonical filesystem path.
    ///
    /// In virtual mode, all paths are treated as relative to `cwd`.
    /// Traversal sequences (`..`) are rejected.
    pub fn resolve_path(&self, path: &str) -> Result<PathBuf, FileOperationError> {
        let path = path.trim();
        if path.is_empty() {
            return Ok(self.inner.cwd.clone());
        }

        // Reject null bytes
        if path.contains('\0') {
            return Err(FileOperationError::SecurityViolation(
                "path contains null byte".to_string(),
            ));
        }

        // Check for path traversal
        if crate::utils::contains_traversal(path) {
            return Err(FileOperationError::SecurityViolation(
                "path traversal detected".to_string(),
            ));
        }

        // Reject tilde expansion
        if path.starts_with('~') {
            return Err(FileOperationError::SecurityViolation(
                "tilde expansion not allowed".to_string(),
            ));
        }

        let resolved = if self.inner.virtual_mode {
            // In virtual mode, strip leading '/' and treat as relative
            let relative = path.strip_prefix('/').unwrap_or(path);
            self.inner.cwd.join(relative)
        } else {
            let p = PathBuf::from(path);
            if p.is_absolute() {
                p
            } else {
                self.inner.cwd.join(path)
            }
        };

        // Verify the resolved path is within cwd (in virtual mode)
        if self.inner.virtual_mode {
            // Normalize without following symlinks — use lexical normalization
            let normalized = lexical_normalize(&resolved);
            let cwd_normalized = lexical_normalize(&self.inner.cwd);
            if !normalized.starts_with(&cwd_normalized) {
                return Err(FileOperationError::SecurityViolation(
                    "resolved path escapes sandbox root".to_string(),
                ));
            }
        }

        Ok(resolved)
    }

    /// Atomic resolve+open using O_NOFOLLOW + /proc/self/fd verification (SEC-001).
    ///
    /// This prevents TOCTOU symlink race conditions by verifying the
    /// actual opened file descriptor points within the allowed root.
    /// Supports both read and write operations.
    #[cfg(unix)]
    fn resolve_and_open(
        &self,
        path: &str,
        write: bool,
    ) -> Result<std::fs::File, FileOperationError> {
        use std::os::unix::fs::OpenOptionsExt;
        use std::os::unix::io::AsRawFd;

        let resolved = self.resolve_path(path)?;

        let mut opts = std::fs::OpenOptions::new();
        opts.read(true);
        if write {
            opts.write(true).create(true);
        }
        opts.custom_flags(libc::O_NOFOLLOW); // Don't follow symlinks

        let file = opts.open(&resolved).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => FileOperationError::FileNotFound,
            std::io::ErrorKind::PermissionDenied => FileOperationError::PermissionDenied,
            _ => FileOperationError::IoError(e.to_string()),
        })?;

        // Post-open verification via /proc/self/fd/N (Linux) or F_GETPATH (macOS)
        if self.inner.virtual_mode {
            #[cfg(target_os = "linux")]
            {
                let fd_path = format!("/proc/self/fd/{}", file.as_raw_fd());
                let real_path = std::fs::read_link(&fd_path)
                    .map_err(|e| FileOperationError::IoError(e.to_string()))?;

                let cwd_canonical = self
                    .inner
                    .cwd
                    .canonicalize()
                    .unwrap_or_else(|_| self.inner.cwd.clone());

                if !real_path.starts_with(&cwd_canonical) {
                    return Err(FileOperationError::PathEscapesRoot(path.to_string()));
                }
            }

            #[cfg(target_os = "macos")]
            {
                use std::ffi::OsStr;
                use std::os::unix::ffi::OsStrExt;

                let mut buf = vec![0u8; libc::PATH_MAX as usize];
                let fd = file.as_raw_fd();

                unsafe {
                    if libc::fcntl(fd, libc::F_GETPATH, buf.as_mut_ptr()) == -1 {
                        return Err(FileOperationError::IoError("F_GETPATH failed".into()));
                    }
                }

                let real_path = std::path::PathBuf::from(
                    OsStr::from_bytes(&buf[..buf.iter().position(|&b| b == 0).unwrap_or(0)])
                );

                let cwd_canonical = self
                    .inner
                    .cwd
                    .canonicalize()
                    .unwrap_or_else(|_| self.inner.cwd.clone());

                if !real_path.starts_with(&cwd_canonical) {
                    return Err(FileOperationError::PathEscapesRoot(path.to_string()));
                }
            }

            // For other Unix platforms, fall back to basic check
            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            {
                // Basic canonicalization check (less robust but better than nothing)
                if let Ok(canonical) = std::fs::canonicalize(&resolved) {
                    let cwd_canonical = self
                        .inner
                        .cwd
                        .canonicalize()
                        .unwrap_or_else(|_| self.inner.cwd.clone());

                    if !canonical.starts_with(&cwd_canonical) {
                        return Err(FileOperationError::PathEscapesRoot(path.to_string()));
                    }
                }
            }
        }

        Ok(file)
    }

    /// Synchronous read_file implementation for use within spawn_blocking.
    fn read_file_sync(
        &self,
        file_path: &str,
        offset: usize,
        limit: usize,
    ) -> Result<String, FileOperationError> {
        #[cfg(unix)]
        {
            // Use atomic resolve+open with TOCTOU protection
            use std::io::Read;

            let file = self.resolve_and_open(file_path, false)?;
            let metadata = file.metadata().map_err(|e| FileOperationError::IoError(e.to_string()))?;

            if metadata.is_dir() {
                return Err(FileOperationError::IsDirectory);
            }

            if metadata.len() > self.inner.max_file_size_bytes {
                return Err(FileOperationError::SecurityViolation(format!(
                    "file size {} exceeds limit {}",
                    metadata.len(),
                    self.inner.max_file_size_bytes
                )));
            }

            let mut content = String::new();
            let mut reader = std::io::BufReader::new(file);
            reader.read_to_string(&mut content)
                .map_err(|e| FileOperationError::IoError(e.to_string()))?;

            let lines: Vec<&str> = content.lines().collect();
            let total = lines.len();
            let start = offset.min(total);
            let end = if limit == 0 {
                total
            } else {
                (start + limit).min(total)
            };

            let selected_content = lines[start..end].join("\n");
            Ok(format_content_with_line_numbers(
                &selected_content,
                start + 1,
                2000,
            ))
        }

        #[cfg(not(unix))]
        {
            // Fallback for non-Unix platforms
            let resolved = self.resolve_path(file_path)?;

            let metadata = std::fs::metadata(&resolved).map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => FileOperationError::FileNotFound,
                std::io::ErrorKind::PermissionDenied => FileOperationError::PermissionDenied,
                _ => FileOperationError::InvalidPath,
            })?;

            if metadata.is_dir() {
                return Err(FileOperationError::IsDirectory);
            }

            if metadata.len() > self.inner.max_file_size_bytes {
                return Err(FileOperationError::SecurityViolation(format!(
                    "file size {} exceeds limit {}",
                    metadata.len(),
                    self.inner.max_file_size_bytes
                )));
            }

            let content =
                std::fs::read_to_string(&resolved).map_err(|e| match e.kind() {
                    std::io::ErrorKind::NotFound => FileOperationError::FileNotFound,
                    std::io::ErrorKind::PermissionDenied => FileOperationError::PermissionDenied,
                    _ => FileOperationError::InvalidPath,
                })?;

            let lines: Vec<&str> = content.lines().collect();
            let total = lines.len();
            let start = offset.min(total);
            let end = if limit == 0 {
                total
            } else {
                (start + limit).min(total)
            };

            let selected_content = lines[start..end].join("\n");
            Ok(format_content_with_line_numbers(
                &selected_content,
                start + 1,
                2000,
            ))
        }
    }

    /// Synchronous grep using literal string matching (ADR-103 C13).
    fn grep_sync(
        &self,
        pattern: &str,
        path: Option<&str>,
        include_glob: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String> {
        let search_root = if let Some(p) = path {
            self.resolve_path(p).map_err(|e| e.to_string())?
        } else {
            self.inner.cwd.clone()
        };

        let glob_pattern = include_glob.and_then(|g| glob::Pattern::new(g).ok());

        let mut matches = Vec::new();

        let walker = walkdir::WalkDir::new(&search_root)
            .follow_links(false) // ADR-103 C1 — never follow symlinks
            .into_iter()
            .filter_map(|e| e.ok());

        for entry in walker {
            if !entry.file_type().is_file() {
                continue;
            }

            let entry_path = entry.path();

            // Apply glob filter if provided
            if let Some(ref gp) = glob_pattern {
                let file_name = entry_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                if !gp.matches(file_name) {
                    continue;
                }
            }

            // Read and search the file (skip binary/unreadable)
            let content = match std::fs::read_to_string(entry_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let relative_path = entry_path
                .strip_prefix(&self.inner.cwd)
                .unwrap_or(entry_path);
            let path_str = relative_path.to_string_lossy().to_string();

            // Literal string matching (ADR-103 C13) — not regex
            for (line_idx, line) in content.lines().enumerate() {
                if line.contains(pattern) {
                    matches.push(GrepMatch {
                        path: path_str.clone(),
                        line: (line_idx + 1) as u32,
                        text: line.to_string(),
                    });
                }
            }
        }

        Ok(matches)
    }

    /// Synchronous glob_info using walkdir with follow_links(false).
    fn glob_info_sync(&self, pattern: &str, path: &str) -> Vec<FileInfo> {
        let search_root = match self.resolve_path(path) {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        let glob_pattern = match glob::Pattern::new(pattern) {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        let mut results = Vec::new();
        let walker = walkdir::WalkDir::new(&search_root)
            .follow_links(false) // ADR-103 C1
            .into_iter()
            .filter_map(|e| e.ok());

        for entry in walker {
            let entry_path = entry.path();
            let relative = entry_path
                .strip_prefix(&self.inner.cwd)
                .unwrap_or(entry_path);
            let path_str = relative.to_string_lossy().to_string();

            if glob_pattern.matches(&path_str) || glob_pattern.matches(
                entry_path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
            ) {
                let (size, modified_at) = entry
                    .metadata()
                    .map(|m| {
                        let size = m.len();
                        let modified = m
                            .modified()
                            .ok()
                            .and_then(|t| {
                                let dt: chrono::DateTime<chrono::Utc> = t.into();
                                Some(dt.to_rfc3339())
                            });
                        (size, modified)
                    })
                    .unwrap_or((0, None));

                results.push(FileInfo {
                    path: path_str,
                    is_dir: entry.file_type().is_dir(),
                    size,
                    modified_at,
                });
            }
        }

        results.sort_by(|a, b| a.path.cmp(&b.path));
        results
    }

    /// Synchronous ls_info.
    fn ls_info_sync(&self, path: &str) -> Vec<FileInfo> {
        let resolved = match self.resolve_path(path) {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        let entries = match std::fs::read_dir(&resolved) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };

        let mut results = Vec::new();
        for entry in entries.flatten() {
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            let path_str = entry
                .path()
                .strip_prefix(&self.inner.cwd)
                .unwrap_or(&entry.path())
                .to_string_lossy()
                .to_string();
            let modified_at = meta
                .modified()
                .ok()
                .map(|t| {
                    let dt: chrono::DateTime<chrono::Utc> = t.into();
                    dt.to_rfc3339()
                });

            results.push(FileInfo {
                path: path_str,
                is_dir: meta.is_dir(),
                size: meta.len(),
                modified_at,
            });
        }

        results.sort_by(|a, b| a.path.cmp(&b.path));
        results
    }

    /// Synchronous write_file.
    fn write_file_sync(&self, file_path: &str, content: &str) -> WriteResult {
        #[cfg(unix)]
        {
            // Use atomic resolve+open with TOCTOU protection
            use std::io::Write;

            let resolved = match self.resolve_path(file_path) {
                Ok(p) => p,
                Err(e) => {
                    return WriteResult {
                        error: Some(e.to_string()),
                        path: None,
                        files_update: None,
                    };
                }
            };

            // Create parent directories
            if let Some(parent) = resolved.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return WriteResult {
                        error: Some(format!("failed to create directories: {}", e)),
                        path: None,
                        files_update: None,
                    };
                }
            }

            match self.resolve_and_open(file_path, true) {
                Ok(mut file) => {
                    // Truncate file before writing
                    if let Err(e) = file.set_len(0) {
                        return WriteResult {
                            error: Some(format!("failed to truncate file: {}", e)),
                            path: None,
                            files_update: None,
                        };
                    }

                    match file.write_all(content.as_bytes()) {
                        Ok(_) => WriteResult {
                            error: None,
                            path: Some(file_path.to_string()),
                            files_update: None,
                        },
                        Err(e) => WriteResult {
                            error: Some(e.to_string()),
                            path: None,
                            files_update: None,
                        },
                    }
                }
                Err(e) => WriteResult {
                    error: Some(e.to_string()),
                    path: None,
                    files_update: None,
                },
            }
        }

        #[cfg(not(unix))]
        {
            // Fallback for non-Unix platforms
            let resolved = match self.resolve_path(file_path) {
                Ok(p) => p,
                Err(e) => {
                    return WriteResult {
                        error: Some(e.to_string()),
                        path: None,
                        files_update: None,
                    };
                }
            };

            // Create parent directories
            if let Some(parent) = resolved.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return WriteResult {
                        error: Some(format!("failed to create directories: {}", e)),
                        path: None,
                        files_update: None,
                    };
                }
            }

            match std::fs::write(&resolved, content) {
                Ok(_) => WriteResult {
                    error: None,
                    path: Some(file_path.to_string()),
                    files_update: None,
                },
                Err(e) => WriteResult {
                    error: Some(e.to_string()),
                    path: None,
                    files_update: None,
                },
            }
        }
    }

    /// Synchronous edit_file.
    fn edit_file_sync(
        &self,
        file_path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> EditResult {
        let resolved = match self.resolve_path(file_path) {
            Ok(p) => p,
            Err(e) => {
                return EditResult {
                    error: Some(e.to_string()),
                    path: None,
                    files_update: None,
                    occurrences: None,
                };
            }
        };

        let content = match std::fs::read_to_string(&resolved) {
            Ok(c) => c,
            Err(e) => {
                return EditResult {
                    error: Some(format!("failed to read file: {}", e)),
                    path: Some(file_path.to_string()),
                    files_update: None,
                    occurrences: None,
                };
            }
        };

        let count = content.matches(old_string).count() as u32;
        if count == 0 {
            return EditResult {
                error: Some(format!("old_string not found in {}", file_path)),
                path: Some(file_path.to_string()),
                files_update: None,
                occurrences: Some(0),
            };
        }

        if !replace_all && count > 1 {
            return EditResult {
                error: Some(format!(
                    "old_string found {} times — must be unique (or use replace_all)",
                    count
                )),
                path: Some(file_path.to_string()),
                files_update: None,
                occurrences: Some(count),
            };
        }

        let new_content = if replace_all {
            content.replace(old_string, new_string)
        } else {
            content.replacen(old_string, new_string, 1)
        };

        let replaced_count = if replace_all { count } else { 1 };

        match std::fs::write(&resolved, &new_content) {
            Ok(_) => EditResult {
                error: None,
                path: Some(file_path.to_string()),
                files_update: None,
                occurrences: Some(replaced_count),
            },
            Err(e) => EditResult {
                error: Some(format!("failed to write file: {}", e)),
                path: Some(file_path.to_string()),
                files_update: None,
                occurrences: Some(replaced_count),
            },
        }
    }
}

/// Lexical path normalization without filesystem access.
/// Resolves `.` and `..` components purely lexically.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for comp in path.components() {
        match comp {
            std::path::Component::ParentDir => {
                if !components.is_empty() {
                    components.pop();
                }
            }
            std::path::Component::CurDir => {}
            other => components.push(other),
        }
    }
    components.iter().collect()
}

#[async_trait]
impl Backend for FilesystemBackend {
    async fn ls_info(&self, path: &str) -> Vec<FileInfo> {
        let backend = self.clone();
        let path = path.to_string();
        tokio::task::spawn_blocking(move || backend.ls_info_sync(&path))
            .await
            .unwrap_or_default()
    }

    async fn read_file(
        &self,
        file_path: &str,
        offset: usize,
        limit: usize,
    ) -> Result<String, FileOperationError> {
        let backend = self.clone();
        let file_path = file_path.to_string();
        tokio::task::spawn_blocking(move || backend.read_file_sync(&file_path, offset, limit))
            .await
            .unwrap_or(Err(FileOperationError::InvalidPath))
    }

    async fn write_file(&self, file_path: &str, content: &str) -> WriteResult {
        let backend = self.clone();
        let file_path = file_path.to_string();
        let content = content.to_string();
        tokio::task::spawn_blocking(move || backend.write_file_sync(&file_path, &content))
            .await
            .unwrap_or_else(|e| WriteResult {
                error: Some(format!("spawn_blocking failed: {}", e)),
                path: None,
                files_update: None,
            })
    }

    async fn edit_file(
        &self,
        file_path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> EditResult {
        let backend = self.clone();
        let file_path = file_path.to_string();
        let old_string = old_string.to_string();
        let new_string = new_string.to_string();
        tokio::task::spawn_blocking(move || {
            backend.edit_file_sync(&file_path, &old_string, &new_string, replace_all)
        })
        .await
        .unwrap_or_else(|e| EditResult {
            error: Some(format!("spawn_blocking failed: {}", e)),
            path: None,
            files_update: None,
            occurrences: None,
        })
    }

    async fn glob_info(&self, pattern: &str, path: &str) -> Vec<FileInfo> {
        let backend = self.clone();
        let pattern = pattern.to_string();
        let path = path.to_string();
        tokio::task::spawn_blocking(move || backend.glob_info_sync(&pattern, &path))
            .await
            .unwrap_or_default()
    }

    async fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        include_glob: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String> {
        let backend = self.clone();
        let pattern = pattern.to_string();
        let path = path.map(|p| p.to_string());
        let include_glob = include_glob.map(|g| g.to_string());
        tokio::task::spawn_blocking(move || {
            backend.grep_sync(
                &pattern,
                path.as_deref(),
                include_glob.as_deref(),
            )
        })
        .await
        .unwrap_or_else(|e| Err(format!("spawn_blocking failed: {}", e)))
    }

    async fn download_files(&self, paths: &[String]) -> Vec<FileDownloadResponse> {
        let backend = self.clone();
        let paths = paths.to_vec();
        tokio::task::spawn_blocking(move || {
            paths
                .iter()
                .map(|p| {
                    let resolved = match backend.resolve_path(p) {
                        Ok(r) => r,
                        Err(e) => {
                            return FileDownloadResponse {
                                path: p.clone(),
                                content: None,
                                error: Some(e),
                            };
                        }
                    };
                    match std::fs::read(&resolved) {
                        Ok(content) => FileDownloadResponse {
                            path: p.clone(),
                            content: Some(content),
                            error: None,
                        },
                        Err(e) => FileDownloadResponse {
                            path: p.clone(),
                            content: None,
                            error: Some(match e.kind() {
                                std::io::ErrorKind::NotFound => FileOperationError::FileNotFound,
                                std::io::ErrorKind::PermissionDenied => {
                                    FileOperationError::PermissionDenied
                                }
                                _ => FileOperationError::InvalidPath,
                            }),
                        },
                    }
                })
                .collect()
        })
        .await
        .unwrap_or_default()
    }

    async fn upload_files(&self, files: &[(String, Vec<u8>)]) -> Vec<FileUploadResponse> {
        let backend = self.clone();
        let files = files.to_vec();
        tokio::task::spawn_blocking(move || {
            files
                .iter()
                .map(|(path, content)| {
                    let resolved = match backend.resolve_path(path) {
                        Ok(r) => r,
                        Err(e) => {
                            return FileUploadResponse {
                                path: path.clone(),
                                error: Some(e),
                            };
                        }
                    };
                    if let Some(parent) = resolved.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    match std::fs::write(&resolved, content) {
                        Ok(_) => FileUploadResponse {
                            path: path.clone(),
                            error: None,
                        },
                        Err(e) => FileUploadResponse {
                            path: path.clone(),
                            error: Some(match e.kind() {
                                std::io::ErrorKind::PermissionDenied => {
                                    FileOperationError::PermissionDenied
                                }
                                _ => FileOperationError::InvalidPath,
                            }),
                        },
                    }
                })
                .collect()
        })
        .await
        .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup() -> (TempDir, FilesystemBackend) {
        let tmp = TempDir::new().unwrap();
        let backend = FilesystemBackend::new(tmp.path().to_path_buf());
        (tmp, backend)
    }

    #[test]
    fn test_resolve_path_normal() {
        let (_tmp, backend) = setup();
        let resolved = backend.resolve_path("src/main.rs").unwrap();
        assert!(resolved.ends_with("src/main.rs"));
    }

    #[test]
    fn test_resolve_path_traversal_blocked() {
        let (_tmp, backend) = setup();
        let result = backend.resolve_path("../etc/passwd");
        assert!(result.is_err());
        match result.unwrap_err() {
            FileOperationError::SecurityViolation(msg) => {
                assert!(msg.contains("traversal"));
            }
            other => panic!("Expected SecurityViolation, got {:?}", other),
        }
    }

    #[test]
    fn test_resolve_path_null_byte_blocked() {
        let (_tmp, backend) = setup();
        let result = backend.resolve_path("file\0.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_path_tilde_blocked() {
        let (_tmp, backend) = setup();
        let result = backend.resolve_path("~/.ssh/id_rsa");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_path_absolute_in_virtual_mode() {
        let (_tmp, backend) = setup();
        // In virtual mode, absolute paths have leading '/' stripped
        let resolved = backend.resolve_path("/foo/bar.txt").unwrap();
        assert!(resolved.ends_with("foo/bar.txt"));
        assert!(resolved.starts_with(backend.cwd()));
    }

    #[test]
    fn test_resolve_path_empty() {
        let (_tmp, backend) = setup();
        let resolved = backend.resolve_path("").unwrap();
        assert_eq!(resolved, backend.cwd());
    }

    #[test]
    fn test_resolve_path_double_dot_in_middle() {
        let (_tmp, backend) = setup();
        let result = backend.resolve_path("foo/../../../etc/passwd");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_write_and_read_file() {
        let (_tmp, backend) = setup();
        let write_result = backend.write_file("test.txt", "hello\nworld").await;
        assert!(write_result.error.is_none());

        let content = backend.read_file("test.txt", 0, 0).await.unwrap();
        assert!(content.contains("hello"));
        assert!(content.contains("world"));
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let (_tmp, backend) = setup();
        let result = backend.read_file("nonexistent.txt", 0, 0).await;
        assert_eq!(result.unwrap_err(), FileOperationError::FileNotFound);
    }

    #[tokio::test]
    async fn test_read_directory_returns_error() {
        let (tmp, backend) = setup();
        std::fs::create_dir(tmp.path().join("subdir")).unwrap();
        let result = backend.read_file("subdir", 0, 0).await;
        assert_eq!(result.unwrap_err(), FileOperationError::IsDirectory);
    }

    #[tokio::test]
    async fn test_edit_file() {
        let (_tmp, backend) = setup();
        backend.write_file("test.txt", "hello world").await;
        let result = backend
            .edit_file("test.txt", "hello", "goodbye", false)
            .await;
        assert!(result.error.is_none());

        let content = backend.read_file("test.txt", 0, 0).await.unwrap();
        assert!(content.contains("goodbye"));
        assert!(!content.contains("hello"));
    }

    #[tokio::test]
    async fn test_grep_literal() {
        let (_tmp, backend) = setup();
        backend
            .write_file("test.rs", "fn main() {}\nlet x = 42;\nfn helper() {}")
            .await;

        // Literal matching, not regex
        let results = backend.grep("fn ", None, None).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_grep_regex_chars_are_literal() {
        let (_tmp, backend) = setup();
        backend
            .write_file("test.txt", "hello (world)\nhello world")
            .await;

        // "(world)" should be treated literally, not as regex
        let results = backend.grep("(world)", None, None).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_ls_info() {
        let (_tmp, backend) = setup();
        backend.write_file("a.txt", "aaa").await;
        backend.write_file("b.txt", "bbb").await;
        let items = backend.ls_info("").await;
        assert_eq!(items.len(), 2);
    }

    #[tokio::test]
    async fn test_upload_download() {
        let (_tmp, backend) = setup();
        let uploads = backend
            .upload_files(&[("doc.bin".to_string(), vec![0xDE, 0xAD])])
            .await;
        assert!(uploads[0].error.is_none());

        let downloads = backend.download_files(&["doc.bin".to_string()]).await;
        assert_eq!(downloads[0].content.as_ref().unwrap(), &[0xDE, 0xAD]);
    }

    #[test]
    fn test_lexical_normalize() {
        let p = PathBuf::from("/a/b/../c/./d");
        assert_eq!(lexical_normalize(&p), PathBuf::from("/a/c/d"));
    }

    #[test]
    fn test_virtual_mode_default_true() {
        let backend = FilesystemBackend::new(PathBuf::from("/tmp"));
        assert!(backend.virtual_mode());
    }
}
