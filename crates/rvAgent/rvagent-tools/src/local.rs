//! Local filesystem backend — the real `Backend` used by the CLI and by the
//! end-to-end harness tests.
//!
//! # Path confinement
//!
//! Every path a tool supplies is model-controlled input, so it is resolved
//! against a fixed root and rejected if it escapes. Resolution is symlink-aware:
//! the deepest existing ancestor is canonicalized before the containment check,
//! so a symlink pointing outside the root cannot be used as a bridge. Without
//! this, `read_file {"path": "/etc/passwd"}` — or a `write_file` anywhere on
//! disk — is a single tool call away.

use std::path::{Component, Path, PathBuf};

use crate::{Backend, ExecuteResponse, FileInfo, GrepMatch, WriteResult};

/// A filesystem backend rooted at, and confined to, a working directory.
pub struct LocalFsBackend {
    /// Canonical root. All resolved paths must live under this.
    root: PathBuf,
}

impl LocalFsBackend {
    /// Create a backend confined to `root`.
    ///
    /// The root is canonicalized so that containment checks compare real paths.
    /// If it cannot be canonicalized (e.g. it does not exist yet) the path is
    /// used as given — resolution still applies, it is simply not symlink-proof
    /// above the root itself.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let root = std::fs::canonicalize(&root).unwrap_or(root);
        Self { root }
    }

    /// The confinement root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a tool-supplied path to a real path inside the root.
    ///
    /// Relative paths are joined to the root; absolute paths must already be
    /// inside it. `..` is normalized lexically first so it cannot be used to
    /// climb out, then the deepest existing ancestor is canonicalized to defeat
    /// symlink escapes.
    fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let raw = Path::new(path);
        let joined = if raw.is_absolute() {
            raw.to_path_buf()
        } else if path.is_empty() || path == "." {
            self.root.clone()
        } else {
            self.root.join(raw)
        };

        let normalized = lexical_normalize(&joined);
        let resolved = canonicalize_existing_prefix(&normalized);

        if resolved.starts_with(&self.root) {
            Ok(resolved)
        } else {
            Err(format!(
                "Error: path '{path}' resolves outside the workspace root"
            ))
        }
    }

    /// Resolve for write-style operations, returning the error as a
    /// `WriteResult` rather than a bare string.
    fn resolve_for_write(&self, path: &str) -> Result<PathBuf, WriteResult> {
        self.resolve(path).map_err(|e| WriteResult {
            error: Some(e),
            ..Default::default()
        })
    }
}

/// Remove `.` and resolve `..` lexically, without touching the filesystem.
///
/// Purely lexical so it works for paths that do not exist yet (`write_file` to
/// a new file). `..` at or above the root simply cannot climb past the prefix.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => out.push(Component::RootDir.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                // Never pop past the root component itself.
                if out.parent().is_some() {
                    out.pop();
                }
            }
            Component::Normal(seg) => out.push(seg),
        }
    }
    out
}

/// Canonicalize the deepest existing ancestor of `path`, re-appending the
/// non-existent tail.
///
/// This is what makes the containment check symlink-aware: if any existing
/// component is a symlink out of the root, canonicalization exposes it before
/// the `starts_with` test. `path` must already be lexically normalized, so
/// re-appending the tail cannot reintroduce `..`.
fn canonicalize_existing_prefix(path: &Path) -> PathBuf {
    let mut existing = path;
    let mut tail: Vec<&std::ffi::OsStr> = Vec::new();

    loop {
        if let Ok(canonical) = std::fs::canonicalize(existing) {
            let mut out = canonical;
            for seg in tail.iter().rev() {
                out.push(seg);
            }
            return out;
        }
        match (existing.file_name(), existing.parent()) {
            (Some(name), Some(parent)) => {
                tail.push(name);
                existing = parent;
            }
            // Nothing along the chain exists; fall back to the lexical path.
            _ => return path.to_path_buf(),
        }
    }
}

impl Backend for LocalFsBackend {
    fn ls_info(&self, path: &str) -> Result<Vec<FileInfo>, String> {
        let target = self.resolve(path)?;
        let entries = std::fs::read_dir(&target)
            .map_err(|e| format!("ls failed on '{}': {}", target.display(), e))?;
        let mut infos = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| format!("read_dir entry error: {}", e))?;
            let meta = entry
                .metadata()
                .map_err(|e| format!("metadata error: {}", e))?;
            let file_type = if meta.is_dir() {
                "directory"
            } else if meta.is_symlink() {
                "symlink"
            } else {
                "file"
            };
            infos.push(FileInfo {
                name: entry.file_name().to_string_lossy().into_owned(),
                file_type: file_type.to_string(),
                permissions: String::new(),
                size: meta.len(),
            });
        }
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(infos)
    }

    fn read(&self, path: &str, offset: usize, limit: usize) -> Result<String, String> {
        let target = self.resolve(path)?;
        let content = std::fs::read_to_string(&target)
            .map_err(|e| format!("read '{}': {}", target.display(), e))?;
        let lines: Vec<&str> = content.lines().collect();
        if offset >= lines.len() {
            return Ok(String::new());
        }
        let end = (offset + limit).min(lines.len());
        Ok(lines[offset..end].join("\n"))
    }

    fn write(&self, path: &str, content: &str) -> WriteResult {
        let target = match self.resolve_for_write(path) {
            Ok(t) => t,
            Err(e) => return e,
        };
        if target.exists() {
            return WriteResult {
                error: Some(format!(
                    "Error: file {} already exists. Use force flag to overwrite.",
                    target.display()
                )),
                ..Default::default()
            };
        }
        if let Some(parent) = target.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return WriteResult {
                    error: Some(format!("mkdir failed: {}", e)),
                    ..Default::default()
                };
            }
        }
        if let Err(e) = std::fs::write(&target, content) {
            return WriteResult {
                error: Some(format!("write '{}': {}", target.display(), e)),
                ..Default::default()
            };
        }
        match verify_written(&target, content) {
            Ok(()) => WriteResult::default(),
            Err(e) => WriteResult {
                error: Some(e),
                ..Default::default()
            },
        }
    }

    fn edit(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> WriteResult {
        let target = match self.resolve_for_write(path) {
            Ok(t) => t,
            Err(e) => return e,
        };
        let content = match std::fs::read_to_string(&target) {
            Ok(c) => c,
            Err(e) => {
                return WriteResult {
                    error: Some(format!("read '{}': {}", target.display(), e)),
                    ..Default::default()
                }
            }
        };
        let count = content.matches(old_string).count();
        if count == 0 {
            // Say why it failed, not just that it did (ADR-273 §3.1/§3.3).
            return WriteResult {
                error: Some(crate::diagnose_edit_failure(
                    &content,
                    old_string,
                    &target.display().to_string(),
                )),
                ..Default::default()
            };
        }
        if count > 1 && !replace_all {
            return WriteResult {
                error: Some(format!(
                    "Error: old_string is not unique in {} ({} occurrences). Use replace_all=true.",
                    target.display(),
                    count
                )),
                ..Default::default()
            };
        }
        let new_content = if replace_all {
            content.replace(old_string, new_string)
        } else {
            content.replacen(old_string, new_string, 1)
        };
        if let Err(e) = std::fs::write(&target, &new_content) {
            return WriteResult {
                error: Some(format!("write '{}': {}", target.display(), e)),
                ..Default::default()
            };
        }
        match verify_written(&target, &new_content) {
            Ok(()) => WriteResult {
                error: None,
                occurrences: Some(if replace_all { count } else { 1 }),
                ..Default::default()
            },
            Err(e) => WriteResult {
                error: Some(e),
                ..Default::default()
            },
        }
    }

    fn glob_info(&self, pattern: &str, path: &str) -> Result<Vec<String>, String> {
        let base = self.resolve(path)?;
        // Simple glob: walk the directory and match by name suffix. Handles the
        // common `*.rs` / `**/*.toml` shapes without pulling in a glob crate.
        let suffix = pattern
            .trim_start_matches('*')
            .trim_start_matches('/')
            .trim_start_matches('*');
        let mut results = Vec::new();
        collect_glob_matches(&base, suffix, &mut results);
        results.sort();
        Ok(results)
    }

    fn grep_raw(
        &self,
        pattern: &str,
        path: Option<&str>,
        _include: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String> {
        let search_dir = self.resolve(path.unwrap_or("."))?;
        let mut matches = Vec::new();
        if search_dir.is_file() {
            grep_file(&search_dir, pattern, &mut matches)?;
        } else if search_dir.is_dir() {
            grep_dir(&search_dir, pattern, &mut matches)?;
        }
        Ok(matches)
    }

    fn execute(&self, command: &str, timeout_secs: u32) -> Result<ExecuteResponse, String> {
        use std::process::{Command, Stdio};
        use std::time::Duration;

        // Security: environment sanitization — strip sensitive variables
        // (SEC-005 / ADR-103 C2). Only a safe allowlist reaches the child.
        const SAFE_ENV_VARS: &[&str] = &[
            "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "TZ",
        ];
        // Patterns identifying vars that must never reach child processes.
        const SENSITIVE_PATTERNS: &[&str] = &[
            "SECRET",
            "KEY",
            "TOKEN",
            "PASSWORD",
            "CREDENTIAL",
            "AWS_",
            "AZURE_",
            "GCP_",
            "DATABASE_URL",
            "PRIVATE",
            "API_KEY",
            "AUTH",
            "BEARER",
            "JWT",
            "SESSION",
        ];

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command).current_dir(&self.root);
        cmd.env_clear();
        for var in SAFE_ENV_VARS {
            if let Ok(val) = std::env::var(var) {
                let upper = var.to_uppercase();
                let sensitive = SENSITIVE_PATTERNS.iter().any(|pat| upper.contains(pat));
                if !sensitive {
                    cmd.env(var, val);
                }
            }
        }
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let timeout = if timeout_secs == 0 { 30 } else { timeout_secs };
        let deadline = std::time::Instant::now() + Duration::from_secs(timeout as u64);

        let mut child = cmd.spawn().map_err(|e| format!("execute failed: {}", e))?;

        // Poll for completion with a deadline to enforce the timeout.
        loop {
            match child
                .try_wait()
                .map_err(|e| format!("wait failed: {}", e))?
            {
                Some(_) => break,
                None => {
                    if std::time::Instant::now() >= deadline {
                        let _ = child.kill();
                        return Ok(ExecuteResponse {
                            output: format!("Command timed out after {} seconds", timeout),
                            exit_code: -1,
                        });
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }

        let output = child
            .wait_with_output()
            .map_err(|e| format!("output collection failed: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut combined = if stderr.is_empty() {
            stdout.into_owned()
        } else {
            format!("{}\n{}", stdout, stderr)
        };

        // Security: cap output size to prevent memory exhaustion.
        const MAX_OUTPUT_BYTES: usize = 1024 * 1024;
        if combined.len() > MAX_OUTPUT_BYTES {
            combined.truncate(MAX_OUTPUT_BYTES);
            combined.push_str("\n... [output truncated at 1 MB]");
        }

        Ok(ExecuteResponse {
            output: combined,
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
}

/// Confirm a write actually landed, by reading the file back (ADR-273 §3.1).
///
/// `std::fs::write` returning `Ok` means the syscalls succeeded, not that the
/// bytes are on disk and readable: a full filesystem, a quota, a racing writer,
/// or an unusual mount can all produce a successful-looking write whose content
/// differs. Reporting success in that case is the worst outcome, because the
/// agent proceeds believing the edit is applied and every later step is built
/// on a false premise — the exact failure mode that dominates harness ablations.
///
/// Costs one read per write, which is negligible against a model round trip.
fn verify_written(target: &Path, expected: &str) -> Result<(), String> {
    match std::fs::read(target) {
        Ok(actual) if actual == expected.as_bytes() => Ok(()),
        Ok(actual) => Err(format!(
            "Error: write to '{}' did not verify — expected {} bytes, file now holds {}. \
             The file may have been modified concurrently or the write was truncated. \
             Re-read the file before making further changes.",
            target.display(),
            expected.len(),
            actual.len()
        )),
        Err(e) => Err(format!(
            "Error: write to '{}' could not be verified: {e}. \
             Treat the file's contents as unknown and re-read it.",
            target.display()
        )),
    }
}

/// Recursively collect files matching a name suffix (simple glob substitute).
fn collect_glob_matches(dir: &Path, suffix: &str, results: &mut Vec<String>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if path.is_file() && name.ends_with(suffix) {
            results.push(path.to_string_lossy().into_owned());
        } else if path.is_dir() && !name.starts_with('.') {
            collect_glob_matches(&path, suffix, results);
        }
    }
}

/// Grep a single file for a pattern.
fn grep_file(path: &Path, pattern: &str, matches: &mut Vec<GrepMatch>) -> Result<(), String> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Ok(()), // skip binary / unreadable files
    };
    for (i, line) in content.lines().enumerate() {
        if line.contains(pattern) {
            matches.push(GrepMatch {
                file: path.to_string_lossy().into_owned(),
                line_number: i + 1,
                text: line.to_string(),
            });
        }
    }
    Ok(())
}

/// Recursively grep a directory, skipping hidden directories.
fn grep_dir(dir: &Path, pattern: &str, matches: &mut Vec<GrepMatch>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read_dir: {}", e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("entry: {}", e))?;
        let path = entry.path();
        if path.is_file() {
            grep_file(&path, pattern, matches)?;
        } else if path.is_dir() {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if !name.starts_with('.') {
                grep_dir(&path, pattern, matches)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> (tempfile::TempDir, LocalFsBackend) {
        let dir = tempfile::tempdir().unwrap();
        let backend = LocalFsBackend::new(dir.path());
        (dir, backend)
    }

    #[test]
    fn relative_paths_resolve_inside_root() {
        let (dir, backend) = backend();
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        assert_eq!(backend.read("a.txt", 0, 10).unwrap(), "hello");
    }

    #[test]
    fn absolute_path_outside_root_is_rejected() {
        let (_dir, backend) = backend();
        let err = backend.read("/etc/passwd", 0, 10).unwrap_err();
        assert!(err.contains("outside the workspace root"), "got: {err}");
    }

    #[test]
    fn parent_traversal_is_rejected() {
        let (_dir, backend) = backend();
        let err = backend.read("../../../etc/passwd", 0, 10).unwrap_err();
        assert!(err.contains("outside the workspace root"), "got: {err}");
    }

    #[test]
    fn write_outside_root_is_rejected() {
        let (_dir, backend) = backend();
        let result = backend.write("/tmp/rvagent-escape-probe.txt", "pwned");
        assert!(result
            .error
            .as_deref()
            .is_some_and(|e| e.contains("outside the workspace root")));
        assert!(
            !Path::new("/tmp/rvagent-escape-probe.txt").exists(),
            "escaping write must not touch the filesystem"
        );
    }

    #[test]
    fn edit_outside_root_is_rejected() {
        let (_dir, backend) = backend();
        let result = backend.edit("/etc/hosts", "a", "b", false);
        assert!(result
            .error
            .as_deref()
            .is_some_and(|e| e.contains("outside the workspace root")));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_is_rejected() {
        let (dir, backend) = backend();
        // A symlink inside the root pointing out of it must not be a bridge.
        std::os::unix::fs::symlink("/etc", dir.path().join("escape")).unwrap();
        let err = backend.read("escape/passwd", 0, 10).unwrap_err();
        assert!(err.contains("outside the workspace root"), "got: {err}");
    }

    #[test]
    fn write_then_read_roundtrip_inside_root() {
        let (_dir, backend) = backend();
        let result = backend.write("nested/dir/new.txt", "content");
        assert!(result.error.is_none(), "unexpected: {:?}", result.error);
        assert_eq!(backend.read("nested/dir/new.txt", 0, 10).unwrap(), "content");
    }

    #[test]
    fn write_is_verified_by_reading_back() {
        let (dir, backend) = backend();
        let result = backend.write("verified.txt", "exact contents");
        assert!(result.error.is_none());
        // The verification path must accept a correct write, not just reject
        // bad ones — otherwise it would be a permanent false alarm.
        assert_eq!(
            std::fs::read_to_string(dir.path().join("verified.txt")).unwrap(),
            "exact contents"
        );
    }

    #[test]
    fn edit_is_verified_by_reading_back() {
        let (dir, backend) = backend();
        std::fs::write(dir.path().join("e.txt"), "alpha beta").unwrap();
        let result = backend.edit("e.txt", "alpha", "gamma", false);
        assert!(result.error.is_none(), "unexpected: {:?}", result.error);
        assert_eq!(result.occurrences, Some(1));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("e.txt")).unwrap(),
            "gamma beta"
        );
    }

    #[test]
    fn verification_reports_a_content_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.txt");
        std::fs::write(&path, "actual").unwrap();
        // Simulate the case the check exists for: what landed differs from
        // what was asked for.
        let err = verify_written(&path, "expected something longer").unwrap_err();
        assert!(err.contains("did not verify"), "got: {err}");
        assert!(err.contains("Re-read the file"), "must be actionable: {err}");
    }

    #[test]
    fn verification_reports_an_unreadable_file() {
        let dir = tempfile::tempdir().unwrap();
        let err = verify_written(&dir.path().join("missing.txt"), "anything").unwrap_err();
        assert!(err.contains("could not be verified"), "got: {err}");
        assert!(err.contains("re-read"), "must be actionable: {err}");
    }

    #[test]
    fn verification_handles_empty_and_multibyte_content() {
        let (_dir, backend) = backend();
        assert!(backend.write("empty.txt", "").error.is_none());
        assert!(backend.write("utf8.txt", "héllo 🙂 wörld").error.is_none());
    }

    #[test]
    fn dotdot_inside_root_still_works() {
        let (_dir, backend) = backend();
        assert!(backend.write("sub/file.txt", "x").error.is_none());
        // Climbs out of `sub` but stays under the root — legitimate.
        assert_eq!(backend.read("sub/../sub/file.txt", 0, 10).unwrap(), "x");
    }
}
