//! Local filesystem backend — the real `Backend` used by the CLI and by the
//! end-to-end harness tests.
//!
//! # Path confinement
//!
//! Every path a tool supplies is model-controlled input, so it is resolved
//! against a fixed root and rejected if it escapes. Without this,
//! `read_file {"path": "/etc/passwd"}` — or a `write_file` anywhere on disk —
//! is a single tool call away.
//!
//! What confinement guarantees:
//!
//! - **Lexical escapes** (`..`, absolute paths) are normalized and then
//!   rejected by a containment check against the canonical root.
//! - **Symlinks** cannot bridge out. The deepest existing ancestor is
//!   canonicalized before the containment check, a dangling link is refused
//!   outright rather than treated as a not-yet-existing path, and writes
//!   refuse to travel through a link at all — including one pointing back
//!   inside the root, which would otherwise pass containment.
//! - **Hard links** cannot bridge out on unix: a write to a file with
//!   `nlink > 1` is refused, because the same inode may have another name
//!   outside the root and nothing in the path tells us it does not.
//!
//! What it does not guarantee:
//!
//! - **TOCTOU races.** Resolution and the write are separate syscalls, so a
//!   concurrent process that swaps a path component between them can still
//!   redirect a write. Defending that needs `openat`-based traversal with
//!   `O_NOFOLLOW`, which is out of scope here; the threat model is a
//!   misbehaving model, not a local attacker racing the agent.
//! - **Non-unix hard links.** The `nlink` check is unix-only.
//! - **Anything below the root.** Confinement is a boundary, not a
//!   permission system: every file under the root is writable.

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
        let normalized = self.literal_path(path);
        let resolved = canonicalize_existing_prefix(&normalized)
            .map_err(|e| format!("Error: path '{path}' is not resolvable: {e}"))?;

        if resolved.starts_with(&self.root) {
            Ok(resolved)
        } else {
            Err(format!(
                "Error: path '{path}' resolves outside the workspace root"
            ))
        }
    }

    /// The lexically normalized path, before any symlink resolution.
    ///
    /// This is what the caller literally named. `resolve` canonicalizes it for
    /// the containment check, which erases the fact that a component *was* a
    /// symlink — so checks that care about that must run against this.
    fn literal_path(&self, path: &str) -> PathBuf {
        let raw = Path::new(path);
        let joined = if raw.is_absolute() {
            raw.to_path_buf()
        } else if path.is_empty() || path == "." {
            self.root.clone()
        } else {
            self.root.join(raw)
        };
        lexical_normalize(&joined)
    }

    /// Resolve for write-style operations, returning the error as a
    /// `WriteResult` rather than a bare string.
    ///
    /// Beyond containment, a write must never travel *through* a link.
    ///
    /// `std::fs::write` follows a symlink, so a link inside the root is a
    /// write primitive for wherever it points — and `resolve` cannot catch
    /// it, because canonicalizing a link to an in-root file yields an in-root
    /// path. The link is checked pre-resolution instead.
    ///
    /// A hard link is invisible to both checks: there is no link to follow and
    /// `symlink_metadata` reports an ordinary file. The only signal is the
    /// inode's link count, so a file with more than one name is refused —
    /// another of those names may well be outside the root.
    fn resolve_for_write(&self, path: &str) -> Result<PathBuf, WriteResult> {
        let fail = |e: String| WriteResult {
            error: Some(e),
            ..Default::default()
        };
        let resolved = self.resolve(path).map_err(fail)?;
        let literal = self.literal_path(path);
        if let Ok(meta) = std::fs::symlink_metadata(&literal) {
            if meta.file_type().is_symlink() {
                return Err(fail(format!(
                    "Error: refusing to write through symlink '{}'. \
                     Write to the link's target directly.",
                    literal.display()
                )));
            }
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if let Ok(meta) = std::fs::metadata(&resolved) {
                // Directories always have several names (`.`, `..`, each
                // child), so the count only means anything for a file.
                if meta.is_file() && meta.nlink() > 1 {
                    return Err(fail(format!(
                        "Error: refusing to write through a hard link (nlink > 1): '{}'. \
                         The file has another name, possibly outside the workspace, \
                         which this write would also change.",
                        resolved.display()
                    )));
                }
            }
        }
        Ok(resolved)
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
///
/// A *dangling* symlink is rejected outright. It cannot be canonicalized, so
/// treating it as part of the non-existent tail would leave it looking like a
/// fresh path under the canonical root — while an actual write through it
/// lands wherever the link points, anywhere on disk.
fn canonicalize_existing_prefix(path: &Path) -> Result<PathBuf, String> {
    let mut existing = path;
    let mut tail: Vec<&std::ffi::OsStr> = Vec::new();

    loop {
        if let Ok(meta) = std::fs::symlink_metadata(existing) {
            if meta.file_type().is_symlink() && std::fs::canonicalize(existing).is_err() {
                return Err(format!(
                    "'{}' is a broken symlink; refusing to resolve through it",
                    existing.display()
                ));
            }
        }
        if let Ok(canonical) = std::fs::canonicalize(existing) {
            let mut out = canonical;
            for seg in tail.iter().rev() {
                out.push(seg);
            }
            return Ok(out);
        }
        match (existing.file_name(), existing.parent()) {
            (Some(name), Some(parent)) => {
                tail.push(name);
                existing = parent;
            }
            // Nothing along the chain exists; fall back to the lexical path.
            _ => return Ok(path.to_path_buf()),
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
        // `limit` is model-supplied and arrives as an unbounded integer, so a
        // plain `offset + limit` overflows and panics on values like u64::MAX.
        // Saturating means "the rest of the file", which is what such a limit
        // asks for anyway.
        let end = offset.saturating_add(limit).min(lines.len());
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
        // (SEC-005 / ADR-103 C2). The allowlist *is* the control: nothing
        // outside it reaches the child, so no name-pattern denylist is needed.
        const SAFE_ENV_VARS: &[&str] = &[
            "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "TZ",
        ];

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command).current_dir(&self.root);
        cmd.env_clear();
        for var in SAFE_ENV_VARS {
            if let Ok(val) = std::env::var(var) {
                cmd.env(var, val);
            }
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Put the child in its own process group so a timeout can kill the
        // whole tree. `child.kill()` signals only `sh`, leaving any grandchild
        // running — and holding the pipe open — after we have given up on it.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        let timeout = if timeout_secs == 0 { 30 } else { timeout_secs };
        let deadline = std::time::Instant::now() + Duration::from_secs(timeout as u64);

        let mut child = cmd.spawn().map_err(|e| format!("execute failed: {}", e))?;

        // Drain both pipes concurrently. A pipe holds only ~64 KB; if nobody
        // reads while we poll, a chatty command blocks on write until the
        // deadline and is then reported as a timeout it never actually hit.
        //
        // Each stream keeps one byte past the cap so that an oversized stream
        // still trips the `> cap` test below and gets the truncation marker,
        // rather than arriving as a silently shortened but cap-sized string.
        let keep = MAX_OUTPUT_BYTES.saturating_add(1);
        let out_pipe = child.stdout.take();
        let err_pipe = child.stderr.take();
        let out_reader = std::thread::spawn(move || match out_pipe {
            Some(p) => drain_capped(p, keep),
            None => Vec::new(),
        });
        let err_reader = std::thread::spawn(move || match err_pipe {
            Some(p) => drain_capped(p, keep),
            None => Vec::new(),
        });

        let mut timed_out = false;
        let status = loop {
            match child
                .try_wait()
                .map_err(|e| format!("wait failed: {}", e))?
            {
                Some(status) => break Some(status),
                None => {
                    if std::time::Instant::now() >= deadline {
                        kill_process_tree(&mut child);
                        // Reap, or the child lingers as a zombie for the life
                        // of the process.
                        let _ = child.wait();
                        timed_out = true;
                        break None;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        };

        // The pipes are closed once the tree is gone, so these always finish.
        let stdout_bytes = out_reader.join().unwrap_or_default();
        let stderr_bytes = err_reader.join().unwrap_or_default();

        if timed_out {
            return Ok(ExecuteResponse {
                output: format!("Command timed out after {} seconds", timeout),
                exit_code: -1,
            });
        }

        let stdout = String::from_utf8_lossy(&stdout_bytes);
        let stderr = String::from_utf8_lossy(&stderr_bytes);
        let combined = if stderr.is_empty() {
            stdout.into_owned()
        } else {
            format!("{}\n{}", stdout, stderr)
        };

        Ok(ExecuteResponse {
            output: truncate_output(combined, MAX_OUTPUT_BYTES),
            exit_code: status.and_then(|s| s.code()).unwrap_or(-1),
        })
    }
}

/// Security: cap on captured command output, to prevent memory exhaustion.
const MAX_OUTPUT_BYTES: usize = 1024 * 1024;

/// Read a pipe to EOF, keeping at most `cap` bytes.
///
/// Reading past the cap and discarding is deliberate: stopping early would
/// leave the writer blocked on a full pipe, which is the deadlock this exists
/// to avoid.
fn drain_capped<R: std::io::Read>(mut reader: R, cap: usize) -> Vec<u8> {
    let mut kept = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                if kept.len() < cap {
                    let take = n.min(cap - kept.len());
                    kept.extend_from_slice(&buf[..take]);
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    kept
}

/// Kill the child and everything it spawned.
#[cfg(unix)]
fn kill_process_tree(child: &mut std::process::Child) {
    // The child leads its own group (see `process_group(0)`), so its pid is
    // the pgid and a negative pid signals the whole group.
    let pid = child.id() as i32;
    unsafe {
        libc::kill(-pid, libc::SIGKILL);
    }
    let _ = child.kill();
}

/// Kill the child. Without process groups only the direct child is reachable.
#[cfg(not(unix))]
fn kill_process_tree(child: &mut std::process::Child) {
    let _ = child.kill();
}

/// Cap `s` at `cap` bytes, cutting on a character boundary.
///
/// `String::truncate` panics when the cut lands inside a multi-byte character,
/// which is reachable whenever a command's output happens to straddle the cap.
fn truncate_output(mut s: String, cap: usize) -> String {
    if s.len() <= cap {
        return s;
    }
    let mut end = cap;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    s.push_str(&format!("\n... [output truncated at {} bytes]", cap));
    s
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
///
/// Symlinks are skipped rather than followed. `Path::is_dir` follows them, and
/// a single `ln -s . loop` inside the tree then recurses until the stack
/// overflows — an unauthenticated crash from an ordinary directory listing.
fn collect_glob_matches(dir: &Path, suffix: &str, results: &mut Vec<String>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        // `DirEntry::file_type` does not follow symlinks, so a link is neither
        // a file nor a directory here and falls through untouched.
        let file_type = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        let path = entry.path();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if file_type.is_file() && name.ends_with(suffix) {
            results.push(path.to_string_lossy().into_owned());
        } else if file_type.is_dir() && !name.starts_with('.') {
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

/// Recursively grep a directory, skipping hidden directories and symlinks.
///
/// Symlinks are skipped for the same reason as in `collect_glob_matches`: a
/// self-referencing link makes the descent unbounded.
fn grep_dir(dir: &Path, pattern: &str, matches: &mut Vec<GrepMatch>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read_dir: {}", e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("entry: {}", e))?;
        let file_type = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        let path = entry.path();
        if file_type.is_file() {
            grep_file(&path, pattern, matches)?;
        } else if file_type.is_dir() {
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

    #[cfg(unix)]
    #[test]
    fn write_to_a_dangling_symlink_is_rejected() {
        let (dir, backend) = backend();
        let outside = tempfile::tempdir().unwrap();
        let escaped = outside.path().join("escaped.txt");
        // The link target does not exist, so `canonicalize` fails and the path
        // looks like a fresh file under the root — but `fs::write` would
        // follow the link and create the file out there.
        std::os::unix::fs::symlink(&escaped, dir.path().join("link")).unwrap();

        let result = backend.write("link", "pwned");
        assert!(
            result.error.is_some(),
            "write through a broken link must fail"
        );
        assert!(
            !escaped.exists(),
            "write must not have escaped to {}",
            escaped.display()
        );
    }

    #[cfg(unix)]
    #[test]
    fn write_through_a_dangling_symlink_directory_is_rejected() {
        let (dir, backend) = backend();
        let outside = tempfile::tempdir().unwrap();
        let escaped_dir = outside.path().join("nope");
        std::os::unix::fs::symlink(&escaped_dir, dir.path().join("linkdir")).unwrap();

        let result = backend.write("linkdir/f.txt", "pwned");
        assert!(
            result.error.is_some(),
            "write through a broken link must fail"
        );
        assert!(
            !escaped_dir.exists(),
            "write must not have created the target"
        );
    }

    #[cfg(unix)]
    #[test]
    fn write_through_a_symlink_to_an_inside_file_is_rejected() {
        let (dir, backend) = backend();
        std::fs::write(dir.path().join("real.txt"), "original").unwrap();
        std::os::unix::fs::symlink(dir.path().join("real.txt"), dir.path().join("alias")).unwrap();

        let result = backend.write("alias", "replaced");
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|e| e.contains("symlink")),
            "got: {:?}",
            result.error
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("real.txt")).unwrap(),
            "original"
        );
    }

    #[cfg(unix)]
    #[test]
    fn write_or_edit_through_a_hard_link_is_rejected() {
        let (dir, backend) = backend();
        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("secret.txt");
        std::fs::write(&outside_file, "original").unwrap();
        // A hard link has no target to canonicalize and looks like an ordinary
        // file to `symlink_metadata`, so only the link count gives it away.
        // Git cannot store one, but a prepared workspace can plant one.
        std::fs::hard_link(&outside_file, dir.path().join("alias.txt")).unwrap();

        let written = backend.write("alias.txt", "pwned");
        assert!(
            written
                .error
                .as_deref()
                .is_some_and(|e| e.contains("hard link")),
            "got: {:?}",
            written.error
        );

        let edited = backend.edit("alias.txt", "original", "pwned", false);
        assert!(
            edited
                .error
                .as_deref()
                .is_some_and(|e| e.contains("hard link")),
            "got: {:?}",
            edited.error
        );

        assert_eq!(
            std::fs::read_to_string(&outside_file).unwrap(),
            "original",
            "the file outside the root must be untouched"
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_ordinary_file_is_not_mistaken_for_a_hard_link() {
        // The nlink check must not fire on the normal case, or every edit in
        // the workspace breaks.
        let (dir, backend) = backend();
        std::fs::write(dir.path().join("plain.txt"), "alpha").unwrap();
        let edited = backend.edit("plain.txt", "alpha", "beta", false);
        assert!(edited.error.is_none(), "unexpected: {:?}", edited.error);
        assert!(backend.write("fresh.txt", "new").error.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn traversal_does_not_follow_a_self_referencing_symlink() {
        let (dir, backend) = backend();
        std::fs::write(dir.path().join("a.rs"), "needle").unwrap();
        // Without a symlink check this recurses until the stack overflows.
        std::os::unix::fs::symlink(dir.path(), dir.path().join("loop")).unwrap();

        assert_eq!(backend.glob_info("*.rs", ".").unwrap().len(), 1);
        assert_eq!(
            backend.grep_raw("needle", Some("."), None).unwrap().len(),
            1
        );
    }

    #[test]
    fn read_with_an_enormous_limit_does_not_overflow() {
        let (dir, backend) = backend();
        std::fs::write(dir.path().join("big.txt"), "one\ntwo\nthree").unwrap();
        // A model can send any integer; `offset + limit` used to wrap and panic.
        assert_eq!(
            backend.read("big.txt", 0, u64::MAX as usize).unwrap(),
            "one\ntwo\nthree"
        );
        assert_eq!(
            backend.read("big.txt", 1, usize::MAX).unwrap(),
            "two\nthree"
        );
        // Past EOF is an empty read, not a panic.
        assert_eq!(backend.read("big.txt", 99, usize::MAX).unwrap(), "");
    }

    #[test]
    fn output_truncation_cuts_on_a_char_boundary() {
        // 'é' is two bytes, so a cap of 5 lands inside the third one — the
        // case where `String::truncate` panics.
        let out = truncate_output("ééé".to_string(), 5);
        assert!(out.starts_with("éé"), "got: {out}");
        assert!(out.contains("[output truncated"), "got: {out}");
        let untouched = truncate_output("ééé".to_string(), 64);
        assert_eq!(untouched, "ééé");
    }

    #[cfg(unix)]
    #[test]
    fn execute_captures_oversized_output_without_hanging() {
        use std::time::{Duration, Instant};
        let (_dir, backend) = backend();
        let started = Instant::now();
        // 2 MB — far past a pipe buffer, so an undrained child would block
        // until the deadline and be misreported as a timeout.
        let result = backend
            .execute("head -c 2000000 /dev/zero | tr '\\0' 'a'", 60)
            .unwrap();
        assert_eq!(result.exit_code, 0, "got: {}", result.output);
        assert!(
            result.output.contains("[output truncated"),
            "expected truncation, got {} bytes",
            result.output.len()
        );
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "took {:?} — the pipe was not drained",
            started.elapsed()
        );
    }

    #[cfg(unix)]
    #[test]
    fn execute_timeout_kills_the_whole_process_tree() {
        use std::time::{Duration, Instant};
        let (_dir, backend) = backend();
        let started = Instant::now();
        // `sh` waits on a grandchild; killing only `sh` would leave it running.
        let result = backend.execute("sleep 30 & wait", 1).unwrap();
        assert_eq!(result.exit_code, -1);
        assert!(
            result.output.contains("timed out"),
            "got: {}",
            result.output
        );
        assert!(
            started.elapsed() < Duration::from_secs(15),
            "timeout path hung for {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn write_then_read_roundtrip_inside_root() {
        let (_dir, backend) = backend();
        let result = backend.write("nested/dir/new.txt", "content");
        assert!(result.error.is_none(), "unexpected: {:?}", result.error);
        assert_eq!(
            backend.read("nested/dir/new.txt", 0, 10).unwrap(),
            "content"
        );
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
        assert!(
            err.contains("Re-read the file"),
            "must be actionable: {err}"
        );
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
