//! SHA-256 checksums for produced artifacts (requirements §3.8).
//!
//! Paths in a manifest are relative and sanitized. A checksum manifest is
//! shipped alongside installers and consumed on another machine, so a manifest
//! that could name `../../etc/ssh/sshd_config` would turn verification into an
//! arbitrary-read primitive. [`Sha256Manifest`] therefore rejects absolute
//! paths, `..` components, and Windows path prefixes on the way in and again
//! on the way out.

use crate::canonical::hex_lower;
use crate::error::{ForgeError, Result};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

/// Schema identifier carried by every [`Sha256Manifest`].
pub const CHECKSUM_SCHEMA: &str = "rvforge.checksums/1";

/// Read granularity when hashing an artifact. Installers reach gigabytes, so
/// hashing streams rather than reading the file into memory.
const CHUNK: usize = 64 * 1024;

/// One artifact's checksum.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sha256Entry {
    /// Relative path, with `/` separators.
    pub path: String,
    /// SHA-256 of the file contents, lowercase hex.
    pub sha256: String,
    /// File size in bytes.
    pub bytes: u64,
}

/// Checksums for every artifact of one build.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sha256Manifest {
    /// Schema identifier; always [`CHECKSUM_SCHEMA`].
    #[serde(default = "default_schema")]
    pub schema: String,
    /// Entries sorted by path, so the manifest is byte-stable regardless of
    /// the order the caller listed its artifacts in.
    pub entries: Vec<Sha256Entry>,
}

fn default_schema() -> String {
    CHECKSUM_SCHEMA.to_string()
}

/// The outcome of re-checking one entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChecksumVerification {
    /// The entry's path.
    pub path: String,
    /// Whether the file on disk matches the manifest.
    pub matched: bool,
    /// What went wrong, when it did not match.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl Sha256Manifest {
    /// Whether the manifest has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many artifacts are covered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The `sha256sum(1)` text form: `<hex>  <path>`, one line per entry.
    pub fn to_sha256sum_text(&self) -> String {
        let mut s = String::new();
        for e in &self.entries {
            s.push_str(&e.sha256);
            s.push_str("  ");
            s.push_str(&e.path);
            s.push('\n');
        }
        s
    }

    /// Parse the `sha256sum(1)` text form.
    ///
    /// # Errors
    ///
    /// [`ForgeError::Manifest`] for a malformed line or an unsafe path. Sizes
    /// are not recoverable from this format and are recorded as zero.
    pub fn from_sha256sum_text(text: &str) -> Result<Self> {
        let mut entries = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            let (hash, path) = line.split_once("  ").ok_or_else(|| ForgeError::Manifest {
                detail: format!("line {}: expected '<sha256>  <path>'", n + 1),
            })?;
            if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(ForgeError::Manifest {
                    detail: format!("line {}: {hash:?} is not a hex SHA-256", n + 1),
                });
            }
            entries.push(Sha256Entry {
                path: sanitize_relative(path)?,
                sha256: hash.to_ascii_lowercase(),
                bytes: 0,
            });
        }
        entries.sort();
        Ok(Self {
            schema: default_schema(),
            entries,
        })
    }
}

/// Hash every path in `paths`, recording each relative to `root`.
///
/// # Errors
///
/// [`ForgeError::Io`] if a file cannot be read; [`ForgeError::Manifest`] if a
/// path escapes `root` or is not valid UTF-8.
pub fn checksums<P: AsRef<Path>>(root: impl AsRef<Path>, paths: &[P]) -> Result<Sha256Manifest> {
    let root = root.as_ref();
    let mut entries = Vec::with_capacity(paths.len());

    for p in paths {
        let path = p.as_ref();
        // has_root(), not is_absolute(): on Windows "/etc/hostname" has a
        // root but no drive prefix, so is_absolute() is false — yet it is
        // not a safe relative path either. Any rooted path takes the
        // containment-checked branch on every platform.
        let absolute = if path.has_root() {
            path.to_path_buf()
        } else {
            root.join(path)
        };
        let relative = relative_to(root, path)?;
        let (digest, bytes) = hash_file(&absolute)?;
        entries.push(Sha256Entry {
            path: relative,
            sha256: hex_lower(&digest),
            bytes,
        });
    }

    entries.sort();
    entries.dedup();
    Ok(Sha256Manifest {
        schema: default_schema(),
        entries,
    })
}

/// Re-hash every entry of `manifest` against the files under `root`.
///
/// Returns one [`ChecksumVerification`] per entry, in manifest order. A
/// missing or mismatched file yields `matched: false` rather than an error, so
/// that a caller sees every discrepancy in one pass instead of only the first.
///
/// # Errors
///
/// [`ForgeError::Manifest`] if an entry's path is unsafe.
pub fn verify_checksums(
    manifest: &Sha256Manifest,
    root: impl AsRef<Path>,
) -> Result<Vec<ChecksumVerification>> {
    let root = root.as_ref();
    let mut out = Vec::with_capacity(manifest.entries.len());

    for entry in &manifest.entries {
        let safe = sanitize_relative(&entry.path)?;
        let path = root.join(&safe);
        let verification = match hash_file(&path) {
            Err(e) => ChecksumVerification {
                path: entry.path.clone(),
                matched: false,
                detail: Some(e.to_string()),
            },
            Ok((digest, bytes)) => {
                let actual = hex_lower(&digest);
                if actual == entry.sha256.to_ascii_lowercase() {
                    ChecksumVerification {
                        path: entry.path.clone(),
                        matched: true,
                        detail: None,
                    }
                } else {
                    ChecksumVerification {
                        path: entry.path.clone(),
                        matched: false,
                        detail: Some(format!(
                            "expected {}, file hashes to {actual} ({bytes} bytes)",
                            entry.sha256
                        )),
                    }
                }
            }
        };
        out.push(verification);
    }
    Ok(out)
}

/// Stream `path` through SHA-256, returning the digest and the byte count.
fn hash_file(path: &Path) -> Result<([u8; 32], u64)> {
    let file = std::fs::File::open(path).map_err(|e| ForgeError::io(path, &e))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = rvf_types::Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    let mut total = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| ForgeError::io(path, &e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hasher.finalize(), total))
}

/// Express `path` relative to `root`, rejecting anything that escapes it.
fn relative_to(root: &Path, path: &Path) -> Result<String> {
    let relative = if path.has_root() {
        path.strip_prefix(root).map_err(|_| ForgeError::Manifest {
            detail: format!(
                "{} is outside the artifact root {}",
                path.display(),
                root.display()
            ),
        })?
    } else {
        path
    };
    sanitize_relative(&relative.to_string_lossy())
}

/// Normalize a relative path to `/`-separated form, refusing absolute paths,
/// `..` components, and Windows prefixes.
fn sanitize_relative(path: &str) -> Result<String> {
    if path.is_empty() {
        return Err(ForgeError::Manifest {
            detail: "artifact path is empty".to_string(),
        });
    }

    let mut parts = Vec::new();
    for component in Path::new(path).components() {
        match component {
            Component::Normal(part) => {
                let part = part.to_str().ok_or_else(|| ForgeError::Manifest {
                    detail: format!("artifact path {path:?} is not valid UTF-8"),
                })?;
                parts.push(part.to_string());
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(ForgeError::Manifest {
                    detail: format!(
                        "artifact path {path:?} must be relative and must not traverse upward"
                    ),
                })
            }
        }
    }

    if parts.is_empty() {
        return Err(ForgeError::Manifest {
            detail: format!("artifact path {path:?} names no file"),
        });
    }
    Ok(parts.join("/"))
}

/// Join `root` with a sanitized relative path.
///
/// Exposed so a caller that resolves artifact paths itself gets the same
/// traversal refusal that [`verify_checksums`] applies.
///
/// # Errors
///
/// [`ForgeError::Manifest`] if `relative` is not a safe relative path.
pub fn resolve_artifact(root: impl AsRef<Path>, relative: &str) -> Result<PathBuf> {
    Ok(root.as_ref().join(sanitize_relative(relative)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_with(files: &[(&str, &[u8])]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, body) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, body).unwrap();
        }
        dir
    }

    #[test]
    fn checksums_round_trip_through_verification() {
        let dir = tmp_with(&[("Agent.msi", b"installer"), ("Agent.rvf", b"artifact")]);
        let m = checksums(dir.path(), &["Agent.msi", "Agent.rvf"]).unwrap();
        assert_eq!(m.len(), 2);

        let results = verify_checksums(&m, dir.path()).unwrap();
        assert!(results.iter().all(|r| r.matched), "{results:?}");
    }

    #[test]
    fn a_modified_artifact_fails_verification() {
        let dir = tmp_with(&[("Agent.msi", b"installer")]);
        let m = checksums(dir.path(), &["Agent.msi"]).unwrap();
        std::fs::write(dir.path().join("Agent.msi"), b"tampered").unwrap();

        let results = verify_checksums(&m, dir.path()).unwrap();
        assert!(!results[0].matched);
        assert!(results[0].detail.as_ref().unwrap().contains("hashes to"));
    }

    #[test]
    fn a_missing_artifact_fails_verification_without_erroring() {
        let dir = tmp_with(&[("Agent.msi", b"installer")]);
        let m = checksums(dir.path(), &["Agent.msi"]).unwrap();
        std::fs::remove_file(dir.path().join("Agent.msi")).unwrap();

        let results = verify_checksums(&m, dir.path()).unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].matched);
    }

    #[test]
    fn entries_are_sorted_regardless_of_input_order() {
        let dir = tmp_with(&[("b.msi", b"b"), ("a.msi", b"a")]);
        let forward = checksums(dir.path(), &["a.msi", "b.msi"]).unwrap();
        let reverse = checksums(dir.path(), &["b.msi", "a.msi"]).unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(forward.entries[0].path, "a.msi");
    }

    #[test]
    fn hash_matches_a_known_sha256() {
        // Well-known vector: SHA-256 of "abc".
        let dir = tmp_with(&[("f", b"abc")]);
        let m = checksums(dir.path(), &["f"]).unwrap();
        assert_eq!(
            m.entries[0].sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(m.entries[0].bytes, 3);
    }

    #[test]
    fn large_files_hash_across_chunk_boundaries() {
        let body = vec![0x5au8; CHUNK * 2 + 7];
        let dir = tmp_with(&[("big.bin", body.as_slice())]);
        let m = checksums(dir.path(), &["big.bin"]).unwrap();
        assert_eq!(m.entries[0].sha256, hex_lower(&rvf_types::sha256(&body)));
        assert_eq!(m.entries[0].bytes, body.len() as u64);
    }

    #[test]
    fn nested_paths_use_forward_slashes() {
        let dir = tmp_with(&[("out/linux/Agent.deb", b"deb")]);
        let m = checksums(dir.path(), &["out/linux/Agent.deb"]).unwrap();
        assert_eq!(m.entries[0].path, "out/linux/Agent.deb");
        assert!(verify_checksums(&m, dir.path()).unwrap()[0].matched);
    }

    #[test]
    fn traversal_paths_are_refused() {
        let m = Sha256Manifest {
            schema: default_schema(),
            entries: vec![Sha256Entry {
                path: "../../etc/passwd".into(),
                sha256: "00".repeat(32),
                bytes: 0,
            }],
        };
        let err = verify_checksums(&m, "/tmp").unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Manifest);
        assert!(err.to_string().contains("traverse upward"), "{err}");
    }

    #[test]
    fn absolute_paths_are_refused() {
        assert!(sanitize_relative("/etc/passwd").is_err());
        assert!(resolve_artifact("/tmp", "../escape").is_err());
        assert_eq!(
            resolve_artifact("/tmp", "./out/Agent.msi").unwrap(),
            PathBuf::from("/tmp/out/Agent.msi")
        );
    }

    #[test]
    fn absolute_input_paths_outside_the_root_are_refused() {
        let dir = tmp_with(&[("Agent.msi", b"x")]);
        let err = checksums(dir.path(), &["/etc/hostname"]).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Manifest);
        assert!(
            err.to_string().contains("outside the artifact root"),
            "{err}"
        );
    }

    #[test]
    fn sha256sum_text_round_trips() {
        let dir = tmp_with(&[("a.msi", b"a"), ("b.msi", b"b")]);
        let m = checksums(dir.path(), &["a.msi", "b.msi"]).unwrap();
        let text = m.to_sha256sum_text();
        assert_eq!(text.lines().count(), 2);

        let parsed = Sha256Manifest::from_sha256sum_text(&text).unwrap();
        let paths: Vec<&str> = parsed.entries.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(paths, vec!["a.msi", "b.msi"]);
        assert_eq!(parsed.entries[0].sha256, m.entries[0].sha256);
    }

    #[test]
    fn malformed_sha256sum_text_is_refused() {
        assert!(Sha256Manifest::from_sha256sum_text("nonsense").is_err());
        assert!(Sha256Manifest::from_sha256sum_text("zz  a.msi").is_err());
        assert!(
            Sha256Manifest::from_sha256sum_text(&format!("{}  ../a", "ab".repeat(32))).is_err()
        );
    }

    #[test]
    fn a_missing_file_is_an_io_error_at_manifest_time() {
        let dir = tempfile::tempdir().unwrap();
        let err = checksums(dir.path(), &["nope.msi"]).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Io);
    }
}
