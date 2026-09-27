//! Hashing, canonical JSON and atomic file writes shared by every module.

use anyhow::{Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::Path;

/// Lower-case hex sha256 of `bytes`.
pub fn sha256_hex(bytes: impl AsRef<[u8]>) -> String {
    let d = Sha256::digest(bytes.as_ref());
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// sha256 of a rank (or id) vector: each value as a u32, little-endian,
/// concatenated. This is the rank-vector hash recorded in receipts and in the
/// ledger; `--verify-final` recomputes it the same way.
pub fn sha256_u32s(v: &[usize]) -> String {
    let mut h = Sha256::new();
    for &x in v {
        h.update((x as u32).to_le_bytes());
    }
    let mut s = String::with_capacity(64);
    for b in h.finalize() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Canonical JSON: object keys sorted recursively (by UTF-16 code units, the
/// order `receipt.mjs`' `canonicalJson` uses), no whitespace, `null` fields
/// kept. Independent of `serde_json`'s `preserve_order` feature.
pub fn canonical_json(v: &Value) -> String {
    let mut out = String::new();
    write_canonical(v, &mut out);
    out
}

fn write_canonical(v: &Value, out: &mut String) {
    match v {
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(k.clone()).to_string());
                out.push(':');
                write_canonical(&m[k], out);
            }
            out.push('}');
        }
        other => out.push_str(&other.to_string()),
    }
}

/// sha256 of the canonical JSON of `v`.
pub fn canonical_hash(v: &Value) -> String {
    sha256_hex(canonical_json(v))
}

/// Write `bytes` to `path` atomically: a sibling temp file is written and
/// fsynced, then renamed over `path`, then the directory is fsynced. A reader
/// sees either the old file or the new one, never a torn write.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let name = path
        .file_name()
        .context("atomic_write: path has no file name")?;
    let tmp = dir.join(format!(
        ".{}.tmp-{}",
        name.to_string_lossy(),
        std::process::id()
    ));
    {
        let mut f = fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))?;
    if let Ok(d) = fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

/// Current UTC time as RFC 3339 (seconds precision).
pub fn utc_now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_json_sorts_keys_recursively() {
        let a = json!({"b": 1, "a": {"y": [1, {"d": 2, "c": 3}], "x": null}});
        let b = json!({"a": {"x": null, "y": [1, {"c": 3, "d": 2}]}, "b": 1});
        assert_eq!(canonical_json(&a), canonical_json(&b));
        assert_eq!(
            canonical_json(&a),
            r#"{"a":{"x":null,"y":[1,{"c":3,"d":2}]},"b":1}"#
        );
    }

    #[test]
    fn sha256_known_vector() {
        // sha256("a\tr\tb\n") — the M0 tamper fixture quoted in ADR-007 §2.4.
        assert!(sha256_hex("a\tr\tb\n").starts_with("b664af6c"));
        assert_eq!(sha256_u32s(&[1, 2]), sha256_hex([1u8, 0, 0, 0, 2, 0, 0, 0]));
    }

    #[test]
    fn atomic_write_replaces() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("f.bin");
        atomic_write(&p, b"one").unwrap();
        atomic_write(&p, b"two").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"two");
        assert_eq!(
            fs::read_dir(d.path()).unwrap().count(),
            1,
            "no temp files left"
        );
    }
}
