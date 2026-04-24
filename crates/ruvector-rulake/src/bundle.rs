//! The `table.rulake.json` bundle sidecar.
//!
//! This is the portable unit that defines the reproducibility and
//! governance scope of a ruLake-compatible table. It does not contain
//! the vectors themselves; it *points at* them (via `data_ref`) and
//! carries the cryptographically-linked metadata needed to prove that
//! a given cache entry was built from exactly those bytes.
//!
//! ## Why this matters more than the UDF
//!
//! The remote function, the BigQuery integration, the cache service —
//! all of them are implementation details that can be swapped. The
//! bundle is what travels: between teams, between clouds, between
//! backups-and-restores. If the bundle format is wrong, every
//! layer on top of it leaks.
//!
//! ## Contract
//!
//! A ruLake bundle MUST carry:
//!
//! - **`data_ref`** — URI pointing at the authoritative byte stream
//!   (Parquet on GCS, an Iceberg table, an RVF segment, etc). ruLake
//!   does not own these bytes.
//! - **`dim`** — vector dimensionality. Immutable once set.
//! - **`rotation_seed`** — the Haar-uniform rotation seed used by the
//!   RaBitQ compressor. Deterministic — two caches built with the
//!   same seed produce byte-identical codes.
//! - **`rerank_factor`** — RaBitQ rerank factor. Declared so consumers
//!   can reason about the recall guarantee without re-running the
//!   sweep.
//! - **`generation`** — the backend's own coherence token at the time
//!   the bundle was produced (mtime, snapshot id, etc). Opaque `Bytes`
//!   so we're not locked to u64 when Iceberg hands us a UUID.
//! - **`rvf_witness`** — SHAKE-256 digest over `(data_ref, dim,
//!   rotation_seed, rerank_factor, generation)`. The cache-key anchor.
//!   Two bundles with the same witness are interchangeable for any
//!   query.
//! - **`pii_policy`** — opaque per-lake classifier reference. ruLake
//!   doesn't interpret it in v1; it passes it through so governance
//!   layers can enforce.
//! - **`lineage_id`** — OpenLineage job id that produced this bundle.
//!
//! The witness makes bundles equality-testable without comparing
//! payloads — two instances of ruLake observing the same bundle from
//! different backends can cache-share safely.

use serde::{Deserialize, Serialize};

/// Opaque coherence token. Large enough to hold a Parquet mtime
/// (u64 seconds), an Iceberg snapshot id (i64 positive), or a
/// Snowflake change-stream offset (u64 nanos). For longer tokens —
/// most notably multi-part Delta transaction logs or raw UUIDs — use
/// the `Opaque` variant and store the serialized bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Generation {
    /// Numeric token (mtime, version, snapshot id).
    Num(u64),
    /// Opaque string (UUID, hash, base64 blob).
    Opaque(String),
}

impl Generation {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Num(n) => Some(*n),
            Self::Opaque(_) => None,
        }
    }

    /// Concatenate into a byte stream for witness digest input.
    ///
    /// Prepends a 1-byte variant tag (`0x00` for `Num`, `0x01` for
    /// `Opaque`) so the two variants can never produce the same
    /// hash-bytes stream. Without the tag, `Num(7)` and
    /// `Opaque("\x07\0\0\0\0\0\0\0")` would collide — a correctness
    /// bug surfaced by the 2026-04-23 security audit. The witness
    /// format now includes this tag; old witnesses computed without
    /// it will not match newly-computed ones, which is intended:
    /// it's a breaking change to the witness, bumped as such in the
    /// bundle's `format_version = 2`.
    pub fn hash_bytes(&self) -> Vec<u8> {
        match self {
            Self::Num(n) => {
                let mut out = Vec::with_capacity(1 + 8);
                out.push(0x00);
                out.extend_from_slice(&n.to_le_bytes());
                out
            }
            Self::Opaque(s) => {
                let mut out = Vec::with_capacity(1 + s.len());
                out.push(0x01);
                out.extend_from_slice(s.as_bytes());
                out
            }
        }
    }
}

impl From<u64> for Generation {
    fn from(n: u64) -> Self {
        Self::Num(n)
    }
}

impl From<String> for Generation {
    fn from(s: String) -> Self {
        Self::Opaque(s)
    }
}

/// The bundle payload. Serializes to `table.rulake.json` verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuLakeBundle {
    /// Semantic version of the bundle format. Bump on breaking changes.
    pub format_version: u32,

    /// URI of the authoritative byte stream. Not owned by ruLake.
    pub data_ref: String,

    /// Vector dimensionality.
    pub dim: usize,

    /// RaBitQ rotation seed. Determines the cache's compressed codes.
    pub rotation_seed: u64,

    /// RaBitQ rerank factor. Determines the recall guarantee.
    pub rerank_factor: usize,

    /// Backend-reported coherence token at bundle-emission time.
    pub generation: Generation,

    /// Anchored digest over the other fields — the cache-key anchor.
    /// SHAKE-256(32) hex-encoded.
    pub rvf_witness: String,

    /// Opaque PII-policy handle — passed to the governance layer.
    pub pii_policy: Option<String>,

    /// OpenLineage job id that produced this bundle.
    pub lineage_id: Option<String>,

    /// Caller-defined memory-class tag (ADR-156 substrate framing).
    /// Agent systems tag bundles with cognitive labels like
    /// `"episodic"` / `"semantic"` / `"procedural"` / `"identity"`;
    /// ruLake stores the string and surfaces it through bundles and
    /// stats but never interprets it. Opaque by design — brain
    /// systems own the semantics, substrate owns the persistence.
    ///
    /// Not part of the witness: changing the memory-class tag on the
    /// same vectors does not invalidate a cache entry. Two bundles
    /// with identical data but different classes share the cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_class: Option<String>,
}

impl RuLakeBundle {
    /// Witness-format version. Bumped to 2 on 2026-04-23 when the
    /// `Generation` variant tag byte was added to `hash_bytes()` (see
    /// security audit §4). Version 1 witnesses are NOT forward-
    /// compatible with version 2 — operators with persisted v1
    /// bundles must re-prime the cache.
    pub const FORMAT_VERSION: u32 = 2;

    /// Construct a bundle, computing the witness from the other fields.
    pub fn new(
        data_ref: impl Into<String>,
        dim: usize,
        rotation_seed: u64,
        rerank_factor: usize,
        generation: Generation,
    ) -> Self {
        let data_ref = data_ref.into();
        let witness = compute_witness(&data_ref, dim, rotation_seed, rerank_factor, &generation);
        Self {
            format_version: Self::FORMAT_VERSION,
            data_ref,
            dim,
            rotation_seed,
            rerank_factor,
            generation,
            rvf_witness: witness,
            pii_policy: None,
            lineage_id: None,
            memory_class: None,
        }
    }

    /// Recompute the witness from the other fields. Useful after
    /// deserialization to verify a bundle hasn't been tampered with.
    pub fn verify_witness(&self) -> bool {
        let expected = compute_witness(
            &self.data_ref,
            self.dim,
            self.rotation_seed,
            self.rerank_factor,
            &self.generation,
        );
        expected == self.rvf_witness
    }

    /// Serialize to JSON (the on-disk `table.rulake.json` form).
    pub fn to_json(&self) -> crate::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::RuLakeError::InvalidParameter(format!("bundle serialize: {e}")))
    }

    /// Deserialize from JSON. Fails if the format version is newer than
    /// this binary supports, the JSON exceeds the max size, or any
    /// metadata field violates its length cap.
    ///
    /// Length caps are a hardening measure against malicious bundles
    /// (e.g. a compromised GCS object) that would otherwise force the
    /// reader to allocate unbounded memory during deserialization.
    pub fn from_json(s: &str) -> crate::Result<Self> {
        // Hard cap on the input size. A legitimate bundle is a few
        // hundred bytes; 64 KiB is 100× headroom for custom metadata.
        const MAX_JSON_BYTES: usize = 64 * 1024;
        const MAX_FIELD_BYTES: usize = 4 * 1024;
        if s.len() > MAX_JSON_BYTES {
            return Err(crate::RuLakeError::InvalidParameter(format!(
                "bundle parse: exceeds {MAX_JSON_BYTES} bytes"
            )));
        }
        let b: Self = serde_json::from_str(s)
            .map_err(|e| crate::RuLakeError::InvalidParameter(format!("bundle parse: {e}")))?;
        if b.format_version > Self::FORMAT_VERSION {
            return Err(crate::RuLakeError::InvalidParameter(format!(
                "bundle format_version={} newer than this binary supports ({})",
                b.format_version,
                Self::FORMAT_VERSION
            )));
        }
        // Per-field length caps. A single field over 4 KiB is almost
        // certainly wrong; legitimate values (URIs, UUIDs, policy IDs)
        // are orders of magnitude smaller.
        for (name, opt) in [
            ("pii_policy", b.pii_policy.as_deref()),
            ("lineage_id", b.lineage_id.as_deref()),
            ("memory_class", b.memory_class.as_deref()),
        ] {
            if let Some(v) = opt {
                if v.len() > MAX_FIELD_BYTES {
                    return Err(crate::RuLakeError::InvalidParameter(format!(
                        "bundle parse: {name} exceeds {MAX_FIELD_BYTES} bytes"
                    )));
                }
            }
        }
        if b.data_ref.len() > MAX_FIELD_BYTES {
            return Err(crate::RuLakeError::InvalidParameter(format!(
                "bundle parse: data_ref exceeds {MAX_FIELD_BYTES} bytes"
            )));
        }
        if b.rvf_witness.len() > 128 {
            // SHAKE-256(32) hex is exactly 64; reject anything wildly
            // off before we try to compare witnesses.
            return Err(crate::RuLakeError::InvalidParameter(
                "bundle parse: rvf_witness not a hex-encoded SHAKE-256(32)".to_string(),
            ));
        }
        Ok(b)
    }

    pub fn with_pii_policy(mut self, p: impl Into<String>) -> Self {
        self.pii_policy = Some(p.into());
        self
    }

    pub fn with_lineage_id(mut self, id: impl Into<String>) -> Self {
        self.lineage_id = Some(id.into());
        self
    }

    /// Tag this bundle with a memory class (ADR-156). Opaque to
    /// ruLake; meaningful to the consuming brain system.
    pub fn with_memory_class(mut self, class: impl Into<String>) -> Self {
        self.memory_class = Some(class.into());
        self
    }

    /// Canonical filename for the on-disk bundle sidecar.
    pub const SIDECAR_FILENAME: &'static str = "table.rulake.json";

    /// Atomically write the bundle to `dir/table.rulake.json`.
    ///
    /// Writes to a sibling temp file then renames into place — so a
    /// concurrent reader either sees the previous bundle or the new one,
    /// never a truncated file. This matches the pattern the BQ UDF +
    /// cache sidecar uses when the warehouse pushes a new generation.
    pub fn write_to_dir(
        &self,
        dir: impl AsRef<std::path::Path>,
    ) -> crate::Result<std::path::PathBuf> {
        use std::io::Write;
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(|e| {
            crate::RuLakeError::InvalidParameter(format!(
                "bundle write: mkdir {}: {e}",
                dir.display()
            ))
        })?;
        let final_path = dir.join(Self::SIDECAR_FILENAME);
        let tmp_path = dir.join(format!(
            ".{}.tmp.{}",
            Self::SIDECAR_FILENAME,
            std::process::id()
        ));
        let body = self.to_json()?;
        {
            let mut f = std::fs::File::create(&tmp_path).map_err(|e| {
                crate::RuLakeError::InvalidParameter(format!(
                    "bundle write: create {}: {e}",
                    tmp_path.display()
                ))
            })?;
            f.write_all(body.as_bytes()).map_err(|e| {
                crate::RuLakeError::InvalidParameter(format!("bundle write: write: {e}"))
            })?;
            f.sync_all().map_err(|e| {
                crate::RuLakeError::InvalidParameter(format!("bundle write: fsync: {e}"))
            })?;
        }
        std::fs::rename(&tmp_path, &final_path).map_err(|e| {
            crate::RuLakeError::InvalidParameter(format!(
                "bundle write: rename {} → {}: {e}",
                tmp_path.display(),
                final_path.display()
            ))
        })?;
        Ok(final_path)
    }

    /// Read `dir/table.rulake.json` and verify its witness.
    ///
    /// Returns an error when the sidecar is missing, malformed, carries a
    /// newer `format_version`, or when the on-disk witness does not match
    /// the recomputed value. The witness check is the anchor that makes
    /// a bundle trustworthy across the cache + UDF boundary.
    pub fn read_from_dir(dir: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        let path = dir.as_ref().join(Self::SIDECAR_FILENAME);
        let body = std::fs::read_to_string(&path).map_err(|e| {
            crate::RuLakeError::InvalidParameter(format!(
                "bundle read: open {}: {e}",
                path.display()
            ))
        })?;
        let bundle = Self::from_json(&body)?;
        if !bundle.verify_witness() {
            return Err(crate::RuLakeError::InvalidParameter(format!(
                "bundle read: witness mismatch at {}",
                path.display()
            )));
        }
        Ok(bundle)
    }
}

/// SHAKE-256(32) over a stable concatenation of the bundle fields.
/// Deliberately domain-separated by a fixed prefix so witnesses can't
/// collide with other uses of SHAKE.
fn compute_witness(
    data_ref: &str,
    dim: usize,
    rotation_seed: u64,
    rerank_factor: usize,
    generation: &Generation,
) -> String {
    use sha3::{
        digest::{ExtendableOutput, Update},
        Shake256,
    };
    let mut h = Shake256::default();
    h.update(b"rulake-bundle-witness-v1|");
    h.update(&(data_ref.len() as u64).to_le_bytes());
    h.update(data_ref.as_bytes());
    h.update(b"|");
    h.update(&(dim as u64).to_le_bytes());
    h.update(&rotation_seed.to_le_bytes());
    h.update(&(rerank_factor as u64).to_le_bytes());
    h.update(b"|");
    let g = generation.hash_bytes();
    h.update(&(g.len() as u64).to_le_bytes());
    h.update(&g);
    let mut reader = h.finalize_xof();
    let mut out = [0u8; 32];
    use sha3::digest::XofReader;
    reader.read(&mut out);
    hex::encode(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_is_deterministic() {
        let a = RuLakeBundle::new("gs://bucket/x.parquet", 128, 42, 20, Generation::Num(7));
        let b = RuLakeBundle::new("gs://bucket/x.parquet", 128, 42, 20, Generation::Num(7));
        assert_eq!(a.rvf_witness, b.rvf_witness);
    }

    #[test]
    fn witness_changes_on_any_field() {
        let a = RuLakeBundle::new("gs://bucket/x.parquet", 128, 42, 20, Generation::Num(7));
        // Different data_ref.
        let b = RuLakeBundle::new("gs://bucket/y.parquet", 128, 42, 20, Generation::Num(7));
        assert_ne!(a.rvf_witness, b.rvf_witness);
        // Different dim.
        let c = RuLakeBundle::new("gs://bucket/x.parquet", 256, 42, 20, Generation::Num(7));
        assert_ne!(a.rvf_witness, c.rvf_witness);
        // Different seed.
        let d = RuLakeBundle::new("gs://bucket/x.parquet", 128, 43, 20, Generation::Num(7));
        assert_ne!(a.rvf_witness, d.rvf_witness);
        // Different rerank factor.
        let e = RuLakeBundle::new("gs://bucket/x.parquet", 128, 42, 5, Generation::Num(7));
        assert_ne!(a.rvf_witness, e.rvf_witness);
        // Different generation.
        let f = RuLakeBundle::new("gs://bucket/x.parquet", 128, 42, 20, Generation::Num(8));
        assert_ne!(a.rvf_witness, f.rvf_witness);
    }

    #[test]
    fn generation_num_and_opaque_cannot_collide() {
        // Regression against the security-audit finding: before the
        // tag byte, Num(7) and Opaque("\x07\0\0\0\0\0\0\0") shared
        // hash_bytes() output, so their witnesses collided.
        let a = RuLakeBundle::new("x", 1, 0, 1, Generation::Num(7));
        let b = RuLakeBundle::new(
            "x",
            1,
            0,
            1,
            Generation::Opaque("\x07\0\0\0\0\0\0\0".to_string()),
        );
        assert_ne!(
            a.rvf_witness, b.rvf_witness,
            "Num and Opaque witnesses must be distinguishable"
        );
    }

    #[test]
    fn witness_is_length_prefixed() {
        // Regression against "a|b" colliding with "ab|" on concat-only
        // witness schemes. Length prefixes should prevent that.
        let g1 = Generation::Opaque("a|b".to_string());
        let g2 = Generation::Opaque("ab|".to_string());
        let a = RuLakeBundle::new("x", 1, 0, 1, g1);
        let b = RuLakeBundle::new("x", 1, 0, 1, g2);
        assert_ne!(a.rvf_witness, b.rvf_witness);
    }

    #[test]
    fn opaque_generation_serialization_roundtrip() {
        let bundle = RuLakeBundle::new(
            "iceberg://catalog/db/table",
            384,
            0xdead_beef,
            5,
            Generation::Opaque("01JCX7NK6G5R9G1YZ7QH".to_string()),
        );
        let s = bundle.to_json().unwrap();
        let parsed = RuLakeBundle::from_json(&s).unwrap();
        assert_eq!(parsed.rvf_witness, bundle.rvf_witness);
        assert!(parsed.verify_witness());
    }

    #[test]
    fn verify_witness_catches_tamper() {
        let mut b = RuLakeBundle::new("x", 128, 1, 20, Generation::Num(1));
        assert!(b.verify_witness());
        // Tamper with a field without recomputing the witness.
        b.dim = 256;
        assert!(!b.verify_witness(), "witness verify must catch dim tamper");
    }

    #[test]
    fn format_version_downgrade_rejected() {
        let good = RuLakeBundle::new("x", 128, 1, 20, Generation::Num(1));
        let mut s = good.to_json().unwrap();
        // Simulate a future bundle with format_version: 99.
        s = s.replace("\"format_version\": 2", "\"format_version\": 99");
        let err = RuLakeBundle::from_json(&s).unwrap_err();
        assert!(matches!(err, crate::RuLakeError::InvalidParameter(_)));
    }

    fn tempdir(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!(
            "rulake-bundle-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn fs_roundtrip_writes_and_reads_canonical_sidecar() {
        let dir = tempdir("roundtrip");
        let orig = RuLakeBundle::new("gs://bucket/x.parquet", 128, 42, 20, Generation::Num(7))
            .with_pii_policy("pii://policies/default")
            .with_lineage_id("ol://jobs/ingest-42");

        let written = orig.write_to_dir(&dir).unwrap();
        assert_eq!(written.file_name().unwrap(), RuLakeBundle::SIDECAR_FILENAME);

        let loaded = RuLakeBundle::read_from_dir(&dir).unwrap();
        assert_eq!(loaded.rvf_witness, orig.rvf_witness);
        assert_eq!(loaded.data_ref, orig.data_ref);
        assert_eq!(loaded.pii_policy.as_deref(), Some("pii://policies/default"));
        assert_eq!(loaded.lineage_id.as_deref(), Some("ol://jobs/ingest-42"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fs_read_rejects_tampered_sidecar() {
        let dir = tempdir("tamper");
        let bundle = RuLakeBundle::new("x", 128, 1, 20, Generation::Num(1));
        bundle.write_to_dir(&dir).unwrap();

        // Tamper: overwrite dim on disk without recomputing witness.
        let path = dir.join(RuLakeBundle::SIDECAR_FILENAME);
        let body = std::fs::read_to_string(&path).unwrap();
        let tampered = body.replace("\"dim\": 128", "\"dim\": 256");
        std::fs::write(&path, tampered).unwrap();

        let err = RuLakeBundle::read_from_dir(&dir).unwrap_err();
        match err {
            crate::RuLakeError::InvalidParameter(msg) => {
                assert!(msg.contains("witness"), "unexpected err: {msg}");
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_json_rejects_oversize_input() {
        // 128 KiB of whitespace inside a bundle → must be rejected
        // before it reaches serde allocations.
        let payload = format!(
            "{{\"format_version\":1,\"data_ref\":\"{}\",\"dim\":1,\"rotation_seed\":0,\"rerank_factor\":1,\"generation\":1,\"rvf_witness\":\"\"}}",
            "x".repeat(128 * 1024)
        );
        let err = RuLakeBundle::from_json(&payload).unwrap_err();
        match err {
            crate::RuLakeError::InvalidParameter(m) => {
                assert!(m.contains("exceeds") && m.contains("bytes"))
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn from_json_rejects_oversize_metadata_field() {
        let good = RuLakeBundle::new("x", 1, 0, 1, Generation::Num(1));
        let mut v = serde_json::to_value(&good).unwrap();
        v["memory_class"] = serde_json::Value::String("m".repeat(5000));
        let s = serde_json::to_string(&v).unwrap();
        let err = RuLakeBundle::from_json(&s).unwrap_err();
        match err {
            crate::RuLakeError::InvalidParameter(m) => assert!(m.contains("memory_class")),
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn memory_class_roundtrips_and_does_not_affect_witness() {
        let plain = RuLakeBundle::new("x", 8, 1, 5, Generation::Num(1));
        let tagged =
            RuLakeBundle::new("x", 8, 1, 5, Generation::Num(1)).with_memory_class("episodic");
        // Witnesses match — class is not part of the digest.
        assert_eq!(plain.rvf_witness, tagged.rvf_witness);
        // Serde roundtrip preserves the tag.
        let s = tagged.to_json().unwrap();
        let parsed = RuLakeBundle::from_json(&s).unwrap();
        assert_eq!(parsed.memory_class.as_deref(), Some("episodic"));
        // Old-format (no memory_class field) still parses.
        let old_format = r#"{"format_version":1,"data_ref":"x","dim":8,"rotation_seed":1,"rerank_factor":5,"generation":1,"rvf_witness":""#;
        let legacy = format!(
            "{}{}\",\"pii_policy\":null,\"lineage_id\":null}}",
            old_format, plain.rvf_witness
        );
        let loaded = RuLakeBundle::from_json(&legacy).unwrap();
        assert_eq!(loaded.memory_class, None);
    }

    #[test]
    fn fs_write_is_atomic_under_crash_simulation() {
        // Simulate a crash between tmp-create and rename by writing a
        // canonical sidecar first, then attempting a second write that
        // we interrupt by leaving a leftover `.tmp.*` file in place.
        // Readers should still see the valid sidecar.
        let dir = tempdir("atomic");
        let v1 = RuLakeBundle::new("x", 128, 1, 20, Generation::Num(1));
        v1.write_to_dir(&dir).unwrap();

        // Drop a leftover tmp file; this mimics a crashed writer.
        let leftover = dir.join(format!(
            ".{}.tmp.{}",
            RuLakeBundle::SIDECAR_FILENAME,
            u32::MAX
        ));
        std::fs::write(&leftover, b"garbage-not-json").unwrap();

        // Reader still observes v1 untouched.
        let loaded = RuLakeBundle::read_from_dir(&dir).unwrap();
        assert_eq!(loaded.rvf_witness, v1.rvf_witness);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
