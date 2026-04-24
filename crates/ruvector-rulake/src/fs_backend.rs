//! `FsBackend` — a filesystem-backed adapter reading vectors from a
//! simple binary file format.
//!
//! This is the concrete M2 on-ramp: it proves that the bundle +
//! witness + cache loop works end-to-end against real persistent data
//! (mtime-as-generation, file-URI-as-data_ref), without dragging in
//! arrow / parquet / delta / iceberg dependencies. A real
//! `ParquetBackend` will reuse this exact shape — only the decoder
//! and the generation source change.
//!
//! ## On-disk format (`ruvec1`)
//!
//! ```text
//!   bytes  field
//!   0..8   magic         = b"ruvec1\0\0"
//!   8..16  count : u64   little-endian
//!  16..20  dim   : u32   little-endian
//!  20..24  _reserved     (must be zero)
//!  24..    records × count, each:
//!            id : u64  little-endian
//!            v  : f32 × dim  little-endian
//! ```
//!
//! Fixed-stride records so pulling is a single `read + transmute` on
//! little-endian platforms. Big-endian platforms take a byte-swap pass.
//!
//! ## Why mtime is enough for the generation
//!
//! ruLake's coherence check runs on `ensure_fresh`, i.e. once per
//! search-that-needs-checking. The backend reports `generation()` as
//! `mtime.seconds_since_epoch`; ruLake folds it into the witness via
//! the bundle. If the file changes, `mtime` changes, the witness
//! changes, the cache primes a new entry. Sub-second re-writes look
//! stale for up to one second — matches the Parquet-on-S3 story and
//! is documented in ADR-155 §Decision 3.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::UNIX_EPOCH;

use crate::backend::{BackendAdapter, CollectionId, PulledBatch};
use crate::bundle::{Generation, RuLakeBundle};
use crate::error::{Result, RuLakeError};

const MAGIC: [u8; 8] = *b"ruvec1\0\0";
const HEADER_BYTES: usize = 24;

/// File-backed vector collection. Each registered collection maps to
/// exactly one `ruvec1` file on disk.
pub struct FsBackend {
    id: String,
    root: PathBuf,
    // Collection → filename (relative to `root`). Kept separate from the
    // filename so operators can rename files without breaking the
    // collection namespace — the witness is still anchored on `data_ref`.
    index: RwLock<std::collections::HashMap<CollectionId, String>>,
}

impl FsBackend {
    pub fn new(id: impl Into<String>, root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        if !root.exists() {
            std::fs::create_dir_all(&root).map_err(|e| {
                RuLakeError::InvalidParameter(format!("FsBackend: mkdir {}: {e}", root.display()))
            })?;
        }
        Ok(Self {
            id: id.into(),
            root,
            index: RwLock::new(std::collections::HashMap::new()),
        })
    }

    /// Register `collection` as the name for `filename` (relative to
    /// root). The file doesn't have to exist yet — writes (`write`) and
    /// pulls (`pull_vectors`) will fail at call time if it's missing.
    ///
    /// Errors when `filename` contains path components that would
    /// escape the root directory (absolute path, `..`, leading `/`).
    /// This is the primary defense against path-traversal — an
    /// operator who lets user input flow into this API gets a hard
    /// failure, not a filesystem escape.
    pub fn register(
        &self,
        collection: impl Into<String>,
        filename: impl Into<String>,
    ) -> Result<()> {
        let filename = filename.into();
        Self::validate_filename(&filename)?;
        let mut idx = self.index.write().unwrap();
        idx.insert(collection.into(), filename);
        Ok(())
    }

    /// Reject any filename that could escape the root directory.
    ///
    /// Rules (matched by every major OS + strict-whitelist-safe):
    ///   - non-empty, ASCII printable, no control bytes;
    ///   - no `/` or `\` separators (this is a filename, not a path);
    ///   - no `.` or `..` components;
    ///   - no drive letter or Windows UNC prefix;
    ///   - length ≤ 255 bytes (POSIX NAME_MAX).
    fn validate_filename(f: &str) -> Result<()> {
        let invalid = |msg: &str| {
            Err(RuLakeError::InvalidParameter(format!(
                "FsBackend: illegal filename {:?}: {}",
                f, msg
            )))
        };
        if f.is_empty() {
            return invalid("empty");
        }
        if f.len() > 255 {
            return invalid("exceeds 255 bytes (POSIX NAME_MAX)");
        }
        if f == "." || f == ".." {
            return invalid("reserved component");
        }
        for b in f.bytes() {
            if b < 0x20 || b == 0x7f {
                return invalid("control byte");
            }
            if b == b'/' || b == b'\\' {
                return invalid("path separator");
            }
        }
        // Reject Windows drive prefix + UNC paths. (A bare `:` in the
        // middle of the name is fine on POSIX but we reject to keep
        // cross-platform semantics.)
        if f.contains(':') {
            return invalid("colon");
        }
        Ok(())
    }

    /// Write a collection to disk in `ruvec1` format. Registers the
    /// `collection → filename` mapping if not already registered.
    pub fn write(
        &self,
        collection: impl Into<String>,
        filename: impl Into<String>,
        dim: usize,
        ids: &[u64],
        vectors: &[Vec<f32>],
    ) -> Result<PathBuf> {
        let collection = collection.into();
        let filename = filename.into();
        Self::validate_filename(&filename)?;
        if ids.len() != vectors.len() {
            return Err(RuLakeError::InvalidParameter(format!(
                "FsBackend::write: ids.len={} != vectors.len={}",
                ids.len(),
                vectors.len()
            )));
        }
        for v in vectors {
            if v.len() != dim {
                return Err(RuLakeError::DimensionMismatch {
                    expected: dim,
                    actual: v.len(),
                });
            }
        }
        let path = self.root.join(&filename);
        let tmp = self.root.join(format!(".{filename}.tmp"));
        {
            let mut f = File::create(&tmp).map_err(|e| {
                RuLakeError::InvalidParameter(format!("FsBackend::write: create: {e}"))
            })?;
            f.write_all(&MAGIC)
                .and_then(|_| f.write_all(&(ids.len() as u64).to_le_bytes()))
                .and_then(|_| f.write_all(&(dim as u32).to_le_bytes()))
                .and_then(|_| f.write_all(&0u32.to_le_bytes()))
                .map_err(|e| {
                    RuLakeError::InvalidParameter(format!("FsBackend::write: header: {e}"))
                })?;
            for (id, v) in ids.iter().zip(vectors.iter()) {
                f.write_all(&id.to_le_bytes()).map_err(|e| {
                    RuLakeError::InvalidParameter(format!("FsBackend::write: id: {e}"))
                })?;
                for x in v {
                    f.write_all(&x.to_le_bytes()).map_err(|e| {
                        RuLakeError::InvalidParameter(format!("FsBackend::write: f32: {e}"))
                    })?;
                }
            }
            f.sync_all().map_err(|e| {
                RuLakeError::InvalidParameter(format!("FsBackend::write: fsync: {e}"))
            })?;
        }
        std::fs::rename(&tmp, &path).map_err(|e| {
            RuLakeError::InvalidParameter(format!(
                "FsBackend::write: rename {} → {}: {e}",
                tmp.display(),
                path.display()
            ))
        })?;
        self.register(collection, filename)?;
        Ok(path)
    }

    fn path_of(&self, collection: &str) -> Result<PathBuf> {
        let idx = self.index.read().unwrap();
        let fname = idx
            .get(collection)
            .ok_or_else(|| RuLakeError::UnknownCollection {
                backend: self.id.clone(),
                collection: collection.to_string(),
            })?;
        Ok(self.root.join(fname))
    }

    fn generation_of(&self, path: &Path) -> Result<u64> {
        let meta = std::fs::metadata(path).map_err(|e| {
            RuLakeError::InvalidParameter(format!("FsBackend: stat {}: {e}", path.display()))
        })?;
        let mtime = meta
            .modified()
            .map_err(|e| RuLakeError::InvalidParameter(format!("FsBackend: mtime: {e}")))?;
        let secs = mtime
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Ok(secs)
    }
}

impl BackendAdapter for FsBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn list_collections(&self) -> Result<Vec<CollectionId>> {
        Ok(self.index.read().unwrap().keys().cloned().collect())
    }

    fn pull_vectors(&self, collection: &str) -> Result<PulledBatch> {
        let path = self.path_of(collection)?;
        let mut f = File::open(&path).map_err(|e| {
            RuLakeError::InvalidParameter(format!("FsBackend::pull: open {}: {e}", path.display()))
        })?;
        let mut header = [0u8; HEADER_BYTES];
        f.read_exact(&mut header)
            .map_err(|e| RuLakeError::InvalidParameter(format!("FsBackend::pull: header: {e}")))?;
        if header[..8] != MAGIC {
            return Err(RuLakeError::InvalidParameter(format!(
                "FsBackend::pull: bad magic at {}",
                path.display()
            )));
        }
        // Bounds-check the on-disk header BEFORE any allocation. A
        // corrupt or hostile file claiming count=u64::MAX or
        // dim=u32::MAX would otherwise crash the host — use try_from
        // so 32-bit targets reject oversized counts instead of
        // silently truncating, and checked_mul so the vec-buffer
        // size can't overflow.
        let count_u64 = u64::from_le_bytes(header[8..16].try_into().unwrap());
        let dim_u32 = u32::from_le_bytes(header[16..20].try_into().unwrap());
        let count = usize::try_from(count_u64).map_err(|_| {
            RuLakeError::InvalidParameter(format!(
                "FsBackend::pull: count={count_u64} exceeds usize"
            ))
        })?;
        let dim = dim_u32 as usize;
        if count > crate::backend::MAX_PULLED_VECTORS {
            return Err(RuLakeError::InvalidParameter(format!(
                "FsBackend::pull: count={count} exceeds MAX_PULLED_VECTORS"
            )));
        }
        if dim == 0 || dim > crate::backend::MAX_PULLED_DIM {
            return Err(RuLakeError::InvalidParameter(format!(
                "FsBackend::pull: dim={dim} outside (0, MAX_PULLED_DIM]"
            )));
        }
        // header[20..24] reserved — ignored.

        let vec_buf_bytes = dim.checked_mul(4).ok_or_else(|| {
            RuLakeError::InvalidParameter("FsBackend::pull: dim*4 overflow".to_string())
        })?;
        let mut ids = Vec::with_capacity(count);
        let mut vectors = Vec::with_capacity(count);
        let mut id_buf = [0u8; 8];
        let mut vec_bytes = vec![0u8; vec_buf_bytes];
        for _ in 0..count {
            f.read_exact(&mut id_buf)
                .map_err(|e| RuLakeError::InvalidParameter(format!("FsBackend::pull: id: {e}")))?;
            f.read_exact(&mut vec_bytes)
                .map_err(|e| RuLakeError::InvalidParameter(format!("FsBackend::pull: vec: {e}")))?;
            ids.push(u64::from_le_bytes(id_buf));
            let mut v = Vec::with_capacity(dim);
            for k in 0..dim {
                let lo = k * 4;
                v.push(f32::from_le_bytes(
                    vec_bytes[lo..lo + 4].try_into().unwrap(),
                ));
            }
            vectors.push(v);
        }
        let generation = self.generation_of(&path)?;
        Ok(PulledBatch {
            collection: collection.to_string(),
            ids,
            vectors,
            dim,
            generation,
        })
    }

    fn generation(&self, collection: &str) -> Result<u64> {
        let path = self.path_of(collection)?;
        self.generation_of(&path)
    }

    /// Override: `FsBackend` knows its data_ref (file://...) and can
    /// read the header cheaply for dim without loading every vector.
    /// This is the hot-path ergonomics a real Parquet/Iceberg backend
    /// needs — `current_bundle` runs per search-that-checks, and a
    /// full pull there would be catastrophic.
    fn current_bundle(
        &self,
        collection: &str,
        rotation_seed: u64,
        rerank_factor: usize,
    ) -> Result<RuLakeBundle> {
        let path = self.path_of(collection)?;
        // Read only the 24-byte header to pick up the dim.
        let mut f = File::open(&path).map_err(|e| {
            RuLakeError::InvalidParameter(format!(
                "FsBackend::current_bundle: open {}: {e}",
                path.display()
            ))
        })?;
        let mut header = [0u8; HEADER_BYTES];
        f.read_exact(&mut header).map_err(|e| {
            RuLakeError::InvalidParameter(format!("FsBackend::current_bundle: header: {e}"))
        })?;
        if header[..8] != MAGIC {
            return Err(RuLakeError::InvalidParameter(format!(
                "FsBackend::current_bundle: bad magic at {}",
                path.display()
            )));
        }
        let dim = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
        let gen_ = self.generation_of(&path)?;
        let data_ref = format!("file://{}", path.display());
        Ok(RuLakeBundle::new(
            data_ref,
            dim,
            rotation_seed,
            rerank_factor,
            Generation::Num(gen_),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("rulake-fs-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn fs_write_then_pull_roundtrip() {
        let dir = tempdir("rt");
        let back = FsBackend::new("disk", &dir).unwrap();
        back.write(
            "c",
            "c.bin",
            3,
            &[10, 20, 30],
            &[
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
        )
        .unwrap();
        let batch = back.pull_vectors("c").unwrap();
        assert_eq!(batch.dim, 3);
        assert_eq!(batch.ids, vec![10, 20, 30]);
        assert_eq!(batch.vectors.len(), 3);
        assert_eq!(batch.vectors[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(batch.vectors[2], vec![7.0, 8.0, 9.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fs_bundle_has_file_uri_and_header_dim() {
        let dir = tempdir("bundle");
        let back = FsBackend::new("disk", &dir).unwrap();
        back.write("c", "c.bin", 4, &[1], &[vec![0.0; 4]]).unwrap();
        let b = back.current_bundle("c", 42, 20).unwrap();
        assert!(b.data_ref.starts_with("file://"), "got {}", b.data_ref);
        assert_eq!(b.dim, 4);
        assert!(b.verify_witness());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fs_register_rejects_path_traversal() {
        // Security gate: user-controlled filenames cannot escape the
        // backend's root directory. Every mode of escape must fail
        // fast with InvalidParameter.
        let dir = tempdir("pt");
        let back = FsBackend::new("disk", &dir).unwrap();
        let cases: &[&str] = &[
            "../escape",        // parent reference
            "../../etc/passwd", // nested parent
            "./secret",         // current-dir (reserved component)
            "",                 // empty
            "/absolute",        // leading slash
            "sub/foo",          // separator
            "back\\slash",      // Windows separator
            ".",                // reserved
            "..",               // reserved
            "foo\0bar",         // null byte
            "foo\nbar",         // control char
            "C:name",           // drive letter
        ];
        for bad in cases {
            assert!(
                back.register("c", *bad).is_err(),
                "register accepted illegal filename {:?}",
                bad
            );
            assert!(
                back.write("c", *bad, 1, &[0], &[vec![0.0]]).is_err(),
                "write accepted illegal filename {:?}",
                bad
            );
        }
        // The legitimate filename still works.
        back.register("c", "ok.bin").unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fs_pull_rejects_bad_magic() {
        let dir = tempdir("bad");
        let back = FsBackend::new("disk", &dir).unwrap();
        // Write a bogus file directly.
        let p = dir.join("bad.bin");
        std::fs::write(&p, b"NOTVECS\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0").unwrap();
        back.register("c", "bad.bin").unwrap();
        let err = back.pull_vectors("c").unwrap_err();
        match err {
            RuLakeError::InvalidParameter(m) => assert!(m.contains("magic"), "got: {m}"),
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
