//! Persistence for [`RabitqPlusIndex`] — seed-based reconstruction format.
//!
//! ## Why seed-based, not direct field serialization
//!
//! `RabitqPlusIndex` keeps its inner [`RabitqIndex`], `originals_flat`, and
//! `rerank_factor` behind private fields and does not expose getters for them.
//! Rather than widen that encapsulation just for a disk format, this module
//! relies on a stronger property already documented by the crate:
//!
//! > Deterministic: `(dim, seed, data)` triple → bit-identical rotation +
//! > index build + search output across runs.
//!
//! So we persist exactly what's needed to replay the build — `(dim, seed,
//! rerank_factor, items)` — and reconstruct on load via the public
//! [`RabitqPlusIndex::from_vectors_parallel`] constructor. This keeps the
//! on-disk format small (no 4·D² rotation matrix, no `n·n_words·8` packed
//! codes, no cos-LUT), fully portable across machines, and immune to drift in
//! the private layout.
//!
//! ## On-disk layout
//!
//! | Offset | Size (bytes) | Field |
//! |--------|--------------|-------|
//! | 0  | 8   | magic = `b"rbpx0001"` |
//! | 8  | 4   | version (u32 LE) |
//! | 12 | 4   | dim (u32 LE) |
//! | 16 | 8   | seed (u64 LE) |
//! | 24 | 4   | rerank_factor (u32 LE) |
//! | 28 | 4   | n (u32 LE) |
//! | 32 | n × (4 + dim·4) | entries: id (u32 LE), then dim f32 LE |
//!
//! All multi-byte integers and floats are little-endian.
//!
//! ## Bounds
//!
//! Load rejects files whose header claims:
//! - magic ≠ `b"rbpx0001"`,
//! - version > current (1),
//! - dim == 0 or dim > [`MAX_DIM`] (8192),
//! - n > [`MAX_N`] (100M),
//! - rerank_factor > [`MAX_RERANK_FACTOR`] (1024).

use std::io::{Read, Write};

use crate::error::{RabitqError, Result};
use crate::index::{AnnIndex, RabitqPlusIndex};

/// 8-byte magic prefix identifying the `rbpx` container, version stripe `0001`.
pub const MAGIC: &[u8; 8] = b"rbpx0001";
/// Current on-disk format version. Bumped on any layout change.
pub const VERSION: u32 = 1;

/// Upper bound on `dim` accepted by [`load_index`] — 8192 covers every
/// production embedding model we target and keeps header validation cheap.
pub const MAX_DIM: u32 = 8192;
/// Upper bound on `n` accepted by [`load_index`] — 100 M entries.
pub const MAX_N: u32 = 100_000_000;
/// Upper bound on `rerank_factor` accepted by [`load_index`].
pub const MAX_RERANK_FACTOR: u32 = 1024;

// ── internal helpers ────────────────────────────────────────────────────────

fn io_err(msg: impl Into<String>) -> RabitqError {
    RabitqError::InvalidParameter(msg.into())
}

fn write_all<W: Write>(w: &mut W, buf: &[u8]) -> Result<()> {
    w.write_all(buf).map_err(|e| io_err(format!("write: {e}")))
}

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<()> {
    r.read_exact(buf).map_err(|e| io_err(format!("read: {e}")))
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<()> {
    write_all(w, &v.to_le_bytes())
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> Result<()> {
    write_all(w, &v.to_le_bytes())
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> Result<()> {
    write_all(w, &v.to_le_bytes())
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    read_exact(r, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    read_exact(r, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut b = [0u8; 4];
    read_exact(r, &mut b)?;
    Ok(f32::from_le_bytes(b))
}

// ── public API ──────────────────────────────────────────────────────────────

/// Serialize a [`RabitqPlusIndex`] by persisting the inputs required to
/// deterministically rebuild it.
///
/// Because `RabitqPlusIndex` does not expose its inner rotation matrix or
/// `originals_flat` through the public API, the caller must supply the
/// `(seed, items)` pair that was used to build `idx` (typically via
/// [`RabitqPlusIndex::from_vectors_parallel`] or a sequence of `add` calls).
/// The `idx` argument is used to read `len()`, `dim()`, and `rerank_factor()`
/// for cross-checking against `items` — this catches drift between the
/// in-memory index and the `items` the caller thinks produced it before bad
/// bytes hit disk.
pub fn save_index<W: Write>(
    idx: &RabitqPlusIndex,
    seed: u64,
    items: &[(usize, Vec<f32>)],
    w: &mut W,
) -> Result<()> {
    let dim = idx.dim();
    let n = idx.len();
    let rerank_factor = idx.rerank_factor();

    // Cross-check the caller's inputs match the index they claim to represent.
    if items.len() != n {
        return Err(io_err(format!(
            "items.len()={} but index.len()={}",
            items.len(),
            n
        )));
    }
    for (i, (_, v)) in items.iter().enumerate() {
        if v.len() != dim {
            return Err(RabitqError::DimensionMismatch {
                expected: dim,
                actual: v.len(),
            })
            .map_err(|_| io_err(format!("item {i}: vector dim {} != {}", v.len(), dim)));
        }
    }

    // Bounds — keep the disk format inside the same caps load_index enforces.
    if dim == 0 || dim as u32 > MAX_DIM {
        return Err(io_err(format!("dim {dim} out of range (1..={MAX_DIM})")));
    }
    if n as u64 > MAX_N as u64 {
        return Err(io_err(format!("n {n} exceeds cap {MAX_N}")));
    }
    if rerank_factor as u32 > MAX_RERANK_FACTOR {
        return Err(io_err(format!(
            "rerank_factor {rerank_factor} exceeds cap {MAX_RERANK_FACTOR}"
        )));
    }

    // Header.
    write_all(w, MAGIC)?;
    write_u32(w, VERSION)?;
    write_u32(w, dim as u32)?;
    write_u64(w, seed)?;
    write_u32(w, rerank_factor as u32)?;
    write_u32(w, n as u32)?;

    // Payload.
    for (id, v) in items {
        // u32 id — upstream uses usize but RabitqIndex already stores u32
        // internally, so we inherit the same narrowing at the API boundary.
        if *id > u32::MAX as usize {
            return Err(io_err(format!("id {id} exceeds u32::MAX")));
        }
        write_u32(w, *id as u32)?;
        for &x in v {
            write_f32(w, x)?;
        }
    }
    Ok(())
}

/// Deserialize a [`RabitqPlusIndex`] previously written by [`save_index`].
///
/// The rotation matrix, binary codes, cos-LUT, and `last_word_mask` are all
/// rebuilt deterministically from `(dim, seed)` — no per-field round-trip is
/// needed and the reconstructed index is byte-identical to the saved one.
pub fn load_index<R: Read>(r: &mut R) -> Result<RabitqPlusIndex> {
    // Magic.
    let mut magic = [0u8; 8];
    read_exact(r, &mut magic)?;
    if &magic != MAGIC {
        return Err(io_err(format!(
            "bad magic: expected {:?}, got {:?}",
            MAGIC, &magic
        )));
    }

    // Version.
    let version = read_u32(r)?;
    if version > VERSION {
        return Err(io_err(format!(
            "unsupported version {version} (max {VERSION})"
        )));
    }

    // Header fields, each bounded.
    let dim = read_u32(r)?;
    if dim == 0 || dim > MAX_DIM {
        return Err(io_err(format!("dim {dim} out of range (1..={MAX_DIM})")));
    }
    let seed = read_u64(r)?;
    let rerank_factor = read_u32(r)?;
    if rerank_factor > MAX_RERANK_FACTOR {
        return Err(io_err(format!(
            "rerank_factor {rerank_factor} exceeds cap {MAX_RERANK_FACTOR}"
        )));
    }
    let n = read_u32(r)?;
    if n > MAX_N {
        return Err(io_err(format!("n {n} exceeds cap {MAX_N}")));
    }

    // Payload.
    let dim_usize = dim as usize;
    let mut items: Vec<(usize, Vec<f32>)> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let id = read_u32(r)? as usize;
        let mut v = Vec::with_capacity(dim_usize);
        for _ in 0..dim_usize {
            v.push(read_f32(r)?);
        }
        items.push((id, v));
    }

    // Deterministic rebuild — same (dim, seed, data) → bit-identical index.
    RabitqPlusIndex::from_vectors_parallel(dim_usize, seed, rerank_factor as usize, items)
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::AnnIndex;
    use rand::{Rng as _, SeedableRng as _};

    fn make_dataset(n: usize, d: usize, seed: u64) -> Vec<(usize, Vec<f32>)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                (i, v)
            })
            .collect()
    }

    #[test]
    fn serialize_roundtrip_preserves_search_results() {
        let d = 32;
        let n = 100;
        let seed = 1337u64;
        let rerank_factor = 3;

        let data = make_dataset(n, d, seed);

        let mut original = RabitqPlusIndex::new(d, seed, rerank_factor);
        for (id, v) in &data {
            original.add(*id, v.clone()).unwrap();
        }

        // Save.
        let mut buf: Vec<u8> = Vec::new();
        save_index(&original, seed, &data, &mut buf).unwrap();

        // Header size = 8 + 4 + 4 + 8 + 4 + 4 = 32 bytes, payload = n*(4 + d*4).
        assert_eq!(buf.len(), 32 + n * (4 + d * 4));

        // Load.
        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = load_index(&mut cursor).unwrap();

        assert_eq!(loaded.len(), n);
        assert_eq!(loaded.dim(), d);
        assert_eq!(loaded.rerank_factor(), rerank_factor);

        // External ids must survive the roundtrip byte-for-byte — the
        // loader re-applies the persisted ids rather than flattening to
        // `0..n`. This is the post-condition `ruvector-rulake`'s
        // `warm_from_dir` now relies on to reconstruct `pos_to_id`
        // without the dense-id workaround.
        assert_eq!(
            original.external_ids(),
            loaded.external_ids(),
            "external ids must be preserved through persist roundtrip",
        );

        // Run 10 queries and assert ids + scores match exactly.
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(7));
        let k = 5;
        for q_idx in 0..10 {
            let q: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            let a = original.search(&q, k).unwrap();
            let b = loaded.search(&q, k).unwrap();
            assert_eq!(a.len(), b.len(), "query {q_idx}: result count");
            for (ra, rb) in a.iter().zip(b.iter()) {
                assert_eq!(ra.id, rb.id, "query {q_idx}: id mismatch");
                // Scores come from exact f32 rerank over the same candidate set —
                // bit-identical rebuild means they must match exactly.
                assert_eq!(
                    ra.score.to_bits(),
                    rb.score.to_bits(),
                    "query {q_idx}: score bits differ ({} vs {})",
                    ra.score,
                    rb.score
                );
            }
        }
    }

    /// Non-dense external ids must survive `save_index` → `load_index`
    /// exactly. The dense `(0, 1, …, n-1)` ids in the main roundtrip
    /// test would mask a regression where the loader flattens ids to
    /// row positions (they happen to coincide at `0..n`). This test
    /// uses the arithmetic progression `13·i + 7` so no id equals its
    /// row index — any loader that stores `pos` instead of the
    /// persisted id is caught immediately.
    #[test]
    fn persist_preserves_non_dense_ids() {
        let d = 24;
        let n = 50;
        let seed = 20_260_423_u64;
        let rerank_factor = 4;

        // External ids: 7, 20, 33, 46, … — non-dense, strictly
        // monotonic but unequal to `0..n`. Largest is 13·49 + 7 = 644
        // so still fits comfortably in u32.
        let external_ids: Vec<usize> = (0..n).map(|i| i * 13 + 7).collect();

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let items: Vec<(usize, Vec<f32>)> = external_ids
            .iter()
            .map(|&id| {
                let v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                (id, v)
            })
            .collect();

        let mut original = RabitqPlusIndex::new(d, seed, rerank_factor);
        for (id, v) in &items {
            original.add(*id, v.clone()).unwrap();
        }

        // Pre-condition: the source index already carries the non-dense
        // ids via `add`. If this fails the bug is in `add`/`RabitqIndex`,
        // not persist.
        let expected_u32: Vec<u32> = external_ids.iter().map(|&id| id as u32).collect();
        assert_eq!(
            original.external_ids(),
            expected_u32.as_slice(),
            "source index dropped non-dense ids before persist",
        );

        // Save → load.
        let mut buf: Vec<u8> = Vec::new();
        save_index(&original, seed, &items, &mut buf).unwrap();
        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = load_index(&mut cursor).unwrap();

        // The contract this test defends: ids survive byte-for-byte.
        assert_eq!(
            loaded.external_ids(),
            expected_u32.as_slice(),
            "load_index flattened non-dense ids — regression of the \
             rulake warm_from_dir limitation",
        );
        // And the widening accessor must agree.
        let expected_u64: Vec<u64> = external_ids.iter().map(|&id| id as u64).collect();
        assert_eq!(loaded.ids_u64(), expected_u64);

        // Search results must quote the persisted ids, not row indices.
        // Running 5 queries keyed off the same seed guarantees the
        // check covers a spread of candidate sets.
        let mut qrng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(42));
        let k = 5;
        for q_idx in 0..5 {
            let q: Vec<f32> = (0..d).map(|_| qrng.gen::<f32>() * 2.0 - 1.0).collect();
            let a = original.search(&q, k).unwrap();
            let b = loaded.search(&q, k).unwrap();
            assert_eq!(a.len(), b.len(), "query {q_idx}: result count");
            for (ra, rb) in a.iter().zip(b.iter()) {
                assert_eq!(ra.id, rb.id, "query {q_idx}: id mismatch after load");
                // Every result id must be one of the persisted
                // external ids — never a row index from `0..n`.
                assert!(
                    external_ids.contains(&ra.id),
                    "query {q_idx}: returned id {} is not in the \
                     persisted id set (smells like a row index)",
                    ra.id,
                );
            }
        }
    }

    /// `RabitqPlusIndex` doesn't derive `Debug`, so `Result::unwrap_err()`
    /// is unavailable. This helper extracts the error without requiring
    /// `Debug` on `T`.
    fn expect_err(res: Result<RabitqPlusIndex>) -> RabitqError {
        match res {
            Ok(_) => panic!("expected load_index to reject the input"),
            Err(e) => e,
        }
    }

    #[test]
    fn reject_bad_magic() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"NOPEBAD!");
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        let mut cursor = std::io::Cursor::new(&buf);
        let err = expect_err(load_index(&mut cursor));
        let msg = format!("{err}");
        assert!(msg.contains("bad magic"), "got: {msg}");
    }

    #[test]
    fn reject_version_too_new() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(VERSION + 1).to_le_bytes());
        let mut cursor = std::io::Cursor::new(&buf);
        let err = expect_err(load_index(&mut cursor));
        let msg = format!("{err}");
        assert!(msg.contains("unsupported version"), "got: {msg}");
    }

    #[test]
    fn reject_oversize_fields() {
        // dim too large.
        {
            let mut buf: Vec<u8> = Vec::new();
            buf.extend_from_slice(MAGIC);
            buf.extend_from_slice(&VERSION.to_le_bytes());
            buf.extend_from_slice(&(MAX_DIM + 1).to_le_bytes());
            let mut cursor = std::io::Cursor::new(&buf);
            let err = expect_err(load_index(&mut cursor));
            let msg = format!("{err}");
            assert!(msg.contains("dim"), "got: {msg}");
        }
        // dim zero.
        {
            let mut buf: Vec<u8> = Vec::new();
            buf.extend_from_slice(MAGIC);
            buf.extend_from_slice(&VERSION.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            let mut cursor = std::io::Cursor::new(&buf);
            let err = expect_err(load_index(&mut cursor));
            let msg = format!("{err}");
            assert!(msg.contains("dim"), "got: {msg}");
        }
        // rerank_factor too large.
        {
            let mut buf: Vec<u8> = Vec::new();
            buf.extend_from_slice(MAGIC);
            buf.extend_from_slice(&VERSION.to_le_bytes());
            buf.extend_from_slice(&32u32.to_le_bytes()); // dim
            buf.extend_from_slice(&0u64.to_le_bytes()); // seed
            buf.extend_from_slice(&(MAX_RERANK_FACTOR + 1).to_le_bytes());
            let mut cursor = std::io::Cursor::new(&buf);
            let err = expect_err(load_index(&mut cursor));
            let msg = format!("{err}");
            assert!(msg.contains("rerank_factor"), "got: {msg}");
        }
        // n too large.
        {
            let mut buf: Vec<u8> = Vec::new();
            buf.extend_from_slice(MAGIC);
            buf.extend_from_slice(&VERSION.to_le_bytes());
            buf.extend_from_slice(&32u32.to_le_bytes()); // dim
            buf.extend_from_slice(&0u64.to_le_bytes()); // seed
            buf.extend_from_slice(&1u32.to_le_bytes()); // rerank_factor
            buf.extend_from_slice(&(MAX_N + 1).to_le_bytes());
            let mut cursor = std::io::Cursor::new(&buf);
            let err = expect_err(load_index(&mut cursor));
            let msg = format!("{err}");
            assert!(msg.contains("n "), "got: {msg}");
        }
    }
}
