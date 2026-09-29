//! The rvf-runtime 0.2.0 on-disk contract (what `rvf inspect/query` open),
//! re-implemented on bytes only — no `rvf-runtime`, no fs, no clock.
//!
//! * Segments are an unpadded 64-byte header + payload; `checksum_algo = 0`
//!   with the runtime's legacy hash (IEEE CRC32 rotated into four lanes).
//!   rvf-wire's XXH3 would be rejected by the runtime as a checksum mismatch.
//! * `VEC_SEG` payload: `dim u16 | count u32 | count × (id u64 | f32 LE × dim)`.
//! * `MANIFEST_SEG` payload: `epoch u32 | dim u16 | total u64 | seg_count u32 |
//!   profile u8 | metric u8 | 2 reserved | seg_count × (seg_id u64 | offset u64 |
//!   payload_len u64 | type u8) | del_count u32 | del_count × u64`, written last.
//! * Edge sidecar: a `PROFILE_SEG` (0x0B, never read by the runtime at boot)
//!   placed before each `VEC_SEG`, carrying the string ids and metadata of
//!   that segment's rows in order. `META_SEG` is avoided on purpose: the
//!   runtime decodes it and would report foreign payloads as damaged.

use crate::bytes::{put_str16, Reader};
use crate::types::{MAX_ID_BYTES, MAX_METADATA_BYTES};
use rvf_types::{SegmentHeader, SEGMENT_HEADER_SIZE, SEGMENT_MAGIC, SEGMENT_VERSION};

/// `VEC_SEG`.
pub const SEG_VEC: u8 = 0x01;
/// `MANIFEST_SEG`.
pub const SEG_MANIFEST: u8 = 0x05;
/// `PROFILE_SEG` (carries the edge id/metadata sidecar).
pub const SEG_PROFILE: u8 = 0x0B;
/// Header size.
pub const HEADER: usize = SEGMENT_HEADER_SIZE;
/// Largest segment payload the exporter writes and the importer accepts by default.
pub const MAX_SEGMENT_PAYLOAD: usize = 8 << 20;

const SIDECAR_MAGIC: &[u8; 4] = b"RVEI";
const SIDECAR_VERSION: u8 = 1;
const MANIFEST_FIXED: usize = 22;
const DIR_ENTRY: usize = 25;

/// Runtime legacy content hash (`crc32fast` IEEE CRC32, rotated 0/8/16/24).
pub fn legacy_content_hash(data: &[u8]) -> [u8; 16] {
    let crc = crc32fast::hash(data);
    let mut h = [0u8; 16];
    for i in 0..4 {
        h[i * 4..(i + 1) * 4].copy_from_slice(&crc.rotate_left(i as u32 * 8).to_le_bytes());
    }
    h
}

/// Runtime-compatible 64-byte header for `payload`.
pub(crate) fn header_bytes(seg_type: u8, seg_id: u64, payload: &[u8]) -> [u8; HEADER] {
    let mut b = [0u8; HEADER];
    b[0..4].copy_from_slice(&SEGMENT_MAGIC.to_le_bytes());
    b[4] = SEGMENT_VERSION;
    b[5] = seg_type;
    b[8..16].copy_from_slice(&seg_id.to_le_bytes());
    b[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    // timestamp 0, checksum_algo 0 (legacy), compression 0, reserved 0.
    b[0x28..0x38].copy_from_slice(&legacy_content_hash(payload));
    b
}

/// `true` if the header's content hash verifies under the runtime's legacy
/// hash (algo 0) or any rvf-wire algorithm. An all-zero ("unset") hash is
/// refused: imports never accept unverifiable segments.
pub(crate) fn payload_hash_ok(hdr: &SegmentHeader, payload: &[u8]) -> bool {
    if hdr.content_hash == [0u8; 16] {
        return false;
    }
    (hdr.checksum_algo == 0 && legacy_content_hash(payload) == hdr.content_hash)
        || rvf_wire::hash::verify_content_hash(hdr, payload)
}

/// Append one sidecar row.
pub(crate) fn put_sidecar_row(out: &mut Vec<u8>, id: &str, meta: Option<&str>) {
    put_str16(out, id);
    match meta {
        None => out.push(0),
        Some(m) => {
            out.push(1);
            put_str16(out, m);
        }
    }
}

/// Sidecar header (count patched at `[5..9]`).
pub(crate) fn begin_sidecar(out: &mut Vec<u8>) {
    out.extend_from_slice(SIDECAR_MAGIC);
    out.push(SIDECAR_VERSION);
    out.extend_from_slice(&0u32.to_le_bytes());
}

/// Patch the sidecar row count.
pub(crate) fn seal_sidecar(buf: &mut [u8], count: u32) {
    buf[5..9].copy_from_slice(&count.to_le_bytes());
}

/// `true` if a `PROFILE_SEG` payload is an edge sidecar.
pub(crate) fn is_sidecar(payload: &[u8]) -> bool {
    payload.starts_with(SIDECAR_MAGIC)
}

/// Sidecar header length (`magic | version | count`).
pub(crate) const SIDECAR_HEADER: usize = 9;

/// One sidecar row borrowed from the payload: `(id, metadata)`.
pub(crate) type SidecarRow<'a> = (&'a str, Option<&'a str>);

/// Read the sidecar row starting at `*pos` (an offset into `payload`) and
/// advance `*pos`. Allocates nothing.
pub(crate) fn sidecar_row<'a>(payload: &'a [u8], pos: &mut usize) -> Option<SidecarRow<'a>> {
    let mut r = Reader::new(payload.get(*pos..)?);
    let id = r.str16(MAX_ID_BYTES)?;
    let meta = match r.u8()? {
        0 => None,
        1 => Some(r.str16(MAX_METADATA_BYTES)?),
        _ => return None,
    };
    *pos += r.pos();
    Some((id, meta))
}

/// Strictly validate a sidecar **without allocating**: header, row count
/// equal to the paired `VEC_SEG`'s `count`, every row well-formed with a
/// non-empty id, no trailing bytes. Run before any row of the pair is used,
/// so a hostile sidecar costs O(1) memory whatever count it declares.
pub(crate) fn validate_sidecar(payload: &[u8], vec_count: u32) -> Result<(), &'static str> {
    let mut r = Reader::new(payload);
    if r.take(4) != Some(SIDECAR_MAGIC.as_slice()) || r.u8() != Some(SIDECAR_VERSION) {
        return Err("sidecar header");
    }
    if r.u32() != Some(vec_count) {
        return Err("sidecar count");
    }
    let mut pos = SIDECAR_HEADER;
    for _ in 0..vec_count {
        match sidecar_row(payload, &mut pos) {
            Some((id, _)) if !id.is_empty() => {}
            _ => return Err("sidecar row"),
        }
    }
    if pos != payload.len() {
        return Err("sidecar trailing bytes");
    }
    Ok(())
}

/// `VEC_SEG` header: `dim`, `count` (count patched at `[2..6]`).
pub(crate) fn begin_vec(out: &mut Vec<u8>, dim: u16) {
    out.extend_from_slice(&dim.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
}

/// Patch the `VEC_SEG` count.
pub(crate) fn seal_vec(buf: &mut [u8], count: u32) {
    buf[2..6].copy_from_slice(&count.to_le_bytes());
}

/// Parsed `VEC_SEG` shape: `(dim, count)`, with the exact length checked.
pub(crate) fn vec_shape(payload: &[u8]) -> Result<(u16, u32), &'static str> {
    let mut r = Reader::new(payload);
    let dim = r.u16().ok_or("vec header")?;
    let count = r.u32().ok_or("vec header")?;
    let per = 8u64 + u64::from(dim) * 4;
    let need = u64::from(count)
        .checked_mul(per)
        .and_then(|v| v.checked_add(6));
    if need != Some(payload.len() as u64) {
        return Err("vec length");
    }
    Ok((dim, count))
}

/// Directory entry of a runtime manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirEntry {
    /// Segment id.
    pub seg_id: u64,
    /// Absolute byte offset of the segment header.
    pub offset: u64,
    /// Payload length.
    pub payload_len: u64,
    /// Segment type.
    pub seg_type: u8,
}

/// A runtime manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RvfManifest {
    /// Store epoch.
    pub epoch: u32,
    /// Dimension.
    pub dim: u16,
    /// Live + deleted vectors counted by the runtime.
    pub total_vectors: u64,
    /// Profile id.
    pub profile: u8,
    /// Metric byte (see [`crate::Metric::rvf_id`]).
    pub metric_id: u8,
    /// Segment directory.
    pub dir: Vec<DirEntry>,
    /// Deleted vector ids.
    pub deleted: Vec<u64>,
}

impl RvfManifest {
    /// Encode (no FileIdentity trailer).
    pub fn encode(&self) -> Vec<u8> {
        let mut p = Vec::with_capacity(
            MANIFEST_FIXED + self.dir.len() * DIR_ENTRY + 4 + self.deleted.len() * 8,
        );
        p.extend_from_slice(&self.epoch.to_le_bytes());
        p.extend_from_slice(&self.dim.to_le_bytes());
        p.extend_from_slice(&self.total_vectors.to_le_bytes());
        p.extend_from_slice(&(self.dir.len() as u32).to_le_bytes());
        p.push(self.profile);
        p.push(self.metric_id);
        p.extend_from_slice(&[0, 0]);
        for e in &self.dir {
            p.extend_from_slice(&e.seg_id.to_le_bytes());
            p.extend_from_slice(&e.offset.to_le_bytes());
            p.extend_from_slice(&e.payload_len.to_le_bytes());
            p.push(e.seg_type);
        }
        p.extend_from_slice(&(self.deleted.len() as u32).to_le_bytes());
        for d in &self.deleted {
            p.extend_from_slice(&d.to_le_bytes());
        }
        p
    }

    /// Decode, bounding directory and deletion counts by the payload length
    /// and by `max_entries`. Trailing bytes (e.g. a FileIdentity) are allowed.
    pub fn decode(payload: &[u8], max_entries: usize) -> Result<RvfManifest, &'static str> {
        let mut r = Reader::new(payload);
        let bad = "manifest header";
        let epoch = r.u32().ok_or(bad)?;
        let dim = r.u16().ok_or(bad)?;
        let total_vectors = r.u64().ok_or(bad)?;
        let n = r.u32().ok_or(bad)? as usize;
        let profile = r.u8().ok_or(bad)?;
        let metric_id = r.u8().ok_or(bad)?;
        r.take(2).ok_or(bad)?;
        if n > max_entries || n.saturating_mul(DIR_ENTRY) > r.remaining() {
            return Err("manifest directory size");
        }
        let mut dir = Vec::with_capacity(n);
        for _ in 0..n {
            dir.push(DirEntry {
                seg_id: r.u64().ok_or(bad)?,
                offset: r.u64().ok_or(bad)?,
                payload_len: r.u64().ok_or(bad)?,
                seg_type: r.u8().ok_or(bad)?,
            });
        }
        let mut deleted = Vec::new();
        if r.remaining() >= 4 {
            let d = r.u32().ok_or(bad)? as usize;
            if d > max_entries || d.saturating_mul(8) > r.remaining() {
                return Err("manifest deletion size");
            }
            deleted.reserve(d);
            for _ in 0..d {
                deleted.push(r.u64().ok_or(bad)?);
            }
        }
        debug_assert!(r.pos() <= payload.len());
        Ok(RvfManifest {
            epoch,
            dim,
            total_vectors,
            profile,
            metric_id,
            dir,
            deleted,
        })
    }
}

/// Parse a raw 64-byte header (magic + version checked; hash not checked).
pub(crate) fn parse_header(b: &[u8]) -> Result<SegmentHeader, &'static str> {
    rvf_wire::read_segment_header(b).map_err(|_| "segment header")
}
