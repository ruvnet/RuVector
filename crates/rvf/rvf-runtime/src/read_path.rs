//! Progressive read logic for the RVF runtime.
//!
//! Boot sequence:
//! 1. Seek to EOF - 4096, parse Level 0 root manifest
//! 2. Extract hotset pointers, mmap hot segments
//! 3. Background: parse Level 1 -> full segment directory
//! 4. On-demand: load cold segments as queries need them

use crate::options::DistanceMetric;
use crate::write_path::{SEG_HASH_ENTRY_LEN, SEG_HASH_TRAILER_MARKER};
use rvf_types::{
    FileIdentity, SegmentHeader, SEGMENT_ALIGNMENT, SEGMENT_HEADER_SIZE, SEGMENT_MAGIC,
};
use rvf_wire::reader::read_segment_header;
use rvf_wire::tail_scan::{extent_fits, manifest_candidates, ScanMode};
use std::io::{self, Read, Seek, SeekFrom};

/// In-memory vector storage. The contiguous-slab implementation lives in
/// [`crate::vector_slab`]; re-exported here so existing consumers
/// (`store`, `index_path`, `rabitq_path`) keep their import path.
pub(crate) use crate::vector_slab::VectorData;

/// A parsed segment directory entry.
#[derive(Clone, Debug)]
pub(crate) struct SegDirEntry {
    pub seg_id: u64,
    pub offset: u64,
    pub payload_length: u64,
    pub seg_type: u8,
}

/// Parsed manifest data from the file.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct ParsedManifest {
    pub epoch: u32,
    pub dimension: u16,
    pub total_vectors: u64,
    pub profile_id: u8,
    /// Distance metric decoded from byte [19] of the manifest header.
    ///
    /// Stores written before this field existed have 0x00 there (reserved),
    /// which decodes as `DistanceMetric::L2` — the backward-compatible default.
    pub metric: DistanceMetric,
    pub segment_dir: Vec<SegDirEntry>,
    pub deleted_ids: Vec<u64>,
    pub file_identity: Option<FileIdentity>,
    /// Per-segment committed hashes (signed stores only): `(segment_id, hash)`.
    pub seg_hashes: Option<Vec<(u64, [u8; 16])>>,
}

/// A manifest located by the tail scan, with the evidence a policy check needs.
#[derive(Clone, Debug)]
pub(crate) struct LocatedManifest {
    pub manifest: ParsedManifest,
    /// File offset of the MANIFEST_SEG header.
    pub offset: u64,
    pub header: SegmentHeader,
    /// Raw manifest payload (hashed by the Strict policy check).
    pub payload: Vec<u8>,
    /// Scan mode that found it. `ByteGranular` means legacy dialect.
    pub mode: ScanMode,
    /// Under a signed page: the newest parseable manifest beyond the
    /// page-named one `(offset, epoch)`, which was NOT used.
    pub newer_ignored: Option<(u64, u32)>,
    /// Under a signed page: offset of the first byte after the signed manifest
    /// when anything at all (even unparseable bytes) follows it before the
    /// page. A correctly written signed file ends manifest, then page.
    pub unsigned_tail: Option<u64>,
}

/// Work done by one manifest scan. Exposed for the DoS regression test, which
/// bounds attempts and copied bytes rather than wall time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanStats {
    /// Manifest header candidates examined.
    pub candidates: u32,
    /// Candidates whose payload was handed to the manifest parser.
    pub parse_attempts: u32,
    /// Sum of payload lengths handed to the parser.
    pub payload_bytes_parsed: u64,
    /// Bytes copied into a fresh allocation (a winning payload; payloads read
    /// from the file because they extend past the window).
    pub bytes_copied: u64,
    /// Largest buffer allocated for a window or a payload read.
    pub max_buffer: u64,
}

/// Parse attempts per open. A hostile file of fake MANIFEST headers can offer
/// millions of candidates; a legitimate file has the real one first or has a
/// handful of torn leftovers newer than it.
const MAX_PARSE_ATTEMPTS: u32 = 256;
/// Payloads at most this large are cheap to reject and are not counted
/// toward [`MAX_PARSE_ATTEMPTS`] (they are still charged to the byte budget).
const HEAVY_PAYLOAD: u64 = 4096;
/// First window read from the tail, then fixed-size blocks going backwards.
const FIRST_BLOCK: u64 = 256 << 10;
const BLOCK: u64 = 8 << 20;

/// Extra knowledge the caller has before scanning.
#[derive(Clone, Copy, Default)]
pub(crate) struct ScanHint {
    /// Under `Strict`/`Paranoid` with a root page present: the offset the
    /// signed page commits to. Candidates older than it are never examined,
    /// and newer ones are examined only up to the attempt budget.
    pub expect_offset: Option<u64>,
}

fn align_up(x: u64) -> u64 {
    x.saturating_add(63) & !63
}

fn header_at<R: Read + Seek>(r: &mut R, off: u64, len: u64) -> io::Result<Option<SegmentHeader>> {
    if off
        .checked_add(SEGMENT_HEADER_SIZE as u64)
        .is_none_or(|e| e > len)
    {
        return Ok(None);
    }
    r.seek(SeekFrom::Start(off))?;
    let mut b = [0u8; SEGMENT_HEADER_SIZE];
    r.read_exact(&mut b)?;
    Ok(read_segment_header(&b).ok())
}

/// Choose the scan mode from the file's actual layout, not from one byte.
///
/// Walk the first headers. At the first segment whose header + payload does
/// not end on a 64-byte boundary, the next header sits either at the padded
/// boundary (wire dialect) or immediately after the payload (legacy). Exactly
/// one of the two positions holding a valid header decides it. Only when the
/// layout is undecidable does the first segment's `checksum_algo = 0`
/// legacy marker break the tie. An attacker who flips that unhashed byte on a
/// wire file therefore does not select the weaker byte-granular scan.
fn mode_for_file<R: Read + Seek>(reader: &mut R, file_size: u64) -> io::Result<ScanMode> {
    let first_algo0 = match header_at(reader, 0, file_size)? {
        Some(h) => h.checksum_algo == 0,
        None => return Ok(ScanMode::Aligned),
    };
    let mut off = 0u64;
    for _ in 0..8 {
        let Some(h) = header_at(reader, off, file_size)? else {
            break;
        };
        let unpadded = off
            .saturating_add(h.payload_length)
            .saturating_add(SEGMENT_HEADER_SIZE as u64);
        let padded = align_up(unpadded);
        if unpadded == padded {
            off = unpadded;
            continue;
        }
        let at_padded = header_at(reader, padded, file_size)?.is_some();
        let at_unpadded = header_at(reader, unpadded, file_size)?.is_some();
        return Ok(match (at_padded, at_unpadded) {
            (true, false) => ScanMode::Aligned,
            (false, true) => ScanMode::ByteGranular,
            // A valid-looking header at BOTH positions means bytes were planted
            // in the unhashed padding to force the weaker scan (F2-10): never
            // pick byte-granular on ambiguity.
            (true, true) => ScanMode::Aligned,
            _ if first_algo0 => ScanMode::ByteGranular,
            _ => ScanMode::Aligned,
        });
    }
    Ok(if first_algo0 {
        ScanMode::ByteGranular
    } else {
        ScanMode::Aligned
    })
}

/// End of the segment area: the file minus a trailing Level-0 root page.
fn body_end<R: Read + Seek>(reader: &mut R, file_size: u64) -> io::Result<u64> {
    let page = rvf_types::ROOT_MANIFEST_SIZE as u64;
    if file_size < page + SEGMENT_HEADER_SIZE as u64 {
        return Ok(file_size);
    }
    reader.seek(SeekFrom::Start(file_size - page))?;
    let mut m = [0u8; 4];
    reader.read_exact(&mut m)?;
    Ok(if m == rvf_types::ROOT_MANIFEST_MAGIC.to_le_bytes() {
        file_size - page
    } else {
        file_size
    })
}

/// Scan backwards from EOF to find and parse the latest valid manifest.
///
/// The scan itself is `rvf_wire::tail_scan` (the single tail scanner); this
/// function supplies bounded blocks of the file and the runtime's payload
/// parser. Wire-dialect files are scanned at 64-byte alignment; legacy files
/// (see [`mode_for_file`]) byte-granular, the documented fallback for unpadded
/// segments. If an aligned scan finds no manifest header anywhere (and only
/// then) the byte-granular scan is tried and the result reported as such; a
/// parse failure of an aligned candidate never triggers the fallback.
///
/// Cost is bounded for hostile files: at most [`MAX_PARSE_ATTEMPTS`] payloads
/// are parsed, their total size is capped by `max(16 MiB, file size)` after
/// the first, candidates are parsed in place from the block buffer (no copy
/// unless the parse succeeds), and no allocation exceeds one block or one
/// payload.
pub(crate) fn find_latest_manifest<R: Read + Seek>(
    reader: &mut R,
    hint: ScanHint,
    stats: &mut ScanStats,
) -> io::Result<Option<LocatedManifest>> {
    let file_size = reader.seek(SeekFrom::End(0))?;
    if file_size < SEGMENT_HEADER_SIZE as u64 {
        return Ok(None);
    }
    let end = body_end(reader, file_size)?;
    // Under a signature-checking policy the page names the manifest: read it
    // directly. Newer lookalikes (crash leftovers, planted headers) are not
    // consulted at all, so they cannot hide it or replace it (F2-07).
    if let Some(o) = hint.expect_offset {
        if let Some(mut m) = manifest_at(reader, o, file_size, end, stats)? {
            // Look (within the usual budgets) for a parseable manifest NEWER
            // than the signed one, so it can be reported instead of silently
            // ignored (F3-02).
            let mode = mode_for_file(reader, file_size)?;
            let newer_hint = ScanHint {
                expect_offset: Some(o + 1),
            };
            if let (Some(n), _, _) = scan_blocks(reader, file_size, end, mode, newer_hint, stats)? {
                m.newer_ignored = Some((n.offset, n.manifest.epoch));
            }
            let signed_end = o + rvf_wire::writer::calculate_padded_size(
                SEGMENT_HEADER_SIZE,
                m.header.payload_length as usize,
            ) as u64;
            if end > signed_end {
                m.unsigned_tail = Some(signed_end);
            }
            return Ok(Some(m));
        }
    }
    let mode = mode_for_file(reader, file_size)?;
    let (found, saw_any, exhausted) = scan_blocks(reader, file_size, end, mode, hint, stats)?;
    if found.is_some() || exhausted {
        return Ok(found);
    }
    if mode == ScanMode::Aligned && !saw_any && hint.expect_offset.is_none() {
        return Ok(scan_blocks(reader, file_size, end, ScanMode::ByteGranular, hint, stats)?.0);
    }
    Ok(None)
}

/// Classify the bytes after the manifest that open selected (bounded walk).
///
/// See [`crate::options::TailRecovery`]: open ignores them (ADR-280 crash
/// rule) but reports what it ignored.
pub(crate) fn inspect_tail<R: Read + Seek>(
    reader: &mut R,
    located: &LocatedManifest,
) -> io::Result<(crate::options::TailRecovery, Vec<u64>)> {
    let file_size = reader.seek(SeekFrom::End(0))?;
    let end = body_end(reader, file_size)?;
    let step = |off: u64, h: &SegmentHeader| {
        let unpadded = off
            .saturating_add(SEGMENT_HEADER_SIZE as u64)
            .saturating_add(h.payload_length);
        if located.mode == ScanMode::ByteGranular {
            unpadded
        } else {
            align_up(unpadded)
        }
    };
    let manifest_end = step(located.offset, &located.header).min(end);
    let mut info = crate::options::TailRecovery {
        bytes_after_manifest: end - manifest_end,
        ..Default::default()
    };
    let mut pos = manifest_end;
    let mut tail_witnesses: Vec<u64> = Vec::new();
    // Zero bytes are alignment padding; anything else that is not a whole
    // segment is a cut-off or garbage tail (checks the first 4 KiB only).
    let residual_nonzero = |reader: &mut R, from: u64| -> io::Result<bool> {
        let n = (end - from).min(4096) as usize;
        let mut buf = vec![0u8; n];
        reader.seek(SeekFrom::Start(from))?;
        reader.read_exact(&mut buf)?;
        Ok(buf.iter().any(|&b| b != 0))
    };
    // Bound the walk: a hostile tail of tiny segments must not cost O(file).
    for _ in 0..4096 {
        if pos >= end {
            break;
        }
        let avail = end - pos;
        if avail < SEGMENT_HEADER_SIZE as u64 {
            // Fewer bytes than a header: non-zero residue is a cut-off write.
            if residual_nonzero(reader, pos)? {
                info.torn_at = Some(pos);
            }
            break;
        }
        let Some(h) = header_at(reader, pos, end)? else {
            if residual_nonzero(reader, pos)? {
                info.torn_at = Some(pos);
            }
            break;
        };
        let payload_end = pos
            .saturating_add(SEGMENT_HEADER_SIZE as u64)
            .saturating_add(h.payload_length);
        if payload_end > end {
            info.torn_at = Some(pos);
            break;
        }
        if h.seg_type != rvf_types::SegmentType::Manifest as u8 {
            info.uncommitted_segments += 1;
        }
        if h.seg_type == rvf_types::SegmentType::Witness as u8 {
            tail_witnesses.push(pos);
        }
        pos = step(pos, &h);
    }
    Ok((info, tail_witnesses))
}

/// If the file starts with a segment header carrying our magic but a segment
/// version this reader does not support, return that version byte. Used to
/// report "unsupported / newer format" instead of a generic missing-manifest
/// or checksum failure (#1276 item 2).
pub(crate) fn unsupported_version_at_start<R: Read + Seek>(
    reader: &mut R,
) -> io::Result<Option<u8>> {
    let len = reader.seek(SeekFrom::End(0))?;
    if len < SEGMENT_HEADER_SIZE as u64 {
        return Ok(None);
    }
    reader.seek(SeekFrom::Start(0))?;
    let mut b = [0u8; 8];
    reader.read_exact(&mut b)?;
    let magic = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    Ok((magic == SEGMENT_MAGIC && b[4] != rvf_types::SEGMENT_VERSION).then_some(b[4]))
}

/// Read and parse the manifest segment at exactly `offset`, or `None`.
fn manifest_at<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    file_size: u64,
    body_end: u64,
    stats: &mut ScanStats,
) -> io::Result<Option<LocatedManifest>> {
    let Some(header) = header_at(reader, offset, file_size)? else {
        return Ok(None);
    };
    if header.seg_type != rvf_types::SegmentType::Manifest as u8
        || header.payload_length > MAX_READ_PAYLOAD
        || !offset.is_multiple_of(SEGMENT_ALIGNMENT as u64)
    {
        return Ok(None);
    }
    let end = offset
        .saturating_add(SEGMENT_HEADER_SIZE as u64)
        .saturating_add(header.payload_length);
    if end > body_end || end > file_size {
        return Ok(None);
    }
    let mut payload = vec![0u8; header.payload_length as usize];
    reader.seek(SeekFrom::Start(offset + SEGMENT_HEADER_SIZE as u64))?;
    stats.parse_attempts += 1;
    stats.payload_bytes_parsed += header.payload_length;
    stats.bytes_copied += header.payload_length;
    stats.max_buffer = stats.max_buffer.max(header.payload_length);
    reader.read_exact(&mut payload)?;
    Ok(
        parse_manifest_payload(&payload).map(|manifest| LocatedManifest {
            manifest,
            offset,
            header,
            payload,
            mode: ScanMode::Aligned,
            newer_ignored: None,
            unsigned_tail: None,
        }),
    )
}

/// Walk the segment area backwards in bounded blocks. Returns the manifest,
/// whether any header candidate was seen, and whether a budget stopped the scan.
fn scan_blocks<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    body_end: u64,
    mode: ScanMode,
    hint: ScanHint,
    stats: &mut ScanStats,
) -> io::Result<(Option<LocatedManifest>, bool, bool)> {
    let budget = file_size.max(16 << 20);
    let mut saw = false;
    // Candidate positions p satisfy p + 64 <= body_end.
    let mut hi = body_end.saturating_sub(SEGMENT_HEADER_SIZE as u64 - 1);
    let mut first = true;
    while hi > 0 {
        let size = if first { FIRST_BLOCK } else { BLOCK };
        first = false;
        let lo = hi.saturating_sub(size) & !(SEGMENT_ALIGNMENT as u64 - 1);
        let buf_end = (hi + SEGMENT_HEADER_SIZE as u64 - 1).min(body_end);
        reader.seek(SeekFrom::Start(lo))?;
        let mut buf = vec![0u8; (buf_end - lo) as usize];
        reader.read_exact(&mut buf)?;
        stats.max_buffer = stats.max_buffer.max(buf.len() as u64);
        for (i, header) in manifest_candidates(&buf, mode) {
            if i as u64 >= hi - lo {
                continue; // belongs to the newer block (header overlap)
            }
            saw = true;
            stats.candidates += 1;
            let file_offset = lo + i as u64;
            if hint.expect_offset.is_some_and(|e| file_offset < e) {
                return Ok((None, saw, true)); // older than the signed manifest
            }
            if header.payload_length > MAX_READ_PAYLOAD {
                continue;
            }
            let plen = header.payload_length;
            let payload_start = i + SEGMENT_HEADER_SIZE;
            let Some(payload_end) = payload_start.checked_add(plen as usize) else {
                continue;
            };
            if !extent_fits(
                usize::try_from(file_offset).unwrap_or(usize::MAX),
                &header,
                usize::try_from(file_size).unwrap_or(usize::MAX),
            ) {
                continue;
            }
            // Only candidates that cost a real payload parse count toward the
            // attempt cap, so a swarm of tiny fake headers cannot hide the
            // real manifest (F2-07); the byte budget still bounds the total.
            let heavy = plen > HEAVY_PAYLOAD;
            if (heavy && stats.parse_attempts >= MAX_PARSE_ATTEMPTS)
                || (stats.payload_bytes_parsed > 0 && stats.payload_bytes_parsed + plen > budget)
            {
                return Ok((None, saw, true));
            }
            if heavy {
                stats.parse_attempts += 1;
            }
            stats.payload_bytes_parsed += plen;
            let owned: Option<Vec<u8>>;
            let slice: &[u8] = if payload_end <= buf.len() {
                &buf[payload_start..payload_end]
            } else {
                // The payload runs past this block: read it from the file.
                reader.seek(SeekFrom::Start(file_offset + SEGMENT_HEADER_SIZE as u64))?;
                let mut p = vec![0u8; plen as usize];
                stats.max_buffer = stats.max_buffer.max(plen);
                stats.bytes_copied += plen;
                if reader.read_exact(&mut p).is_err() {
                    continue;
                }
                owned = Some(p);
                owned.as_deref().unwrap_or(&[])
            };
            if let Some(manifest) = parse_manifest_payload(slice) {
                let payload = slice.to_vec();
                stats.bytes_copied += payload.len() as u64;
                return Ok((
                    Some(LocatedManifest {
                        manifest,
                        offset: file_offset,
                        header,
                        payload,
                        mode,
                        newer_ignored: None,
                        unsigned_tail: None,
                    }),
                    saw,
                    false,
                ));
            }
        }
        hi = lo;
    }
    Ok((None, saw, false))
}

/// Locate the latest manifest in a complete in-memory file with the same
/// code the store uses at open. Returns `(offset, legacy)` where `legacy`
/// means the byte-granular mode found it.
///
/// Public only so the differential test and the fuzz target can compare the
/// runtime against `rvf-wire`'s scanner on identical bytes; not a stable API.
#[doc(hidden)]
pub fn locate_manifest_in_bytes(data: &[u8]) -> Option<(u64, bool)> {
    locate_manifest_with_stats(data).0
}

/// [`locate_manifest_in_bytes`] plus the scan's work counters.
#[doc(hidden)]
pub fn locate_manifest_with_stats(data: &[u8]) -> (Option<(u64, bool)>, ScanStats) {
    let mut stats = ScanStats::default();
    let mut cursor = io::Cursor::new(data);
    let r = find_latest_manifest(&mut cursor, ScanHint::default(), &mut stats)
        .ok()
        .flatten()
        .map(|m| (m.offset, m.mode == ScanMode::ByteGranular));
    (r, stats)
}

/// Which scan mode the store would pick for `data` (`true` = byte-granular).
#[doc(hidden)]
pub fn scan_is_legacy_for_bytes(data: &[u8]) -> bool {
    let mut cursor = io::Cursor::new(data);
    mode_for_file(&mut cursor, data.len() as u64)
        .map(|m| m == ScanMode::ByteGranular)
        .unwrap_or(false)
}

/// Whether `payload` parses as a runtime manifest. Test/fuzz support only.
#[doc(hidden)]
pub fn manifest_payload_parses(payload: &[u8]) -> bool {
    parse_manifest_payload(payload).is_some()
}

/// Parse a manifest payload into structured data.
fn parse_manifest_payload(payload: &[u8]) -> Option<ParsedManifest> {
    // Minimum header: epoch(4) + dim(2) + total_vectors(8) + seg_count(4) + profile(1) + pad(3) = 22
    if payload.len() < 22 {
        return None;
    }

    let epoch = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let dimension = u16::from_le_bytes([payload[4], payload[5]]);
    let total_vectors = u64::from_le_bytes([
        payload[6],
        payload[7],
        payload[8],
        payload[9],
        payload[10],
        payload[11],
        payload[12],
        payload[13],
    ]);
    let seg_count = u32::from_le_bytes([payload[14], payload[15], payload[16], payload[17]]);
    let profile_id = payload[18];
    // Byte [19] encodes the distance metric (was reserved zero in older stores).
    // DistanceMetric::from_id(0) == L2, so old stores boot correctly.
    let metric = DistanceMetric::from_id(payload[19]);

    let mut offset = 22; // past header (4+2+8+4+1+3)

    // Validate that seg_count does not exceed what the payload can actually hold.
    // Each directory entry is 25 bytes, so seg_count * 25 + 22 must fit in the payload.
    let max_possible_entries = payload.len().saturating_sub(22) / 25;
    if (seg_count as usize) > max_possible_entries {
        return None;
    }

    // Parse segment directory.
    let mut segment_dir = Vec::with_capacity(seg_count as usize);
    for _ in 0..seg_count {
        if offset + 25 > payload.len() {
            return None;
        }
        let seg_id = u64::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
            payload[offset + 4],
            payload[offset + 5],
            payload[offset + 6],
            payload[offset + 7],
        ]);
        let seg_offset = u64::from_le_bytes([
            payload[offset + 8],
            payload[offset + 9],
            payload[offset + 10],
            payload[offset + 11],
            payload[offset + 12],
            payload[offset + 13],
            payload[offset + 14],
            payload[offset + 15],
        ]);
        let plen = u64::from_le_bytes([
            payload[offset + 16],
            payload[offset + 17],
            payload[offset + 18],
            payload[offset + 19],
            payload[offset + 20],
            payload[offset + 21],
            payload[offset + 22],
            payload[offset + 23],
        ]);
        let stype = payload[offset + 24];
        segment_dir.push(SegDirEntry {
            seg_id,
            offset: seg_offset,
            payload_length: plen,
            seg_type: stype,
        });
        offset += 25;
    }

    // Parse deletion bitmap.
    let mut deleted_ids = Vec::new();
    if offset + 4 <= payload.len() {
        let del_count = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
        offset += 4;
        for _ in 0..del_count {
            if offset + 8 > payload.len() {
                break;
            }
            let did = u64::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
                payload[offset + 4],
                payload[offset + 5],
                payload[offset + 6],
                payload[offset + 7],
            ]);
            deleted_ids.push(did);
            offset += 8;
        }
    }

    // Try to parse FileIdentity trailer (backward-compatible).
    // Look for magic marker 0x46494449 ("FIDI") followed by 68 bytes.
    let file_identity = if offset + 4 + 68 <= payload.len() {
        let marker = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
        if marker == 0x4649_4449 {
            offset += 4;
            let fi_data: &[u8; 68] = payload[offset..offset + 68].try_into().ok()?;
            offset += 68;
            Some(FileIdentity::from_bytes(fi_data))
        } else {
            None
        }
    } else {
        None
    };

    // Optional per-segment hash trailer (signed stores). A present-but-
    // malformed trailer rejects the candidate rather than being ignored: an
    // ignored trailer would silently downgrade a signed store to unchecked.
    let mut seg_hashes = None;
    if offset + 8 <= payload.len() {
        let marker = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
        if marker == SEG_HASH_TRAILER_MARKER {
            let count = u32::from_le_bytes([
                payload[offset + 4],
                payload[offset + 5],
                payload[offset + 6],
                payload[offset + 7],
            ]) as usize;
            let body = &payload[offset + 8..];
            if count.checked_mul(SEG_HASH_ENTRY_LEN)? != body.len() {
                return None;
            }
            let mut entries = Vec::with_capacity(count);
            for chunk in body.chunks_exact(SEG_HASH_ENTRY_LEN) {
                let sid = u64::from_le_bytes(chunk[..8].try_into().ok()?);
                let mut h = [0u8; 16];
                h.copy_from_slice(&chunk[8..]);
                entries.push((sid, h));
            }
            seg_hashes = Some(entries);
        }
    }

    Some(ParsedManifest {
        epoch,
        dimension,
        total_vectors,
        profile_id,
        metric,
        segment_dir,
        deleted_ids,
        file_identity,
        seg_hashes,
    })
}

/// Read a VEC_SEG payload and return (id, vector) pairs.
pub(crate) fn read_vec_seg_payload(payload: &[u8]) -> Option<Vec<(u64, Vec<f32>)>> {
    if payload.len() < 6 {
        return None;
    }

    let dimension = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    let vector_count =
        u32::from_le_bytes([payload[2], payload[3], payload[4], payload[5]]) as usize;

    // Checked u64 arithmetic: on 32-bit targets (wasm32) a crafted
    // `vector_count` can wrap `vector_count * (8 + bytes_per_vec)` in
    // usize, pass this length check, and panic on out-of-bounds reads in
    // the loop below.
    let bytes_per_vec = (dimension as u64) * 4;
    let expected_size = (vector_count as u64)
        .checked_mul(8 + bytes_per_vec)
        .and_then(|v| v.checked_add(6))?;
    if (payload.len() as u64) < expected_size {
        return None;
    }

    let mut result = Vec::with_capacity(vector_count);
    let mut offset = 6;

    for _ in 0..vector_count {
        let vec_id = u64::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
            payload[offset + 4],
            payload[offset + 5],
            payload[offset + 6],
            payload[offset + 7],
        ]);
        offset += 8;

        let mut vec_data = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            let val = f32::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]);
            vec_data.push(val);
            offset += 4;
        }

        result.push((vec_id, vec_data));
    }

    Some(result)
}

/// Maximum allowed payload size when reading segments (256 MiB).
/// This prevents a malicious payload_length field from causing OOM.
const MAX_READ_PAYLOAD: u64 = 256 * 1024 * 1024;

/// Read a segment's payload from the file given its offset.
///
/// Validates magic, enforces a maximum payload size, and verifies the
/// content hash before returning the data.
pub(crate) fn read_segment_payload<R: Read + Seek>(
    reader: &mut R,
    seg_offset: u64,
) -> io::Result<(SegmentHeader, Vec<u8>)> {
    reader.seek(SeekFrom::Start(seg_offset))?;

    let mut hdr_buf = [0u8; SEGMENT_HEADER_SIZE];
    reader.read_exact(&mut hdr_buf)?;

    let magic = u32::from_le_bytes([hdr_buf[0], hdr_buf[1], hdr_buf[2], hdr_buf[3]]);
    if magic != SEGMENT_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid segment magic",
        ));
    }

    let payload_length = u64::from_le_bytes([
        hdr_buf[0x10],
        hdr_buf[0x11],
        hdr_buf[0x12],
        hdr_buf[0x13],
        hdr_buf[0x14],
        hdr_buf[0x15],
        hdr_buf[0x16],
        hdr_buf[0x17],
    ]);

    // Enforce maximum payload size to prevent OOM from crafted files.
    if payload_length > MAX_READ_PAYLOAD {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "segment payload too large: {} bytes (max {})",
                payload_length, MAX_READ_PAYLOAD
            ),
        ));
    }

    let header = SegmentHeader {
        magic,
        version: hdr_buf[0x04],
        seg_type: hdr_buf[0x05],
        flags: u16::from_le_bytes([hdr_buf[0x06], hdr_buf[0x07]]),
        segment_id: u64::from_le_bytes([
            hdr_buf[0x08],
            hdr_buf[0x09],
            hdr_buf[0x0A],
            hdr_buf[0x0B],
            hdr_buf[0x0C],
            hdr_buf[0x0D],
            hdr_buf[0x0E],
            hdr_buf[0x0F],
        ]),
        payload_length,
        timestamp_ns: u64::from_le_bytes([
            hdr_buf[0x18],
            hdr_buf[0x19],
            hdr_buf[0x1A],
            hdr_buf[0x1B],
            hdr_buf[0x1C],
            hdr_buf[0x1D],
            hdr_buf[0x1E],
            hdr_buf[0x1F],
        ]),
        checksum_algo: hdr_buf[0x20],
        compression: hdr_buf[0x21],
        reserved_0: u16::from_le_bytes([hdr_buf[0x22], hdr_buf[0x23]]),
        reserved_1: u32::from_le_bytes([
            hdr_buf[0x24],
            hdr_buf[0x25],
            hdr_buf[0x26],
            hdr_buf[0x27],
        ]),
        content_hash: {
            let mut h = [0u8; 16];
            h.copy_from_slice(&hdr_buf[0x28..0x38]);
            h
        },
        uncompressed_len: u32::from_le_bytes([
            hdr_buf[0x38],
            hdr_buf[0x39],
            hdr_buf[0x3A],
            hdr_buf[0x3B],
        ]),
        alignment_pad: u32::from_le_bytes([
            hdr_buf[0x3C],
            hdr_buf[0x3D],
            hdr_buf[0x3E],
            hdr_buf[0x3F],
        ]),
    };

    // payload_length is guaranteed <= MAX_READ_PAYLOAD (256 MiB) which fits in usize.
    let mut payload = vec![0u8; payload_length as usize];
    reader.read_exact(&mut payload)?;

    // Verify the content hash if it is non-zero (zero hash means "not set").
    // Dual-accept: the algorithm named by the header selects the dialect
    // (see `crate::hashing`); an unknown algorithm is refused.
    if header.content_hash != [0u8; 16]
        && !crate::hashing::verify_payload_hash(
            header.checksum_algo,
            &payload,
            &header.content_hash,
        )
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "segment content hash mismatch",
        ));
    }

    Ok((header, payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_manifest() {
        assert!(parse_manifest_payload(&[]).is_none());
        assert!(parse_manifest_payload(&[0u8; 10]).is_none());
    }

    #[test]
    fn vec_seg_round_trip() {
        // Build a VEC_SEG payload: dim=2, count=2, vectors.
        let dim: u16 = 2;
        let count: u32 = 2;
        let mut payload = Vec::new();
        payload.extend_from_slice(&dim.to_le_bytes());
        payload.extend_from_slice(&count.to_le_bytes());
        // Vector 0: id=10, [1.0, 2.0]
        payload.extend_from_slice(&10u64.to_le_bytes());
        payload.extend_from_slice(&1.0f32.to_le_bytes());
        payload.extend_from_slice(&2.0f32.to_le_bytes());
        // Vector 1: id=20, [3.0, 4.0]
        payload.extend_from_slice(&20u64.to_le_bytes());
        payload.extend_from_slice(&3.0f32.to_le_bytes());
        payload.extend_from_slice(&4.0f32.to_le_bytes());

        let result = read_vec_seg_payload(&payload).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 10);
        assert_eq!(result[0].1, vec![1.0, 2.0]);
        assert_eq!(result[1].0, 20);
        assert_eq!(result[1].1, vec![3.0, 4.0]);
    }

    #[test]
    fn vec_seg_rejects_oversized_counts_without_panicking() {
        // Maximal dimension and vector_count: the size product
        // (~1.1e15 bytes) wraps a 32-bit usize. The checked u64 size
        // computation must reject the payload (None), never pass the
        // length check and read out of bounds.
        let mut payload = Vec::new();
        payload.extend_from_slice(&u16::MAX.to_le_bytes()); // dimension
        payload.extend_from_slice(&u32::MAX.to_le_bytes()); // vector_count
        payload.extend_from_slice(&[0u8; 32]); // a little body, far too short
        assert!(read_vec_seg_payload(&payload).is_none());

        // A count that wraps usize-32 to a tiny value (dim 0 -> 8 bytes
        // per record; count * 8 wraps at 2^29 records on 32-bit).
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&(1u32 << 29).to_le_bytes());
        payload.extend_from_slice(&[0u8; 64]);
        assert!(read_vec_seg_payload(&payload).is_none());
    }
}
