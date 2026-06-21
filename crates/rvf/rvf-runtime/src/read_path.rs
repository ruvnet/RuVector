//! Progressive read logic for the RVF runtime.
//!
//! Boot sequence:
//! 1. Seek to EOF - 4096, parse Level 0 root manifest
//! 2. Extract hotset pointers, mmap hot segments
//! 3. Background: parse Level 1 -> full segment directory
//! 4. On-demand: load cold segments as queries need them

use rvf_types::{FileIdentity, SegmentHeader, SegmentType, SEGMENT_HEADER_SIZE, SEGMENT_MAGIC};
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
    pub segment_dir: Vec<SegDirEntry>,
    pub deleted_ids: Vec<u64>,
    pub file_identity: Option<FileIdentity>,
}

/// Scan backwards from EOF to find and parse the latest valid manifest.
///
/// Reads a tail chunk and scans byte-by-byte for the magic + manifest-type
/// pattern, since segment headers are NOT necessarily 64-byte aligned from EOF.
pub(crate) fn find_latest_manifest<R: Read + Seek>(
    reader: &mut R,
) -> io::Result<Option<ParsedManifest>> {
    let file_size = reader.seek(SeekFrom::End(0))?;
    if file_size < SEGMENT_HEADER_SIZE as u64 {
        return Ok(None);
    }

    // The manifest grows with segment count, so it can extend arbitrarily far
    // back from EOF. Progressively widen the backward scan window (64 KB ->
    // 1 MB -> 16 MB -> whole file) until a valid manifest is found or the
    // file start is reached.
    const SCAN_WINDOWS: [u64; 4] = [64 << 10, 1 << 20, 16 << 20, u64::MAX];
    let mut prev_scan_size = 0u64;
    for window in SCAN_WINDOWS {
        let scan_size = window.min(file_size);
        if scan_size == prev_scan_size {
            break; // Already scanned the whole file.
        }
        if let Some(manifest) = scan_tail_for_manifest(reader, file_size, scan_size as usize)? {
            return Ok(Some(manifest));
        }
        prev_scan_size = scan_size;
    }

    Ok(None)
}

/// Scan the final `scan_size` bytes of the file for the latest valid manifest.
fn scan_tail_for_manifest<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    scan_size: usize,
) -> io::Result<Option<ParsedManifest>> {
    let scan_start = file_size - scan_size as u64;
    reader.seek(SeekFrom::Start(scan_start))?;
    let mut buf = vec![0u8; scan_size];
    reader.read_exact(&mut buf)?;

    let magic_bytes = SEGMENT_MAGIC.to_le_bytes();
    let manifest_type = SegmentType::Manifest as u8;

    // Scan backwards through the buffer looking for magic + manifest type.
    // We need at least SEGMENT_HEADER_SIZE bytes from the candidate position.
    if buf.len() < SEGMENT_HEADER_SIZE {
        return Ok(None);
    }

    let last_possible = buf.len() - SEGMENT_HEADER_SIZE;
    for i in (0..=last_possible).rev() {
        if buf[i..i + 4] == magic_bytes && buf[i + 5] == manifest_type {
            // Found a candidate manifest header at offset `i` within the buffer.
            let hdr_buf = &buf[i..i + SEGMENT_HEADER_SIZE];
            let payload_length_u64 = u64::from_le_bytes([
                hdr_buf[0x10],
                hdr_buf[0x11],
                hdr_buf[0x12],
                hdr_buf[0x13],
                hdr_buf[0x14],
                hdr_buf[0x15],
                hdr_buf[0x16],
                hdr_buf[0x17],
            ]);

            // Reject implausible payload lengths to prevent OOM.
            if payload_length_u64 > MAX_READ_PAYLOAD {
                continue;
            }
            let payload_length = payload_length_u64 as usize;

            let payload_start = i + SEGMENT_HEADER_SIZE;
            let payload_end = match payload_start.checked_add(payload_length) {
                Some(end) => end,
                None => continue, // overflow: skip this candidate
            };

            if payload_end <= buf.len() {
                // Payload is within our buffer — parse directly.
                if let Some(manifest) = parse_manifest_payload(&buf[payload_start..payload_end]) {
                    return Ok(Some(manifest));
                }
            } else {
                // Payload extends beyond our buffer — read from file.
                let file_offset = scan_start + i as u64 + SEGMENT_HEADER_SIZE as u64;
                reader.seek(SeekFrom::Start(file_offset))?;
                let mut payload = vec![0u8; payload_length];
                if reader.read_exact(&mut payload).is_ok() {
                    if let Some(manifest) = parse_manifest_payload(&payload) {
                        return Ok(Some(manifest));
                    }
                }
            }
        }
    }

    Ok(None)
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
            Some(FileIdentity::from_bytes(fi_data))
        } else {
            None
        }
    } else {
        None
    };

    Some(ParsedManifest {
        epoch,
        dimension,
        total_vectors,
        profile_id,
        segment_dir,
        deleted_ids,
        file_identity,
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

    // Verify content hash if it is non-zero (zero hash means "not set").
    if header.content_hash != [0u8; 16] {
        let computed = compute_content_hash(&payload);
        if computed != header.content_hash {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "segment content hash mismatch",
            ));
        }
    }

    Ok((header, payload))
}

/// Compute a 16-byte content hash matching the write path's algorithm.
/// Delegates to the single shared implementation in [`crate::hashing`].
fn compute_content_hash(data: &[u8]) -> [u8; 16] {
    crate::hashing::legacy_content_hash(data)
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
