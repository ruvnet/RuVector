//! Segment parsing and inspection exports for WASM.

extern crate alloc;

use alloc::vec::Vec;
use rvf_types::constants::{
    ROOT_MANIFEST_MAGIC_BYTES, ROOT_MANIFEST_SIZE, SEGMENT_HEADER_SIZE, SEGMENT_MAGIC,
    SEGMENT_MAGIC_BYTES, SEGMENT_VERSION,
};

use crate::store::VecEntry;

/// Parsed segment info for WASM consumers.
pub struct SegmentInfo {
    pub seg_id: u64,
    pub seg_type: u8,
    pub payload_length: u64,
    pub offset: usize,
}

/// Parse all segments from a raw .rvf byte buffer.
pub fn parse_segments(buf: &[u8]) -> Vec<SegmentInfo> {
    let mut segments = Vec::new();
    let magic_bytes = SEGMENT_MAGIC.to_le_bytes();

    if buf.len() < SEGMENT_HEADER_SIZE {
        return segments;
    }

    let mut i = 0;
    let last = buf.len().saturating_sub(SEGMENT_HEADER_SIZE);

    while i <= last {
        if buf[i..i + 4] == magic_bytes {
            let version = buf[i + 4];
            if version != 1 {
                i += 1;
                continue;
            }
            let seg_type = buf[i + 5];
            let seg_id = u64::from_le_bytes([
                buf[i + 8],
                buf[i + 9],
                buf[i + 10],
                buf[i + 11],
                buf[i + 12],
                buf[i + 13],
                buf[i + 14],
                buf[i + 15],
            ]);
            let payload_length = u64::from_le_bytes([
                buf[i + 16],
                buf[i + 17],
                buf[i + 18],
                buf[i + 19],
                buf[i + 20],
                buf[i + 21],
                buf[i + 22],
                buf[i + 23],
            ]);

            segments.push(SegmentInfo {
                seg_id,
                seg_type,
                payload_length,
                offset: i,
            });

            // Skip past this segment
            let total = SEGMENT_HEADER_SIZE + payload_length as usize;
            if let Some(next) = i.checked_add(total) {
                if next > i {
                    i = next;
                    continue;
                }
            }
        }
        i += 1;
    }

    segments
}

/// Decode a store only when every segment and every vector record is complete.
/// This intentionally does not scan for magic inside arbitrary input: a store
/// must start with a segment header (and may end with a Level-0 root manifest).
pub fn parse_store(buf: &[u8]) -> Option<(u32, Vec<VecEntry>)> {
    let mut offset = 0usize;
    let mut dimension = None;
    let mut entries = Vec::new();

    while offset < buf.len() {
        let remaining = &buf[offset..];
        if remaining.len() == ROOT_MANIFEST_SIZE
            && remaining.starts_with(&ROOT_MANIFEST_MAGIC_BYTES)
        {
            break;
        }
        if remaining.len() < SEGMENT_HEADER_SIZE
            || !remaining.starts_with(&SEGMENT_MAGIC_BYTES)
            || remaining[4] != SEGMENT_VERSION
        {
            return None;
        }

        let payload_len =
            usize::try_from(u64::from_le_bytes(remaining[16..24].try_into().ok()?)).ok()?;
        let payload_start = offset.checked_add(SEGMENT_HEADER_SIZE)?;
        let payload_end = payload_start.checked_add(payload_len)?;
        let padding = u32::from_le_bytes(remaining[60..64].try_into().ok()?) as usize;
        let next = payload_end.checked_add(padding)?;
        if next > buf.len() || buf[payload_end..next].iter().any(|&byte| byte != 0) {
            return None;
        }

        if remaining[5] == rvf_types::SegmentType::Vec as u8 {
            // A compressed Vec segment cannot be decoded as raw f32 records.
            let flags = u16::from_le_bytes([remaining[6], remaining[7]]);
            if remaining[33] != 0
                || flags & rvf_types::SegmentFlags::COMPRESSED != 0
                || payload_len < 6
            {
                return None;
            }
            let payload = &buf[payload_start..payload_end];
            let count = u16::from_le_bytes(payload[0..2].try_into().ok()?) as usize;
            let dim = u32::from_le_bytes(payload[2..6].try_into().ok()?);
            if dim == 0 || dim > i32::MAX as u32 || dimension.is_some_and(|d| d != dim) {
                return None;
            }
            let record_len = (dim as usize).checked_mul(4)?.checked_add(8)?;
            let expected = count.checked_mul(record_len)?.checked_add(6)?;
            if expected != payload_len {
                return None;
            }
            dimension = Some(dim);
            for record in payload[6..].chunks_exact(record_len) {
                let id = u64::from_le_bytes(record[0..8].try_into().ok()?);
                let mut data = Vec::with_capacity(dim as usize);
                for bytes in record[8..].chunks_exact(4) {
                    data.push(f32::from_le_bytes(bytes.try_into().ok()?));
                }
                entries.push(VecEntry {
                    id,
                    data,
                    deleted: false,
                });
            }
        }
        offset = next;
    }

    Some((dimension?, entries))
}

/// Parse a segment header from raw bytes.
/// Writes to out_ptr: [magic: u32, version: u8, type: u8, flags: u16, seg_id: u64, payload_len: u64]
/// = 24 bytes
pub fn parse_header_to_buf(buf: &[u8], out_ptr: *mut u8) -> i32 {
    if buf.len() < SEGMENT_HEADER_SIZE {
        return -1;
    }

    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != SEGMENT_MAGIC {
        return -2;
    }

    // Copy first 24 bytes of header (magic through payload_length)
    unsafe {
        for i in 0..24 {
            *out_ptr.add(i) = buf[i];
        }
    }
    0
}

/// Verify CRC32C of a buffer. Returns 1 if valid (matches expected), 0 if not.
pub fn verify_crc32c(buf: &[u8], expected: u32) -> i32 {
    let computed = crc32c_compute(buf);
    if computed == expected {
        1
    } else {
        0
    }
}

fn crc32c_compute(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0x82F6_3B78;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(seg_type: u8, payload: &[u8], padded: bool) -> Vec<u8> {
        let pad = if padded {
            (64 - payload.len() % 64) % 64
        } else {
            0
        };
        let mut bytes = alloc::vec![0; SEGMENT_HEADER_SIZE];
        bytes[0..4].copy_from_slice(&SEGMENT_MAGIC_BYTES);
        bytes[4] = SEGMENT_VERSION;
        bytes[5] = seg_type;
        bytes[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes[60..64].copy_from_slice(&(pad as u32).to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes.resize(bytes.len() + pad, 0);
        bytes
    }

    fn vec_payload(count: u16, dim: u32, records: &[(u64, &[f32])]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&count.to_le_bytes());
        payload.extend_from_slice(&dim.to_le_bytes());
        for (id, vector) in records {
            payload.extend_from_slice(&id.to_le_bytes());
            for value in *vector {
                payload.extend_from_slice(&value.to_le_bytes());
            }
        }
        payload
    }

    #[test]
    fn opens_complete_vec_records_with_correct_count_and_dimension() {
        let payload = vec_payload(2, 3, &[(7, &[1.0, 2.0, 3.0]), (9, &[4.0, 5.0, 6.0])]);
        let mut bytes = segment(1, &payload, true);
        bytes.extend(segment(4, &[1, 2, 3], true));
        let (dim, entries) = parse_store(&bytes).expect("valid store");
        assert_eq!(dim, 3);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, 7);
        assert_eq!(entries[0].data, [1.0, 2.0, 3.0]);
        assert_eq!(entries[1].id, 9);
    }

    #[test]
    fn rejects_noise_missing_vec_and_embedded_magic() {
        assert!(parse_store(&[0x41; 128]).is_none());
        assert!(parse_store(&segment(4, &[1, 2, 3], true)).is_none());
        let mut prefixed = alloc::vec![0x41; 64];
        prefixed.extend(segment(1, &vec_payload(0, 2, &[]), true));
        assert!(parse_store(&prefixed).is_none());
    }

    #[test]
    fn rejects_truncated_or_inconsistent_vectors() {
        let payload = vec_payload(2, 3, &[(7, &[1.0, 2.0, 3.0])]);
        assert!(parse_store(&segment(1, &payload, false)).is_none());
        let mut valid = segment(1, &vec_payload(1, 2, &[(7, &[1.0, 2.0])]), true);
        valid.pop();
        assert!(parse_store(&valid).is_none());
        assert!(parse_store(&segment(1, &vec_payload(0, 0, &[]), false)).is_none());
    }

    #[test]
    fn rejects_mismatched_dimensions_and_nonzero_padding() {
        let mut bytes = segment(1, &vec_payload(0, 2, &[]), true);
        bytes.extend(segment(1, &vec_payload(0, 3, &[]), false));
        assert!(parse_store(&bytes).is_none());

        let mut padded = segment(1, &vec_payload(0, 2, &[]), true);
        *padded.last_mut().unwrap() = 1;
        assert!(parse_store(&padded).is_none());
    }

    #[test]
    fn accepts_unpadded_store_export_and_rejects_compressed_vec() {
        let payload = vec_payload(1, 2, &[(7, &[1.0, 2.0])]);
        let mut exported = segment(1, &payload, false);
        exported.extend(segment(4, &[1, 2, 3], false));
        assert_eq!(parse_store(&exported).unwrap().1.len(), 1);

        let mut compressed = segment(1, &payload, false);
        compressed[6] = rvf_types::SegmentFlags::COMPRESSED as u8;
        assert!(parse_store(&compressed).is_none());
    }
}
