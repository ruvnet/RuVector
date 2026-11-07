//! Segment parsing and inspection exports for WASM.

extern crate alloc;

use alloc::vec::Vec;
use rvf_types::constants::{SEGMENT_HEADER_SIZE, SEGMENT_MAGIC};

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

            // Skip past this segment. The length is untrusted: checked
            // conversion (u64 -> 32-bit usize on wasm32) and checked adds.
            let next = usize::try_from(payload_length)
                .ok()
                .and_then(|len| len.checked_add(SEGMENT_HEADER_SIZE))
                .and_then(|total| i.checked_add(total));
            if let Some(next) = next {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn header(payload_length: u64) -> Vec<u8> {
        let mut h = Vec::new();
        h.extend_from_slice(&SEGMENT_MAGIC.to_le_bytes());
        h.push(1);
        h.push(0x01);
        h.extend_from_slice(&[0, 0]);
        h.extend_from_slice(&7u64.to_le_bytes());
        h.extend_from_slice(&payload_length.to_le_bytes());
        h.resize(SEGMENT_HEADER_SIZE, 0);
        h
    }

    #[test]
    fn huge_payload_length_does_not_panic() {
        // `SEGMENT_HEADER_SIZE + payload_length as usize` overflowed.
        for len in [u64::MAX, u64::MAX - 10, u64::MAX - 63] {
            let mut buf = header(len);
            buf.extend_from_slice(&header(0));
            let segs = parse_segments(&buf);
            assert_eq!(segs[0].payload_length, len);
            // An unrepresentable length cannot be skipped, so the scan keeps
            // going and finds the well-formed segment after it.
            assert_eq!(segs.last().map(|s| s.offset), Some(SEGMENT_HEADER_SIZE));
        }
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
