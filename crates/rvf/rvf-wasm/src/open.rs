//! Decoding of VEC_SEG payloads from untrusted `.rvf` bytes (`rvf_store_open`).

extern crate alloc;

use crate::segment;
use alloc::vec::Vec;

/// A decoded vector: (id, components).
pub type Entry = (u64, Vec<f32>);

/// Why `rvf_store_open` refuses a buffer instead of allocating a handle.
#[derive(Debug, PartialEq, Eq)]
pub enum OpenError {
    /// A VEC segment is truncated, declares no dimension, or disagrees with
    /// the dimension already read. Skipping it would hide the damage.
    Malformed,
    /// No VEC segment declared a dimension. Callers must not invent one.
    Dimensionless,
}

/// Decode every VEC_SEG in `buf`.
///
/// Every returned entry has exactly the store dimension. A malformed or
/// dimensionless buffer is an error; it is not opened as an empty dim-1 store.
pub fn parse_vec_entries(buf: &[u8]) -> Result<(u32, Vec<Entry>), OpenError> {
    let segments = segment::parse_segments(buf);
    let mut dim: u32 = 0;
    let mut entries: Vec<Entry> = Vec::new();

    for seg in &segments {
        // SegmentType::Vec = 0x01
        if seg.seg_type != 0x01 {
            continue;
        }
        // payload_length is untrusted: on wasm32 `as usize` truncates and the
        // add can wrap below the start, so the slice below would trap.
        let payload_start = seg.offset + rvf_types::constants::SEGMENT_HEADER_SIZE;
        let payload_end = match usize::try_from(seg.payload_length)
            .ok()
            .and_then(|len| payload_start.checked_add(len))
        {
            Some(end) if end <= buf.len() && seg.payload_length >= 6 => end,
            _ => return Err(OpenError::Malformed),
        };
        let payload = &buf[payload_start..payload_end];

        // Same layout as the native writer (rvf-runtime write_vec_seg):
        // dimension u16 | vector_count u32 | (id u64, f32 * dimension)*.
        let seg_dim = u32::from(u16::from_le_bytes([payload[0], payload[1]]));
        let count = u32::from_le_bytes([payload[2], payload[3], payload[4], payload[5]]) as usize;
        // The store has one dimension; a segment that disagrees (or declares
        // none) would leave entries shorter than it, and query/export index
        // every entry by the store dimension.
        if seg_dim == 0 || (dim != 0 && seg_dim != dim) {
            return Err(OpenError::Malformed);
        }
        // seg_dim fits in u16, so this cannot overflow even on wasm32.
        let vec_bytes = seg_dim as usize * 4;
        dim = seg_dim;

        let mut offset = 6;
        for _ in 0..count {
            // offset <= payload.len() always holds here, so these cannot wrap.
            if payload.len() - offset < 8 {
                return Err(OpenError::Malformed);
            }
            let mut id_bytes = [0u8; 8];
            id_bytes.copy_from_slice(&payload[offset..offset + 8]);
            let id = u64::from_le_bytes(id_bytes);
            offset += 8;
            if payload.len() - offset < vec_bytes {
                return Err(OpenError::Malformed);
            }
            let mut vec_data = Vec::with_capacity(seg_dim as usize);
            for d in 0..seg_dim as usize {
                let at = offset + d * 4;
                vec_data.push(f32::from_le_bytes([
                    payload[at],
                    payload[at + 1],
                    payload[at + 2],
                    payload[at + 3],
                ]));
            }
            offset += vec_bytes;
            entries.push((id, vec_data));
        }
    }

    if dim == 0 {
        return Err(OpenError::Dimensionless);
    }
    Ok((dim, entries))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{VecEntry, WasmStore};
    use rvf_types::constants::{SEGMENT_HEADER_SIZE, SEGMENT_MAGIC};

    fn header(seg_type: u8, payload_length: u64) -> Vec<u8> {
        let mut h = Vec::new();
        h.extend_from_slice(&SEGMENT_MAGIC.to_le_bytes());
        h.push(1);
        h.push(seg_type);
        h.extend_from_slice(&[0, 0]);
        h.extend_from_slice(&1u64.to_le_bytes());
        h.extend_from_slice(&payload_length.to_le_bytes());
        h.resize(SEGMENT_HEADER_SIZE, 0);
        h
    }

    fn vec_seg(dim: u16, vecs: &[(u64, &[f32])]) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&dim.to_le_bytes());
        p.extend_from_slice(&(vecs.len() as u32).to_le_bytes());
        for (id, v) in vecs {
            p.extend_from_slice(&id.to_le_bytes());
            for x in *v {
                p.extend_from_slice(&x.to_le_bytes());
            }
        }
        let mut s = header(0x01, p.len() as u64);
        s.extend_from_slice(&p);
        s
    }

    #[test]
    fn round_trips_a_well_formed_segment() {
        let buf = vec_seg(2, &[(7, &[1.0, 2.0]), (9, &[3.0, 4.0])]);
        let (dim, entries) = parse_vec_entries(&buf).unwrap();
        assert_eq!(dim, 2);
        assert_eq!(
            entries,
            alloc::vec![(7, alloc::vec![1.0, 2.0]), (9, alloc::vec![3.0, 4.0])]
        );
    }

    #[test]
    fn huge_payload_length_does_not_panic() {
        // payload_length near u64::MAX made `payload_start + len` overflow and
        // the wrapped end fall below the start: `buf[start..end]` trapped.
        for len in [u64::MAX, u64::MAX - 40, u64::MAX - 63, 1u64 << 63] {
            let mut buf = header(0x01, len);
            buf.extend_from_slice(&[0u8; 32]);
            assert_eq!(parse_vec_entries(&buf), Err(OpenError::Malformed));
        }
    }

    #[test]
    fn mixed_dimension_segments_cannot_poison_the_store() {
        // First segment fixes dim = 4; a second one smuggles a 2-dim vector.
        let mut buf = vec_seg(4, &[(1, &[0.0, 0.0, 0.0, 1.0])]);
        buf.extend_from_slice(&vec_seg(2, &[(2, &[1.0, 1.0])]));
        assert_eq!(parse_vec_entries(&buf), Err(OpenError::Malformed));
    }

    #[test]
    fn export_never_writes_a_lossy_file() {
        // A u16 count wrote 65_536 live vectors as count = 0, so reopening the
        // exported file silently lost everything. The count is u32 now.
        let n = u16::MAX as usize + 1;
        let mut s = WasmStore::new(1, 0);
        for i in 0..n {
            s.entries.push(VecEntry {
                id: i as u64,
                data: alloc::vec![i as f32],
                deleted: false,
            });
        }
        let mut buf = alloc::vec![0u8; 64 + 6 + n * 12 + 64 + 44];
        let written = s.export(buf.as_mut_ptr(), buf.len() as u32);
        assert!(written > 0, "export failed: {written}");
        let (_, entries) = parse_vec_entries(&buf[..written as usize]).unwrap();
        assert_eq!(entries.len(), n, "export round-trip lost vectors");
    }

    #[test]
    fn export_refuses_a_dimension_the_format_cannot_hold() {
        let mut s = WasmStore::new(u32::from(u16::MAX) + 1, 0);
        s.entries.push(VecEntry {
            id: 1,
            data: alloc::vec![0.0; u16::MAX as usize + 1],
            deleted: false,
        });
        let mut buf = alloc::vec![0u8; 1 << 20];
        assert_eq!(s.export(buf.as_mut_ptr(), buf.len() as u32), -1);
    }

    #[test]
    fn opens_a_file_written_by_the_native_runtime() {
        extern crate std;
        use rvf_runtime::{RvfOptions, RvfStore};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("native.rvf");
        let vectors: [[f32; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let mut store = RvfStore::create(
            &path,
            RvfOptions {
                dimension: 3,
                ..Default::default()
            },
        )
        .unwrap();
        store.ingest_batch(&refs, &[10, 20], None).unwrap();
        store.close().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let (dim, entries) = parse_vec_entries(&bytes).unwrap();
        assert_eq!(dim, 3);
        assert_eq!(
            entries,
            alloc::vec![
                (10, alloc::vec![1.0, 2.0, 3.0]),
                (20, alloc::vec![4.0, 5.0, 6.0])
            ]
        );
    }

    #[test]
    fn zero_dimension_segment_cannot_poison_the_store() {
        // A dim=0 segment used to push zero-length vectors before a later
        // segment set dim, leaving entries shorter than the store dimension.
        let mut buf = vec_seg(0, &[(1, &[]), (2, &[])]);
        buf.extend_from_slice(&vec_seg(3, &[(3, &[1.0, 2.0, 3.0])]));
        assert_eq!(parse_vec_entries(&buf), Err(OpenError::Malformed));
        assert_eq!(parse_vec_entries(&[]), Err(OpenError::Dimensionless));
    }
}
