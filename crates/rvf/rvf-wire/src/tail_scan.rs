//! Tail-scan algorithm for finding the latest manifest segment.
//!
//! This module is the **single tail scanner** for the RVF stack (ADR-354
//! item 4). `rvf-runtime` calls it; it does not carry its own scanner.
//!
//! An RVF file is discovered from its tail. A trailing Level-0 root manifest
//! page (4096 bytes, magic `RVM0`) is trimmed first, then the body is scanned
//! backward for a MANIFEST_SEG header.
//!
//! # Scan modes
//!
//! * [`ScanMode::Aligned`] visits only 64-byte boundaries. This is the wire
//!   dialect: every segment starts on a 64-byte boundary.
//! * [`ScanMode::ByteGranular`] visits every byte offset. It exists solely for
//!   the *legacy* dialect written by `rvf-runtime` before ADR-354, whose
//!   segments were not padded. It is a strict superset of `Aligned` and is
//!   weaker: any `RVFS` + MANIFEST byte pattern inside a payload is a
//!   candidate. Callers must select it deliberately and report the file as
//!   legacy-dialect.
//!
//! Neither mode authenticates anything. A scan result is a *claim* about where
//! the manifest is; authenticity comes from a signed root page that commits to
//! the manifest (see `rvf-runtime`'s `Strict` policy).

use crate::reader::read_segment_header;
use rvf_types::{
    ErrorCode, RvfError, SegmentHeader, SegmentType, ROOT_MANIFEST_MAGIC, ROOT_MANIFEST_SIZE,
    SEGMENT_ALIGNMENT, SEGMENT_HEADER_SIZE, SEGMENT_MAGIC, SEGMENT_MAGIC_BYTES, SEGMENT_VERSION,
};

/// Which offsets a tail scan visits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanMode {
    /// Wire dialect: only offsets that are a multiple of 64.
    Aligned,
    /// Legacy dialect: every byte offset. Weaker; see the module docs.
    ByteGranular,
}

/// Split a trailing Level-0 root manifest page off `data`.
///
/// Returns the segment area and, when the last 4096 bytes start with the
/// `RVM0` magic, the page. At least one segment header must fit ahead of the
/// page, so a file that is only a page is returned whole.
pub fn split_root_page(data: &[u8]) -> (&[u8], Option<&[u8]>) {
    if data.len() < ROOT_MANIFEST_SIZE + SEGMENT_HEADER_SIZE {
        return (data, None);
    }
    let start = data.len() - ROOT_MANIFEST_SIZE;
    let magic = u32::from_le_bytes([
        data[start],
        data[start + 1],
        data[start + 2],
        data[start + 3],
    ]);
    if magic == ROOT_MANIFEST_MAGIC {
        (&data[..start], Some(&data[start..]))
    } else {
        (data, None)
    }
}

/// Iterator over MANIFEST_SEG header candidates, highest offset first.
///
/// A candidate has the segment magic, the supported version, type
/// `MANIFEST_SEG`, and a header that parses. Payload extents are **not**
/// checked here: callers decide what "complete" means for their buffer.
///
/// For [`ScanMode::Aligned`], `data` must begin on a 64-byte file boundary or
/// the alignment it applies is not the file's.
pub struct ManifestCandidates<'a> {
    data: &'a [u8],
    mode: ScanMode,
    next: Option<usize>,
}

/// Iterate manifest header candidates in `data`, newest first.
pub fn manifest_candidates(data: &[u8], mode: ScanMode) -> ManifestCandidates<'_> {
    let next = if data.len() < SEGMENT_HEADER_SIZE {
        None
    } else {
        let last = data.len() - SEGMENT_HEADER_SIZE;
        Some(match mode {
            ScanMode::Aligned => last & !(SEGMENT_ALIGNMENT - 1),
            ScanMode::ByteGranular => last,
        })
    };
    ManifestCandidates { data, mode, next }
}

impl Iterator for ManifestCandidates<'_> {
    type Item = (usize, SegmentHeader);

    fn next(&mut self) -> Option<Self::Item> {
        let magic_first = SEGMENT_MAGIC_BYTES[0];
        loop {
            let pos = self.next?;
            let step = match self.mode {
                ScanMode::Aligned => SEGMENT_ALIGNMENT,
                ScanMode::ByteGranular => 1,
            };
            self.next = pos.checked_sub(step);
            // Quick rejection on the first magic byte before the full check.
            if self.data[pos] != magic_first {
                continue;
            }
            let window = &self.data[pos..];
            let magic = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
            if magic == SEGMENT_MAGIC
                && window[4] == SEGMENT_VERSION
                && window[5] == SegmentType::Manifest as u8
            {
                if let Ok(header) = read_segment_header(window) {
                    return Some((pos, header));
                }
            }
        }
    }
}

/// True when the candidate's declared extent lies within `len` bytes.
pub fn extent_fits(offset: usize, header: &SegmentHeader, len: usize) -> bool {
    usize::try_from(header.payload_length)
        .ok()
        .and_then(|p| p.checked_add(SEGMENT_HEADER_SIZE))
        .and_then(|t| offset.checked_add(t))
        .is_some_and(|end| end <= len)
}

/// Find the latest manifest segment in `data`, wire dialect (64-byte aligned).
///
/// A trailing root manifest page is trimmed first. The newest aligned
/// MANIFEST_SEG header whose declared extent lies inside `data` wins; no hash
/// or signature is checked.
///
/// Returns `(byte_offset, SegmentHeader)` of the manifest segment.
///
/// # Errors
///
/// - `ManifestNotFound` if no valid MANIFEST_SEG is found in the entire file.
/// - `TruncatedSegment` if the file is too short to contain any segment.
pub fn find_latest_manifest(data: &[u8]) -> Result<(usize, SegmentHeader), RvfError> {
    find_latest_manifest_with_mode(data, ScanMode::Aligned)
}

/// [`find_latest_manifest`] with an explicit [`ScanMode`].
pub fn find_latest_manifest_with_mode(
    data: &[u8],
    mode: ScanMode,
) -> Result<(usize, SegmentHeader), RvfError> {
    if data.len() < SEGMENT_HEADER_SIZE {
        return Err(RvfError::Code(ErrorCode::TruncatedSegment));
    }
    let (body, _page) = split_root_page(data);
    manifest_candidates(body, mode)
        .find(|(off, hdr)| extent_fits(*off, hdr, data.len()))
        .ok_or(RvfError::Code(ErrorCode::ManifestNotFound))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::write_segment;
    use rvf_types::SegmentFlags;

    fn make_manifest_segment(segment_id: u64, payload: &[u8]) -> Vec<u8> {
        write_segment(
            SegmentType::Manifest as u8,
            payload,
            SegmentFlags::empty(),
            segment_id,
        )
    }

    #[test]
    fn find_single_manifest() {
        let vec_seg = write_segment(
            SegmentType::Vec as u8,
            &[1u8; 100],
            SegmentFlags::empty(),
            0,
        );
        let manifest_payload = vec![0u8; 64];
        let manifest_seg = make_manifest_segment(1, &manifest_payload);
        let mut file = vec_seg.clone();
        let manifest_offset = file.len();
        file.extend_from_slice(&manifest_seg);

        let (offset, header) = find_latest_manifest(&file).unwrap();
        assert_eq!(offset, manifest_offset);
        assert_eq!(header.seg_type, SegmentType::Manifest as u8);
        assert_eq!(header.segment_id, 1);
    }

    #[test]
    fn find_latest_of_multiple_manifests() {
        let vec_seg = write_segment(SegmentType::Vec as u8, &[0u8; 32], SegmentFlags::empty(), 0);
        let m1 = make_manifest_segment(1, &[0u8; 32]);
        let m2 = make_manifest_segment(2, &[0u8; 32]);
        let mut file = vec_seg;
        file.extend_from_slice(&m1);
        let m2_offset = file.len();
        file.extend_from_slice(&m2);

        let (offset, header) = find_latest_manifest(&file).unwrap();
        assert_eq!(offset, m2_offset);
        assert_eq!(header.segment_id, 2);
    }

    #[test]
    fn no_manifest_returns_error() {
        let vec_seg = write_segment(
            SegmentType::Vec as u8,
            &[0u8; 100],
            SegmentFlags::empty(),
            0,
        );
        let result = find_latest_manifest(&vec_seg);
        assert!(result.is_err());
    }

    #[test]
    fn too_short_returns_error() {
        let result = find_latest_manifest(&[0u8; 10]);
        assert!(result.is_err());
    }

    fn root_page() -> Vec<u8> {
        let mut page = vec![0u8; ROOT_MANIFEST_SIZE];
        page[..4].copy_from_slice(&ROOT_MANIFEST_MAGIC.to_le_bytes());
        page
    }

    #[test]
    fn trailing_root_page_is_trimmed_not_scanned() {
        // The page follows the manifest segment as a separate region (the
        // layout rvf-forge-core writes). Plant a fake MANIFEST header inside
        // the page: it must not win.
        let m = make_manifest_segment(1, &[7u8; 40]);
        let mut file = m.clone();
        let mut page = root_page();
        let fake = make_manifest_segment(9, &[0u8; 8]);
        page[256..256 + fake.len()].copy_from_slice(&fake);
        file.extend_from_slice(&page);

        let (offset, header) = find_latest_manifest(&file).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(header.segment_id, 1);
        let (body, got) = split_root_page(&file);
        assert_eq!(body.len(), m.len());
        assert_eq!(got.map(<[u8]>::len), Some(ROOT_MANIFEST_SIZE));
    }

    #[test]
    fn page_inside_manifest_tail_still_finds_the_manifest() {
        // Spec layout: the page is the last 4096 bytes of the MANIFEST_SEG
        // payload, so the header's extent reaches past the trimmed body.
        let mut payload = vec![3u8; 128];
        payload.extend_from_slice(&root_page());
        let seg = write_segment(
            SegmentType::Manifest as u8,
            &payload,
            SegmentFlags::empty(),
            4,
        );
        let mut file = write_segment(SegmentType::Vec as u8, &[0u8; 8], SegmentFlags::empty(), 0);
        let m_off = file.len();
        file.extend_from_slice(&seg);
        // Padding after the payload would break "page is the last 4096 bytes",
        // so require the segment to end exactly at the file end.
        assert_eq!((128 + ROOT_MANIFEST_SIZE + 64) % 64, 0);
        let (offset, header) = find_latest_manifest(&file).unwrap();
        assert_eq!(offset, m_off);
        assert_eq!(header.segment_id, 4);
    }

    #[test]
    fn truncated_candidate_is_skipped() {
        let m1 = make_manifest_segment(1, &[0u8; 32]);
        let m2 = make_manifest_segment(2, &[0u8; 512]);
        let mut file = m1;
        file.extend_from_slice(&m2[..200]); // m2 cut mid-payload
        let (_, header) = find_latest_manifest(&file).unwrap();
        assert_eq!(header.segment_id, 1);
    }

    #[test]
    fn aligned_ignores_unaligned_header_byte_granular_finds_it() {
        let mut file = write_segment(SegmentType::Vec as u8, &[0u8; 8], SegmentFlags::empty(), 0);
        let real = file.len();
        file.extend_from_slice(&make_manifest_segment(1, &[0u8; 16]));
        // Unaligned manifest after it (legacy-style packing).
        file.extend_from_slice(&[0u8; 5]);
        let unaligned = file.len();
        assert_ne!(unaligned % SEGMENT_ALIGNMENT, 0);
        let mut raw = make_manifest_segment(2, &[0u8; 16]);
        raw.truncate(SEGMENT_HEADER_SIZE + 16); // legacy: no tail padding
        file.extend_from_slice(&raw);

        let (a, _) = find_latest_manifest_with_mode(&file, ScanMode::Aligned).unwrap();
        let (b, hb) = find_latest_manifest_with_mode(&file, ScanMode::ByteGranular).unwrap();
        assert_eq!(a, real);
        assert_eq!(b, unaligned);
        assert_eq!(hb.segment_id, 2);
    }

    #[test]
    fn aligned_candidates_are_a_subsequence_of_byte_granular() {
        let mut file = Vec::new();
        for id in 0..4u64 {
            file.extend_from_slice(&make_manifest_segment(id, &[id as u8; 70]));
            file.extend_from_slice(&[0xAA; 3]);
        }
        let aligned: Vec<_> = manifest_candidates(&file, ScanMode::Aligned)
            .map(|c| c.0)
            .collect();
        let byte: Vec<_> = manifest_candidates(&file, ScanMode::ByteGranular)
            .map(|c| c.0)
            .collect();
        assert!(!aligned.is_empty());
        assert!(aligned.iter().all(|a| byte.contains(a)));
        assert!(aligned.windows(2).all(|w| w[0] > w[1]));
    }
}
