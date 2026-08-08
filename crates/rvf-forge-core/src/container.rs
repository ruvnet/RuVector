//! Inspection-only walk over an RVF segment stream.
//!
//! This module is the single place that turns raw bytes into a segment
//! inventory, and it is deliberately the *only* place that touches offsets.
//! It upholds core invariant 2 (ADR-283 §2): the walk reads 64-byte headers
//! and records byte ranges. It never maps, links, interprets, or executes a
//! payload. Executable payload bytes leave this module only as `&[u8]` handed
//! to a hash or signature check.
//!
//! Layout of one segment on the wire:
//!
//! ```text
//! [ 64-byte header ][ payload_length bytes ][ signature footer? ][ zero pad ]
//! ```
//!
//! The footer is present exactly when the header's `SIGNED` flag is set, and
//! its own trailing `footer_length` field gives its size. The next segment
//! begins at the next 64-byte boundary.
//!
//! A complete RVF file is that segment stream followed by a fixed 4096-byte
//! Level-0 root manifest page at EOF (`rvf-manifest/src/level0.rs`), which is
//! not itself a segment and carries `RVM0` rather than the segment magic. The
//! walk therefore trims that page before it starts; see
//! [`split_root_manifest`]. Containers that are a bare segment stream — the
//! shape [`crate::testkit`] built before authoring existed — have no such page
//! and are walked whole.

use crate::error::{ForgeError, Result};
use rvf_types::{
    SegmentFlags, SegmentHeader, SegmentType, MAX_SEGMENT_PAYLOAD, ROOT_MANIFEST_MAGIC,
    ROOT_MANIFEST_SIZE, SEGMENT_ALIGNMENT, SEGMENT_HEADER_SIZE,
};
use std::ops::Range;

/// Upper bound on segments in one container. A malformed length field cannot
/// turn the walk into an unbounded loop; this makes that explicit as well.
const MAX_SEGMENTS: usize = 1 << 20;

/// Segment types whose payload is code the runtime would execute.
///
/// ADR-283 core invariant 3 and the RVM loading requirements (§4.3) treat
/// these as the segments that must be signed before a runtime may load them.
pub(crate) const EXECUTABLE_SEGMENT_TYPES: [SegmentType; 3] =
    [SegmentType::Kernel, SegmentType::Ebpf, SegmentType::Wasm];

/// True when `seg_type` names an executable segment.
pub(crate) fn is_executable(seg_type: u8) -> bool {
    SegmentType::try_from(seg_type).is_ok_and(|t| EXECUTABLE_SEGMENT_TYPES.contains(&t))
}

/// Stable, human-readable name for a segment type discriminator.
pub(crate) fn segment_type_name(seg_type: u8) -> String {
    match SegmentType::try_from(seg_type) {
        Ok(t) => format!("{t:?}").to_ascii_lowercase(),
        Err(raw) => format!("unknown(0x{raw:02x})"),
    }
}

/// One segment located in the byte stream. Holds ranges, never payload copies.
#[derive(Clone, Debug)]
pub(crate) struct ParsedSegment {
    /// Zero-based position in the stream.
    pub index: usize,
    /// Byte offset of the segment header.
    pub offset: usize,
    /// The parsed 64-byte header.
    pub header: SegmentHeader,
    /// Byte range of the payload within the container.
    pub payload: Range<usize>,
    /// Byte range of the signature footer, when the `SIGNED` flag is set.
    pub footer: Option<Range<usize>>,
}

impl ParsedSegment {
    /// Whether the header declares a signature footer.
    pub fn is_signed(&self) -> bool {
        SegmentFlags::from_raw(self.header.flags).contains(SegmentFlags::SIGNED)
    }

    /// Whether the header declares an encrypted payload.
    pub fn is_encrypted(&self) -> bool {
        SegmentFlags::from_raw(self.header.flags).contains(SegmentFlags::ENCRYPTED)
    }

    /// Whether this segment's payload is executable code.
    pub fn is_executable(&self) -> bool {
        is_executable(self.header.seg_type)
    }
}

/// Split a container into its segment area and its trailing Level-0 root
/// manifest page.
///
/// The page is identified by its `RVM0` magic at exactly `len - 4096`, which
/// is the only place the format permits it. Recognising it by magic rather
/// than by size alone means a bare segment stream that happens to be at least
/// 4096 bytes long is still walked whole.
pub(crate) fn split_root_manifest(data: &[u8]) -> (&[u8], Option<Range<usize>>) {
    // Require room for at least one segment ahead of the page: a file that is
    // nothing but a root manifest has no segments, and reporting that as
    // "holds no segments" is clearer than reporting a bad magic.
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
        (&data[..start], Some(start..data.len()))
    } else {
        (data, None)
    }
}

/// Walk `data` and return every segment in stream order.
///
/// A trailing Level-0 root manifest page is trimmed first and is not reported
/// as a segment, because it is not one.
///
/// # Errors
///
/// [`ForgeError::InvalidRvf`] if the stream is empty, truncated, declares an
/// implausible payload length, carries a malformed signature footer, or has
/// trailing bytes that are not a segment.
pub(crate) fn walk(data: &[u8]) -> Result<Vec<ParsedSegment>> {
    let (body, _) = split_root_manifest(data);
    walk_segments(body)
}

/// Walk a segment area that has already had any root manifest page trimmed.
fn walk_segments(data: &[u8]) -> Result<Vec<ParsedSegment>> {
    if data.len() < SEGMENT_HEADER_SIZE {
        return Err(ForgeError::invalid_rvf(format!(
            "container is {} bytes, shorter than one {SEGMENT_HEADER_SIZE}-byte segment header",
            data.len()
        )));
    }

    let mut segments = Vec::new();
    let mut offset = 0usize;

    while offset < data.len() {
        let remaining = data.len() - offset;

        // A short, all-zero tail is alignment padding, not a truncated segment.
        if remaining < SEGMENT_HEADER_SIZE {
            if data[offset..].iter().all(|&b| b == 0) {
                break;
            }
            return Err(ForgeError::invalid_rvf(format!(
                "{remaining} trailing bytes at offset {offset} are not a segment header"
            )));
        }

        if segments.len() >= MAX_SEGMENTS {
            return Err(ForgeError::invalid_rvf(format!(
                "container declares more than {MAX_SEGMENTS} segments"
            )));
        }

        let header = rvf_wire::read_segment_header(&data[offset..])
            .map_err(|e| ForgeError::invalid_rvf(format!("segment at offset {offset}: {e:?}")))?;

        if header.payload_length > MAX_SEGMENT_PAYLOAD {
            return Err(ForgeError::invalid_rvf(format!(
                "segment at offset {offset} declares a {}-byte payload, over the \
                 {MAX_SEGMENT_PAYLOAD}-byte maximum",
                header.payload_length
            )));
        }

        let payload_start = offset + SEGMENT_HEADER_SIZE;
        let payload_end = payload_start
            .checked_add(header.payload_length as usize)
            .ok_or_else(|| {
                ForgeError::invalid_rvf(format!(
                    "segment at offset {offset} has a payload length that overflows the address space"
                ))
            })?;
        if payload_end > data.len() {
            return Err(ForgeError::invalid_rvf(format!(
                "segment at offset {offset} declares {} payload bytes but only {} remain",
                header.payload_length,
                data.len() - payload_start
            )));
        }

        let mut end = payload_end;
        let mut footer = None;
        if SegmentFlags::from_raw(header.flags).contains(SegmentFlags::SIGNED) {
            let decoded =
                rvf_crypto::decode_signature_footer(&data[payload_end..]).map_err(|e| {
                    ForgeError::invalid_rvf(format!(
                        "segment at offset {offset} sets SIGNED but its footer is malformed: {e:?}"
                    ))
                })?;
            // A segment's zero padding decodes as a structurally valid footer
            // with sig_algo 0, sig_length 0, footer_length 0. Accepting that
            // would let any segment claim SIGNED by setting one flag bit and
            // changing nothing else, so require the footer to be internally
            // consistent and to actually carry a signature.
            if decoded.sig_length == 0 {
                return Err(ForgeError::invalid_rvf(format!(
                    "segment at offset {offset} sets SIGNED but its footer declares a \
                     zero-length signature"
                )));
            }
            let expected = rvf_types::SignatureFooter::compute_footer_length(decoded.sig_length);
            if decoded.footer_length != expected {
                return Err(ForgeError::invalid_rvf(format!(
                    "segment at offset {offset} has a footer declaring length {} for a \
                     {}-byte signature, which should be {expected}",
                    decoded.footer_length, decoded.sig_length
                )));
            }

            let footer_len = decoded.footer_length as usize;
            let footer_end = payload_end
                .checked_add(footer_len)
                .filter(|e| *e <= data.len());
            let footer_end = footer_end.ok_or_else(|| {
                ForgeError::invalid_rvf(format!(
                    "segment at offset {offset} declares a {footer_len}-byte signature footer \
                     that runs past the end of the container"
                ))
            })?;
            footer = Some(payload_end..footer_end);
            end = footer_end;
        }

        let next = align_up(end);
        if next <= offset {
            return Err(ForgeError::invalid_rvf(format!(
                "segment at offset {offset} does not advance the read cursor"
            )));
        }

        segments.push(ParsedSegment {
            index: segments.len(),
            offset,
            header,
            payload: payload_start..payload_end,
            footer,
        });

        offset = next.min(data.len());
        if next > data.len() {
            // The final segment's alignment padding was truncated. The segment
            // itself is intact, so this is not an error.
            break;
        }
    }

    if segments.is_empty() {
        return Err(ForgeError::invalid_rvf("container holds no segments"));
    }
    Ok(segments)
}

/// Round `n` up to the next 64-byte segment boundary.
fn align_up(n: usize) -> usize {
    n.div_ceil(SEGMENT_ALIGNMENT) * SEGMENT_ALIGNMENT
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvf_wire::write_segment;

    #[test]
    fn walks_a_two_segment_stream() {
        let mut data = write_segment(SegmentType::Meta as u8, b"first", SegmentFlags::empty(), 1);
        data.extend(write_segment(
            SegmentType::Vec as u8,
            b"second payload",
            SegmentFlags::empty(),
            2,
        ));

        let segs = walk(&data).unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].header.segment_id, 1);
        assert_eq!(&data[segs[0].payload.clone()], b"first");
        assert_eq!(segs[1].header.segment_id, 2);
        assert_eq!(&data[segs[1].payload.clone()], b"second payload");
        assert_eq!(segs[1].offset, SEGMENT_ALIGNMENT * 2);
    }

    #[test]
    fn rejects_a_stream_shorter_than_a_header() {
        let err = walk(&[0u8; 10]).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::InvalidRvf);
    }

    #[test]
    fn rejects_a_bad_magic() {
        let mut data = write_segment(SegmentType::Meta as u8, b"x", SegmentFlags::empty(), 1);
        data[0] ^= 0xff;
        assert_eq!(
            walk(&data).unwrap_err().code(),
            crate::ErrorCode::InvalidRvf
        );
    }

    #[test]
    fn rejects_a_payload_length_past_the_end() {
        let mut data = write_segment(SegmentType::Meta as u8, b"x", SegmentFlags::empty(), 1);
        // payload_length lives at header offset 0x10.
        data[0x10..0x18].copy_from_slice(&4096u64.to_le_bytes());
        let err = walk(&data).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::InvalidRvf);
        assert!(err.to_string().contains("only"), "{err}");
    }

    #[test]
    fn rejects_signed_flag_without_a_footer() {
        // `write_segment` pads with zeros, which decode as a structurally
        // valid but empty footer. Setting SIGNED must not be enough on its own.
        let data = write_segment(
            SegmentType::Wasm as u8,
            b"code",
            SegmentFlags::empty().with(SegmentFlags::SIGNED),
            1,
        );
        let err = walk(&data).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::InvalidRvf);
        assert!(err.to_string().contains("zero-length signature"), "{err}");
    }

    #[test]
    fn rejects_a_footer_whose_declared_length_is_inconsistent() {
        let kp = crate::testkit::TestKeypair::deterministic(3);
        let mut data = crate::testkit::signed_segment(SegmentType::Wasm, b"code", 1, &kp);
        // The footer's trailing footer_length is the last 4 bytes before pad.
        let footer_start = SEGMENT_HEADER_SIZE + 4;
        let len_at = footer_start + 4 + 64;
        data[len_at..len_at + 4].copy_from_slice(&999u32.to_le_bytes());

        let err = walk(&data).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::InvalidRvf);
        assert!(err.to_string().contains("should be 72"), "{err}");
    }

    #[test]
    fn accepts_a_well_formed_signed_segment() {
        let kp = crate::testkit::TestKeypair::deterministic(3);
        let data = crate::testkit::signed_segment(SegmentType::Wasm, b"code", 1, &kp);
        let segs = walk(&data).unwrap();
        assert_eq!(segs.len(), 1);
        assert!(segs[0].is_signed());
        let footer = segs[0].footer.clone().unwrap();
        assert_eq!(footer.len(), 72);
    }

    #[test]
    fn rejects_non_zero_trailing_bytes() {
        let mut data = write_segment(SegmentType::Meta as u8, b"x", SegmentFlags::empty(), 1);
        data.extend_from_slice(b"junk");
        let err = walk(&data).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::InvalidRvf);
        assert!(err.to_string().contains("trailing"), "{err}");
    }

    #[test]
    fn a_trailing_root_manifest_page_is_trimmed_not_walked() {
        let built = crate::author::ContainerBuilder::new()
            .segment(SegmentType::Vec, b"vectors".to_vec())
            .build()
            .unwrap();

        let (body, page) = split_root_manifest(&built.bytes);
        assert_eq!(
            page,
            Some(built.bytes.len() - ROOT_MANIFEST_SIZE..built.bytes.len())
        );
        assert_eq!(body.len(), built.bytes.len() - ROOT_MANIFEST_SIZE);

        // The page is not reported as a segment.
        let segs = walk(&built.bytes).unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[1].header.seg_type, SegmentType::Manifest as u8);
    }

    #[test]
    fn a_bare_segment_stream_has_no_root_manifest_page() {
        let data = write_segment(
            SegmentType::Meta as u8,
            &[0u8; 8192],
            SegmentFlags::empty(),
            1,
        );
        let (body, page) = split_root_manifest(&data);
        assert_eq!(page, None);
        assert_eq!(body.len(), data.len());
        assert_eq!(walk(&data).unwrap().len(), 1);
    }

    #[test]
    fn trailing_bytes_that_are_not_a_root_manifest_are_still_rejected() {
        let mut data = write_segment(SegmentType::Meta as u8, b"x", SegmentFlags::empty(), 1);
        data.extend(std::iter::repeat_n(0xEE, ROOT_MANIFEST_SIZE));
        let err = walk(&data).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::InvalidRvf);
    }

    #[test]
    fn executable_types_are_kernel_ebpf_and_wasm() {
        assert!(is_executable(SegmentType::Wasm as u8));
        assert!(is_executable(SegmentType::Kernel as u8));
        assert!(is_executable(SegmentType::Ebpf as u8));
        assert!(!is_executable(SegmentType::Vec as u8));
        assert!(!is_executable(SegmentType::Manifest as u8));
        assert!(!is_executable(0xEE));
    }

    #[test]
    fn segment_type_names_are_lowercase_and_stable() {
        assert_eq!(segment_type_name(SegmentType::Wasm as u8), "wasm");
        assert_eq!(segment_type_name(SegmentType::Manifest as u8), "manifest");
        assert_eq!(segment_type_name(0xEE), "unknown(0xee)");
    }

    #[test]
    fn align_up_rounds_to_64() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 64);
        assert_eq!(align_up(64), 64);
        assert_eq!(align_up(65), 128);
    }
}
