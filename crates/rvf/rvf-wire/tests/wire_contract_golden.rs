//! Golden byte vectors for the RVF v1 wire contract (ADR-009).
//!
//! These tests pin the exact bytes that v1 readers in the field already
//! accept. A diff here is not a test failure to be patched away: it means the
//! wire format changed, which requires a new format version, dual-version
//! readers, and fresh golden vectors for both versions.

use std::fs;

use rvf_types::{
    SegmentFlags, SegmentType, ROOT_MANIFEST_MAGIC_BYTES, ROOT_MANIFEST_SIZE, SEGMENT_ALIGNMENT,
    SEGMENT_HEADER_SIZE, SEGMENT_MAGIC_BYTES,
};
use rvf_wire::manifest_codec::{read_root_manifest, write_root_manifest, Level0Root};
use rvf_wire::writer::write_segment_with_algo;
use rvf_wire::{find_latest_manifest, write_segment};

/// Checksum algorithm 2: SHAKE-256 truncated to 128 bits.
const CHECKSUM_ALGO_SHAKE256: u8 = 2;

/// The canonical empty segment: a META segment with an empty payload, segment
/// id 0, no flags, and a SHAKE-256 content hash.
///
/// Bytes 0x28..0x38 are SHAKE-256 of the empty input truncated to 128 bits,
/// which is the NIST-published value `46b9dd2b0ba88d13233b3feb743eeb24`.
const GOLDEN_EMPTY_SHAKE256_SEGMENT: [u8; SEGMENT_HEADER_SIZE] = [
    // 0x00 magic (LE), 0x04 version, 0x05 seg_type (META), 0x06 flags
    0x53, 0x46, 0x56, 0x52, 0x01, 0x07, 0x00, 0x00, //
    // 0x08 segment_id
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //
    // 0x10 payload_length
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //
    // 0x18 timestamp_ns
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //
    // 0x20 checksum_algo, 0x21 compression, 0x22 reserved_0, 0x24 reserved_1
    0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //
    // 0x28 content_hash (16 bytes)
    0x46, 0xB9, 0xDD, 0x2B, 0x0B, 0xA8, 0x8D, 0x13, //
    0x23, 0x3B, 0x3F, 0xEB, 0x74, 0x3E, 0xEB, 0x24, //
    // 0x38 uncompressed_len, 0x3C alignment_pad
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// First eight bytes of a default Level-0 root manifest: the little-endian
/// root magic followed by `version = 1` and `flags = 0`.
const GOLDEN_ROOT_PREFIX: [u8; 8] = [0x30, 0x4D, 0x56, 0x52, 0x01, 0x00, 0x00, 0x00];

/// CRC32C over bytes 0x000..0xFFC of a default Level-0 root manifest.
const GOLDEN_ROOT_CRC32C: [u8; 4] = [0xFF, 0xDD, 0x18, 0x14];

#[test]
fn canonical_empty_segment_header_matches_golden_bytes() {
    let segment = write_segment_with_algo(
        SegmentType::Meta as u8,
        &[],
        SegmentFlags::empty(),
        0,
        CHECKSUM_ALGO_SHAKE256,
    );

    assert_eq!(
        segment.len(),
        SEGMENT_HEADER_SIZE,
        "an empty payload occupies exactly one 64-byte header and no padding"
    );
    assert_eq!(segment.as_slice(), GOLDEN_EMPTY_SHAKE256_SEGMENT.as_slice());
    assert_eq!(&segment[0..4], SEGMENT_MAGIC_BYTES.as_slice());
}

#[test]
fn default_root_manifest_matches_golden_prefix_and_checksum() {
    let root = write_root_manifest(&Level0Root::default());

    assert_eq!(root.len(), ROOT_MANIFEST_SIZE);
    assert_eq!(&root[0..8], GOLDEN_ROOT_PREFIX.as_slice());
    assert_eq!(&root[0..4], ROOT_MANIFEST_MAGIC_BYTES.as_slice());
    assert_eq!(&root[0xFFC..], GOLDEN_ROOT_CRC32C.as_slice());

    // Everything between the header prefix and the trailing checksum is zero
    // for a default root, so the golden prefix plus CRC pins the whole page.
    assert!(
        root[8..0xFFC].iter().all(|&b| b == 0),
        "default root manifest must be zero-filled between header and checksum"
    );
}

#[test]
fn root_manifest_is_discovered_from_the_tail_without_an_offset_zero_header() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("tail_discovery.rvf");

    // A leading data segment stands in for arbitrary prior content. Nothing
    // about it identifies the file as RVF at offset zero.
    let vec_segment = write_segment(
        SegmentType::Vec as u8,
        &[0xA5; 200],
        SegmentFlags::empty(),
        7,
    );

    // The manifest payload carries directory bytes followed by the Level-0
    // root, which must occupy the final ROOT_MANIFEST_SIZE bytes.
    let root = write_root_manifest(&Level0Root::default());
    let mut manifest_payload = vec![0x11u8; SEGMENT_ALIGNMENT];
    manifest_payload.extend_from_slice(&root);
    let manifest_segment = write_segment(
        SegmentType::Manifest as u8,
        &manifest_payload,
        SegmentFlags::empty(),
        8,
    );

    let manifest_offset = vec_segment.len();
    let mut file_bytes = vec_segment;
    file_bytes.extend_from_slice(&manifest_segment);
    fs::write(&path, &file_bytes).unwrap();

    let on_disk = fs::read(&path).unwrap();

    // Offset zero holds an ordinary segment, not a container header, and it is
    // not parseable as a root manifest.
    assert_eq!(&on_disk[0..4], SEGMENT_MAGIC_BYTES.as_slice());
    assert_ne!(&on_disk[0..4], ROOT_MANIFEST_MAGIC_BYTES.as_slice());
    assert!(read_root_manifest(&on_disk).is_err());

    // Discovery works purely from the tail.
    let (offset, header) = find_latest_manifest(&on_disk).unwrap();
    assert_eq!(offset, manifest_offset);
    assert_eq!(header.seg_type, SegmentType::Manifest as u8);
    assert_eq!(header.segment_id, 8);

    let root_start = on_disk.len() - ROOT_MANIFEST_SIZE;
    let decoded = read_root_manifest(&on_disk[root_start..]).unwrap();
    assert_eq!(decoded.version, 1);
    assert_eq!(
        decoded.root_checksum,
        u32::from_le_bytes(GOLDEN_ROOT_CRC32C)
    );
}
