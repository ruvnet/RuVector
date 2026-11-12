//! Magic numbers, alignment requirements, and size limits for the RVF format.
//!
//! # Mnemonics versus wire bytes
//!
//! The magic values below are numeric constants whose *big-endian* rendering
//! spells a human-readable mnemonic ("RVFS", "RVM0"). RVF v1 serializes every
//! multi-byte integer little-endian, so the bytes that actually appear on the
//! wire are the reverse of the mnemonic. Reading the mnemonic as if it were
//! the byte sequence is the single most common way to misparse an RVF file.
//!
//! Use [`SEGMENT_MAGIC_BYTES`] and [`ROOT_MANIFEST_MAGIC_BYTES`] whenever you
//! need to compare or emit raw bytes; never hand-write an ASCII literal.
//!
//! See ADR-009 (RVF Version 1 Wire Contract) for the normative statement.

/// Segment header magic. Mnemonic "RVFS" (big-endian rendering of the number).
///
/// On the wire this serializes little-endian as `53 46 56 52`. See
/// [`SEGMENT_MAGIC_BYTES`] for the exact byte sequence.
pub const SEGMENT_MAGIC: u32 = 0x5256_4653;

/// Root manifest magic. Mnemonic "RVM0" (big-endian rendering of the number).
///
/// On the wire this serializes little-endian as `30 4D 56 52`. See
/// [`ROOT_MANIFEST_MAGIC_BYTES`] for the exact byte sequence.
pub const ROOT_MANIFEST_MAGIC: u32 = 0x5256_4D30;

/// The exact four bytes a v1 segment header starts with: `53 46 56 52`.
///
/// Byte-slice comparisons against a segment header must use this constant, not
/// an ASCII literal such as `b"RVFS"`.
pub const SEGMENT_MAGIC_BYTES: [u8; 4] = SEGMENT_MAGIC.to_le_bytes();

/// The exact four bytes a v1 Level-0 root manifest starts with: `30 4D 56 52`.
///
/// Byte-slice comparisons against a root manifest must use this constant, not
/// an ASCII literal such as `b"RVM0"`.
pub const ROOT_MANIFEST_MAGIC_BYTES: [u8; 4] = ROOT_MANIFEST_MAGIC.to_le_bytes();

/// All segments must start at a 64-byte aligned boundary (AVX-512 / cache-line width).
pub const SEGMENT_ALIGNMENT: usize = 64;

/// The Level 0 root manifest is always exactly 4096 bytes (one OS page / disk sector).
pub const ROOT_MANIFEST_SIZE: usize = 4096;

/// Maximum payload size for a single segment (4 GiB).
pub const MAX_SEGMENT_PAYLOAD: u64 = 4 * 1024 * 1024 * 1024;

/// Size of the segment header in bytes.
pub const SEGMENT_HEADER_SIZE: usize = 64;

/// Current segment format version.
pub const SEGMENT_VERSION: u8 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_magic_has_stable_numeric_and_wire_forms() {
        // The numeric constant is frozen for format version 1.
        assert_eq!(SEGMENT_MAGIC, 0x5256_4653);
        // Little-endian serialization is what a reader actually sees.
        assert_eq!(SEGMENT_MAGIC_BYTES, [0x53, 0x46, 0x56, 0x52]);
        assert_eq!(SEGMENT_MAGIC.to_le_bytes(), SEGMENT_MAGIC_BYTES);
        // Round-tripping the wire bytes recovers the numeric constant.
        assert_eq!(u32::from_le_bytes(SEGMENT_MAGIC_BYTES), SEGMENT_MAGIC);
        // The mnemonic only appears under big-endian rendering; it is a naming
        // convention, not the wire encoding.
        assert_eq!(&SEGMENT_MAGIC.to_be_bytes(), b"RVFS");
        assert_ne!(&SEGMENT_MAGIC_BYTES, b"RVFS");
    }

    #[test]
    fn root_manifest_magic_has_stable_numeric_and_wire_forms() {
        assert_eq!(ROOT_MANIFEST_MAGIC, 0x5256_4D30);
        assert_eq!(ROOT_MANIFEST_MAGIC_BYTES, [0x30, 0x4D, 0x56, 0x52]);
        assert_eq!(ROOT_MANIFEST_MAGIC.to_le_bytes(), ROOT_MANIFEST_MAGIC_BYTES);
        assert_eq!(
            u32::from_le_bytes(ROOT_MANIFEST_MAGIC_BYTES),
            ROOT_MANIFEST_MAGIC
        );
        assert_eq!(&ROOT_MANIFEST_MAGIC.to_be_bytes(), b"RVM0");
        assert_ne!(&ROOT_MANIFEST_MAGIC_BYTES, b"RVM0");
    }

    #[test]
    fn alignment_is_power_of_two() {
        assert!(SEGMENT_ALIGNMENT.is_power_of_two());
    }

    #[test]
    fn max_payload_is_4gb() {
        assert_eq!(MAX_SEGMENT_PAYLOAD, 0x1_0000_0000);
    }
}
