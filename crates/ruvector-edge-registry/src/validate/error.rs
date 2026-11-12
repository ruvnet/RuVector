//! Typed validation failures. Every variant is a `400`-class refusal of the
//! uploaded bytes except [`ValidationError::TooLarge`] (`413`); none of them
//! echo payload bytes.

use thiserror::Error;

/// Why an RVF byte stream was refused. Offsets are absolute byte offsets in
/// the uploaded object.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ValidationError {
    /// Zero bytes were supplied.
    #[error("empty upload")]
    Empty,
    /// The stream (or a declared segment) exceeds `max_total_bytes`.
    #[error("upload exceeds the {limit}-byte limit")]
    TooLarge {
        /// The configured limit.
        limit: u64,
    },
    /// More segments than `max_segments` (segment bomb).
    #[error("more than {limit} segments")]
    TooManySegments {
        /// The configured limit.
        limit: u32,
    },
    /// The stream ended inside a structure.
    #[error("truncated {what} at offset {offset}")]
    Truncated {
        /// Offset of the structure that was cut short.
        offset: u64,
        /// Which structure.
        what: &'static str,
    },
    /// A segment boundary does not start with the segment (or root) magic.
    #[error("bad magic at offset {offset}")]
    BadMagic {
        /// Offset of the bad header.
        offset: u64,
    },
    /// Unsupported segment format version.
    #[error("unsupported segment version at offset {offset}")]
    BadVersion {
        /// Offset of the header.
        offset: u64,
    },
    /// The segment type is reserved, zero or unknown.
    #[error("unknown segment type 0x{seg_type:02x} at offset {offset}")]
    UnknownSegmentType {
        /// Offset of the header.
        offset: u64,
        /// The raw type byte.
        seg_type: u8,
    },
    /// Header flag bits outside the known mask.
    #[error("unknown flag bits at offset {offset}")]
    UnknownFlags {
        /// Offset of the header.
        offset: u64,
    },
    /// A reserved header field is non-zero.
    #[error("reserved header field {field} is non-zero at offset {offset}")]
    ReservedNonZero {
        /// Offset of the header.
        offset: u64,
        /// Field name.
        field: &'static str,
    },
    /// Compressed segments are refused (decompression-bomb surface).
    #[error("compressed segment at offset {offset} is not accepted")]
    UnsupportedCompression {
        /// Offset of the header.
        offset: u64,
    },
    /// Checksum algorithm is reserved (3) or unknown.
    #[error("unsupported checksum algorithm {algo} at offset {offset}")]
    UnsupportedChecksum {
        /// Offset of the header.
        offset: u64,
        /// The algorithm byte.
        algo: u8,
    },
    /// A declared payload is larger than `max_segment_payload`.
    #[error("segment at offset {offset} declares {len} payload bytes (limit {limit})")]
    SegmentTooLarge {
        /// Offset of the header.
        offset: u64,
        /// Declared payload length.
        len: u64,
        /// The configured limit.
        limit: u64,
    },
    /// The payload does not match the header's content hash.
    #[error("content hash mismatch for segment at offset {offset}")]
    ChecksumMismatch {
        /// Offset of the header.
        offset: u64,
    },
    /// `alignment_pad` is out of range or the padding is not zero.
    #[error("bad alignment padding for segment at offset {offset}")]
    BadPadding {
        /// Offset of the header.
        offset: u64,
    },
    /// A `SIGNED` segment's footer is malformed.
    #[error("malformed signature footer for segment at offset {offset}")]
    BadFooter {
        /// Offset of the header.
        offset: u64,
    },
    /// Executable segments (kernel, eBPF, WASM) are refused by policy.
    #[error("executable segment at offset {offset} is not accepted")]
    ExecutableSegment {
        /// Offset of the header.
        offset: u64,
    },
    /// A manifest segment payload exceeds `max_manifest_payload`.
    #[error("manifest segment at offset {offset} exceeds {limit} bytes")]
    ManifestTooLarge {
        /// Offset of the header.
        offset: u64,
        /// The configured limit.
        limit: u32,
    },
    /// A manifest segment payload does not parse.
    #[error("malformed manifest at offset {offset}: {reason}")]
    MalformedManifest {
        /// Offset of the manifest segment.
        offset: u64,
        /// What was wrong.
        reason: &'static str,
    },
    /// A manifest directory entry names no segment seen before it.
    #[error("manifest at offset {offset} references a missing segment at {target}")]
    DirectoryMismatch {
        /// Offset of the manifest segment.
        offset: u64,
        /// The directory entry's segment offset.
        target: u64,
    },
    /// A vector segment payload does not match its declared layout.
    #[error("malformed vector segment at offset {offset}: {reason}")]
    VecSegMalformed {
        /// Offset of the segment.
        offset: u64,
        /// What was wrong.
        reason: &'static str,
    },
    /// Stream contains no segment at all.
    #[error("no segments")]
    NoSegments,
    /// Stream contains no manifest segment.
    #[error("no manifest segment")]
    NoManifest,
    /// Dimension is zero or above `max_dimension`.
    #[error("dimension {dim} out of range")]
    BadDimension {
        /// The declared dimension.
        dim: u16,
    },
    /// A vector segment or root page disagrees with the manifest dimension.
    #[error("dimension mismatch: manifest {expected}, found {found}")]
    DimensionMismatch {
        /// Manifest dimension.
        expected: u16,
        /// Conflicting dimension.
        found: u16,
    },
    /// The metric byte is not 0 (L2), 1 (inner product) or 2 (cosine).
    #[error("unknown metric id {0}")]
    UnknownMetric(u8),
    /// The trailing Level-0 root page failed its magic or CRC check.
    #[error("invalid root manifest page at offset {offset}")]
    RootManifestInvalid {
        /// Offset of the page.
        offset: u64,
    },
    /// Bytes after the Level-0 root page.
    #[error("trailing bytes at offset {offset}")]
    TrailingBytes {
        /// First trailing byte.
        offset: u64,
    },
}

impl ValidationError {
    /// `413` for size limits, `400` otherwise.
    pub fn http_status(&self) -> u16 {
        match self {
            ValidationError::TooLarge { .. } | ValidationError::SegmentTooLarge { .. } => 413,
            _ => 400,
        }
    }
}
