//! Typed errors. Hand-written `Display` keeps the crate dependency-light.

use std::fmt;

/// Invalid input or state on the insert/search path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexError {
    /// Vector length differs from the index dimension.
    DimMismatch {
        /// Index dimension.
        expected: usize,
        /// Supplied length.
        got: usize,
    },
    /// A NaN or infinite component.
    NonFinite,
    /// Zero-norm vector under cosine.
    ZeroNorm,
    /// Invalid construction parameters.
    InvalidParams(&'static str),
    /// Inserting would exceed the slot cap (bounded memory; a huge iid
    /// must not allocate a huge gap).
    CapacityExceeded {
        /// Configured slot cap.
        max_slots: u32,
    },
    /// `insert` on an iid that already holds a node (use `update` or
    /// `upsert` to move it).
    Occupied(u32),
    /// `update` on an iid that holds no node.
    NotFound(u32),
    /// iid below the index's `iid_base`.
    IidBelowBase {
        /// Offending iid.
        iid: u32,
        /// Configured base.
        base: u32,
    },
    /// Quantizer epoch of an index differs from the one supplied.
    QuantEpochMismatch {
        /// Epoch the index was built with.
        index: u64,
        /// Epoch supplied.
        got: u64,
    },
}

/// Refusal to serialise an index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// The iid space has never-assigned slots: `node_count != max_iid + 1`
    /// over present nodes. The store must renumber (compact) first.
    GappedIid {
        /// Lowest never-assigned iid.
        first_missing: u32,
        /// Number of never-assigned slots.
        missing: u32,
    },
    /// Internal structural invariant violated (a bug, never silent).
    Invariant(&'static str),
    /// Requested chunk size cannot hold a header plus one body byte, or
    /// exceeds [`crate::MAX_CHUNK_BYTES`].
    BadChunkSize(usize),
}

/// Rejection of stored `index_chunks`. Every variant means "do not use this
/// epoch": the caller loads the previous epoch or replays the op log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// No chunks supplied.
    Empty,
    /// Chunk shorter than its header, or its length differs from the
    /// declared body length.
    Truncated {
        /// Row part number.
        part: u32,
    },
    /// Magic bytes do not match.
    BadMagic {
        /// Row part number.
        part: u32,
    },
    /// Unknown format version.
    UnsupportedVersion(u16),
    /// Per-chunk sha256 mismatch (bit rot, partial write).
    ChecksumMismatch {
        /// Row part number.
        part: u32,
    },
    /// Row `(epoch, part)` disagrees with the chunk header.
    RowMismatch {
        /// Row part number.
        part: u32,
    },
    /// Chunks not in ascending part order starting at 0.
    PartOutOfOrder {
        /// Expected part.
        expected: u32,
        /// Found part.
        got: u32,
    },
    /// Parts from different epochs, totals, kinds or payload digests.
    Inconsistent {
        /// Row part number.
        part: u32,
    },
    /// Fewer or more chunks than the header's `total_parts`.
    PartCount {
        /// Declared total.
        expected: u32,
        /// Supplied.
        got: u32,
    },
    /// Concatenated payload length or sha256 differs from the header, or
    /// from the digest the caller recorded in `meta.index_sha256`.
    DigestMismatch,
    /// Chunks hold a different index kind than requested.
    WrongKind,
    /// Payload failed structural validation (after checksums passed).
    Malformed(&'static str),
    /// Declared payload length exceeds the caller's limit (refused before
    /// any allocation).
    TooLarge {
        /// Declared payload bytes.
        declared: u64,
        /// Caller limit.
        limit: u64,
    },
}

/// Failure of a streaming encode ([`crate::HnswIndex::encode_into`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError<E> {
    /// The index refused to encode.
    Encode(EncodeError),
    /// The chunk sink failed (e.g. a SQL write); later parts were not sent.
    Sink(E),
}

/// Rerank failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RerankError<E> {
    /// The query failed boundary validation.
    Invalid(IndexError),
    /// The fetch callback failed.
    Fetch(E),
    /// Fetched vector has the wrong dimension (store corruption).
    DimMismatch {
        /// Offending iid.
        iid: u32,
    },
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::NonFinite => f.write_str("vector has a non-finite component"),
            Self::ZeroNorm => f.write_str("zero-norm vector under cosine"),
            Self::InvalidParams(m) => write!(f, "invalid parameters: {m}"),
            Self::CapacityExceeded { max_slots } => {
                write!(f, "index slot cap {max_slots} exceeded")
            }
            Self::Occupied(i) => write!(f, "iid {i} already holds a node"),
            Self::NotFound(i) => write!(f, "iid {i} holds no node"),
            Self::IidBelowBase { iid, base } => write!(f, "iid {iid} below iid_base {base}"),
            Self::QuantEpochMismatch { index, got } => {
                write!(f, "quantizer epoch mismatch: index {index}, got {got}")
            }
        }
    }
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GappedIid {
                first_missing,
                missing,
            } => write!(
                f,
                "gapped iid space: {missing} never-assigned slots (first {first_missing}); compact before encoding"
            ),
            Self::Invariant(m) => write!(f, "index invariant violated: {m}"),
            Self::BadChunkSize(n) => write!(f, "bad chunk size {n}"),
        }
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("no index chunks"),
            Self::Truncated { part } => write!(f, "chunk {part} truncated"),
            Self::BadMagic { part } => write!(f, "chunk {part} bad magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported chunk format version {v}"),
            Self::ChecksumMismatch { part } => write!(f, "chunk {part} checksum mismatch"),
            Self::RowMismatch { part } => write!(f, "chunk {part} row/header mismatch"),
            Self::PartOutOfOrder { expected, got } => {
                write!(f, "chunk out of order: expected part {expected}, got {got}")
            }
            Self::Inconsistent { part } => write!(f, "chunk {part} inconsistent with part 0"),
            Self::PartCount { expected, got } => {
                write!(f, "expected {expected} chunks, got {got}")
            }
            Self::DigestMismatch => f.write_str("index payload digest mismatch"),
            Self::WrongKind => f.write_str("chunks hold a different index kind"),
            Self::Malformed(m) => write!(f, "malformed index payload: {m}"),
            Self::TooLarge { declared, limit } => {
                write!(f, "declared payload {declared} B exceeds limit {limit} B")
            }
        }
    }
}

impl<E: fmt::Display> fmt::Display for EmitError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(e) => write!(f, "{e}"),
            Self::Sink(e) => write!(f, "chunk sink failed: {e}"),
        }
    }
}

impl<E: fmt::Display> fmt::Display for RerankError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(e) => write!(f, "invalid query: {e}"),
            Self::Fetch(e) => write!(f, "rerank fetch failed: {e}"),
            Self::DimMismatch { iid } => write!(f, "fetched vector {iid} has wrong dimension"),
        }
    }
}

impl std::error::Error for IndexError {}
impl std::error::Error for EncodeError {}
impl std::error::Error for DecodeError {}
impl<E: fmt::Debug + fmt::Display> std::error::Error for RerankError<E> {}
impl<E: fmt::Debug + fmt::Display> std::error::Error for EmitError<E> {}
