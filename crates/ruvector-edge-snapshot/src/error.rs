//! Typed errors. Every refusal a restore, import, job or embedding call can
//! produce has its own variant so the gateway maps it to a stable problem
//! `code` and tests can assert *which* check fired.

use thiserror::Error;

/// Snapshot write / restore / witness failures.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SnapshotError {
    /// A boundary identifier (tenant key, collection uid, service) is malformed.
    #[error("invalid identifier: {0}")]
    InvalidIdentifier(&'static str),
    /// A row failed validation (dimension, id, metadata, non-finite value, order).
    #[error("invalid row {index}: {reason}")]
    InvalidRow {
        /// Zero-based row ordinal within the snapshot.
        index: u64,
        /// Static reason.
        reason: &'static str,
    },
    /// A declared or observed size exceeds the caller's quota or a hard limit.
    #[error("quota exceeded: {what} {requested} > {limit}")]
    QuotaExceeded {
        /// What was measured.
        what: &'static str,
        /// The limit.
        limit: u64,
        /// The requested / declared amount.
        requested: u64,
    },
    /// The manifest object could not be parsed.
    #[error("malformed manifest: {0}")]
    Malformed(&'static str),
    /// The manifest format version is not one this build reads.
    #[error("unknown snapshot format version {0}")]
    UnknownFormat(u16),
    /// The row schema version is not one this build restores.
    #[error("unknown schema version {0}")]
    UnknownSchema(u32),
    /// The manifest belongs to another tenant than the caller.
    #[error("tenant mismatch")]
    TenantMismatch,
    /// The manifest names another service or collection than the target.
    #[error("collection mismatch")]
    CollectionMismatch,
    /// The manifest names another shard than the target.
    #[error("shard mismatch")]
    ShardMismatch,
    /// The manifest dimension or metric differs from the target collection.
    #[error("dimension or metric mismatch")]
    DimensionMismatch,
    /// The recomputed manifest root differs from the stored root.
    #[error("manifest checksum mismatch")]
    ManifestChecksum,
    /// A witness-chain entry does not link to its predecessor or its hash is wrong.
    #[error("witness chain broken at seq {seq}")]
    ChainBreak {
        /// Sequence number of the first bad entry.
        seq: u64,
    },
    /// The chain does not end at the trusted head held by the ledger.
    #[error("witness chain head mismatch")]
    ChainHeadMismatch,
    /// The manifest root is not recorded in the tenant's witness chain.
    #[error("manifest not in witness chain")]
    NotInChain,
    /// A signature was required but the manifest carries none.
    #[error("signature missing")]
    SignatureMissing,
    /// The signature does not verify under the named key.
    #[error("signature invalid")]
    SignatureInvalid,
    /// The signer failed (no key material lives in this crate).
    #[error("signer failed: {0}")]
    SignerFailed(&'static str),
    /// Chunks must be fed in manifest order.
    #[error("chunk out of order: expected {expected}, got {got}")]
    ChunkOutOfOrder {
        /// Next expected chunk index.
        expected: u32,
        /// Index supplied.
        got: u32,
    },
    /// A chunk's byte length differs from the manifest (truncation / padding).
    #[error("chunk {index} size mismatch")]
    ChunkSize {
        /// Chunk index.
        index: u32,
    },
    /// A chunk's sha256 differs from the manifest.
    #[error("chunk {index} checksum mismatch")]
    ChunkChecksum {
        /// Chunk index.
        index: u32,
    },
    /// A chunk passed its sha256 but its segments do not decode.
    #[error("chunk {index} corrupt: {reason}")]
    SegmentCorrupt {
        /// Chunk index.
        index: u32,
        /// Static reason.
        reason: &'static str,
    },
    /// `finish` was called before every chunk was accepted.
    #[error("missing chunks: expected {expected}, got {got}")]
    ChunkMissing {
        /// Chunks in the manifest.
        expected: u32,
        /// Chunks accepted.
        got: u32,
    },
    /// Decoded rows do not add up to the manifest row count.
    #[error("row count mismatch")]
    RowCountMismatch,
}

/// RVF import failures.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ImportError {
    /// Structural decode failure.
    #[error("malformed rvf: {0}")]
    Malformed(&'static str),
    /// A segment declares a payload larger than the import limit.
    #[error("segment too large: {0} bytes")]
    SegmentTooLarge(u64),
    /// The upload exceeds the file-size limit.
    #[error("file too large")]
    FileTooLarge,
    /// A segment's content hash does not verify.
    #[error("segment checksum mismatch at offset {0}")]
    Checksum(u64),
    /// Vector dimension differs from the collection.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Collection dimension.
        expected: u16,
        /// File dimension.
        got: u16,
    },
    /// The file's distance metric differs from the collection.
    #[error("metric mismatch")]
    MetricMismatch,
    /// Importing would exceed the tenant's row quota.
    #[error("row quota exceeded: limit {limit}")]
    QuotaExceeded {
        /// Rows allowed for this import.
        limit: u64,
    },
    /// A row failed validation.
    #[error("invalid row {index}: {reason}")]
    InvalidRow {
        /// Zero-based row ordinal within the file.
        index: u64,
        /// Static reason.
        reason: &'static str,
    },
    /// No manifest segment was found.
    #[error("no manifest")]
    NoManifest,
    /// The file's manifest disagrees with the imported segments or the tail summary.
    #[error("manifest disagrees with stream: {0}")]
    ManifestMismatch(&'static str),
    /// The stream ended inside a segment.
    #[error("truncated rvf")]
    Truncated,
}

/// Bulk-import job state-machine violations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum JobError {
    /// The transition is not allowed from the current state.
    #[error("illegal transition from {from}")]
    IllegalTransition {
        /// Current state name.
        from: &'static str,
    },
    /// A batch arrived out of sequence (gap).
    #[error("batch gap: expected {expected}, got {got}")]
    BatchGap {
        /// Next expected batch sequence.
        expected: u64,
        /// Supplied sequence.
        got: u64,
    },
    /// Persisted job bytes do not decode.
    #[error("corrupt job record: {0}")]
    Corrupt(&'static str),
    /// A job identifier is malformed.
    #[error("invalid identifier: {0}")]
    InvalidIdentifier(&'static str),
    /// The staging object or stream differs from the declared / pinned upload.
    #[error("upload mismatch: {0}")]
    UploadMismatch(&'static str),
}

/// Embedding batcher / port failures.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EmbedError {
    /// Input text at `index` is empty.
    #[error("text {index} is empty")]
    EmptyText {
        /// Input index.
        index: usize,
    },
    /// Input text at `index` exceeds the byte limit.
    #[error("text {index} too long")]
    TextTooLong {
        /// Input index.
        index: usize,
    },
    /// Input text at `index` exceeds the per-text token estimate.
    #[error("text {index} exceeds token limit")]
    TooManyTokens {
        /// Input index.
        index: usize,
    },
    /// Too many texts in one request.
    #[error("too many texts: {0}")]
    TooManyTexts(usize),
    /// The model answered with the wrong number of vectors.
    #[error("response count mismatch")]
    ResponseCount,
    /// A returned vector has the wrong dimension or a non-finite value.
    #[error("bad vector {index} in response")]
    BadVector {
        /// Input index.
        index: usize,
    },
    /// The port (Workers AI binding) failed.
    #[error("embedding port failed: {0}")]
    Port(&'static str),
}
