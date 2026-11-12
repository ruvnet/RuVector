//! Typed errors for the quant shard, mapped onto the ADR-351 §16.3 codes.
//!
//! The mapping that matters for M4 acceptance: every budget refusal is
//! `413 budget_exceeded` (never `500`), and a corrupt or unreadable snapshot
//! is `503 shard_unavailable` (the host re-derives the shard from the op log;
//! it is not the caller's fault and not an internal panic either).

use ruvector_edge_store::{ErrorCode, OpError};
use thiserror::Error;

/// Which budget a refusal was charged against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetResource {
    /// Resident bytes of the shard (codes, norms, keys, key index, rotation).
    ResidentBytes,
    /// Vector count cap of the shard.
    Vectors,
    /// Work units of one query (scan + rerank + query rotation).
    QueryUnits,
    /// Work units of one cold load (bytes decoded + rotation regeneration).
    LoadUnits,
    /// Rerank candidates of one query (rows fetched from the f32 store).
    RerankCandidates,
    /// `top_k` above the per-query limit.
    TopK,
}

impl BudgetResource {
    /// Stable wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            BudgetResource::ResidentBytes => "resident_bytes",
            BudgetResource::Vectors => "vectors",
            BudgetResource::QueryUnits => "query_units",
            BudgetResource::LoadUnits => "load_units",
            BudgetResource::RerankCandidates => "rerank_candidates",
            BudgetResource::TopK => "top_k",
        }
    }
}

/// Why a persisted snapshot was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CorruptKind {
    /// Missing or wrong magic bytes.
    #[error("bad magic")]
    BadMagic,
    /// A format version this build cannot read.
    #[error("unsupported version {0}")]
    UnsupportedVersion(u16),
    /// Header CRC mismatch.
    #[error("header checksum mismatch")]
    HeaderChecksum,
    /// A data frame's payload CRC mismatch.
    #[error("frame {0} checksum mismatch")]
    FrameChecksum(u32),
    /// The footer's rolling CRC over all frame CRCs does not match.
    #[error("footer checksum mismatch")]
    FooterChecksum,
    /// The input ended before the declared layout was complete.
    #[error("truncated")]
    Truncated,
    /// Frames out of order, missing or duplicated.
    #[error("frame sequence: expected {expected}, got {got}")]
    FrameSequence {
        /// Expected frame index.
        expected: u32,
        /// Frame index found.
        got: u32,
    },
    /// Bytes after the footer, or more frames than the header declares.
    #[error("trailing data")]
    TrailingData,
    /// A structurally invalid field (sizes, enum tags, section order).
    #[error("malformed: {0}")]
    Malformed(&'static str),
    /// The rotation regenerated from `(kind, seed, dim)` does not produce
    /// the fingerprint recorded at save time.
    #[error("rotation fingerprint mismatch")]
    RotationMismatch,
    /// A stored norm is negative or non-finite.
    #[error("invalid norm")]
    InvalidNorm,
    /// Padding bits past `dim` in a code's last word are set.
    #[error("padding bits set")]
    PaddingBits,
    /// Two rows share a key.
    #[error("duplicate key")]
    DuplicateKey,
}

/// Every failure the quant shard reports.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum QuantError {
    /// A hard budget would be exceeded. Checked *before* the work or
    /// allocation it guards; maps to `413 budget_exceeded`.
    #[error("budget exceeded: {} (limit {limit}, requested {requested})", resource.as_str())]
    BudgetExceeded {
        /// Which budget.
        resource: BudgetResource,
        /// Configured limit.
        limit: u64,
        /// What the operation would have needed.
        requested: u64,
    },
    /// Vector or query length differs from the shard dimension.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Shard dimension.
        expected: usize,
        /// Length supplied.
        actual: usize,
    },
    /// NaN/inf component, or a zero vector under the cosine metric.
    #[error("non-finite or degenerate vector")]
    NonFinite,
    /// Invalid shard configuration (dim 0, dim above the ADR limit).
    #[error("invalid config: {0}")]
    InvalidConfig(&'static str),
    /// A persisted snapshot failed validation.
    #[error("corrupt snapshot: {0}")]
    Corrupt(CorruptKind),
    /// The rerank source failed (the store could not read f32 rows).
    #[error("rerank source failed: {0}")]
    Rerank(String),
    /// An I/O error other than end-of-input while streaming a snapshot.
    #[error("io: {0}")]
    Io(String),
}

impl QuantError {
    /// Shorthand for a corruption error.
    pub fn corrupt(kind: CorruptKind) -> Self {
        QuantError::Corrupt(kind)
    }

    /// The ADR-351 error code this failure surfaces as.
    pub fn code(&self) -> ErrorCode {
        match self {
            QuantError::BudgetExceeded { .. } => ErrorCode::BudgetExceeded,
            QuantError::DimensionMismatch { .. } => ErrorCode::DimensionMismatch,
            QuantError::NonFinite => ErrorCode::NonFiniteValue,
            QuantError::InvalidConfig(_) => ErrorCode::InvalidRequest,
            QuantError::Corrupt(_) | QuantError::Rerank(_) | QuantError::Io(_) => {
                ErrorCode::ShardUnavailable
            }
        }
    }

    /// HTTP status (`413` for every budget refusal).
    pub fn status(&self) -> u16 {
        self.code().status()
    }
}

impl From<QuantError> for OpError {
    fn from(e: QuantError) -> Self {
        let detail: &'static str = match &e {
            QuantError::BudgetExceeded { resource, .. } => match resource {
                BudgetResource::ResidentBytes => "quant shard resident byte budget",
                BudgetResource::Vectors => "quant shard vector budget",
                BudgetResource::QueryUnits => "quant query work budget",
                BudgetResource::LoadUnits => "quant cold-load work budget",
                BudgetResource::RerankCandidates => "quant rerank candidate budget",
                BudgetResource::TopK => "top_k above limit",
            },
            QuantError::DimensionMismatch { .. } => "dimension mismatch",
            QuantError::NonFinite => "non-finite value",
            QuantError::InvalidConfig(d) => d,
            QuantError::Corrupt(_) => "quant snapshot unreadable",
            QuantError::Rerank(_) => "rerank source unavailable",
            QuantError::Io(_) => "quant snapshot read failed",
        };
        OpError::new(e.code(), detail)
    }
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, QuantError>;
