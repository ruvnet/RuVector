//! Typed analytics errors and their mapping onto the edge-store wire codes
//! (ADR-351 §7, §16.3). Every limit or budget failure is a 413, never a 500.

use ruvector_edge_store::{ErrorCode, OpError};
use thiserror::Error;

/// Which configured limit was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitKind {
    /// Distinct vertices in the graph.
    Vertices,
    /// Undirected edges in the graph.
    Edges,
}

/// Why persisted bytes were refused. Details are static: nothing from the
/// input is echoed back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorruptKind {
    /// Magic, version or record kind do not match.
    Header,
    /// A sha256 trailer does not match the bytes it covers.
    Checksum,
    /// The record ends early or has trailing bytes.
    Truncated,
    /// A chunk belongs to a different graph, revision or flags.
    ForeignChunk,
    /// A chunk is out of order, missing or not listed in the manifest.
    ChunkOrder,
    /// Decoded counts disagree with the manifest.
    CountMismatch,
    /// A varint is overlong, overflows or a delta leaves the id space.
    Encoding,
}

/// Errors from the analytics core.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AnalyticsError {
    /// A size limit was exceeded (`payload_too_large`, 413).
    #[error("limit exceeded: {kind:?} {actual} > {max}")]
    LimitExceeded {
        /// Which limit.
        kind: LimitKind,
        /// Observed value.
        actual: u64,
        /// Configured maximum.
        max: u64,
    },
    /// The estimated work or memory exceeds the budget (`budget_exceeded`,
    /// 413). `job_eligible` says whether the async job profile would accept it.
    #[error("budget exceeded: {resource} estimated {estimated} > {budget}")]
    BudgetExceeded {
        /// `"work"` or `"memory_bytes"`.
        resource: &'static str,
        /// Cost-model estimate.
        estimated: u64,
        /// Budget it was checked against.
        budget: u64,
        /// Whether `/v1/mincut/jobs` would accept the same query.
        job_eligible: bool,
    },
    /// Persisted bytes failed verification (`server_error` class: stored data
    /// is the service's responsibility, but it never panics).
    #[error("corrupt edge list: {0:?}")]
    Corrupt(CorruptKind),
    /// The caller supplied an invalid graph or parameter (`invalid_request`).
    #[error("invalid: {0}")]
    Invalid(&'static str),
    /// A job transition is not allowed from the current state (`conflict`).
    #[error("job state conflict: {0}")]
    JobState(&'static str),
    /// The solver refused a graph that passed validation (should not happen).
    #[error("solver error")]
    Solver,
}

impl AnalyticsError {
    /// Stable wire code.
    pub fn code(&self) -> ErrorCode {
        match self {
            AnalyticsError::LimitExceeded { .. } => ErrorCode::PayloadTooLarge,
            AnalyticsError::BudgetExceeded { .. } => ErrorCode::BudgetExceeded,
            AnalyticsError::Invalid(_) => ErrorCode::InvalidRequest,
            AnalyticsError::JobState(_) => ErrorCode::Conflict,
            AnalyticsError::Corrupt(_) | AnalyticsError::Solver => ErrorCode::ServerError,
        }
    }

    /// HTTP status (`413` for every limit and budget failure).
    pub fn status(&self) -> u16 {
        self.code().status()
    }

    /// Static, non-echoing detail string.
    pub fn detail(&self) -> &'static str {
        match self {
            AnalyticsError::LimitExceeded { kind, .. } => match kind {
                LimitKind::Vertices => "graph vertex limit exceeded",
                LimitKind::Edges => "graph edge limit exceeded",
            },
            AnalyticsError::BudgetExceeded { job_eligible, .. } => {
                if *job_eligible {
                    "min-cut budget exceeded; submit it as a job"
                } else {
                    "min-cut budget exceeded"
                }
            }
            AnalyticsError::Corrupt(_) => "stored edge list failed verification",
            AnalyticsError::Invalid(d) | AnalyticsError::JobState(d) => d,
            AnalyticsError::Solver => "min-cut solver failure",
        }
    }
}

impl From<AnalyticsError> for OpError {
    fn from(e: AnalyticsError) -> Self {
        OpError::new(e.code(), e.detail())
    }
}

/// Crate result alias.
pub type Result<T> = core::result::Result<T, AnalyticsError>;
