use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShardError {
    #[error("vector slice length is not a multiple of dim")]
    InvalidDimension,

    #[error("graph has no nodes")]
    EmptyGraph,

    #[error("budget is zero")]
    ZeroBudget,

    #[error("anchor index {0} is out of range")]
    AnchorOutOfRange(u32),

    #[error("wire format: bad magic bytes")]
    BadMagic,

    #[error("wire format: unsupported version {0}")]
    UnsupportedVersion(u32),

    #[error("wire format: truncated data at offset {0}")]
    Truncated(usize),

    #[error("wire format: node_count {0} exceeds sanity limit")]
    NodeCountTooLarge(u64),
}

pub type ShardResult<T> = Result<T, ShardError>;
