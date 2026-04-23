use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbedError {
    #[error("node {0} out of range")]
    NodeOutOfRange(usize),
    #[error("anchor {0} out of range")]
    AnchorOutOfRange(u32),
    #[error("distance {0} exceeds max_dist {1}")]
    DistanceTooLarge(u8, u8),
    #[error("empty token sequence for node {0}")]
    EmptyTokens(usize),
}
