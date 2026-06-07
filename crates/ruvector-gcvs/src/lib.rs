pub mod distance;
pub mod flat;
pub mod graph;
pub mod graph_aug;
pub mod graph_coh;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GcvsError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("empty index")]
    Empty,
    #[error("k must be > 0")]
    InvalidK,
}

pub type Result<T> = std::result::Result<T, GcvsError>;

/// Search result carrying the vector id and its score (higher = more similar).
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: usize,
    pub score: f32,
}

/// Common trait for all GCVS index variants.
pub trait GcvsIndex {
    /// Insert a vector. The caller assigns the id.
    fn insert(&mut self, id: usize, vector: Vec<f32>) -> Result<()>;

    /// Return up to k most similar vectors to query (highest cosine similarity first).
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>>;

    /// Number of indexed vectors.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Name for display.
    fn name(&self) -> &'static str;
}
