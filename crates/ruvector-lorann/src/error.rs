use thiserror::Error;

#[derive(Debug, Error)]
pub enum LorannError {
    #[error("empty dataset")]
    EmptyDataset,

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },

    #[error("k-means failed to converge after {max_iter} iterations")]
    KMeansTimeout { max_iter: usize },

    #[error("SVD failed for cluster {cluster_id}: matrix is {rows}×{cols} with rank {rank}")]
    SvdFailed {
        cluster_id: usize,
        rows: usize,
        cols: usize,
        rank: usize,
    },

    #[error("cluster {id} has {size} vectors; need ≥ {min} for rank-{rank} factorisation")]
    ClusterTooSmall {
        id: usize,
        size: usize,
        min: usize,
        rank: usize,
    },

    #[error("n_probe ({n_probe}) exceeds n_clusters ({n_clusters})")]
    NProbeExceedsClusters { n_probe: usize, n_clusters: usize },
}

pub type Result<T> = std::result::Result<T, LorannError>;
