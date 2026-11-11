//! e-means CPU reference implementation of Bessa et al., arXiv:2611.09501v1.
//!
//! Uses exact squared-L2 assignment, f64 sufficient statistics and seeded
//! ChaCha8 shuffles. This does not implement the paper's GPU or approximate IVF
//! kernels. See [`fit`] and [`Config`].

mod config;
mod distance;
pub mod ivf;
pub mod quantization;
mod trainer;
#[cfg(feature = "wasm")]
mod wasm;

pub use config::{cosine_step, Config, Finalization, Parameters};
pub(crate) use distance::nearest;
pub use ivf::IvfPartition;
pub use quantization::ProductQuantizer;
pub use trainer::fit;

/// Validation failure; invalid input is rejected before training starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub(crate) &'static str);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for Error {}

/// A trained flat, row-major codebook and freshly recomputed training metrics.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct Model {
    pub centroids: Vec<f32>,
    pub dimensions: usize,
    /// Nearest-centroid assignments in original input order, after finalization.
    pub assignments: Vec<usize>,
    /// Mean squared L2 distance per vector, not divided by dimensionality.
    pub mse: f64,
    pub batch_size: usize,
    pub peak_step: f64,
    pub updates: usize,
    pub revived_clusters: usize,
}

impl Model {
    /// Assign new vectors to this codebook. Empty queries return no assignments.
    /// Public/deserialized model fields are validated on every call.
    pub fn predict(&self, data: &[f32]) -> Result<Vec<usize>, Error> {
        predict(data, self.dimensions, &self.centroids)
    }
}

/// Exact squared-L2 prediction with deterministic lowest-index tie breaking.
pub fn predict(data: &[f32], dimensions: usize, centroids: &[f32]) -> Result<Vec<usize>, Error> {
    validate_rows(data, dimensions)?;
    validate_rows(centroids, dimensions)?;
    if centroids.is_empty() {
        return Err(Error("centroids must not be empty"));
    }
    Ok(data
        .chunks_exact(dimensions)
        .map(|row| nearest(row, centroids, dimensions).0)
        .collect())
}

pub(crate) fn validate_rows(data: &[f32], dimensions: usize) -> Result<(), Error> {
    if dimensions == 0 || data.len() % dimensions != 0 {
        return Err(Error(
            "dimensions must be positive and divide the data length",
        ));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(Error("vectors must contain only finite values"));
    }
    Ok(())
}
